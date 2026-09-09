"""
framework_config — per-project configuration for the LAFT pipeline.

Loads config/project.toml (see that file for the schema) and exposes the
values the phase tools need: source files, the out/ directory layout, and the
shared name-set heuristics that were previously duplicated across phases.

Contract: the phase tools are run from the project root (the directory that
contains config/), exactly as before this module existed. All paths returned
here are root-relative Path objects, so generated artifacts keep the same
embedded path strings as the pre-config pipeline.

Config discovery order:
  1. explicit path passed to get_config()/init() (each tool's --config flag)
  2. $FORTRAN2JAX_CONFIG
  3. config/project.toml in the current directory, then in parent directories
  4. config/project.toml next to this config/ directory

Stdlib-only: tomllib (Python 3.11+) with tomli fallback (Python 3.10).
"""

from __future__ import annotations

import keyword
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10 (jax-validate env)
    import tomli as tomllib  # type: ignore[no-redef]

_CONFIG_SUBPATH = Path("config") / "project.toml"


# ---------------------------------------------------------------------------
# Fortran identifier -> Python identifier
# ---------------------------------------------------------------------------
# Fortran's reserved words are not Python's. One scheme names its grid start
# index `is` (paired with `ie`, like `js`/`je` and `ks`/`ke`) — legal Fortran,
# a reserved word in Python. Emitting it produced 13 bridges that were not
# valid Python at all, and would have told the translator to write
# `def proc(..., is, ...)`.
#
# EVERY generator that turns a Fortran name into a Python identifier must route
# it through here, so the bridge, the wrapper and the prompt agree on one name.
# Convention is PEP 8's: trailing underscore (`is` -> `is_`, like `class_`).
#
# Bridges call the kernel positionally, so argument ORDER is what has to match
# between bridge and translation — but the translated function's own parameter
# list still has to be valid Python, which is why the prompt generator needs
# this too.
#
# Projects with no colliding names (e.g. Kessler) are unaffected: this
# is the identity function for every name that is already a legal identifier.

_PY_SOFT_RESERVED = frozenset({"none", "true", "false", "match", "case", "type"})


def safe_py_name(name: str) -> str:
    """Return a Fortran identifier as a valid, non-reserved Python identifier."""
    if not name:
        return name
    low = name.lower()
    if keyword.iskeyword(low) or low in _PY_SOFT_RESERVED or keyword.iskeyword(name):
        return f"{name}_"
    return name


def safe_py_names(names) -> List[str]:
    """safe_py_name over a sequence, preserving order."""
    return [safe_py_name(n) for n in names]


def _discover_config_path() -> Path:
    env = os.environ.get("FORTRAN2JAX_CONFIG")
    if env:
        p = Path(env)
        if p.exists():
            return p
        raise FileNotFoundError(f"$FORTRAN2JAX_CONFIG points to missing file: {p}")

    cur = Path.cwd()
    for base in [cur, *cur.parents]:
        cand = base / _CONFIG_SUBPATH
        if cand.exists():
            return cand

    cand = Path(__file__).resolve().parent.parent / _CONFIG_SUBPATH
    if cand.exists():
        return cand

    raise FileNotFoundError(
        "No config/project.toml found (searched $FORTRAN2JAX_CONFIG, the "
        "current directory and its parents, and the tools/ install location). "
        "Run from the project root or pass --config."
    )


class Config:
    def __init__(self, config_path: Path):
        self.config_path = config_path.resolve()
        self.root = self.config_path.parent.parent  # <root>/config/project.toml
        with open(self.config_path, "rb") as f:
            self.raw: Dict[str, Any] = tomllib.load(f)

        heur = self.section("heuristics")
        self._dim_names = frozenset(n.lower() for n in heur.get("dim_names", []))
        self._per_step_names = tuple(heur.get("per_step_scalar_names", []))
        self._scalar_flag_names = frozenset(
            n.lower() for n in heur.get("scalar_flag_names", [])
        )

    # ---------------- generic access ----------------

    def section(self, name: str) -> Dict[str, Any]:
        v = self.raw.get(name, {})
        return v if isinstance(v, dict) else {}

    # ---------------- project / source ----------------

    @property
    def project_name(self) -> str:
        return self.section("project").get("name", "project")

    @property
    def fortran_files(self) -> List[Path]:
        return [Path(p) for p in self.section("source").get("fortran_files", [])]

    # ---------------- out/ layout ----------------
    # Subdirectory names are framework convention; only out_root is configured.

    @property
    def out_root(self) -> Path:
        return Path(self.section("source").get("out_root", "out"))

    @property
    def procedures_dir(self) -> Path:
        return self.out_root / "procedures"

    @property
    def programs_dir(self) -> Path:
        return self.out_root / "programs"

    @property
    def modules_dir(self) -> Path:
        return self.out_root / "modules"

    @property
    def packets_dir(self) -> Path:
        return self.out_root / "packets"

    @property
    def prompts_dir(self) -> Path:
        return self.out_root / "prompts"

    @property
    def jax_dir(self) -> Path:
        return self.out_root / "jax"

    @property
    def wrappers_dir(self) -> Path:
        # Reference wrapper skeletons (phase04_02). Kept OUT of out/jax so a
        # rerun can never overwrite the LLM translations that live there.
        return self.out_root / "wrappers"

    @property
    def bridge_dir(self) -> Path:
        return self.out_root / "bridge"

    @property
    def reports_dir(self) -> Path:
        # Consolidated workflow reports: out/reports/<workflow>/ — one place
        # to answer "did each workflow finish green" (bridge / translation /
        # profile). Working artifacts consumed by later stages stay where
        # their consumers expect them (driver/comparison JSON in out/driver/,
        # fix_log.md in out/issues/).
        return self.out_root / "reports"

    @property
    def bridge_reports_dir(self) -> Path:
        return self.reports_dir / "bridge"

    @property
    def lint_dir(self) -> Path:
        return self.out_root / "lint"

    @property
    def validation_dir(self) -> Path:
        return self.out_root / "validation"

    @property
    def issues_dir(self) -> Path:
        return self.out_root / "issues"

    @property
    def phase1_index(self) -> Path:
        return self.out_root / "phase1_index.json"

    @property
    def all_deps_file(self) -> Path:
        return self.packets_dir / "_ALL_deps.json"

    @property
    def module_deps_file(self) -> Path:
        return self.out_root / "module_dependencies.json"

    @property
    def call_graph_svg(self) -> Path:
        return self.out_root / "procedure_call_graph.svg"

    # ---------------- driver contracts ----------------
    # [driver].contracts — the project's bridge-contract decision record.
    # [1, 2] = undecided (exercise both until the project's benchmark
    # decides); a single value = pinned. Key absent = [1], so configs
    # predating the declaration keep host-only behavior.

    _DRIVER_MODE_BY_CONTRACT = {1: "host", 2: "device_resident"}

    @property
    def driver_contracts(self) -> List[int]:
        """Declared bridge contracts, sorted, from [driver].contracts."""
        raw = self.section("driver").get("contracts", [1])
        if isinstance(raw, int):
            raw = [raw]
        vals = sorted({int(v) for v in raw})
        bad = [v for v in vals if v not in self._DRIVER_MODE_BY_CONTRACT]
        if bad or not vals:
            raise ValueError(
                f"[driver].contracts must be a non-empty subset of [1, 2], "
                f"got {raw!r} in {self.config_path}"
            )
        return vals

    @property
    def driver_modes(self) -> List[str]:
        """LAFT_DRIVER_MODE value per declared contract (1 -> 'host',
        2 -> 'device_resident'). The hand-authored driver reads the
        LAFT_DRIVER_MODE environment variable (default 'host' when unset);
        Stage 2 runs driver + comparison once per mode listed here."""
        return [self._DRIVER_MODE_BY_CONTRACT[c] for c in self.driver_contracts]

    # ---------------- prompts ----------------

    @property
    def policies_dir(self) -> Path:
        return Path(self.section("prompts").get("policies_dir", "workflow_translator/prompt_policies"))

    @property
    def project_policies(self) -> List[str]:
        """Project-specific policy files appended to Pass-1 prompts (may be empty)."""
        return list(self.section("prompts").get("project_policies", []))

    @property
    def error_arg_names(self) -> List[str]:
        """Argument names that trigger inclusion of error_policy.md."""
        return list(self.section("prompts").get("error_arg_names", ["errmsg", "errflg"]))

    # ---------------- shared heuristics ----------------
    # Single source for the name-set checks previously copied into
    # phase02_02, phase02_03, phase04_01, phase04_02 and phase05_01.

    def is_dim_name(self, name: str) -> bool:
        """Dimension/size argument names that must never be treated as outputs."""
        return name.lower() in self._dim_names

    # Built-in defaults; override/extend with [heuristics].per_step_scalar_names
    _PER_STEP_DEFAULTS = frozenset({"it", "kount", "itimestep", "istep"})

    def is_per_step_scalar(self, name: str) -> bool:
        """Scalars that vary every call (timestep counters, sim time).

        These must NEVER become jit statics: a static arg is baked into the
        compiled binary, so a per-step value re-specializes/recompiles the
        kernel every step (~18 s/step observed on a large multi-stage orchestrator procedure). phase03 excludes these from
        the generated bridge's static_argnames.
        """
        n = name.lower()
        extra = frozenset(
            x.lower() for x in getattr(self, "_per_step_names", ()) or ())
        return n in self._PER_STEP_DEFAULTS or n in extra

    def is_scalar_output_flag(self, name: str) -> bool:
        """Scalar status-flag names (typically INTENT(OUT) flags), plus 'is_*'.

        The generic prefix rule requires 'is_' (not bare 'is'): one scheme
        had intent(out) ARRAYS whose names begin with 'is', and a bare
        'is*' rule misclassified them as droppable scalar flags — phase03
        (bridge) kept them while phase04/phase05 dropped them, an
        unsatisfiable wrapper contract. 'is_' still matches real flags
        (is_ok is in the explicit list; an 'is_activated' flag is still
        classified as a flag) and the bare renamed index 'is_'
        (safe_py_name of Fortran 'is') stays excluded via the length guard.
        """
        n = name.lower()
        if n in self._scalar_flag_names:
            return True
        return n.startswith("is_") and len(n) > 3


_CONFIG: Optional[Config] = None


def init(config_path: Optional[str] = None) -> Config:
    """Load (or reload) the config, optionally from an explicit path."""
    global _CONFIG
    path = Path(config_path) if config_path else _discover_config_path()
    _CONFIG = Config(path)
    return _CONFIG


def get_config() -> Config:
    """Return the loaded config, discovering it on first use."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config(_discover_config_path())
    return _CONFIG


def add_config_arg(parser) -> None:
    """Attach the standard --config option to an argparse parser."""
    parser.add_argument(
        "--config",
        default=None,
        help="Path to project config TOML (default: auto-discover config/project.toml)",
    )


def packet_module_vars(merged_packet: Dict[str, Any]) -> List[str]:
    """The module-var list every generated signature must agree on.

    Single source of truth for phase03 (bridges), phase04_01/02
    (prompts/wrappers), and phase05_01/02 (lint/runtime validation) — these
    five must use the SAME set and order or the pipeline contradicts itself.

    Rule (2026-07-17, reverse-engineered from a campaign of validated
    translations of a multi-procedure scheme):
      - base = `module_vars_used`, the TRANSITIVE CLOSURE over the call
        graph: a procedure must accept every module var any of its (direct
        or indirect) callees uses, because those globals are threaded down
        as explicit parameters (e.g. sedimentation_liquid's validated
        translation takes autoaccr_param/cons1/pi/rhow/thrd purely to pass
        them to callees). Emitted in module DECLARATION order by phase02_03;
      - parameter constants are already excluded upstream (phase02_01
        filters them before the closure is built) — compile-time constants
        are baked into translations, never runtime state;
      - LUT tables are pulled out separately as pass-through kwargs via
        [bridge.optional_kwargs] at generation time, not here;
      - drop names listed in [bridge].exclude_module_vars — per-project
        judgment calls the dependency analysis cannot make (e.g. a global
        status variable: error plumbing the translations handle internally).
    """
    mv = merged_packet.get("module_vars_used", []) or []
    exclude = {v.lower()
               for v in get_config().section("bridge").get("exclude_module_vars", [])}
    return [v for v in mv if v.lower() not in exclude]
