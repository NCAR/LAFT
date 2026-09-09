#!/usr/bin/env python3
"""Shared profiler harness — driver-agnostic. Two responsibilities:

1. **Staged-code loading** (``load_staged_bridge``): the profiling loop
   iterates on a staged copy of the translation ([profiler].staged_code).
   The staged module is injected into ``sys.modules`` under the canonical
   name (out.jax.<proc>) BEFORE the bridge module is imported, so the bridge
   binds the staged code instead of the committed translation. Identical
   mechanism for any project.

2. **Dispatch to the project's own input layer** (``capture_state`` /
   ``tile_state``): *how* to get one realistic bridge call's worth of state
   out of a project's driver, and *how* to expand it to ncol columns, is
   driver-shaped, not framework-shaped — one project's driver is a
   single-column, time-stepping model (capture keyed on a minute -> `it`
   step, tile by replicating axis 0 from size 1); Kessler's driver is a
   single-shot, real-128-column batch (capture the one call, tile by cycling
   axis 1 from size 128). Forcing both through one "generic" implementation
   just grows special cases per project, so — like [driver].script and
   [comparison].script — each project hand-authors its own
   [profiler].inputs_script implementing ``capture_state``/``tile_state``;
   this module only loads it and forwards the calls. See
   PROFILE_WORKFLOW.md and an existing project's inputs_script (e.g.
   out/profiled/kessler_profiler_inputs.py) for the exact contract.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


# workflow_profiler/ is a per-project SYMLINK to LAFT/workflow_profiler/, so
# Path(__file__).resolve() follows it back to the shared LAFT/ tree — using
# that as PROJECT_ROOT would silently resolve every per-project path (out/,
# data/, ...) against LAFT/ instead of the actual project root. But resolving
# to the sibling LAFT/config/ is exactly what we want, purely to import
# framework_config; the real PROJECT_ROOT then comes from get_config().root,
# which discovers it by walking up from the cwd (PBS jobs / manual runs
# always cd to the project root first) — the same convention framework_config
# itself uses.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "config"))

from framework_config import get_config  # noqa: E402

PROJECT_ROOT = get_config().root
sys.path.insert(0, str(PROJECT_ROOT))


def profiler_cfg() -> dict:
    cfg = get_config().section("profiler")
    if not cfg:
        raise SystemExit("No [profiler] section in config/project.toml")
    return cfg


_PROJECT_INPUTS_MODULE = None


def _project_inputs():
    """Dynamically load [profiler].inputs_script — the hand-authored,
    per-project file (like [driver].script / [comparison].script) that
    implements capture_state/tile_state for this project's own driver.
    """
    global _PROJECT_INPUTS_MODULE
    if _PROJECT_INPUTS_MODULE is not None:
        return _PROJECT_INPUTS_MODULE

    cfg = profiler_cfg()
    script = cfg.get("inputs_script")
    if not script:
        raise SystemExit(
            "[profiler].inputs_script not set in config/project.toml — "
            "see PROFILE_WORKFLOW.md (a per-project file implementing "
            "capture_state/tile_state for this project's driver, analogous "
            "to [driver].script)")
    path = PROJECT_ROOT / script
    if not path.exists():
        raise SystemExit(f"[profiler].inputs_script not found: {path}")

    spec = importlib.util.spec_from_file_location("_project_profiler_inputs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _PROJECT_INPUTS_MODULE = mod
    return mod


def capture_state(force: bool = False) -> dict:
    """Return one realistic bridge call's worth of kwargs, captured from
    this project's validated driver. See PROFILE_WORKFLOW.md and
    [profiler].inputs_script for how — that's project-specific."""
    return _project_inputs().capture_state(force=force)


def tile_state(state: dict, ncol: int, seed: int = 42) -> dict:
    """Expand a captured state to ncol columns. See PROFILE_WORKFLOW.md and
    [profiler].inputs_script for how — that's project-specific."""
    return _project_inputs().tile_state(state, ncol, seed=seed)


def load_staged_bridge(staged_path: Path):
    """Import the staged translation under its canonical module name, then
    import the bridge so it binds the staged code. Returns the bridge callable.
    """
    cfg = profiler_cfg()
    target_proc = cfg["target_proc"]
    bridge_module = cfg["bridge_module"]
    bridge_func = cfg["bridge_func"]
    canonical = f"out.jax.{target_proc}"

    if bridge_module in sys.modules:
        del sys.modules[bridge_module]
    if canonical in sys.modules:
        del sys.modules[canonical]

    spec = importlib.util.spec_from_file_location(canonical, staged_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[canonical] = mod          # register BEFORE exec + bridge import
    spec.loader.exec_module(mod)

    bridge_mod = importlib.import_module(bridge_module)
    return getattr(bridge_mod, bridge_func)


def get_profiling_inputs(ncol: int, seed: int = 42) -> dict:
    """Captured (cached) driver state, tiled to ncol."""
    return tile_state(capture_state(), ncol, seed=seed)


def ensure_cache_subprocess() -> None:
    """Build the state cache in a SEPARATE process if it is missing.

    The profiling runners call this instead of capture_state() so the driver
    spin-up never runs inside the instrumented process — it would pollute the
    nsys timeline and add its ncol=1 modules to the XLA HLO dump.
    """
    import subprocess
    cfg = profiler_cfg()
    cache = PROJECT_ROOT / cfg["state_cache"]
    if cache.exists():
        return
    print("[inputs] state cache missing — building it in a subprocess ...")
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--ensure"],
                   check=True, cwd=str(PROJECT_ROOT))
    if not cache.exists():
        raise SystemExit(f"state capture subprocess did not produce {cache}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Profiler state-capture utility")
    ap.add_argument("--ensure", action="store_true",
                    help="Build the state cache if missing (no-op when present).")
    ap.add_argument("--force", action="store_true",
                    help="Re-capture even if the cache exists.")
    a = ap.parse_args()
    st = capture_state(force=a.force)
    n_arr = sum(1 for v in st.values() if isinstance(v, np.ndarray))
    print(f"[inputs] state ready: {len(st)} args, {n_arr} arrays "
          f"(cache: {profiler_cfg()['state_cache']})")
