# -----------------------------------------------------------
# Phase 5.2: Runtime Validation (Smoke + JIT) — Bridge Mode ONLY
#
# Validates generated Python/JAX translation at runtime:
#   - Import module
#   - Call wrapper with dummy inputs
#   - Call jitted _core (if effects.has_io == False and _core exists)
#   - Sanity checks: outputs exist, shapes consistent, finite values
#
# Layout policy:
#   BRIDGE MODE ONLY (row-major compute). NO Policy A, NO swapaxes.
#
# Dummy data:
#   Generic fallbacks for most procedures.
#   Kessler microphysics: atmospheric physics-specific dummy data
#   (theta, qv, qc, qr, cpair, rair, rho, z, pk, etc.)
#   All 2D dummy arrays are created in Fortran layout (ncol, nz).
#
# Call strategy (bridge mode):
#   Wrapper test : calls {proc}_bridge with Fortran-layout inputs (full stack)
#   Core test    : calls {proc}_core with inputs transposed to JAX layout (nz, ncol)
#
# Inputs:
#   out/packets/<PROC>_merged.json
#   out/jax/<PROC>.py
#   out/bridge/<PROC>_bridge.py  (used for wrapper test when present)
#
# Outputs:
#   out/validation/<PROC>_runtime.json
# -----------------------------------------------------------
from __future__ import annotations
import os
os.environ["JAX_ENABLE_X64"] = "1"

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp



import re
import importlib.util
import inspect
import json
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

# framework_config lives in LAFT/config/; this script now lives in
# validation/ (symlinked from LAFT/ into each project) — resolve back to
# LAFT/config to import it. Same convention as workflow_bridge/phase03_make_bridge.py.
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "config"))

from framework_config import get_config, packet_module_vars



def _is_scalar_only_proc(merged: Dict[str, Any]) -> bool:
    """
    Returns True if this procedure has NO array arguments.
    Scalar-only procedures have no _core and must not be JIT-tested.
    Mirrors the detection in phase04_01, phase04_02, phase05_01.
    """
    fortran_source = merged.get("phase2_packet", {}).get("fortran_source", "")
    if not fortran_source:
        return False
    array_patterns = [
        r"::\s*\w+\s*\(",      # :: varname( — array in declaration
        r"\bdimension\s*\(",   # dimension( attribute
    ]
    for pattern in array_patterns:
        if re.search(pattern, fortran_source, re.IGNORECASE):
            return False
    return True


def _find_ci(dirpath: str, filename: str) -> Path:
    """
    Return dirpath/filename, matching case-insensitively when the exact path
    does not exist. Packet proc names (e.g. sedimentation_ice_FF) and the
    generated filenames (sedimentation_ice_ff.py) differ in case; the
    filesystem is case-sensitive. Mirrors the lint tool's _find_packet/_find_bridge.
    """
    exact = Path(dirpath) / filename
    if exact.exists():
        return exact
    low = filename.lower()
    d = Path(dirpath)
    if d.exists():
        for p in d.iterdir():
            if p.name.lower() == low:
                return p
    return exact  # non-existent path so callers can report the error


def _getattr_ci(mod, name: str):
    """
    getattr matching case-insensitively — module function names follow the
    file's casing (sedimentation_ice_ff), not the packet's (sedimentation_ice_FF).
    Returns None if no attribute matches.
    """
    if hasattr(mod, name):
        return getattr(mod, name)
    low = name.lower()
    for attr in dir(mod):
        if attr.lower() == low:
            return getattr(mod, attr)
    return None


def load_merged(proc: str) -> Dict[str, Any]:
    p = _find_ci(get_config().packets_dir, f"{proc}_merged.json")
    if not p.exists():
        raise FileNotFoundError(f"Missing merged packet: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def import_proc_module(proc: str):
    import sys

    pyfile = _find_ci(get_config().jax_dir, f"{proc}.py")
    if not pyfile.exists():
        raise FileNotFoundError(f"Missing translated file: {pyfile}")

    # Translated modules import their dependencies as 'from out.jax.<x> import ...',
    # which needs the project root (parent of 'out/') on sys.path.
    project_root = str(pyfile.resolve().parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    spec = importlib.util.spec_from_file_location(proc, pyfile)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {pyfile}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def import_bridge_module(proc: str):
    """Import the bridge module for a procedure (out/bridge/{proc}_bridge.py).

    The bridge file contains 'from out.jax.<proc> import ...' which requires
    the project root to be in sys.path so Python can resolve the 'out' package.
    """
    import sys

    bridge_file = _find_ci(get_config().bridge_dir, f"{proc}_bridge.py")
    if not bridge_file.exists():
        raise FileNotFoundError(f"Missing bridge file: {bridge_file}")

    # Ensure project root (parent of 'out/') is in sys.path so that
    # 'from out.jax.<proc> import ...' inside the bridge file resolves correctly.
    project_root = str(bridge_file.resolve().parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    spec = importlib.util.spec_from_file_location(f"{proc}_bridge", bridge_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load bridge module spec from {bridge_file}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def to_jax_layout_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Transpose 2D arrays from Fortran layout (ncol, nz) to JAX bridge layout (nz, ncol).

    The dummy inputs are created in Fortran layout for use with the bridge.
    Before passing them directly to _core (which expects JAX layout), we must
    transpose every 2D array.  1D arrays and scalars are left unchanged.
    """
    result = {}
    for k, v in inputs.items():
        if hasattr(v, "ndim") and v.ndim == 2:
            result[k] = jnp.asarray(v.T, dtype=jnp.float64)
        else:
            result[k] = v
    return result


def is_bridge_mode(proc: str, merged: Dict[str, Any]) -> bool:
    """Check if procedure has a bridge file (bridge mode validation)"""
    # Check for bridge Python file (case-insensitive)
    if _find_ci(get_config().bridge_dir, f"{proc}_bridge.py").exists():
        return True
    # Fallback check
    lp = (merged.get("layout_policy") or {})
    return (lp.get("mode") or "").lower() in {"bridge", "row_major", "bridge_mode"}


# ---------------------------------------------------------------------------
# Metadata-driven dummy generation
# ---------------------------------------------------------------------------

DUMMY_N = 4  # size used for every array dimension in dummy inputs

_PHASE1_META: Dict[str, Dict[str, Any]] = {}

def _load_phase1_meta() -> Dict[str, Dict[str, Any]]:
    """Load phase1_index.json once and return {proc_lower: {arg_lower: metadata}}."""
    global _PHASE1_META
    if _PHASE1_META:
        return _PHASE1_META
    p = get_config().phase1_index
    if not p.exists():
        return _PHASE1_META
    idx = json.loads(p.read_text(encoding="utf-8"))
    for entry in idx.get("procedures", []):
        name = (entry.get("name") or "").lower()
        _PHASE1_META[name] = {
            k.lower(): v
            for k, v in (entry.get("arg_metadata") or {}).items()
        }
    return _PHASE1_META


def _dummy_from_meta(meta: Dict[str, Any]) -> Any:
    """Generate a dummy value from Phase 1 arg metadata (dtype + rank)."""
    dtype = meta.get("dtype", "real")
    rank  = meta.get("rank",  0)
    shape = tuple(DUMMY_N for _ in range(rank))

    if dtype == "character":
        return ""
    if dtype == "logical":
        return False
    if dtype == "integer":
        # Integer scalars must be Python ints so they work as loop bounds,
        # static JAX arguments, and Fortran-style index expressions.
        return DUMMY_N if rank == 0 else jnp.ones(shape, dtype=jnp.int32)
    # real (default)
    return jnp.array(1.0, dtype=jnp.float64) if rank == 0 else jnp.ones(shape, dtype=jnp.float64)


_MODULE_VAR_META: Dict[str, Dict[str, Any]] = {}

def _load_module_var_meta() -> Dict[str, Dict[str, Any]]:
    """Load module-variable metadata (type/dims/init) from out/modules/*.json once."""
    global _MODULE_VAR_META
    if _MODULE_VAR_META:
        return _MODULE_VAR_META
    for f in get_config().modules_dir.glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for v in d.get("module_variables", []):
            name = (v.get("name") or "").lower()
            if name:
                _MODULE_VAR_META[name] = v
    return _MODULE_VAR_META


def _fortran_value(txt):
    """Parse a Fortran initialization literal or constant expression to a number."""
    if txt is None:
        return None
    t = txt.strip().lower()
    t = re.sub(r"_\w+\s*$", "", t)                      # kind suffix: 1.0_wp
    t = re.sub(r"([0-9.])d([+-]?\d)", r"\1e\2", t)      # d exponents: 1.d-14
    if not re.fullmatch(r"[0-9eE+\-*/(). ]+", t):
        return None                                     # references other names
    try:
        return eval(t, {"__builtins__": {}}, {})
    except Exception:
        return None


def _resolve_dim(token: str, meta: Dict[str, Dict[str, Any]]) -> int:
    """Resolve a dimension token (literal or parameter name) to an int size."""
    v = _fortran_value(token)
    if v is None:
        entry = meta.get(token.lower())
        if entry:
            v = _fortran_value(entry.get("init"))
    if v is None:
        return DUMMY_N  # runtime-set or unresolvable dimension
    return max(1, int(v))


def _dummy_from_module_var(entry: Dict[str, Any], meta: Dict[str, Dict[str, Any]]) -> Any:
    """
    Build a dummy for a MODULE variable from phase01 metadata:
      - arrays get their true declared shape (dims resolved via parameter values)
      - integer parameters get their true value as a Python int (index bounds)
      - real parameters get their true value (e.g. qsmall=1e-14, not 1.0)
      - runtime-set scalars (no init) fall back to 1.0
    """
    import numpy as np

    dtype = entry.get("type", "real")
    dims = entry.get("dims") or []
    init = _fortran_value(entry.get("init"))

    if dims:
        # NumPy (not jnp): host-side table loaders assign into these
        # in place; jnp ops auto-convert numpy inputs, so compute is unaffected.
        shape = tuple(_resolve_dim(d, meta) for d in dims)
        if dtype == "integer":
            return np.ones(shape, dtype=np.int32)
        if dtype == "logical":
            return np.zeros(shape, dtype=bool)
        return np.ones(shape, dtype=np.float64)

    if dtype == "integer":
        # Non-positive initializers are "not set yet" sentinels
        # (e.g. n_iceCat = -1) — unusable as array extents; use DUMMY_N.
        if init is not None and int(init) > 0:
            return int(init)
        return DUMMY_N
    if dtype == "logical":
        return False
    if dtype == "character":
        return ""
    return jnp.array(init if init is not None else 1.0, dtype=jnp.float64)


def default_dummy_for_arg(arg: str, proc: str, deps: Dict[str, Any]) -> Any:
    """
    Generate a dummy input for one procedure argument.

    Strategy (in priority order):
    1. Non-type-derivable special cases (file paths, format strings).
    2. Phase 1 type metadata: dtype + rank → correct Python/JAX type automatically.
       This covers every codebase without project-specific hardcoding.
    3. MODULE-variable metadata from out/modules (declared shapes for lookup
       tables, true parameter values for size constants and thresholds).
    4. Last-resort fallback (real scalar 1.0).
    """
    arg_l = arg.lower()

    # --- Project-declared argument values (config [runtime_validation.arg_values]) ---
    # Per procedure (or "*" for every procedure): arg name -> value.
    #   str containing "/"  : project-relative path (e.g. a lookup-table directory)
    #   "ramp:<step>"       : array decreasing along the last Fortran axis,
    #                         step*(n, n-1, ..., 1) — e.g. heights with kbot = nk,
    #                         so layer thicknesses are non-zero
    #   number / bool       : scalar, broadcast to the argument's declared shape
    # Use it when the uniform dummies (1.0 / DUMMY_N / "") make the *Fortran*
    # itself undefined (division by zero, unreadable table set, ...).
    arg_values = get_config().raw.get("runtime_validation", {}).get("arg_values", {}) or {}
    for scope in (proc, "*"):
        for k, v in (next((vv for kk, vv in arg_values.items() if kk.lower() == scope.lower()), None) or {}).items():
            if k.lower() != arg_l:
                continue
            if isinstance(v, str) and "/" in v:
                vp = Path(v)
                return str(vp if vp.is_absolute() else (get_config().root / vp))
            meta = _load_phase1_meta().get(proc.lower(), {}).get(arg_l) or {}
            rank = int(meta.get("rank", 0) or 0)
            shape = tuple(DUMMY_N for _ in range(rank))
            is_int = meta.get("dtype") == "integer"
            if isinstance(v, str) and v.startswith("ramp:"):
                step = float(v.split(":", 1)[1])
                if rank == 0:
                    return step
                ramp = step * jnp.arange(shape[-1], 0, -1, dtype=jnp.float64)
                return jnp.broadcast_to(ramp, shape)
            if rank == 0 or isinstance(v, (str, bool)):
                return v
            return jnp.full(shape, v, dtype=jnp.int32 if is_int else jnp.float64)

    # --- Non-type-derivable special cases ---
    if arg_l in {"filename", "file", "path", "filepath", "out", "outfile", "output_file"}:
        validation_dir = get_config().validation_dir
        validation_dir.mkdir(parents=True, exist_ok=True)
        return str(validation_dir / f"{proc}.txt")
    if arg_l in {"output_info", "header", "title", "label"}:
        return f"[runtime_validate] {proc}"
    if arg_l in {"fmt", "format"}:
        return "DUMMY"
    if arg_l == "mode":
        return "a"

    # --- Phase 1 metadata-driven generation ---
    proc_meta = _load_phase1_meta().get(proc.lower(), {})
    meta = proc_meta.get(arg_l)
    if meta:
        return _dummy_from_meta(meta)

    # --- MODULE variables: use phase01 module metadata (type/dims/init) ---
    # Lookup tables get their true declared shapes, integer size parameters
    # their true values, real parameters their true values (e.g. qsmall=1e-14).
    mv_meta = _load_module_var_meta()
    entry = mv_meta.get(arg_l)
    if entry is not None:
        return _dummy_from_module_var(entry, mv_meta)

    # --- Last-resort fallback: real scalar 1.0 ---
    return jnp.array(1.0, dtype=jnp.float64)


def build_inputs(proc: str, deps: Dict[str, Any], module_vars_used: List[str] = None) -> Dict[str, Any]:
    """
    Build dummy inputs for procedure including MODULE variables.
    
    Args:
        proc: Procedure name
        deps: Dependencies dict
        module_vars_used: MODULE variables to add as inputs
    
    Returns:
        Dict of input name -> dummy value
    """
    module_vars_used = module_vars_used or []
    
    args: List[str] = deps.get("args", [])
    inputs = {a: default_dummy_for_arg(a, proc, deps) for a in args}
    
    # Add MODULE variables as inputs
    for var in module_vars_used:
        if var not in inputs:  # Don't override if already in args
            inputs[var] = default_dummy_for_arg(var, proc, deps)
    
    return inputs


def call_fn(fn, kwargs: Dict[str, Any]) -> Any:
    """
    Call fn using only the keyword arguments it actually accepts.
    Prevents failures when wrappers omit INTENT(OUT) args, etc.
    """
    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return fn(**filtered)


def resolve_call_kwargs(fn, inputs: Dict[str, Any], proc: str, deps: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map dummy inputs onto fn's actual parameter names and fill the gaps:
      - case-insensitive name matching (packet arg names are Fortran-cased,
        function params follow the Python file's casing, e.g. qi vs Qi)
      - accepted params not covered by inputs (e.g. vestigial module-var
        parameters some wrappers/bridges carry) get generated dummies —
        module-var metadata provides correct shapes/values for them.
    """
    sig = inspect.signature(fn)
    by_lower = {k.lower(): v for k, v in inputs.items()}
    kwargs: Dict[str, Any] = {}
    for p in sig.parameters:
        if p.lower() in by_lower:
            kwargs[p] = by_lower[p.lower()]
        else:
            kwargs[p] = default_dummy_for_arg(p, proc, deps)

    # Array flavor by callee contract: JAX procs get jnp arrays (they never
    # mutate tables in place, and jnp clamps out-of-range dummy indices where
    # numpy raises); host-side procs (jax_required=False) keep numpy so
    # in-place assignment works (e.g. an init routine filling lookup tables).
    if deps.get("jax_required", True):
        import numpy as _np
        kwargs = {
            k: (jnp.asarray(v) if isinstance(v, _np.ndarray) else v)
            for k, v in kwargs.items()
        }
    return kwargs


def is_finite_tree(x) -> bool:
    leaves = jax.tree_util.tree_leaves(x)
    for v in leaves:
        if isinstance(v, (int, float, bool, str)):
            continue
        if hasattr(v, "dtype") and jnp.issubdtype(v.dtype, jnp.number):
            if not bool(jnp.all(jnp.isfinite(v))):
                return False
    return True


@dataclass
class RuntimeResult:
    proc: str
    imported: bool
    wrapper_called: bool
    core_jit_called: bool
    wrapper_ok: bool
    core_ok: bool
    wrapper_exception: str | None
    core_exception: str | None
    wrapper_finite: bool | None
    core_finite: bool | None
    notes: List[str]


def validate_proc(proc: str) -> RuntimeResult:
    merged = load_merged(proc)

    # Bridge-only enforcement (to avoid silently validating old Policy A code)
    if not is_bridge_mode(proc, merged):
        return RuntimeResult(
            proc=proc,
            imported=False,
            wrapper_called=False,
            core_jit_called=False,
            wrapper_ok=False,
            core_ok=False,
            wrapper_exception=f"{proc}: not in Bridge Mode (missing out/bridge/{proc}_bridge_metadata.json)",
            core_exception=None,
            wrapper_finite=None,
            core_finite=None,
            notes=["Bridge-only validator refused to run."],
        )

    deps = merged["deps"]
    effects = deps.get("effects", {}) or {}
    has_io = bool(effects.get("has_io", False))
    scalar_only = _is_scalar_only_proc(merged)
    notes: List[str] = []

    # Extract MODULE variables — same rule the generators use, so the
    # built inputs match the wrapper signature actually generated.
    module_vars_used = packet_module_vars(merged)

    # Import module
    try:
        mod = import_proc_module(proc)
        imported = True
    except Exception:
        return RuntimeResult(
            proc=proc,
            imported=False,
            wrapper_called=False,
            core_jit_called=False,
            wrapper_ok=False,
            core_ok=False,
            wrapper_exception=traceback.format_exc(),
            core_exception=None,
            wrapper_finite=None,
            core_finite=None,
            notes=["Import failed"],
        )

    wrapper_name = proc
    core_name = f"{proc}_core"
    bridge_name = f"{proc}_bridge"

    # --- Wrapper: prefer bridge function if bridge file exists ---
    # All name lookups are case-insensitive: packet proc names and generated
    # file/function names differ in case (e.g. sedimentation_ice_FF vs
    # sedimentation_ice_ff).
    bridge_path = _find_ci(get_config().bridge_dir, f"{proc}_bridge.py")
    if bridge_path.exists():
        try:
            bridge_mod = import_bridge_module(proc)
            wrapper_fn = _getattr_ci(bridge_mod, bridge_name)
            if wrapper_fn is None:
                raise AttributeError(f"Bridge module has no function {bridge_name}")
            notes.append(f"Wrapper validation uses bridge ({bridge_name}) with Fortran-layout inputs.")
        except Exception as bridge_exc:
            # Fall back to direct JAX wrapper and surface the reason
            notes.append(f"Bridge import failed ({bridge_exc}); falling back to direct {wrapper_name}.")
            wrapper_fn = _getattr_ci(mod, wrapper_name)
            if wrapper_fn is None:
                return RuntimeResult(
                    proc=proc, imported=True,
                    wrapper_called=False, core_jit_called=False,
                    wrapper_ok=False, core_ok=False,
                    wrapper_exception=f"Missing wrapper function: {wrapper_name}",
                    core_exception=None, wrapper_finite=None, core_finite=None,
                    notes=notes + ["Wrapper missing"],
                )
    else:
        wrapper_fn = _getattr_ci(mod, wrapper_name)
        if wrapper_fn is None:
            return RuntimeResult(
                proc=proc, imported=True,
                wrapper_called=False, core_jit_called=False,
                wrapper_ok=False, core_ok=False,
                wrapper_exception=f"Missing wrapper function: {wrapper_name}",
                core_exception=None, wrapper_finite=None, core_finite=None,
                notes=["Wrapper missing"],
            )

    # Core is required for array procedures only (not scalar-only, not wrapper-only I/O).
    core_fn = _getattr_ci(mod, core_name)
    if core_fn is None and (not has_io) and (not scalar_only):
        return RuntimeResult(
            proc=proc,
            imported=True,
            wrapper_called=False,
            core_jit_called=False,
            wrapper_ok=False,
            core_ok=False,
            wrapper_exception=f"Missing core function: {core_name}",
            core_exception=None,
            wrapper_finite=None,
            core_finite=None,
            notes=["Core missing"],
        )
    if core_fn is None and has_io:
        notes.append("Core missing but effects.has_io=True -> wrapper-only allowed; skipping core jit validation.")
    if core_fn is None and scalar_only:
        notes.append("Scalar-only procedure: no _core expected. Skipping JIT validation.")

    # Inputs (including MODULE variables)
    inputs = build_inputs(proc, deps, module_vars_used)

    # Call wrapper
    wrapper_called = False
    wrapper_ok = False
    wrapper_exc = None
    wrapper_out = None
    wrapper_finite = None

    try:
        wrapper_called = True
        wrapper_out = wrapper_fn(**resolve_call_kwargs(wrapper_fn, inputs, proc, deps))
        wrapper_ok = True
        wrapper_finite = is_finite_tree(wrapper_out)
    except Exception:
        wrapper_exc = traceback.format_exc()

    # Call jitted core if allowed (bridge mode: pass row-major args directly; no aF, no swapaxes)
    core_jit_called = False
    core_ok = False
    core_exc = None
    core_out = None
    core_finite = None

    if has_io or scalar_only or (core_fn is None):
        if not scalar_only:  # scalar-only already noted above
            notes.append("effects.has_io=True or core missing -> skipping core jit validation.")
    else:
        try:
            core_jit_called = True

            # Resolve inputs onto the core's actual parameter names
            # (case-insensitive, gaps filled from module-var metadata), then
            # transpose 2D arrays: core expects JAX bridge layout (nz, ncol)
            # while dummy inputs are in Fortran layout (ncol, nz).
            core_kwargs = to_jax_layout_inputs(
                resolve_call_kwargs(core_fn, inputs, proc, deps)
            )

            # Enforce: _core must not accept any string parameter.
            string_params = sorted(
                name for name, val in core_kwargs.items() if isinstance(val, str)
            )
            if string_params:
                raise TypeError(
                    f"_core accepts string parameter(s) {string_params} — strings "
                    f"must stay in the wrapper only (see jax_jit_boundary_rules.md "
                    f"'Strings MUST NOT be passed to _core')."
                )

            static_argnames = []
            for arg_name, arg_value in core_kwargs.items():
                # Static args: non-array scalars (int, bool, float).
                # Strings are rejected above, so they never reach this loop.
                if isinstance(arg_value, (int, bool, float)):
                    static_argnames.append(arg_name)

            # Create JIT with static_argnames if needed
            if static_argnames:
                jitted = jax.jit(core_fn, static_argnames=static_argnames)
            else:
                jitted = jax.jit(core_fn)

            core_out = call_fn(jitted, core_kwargs)

            core_ok = True
            core_finite = is_finite_tree(core_out)
        except Exception:
            core_exc = traceback.format_exc()

    return RuntimeResult(
        proc=proc,
        imported=imported,
        wrapper_called=wrapper_called,
        core_jit_called=core_jit_called,
        wrapper_ok=wrapper_ok,
        core_ok=core_ok if (not has_io and not scalar_only and core_fn is not None) else True,
        wrapper_exception=wrapper_exc,
        core_exception=core_exc,
        wrapper_finite=wrapper_finite,
        core_finite=core_finite if (not has_io and not scalar_only and core_fn is not None) else None,
        notes=notes,
    )


def main():
    cfg = get_config()
    merged_files = sorted(cfg.packets_dir.glob("*_merged.json"))
    if not merged_files:
        raise SystemExit(f"No *_merged.json found in {cfg.packets_dir}/. Run Phase 3.1 merge first.")

    procs = [p.name.replace("_merged.json", "") for p in merged_files]

    out_dir = cfg.validation_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for proc in procs:
        print(f"\n=== Runtime validate {proc} ===")
        res = validate_proc(proc)
        print(f"imported={res.imported} wrapper_ok={res.wrapper_ok} core_ok={res.core_ok}")

        if not (res.imported and res.wrapper_ok and res.core_ok):
            failures += 1

        (out_dir / f"{proc}_runtime.json").write_text(
            json.dumps(asdict(res), indent=2),
            encoding="utf-8",
        )

        if res.wrapper_exception:
            print("[WRAPPER EXC]\n" + res.wrapper_exception)
        if res.core_exception:
            print("[CORE EXC]\n" + res.core_exception)
        for note in res.notes:
            print("[NOTE] " + note)

    raise SystemExit(0 if failures == 0 else 2)


if __name__ == "__main__":
    import argparse
    from framework_config import init as init_config, add_config_arg

    ap = argparse.ArgumentParser(description="Phase 05.2: runtime validation")
    add_config_arg(ap)
    init_config(ap.parse_args().config)
    main()