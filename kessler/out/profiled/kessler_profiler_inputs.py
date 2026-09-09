#!/usr/bin/env python3
"""[profiler].inputs_script for Kessler — capture_state/tile_state for the
project's driver (a single-shot, real-128-column batch call, no
time-stepping loop).

Loaded dynamically by LAFT/workflow_profiler/profiler_inputs.py's
dispatcher; not meant to be run standalone. See PROFILE_WORKFLOW.md.

1. **State capture** (``capture_state``): import the driver module
   ([profiler].driver_module), intercept its bridge call
   ([profiler].bridge_func) with a wrapper that saves the full keyword-
   argument state, then abort the driver loop. The driver file is NEVER
   modified; correctness stays owned by the validation chain. Kessler's
   driver calls the bridge exactly once — there is no ``it`` kwarg and no
   module-level ``DT`` to key a capture minute off of, so the first (and
   only) call is captured unconditionally. The captured state (128 real
   columns, the driver's actual grid) is cached to [profiler].state_cache.

2. **Column tiling** (``tile_state``): the captured state is what the
   *bridge itself receives* — Fortran/col-major convention, shape
   (ncol, nz), columns on axis 0. (The bridge's own ``to_row_major_2d``
   transposes to (nz, ncol) for kessler_run_core's
   ``vmap(in_axes=(...,1,...))``, but that transpose happens *after*
   capture, inside the bridge, on whatever data tile_state hands it — so
   tiling operates on the pre-transpose, axis-0-columns shape.)
   Per-column arrays are cyclically tiled from their real captured column
   count up to ncol — e.g. 128 real columns cycled to fill 1000 — which
   preserves the real per-column heterogeneity of the captured data
   instead of replicating one column. Fields listed in
   [profiler].noise_fields additionally get per-column multiplicative
   noise (1 + amplitude * N(0,1), clipped at 0) so the cycled repeats
   aren't bit-identical; the scalar named in [profiler].ncol_args (``ncol``)
   is set to the target ncol.
"""

from __future__ import annotations

import functools
import importlib.util
import pickle
from pathlib import Path

import numpy as np

from framework_config import get_config

PROJECT_ROOT = get_config().root


def _cfg() -> dict:
    return get_config().section("profiler")


class _CaptureDone(Exception):
    """Raised inside the driver loop once the bridge call has been captured."""


def _to_numpy(v):
    """Detach any JAX array to numpy; pass everything else through."""
    if hasattr(v, "__array__") and not isinstance(v, np.ndarray):
        return np.asarray(v)
    return v


def capture_state(force: bool = False) -> dict:
    """Return the kwargs of the driver's one bridge call.

    Uses the cache at [profiler].state_cache when present; otherwise imports
    the driver module, intercepts its bridge call, runs it, saves, and
    aborts before the driver writes its own output files.
    """
    cfg = _cfg()
    cache = PROJECT_ROOT / cfg["state_cache"]
    if cache.exists() and not force:
        with open(cache, "rb") as f:
            return pickle.load(f)

    driver_path = PROJECT_ROOT / cfg["driver_module"]
    bridge_func = cfg["bridge_func"]

    print(f"[kessler-inputs] capturing driver state at the first "
          f"{bridge_func} call from {driver_path.name} (single-shot driver, "
          f"no time-stepping loop; cached to {cache.name})")

    spec = importlib.util.spec_from_file_location("_profiler_driver", driver_path)
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)

    if not hasattr(drv, bridge_func):
        raise SystemExit(
            f"driver module {driver_path} has no symbol '{bridge_func}' to intercept")

    # Redirect the driver's output files to a scratch dir: drv.main() opens
    # its output for WRITING (including DRIVER_JSON, written unconditionally
    # in a finally block), and an aborted capture run would otherwise
    # overwrite the validated out/driver/kessler_driver.json with a
    # FAIL/traceback stub.
    capture_tmp = cache.parent / "capture_tmp"
    capture_tmp.mkdir(parents=True, exist_ok=True)
    for attr in cfg.get("driver_output_attrs", ["OUT_DIR", "OUT_DAT", "OUT_JSON"]):
        if hasattr(drv, attr):
            orig = Path(getattr(drv, attr))
            setattr(drv, attr,
                    capture_tmp if attr == "OUT_DIR" else capture_tmp / orig.name)

    real_bridge = getattr(drv, bridge_func)
    captured: dict = {}

    # functools.wraps: the driver introspects inspect.signature(<bridge>) to
    # enumerate module-var parameters — the interceptor must look identical.
    @functools.wraps(real_bridge)
    def intercepting(**kwargs):
        for k, v in kwargs.items():
            captured[k] = _to_numpy(v)
        raise _CaptureDone

    setattr(drv, bridge_func, intercepting)
    try:
        drv.main()
        raise SystemExit(
            f"driver finished without ever calling {bridge_func} — "
            f"nothing to capture")
    except _CaptureDone:
        pass
    finally:
        setattr(drv, bridge_func, real_bridge)

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(captured, f)
    n_arrays = sum(1 for v in captured.values() if isinstance(v, np.ndarray))
    print(f"[kessler-inputs] captured {len(captured)} args ({n_arrays} arrays) "
          f"-> {cache}")
    return captured


def tile_state(state: dict, ncol: int, seed: int = 42) -> dict:
    """Cyclically tile the captured 128-column state up to ncol columns."""
    cfg = _cfg()
    noise_fields = set(cfg.get("noise_fields", []))
    amplitude = float(cfg.get("noise_amplitude", 0.0))
    ncol_args = list(cfg.get("ncol_args", []))
    column_axis = int(cfg.get("column_axis", 0))

    src_ncol = None
    for nm in ncol_args:
        if nm in state:
            src_ncol = int(state[nm])
            break
    if src_ncol is None:
        raise SystemExit(
            "tile_state: none of [profiler].ncol_args were found in the "
            "captured state — cannot determine the source column count")

    rng = np.random.default_rng(seed)
    col_factor = 1.0 + amplitude * rng.standard_normal(ncol)
    cyclic_idx = np.arange(ncol) % src_ncol

    out: dict = {}
    for name, v in state.items():
        if name in ncol_args:
            out[name] = ncol
        elif (isinstance(v, np.ndarray) and v.ndim > column_axis
                and v.shape[column_axis] == src_ncol):
            tiled = np.take(v, cyclic_idx, axis=column_axis)
            if name in noise_fields and np.issubdtype(tiled.dtype, np.floating):
                shape = [1] * tiled.ndim
                shape[column_axis] = ncol
                fac = col_factor.reshape(shape)
                tiled = np.maximum(0.0, tiled * fac) if tiled.min() >= 0.0 \
                    else tiled * fac
            out[name] = np.ascontiguousarray(tiled)
        else:
            out[name] = v
    return out
