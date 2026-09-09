#!/usr/bin/env python3
"""nsys hardware characterization of the project's profiled translation.

MANDATORY, POST-CONVERGENCE pass — NOT part of the iteration loop. The
loop's single profiling job is profile_trace_run.py (jax.profiler.trace
+ compiled-HLO), which supplies everything diagnose.py gates on. Run
this pass once, after the loop converges (PROFILE_WORKFLOW.md Step 6),
to characterize the final code at the hardware layer (device kernel
times, CUDA-graph sync idle, memcpy timing) for the headroom analysis
and the report; diagnose.py attaches its stats via the optional
--nsys-json as an informational hardware_corroboration block that
never affects production_ready.

Everything project-specific comes from [profiler] in config/project.toml:
the target procedure, the staged code path, and the input state — captured
once from the project's validated driver mid-run and tiled to the requested
ncol (see profiler_inputs.py). The profiled call is the procedure's BRIDGE
call, i.e. exactly the production entry: for an orchestrator-style
translation that covers the host orchestration plus all
of its internal jitted stage cores.

This is the inner Python entrypoint that `pbsJobs/jax_gpu_profile_nsys.sh`
wraps with `nsys profile`. Given the staged translation, it:

  1. Injects the staged file as out.jax.<proc> and imports the bridge.
  2. Loads the captured driver state, tiled to the requested ncol.
  3. Runs warmup, then `n_calls` profiled calls inside NVTX ranges so the
     diagnoser can recover the iteration count from the timeline.
  4. Writes `<proc>_code_info.json` (ncol, nz, n_calls, jax_op_count,
     output_array_count, source_file) to the output directory — this feeds
     the diagnoser alongside the nsys-derived stats.

Timing is NOT measured here. A host-side wall clock taken inside the
nsys-instrumented process is perturbed and unreliable; the authoritative
timing is the tool-measured device kernel time from the nsys
`cuda_gpu_kern_sum` export (surfaced by diagnose.py as `gpu_kernel_time_ms`).

The script does NOT export nsys stats itself; the PBS wrapper runs
`nsys stats ... --format json` after this process exits and the .nsys-rep
file has been written.

Usage (inside the PBS job; --code-path defaults to [profiler].staged_code):

    nsys profile --trace=cuda,nvtx,osrt --output=<prefix> \\
        python3 workflow_profiler/profile_run.py \\
            --out-dir out/profiled/iteration_<N> \\
            --ncol    1000
"""

from __future__ import annotations

import os
os.environ["JAX_ENABLE_X64"] = "1"  # MUST come before `import jax`

import argparse
import ast
import json
import sys
from pathlib import Path

# workflow_profiler/ is a per-project SYMLINK to LAFT/workflow_profiler/, so
# Path(__file__).resolve() follows it back to the shared LAFT/ tree — using
# that as PROJECT_ROOT would silently resolve every per-project path (out/,
# data/, ...) against LAFT/ instead of the actual project root. See
# profiler_inputs.py for the full explanation; PROJECT_ROOT here must come
# from get_config().root (cwd-discovered), not SCRIPT_DIR.parent.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "config"))

from framework_config import get_config  # noqa: E402

PROJECT_ROOT = get_config().root
sys.path.insert(0, str(PROJECT_ROOT))

from profiler_inputs import (  # noqa: E402
    ensure_cache_subprocess, get_profiling_inputs, load_staged_bridge,
    profiler_cfg,
)

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)

try:
    import nvtx
    HAS_NVTX = True
except ImportError:
    HAS_NVTX = False
    print("WARNING: nvtx not available — nsys profiling runs will lack "
          "per-call NVTX range labels in the timeline (cosmetic only; "
          "while_iterations comes from the jax.profiler.trace pass).")


def infer_nz(inputs: dict) -> int:
    """Vertical levels: the driver's kte argument when present, else the
    second axis of a representative 2D field."""
    if "kte" in inputs:
        return int(inputs["kte"])
    for v in inputs.values():
        if hasattr(v, "ndim") and getattr(v, "ndim", 0) == 2:
            return int(v.shape[1])
    return -1


def count_jax_operations(code_path: Path) -> int:
    """Count jnp.* / lax.* calls in the staged file. Used by diagnose.py to
    compute the (report-only) fusion-efficiency proxy."""
    tree = ast.parse(code_path.read_text())
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if func.value.id in ("jnp", "lax"):
                    count += 1
    return count


def count_output_arrays(ret) -> int:
    """Number of arrays the bridge call returns — measured live, not by AST,
    so it is correct for any translation structure."""
    if isinstance(ret, tuple):
        return sum(1 for item in ret if hasattr(item, "shape"))
    return 1 if hasattr(ret, "shape") else 0


def write_code_info(out_dir: Path, code_path: Path, ncol: int, nz: int,
                    n_calls: int, output_array_count: int) -> Path:
    cfg = profiler_cfg()
    code_info = {
        "ncol": ncol,
        "nz": nz,
        "n_calls": n_calls,
        "jax_op_count": count_jax_operations(code_path),
        "output_array_count": output_array_count,
        "source_file": str(code_path),
    }
    path = out_dir / f"{cfg['target_proc']}_code_info.json"
    path.write_text(json.dumps(code_info, indent=2) + "\n")
    return path


def run(code_path: Path, out_dir: Path, ncol: int, n_calls: int) -> None:
    cfg = profiler_cfg()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[profile] target    = {cfg['target_proc']} (bridge call)")
    print(f"[profile] code_path = {code_path}")
    print(f"[profile] out_dir   = {out_dir}")

    fn = load_staged_bridge(code_path)
    inputs = get_profiling_inputs(ncol)
    nz = infer_nz(inputs)
    print(f"[profile] ncol={ncol}  nz={nz}  n_calls={n_calls}")

    print("[profile] warmup ... ", end="", flush=True)
    ret = jax.block_until_ready(fn(**inputs))
    print("done")

    label = cfg.get("nvtx_label", f"{cfg['target_proc']}_profiled_calls")
    if HAS_NVTX:
        nvtx.push_range(label)

    # Run the JIT-warm calls so nsys can profile the device kernels. No
    # host-side wall clock is taken: a timer inside the nsys-instrumented
    # process is perturbed by the instrumentation. The authoritative timing
    # is the tool-measured device kernel time from the nsys cuda_gpu_kern_sum
    # export. block_until_ready waits on the full output pytree so every
    # call's kernels land in the timeline.
    print(f"[profile] {n_calls} profiled calls ... ", end="", flush=True)
    for i in range(n_calls):
        if HAS_NVTX:
            nvtx.push_range(f"call_{i}")
        jax.block_until_ready(fn(**inputs))
        if HAS_NVTX:
            nvtx.pop_range()
    print("done")

    if HAS_NVTX:
        nvtx.pop_range()

    code_info_path = write_code_info(
        out_dir, code_path, ncol, nz, n_calls, count_output_arrays(ret))
    print(f"[profile] → {code_info_path}")


def main() -> int:
    cfg = profiler_cfg()
    p = argparse.ArgumentParser(
        description=f"nsys profiling runner for {cfg['target_proc']}")
    p.add_argument("--code-path", type=Path,
                   default=PROJECT_ROOT / cfg["staged_code"],
                   help="Staged translation to profile "
                        f"(default: {cfg['staged_code']}).")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Directory for code_info.json (and where nsys-rep will land).")
    p.add_argument("--ncol", type=int, default=int(cfg.get("ncol", 1000)),
                   help="Number of columns.")
    p.add_argument("--n-calls", type=int, default=6,
                   help="Number of profiled JIT-warm calls (default: 6).")
    args = p.parse_args()

    if not args.code_path.is_file():
        print(f"ERROR: code path not found: {args.code_path}", file=sys.stderr)
        return 1

    # Build the state cache OUTSIDE this process so the driver spin-up does
    # not appear in the nsys timeline.
    ensure_cache_subprocess()

    print(f"[profile] JAX backend : {jax.devices()[0].platform}")
    print(f"[profile] JAX devices : {jax.devices()}")

    run(args.code_path, args.out_dir, args.ncol, args.n_calls)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
