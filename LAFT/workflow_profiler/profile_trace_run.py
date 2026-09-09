#!/usr/bin/env python3
"""Profile the project's staged translation with jax.profiler.trace.

This is the *design / structure* profiling pass — the XLA/HLO-graph
counterpart to the nsys hardware pass (profile_run.py). It answers: did the
translation vectorize each loop, and how did XLA fuse the op graph? The
authoritative `while_iterations` signal comes from here, measured directly.

Everything project-specific comes from [profiler] in config/project.toml
(target procedure, staged code path, ncol points) and the input state is the
project driver's own mid-run state, tiled to ncol (profiler_inputs.py). The
profiled call is the procedure's BRIDGE call — the production entry — so for
an orchestrator-style translation the trace covers the
host orchestration plus ALL of its internal jitted stage cores.

HLO note: an orchestrator is not itself one jittable function, so the old
single-function `lower().compile().as_text()` dump does not apply. Instead
the process runs with `--xla_dump_to`, and every module XLA compiles during
the main-ncol warmup (all stage cores + jitted callees) is concatenated into
`<proc>_hlo.txt`. parse_hlo's counters are line-based, so the aggregate
metrics (fusion/scatter/gather/while, ops_per_fusion) remain valid across
the concatenation.

The script:

  1. Injects the staged file as out.jax.<proc> and imports the bridge.
  2. Loads the captured driver state tiled to --ncol; JIT-warms (compilation
     excluded from the trace); collects the XLA HLO dump.
  3. Runs the profiled call(s) inside `jax.profiler.trace(<trace_dir>)`.
  4. Repeats at a second, smaller ncol (--ncol-check) and emits
     `while_iterations_scaling` = wi(ncol_hi) / wi(ncol_lo). This is the
     two-ncol scaling check: a subcycling while-loop inside vectorized code
     has scaling ~ 1 (iteration count is ncol-independent), while a
     sequential column loop scales with ncol. Disable with --ncol-check 0.
  5. Parses the emitted Perfetto trace(s) and writes `<proc>_trace.json`
     (trace metrics + the `hlo` block) and `<proc>_code_info.json`.

This job is the iteration loop's ONLY profiling job and writes everything
diagnose.py needs. The nsys runner (profile_run.py) is the optional
post-convergence hardware characterization, not part of the loop.

Usage (inside the PBS job; --code-path defaults to [profiler].staged_code):

    python3 workflow_profiler/profile_trace_run.py \\
        --out-dir out/profiled/iteration_<N> \\
        --ncol    1000
"""

from __future__ import annotations

import os
os.environ["JAX_ENABLE_X64"] = "1"  # MUST come before `import jax`

import argparse
import json
import sys
import time
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


def _parse_args() -> argparse.Namespace:
    # Parsed BEFORE importing jax: the XLA dump directory must be in
    # XLA_FLAGS before the backend initializes.
    cfg = get_config().section("profiler")
    if not cfg:
        raise SystemExit("No [profiler] section in config/project.toml")

    p = argparse.ArgumentParser(
        description=f"jax.profiler.trace runner for {cfg['target_proc']}")
    p.add_argument("--code-path", type=Path,
                   default=PROJECT_ROOT / cfg["staged_code"],
                   help=f"Staged translation (default: {cfg['staged_code']}).")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Iteration directory; <proc>_trace.json lands here, "
                        "raw trace under <out-dir>/<trace-subdir>/.")
    p.add_argument("--trace-subdir", default="jax_trace",
                   help="Subdirectory for the raw Perfetto trace "
                        "(default: jax_trace).")
    p.add_argument("--ncol", type=int, default=int(cfg.get("ncol", 1000)),
                   help="Number of columns.")
    p.add_argument("--n-calls", type=int,
                   default=int(cfg.get("n_calls", 10)),
                   help="Profiled JIT-warm calls inside the main trace "
                        "(the ncol-check trace always uses 1 call).")
    p.add_argument("--ncol-check", type=int,
                   default=int(cfg.get("ncol_check", 100)),
                   help="Second ncol for the while_iterations scaling check "
                        "(0 disables).")
    return p.parse_args()


ARGS = _parse_args()
HLO_DUMP_DIR = (ARGS.out_dir / "hlo_dump").resolve()
HLO_DUMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "")
    + f" --xla_dump_to={HLO_DUMP_DIR} --xla_dump_hlo_as_text"
).strip()

import jax  # noqa: E402  (AFTER the XLA_FLAGS setup above)
jax.config.update("jax_enable_x64", True)

from profiler_inputs import (  # noqa: E402
    ensure_cache_subprocess, get_profiling_inputs, load_staged_bridge,
    profiler_cfg,
)
from profile_run import (  # noqa: E402
    count_jax_operations, count_output_arrays, infer_nz, write_code_info,
)
from parse_jax_trace import (  # noqa: E402
    compute_while_scaling, find_trace_json, parse_trace,
)
from parse_hlo import parse_hlo_metrics  # noqa: E402


def collect_hlo_dump(out_dir: Path, proc: str, snapshot: set) -> tuple[Path, dict]:
    """Concatenate the optimized-HLO modules dumped since `snapshot` into
    <proc>_hlo.txt and parse aggregate metrics."""
    files = sorted(p for p in HLO_DUMP_DIR.glob("*after_optimizations.txt")
                   if p not in snapshot)
    if not files:
        print("[trace] WARNING: no *after_optimizations.txt modules in the "
              "XLA dump — HLO metrics will be empty")
    parts = []
    for p in files:
        parts.append(f"// ==== module dump: {p.name} ====\n")
        parts.append(p.read_text())
        parts.append("\n")
    hlo_text = "".join(parts)
    hlo_path = out_dir / f"{proc}_hlo.txt"
    hlo_path.write_text(hlo_text)
    print(f"[trace] -> {hlo_path}  ({len(files)} compiled modules)")
    return hlo_path, parse_hlo_metrics(hlo_text)


def trace_once(fn, trace_dir: Path, ncol: int, n_calls: int) -> dict:
    """One warmup + n_calls traced calls at the given ncol; parsed metrics."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    inputs = get_profiling_inputs(ncol)

    print(f"[trace] ncol={ncol}: warmup (compilation excluded from trace) "
          f"... ", end="", flush=True)
    jax.block_until_ready(fn(**inputs))
    print("done")

    print(f"[trace] ncol={ncol}: {n_calls} profiled call(s) inside "
          f"jax.profiler.trace ... ", end="", flush=True)
    t0 = time.perf_counter()
    with jax.profiler.trace(str(trace_dir), create_perfetto_link=False):
        for _ in range(n_calls):
            jax.block_until_ready(fn(**inputs))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"done ({elapsed_ms:.2f} ms wall)")

    trace_json = find_trace_json(trace_dir)
    print(f"[trace] parsing {trace_json}")
    metrics = parse_trace(trace_json, n_calls=n_calls)
    metrics["ncol"] = ncol
    metrics["n_calls"] = n_calls
    return metrics


def run(code_path: Path, out_dir: Path, trace_subdir: str,
        ncol: int, n_calls: int, ncol_check: int) -> None:
    cfg = profiler_cfg()
    proc = cfg["target_proc"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[trace] target    = {proc} (bridge call)")
    print(f"[trace] code_path = {code_path}")
    print(f"[trace] out_dir   = {out_dir}")
    print(f"[trace] ncol={ncol}  n_calls={n_calls}  ncol_check={ncol_check}")

    fn = load_staged_bridge(code_path)

    # --- main-ncol trace (its warmup fills the XLA dump) ----------------
    pre_warmup_snapshot = set(HLO_DUMP_DIR.glob("*after_optimizations.txt"))
    metrics = trace_once(fn, out_dir / trace_subdir, ncol, n_calls)

    # --- compiled-HLO extraction (all modules compiled at the main ncol) --
    hlo_path, hlo_metrics = collect_hlo_dump(out_dir, proc, pre_warmup_snapshot)
    metrics["hlo"] = hlo_metrics
    metrics["hlo_file"] = str(hlo_path)
    metrics["hlo_module_count"] = len(
        set(HLO_DUMP_DIR.glob("*after_optimizations.txt")) - pre_warmup_snapshot)

    # --- two-ncol scaling check --------------------------------------
    # while_iterations alone cannot distinguish legitimate subcycling
    # (e.g. substeps inside vectorized code) from partial sequentialization.
    # Its scaling across two ncol values can: ncol-independent (~1) means
    # vectorized, tracking ncol_ratio means sequential.
    do_check = ncol_check and ncol_check > 0 and ncol_check != ncol
    metrics["while_iterations_scaling"] = None
    metrics["ncol_ratio"] = None
    metrics["scaling_check"] = None
    if do_check:
        # Structural (does wi scale with ncol?), not a timing measurement —
        # one call keeps it cheap.
        check = trace_once(fn, out_dir / f"{trace_subdir}_check",
                           ncol_check, 1)
        metrics["scaling_check"] = {
            k: check.get(k) for k in
            ("ncol", "n_calls", "while_iterations", "while_iterations_raw",
             "source_trace")
        }
        scaling, ncol_ratio = compute_while_scaling(
            ncol, metrics["while_iterations"],
            ncol_check, check["while_iterations"])
        metrics["while_iterations_scaling"] = scaling
        metrics["ncol_ratio"] = ncol_ratio

    # Reproducibility metadata: which stack produced these numbers.
    metrics["jax_version"] = jax.__version__
    metrics["device_kind"] = jax.devices()[0].device_kind
    metrics["xla_flags"] = os.environ.get("XLA_FLAGS")
    metrics["target_proc"] = proc

    out_path = out_dir / f"{proc}_trace.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[trace] -> {out_path}")

    # code_info: static counts diagnose.py needs. output_array_count is
    # measured live from one extra (JIT-warm, untraced) call.
    ret = fn(**get_profiling_inputs(ncol))
    nz = infer_nz(get_profiling_inputs(ncol))
    code_info_path = write_code_info(
        out_dir, code_path, ncol, nz, n_calls, count_output_arrays(ret))
    print(f"[trace] -> {code_info_path}")

    print(f"[trace]   while_iterations={metrics['while_iterations']}  "
          f"while_iterations_scaling={metrics['while_iterations_scaling']}  "
          f"(ncol_ratio={metrics['ncol_ratio']})")
    print(f"[trace]   xla_event_count={metrics['xla_event_count']}  "
          f"unique_hlo_ops={metrics['unique_hlo_ops']}")
    print(f"[trace]   hlo: {hlo_metrics['total_instructions']} instructions "
          f"across {metrics['hlo_module_count']} modules, "
          f"ops={hlo_metrics['ops']}")
    timing = metrics.get("timing", {})
    print(f"[trace]   timing medians (n={timing.get('end_to_end_ms', {}).get('n')}): "
          f"end_to_end_ms={metrics['end_to_end_ms']}  "
          f"gpu_exec_ms={metrics['gpu_exec_ms']}  "
          f"dispatch_overhead_ms={metrics['dispatch_overhead_ms']}")
    for w in metrics.get("warnings", []):
        print(f"[trace]   WARNING: {w}")


def main() -> int:
    args = ARGS
    if not args.code_path.is_file():
        print(f"ERROR: code path not found: {args.code_path}", file=sys.stderr)
        return 1

    # Build the state cache OUTSIDE this process so the driver spin-up does
    # not add its own modules to this process's XLA dump.
    ensure_cache_subprocess()

    print(f"[trace] JAX backend : {jax.devices()[0].platform}")
    print(f"[trace] JAX devices : {jax.devices()}")

    run(args.code_path, args.out_dir, args.trace_subdir,
        args.ncol, args.n_calls, args.ncol_check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
