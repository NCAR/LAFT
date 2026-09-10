"""
Profiling Runner: kessler_run_core  (JD-driver inputs)
=======================================================

Runs one profiling mode at a time against kessler_run_core for the four
paper-1 translation archives (Claude, GPT, Gemini, Qwen). This is the script
that produced the Nsight Systems reports in outputs/reports/.

Inputs are generated with the same logic as data/exp2_jd/src/generate_fortran_inputs_jd.py:
per-column Box-Muller scaling (mean=1, std=0.1), physics-based z/rho/theta profiles,
mirroring JD_kessler_driver.F90.  Grid default: 128 cols x 56 levels.

Usage (from the kessler project root; the nsys and ncu modes are wrapped by
the profiler binary, see ../LAFT/pbsJobs/compare_profiling_LLMs.sh):
    python3 _compare_results/LLMx_jd_profiling_run.py --mode <mode> [--ncol <ncol>] [--steps <steps>]

Modes:
    jax_trace     -- JAX built-in profiler (Perfetto trace). Best for: which
                     XLA kernels are slow, operator fusion, retracing.

    memory        -- JAX device memory profiler + nvidia-smi polling across N
                     simulated time steps. Best for: memory growth / leaks.

    log_compiles  -- Sets JAX_LOG_COMPILES=1 and runs the sweep. Best for:
                     detecting unexpected JIT retracing. Writes a summary
                     table to outputs/profiling/log_compiles/compile_times_table.txt.

    nvidia_smi    -- Polls nvidia-smi before/after each call in a time-step
                     loop. Best for: quick sanity check on GPU memory usage.

    nsys          -- Runs warmup + N profiled calls for a single model under
                     NVIDIA Nsight Systems. nsys wraps this process externally
                     (see ../LAFT/pbsJobs/compare_profiling_LLMs.sh). Requires --model.
                     Uses the same JD inputs as all other modes (qr ~ 0.01).

Outputs:
    _compare_results/outputs/profiling/<mode>/   -- all artifacts for the selected mode
"""

from __future__ import annotations
import os
os.environ["JAX_ENABLE_X64"] = "1"

import sys
import argparse
import importlib.util
import inspect
import subprocess
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # the kessler project root
sys.path.insert(0, str(PROJECT_ROOT))


def to_row_major_2d(arr):
    """Fortran (ncol, nz) host array -> device-resident JAX (nz, ncol) array."""
    return jnp.asarray(np.ascontiguousarray(arr.T), dtype=jnp.float64)


def to_row_major_1d(arr):
    return jnp.asarray(np.ascontiguousarray(arr), dtype=jnp.float64)


# Short keys (used in the reports and output folders) -> the paper-1 archives.
MODELS = {
    "claude": PROJECT_ROOT / "translations/claude-sonnet46-jd/jax/kessler_run.py",
    "gpt":    PROJECT_ROOT / "translations/gpt54thinking-jd/jax/kessler_run.py",
    "gemini": PROJECT_ROOT / "translations/gemini31pro-jd/jax/kessler_run.py",
    "qwen":   PROJECT_ROOT / "translations/qwen25-32b-jd/jax/kessler_run.py",
}

# JD-driver grid defaults (matches JD_kessler_driver.F90)
NZ       = 56
DT       = 60.0
LYR_SURF = 1
LYR_TOA  = NZ
SEED     = 42   # fixed seed for reproducibility

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_core_fn(model: str, path: Path):
    spec = importlib.util.spec_from_file_location(f"kessler_run_{model}", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "kessler_run_core")


def make_inputs(ncol: int, nz: int, seed: int = SEED) -> dict:
    """
    Build JAX input arrays matching data/exp2_jd/src/JD_kessler_driver.F90
    (and the OpenACC driver in data/exp2_jd/acc_src/).

    Physics formulas and all scalar constants are identical to both Fortran
    drivers.  The per-column scaling arr[i] ~ N(mean=1, std=0.1) uses
    np.random.default_rng(seed) rather than Fortran's random_number — the
    statistical distribution is the same but specific values differ, which
    causes minor subcycle count variation (e.g. 10 vs 11 subcycles at ncol=1000).

    arr(i)     = 1.0 + 0.1 * N(0,1)          [per-column scale]
    z(i,k)     = arr(i) * 100.0 * (k-1)      [m]
    rho(i,k)   = arr(i) * 1.2 * exp(-z/8000) [kg/m^3]
    pk(i,k)    = arr(i) * 1.0                 [Exner-like]
    theta(i,k) = arr(i) * (300 - 0.006*z)     [K]
    qv(i,k)    = arr(i) * 0.010               [kg/kg]
    qc(i,k)    = arr(i) * 0.01                [kg/kg]
    qr(i,k)    = arr(i) * 0.01                [kg/kg]
    cpair(i,k) = 1004.0                       [J/(kg K)]
    rair(i,k)  = 287.0                        [J/(kg K)]
    precl(i)   = 0.0                          [m/s]
    lv         = 2.5e6                        [J/kg]
    pref       = 100000.0                     [Pa]
    rhoqr      = 1000.0                       [kg/m^3]
    """
    rng = np.random.default_rng(seed)

    # Per-column scaling: mean=1, stddev=0.1  (mirrors Fortran arr(i))
    arr = 1.0 + 0.1 * rng.standard_normal(ncol)   # (ncol,)

    # Heights (m): k=0..nz-1 mirrors Fortran (k-1) offset
    k_idx = np.arange(nz, dtype=np.float64)        # 0, 1, ..., nz-1
    z_np  = arr[:, np.newaxis] * (100.0 * k_idx)   # (ncol, nz)

    # Physics fields
    cpair_np = np.full((ncol, nz), 1004.0)
    rair_np  = np.full((ncol, nz), 287.0)
    rho_np   = arr[:, np.newaxis] * (1.2 * np.exp(-z_np / 8000.0))
    pk_np    = arr[:, np.newaxis] * np.ones((ncol, nz))
    theta_np = arr[:, np.newaxis] * (300.0 - 0.006 * z_np)
    qv_np    = arr[:, np.newaxis] * np.full((ncol, nz), 0.010)
    qc_np    = arr[:, np.newaxis] * np.full((ncol, nz), 0.01)
    qr_np    = arr[:, np.newaxis] * np.full((ncol, nz), 0.01)
    precl_np = np.zeros(ncol)

    # Transpose from Fortran (ncol, nz) → JAX (nz, ncol) using bridge helpers
    return dict(
        ncol=ncol, nz=nz, dt=DT, lyr_surf=LYR_SURF, lyr_toa=LYR_TOA,
        cpair=to_row_major_2d(cpair_np),
        rair=to_row_major_2d(rair_np),
        rho=to_row_major_2d(rho_np),
        z=to_row_major_2d(z_np),
        pk=to_row_major_2d(pk_np),
        theta=to_row_major_2d(theta_np),
        qv=to_row_major_2d(qv_np),
        qc=to_row_major_2d(qc_np),
        qr=to_row_major_2d(qr_np),
        precl=to_row_major_1d(precl_np),
        relhum=jnp.zeros((nz, ncol),               dtype=jnp.float64),
        scheme_name="kessler", errmsg="", errflg=0,
        lv=jnp.array(2.5e6,      dtype=jnp.float64),
        pref=jnp.array(100000.0, dtype=jnp.float64),
        rhoqr=jnp.array(1000.0,  dtype=jnp.float64),
    )


def jit_fn(fn):
    candidates = ["ncol", "nz", "lyr_surf", "lyr_toa", "scheme_name", "errmsg", "errflg"]
    sig_params  = inspect.signature(fn).parameters
    valid       = [a for a in candidates if a in sig_params]
    return jax.jit(fn, static_argnames=valid)


def filter_inputs(fn_jit, inputs: dict) -> dict:
    """Return only the keys that the function actually accepts."""
    sig_params = inspect.signature(fn_jit).parameters
    return {k: v for k, v in inputs.items() if k in sig_params}


def warmup(fn_jit, inputs):
    """Trigger JIT compilation before profiling."""
    jax.block_until_ready(fn_jit(**filter_inputs(fn_jit, inputs))[0])


def gpu_memory_mb() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        )
        return int(out.decode().strip().split("\n")[0])
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Mode 1 — JAX trace (Perfetto)
# ---------------------------------------------------------------------------

def run_jax_trace(ncol: int, out_dir: Path):
    """
    Wraps a single warmed-up call for each model in jax.profiler.trace.
    Output: _compare_results/outputs/profiling/jax_trace/<model>/  — open in https://ui.perfetto.dev
    """
    print(f"\n[jax_trace] ncol={ncol}, nz={NZ}")
    inputs = make_inputs(ncol, NZ)

    for model, path in MODELS.items():
        model_dir = out_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"  {model} ... ", end="", flush=True)
        try:
            fn_jit = jit_fn(load_core_fn(model, path))
            warmup(fn_jit, inputs)                       # compile outside the trace

            filtered = filter_inputs(fn_jit, inputs)
            with jax.profiler.trace(str(model_dir), create_perfetto_link=False):
                jax.block_until_ready(fn_jit(**filtered)[0])

            print(f"done  →  {model_dir}")
        except Exception as exc:
            print(f"FAILED: {exc}")

    print(f"\nOpen trace files at https://ui.perfetto.dev")
    print(f"Trace directories: {out_dir}/")


# ---------------------------------------------------------------------------
# Mode 2 — JAX memory profiler
# ---------------------------------------------------------------------------

def run_memory(ncol: int, steps: int, out_dir: Path):
    """
    Runs N simulated time steps and saves a device memory profile snapshot
    after each step for each model.
    Output: _compare_results/outputs/profiling/memory/<model>/step_<N>.prof
    Use:    pprof --pdf step_N.prof > step_N.pdf
    """
    print(f"\n[memory] ncol={ncol}, nz={NZ}, steps={steps}")
    inputs = make_inputs(ncol, NZ)

    for model, path in MODELS.items():
        model_dir = out_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  {model}")
        try:
            fn_jit = jit_fn(load_core_fn(model, path))
            warmup(fn_jit, inputs)

            filtered = filter_inputs(fn_jit, inputs)
            for step in range(steps):
                result = fn_jit(**filtered)
                jax.block_until_ready(result[0])

                prof_path = str(model_dir / f"step_{step:04d}.prof")
                jax.profiler.save_device_memory_profile(prof_path)
                mem_mb = gpu_memory_mb()
                print(f"    step {step:>4}  GPU mem={mem_mb} MB  → {prof_path}")

        except Exception as exc:
            print(f"  FAILED: {exc}")


# ---------------------------------------------------------------------------
# Mode 3 — log compiles
# ---------------------------------------------------------------------------

def run_log_compiles(out_dir: Path):
    """
    Enables JAX compile logging and runs the full ncol sweep for each model.
    Every JIT recompilation is printed to stdout, making unexpected retracing
    immediately visible.

    JAX >= 0.4.14 replaced the env var JAX_LOG_COMPILES=1 with:
        jax.config.update("jax_log_compiles", True)
    Both are set here for compatibility across versions.

    Output:
      - stdout: raw JAX compile logs (PBS .o file)
      - outputs/profiling/log_compiles/compile_times_table.txt  — summary table
    Expected: exactly one "Compiling..." line per (model, ncol) pair.
    Any extra lines indicate unexpected retracing.
    """
    import logging

    # Support both old (env var) and new (config) JAX versions
    os.environ["JAX_LOG_COMPILES"] = "1"
    try:
        jax.config.update("jax_log_compiles", True)
    except Exception:
        pass  # older JAX versions don't have this config key

    NCOL_SWEEP = [50, 100, 200, 500, 1000, 2000, 5000, 10_000]
    print(f"\n[log_compiles] ncol sweep: {NCOL_SWEEP}")
    print("Compile logging enabled — every recompilation will be logged below.")
    print("Expected: one 'Compiling...' line per (model, ncol) pair.\n")

    # ── log handler to count recompilations per (model, ncol) ────────────────
    class _CompileCounter(logging.Handler):
        def __init__(self):
            super().__init__()
            self.count = 0
        def emit(self, record):
            if "Compiling" in record.getMessage():
                self.count += 1
        def reset(self):
            self.count = 0

    counter = _CompileCounter()
    logging.getLogger("jax").addHandler(counter)

    # ── collect results ───────────────────────────────────────────────────────
    # records[(model, ncol)] = {compile_s, execute_s, recompiles}
    records = {}

    for model, path in MODELS.items():
        print(f"\n{'='*50}")
        print(f"Model: {model}")
        print(f"{'='*50}")
        try:
            core_fn = load_core_fn(model, path)
            fn_jit  = jit_fn(core_fn)

            for ncol in NCOL_SWEEP:
                inputs  = make_inputs(ncol, NZ)
                filtered = filter_inputs(fn_jit, inputs)

                counter.reset()
                print(f"  ncol={ncol:>6} ... ", end="", flush=True)

                # First call: compile + execute
                t0 = time.perf_counter()
                jax.block_until_ready(fn_jit(**filtered)[0])
                compile_and_exec_s = time.perf_counter() - t0

                # Second call: execute only (JIT cache hit)
                t1 = time.perf_counter()
                jax.block_until_ready(fn_jit(**filtered)[0])
                execute_s = time.perf_counter() - t1

                compile_s  = compile_and_exec_s - execute_s
                recompiles = counter.count
                records[(model, ncol)] = dict(
                    compile_s=compile_s,
                    execute_s=execute_s,
                    recompiles=recompiles,
                )
                print(f"done  (compile≈{compile_s:.3f}s  exec={execute_s*1000:.2f}ms  recompiles={recompiles})")

        except Exception as exc:
            print(f"  FAILED: {exc}")
            for ncol in NCOL_SWEEP:
                records[(model, ncol)] = dict(compile_s=float("nan"),
                                              execute_s=float("nan"),
                                              recompiles=-1)

    logging.getLogger("jax").removeHandler(counter)

    # ── write table ───────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "compile_times_table.txt"

    models_list = list(MODELS.keys())
    col_w = 12

    def fmt(v, nan_str="FAILED"):
        return nan_str if v != v else f"{v:.3f}"

    with open(out_path, "w") as f:
        import datetime
        f.write(f"JAX Compile Times — log_compiles mode\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"ncol sweep: {NCOL_SWEEP}  |  nz={NZ}\n\n")

        # ── compile time table ────────────────────────────────────────────────
        f.write("=== XLA Compile Time (seconds, first call minus cached call) ===\n\n")
        header = f"{'ncol':>8}" + "".join(f"{m:>{col_w}}" for m in models_list)
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for ncol in NCOL_SWEEP:
            row = f"{ncol:>8}"
            for m in models_list:
                v = records.get((m, ncol), {}).get("compile_s", float("nan"))
                row += f"{fmt(v):>{col_w}}"
            f.write(row + "\n")

        # ── execute time table ────────────────────────────────────────────────
        f.write("\n=== Cached Execute Time (ms, second call) ===\n\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for ncol in NCOL_SWEEP:
            row = f"{ncol:>8}"
            for m in models_list:
                v = records.get((m, ncol), {}).get("execute_s", float("nan"))
                row += f"{fmt(v * 1000 if v == v else v):>{col_w}}"
            f.write(row + "\n")

        # ── recompile count table ─────────────────────────────────────────────
        f.write("\n=== Recompile Count (expected: 1 per ncol) ===\n\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for ncol in NCOL_SWEEP:
            row = f"{ncol:>8}"
            for m in models_list:
                v = records.get((m, ncol), {}).get("recompiles", -1)
                flag = "  <-- UNEXPECTED" if isinstance(v, int) and v > 1 else ""
                row += f"{str(v):>{col_w}}"
            f.write(row + flag + "\n")

    print(f"\nResults table written to: {out_path}")


# ---------------------------------------------------------------------------
# Mode 4 — nsys (NVIDIA Nsight Systems)
# ---------------------------------------------------------------------------

def run_nsys(model: str, ncol: int, n_calls: int = 6):
    """
    Runs warmup + n_calls profiled calls for a single model.
    nsys wraps this process externally:
      nsys profile --trace=cuda,nvtx --output=<out> python3 $SCRIPT --mode nsys --model MODEL --ncol NCOL
    Uses the same JD inputs as all other modes (qr = arr * 0.01, nz=56).
    """
    print(f"\n[nsys] model={model}, ncol={ncol}, nz={NZ}, calls={n_calls}")
    inputs = make_inputs(ncol, NZ)

    path = MODELS[model]
    fn_jit = jit_fn(load_core_fn(model, path))

    print(f"  warmup ... ", end="", flush=True)
    warmup(fn_jit, inputs)
    print("done")

    filtered = filter_inputs(fn_jit, inputs)
    print(f"  {n_calls} profiled calls ... ", end="", flush=True)
    for _ in range(n_calls):
        jax.block_until_ready(fn_jit(**filtered)[0])
    print("done")


# ---------------------------------------------------------------------------
# Mode 5 — ncu (NVIDIA Nsight Compute kernel-level)
# ---------------------------------------------------------------------------

def run_ncu(model: str, ncol: int, n_calls: int = 2):
    """
    Runs warmup + n_calls profiled calls for a single model.
    ncu wraps this process externally:
      ncu --set full --export <out> python3 $SCRIPT --mode ncu --model MODEL --ncol NCOL
    Uses the same JD inputs as all other modes (qr = arr * 0.01, nz=56).
    n_calls=2: one warmup call (JIT compile), one profiled call.
    """
    print(f"\n[ncu] model={model}, ncol={ncol}, nz={NZ}, calls={n_calls}")
    inputs = make_inputs(ncol, NZ)

    path = MODELS[model]
    fn_jit = jit_fn(load_core_fn(model, path))

    print(f"  warmup ... ", end="", flush=True)
    warmup(fn_jit, inputs)
    print("done")

    filtered = filter_inputs(fn_jit, inputs)
    print(f"  {n_calls} profiled calls ... ", end="", flush=True)
    for _ in range(n_calls):
        jax.block_until_ready(fn_jit(**filtered)[0])
    print("done")


# ---------------------------------------------------------------------------
# Mode 6 — nvidia-smi polling
# ---------------------------------------------------------------------------


def run_nvidia_smi(ncol: int, steps: int, out_dir: Path):
    """
    Polls nvidia-smi before and after each call in a time-step loop.
    Prints a memory delta table per model.
    Best for: quick sanity check — is memory flat, growing, or spiking?
    """
    print(f"\n[nvidia_smi] ncol={ncol}, nz={NZ}, steps={steps}")
    inputs = make_inputs(ncol, NZ)

    for model, path in MODELS.items():
        print(f"\n  {model}")
        print(f"  {'step':>6}  {'before_MB':>10}  {'after_MB':>10}  {'delta_MB':>9}")
        print(f"  {'-'*45}")
        try:
            fn_jit = jit_fn(load_core_fn(model, path))
            warmup(fn_jit, inputs)

            filtered = filter_inputs(fn_jit, inputs)
            for step in range(steps):
                before = gpu_memory_mb()
                jax.block_until_ready(fn_jit(**filtered)[0])
                after  = gpu_memory_mb()
                delta  = after - before
                flag   = " <-- GROWING" if delta > 0 else ""
                print(f"  {step:>6}  {before:>10}  {after:>10}  {delta:>+9}{flag}")

        except Exception as exc:
            print(f"  FAILED: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="kessler_run_core profiling runner (JD inputs)")
    p.add_argument("--mode",  required=True,
                   choices=["jax_trace", "memory", "log_compiles", "nvidia_smi", "nsys", "ncu"],
                   help="Profiling mode to run (one at a time)")
    p.add_argument("--ncol",  type=int, default=128,
                   help="Number of columns (default: 128, matching JD driver)")
    p.add_argument("--steps", type=int, default=20,
                   help="Number of simulated time steps for memory/nvidia_smi modes (default: 20)")
    p.add_argument("--model", type=str, default=None,
                   choices=list(MODELS.keys()),
                   help="Single model to profile (required for nsys and ncu modes)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"JAX backend : {jax.lib.xla_bridge.get_backend().platform}")
    print(f"JAX devices : {jax.devices()}")
    print(f"Mode        : {args.mode}")
    print(f"ncol        : {args.ncol}  |  nz={NZ}  |  dt={DT}  |  seed={SEED}")

    out_dir = Path(__file__).resolve().parent / "outputs" / "profiling" / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "jax_trace":
        run_jax_trace(args.ncol, out_dir)

    elif args.mode == "memory":
        run_memory(args.ncol, args.steps, out_dir)

    elif args.mode == "log_compiles":
        run_log_compiles(out_dir)

    elif args.mode == "nvidia_smi":
        run_nvidia_smi(args.ncol, args.steps, out_dir)

    elif args.mode == "nsys":
        if args.model is None:
            print("ERROR: --model is required for nsys mode")
            raise SystemExit(1)
        run_nsys(args.model, args.ncol)

    elif args.mode == "ncu":
        if args.model is None:
            print("ERROR: --model is required for ncu mode")
            raise SystemExit(1)
        run_ncu(args.model, args.ncol)
