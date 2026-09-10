"""
Scalability Benchmark: Fortran → JAX Translations of kessler_run  (JD-driver inputs)
======================================================================================

Sweeps ncol at fixed nz=56 and times the JIT-compiled kessler_run_core
for each of the four paper-1 translations in translations/ (Claude, GPT,
Gemini, Qwen). Run from the kessler project root:

    python3 _compare_results/scalability_benchmark_LLMs.py

or submit ../LAFT/pbsJobs/compare_scalability_LLMs.sh (needs one GPU).

Inputs are generated with the same logic as data/exp2_jd/src/generate_fortran_inputs_jd.py:
per-column Box-Muller scaling (mean=1, std=0.1), physics-based z/rho/theta profiles,
mirroring JD_kessler_driver.F90.  Arrays are transposed from Fortran (ncol, nz) to
JAX (nz, ncol) before any timing begins.

Timing methodology:
  - One warmup call per (model, ncol) pair — triggers JAX tracing/compilation
  - N_REPS subsequent timed calls using jax.block_until_ready() for GPU sync
  - Median of N_REPS reported as the benchmark time

Outputs:
  _compare_results/outputs/scalability/LLMs_scalability_results.json  — raw timings
  _compare_results/outputs/plots/LLMs_scalability_plot.png             — time vs ncol plot
"""

from __future__ import annotations
import os
os.environ["JAX_ENABLE_X64"] = "1"

import sys
import time
import json
import importlib.util
import inspect
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

NCOL_SWEEP = [50, 100, 200, 500, 1000, 2000, 5000, 10_000, 100_000, 1_000_000]

# JD-driver grid defaults (matches JD_kessler_driver.F90)
NZ       = 56
DT       = 60.0
LYR_SURF = 1
LYR_TOA  = NZ
SEED     = 42   # fixed seed for reproducibility
N_REPS   = 5   # timed repetitions after warmup

# Short keys (used in the JSON, the plots, and the reports) -> the paper-1 archives.
MODELS = {
    "claude": PROJECT_ROOT / "translations/claude-sonnet46-jd/jax/kessler_run.py",
    "gpt":    PROJECT_ROOT / "translations/gpt54thinking-jd/jax/kessler_run.py",
    "gemini": PROJECT_ROOT / "translations/gemini31pro-jd/jax/kessler_run.py",
    "qwen":   PROJECT_ROOT / "translations/qwen25-32b-jd/jax/kessler_run.py",
}

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "scalability"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_core_fn(model: str, path: Path):
    """Import kessler_run_core from a model's translation file."""
    spec = importlib.util.spec_from_file_location(f"kessler_run_{model}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "kessler_run_core")


def make_inputs(ncol: int, nz: int, seed: int = SEED) -> dict:
    """
    Build JAX input arrays matching data/exp2_jd/src/JD_kessler_driver.F90
    (and the OpenACC driver in data/exp2_jd/acc_src/).

    Physics formulas and all scalar constants are identical to both Fortran
    drivers.  Arrays are generated in Fortran layout (ncol, nz) then transposed
    to JAX layout (nz, ncol) — transposition happens here,
    outside any timing loop.

    The per-column scaling arr[i] ~ N(mean=1, std=0.1) uses
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

    # Physics fields — Fortran layout (ncol, nz)
    cpair_np = np.full((ncol, nz), 1004.0)
    rair_np  = np.full((ncol, nz), 287.0)
    rho_np   = arr[:, np.newaxis] * (1.2 * np.exp(-z_np / 8000.0))
    pk_np    = arr[:, np.newaxis] * np.ones((ncol, nz))
    theta_np = arr[:, np.newaxis] * (300.0 - 0.006 * z_np)
    qv_np    = arr[:, np.newaxis] * np.full((ncol, nz), 0.010)
    qc_np    = arr[:, np.newaxis] * np.full((ncol, nz), 0.01)
    qr_np    = arr[:, np.newaxis] * np.full((ncol, nz), 0.01)
    precl_np = np.zeros(ncol)

    # Transpose from Fortran (ncol, nz) → JAX (nz, ncol)
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


def call_core(fn_jit, inputs: dict):
    """Call the JIT-compiled core and block until GPU work completes."""
    sig_params = inspect.signature(fn_jit).parameters
    filtered   = {k: v for k, v in inputs.items() if k in sig_params}
    result = fn_jit(**filtered)
    # block_until_ready on the first output tensor (theta)
    jax.block_until_ready(result[0])
    return result


def jit_compile(fn, ncol: int, nz: int):
    """JIT-compile fn with static integer shape args, filtered to the function's signature."""
    candidates = ["ncol", "nz", "lyr_surf", "lyr_toa", "scheme_name", "errmsg", "errflg"]
    sig_params  = inspect.signature(fn).parameters
    valid       = [a for a in candidates if a in sig_params]
    return jax.jit(fn, static_argnames=valid)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    print(f"JAX devices: {jax.devices()}")
    print(f"ncol sweep: {NCOL_SWEEP}")
    print(f"nz={NZ}, dt={DT}, seed={SEED}, reps={N_REPS}\n")

    results = {}   # model -> list of {ncol, times_s, median_s, mean_s}

    for model, path in MODELS.items():
        print(f"{'='*60}")
        print(f"Model: {model}  ({path.name})")
        print(f"{'='*60}")

        core_fn = load_core_fn(model, path)
        model_results = []

        # JIT handle is reused across ncol sizes where shapes don't change the
        # compiled kernel — JAX retraces when static args (ncol, nz) change.
        fn_jit = jit_compile(core_fn, None, None)

        for ncol in NCOL_SWEEP:
            if model == "qwen" and ncol > 10_000:
                print(f"  ncol={ncol:>6}  skipped (qwen JIT too slow for large ncol)")
                model_results.append({"ncol": ncol, "skipped": True})
                continue

            inputs = make_inputs(ncol, NZ)

            # --- Warmup (triggers JIT trace + compilation) ---
            print(f"  ncol={ncol:>6}  warmup... ", end="", flush=True)
            try:
                t0 = time.perf_counter()
                call_core(fn_jit, inputs)
                warmup_s = time.perf_counter() - t0
                print(f"done ({warmup_s:.2f}s compile+run)", end="  ")
            except Exception as exc:
                print(f"FAILED: {exc}")
                model_results.append({"ncol": ncol, "error": str(exc)})
                continue

            # --- Timed repetitions ---
            times = []
            for _ in range(N_REPS):
                t0 = time.perf_counter()
                call_core(fn_jit, inputs)
                times.append(time.perf_counter() - t0)

            median_ms = np.median(times) * 1e3
            mean_ms   = np.mean(times)   * 1e3
            print(f"median={median_ms:.3f}ms  mean={mean_ms:.3f}ms")

            model_results.append({
                "ncol":      ncol,
                "warmup_s":  warmup_s,
                "times_s":   times,
                "median_s":  float(np.median(times)),
                "mean_s":    float(np.mean(times)),
                "std_s":     float(np.std(times)),
            })

        results[model] = model_results
        print()

    # --- Save JSON ---
    json_path = OUT_DIR / "LLMs_scalability_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "config": {"ncol_sweep": NCOL_SWEEP, "nz": NZ, "dt": DT, "seed": SEED,
                       "n_reps": N_REPS,
                       "backend": jax.lib.xla_bridge.get_backend().platform},
            "results": results,
        }, f, indent=2)
    print(f"Results saved → {json_path}")

    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_results(results: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    colors = {"claude": "#1f77b4", "gpt": "#ff7f0e", "gemini": "#2ca02c", "qwen": "#9467bd"}
    markers = {"claude": "o", "gpt": "s", "gemini": "^", "qwen": "D"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for model, rows in results.items():
        good = [r for r in rows if "error" not in r and "skipped" not in r]
        if not good:
            continue
        ncols   = [r["ncol"]     for r in good]
        medians = [r["median_s"] * 1e3 for r in good]  # ms
        stds    = [r["std_s"]    * 1e3 for r in good]

        kw = dict(color=colors.get(model, "gray"),
                  marker=markers.get(model, "x"),
                  label=model.capitalize(), linewidth=1.8, markersize=6)

        # Left: JAX vmap models only (no Qwen) — linear scale to show fine differences
        if model != "qwen":
            axes[0].errorbar(ncols, medians, yerr=stds, **kw)
        # Right: all models — log-log scale for full picture
        axes[1].errorbar(ncols, medians, yerr=stds, **kw)

    for ax, scale, title in zip(
        axes,
        [("linear", "linear"), ("log", "log")],
        ["Runtime vs Grid Size — vmap models (linear)", "Runtime vs Grid Size — all models (log-log)"]
    ):
        ax.set_xscale(scale[0])
        ax.set_yscale(scale[1])
        ax.set_xlabel("ncol (number of columns)")
        ax.set_ylabel("Median runtime (ms)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"kessler_run scalability (JD inputs)  |  nz={NZ}, dt={DT}s  |  "
        f"backend={jax.lib.xla_bridge.get_backend().platform.upper()}  |  "
        f"{N_REPS} reps, median±std",
        fontsize=10,
    )
    plt.tight_layout()

    plot_path = OUT_DIR.parent / "plots" / "LLMs_scalability_plot.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved     → {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_benchmark()
    plot_results(results)
