"""
Profiling Runner: driver_kessler (OpenACC Fortran)
==================================================

Profiles the compiled OpenACC Fortran binary ./driver_kessler (built from the
sources in this directory with `make ARCH=GPU`) in the same modes as the LAFT
JAX profiler, so the numbers can be compared with any JAX translation.

The binary takes `ncol nz` on the command line (default ncol=1000, nz=56,
dt=60.0 s) and generates its own inputs internally.  The nsys mode automatically parses kernel/memory stats
and writes a summary table in the same format as the JAX profiling reports:

    | Implementation      | End-to-end (ms) | GPU exec (ms) | Dispatch overhead (ms) |
    | Kernel launches | Unique kernels | While iterations | MemcpyD2H |

Usage:
    python acc_kessler_profiling.py --mode <mode> [--ncol N] [--nz N] [--steps N]

Modes:
    timing      -- 1 warmup + N timed runs (wall-clock ms). Reports steady-state
                   execution time and run-to-run variance.

    nvidia_smi  -- Polls nvidia-smi before/after each run. Best for: quick
                   sanity check on GPU memory usage (flat vs growing).

    memory      -- Repeated runs with nvidia-smi snapshots. Best for: spotting
                   memory growth across simulated time steps.

    nsys        -- Launches the binary under NVIDIA Nsight Systems, then runs
                   nsys stats to extract kernel/memory metrics and writes a
                   markdown summary table comparable to the JAX profiling reports.
                   Requires nsys on PATH (shipped with the NVIDIA HPC SDK).

    ncol_sweep  -- Sweeps ncol over [50,100,200,500,1000,2000,5000,10000,100000,1000000].
                   For each ncol: 1 warmup + STEPS timed runs.
                   Reports first-call time and steady-state mean, analogous to
                   the JAX log_compiles table (XLA compile vs cached execute).
                   Output: outputs/profiling/ncol_sweep/ncol_sweep_table.txt

Outputs:
    outputs/profiling/<mode>/  -- all artifacts for the selected mode
    outputs/profiling/nsys/summary_table.md  -- main comparison table

Prerequisites:
    - Binary compiled:  make COMPILER=nvhpc ARCH=GPU  (in this directory)
    - nsys mode only:   nsys on PATH
"""

from __future__ import annotations
import argparse
import csv
import datetime
import io
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SRC_DIR  = Path(__file__).resolve().parent
BINARY   = SRC_DIR / "driver_kessler"

INPUTS_DIR  = SRC_DIR.parent / "fortran_io" / "inputs"
OUTPUTS_DIR = SRC_DIR.parent / "fortran_io" / "outputs"

# Grid defaults — match JD_kessler_driver.F90 defaults and JAX profiling setup
NCOL = 1000   # matches JAX profiling (was 128 in original Fortran source)
NZ   = 56
DT   = 60.0

# ncol sweep — mirrors JAX log_compiles ncol sweep
NCOL_SWEEP = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 100000, 1000000]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def check_binary():
    if not BINARY.exists():
        print(f"ERROR: binary not found: {BINARY}")
        print("Compile with:  cd data/acc_src && make COMPILER=nvhpc ARCH=GPU")
        sys.exit(1)


def ensure_io_dirs():
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def run_binary(ncol: int = NCOL, nz: int = NZ,
               capture_output: bool = True) -> tuple[float, str, str]:
    """Run the binary once; return (wall_time_s, stdout, stderr)."""
    ensure_io_dirs()
    t0 = time.perf_counter()
    result = subprocess.run(
        [str(BINARY), str(ncol), str(nz)],
        cwd=str(SRC_DIR),
        capture_output=capture_output,
        text=True,
    )
    elapsed = time.perf_counter() - t0
    stdout = result.stdout if capture_output else ""
    stderr = result.stderr if capture_output else ""
    if result.returncode != 0:
        print(f"  WARNING: binary exited with code {result.returncode}")
        if stderr:
            print(f"  stderr: {stderr.strip()[:200]}")
    return elapsed, stdout, stderr


def gpu_memory_mb() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        )
        return int(out.decode().strip().split("\n")[0])
    except Exception:
        return -1


def _extract_iter_count(stdout: str) -> int:
    """Parse 'iteration Cnt: N' from binary stdout; return -1 if not found."""
    for line in stdout.splitlines():
        low = line.lower()
        if "iteration" in low and "cnt" in low:
            for token in reversed(line.split()):
                try:
                    return int(token)
                except ValueError:
                    continue
    return -1


# ---------------------------------------------------------------------------
# nsys stats helpers
# ---------------------------------------------------------------------------

def _run_nsys_stats(rep_file: Path, report: str) -> str:
    """
    Run `nsys stats --report <report> --format csv` on the trace.
    Returns the CSV text (header + data rows, Processing line stripped), or "".

    Try order:
      1. .sqlite  (preferred on nsys >= 2024 — avoids re-export overhead)
      2. .nsys-rep with --output - (stdout)
      3. .nsys-rep without --output flag
      4. sidecar CSV written by nsys next to the .nsys-rep
    """
    sqlite_str = str(rep_file.with_suffix("")) + ".sqlite" \
        if not str(rep_file).endswith(".sqlite") else str(rep_file)
    rep_str = str(rep_file) if str(rep_file).endswith(".nsys-rep") \
        else str(rep_file) + ".nsys-rep"

    candidates = [
        ["nsys", "stats", "--report", report, "--format", "csv", sqlite_str],
        ["nsys", "stats", "--report", report, "--format", "csv", "--output", "-", rep_str],
        ["nsys", "stats", "--report", report, "--format", "csv", rep_str],
    ]

    for cmd in candidates:
        result = subprocess.run(cmd, capture_output=True, text=True)
        # Strip the "Processing [...]" banner line nsys writes to stdout
        cleaned = "\n".join(
            l for l in result.stdout.splitlines() if not l.startswith("Processing [")
        )
        if result.returncode == 0 and cleaned.strip():
            return cleaned

    # sidecar CSV some nsys versions write next to the .nsys-rep
    sidecar = rep_file.parent / f"{rep_file.stem}_{report}.csv"
    if sidecar.exists():
        return sidecar.read_text()

    return ""


def _parse_kern_sum(csv_text: str) -> dict:
    """
    Parse cuda_gpu_kern_sum CSV.
    Returns: gpu_exec_ms, kernel_launches, unique_kernels.
    """
    if not csv_text.strip():
        return {"gpu_exec_ms": float("nan"), "kernel_launches": -1, "unique_kernels": -1}

    total_ns   = 0.0
    total_inst = 0
    names      = set()

    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        # Column names vary slightly across nsys versions — try both styles
        ns = float(
            row.get("Total Time (ns)")
            or row.get("Total Time")
            or row.get("GPU Time (ns)")
            or 0
        )
        inst = int(
            row.get("Instances")
            or row.get("Count")
            or row.get("Calls")
            or 0
        )
        name = (
            row.get("Name")
            or row.get("Kernel Name")
            or row.get("Operation")
            or ""
        ).strip()

        total_ns   += ns
        total_inst += inst
        if name:
            names.add(name)

    return {
        "gpu_exec_ms":    total_ns / 1e6,
        "kernel_launches": total_inst,
        "unique_kernels":  len(names),
    }


def _parse_mem_sum(csv_text: str) -> dict:
    """
    Parse cuda_gpu_mem_time_sum CSV.
    Returns: d2h_count, d2h_ms.
    """
    if not csv_text.strip():
        return {"d2h_count": -1, "d2h_ms": float("nan")}

    d2h_count = 0
    d2h_ns    = 0.0

    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        op = (
            row.get("Operation")
            or row.get("Type")
            or row.get("Name")
            or ""
        ).lower()
        count = int(row.get("Count") or row.get("Instances") or row.get("Calls") or 0)
        ns    = float(row.get("Total Time (ns)") or row.get("Total Time") or 0)

        if "dtoh" in op or "d2h" in op or "device to host" in op:
            d2h_count += count
            d2h_ns    += ns

    return {"d2h_count": d2h_count, "d2h_ms": d2h_ns / 1e6}


# ---------------------------------------------------------------------------
# Mode 1 — timing
# ---------------------------------------------------------------------------

def run_timing(steps: int, out_dir: Path, ncol: int = NCOL, nz: int = NZ):
    """
    1 warmup + STEPS timed runs. Reports steady-state wall-clock time.
    Output: outputs/profiling/timing/timing_table.txt
    """
    print(f"\n[timing]  binary={BINARY.name}  ncol={ncol}  nz={nz}  steps={steps}")

    print("  warmup ... ", end="", flush=True)
    warmup_s, stdout, _ = run_binary(ncol, nz)
    iter_count = _extract_iter_count(stdout)
    print(f"done  ({warmup_s*1000:.2f} ms)  while_iter={iter_count}")

    times = []
    for step in range(steps):
        elapsed, _, _ = run_binary(ncol, nz)
        times.append(elapsed)
        print(f"  step {step:>4}  {elapsed*1000:>8.2f} ms")

    mean_s = sum(times) / len(times)
    min_s  = min(times)
    max_s  = max(times)

    summary = (
        f"\n  Summary ({steps} timed runs):\n"
        f"    warmup : {warmup_s*1000:.2f} ms\n"
        f"    mean   : {mean_s*1000:.2f} ms\n"
        f"    min    : {min_s*1000:.2f} ms\n"
        f"    max    : {max_s*1000:.2f} ms\n"
    )
    print(summary)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "timing_table.txt"
    with open(out_path, "w") as f:
        f.write(f"ACC Fortran Timing — driver_kessler\n")
        f.write(f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Binary    : {BINARY}\n")
        f.write(f"Grid      : ncol={ncol}  nz={nz}  dt={DT} s\n")
        f.write(f"Runs      : 1 warmup + {steps} timed\n\n")
        f.write(f"{'run':>6}  {'time_ms':>10}\n")
        f.write(f"{'warmup':>6}  {warmup_s*1000:>10.2f}\n")
        for i, t in enumerate(times):
            f.write(f"{i:>6}  {t*1000:>10.2f}\n")
        f.write(summary)
    print(f"  Results written to: {out_path}")


# ---------------------------------------------------------------------------
# Mode 2 — nvidia_smi
# ---------------------------------------------------------------------------

def run_nvidia_smi(steps: int, out_dir: Path, ncol: int = NCOL, nz: int = NZ):
    """
    Polls nvidia-smi before and after each run.
    Best for: quick sanity check on GPU memory usage.
    """
    print(f"\n[nvidia_smi]  binary={BINARY.name}  ncol={ncol}  nz={nz}  steps={steps}")
    print(f"\n  {'step':>6}  {'before_MB':>10}  {'after_MB':>10}  {'delta_MB':>9}  {'time_ms':>9}")
    print(f"  {'-'*55}")

    for step in range(steps):
        before = gpu_memory_mb()
        elapsed, _, _ = run_binary(ncol, nz)
        after  = gpu_memory_mb()
        delta  = after - before
        flag   = " <-- GROWING" if delta > 0 else ""
        print(f"  {step:>6}  {before:>10}  {after:>10}  {delta:>+9}  {elapsed*1000:>8.2f}{flag}")


# ---------------------------------------------------------------------------
# Mode 3 — memory
# ---------------------------------------------------------------------------

def run_memory(steps: int, out_dir: Path, ncol: int = NCOL, nz: int = NZ):
    """
    Repeated runs with nvidia-smi snapshots after each step.
    Output: outputs/profiling/memory/memory_log.txt
    """
    print(f"\n[memory]  binary={BINARY.name}  ncol={ncol}  nz={nz}  steps={steps}")

    print("  warmup ... ", end="", flush=True)
    run_binary(ncol, nz)
    print("done")

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "memory_log.txt"

    records = []
    with open(log_path, "w") as f:
        f.write(f"ACC Fortran Memory Log — driver_kessler\n")
        f.write(f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Binary    : {BINARY}\n")
        f.write(f"Grid      : ncol={ncol}  nz={nz}  dt={DT} s\n\n")
        f.write(f"{'step':>6}  {'gpu_mem_MB':>12}  {'time_ms':>9}\n")
        f.write(f"{'-'*35}\n")

        for step in range(steps):
            elapsed, _, _ = run_binary(ncol, nz)
            mem_mb = gpu_memory_mb()
            records.append((step, mem_mb, elapsed))
            print(f"  step {step:>4}  GPU mem={mem_mb:>6} MB  {elapsed*1000:>8.2f} ms")
            f.write(f"{step:>6}  {mem_mb:>12}  {elapsed*1000:>9.2f}\n")

    mems = [r[1] for r in records if r[1] >= 0]
    if len(mems) >= 2 and mems[-1] > mems[0]:
        print(f"\n  WARNING: GPU memory grew from {mems[0]} MB to {mems[-1]} MB (+{mems[-1]-mems[0]} MB)")
    else:
        print(f"\n  OK: GPU memory stable ({mems[0] if mems else 'N/A'} MB)")

    print(f"  Log written to: {log_path}")


# ---------------------------------------------------------------------------
# Mode 4 — nsys  (full pipeline → summary table)
# ---------------------------------------------------------------------------

def _fmt(v, is_int: bool = False, nan_str: str = "N/A") -> str:
    """Format a metric value for the markdown table."""
    if v != v:  # NaN
        return nan_str
    if is_int:
        return f"{int(v):,}"
    return f"{v:.3f}"


def _write_nsys_report(out_dir: Path, metrics: dict):
    """Write the markdown summary table + analysis notes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "summary_table.md"

    e2e_ms      = metrics["end_to_end_ms"]
    gpu_ms      = metrics["gpu_exec_ms"]
    dispatch_ms = metrics["dispatch_overhead_ms"]
    launches    = metrics["kernel_launches"]
    unique_k    = metrics["unique_kernels"]
    while_iter  = metrics["while_iterations"]
    d2h         = metrics["d2h_count"]
    ncol        = metrics["ncol"]
    nz          = metrics["nz"]

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# ACC Fortran Profiling Report — driver_kessler (OpenACC)",
        f"**Generated:** {now}",
        f"**Binary:** `{BINARY.name}`  |  **Grid:** ncol={ncol}, nz={nz}, dt={DT} s",
        f"**Profiler:** NVIDIA Nsight Systems (`nsys profile --trace=openacc,cuda,nvtx`)",
        f"",
        f"---",
        f"",
        f"## Summary Table",
        f"",
        f"| Implementation | End-to-end (ms) | GPU exec (ms) | Dispatch overhead (ms) | Kernel launches | Unique kernels | While iterations | MemcpyD2H |",
        f"|----------------|----------------:|--------------:|-----------------------:|----------------:|---------------:|-----------------:|----------:|",
        f"| ACC Fortran    | {_fmt(e2e_ms)} | {_fmt(gpu_ms)} | {_fmt(dispatch_ms)} | {_fmt(launches, is_int=True)} | {_fmt(unique_k, is_int=True)} | {_fmt(while_iter, is_int=True)} | {_fmt(d2h, is_int=True)} |",
        f"",
        f"> **Column definitions** (matching JAX profiling report convention)",
        f"> - *End-to-end*: wall-clock time of the binary run (includes array init and file I/O in addition to physics).",
        f"> - *GPU exec*: sum of all CUDA kernel durations captured by nsys (`cuda_gpu_kern_sum`).",
        f"> - *Dispatch overhead*: end-to-end − GPU exec (CPU setup, OpenACC runtime, file I/O).",
        f"> - *Kernel launches*: total CUDA kernel invocations recorded by nsys.",
        f"> - *Unique kernels*: number of distinct kernel names.",
        f"> - *While iterations*: subcycling iterations reported by the binary (`iteration Cnt`).",
        f"> - *MemcpyD2H*: device-to-host memory copy events (`cuda_gpu_mem_time_sum`).",
        f"",
        f"> **Note on end-to-end time:** Unlike the JAX profiling (which times only the physics",
        f"> function call), the binary wall time includes Box-Muller random input generation,",
        f"> array initialisation, and writing input/output files to disk. The GPU exec time is",
        f"> the clean comparison metric against JAX `kessler_run_core`.",
        f"",
        f"---",
        f"",
        f"## Raw nsys Artifact",
        f"",
        f"The `.nsys-rep` file is in `outputs/profiling/nsys/driver_kessler.nsys-rep`.",
        f"View with: **Nsight Systems GUI** (`nsys-ui`) for the full OpenACC/CUDA timeline.",
        f"",
    ]

    md_path.write_text("\n".join(lines))
    return md_path


def run_nsys(steps: int, out_dir: Path, ncol: int = NCOL, nz: int = NZ):
    """
    Full nsys pipeline:
      1. Warmup run (outside profiled region)
      2. Timed run under nsys profile
      3. nsys stats: cuda_gpu_kern_sum  → GPU exec, kernel launches, unique kernels
      4. nsys stats: cuda_gpu_mem_time_sum → MemcpyD2H count
      5. Extract while iterations from binary stdout
      6. Write summary_table.md
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_base = out_dir / "driver_kessler"
    rep_file = Path(str(rep_base) + ".nsys-rep")

    ensure_io_dirs()

    # ------------------------------------------------------------------
    # Step 1 — warmup (not profiled)
    # ------------------------------------------------------------------
    print(f"\n[nsys]  binary={BINARY.name}  ncol={ncol}  nz={nz}")
    print("  Step 1/4  warmup ... ", end="", flush=True)
    _, warmup_stdout, _ = run_binary(ncol, nz)
    iter_count = _extract_iter_count(warmup_stdout)
    print(f"done  (while_iter={iter_count})")

    # ------------------------------------------------------------------
    # Step 2 — profiled run under nsys
    # ------------------------------------------------------------------
    print(f"  Step 2/4  nsys profile ...", flush=True)
    cmd = [
        "nsys", "profile",
        "--trace=openacc,cuda,nvtx",
        f"--output={rep_base}",
        "--force-overwrite=true",
        str(BINARY), str(ncol), str(nz),
    ]
    print(f"    {' '.join(cmd)}")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(SRC_DIR), capture_output=True, text=True)
    e2e_s = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  ERROR: nsys exited with code {result.returncode}")
        if result.stderr:
            print(result.stderr.strip()[:400])
        return

    # Prefer iter_count from profiled run (same binary output)
    profiled_iter = _extract_iter_count(result.stdout)
    if profiled_iter >= 0:
        iter_count = profiled_iter

    print(f"    done  ({e2e_s*1000:.1f} ms wall)  → {rep_file.name}")

    # ------------------------------------------------------------------
    # Step 3 — nsys stats: kernel summary
    # ------------------------------------------------------------------
    print(f"  Step 3/4  nsys stats (kernels) ...", end="", flush=True)
    kern_csv  = _run_nsys_stats(rep_base, "cuda_gpu_kern_sum")
    kern      = _parse_kern_sum(kern_csv)
    print(f"  gpu_exec={kern['gpu_exec_ms']:.3f} ms  launches={kern['kernel_launches']}  unique={kern['unique_kernels']}")

    # ------------------------------------------------------------------
    # Step 4 — nsys stats: memory transfer summary
    # ------------------------------------------------------------------
    print(f"  Step 4/4  nsys stats (memcpy)  ...", end="", flush=True)
    mem_csv = _run_nsys_stats(rep_base, "cuda_gpu_mem_time_sum")
    mem     = _parse_mem_sum(mem_csv)
    print(f"  D2H_count={mem['d2h_count']}  D2H_time={mem['d2h_ms']:.3f} ms")

    # ------------------------------------------------------------------
    # Build metrics dict and write report
    # ------------------------------------------------------------------
    gpu_ms      = kern["gpu_exec_ms"]
    dispatch_ms = max(0.0, e2e_s * 1000 - gpu_ms)

    metrics = {
        "end_to_end_ms":        e2e_s * 1000,
        "gpu_exec_ms":          gpu_ms,
        "dispatch_overhead_ms": dispatch_ms,
        "kernel_launches":      kern["kernel_launches"],
        "unique_kernels":       kern["unique_kernels"],
        "while_iterations":     iter_count,
        "d2h_count":            mem["d2h_count"],
        "ncol":                 ncol,
        "nz":                   nz,
    }

    md_path = _write_nsys_report(out_dir, metrics)

    # ------------------------------------------------------------------
    # Print the table to stdout as well
    # ------------------------------------------------------------------
    print(f"\n  === Summary Table ===")
    print(f"  {'Metric':<28} {'Value':>12}")
    print(f"  {'-'*42}")
    print(f"  {'End-to-end (ms)':<28} {_fmt(metrics['end_to_end_ms']):>12}")
    print(f"  {'GPU exec (ms)':<28} {_fmt(metrics['gpu_exec_ms']):>12}")
    print(f"  {'Dispatch overhead (ms)':<28} {_fmt(metrics['dispatch_overhead_ms']):>12}")
    print(f"  {'Kernel launches':<28} {_fmt(metrics['kernel_launches'], is_int=True):>12}")
    print(f"  {'Unique kernels':<28} {_fmt(metrics['unique_kernels'], is_int=True):>12}")
    print(f"  {'While iterations':<28} {_fmt(metrics['while_iterations'], is_int=True):>12}")
    print(f"  {'MemcpyD2H':<28} {_fmt(metrics['d2h_count'], is_int=True):>12}")
    print(f"\n  Report written to: {md_path}")
    print(f"  Trace file:        {rep_file}")
    print(f"  View trace with:   nsys-ui {rep_file.name}")


# ---------------------------------------------------------------------------
# Mode 5 — ncol_sweep
# ---------------------------------------------------------------------------

def _nsys_gpu_exec_for_ncol(ncol: int, nz: int, out_dir: Path) -> float:
    """
    Run one nsys-profiled binary call for the given ncol and return GPU exec
    time in ms from cuda_gpu_kern_sum.  The .nsys-rep/.sqlite are written to
    out_dir/nsys/ncol_<ncol>.{nsys-rep,sqlite}.  Returns float('nan') on failure.

    Two-step approach to avoid the nsys version issue where --stats=false skips
    sqlite creation and nsys stats on .nsys-rep writes CSV to sqlite (not stdout):
      1. nsys profile  → .nsys-rep
      2. nsys stats on .nsys-rep → creates .sqlite as side effect (output discarded)
      3. _run_nsys_stats queries .sqlite directly → returns CSV
    """
    nsys_dir = out_dir / "nsys"
    nsys_dir.mkdir(parents=True, exist_ok=True)
    rep_base = nsys_dir / f"ncol_{ncol}"
    rep_file = Path(str(rep_base) + ".nsys-rep")
    sqlite_file = Path(str(rep_base) + ".sqlite")

    ensure_io_dirs()

    # Step 1 — profile
    cmd = [
        "nsys", "profile",
        "--trace=cuda",
        f"--output={rep_base}",
        "--force-overwrite=true",
        "--stats=false",
        str(BINARY), str(ncol), str(nz),
    ]
    result = subprocess.run(cmd, cwd=str(SRC_DIR), capture_output=True, text=True)
    if result.returncode != 0:
        print(f" [nsys profile ERROR code={result.returncode}]", end="")
        return float("nan")

    # Step 2 — generate sqlite (nsys stats on .nsys-rep creates it as a side effect)
    if not sqlite_file.exists():
        subprocess.run(
            ["nsys", "stats", "--format", "csv", str(rep_file)],
            capture_output=True, text=True,
        )

    # Step 3 — query sqlite
    kern_csv = _run_nsys_stats(rep_base, "cuda_gpu_kern_sum")
    kern     = _parse_kern_sum(kern_csv)
    return kern["gpu_exec_ms"]


def run_ncol_sweep(steps: int, out_dir: Path, nz: int = NZ):
    """
    Sweep ncol over NCOL_SWEEP.  For each ncol:
      - 1 warmup + STEPS unprofiled timed runs  → first-call and steady-state
        wall-clock (full binary process: CPU init + H2D + GPU + D2H).
      - 1 nsys-profiled run  → clean GPU kernel exec time (cuda_gpu_kern_sum).

    Writes ncol_sweep_table.txt with three sections:
      1. First-call wall-clock (ms)
      2. Steady-state wall-clock (ms, mean/min/max)  — full binary process
      3. GPU exec time (ms, from nsys)  — comparable to JAX cached execute
    """
    import statistics

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ncol_sweep_table.txt"

    ncols = NCOL_SWEEP
    first_call:  dict[int, float] = {}
    mean_ms:     dict[int, float] = {}
    min_ms:      dict[int, float] = {}
    max_ms:      dict[int, float] = {}
    gpu_exec_ms: dict[int, float] = {}

    print(f"\n[ncol_sweep]  binary={BINARY.name}  nz={nz}  steps={steps}")
    print(f"  ncol sweep: {ncols}")
    print(f"  per ncol: 1 warmup + {steps} timed runs (wall-clock) + 1 nsys run (GPU exec)\n")

    for ncol in ncols:
        # ── wall-clock timing ──────────────────────────────────────────
        print(f"  ncol={ncol:<6}  warmup ... ", end="", flush=True)
        warmup_s, _, _ = run_binary(ncol, nz)
        first_call[ncol] = warmup_s * 1000
        print(f"{first_call[ncol]:>8.3f} ms", end="", flush=True)

        times = []
        for _ in range(steps):
            elapsed, _, _ = run_binary(ncol, nz)
            times.append(elapsed * 1000)

        mean_ms[ncol] = statistics.mean(times)
        min_ms[ncol]  = min(times)
        max_ms[ncol]  = max(times)
        print(f"  |  mean={mean_ms[ncol]:.3f}  min={min_ms[ncol]:.3f}  max={max_ms[ncol]:.3f} ms", end="", flush=True)

        # ── nsys GPU exec ──────────────────────────────────────────────
        print(f"  |  nsys ... ", end="", flush=True)
        gpu_exec_ms[ncol] = _nsys_gpu_exec_for_ncol(ncol, nz, out_dir)
        print(f"gpu_exec={gpu_exec_ms[ncol]:.3f} ms")

    # ------------------------------------------------------------------
    # Write table
    # ------------------------------------------------------------------
    now   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    col_w = 14

    def _fv(v): return f"{v:>{col_w}.3f}" if v == v else f"{'N/A':>{col_w}}"

    lines = [
        "ACC Fortran — ncol sweep",
        f"Generated: {now}",
        f"ncol sweep: {ncols}  |  nz={nz}",
        f"Wall-clock: 1 warmup + {steps} timed runs per ncol (full binary process)",
        "GPU exec:   1 nsys-profiled run per ncol (cuda_gpu_kern_sum, pure kernel time)",
        "",
        "Note: OpenACC Fortran is compiled ahead-of-time with nvhpc — no per-ncol",
        "JIT compilation.  Wall-clock includes CPU init + H2D + GPU kernels + D2H.",
        "GPU exec is the clean comparison metric against JAX cached execute time.",
        "",
        "WARNING (ncol >= 100000): Wall-clock timing becomes increasingly dominated by",
        "H2D/D2H transfers and CPU array allocation/RNG, not GPU physics.  At ncol=1000000",
        "each double-precision array is ~448 MB; several arrays together reach 3-5 GB of",
        "transfer overhead.  Do NOT use wall-clock columns at these scales to compare against",
        "JAX cached-execute time (which pre-stages arrays on GPU and pays no H2D/D2H cost).",
        "Use GPU exec (nsys cuda_gpu_kern_sum) as the sole valid comparison metric.",
        "",
        "=== First-Call Wall-Clock (ms, warmup run — full binary process) ===",
        "",
        f"{'ncol':>8}  {'ACC Fortran':>{col_w}}",
        "-" * (8 + 2 + col_w),
    ]
    for ncol in ncols:
        lines.append(f"{ncol:>8}  {_fv(first_call[ncol])}")

    lines += [
        "",
        f"=== Steady-State Wall-Clock (ms, mean of {steps} runs — full binary process) ===",
        "",
        f"{'ncol':>8}  {'mean':>{col_w}}  {'min':>{col_w}}  {'max':>{col_w}}",
        "-" * (8 + 2 + col_w * 3 + 4),
    ]
    for ncol in ncols:
        lines.append(
            f"{ncol:>8}  {_fv(mean_ms[ncol])}  {_fv(min_ms[ncol])}  {_fv(max_ms[ncol])}"
        )

    lines += [
        "",
        "=== GPU Exec Time (ms, nsys cuda_gpu_kern_sum — pure kernel time) ===",
        "",
        f"{'ncol':>8}  {'ACC Fortran':>{col_w}}",
        "-" * (8 + 2 + col_w),
    ]
    for ncol in ncols:
        lines.append(f"{ncol:>8}  {_fv(gpu_exec_ms[ncol])}")

    lines += [""]

    out_path.write_text("\n".join(lines))
    print(f"\n  Table written to: {out_path}")
    print(f"  nsys traces in:   {out_dir / 'nsys'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="OpenACC Fortran profiling runner for driver_kessler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode", required=True,
        choices=["timing", "nvidia_smi", "memory", "nsys", "ncol_sweep"],
        help="Profiling mode to run (one at a time)",
    )
    p.add_argument(
        "--ncol", type=int, default=NCOL,
        help=f"Number of columns (default: {NCOL}, matches JAX profiling)",
    )
    p.add_argument(
        "--nz", type=int, default=NZ,
        help=f"Number of vertical levels (default: {NZ})",
    )
    p.add_argument(
        "--steps", type=int, default=20,
        help="Number of runs for timing/memory/nvidia_smi modes (default: 20)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    check_binary()

    print(f"Binary  : {BINARY}")
    print(f"Grid    : ncol={args.ncol}  nz={args.nz}  dt={DT} s")
    print(f"Mode    : {args.mode}")
    if args.mode not in ("nsys", "ncol_sweep"):
        print(f"Steps   : {args.steps}")
    elif args.mode == "ncol_sweep":
        print(f"Steps   : {args.steps} per ncol")

    gpu_mb = gpu_memory_mb()
    if gpu_mb >= 0:
        print(f"GPU mem : {gpu_mb} MB (baseline)")

    out_dir = SRC_DIR / "outputs" / "profiling" / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "timing":
        run_timing(args.steps, out_dir, args.ncol, args.nz)

    elif args.mode == "nvidia_smi":
        run_nvidia_smi(args.steps, out_dir, args.ncol, args.nz)

    elif args.mode == "memory":
        run_memory(args.steps, out_dir, args.ncol, args.nz)

    elif args.mode == "nsys":
        run_nsys(args.steps, out_dir, args.ncol, args.nz)

    elif args.mode == "ncol_sweep":
        run_ncol_sweep(args.steps, out_dir, args.nz)
