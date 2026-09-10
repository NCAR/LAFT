#!/bin/bash -l
#PBS -N acc_kessler_profile
#PBS -A <PROJECT_CODE>
#PBS -q develop
#PBS -j oe
#PBS -o out/jobs/
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=00:30:00

# =============================================================================
# Profiling Runner for driver_kessler (OpenACC Fortran)
# =============================================================================
# Runs data/exp2_jd/acc_src/acc_kessler_profiling.py against the OpenACC
# Fortran binary data/exp2_jd/acc_src/driver_kessler (built there with
# `make ARCH=GPU`).  Grid default: ncol=1000, nz=56, dt=60 s.
#
# Available modes:
#   timing      -- warmup + N timed runs (wall-clock, ms). Best for: steady-
#                  state execution time and run-to-run variance.
#
#   nvidia_smi  -- poll GPU memory before/after each run. Best for: quick
#                  sanity check — is memory flat, growing, or spiking?
#
#   memory      -- repeated runs with nvidia-smi snapshots after each step.
#                  Best for: spotting memory growth across simulated time steps.
#
#   nsys        -- launches binary under NVIDIA Nsight Systems. Best for: full
#                  OpenACC/CUDA kernel timeline. Produces a .nsys-rep file.
#
# -----------------------------------------------------------------------------
# RECOMMENDED ORDER — run one mode per PBS submission, in this sequence:
#
#   Step 1 — nvidia_smi  (quick memory sanity check, ~2 min)
#            Runs STEPS calls and prints a GPU memory delta table.
#            If delta is always 0, memory is clean.  Any positive drift
#            flags a potential leak worth investigating further.
#
#   Step 2 — memory  (~5 min)
#            Records nvidia-smi snapshots after each step.
#            Confirms whether memory growth seen in step 1 is real.
#            Output: data/exp2_jd/acc_src/outputs/profiling/memory/memory_log.txt
#
#   Step 3 — timing  (~5 min)
#            1 warmup + STEPS timed runs.  Reports first-call vs steady-state
#            wall-clock time and writes a summary table.
#            Output: data/exp2_jd/acc_src/outputs/profiling/timing/timing_table.txt
#
#   Step 4 — nsys  (~10 min)
#            Captures the full OpenACC/CUDA timeline.  View the .nsys-rep
#            file in the Nsight Systems GUI to see kernel durations, GPU
#            utilization, and host-device transfer stalls.
#            Output: data/exp2_jd/acc_src/outputs/profiling/nsys/driver_kessler.nsys-rep
#
# All outputs go to: data/exp2_jd/acc_src/outputs/profiling/<mode>/
# Submit FROM THE kessler PROJECT ROOT:
#   qsub ../LAFT/pbsJobs/acc_kessler_profile.sh
# =============================================================================

# --- change this between runs ---
# MODE="nvidia_smi"
# MODE="memory"
# MODE="timing"
MODE="nsys"

NCOL=1000  # number of columns passed to driver_kessler (matches JAX profiling)
NZ=56       # number of vertical levels (default: 56)
STEPS=20    # number of runs (used by timing, nvidia_smi, memory modes)

set -euo pipefail

cd "${PBS_O_WORKDIR:?submit this job with qsub from the kessler project root}"
ACC_SRC=data/exp2_jd/acc_src
SCRIPT=$ACC_SRC/acc_kessler_profiling.py
BINARY=$ACC_SRC/driver_kessler

echo "=== Host ==="
hostname
date

echo "=== Modules ==="
module purge
module load ncarenv/25.10
module load nvhpc/26.1
module load cuda/12.9.0

echo "=== GPU ==="
nvidia-smi || true

echo "=== Binary ==="
ls -lh $BINARY

echo ""
echo "=== Profiling mode: $MODE ==="
echo ""

# ---------------------------------------------------------------------------
# timing / nvidia_smi / memory — run the Python profiling script directly
# ---------------------------------------------------------------------------
if [[ "$MODE" == "timing" || "$MODE" == "nvidia_smi" || "$MODE" == "memory" ]]; then

    python3 $SCRIPT --mode $MODE --ncol $NCOL --nz $NZ --steps $STEPS

# ---------------------------------------------------------------------------
# nsys — run the Python script which internally invokes nsys profile
# ---------------------------------------------------------------------------
elif [[ "$MODE" == "nsys" ]]; then

    python3 $SCRIPT --mode nsys --ncol $NCOL --nz $NZ

else
    echo "Unknown MODE: $MODE"
    echo "Valid modes: timing  nvidia_smi  memory  nsys"
    exit 1
fi

echo ""
echo "=== Done ==="
date
