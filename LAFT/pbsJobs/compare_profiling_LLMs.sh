#!/bin/bash -l
#PBS -N compare_profiling_LLMs
#PBS -A <PROJECT_CODE>
#PBS -q develop
#PBS -j oe
#PBS -o out/jobs/
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=00:30:00

# =============================================================================
# Cross-LLM profiling of kessler_run_core — the job behind the Nsight Systems
# reports in kessler/_compare_results/outputs/reports/ (*_nsys_report_*.md).
# =============================================================================
# Runs _compare_results/LLMx_jd_profiling_run.py against the four translation
# archives. The core is called directly (no bridge) to isolate GPU kernel
# behaviour; inputs mirror the JD Fortran driver at NCOL columns x 56 levels.
#
# Modes (one per submission; set MODE below and resubmit):
#   log_compiles  -- log every JIT recompilation (unexpected retracing)
#   nvidia_smi    -- poll GPU memory before/after each call
#   memory        -- JAX device memory profiler across STEPS time steps
#   jax_trace     -- JAX/Perfetto trace per model (open at https://ui.perfetto.dev)
#   nsys          -- NVIDIA Nsight Systems timeline per model (1 warmup + 6
#                    profiled calls); the paper's reports were derived from the
#                    .nsys-rep / .sqlite files this writes, at NCOL=1000
#   ncu           -- NVIDIA Nsight Compute kernel-level profile, one model
#
# All outputs go to: _compare_results/outputs/profiling/<mode>/
#
# Submit FROM THE kessler PROJECT ROOT:
#   qsub ../LAFT/pbsJobs/compare_profiling_LLMs.sh
# =============================================================================

MODE="nsys"            # log_compiles | nvidia_smi | memory | jax_trace | nsys | ncu
NCOL=1000              # number of columns (the reports use 1000)
STEPS=20               # time steps (memory and nvidia_smi modes)
MODEL_NCU="gemini"     # ncu profiles one model at a time

set -euo pipefail

echo "=== Host ==="
hostname
date

echo "=== Modules ==="
module purge
module load conda/latest
conda activate jax-validate
module unload cuda openmpi || true
module load ncarenv/24.12
module load nvhpc/25.9         # provides nsys and ncu

echo "=== NVIDIA-SMI ==="
nvidia-smi || true

python -c "import jax; print('JAX backend:', jax.lib.xla_bridge.get_backend().platform); print(jax.devices())"

cd "${PBS_O_WORKDIR:?submit this job with qsub from the kessler project root}"
SCRIPT=_compare_results/LLMx_jd_profiling_run.py
OUT=_compare_results/outputs/profiling

echo ""
echo "=== Profiling mode: $MODE ==="
echo ""

if [[ "$MODE" == "jax_trace" || "$MODE" == "memory" || \
      "$MODE" == "log_compiles" || "$MODE" == "nvidia_smi" ]]; then

    python3 -u "$SCRIPT" --mode "$MODE" --ncol "$NCOL" --steps "$STEPS"

elif [[ "$MODE" == "nsys" ]]; then

    mkdir -p "$OUT/nsys"
    for MODEL in claude gpt gemini qwen; do
        echo "--- nsys: $MODEL ---"
        nsys profile \
            --trace=cuda,nvtx \
            --output="$OUT/nsys/${MODEL}" \
            --force-overwrite=true \
            python3 "$SCRIPT" --mode nsys --model "$MODEL" --ncol "$NCOL"
        echo "  -> $OUT/nsys/${MODEL}.nsys-rep"
    done
    # Kernel/memcpy tables, as used in the reports:
    #   nsys stats --report cuda_gpu_kern_sum --format csv $OUT/nsys/<model>.nsys-rep

elif [[ "$MODE" == "ncu" ]]; then

    mkdir -p "$OUT/ncu"
    echo "--- ncu: $MODEL_NCU ---"
    ncu \
        --set full \
        --export "$OUT/ncu/${MODEL_NCU}" \
        --force-overwrite \
        python3 "$SCRIPT" --mode ncu --model "$MODEL_NCU" --ncol "$NCOL"
    echo "  -> $OUT/ncu/${MODEL_NCU}.ncu-rep"

else
    echo "Unknown MODE: $MODE"
    echo "Valid modes: jax_trace  memory  log_compiles  nvidia_smi  nsys  ncu"
    exit 1
fi

echo ""
echo "=== Done ==="
date
