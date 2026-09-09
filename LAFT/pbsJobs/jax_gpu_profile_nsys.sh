#!/bin/bash -l
#PBS -N jax_profile
#PBS -A <PROJECT_CODE>
#PBS -q develop
#PBS -j oe
#PBS -o out/jobs/
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=00:30:00

# =============================================================================
# jax_gpu_profile_nsys.sh — MANDATORY post-convergence hardware characterization
# =============================================================================
# NOT part of the iteration loop. The loop's single profiling job is
# jax_gpu_profile_trace.sh (jax.profiler.trace + compiled-HLO), which writes
# everything diagnose.py gates on. Run THIS job once, after the loop
# converges (PROFILE_WORKFLOW.md Step 6), to characterize the final code at
# the hardware layer (device kernel times, CUDA-graph sync idle, memcpy
# timing) for the headroom analysis (Step 7) and the report.
# diagnose.py attaches the resulting stats via its optional --nsys-json
# flag as an informational hardware_corroboration block — it never affects
# any check or production_ready.
#
# Profiles the staged translation ([profiler].staged_code in
# config/project.toml) and writes artifacts to <out_dir>/iteration_<N>/
# (point ITERATION at the final converged iteration).
#
# Submit FROM THE PROJECT ROOT with an ITERATION selector:
#
#   qsub -v ITERATION=1 pbsJobs/jax_gpu_profile_nsys.sh
#
# Optional overrides (qsub -v):
#   NCOL          columns to profile at        (default: [profiler].ncol)
#   N_CALLS       profiled JIT-warm calls      (default: 6)
#   CODE_PATH     file to profile              (default: [profiler].staged_code)
#
# Outputs (under <out_dir>/iteration_<N>/, <proc> = [profiler].target_proc):
#   <proc>.nsys-rep                      nsys binary timeline (view in nsys-ui)
#   <proc>.sqlite                        nsys database (queryable)
#   <proc>_cuda_gpu_kern_sum.json        cuda_gpu_kern_sum export
#   <proc>_cuda_api_sum.json             cuda_api_sum export
#   <proc>_stats.json                    merged stats — input to diagnose.py
#   <proc>_code_info.json                static code metadata — input to diagnose.py
#
# Log: out/jobs/jax_profile.o<JOBID>. Account, queue, and env are set directly in
# this script's headers, not read from config/project.toml.
# =============================================================================

set -euo pipefail

# --- Validate ITERATION ----------------------------------------------------
ITERATION="${ITERATION:-}"
if [[ -z "${ITERATION}" ]] || ! [[ "${ITERATION}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ITERATION must be a positive integer (use 'qsub -v ITERATION=N')."
    exit 1
fi

# --- Project root: the submit directory ------------------------------------
cd "${PBS_O_WORKDIR:?submit this job with qsub from the project root}"
[ -f config/project.toml ] || { echo "ERROR: $PWD is not a LAFT project root (config/project.toml missing)"; exit 1; }
PROJECT_ROOT="$PWD"

toml_str() { sed -n "s/^$1 *= *\"\([^\"]*\)\".*/\1/p" config/project.toml | head -1; }
PROC="$(toml_str target_proc)"
PROF_OUT="$(toml_str out_dir)"
[ -n "$PROC" ] || { echo "ERROR: [profiler].target_proc not found in config/project.toml"; exit 1; }

OUT_DIR="${PROJECT_ROOT}/${PROF_OUT:-out/profiled}/iteration_${ITERATION}"
RUNNER="${PROJECT_ROOT}/workflow_profiler/profile_run.py"
N_CALLS="${N_CALLS:-6}"
EXTRA_ARGS=()
[[ -n "${CODE_PATH:-}" ]] && EXTRA_ARGS+=(--code-path "${CODE_PATH}")
[[ -n "${NCOL:-}" ]]      && EXTRA_ARGS+=(--ncol "${NCOL}")

if [[ ! -f "${RUNNER}" ]]; then
    echo "ERROR: runner script not found: ${RUNNER}"
    exit 1
fi

mkdir -p "${OUT_DIR}"
PREFIX="${OUT_DIR}/${PROC}"

echo "============================================"
echo " nsys hardware profiling — iteration ${ITERATION}"
echo " Host       : $(hostname)"
echo " Started    : $(date)"
echo " Target     : ${PROC}"
echo " Out dir    : ${OUT_DIR}"
echo " n_calls    : ${N_CALLS}"
echo " Overrides  : ${EXTRA_ARGS[*]:-(config defaults)}"
echo "============================================"

# --- Environment -----------------------------------------------------------
module purge
module load conda/latest
conda activate jax-validate
module unload cuda openmpi || true
module load ncarenv/24.12
module load nvhpc/25.9  # for nsys

echo "[gpu] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true)"
python -c "import jax; print('[jax] devices =', jax.devices())"

# --- Step 0: state cache (OUTSIDE the nsys-instrumented process) ------------
echo ""
echo "[0/3] Ensuring captured driver state ..."
python3 workflow_profiler/profiler_inputs.py --ensure

# --- Step 1: nsys profile --------------------------------------------------
echo ""
echo "[1/3] Running nsys profiler ..."
nsys profile \
    --trace=cuda,nvtx,osrt \
    --stats=true \
    --output="${PREFIX}" \
    --force-overwrite=true \
    --export=sqlite \
    python3 "${RUNNER}" \
        --out-dir "${OUT_DIR}" \
        --n-calls "${N_CALLS}" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "  -> ${PREFIX}.nsys-rep"
echo "  -> ${PREFIX}.sqlite"

# --- Step 2: export nsys stats to JSON -------------------------------------
echo ""
echo "[2/3] Exporting nsys stats to JSON ..."

# nsys stats appends "_<report>" to --output, so --output "${PREFIX}" yields
# "${PREFIX}_cuda_gpu_kern_sum.json" and "${PREFIX}_cuda_api_sum.json".
nsys stats "${PREFIX}.nsys-rep" \
    --report cuda_gpu_kern_sum \
    --format json \
    --output "${PREFIX}" \
    --force-export true

nsys stats "${PREFIX}.nsys-rep" \
    --report cuda_api_sum \
    --format json \
    --output "${PREFIX}" \
    --force-export true

echo "  -> ${PREFIX}_cuda_gpu_kern_sum.json"
echo "  -> ${PREFIX}_cuda_api_sum.json"

# --- Step 3: merge nsys stats into the diagnose.py input format ------------
echo ""
echo "[3/3] Merging stats into diagnose.py format ..."

python3 - "${PREFIX}" <<'PYEOF'
import json
import sys
from pathlib import Path

prefix = Path(sys.argv[1])

kern_path = prefix.with_name(prefix.name + "_cuda_gpu_kern_sum.json")
api_path  = prefix.with_name(prefix.name + "_cuda_api_sum.json")


def norm_key(k):
    # "Total Time (ns)" -> "total_time_ns", "Num Calls" -> "num_calls",
    # "Time (%)" -> "time_pct", "Name" -> "name".
    k = str(k).lower().strip().replace("(%)", "pct").replace("(ns)", "ns")
    for ch in "()%":
        k = k.replace(ch, " ")
    return "_".join(k.split())


def load_rows(path, report):
    # `nsys stats --format json` writes a flat list of row dicts; some
    # versions nest it under the report name. Handle both, and normalize
    # the Title-Case nsys column names to the lowercase keys diagnose.py
    # expects (name, instances, num_calls).
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get(report, [])
    return [{norm_key(k): v for k, v in row.items()} for row in data]


# Note: cumulative counts span the whole process (1 warmup call +
# N_CALLS profiled calls, plus XLA autotuning during compilation).
# This is fine for the block's informational role; it is one reason
# these stats must never gate the loop's checks.
merged = {
    "cuda_gpu_kern_sum": load_rows(kern_path, "cuda_gpu_kern_sum"),
    "cuda_api_sum":      load_rows(api_path, "cuda_api_sum"),
}

stats_path = prefix.with_name(prefix.name + "_stats.json")
with open(stats_path, "w") as f:
    json.dump(merged, f, indent=2)

print(f"  -> {stats_path}")
PYEOF

# --- Done ------------------------------------------------------------------
echo ""
echo "============================================"
echo " Job finished : $(date)"
echo " Artifacts    : ${OUT_DIR}/"
echo "   - ${PROC}.nsys-rep      (open in nsys-ui)"
echo "   - ${PROC}_stats.json    (input to diagnose.py)"
echo "   - ${PROC}_code_info.json"
echo "============================================"
echo ""
echo "Next (optional): attach these hardware stats to the final diagnosis and"
echo "cite them in the profile report:"
echo "  python workflow_profiler/diagnose.py \\"
echo "    --trace-json ${OUT_DIR}/${PROC}_trace.json \\"
echo "    --code-info  ${OUT_DIR}/${PROC}_code_info.json \\"
echo "    --nsys-json  ${OUT_DIR}/${PROC}_stats.json \\"
echo "    --out        ${OUT_DIR}/${PROC}_diagnosis.json"
