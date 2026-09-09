#!/bin/bash -l
#PBS -N jax_trace
#PBS -A <PROJECT_CODE>
#PBS -q develop
#PBS -j oe
#PBS -o out/jobs/
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=00:30:00

# =============================================================================
# jax_gpu_profile_trace.sh — THE profiling job of the iteration loop
# =============================================================================
# The DESIGN / STRUCTURE pass, and the iteration loop's ONLY PBS job. Runs
# the staged translation ([profiler].staged_code in config/project.toml)
# inside jax.profiler.trace (plus the second-ncol scaling check), collects the
# compiled HLO of every jitted module, and writes everything diagnose.py gates
# on into <out_dir>/iteration_<N>/:
#   <proc>_trace.json      (trace metrics + hlo block)
#   <proc>_code_info.json  (static counts)
#
# The nsys hardware lens (jax_gpu_profile_nsys.sh) is NOT part of the loop —
# it is the optional one-time post-convergence characterization for the
# report.
#
# This is "Phase A" of workflow_profiler/PROFILE_WORKFLOW.md — one job,
# one wait gate. Submit FROM THE PROJECT ROOT with an ITERATION selector:
#
#   qsub -v ITERATION=1 pbsJobs/jax_gpu_profile_trace.sh
#
# Optional overrides (qsub -v):
#   NCOL, NCOL_CHECK   profile / scaling-check column counts
#                      (defaults: [profiler].ncol / .ncol_check in the config)
#   CODE_PATH          file to profile (default: [profiler].staged_code)
#   OUT_DIR            artifact dir (default: <[profiler].out_dir>/iteration_<N>)
#
# The raw <host>.xplane.pb is deleted after parsing (TensorBoard-only, can be
# tens of MB). The compact .trace.json.gz is kept for Perfetto inspection.
#
# Log: out/jobs/jax_trace.o<JOBID>  (#PBS -j oe). Account, queue, and env are set
# directly in this script's headers, not read from config/project.toml.
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

# Config values (single-line TOML scalars parsed with sed)
toml_str() { sed -n "s/^$1 *= *\"\([^\"]*\)\".*/\1/p" config/project.toml | head -1; }
PROC="$(toml_str target_proc)"
PROF_OUT="$(toml_str out_dir)"
[ -n "$PROC" ] || { echo "ERROR: [profiler].target_proc not found in config/project.toml"; exit 1; }

OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/${PROF_OUT:-out/profiled}/iteration_${ITERATION}}"
RUNNER="${PROJECT_ROOT}/workflow_profiler/profile_trace_run.py"
EXTRA_ARGS=()
[[ -n "${CODE_PATH:-}" ]]  && EXTRA_ARGS+=(--code-path "${CODE_PATH}")
[[ -n "${NCOL:-}" ]]       && EXTRA_ARGS+=(--ncol "${NCOL}")
[[ -n "${NCOL_CHECK:-}" ]] && EXTRA_ARGS+=(--ncol-check "${NCOL_CHECK}")

if [[ ! -f "${RUNNER}" ]]; then
    echo "ERROR: runner script not found: ${RUNNER}"
    exit 1
fi

mkdir -p "${OUT_DIR}"

echo "============================================"
echo " JAX trace profiling — iteration ${ITERATION}"
echo " Host       : $(hostname)"
echo " Started    : $(date)"
echo " Target     : ${PROC}"
echo " Out dir    : ${OUT_DIR}"
echo " Overrides  : ${EXTRA_ARGS[*]:-(config defaults)}"
echo "============================================"

# --- Environment -----------------------------------------------------------
# No nsys / nvhpc module needed — jax.profiler.trace is in-process.
module purge
module load conda/latest
conda activate jax-validate
module unload cuda openmpi || true
module load ncarenv/24.12

echo "[gpu] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true)"
python -c "import jax; print('[jax] devices =', jax.devices())"

# --- Step 1: state cache (separate process: keeps the HLO dump clean) ------
echo ""
echo "[1/3] Ensuring captured driver state ..."
python3 workflow_profiler/profiler_inputs.py --ensure

# --- Step 2: jax.profiler.trace pass + parse -------------------------------
echo ""
echo "[2/3] Running jax.profiler.trace pass ..."
python3 "${RUNNER}" --out-dir "${OUT_DIR}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

# --- Step 3: prune bulky raw artifacts -------------------------------------
echo ""
echo "[3/3] Pruning raw xplane.pb and the per-module HLO dump ..."
# covers both jax_trace/ and the jax_trace_check/ scaling-check trace;
# the concatenated <proc>_hlo.txt already holds the per-module HLO text
find "${OUT_DIR}" -name "*.xplane.pb" -print -delete || true
rm -rf "${OUT_DIR}/hlo_dump"

# --- Done ------------------------------------------------------------------
echo ""
echo "============================================"
echo " Job finished : $(date)"
echo " Artifacts    : ${OUT_DIR}/"
echo "   - ${PROC}_trace.json      (input to diagnose.py --trace-json)"
echo "   - ${PROC}_code_info.json  (input to diagnose.py --code-info)"
echo "   - ${PROC}_hlo.txt         (concatenated compiled-HLO modules)"
echo "   - jax_trace/plugins/profile/<ts>/<host>.trace.json.gz"
echo "============================================"
echo ""
echo "Next: run diagnose.py (this was the loop's only profiling job):"
echo "  python workflow_profiler/diagnose.py \\"
echo "    --trace-json ${OUT_DIR}/${PROC}_trace.json \\"
echo "    --code-info  ${OUT_DIR}/${PROC}_code_info.json \\"
echo "    --out        ${OUT_DIR}/${PROC}_diagnosis.json"
