#!/bin/bash -l
#PBS -N jax_cpu_test
#PBS -A <PROJECT_CODE>
#PBS -q develop
#PBS -j oe
#PBS -o out/jobs/
#PBS -l select=1:ncpus=1
#PBS -l walltime=00:15:00

# ---------------------------------------------------------------------------
# CPU-only pre-GPU sanity harness. Runs a Python script on a CPU COMPUTE node
# (never the login node) to trace/compile/arity-check translated JAX before
# spending a GPU runtime/driver job.
#
# Usage:
#   qsub -v TEST_SCRIPT=/abs/path/to/script.py pbsJobs/jax_cpu_test.sh
#   qsub -v TEST_SCRIPT=script.py,TEST_ARGS="--flag value" pbsJobs/jax_cpu_test.sh
# Bridge-suite gate (BRIDGE_WORKFLOW Step 2 / TRANSLATE_WORKFLOW Step 4.5):
#   qsub -v TEST_SCRIPT=workflow_bridge/run_bridge_tests.py pbsJobs/jax_cpu_test.sh
#   qsub -v TEST_SCRIPT=workflow_bridge/run_bridge_tests.py,TEST_ARGS=--require-dependent pbsJobs/jax_cpu_test.sh
# TEST_SCRIPT may be relative to the submit directory (the job cd's there).
# The script runs with JAX forced onto CPU so it needs no GPU.
# ---------------------------------------------------------------------------

set -euo pipefail

echo "=== Host ==="; hostname; date
echo "=== Modules ==="
module purge
module load conda/latest
conda activate jax-validate

cd "${PBS_O_WORKDIR:-.}"
export JAX_PLATFORMS=cpu
export PYTHONPATH="${PBS_O_WORKDIR:-.}:${PYTHONPATH:-}"

: "${TEST_SCRIPT:?set TEST_SCRIPT=/abs/path/to/script.py via 'qsub -v TEST_SCRIPT=...'}"
echo "=== Running (CPU): ${TEST_SCRIPT} ${TEST_ARGS:-} ==="
python "${TEST_SCRIPT}" ${TEST_ARGS:-}
echo "=== Done ==="; date
