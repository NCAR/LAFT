#!/bin/bash -l
#PBS -N runtimevalidate
#PBS -A <PROJECT_CODE>
# use -q casper when run on casper or -q main when run on derecho 
#PBS -q develop 
#PBS -j oe
#PBS -o out/jobs/
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=00:20:00

# Runtime validation (TRANSLATE_WORKFLOW.md step 4) for the translated
# scheme. Runs phase05_02_runtime_validate.py, which imports every
# translated module in out/jax/, smoke-tests wrappers through their bridges,
# and JIT-tests each _core. Results land in out/validation/<proc>_runtime.json.

set -euo pipefail

echo "=== Host ==="
hostname
date

echo "=== Modules ==="
module purge
module load conda/latest
conda activate jax-validate
module unload cuda openmpi
module load ncarenv/24.12 
echo "=== NVIDIA-SMI ==="
nvidia-smi || true

python -c "import jax; print(jax.lib.xla_bridge.get_backend().platform)"

# The validator uses relative paths (out/packets, out/jax, out/bridge,
# out/validation), so we must run from the project root.
# Project root: submit this job with qsub FROM the project root — PBS records
# the submit dir in PBS_O_WORKDIR. Account, queue, and conda env are set
# directly in this script's headers, not read from config/project.toml.
cd "${PBS_O_WORKDIR:?submit this job with qsub from the project root}"
[ -f config/project.toml ] || { echo "ERROR: $PWD is not a LAFT project root (config/project.toml missing)"; exit 1; }

# Step 3.5 gate: the semantic audit must have run on the CURRENT translations
# with 0 unwaived FAIL (workflow_translator/audit_gate.py). Refusing here means
# a skipped audit costs one queued job, not four driver/comparison cycles.
python3 workflow_translator/audit_gate.py || { echo "ERROR: semantic-audit gate closed — run 'python workflow_translator/phase05_01b_semantic_audit.py' (TRANSLATE_WORKFLOW.md Step 3.5) before runtime validation"; exit 1; }

python3 validation/phase05_02_runtime_validate.py


