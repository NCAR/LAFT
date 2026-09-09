#!/bin/bash -l
#PBS -N jaxdrvgpu
#PBS -A <PROJECT_CODE>
#PBS -q develop
#PBS -j oe
#PBS -o out/jobs/
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=01:00:00

# JAX driver on GPU: same run as jax_driver.sh ([driver].script in
# config/project.toml) but with a GPU allocated so the jitted cores run on
# the accelerator. For schemes with host-side orchestration over small
# arrays, kernel launch + host<->device transfer overhead can make this
# SLOWER than the CPU run — this job checks correctness/portability.
# Output overwrites [driver].output_file + [driver].summary_json.
#
# #PBS -N above must match [hpc].driver_job_name in config/project.toml if
# this job is configured as the workflow driver.

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

echo "=== NVIDIA-SMI ==="
nvidia-smi || true

python -c "import jax; print('JAX backend:', jax.lib.xla_bridge.get_backend().platform)"

# Project root: submit this job with qsub FROM the project root — PBS records
# the submit dir in PBS_O_WORKDIR. Account, queue, and conda env are set
# directly in this script's headers, not read from config/project.toml.
cd "${PBS_O_WORKDIR:?submit this job with qsub from the project root}"
[ -f config/project.toml ] || { echo "ERROR: $PWD is not a LAFT project root (config/project.toml missing)"; exit 1; }

# cfg_get <section> <key> — read a quoted string from config/project.toml
cfg_get() {
    awk -v sec="$1" -v key="$2" '
        /^\[/ { insec = (index($0, "[" sec "]") == 1) }
        insec && $0 ~ "^" key " *=" {
            if (match($0, /"[^"]*"/)) { print substr($0, RSTART+1, RLENGTH-2); exit }
        }' config/project.toml
}
DRIVER_SCRIPT="$(cfg_get driver script)"
[ -n "${DRIVER_SCRIPT}" ] || { echo "ERROR: [driver].script not found in config/project.toml"; exit 1; }

# -u: unbuffered, so per-minute progress appears in this log as it happens
python3 -u "${DRIVER_SCRIPT}"

echo "=== Done ==="
date
ls -la "$(cfg_get driver output_file)" "$(cfg_get driver summary_json)"
