#!/bin/bash -l
#PBS -N comparevalues
#PBS -A <PROJECT_CODE>
# use -q casper when run on casper or -q main when run on derecho
#PBS -q develop 
#PBS -j oe
#PBS -o out/jobs/
#PBS -l select=1:ncpus=1
#PBS -l walltime=00:05:00

# This job runs the PROJECT'S comparison script ([comparison].script in
# config/project.toml): it reads the reference Fortran outputs and the JAX
# driver outputs and validates the correctness of the JAX translation.
# Comparison is project-specific (output format and pass criteria follow the
# reference driver) — the framework owns only this contract:
#   - the script path comes from [comparison].script
#   - exit 0 = PASS, non-zero = FAIL (this log is authoritative)
#   - it writes a human-readable report + a JSON with a top-level PASS/FAIL
#     status next to the JAX driver output

set -euo pipefail

echo "=== Host ==="
hostname
date

echo "=== Modules ==="
module purge
module load conda/latest
conda activate jax-validate

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
COMPARE_SCRIPT="$(cfg_get comparison script)"
[ -n "${COMPARE_SCRIPT}" ] || { echo "ERROR: [comparison].script not found in config/project.toml"; exit 1; }

python3 "${COMPARE_SCRIPT}" 

