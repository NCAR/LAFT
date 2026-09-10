#!/bin/bash -l
#PBS -N compare_scalability_LLMs
#PBS -A <PROJECT_CODE>
#PBS -q develop
#PBS -j oe
#PBS -o out/jobs/
#PBS -l select=1:ncpus=4:ngpus=1:mem=32GB
#PBS -l walltime=01:00:00

# =============================================================================
# Cross-LLM kernel scalability benchmark — the JAX half of the paper's
# Fortran-vs-JAX comparison. Pairs with compare_scalability_fortran.sh.
# =============================================================================
# Runs _compare_results/scalability_benchmark_LLMs.py on one GPU: times the
# jitted kessler_run_core of each of the four translation archives across the
# ncol sweep [50 ... 1,000,000] at nz=56 (device-resident inputs, 1 warmup +
# 5 timed calls, median reported; Qwen is skipped above ncol=10,000).
#
# Writes: _compare_results/outputs/scalability/LLMs_scalability_results.json
#         _compare_results/outputs/plots/LLMs_scalability_plot.png
# Then draw the combined figure with
#   python3 _compare_results/plot_Fortran_vs_LLMs_scalability.py
#
# Submit FROM THE kessler PROJECT ROOT:
#   qsub ../LAFT/pbsJobs/compare_scalability_LLMs.sh
# =============================================================================

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

cd "${PBS_O_WORKDIR:?submit this job with qsub from the kessler project root}"
python3 -u _compare_results/scalability_benchmark_LLMs.py

echo "=== Done ==="
date
