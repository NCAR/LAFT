#!/bin/bash -l
#PBS -N acc_kessler_ncol_sweep
#PBS -A <PROJECT_CODE>
#PBS -q develop
#PBS -j oe
#PBS -o out/jobs/
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=00:45:00

# =============================================================================
# ncol sweep for driver_kessler (OpenACC Fortran)
# =============================================================================
# Runs data/exp2_jd/acc_src/acc_kessler_profiling.py --mode ncol_sweep against
# the OpenACC binary built there with `make ARCH=GPU`.
#
# For each ncol in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 100000, 1000000]:
#   - 1 warmup run  (first-call time — includes GPU context init on first ncol)
#   - STEPS timed runs  (steady-state mean/min/max)
#
# Output: data/exp2_jd/acc_src/outputs/profiling/ncol_sweep/ncol_sweep_table.txt
#
# The table has the same shape as the JAX per-ncol compile/execute sweep, so
# the two can be compared directly. OpenACC has no per-ncol JIT: compile cost
# is zero at runtime.
#
# Submit FROM THE kessler PROJECT ROOT:
#   qsub ../LAFT/pbsJobs/acc_kessler_ncol_sweep.sh
# =============================================================================

STEPS=5    # timed runs per ncol
NZ=56

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
echo "=== ncol sweep (steps=$STEPS per ncol, nz=$NZ) ==="
echo ""

python3 $SCRIPT --mode ncol_sweep --nz $NZ --steps $STEPS

echo ""
echo "=== Done ==="
date
