#!/bin/bash -l
#PBS -N compare_scalability_fortran
#PBS -A <PROJECT_CODE>
#PBS -q develop
#PBS -j oe
#PBS -o out/jobs/
#PBS -l select=1:ncpus=1
#PBS -l walltime=00:30:00

# =============================================================================
# Serial Fortran (CPU) scalability baseline — the Fortran half of the paper's
# Fortran-vs-JAX comparison. Pairs with compare_scalability_LLMs.sh (the JAX
# half): same grid (nz=56), same ncol sweep [50 ... 1,000,000], so the two
# JSONs are directly comparable.
# =============================================================================
# Compiles and runs data/exp2_jd/src/scalability_benchmark_fortran.F90 with
# gfortran -O2, single core, no OpenACC/OpenMP (the directives in kessler.F90
# are inert). One warmup + 5 timed reps per ncol, median reported.
#
# Output: _compare_results/outputs/scalability/fortran_scalability_results.json
# (the Fortran code writes this path RELATIVE to the project root, so the
#  binary is run from there).
#
# Estimated serial runtimes (~0.05 ms/col): ncol=1e5 -> ~30 s, ncol=1e6 -> ~5 min.
#
# Submit FROM THE kessler PROJECT ROOT:
#   qsub ../LAFT/pbsJobs/compare_scalability_fortran.sh
# =============================================================================

set -euo pipefail

cd "${PBS_O_WORKDIR:?submit this job with qsub from the kessler project root}"
SRC=data/exp2_jd/src
BINARY=$SRC/scalability_benchmark_fortran
OUT_DIR=_compare_results/outputs/scalability

echo "=== Host ==="
hostname
date

echo "=== Modules ==="
module purge
module load ncarenv/25.10      # NCAR site environment; provides gfortran

echo "=== Compiler ==="
gfortran --version

echo "=== Compile ==="
# -ffree-line-length-none: kessler.F90 has source lines > 132 columns.
gfortran -O2 -ffree-line-length-none \
    -o "$BINARY" \
    "$SRC/ccpp_kinds.F90" \
    "$SRC/kessler.F90" \
    "$SRC/scalability_benchmark_fortran.F90"

echo "=== Binary ==="
ls -lh "$BINARY"

mkdir -p "$OUT_DIR"

echo ""
echo "=== ncol sweep ==="
echo ""
"$BINARY"

echo ""
echo "=== Done ==="
date
echo "Results: $OUT_DIR/fortran_scalability_results.json"
