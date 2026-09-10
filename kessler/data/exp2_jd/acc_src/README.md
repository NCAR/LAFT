# acc_src — the OpenACC GPU reference

The manually accelerated OpenACC version of the Kessler scheme used as the
GPU reference in the paper's scalability comparison (Section 5, Figure 3).
It is the exact source state that produced those numbers, so anyone can build
it, run it, and compare the timings with any JAX translation of their own.

| File | What it is |
|---|---|
| `kessler.F90` | The scheme with OpenACC directives (`!$acc parallel loop ...`). The driver-owned arrays are placed on the device by the driver and referenced with `default(present)`; the `DEVICEPTR` macro at the top of the file is intentionally empty for this build. |
| `kessler_update.F90` | The companion CCPP scheme the driver calls after the microphysics for diagnostic checksums. It only reads the Kessler fields and is not part of the timed call or of the translation work. |
| `ccpp_kinds.F90`, `*.meta` | Kind definitions and CCPP metadata, identical to `../src/`. |
| `driver_kessler.F90` | The GPU driver: allocates the arrays on the device (`omp_target_alloc` / `omp_target_associate_ptr`), fills them with physically structured random inputs (fixed seed), calls `kessler_run` once, and prints checksums. Takes `ncol nz` on the command line (default `1000 56`); `dt = 60 s`. |
| `Makefile` | `make ARCH=GPU` builds the OpenACC binary with nvfortran; `make` builds a CPU nvfortran binary; `make COMPILER=gnu` a gfortran one. FMA is disabled (`-Mnofma`) so results match the reference logs. |
| `golden*.log` | Reference checksums for `make compare` (identical to `../src/`). |
| `acc_kessler_profiling.py` | Profiling runner for the built binary: `timing`, `nvidia_smi`, `memory`, `nsys`, and `ncol_sweep` modes, writing under `outputs/profiling/`. The `ncol_sweep` mode with its nsys pass is what produced the OpenACC column of the paper's scalability figure. |

## Build and run

```bash
module load ncarenv/25.10 nvhpc/26.1 cuda/12.9.0   # NCAR Derecho; any NVIDIA HPC SDK with OpenACC support works
make ARCH=GPU
./driver_kessler 1000 56        # ncol nz
```

`driver_kessler` prints the elapsed time of the `kessler_run` call and the
output checksums. `make compare` checks a run against `golden-nofma.nvhpc.log`.

## Profiling

```bash
python acc_kessler_profiling.py --mode ncol_sweep --steps 5     # per-ncol wall-clock + nsys kernel time
python acc_kessler_profiling.py --mode nsys --ncol 1000         # kernel/memcpy summary table
```

The `nsys` and `ncol_sweep` modes need NVIDIA Nsight Systems (`nsys`) on the
PATH. The PBS jobs `../../../../LAFT/pbsJobs/acc_kessler_ncol_sweep.sh` and
`acc_kessler_profile.sh` run these modes on one A100; submit them from the
`kessler/` project root. The comparable numbers for a JAX translation come
from the LAFT profiler and scalability benchmark; the metric to compare
against JAX cached-execute time is the nsys `cuda_gpu_kern_sum` kernel time,
not the binary's wall-clock, which includes host-side setup and transfers.

The paper's OpenACC results are reported there and are not duplicated here.
