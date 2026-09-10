# pbsJobs — HPC batch scripts (NCAR reference)

These are the PBS Pro job scripts used to run the LAFT workflow on NCAR's
**Derecho** and **Casper** systems. They are shipped as working reference
examples: the framework's docs and state tools refer to them by name
(`qsub pbsJobs/<script>`), but nothing in them is specific to LAFT beyond
the Python entry point each one launches.

On any other cluster you will need to adapt the site-specific parts listed
below. The Python code the scripts call is portable and does not depend on
these settings.

## Scripts

| Script | Phase | Resources |
|---|---|---|
| `jax_cpu_test.sh` | CPU-node runner for any JAX check: the bridge-suite gate (`run_bridge_tests.py`, bridge workflow Step 2 and translator Step 4.5) and ad-hoc sanity scripts, via `-v TEST_SCRIPT=...,TEST_ARGS=...` | 1 CPU |
| `jax_gpu_runtimevalid.sh` | Phase 4: runtime validation (import / smoke / jit-trace) | 1 GPU |
| `jax_driver.sh` | Phase 5: project driver on CPU | 1 CPU |
| `jax_driver_gpu.sh` | Phase 5: project driver on GPU | 1 GPU |
| `jax_gpu_compvalues.sh` | Phase 5: compare driver output against the Fortran reference | 1 CPU |
| `jax_gpu_profile_trace.sh` | Profiler: `jax.profiler.trace` + XLA HLO dump | 1 GPU |
| `jax_gpu_profile_nsys.sh` | Profiler: NVIDIA Nsight Systems kernel/memcpy stats | 1 GPU + `nsys` |
| `pbs_qwen25_translate.sh` | Phase 4: the **paper-1 Qwen translation** — Qwen2.5-Coder-32B-Instruct via vLLM with the April 2026 settings fixed | 2 GPUs |
| `acc_kessler_profile.sh` | OpenACC reference (`kessler/data/exp2_jd/acc_src/`): timing / memory / nsys profiling of the compiled Fortran binary, one mode per submission | 1 GPU + `nsys` |
| `acc_kessler_ncol_sweep.sh` | OpenACC reference: per-ncol wall-clock and nsys kernel-time sweep, the OpenACC column of the paper's scalability figure | 1 GPU + `nsys` |

The `jax_*` scripts assume the project layout described in
`LAFT/docs/architecture.md` and are submitted from the project root, so that
`PBS_O_WORKDIR` resolves the `out/` tree. The `acc_*` scripts are submitted from the `kessler/` project root and
expect the OpenACC binary to be built first (`make ARCH=GPU` in
`data/exp2_jd/acc_src/`). `pbs_qwen25_translate.sh` is only
needed to re-run the paper's locally hosted Qwen translation; the
in-context agents (Claude Code, OpenAI Codex, Gemini CLI) author the
translation in their own session and need no batch job for the LLM call.

## What to change for another system

1. **Allocation / account.** Every script carries `#PBS -A <PROJECT_CODE>`.
   Replace the placeholder with your project code, or leave the file alone
   and pass it at submit time, which PBS honours over the in-file directive:

   ```bash
   qsub -A MYPROJ0001 pbsJobs/jax_gpu_runtimevalid.sh
   ```

2. **Queue.** `#PBS -q develop` is the NCAR debug/development queue. Use
   `main` on Derecho, `casper` on Casper, or whatever your site provides.
   Some scripts note this in a comment next to the `-q` line.

3. **Resource strings.** `#PBS -l select=1:ncpus=N:ngpus=M:mem=X` uses PBS Pro
   syntax with NCAR's resource names. Slurm or a differently configured PBS
   will need the equivalent directives.

4. **Environment modules.** The scripts run:

   ```bash
   module load conda/latest
   module load ncarenv/24.12      # NCAR site environment
   module load nvhpc/25.9         # only jax_gpu_profile_nsys.sh, for nsys
   module load ncarenv/25.10 nvhpc/26.1 cuda/12.9.0   # acc_* scripts: nvfortran + nsys
   ```

   Replace these with whatever provides `conda`, a CUDA runtime, and (for the
   nsys job) Nsight Systems on your cluster.

5. **Conda environments.** The `jax_*` scripts activate `jax-validate`,
   built from `../envs/jax-validate.yml`; `pbs_qwen25_translate.sh` activates
   `qwen-vllm`, built from `../envs/qwen-vllm.yml` (override with
   `-v QWEN_ENV=...`). Use the recipes
   as they are, or edit the `conda activate` line to your own env name. See
   `../docs/README.md` §Environments.

6. **Filesystem paths.** The Qwen job defaults the model weights to
   `/glade/work/$USER/models/Qwen2.5-Coder-32B-Instruct` (NCAR's `/glade`
   tree); override at submit time with `qsub -v MODEL=/path/to/weights ...`.

7. **Job output directory.** `#PBS -o out/jobs/` is relative to the submit
   directory; create it first or point it elsewhere.

Nothing else in the scripts is site-dependent, and no secrets are stored in
them: the in-context agents authenticate through their own tooling, and the
Qwen jobs use local weights only.
