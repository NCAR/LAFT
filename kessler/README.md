# kessler — the LAFT paper-1 experiment

Fortran → JAX translation of the Kessler warm-rain microphysics scheme by four
LLMs, validated against the Fortran reference with the LAFT pipeline in
`../LAFT/`. This folder is a LAFT *project*: `config/project.toml` is read with
this directory as the project root, and the pipeline scripts and PBS jobs in
`../LAFT/` are run from here.

## Layout

| Folder | What it holds |
|---|---|
| `config/` | `project.toml` — the LAFT project configuration (source file, reference data, driver, comparison, profiler, HPC job names). |
| `data/exp2_jd/` | The Fortran reference. `src/` has the scheme (`kessler.F90`, `kessler_update.F90`), the CCPP metadata, the reference driver (`JD_kessler_driver.F90`), the input generator, a Makefile, the scalability benchmark source, and the golden compiler logs. `acc_src/` has the manually accelerated OpenACC GPU version of the scheme with its driver, Makefile, and profiling runner (see `acc_src/README.md`). `fortran_io/` has the 128-column × 56-level inputs and the Fortran outputs every translation is compared against. |
| `out/` | Snapshot of the LAFT pipeline artifacts shared by every run: the frontend extraction (`phase1_index.json`, `modules/`, `procedures/`, `packets/`), the generated bridge (`bridge/`), wrappers, the JAX driver and the Fortran-vs-JAX comparison script (`driver/`), and the profiler input script. `out/jax/` is intentionally empty — each model's translation lives under `translations/`. |
| `bridge_test/` | The project's bridge unit tests (pytest). They import from `out/bridge/` and, for the translation-dependent tests, from `out/jax/`; those skip themselves when no translation is staged there. See `bridge_test/TESTING_GUIDE.md`. |
| `translations/` | One archive per model, described below. |
| `_compare_results/` | Cross-model analysis of the April 2026 runs: the performance comparison report, accuracy and GPU scalability plots, and Nsight Systems profiling reports. |

## The four translations

All four were produced in April 2026 from the JD-data Fortran (`data/exp2_jd/`).
On 2026-09-04 each was re-validated, unchanged, under the current LAFT bridge:
the JAX files were copied verbatim into the pipeline (a version header was
added), pushed through lint, semantic audit, runtime validation, the bridge test
suite, the driver, and the Fortran comparison, with the loop and vectorization
structure frozen. Only interface-level repairs were allowed, and each archive's
`issues/fix_log.md` records what, if anything, was fixed.

| Archive | Model | Column strategy | Fixes in re-validation |
|---|---|---|---|
| `claude-sonnet46-jd/` | Claude Sonnet 4.6 | `jax.vmap`, index-array level range with scatters | 1 (error-flag precedence) |
| `gemini31pro-jd/` | Gemini 3.1 Pro (via Gemini CLI) | `jax.vmap`, full-level masked kernel, no scatters | 0 |
| `gpt54thinking-jd/` | GPT-5.4 Thinking | `jax.vmap`, index-array level range with scatters | 1 (error-flag precedence, same as Sonnet) |
| `qwen25-32b-jd/` | Qwen2.5-32B (Qwen2.5-Coder-32B-Instruct via vLLM) | serial `lax.fori_loop` over columns, no `vmap` | 1 (single-level harness guard) |

Every archive has the same shape: `README.md` (model, provenance, gate results,
design summary), `jax/` (the translation, plus the `.validated.py` copies the
gates approved), `prompts/`, `lint/`, `validation/`, `issues/` (semantic audit
and fix log), `reports/` (translation and bridge reports), and
`driver/results/` (the JAX outputs and the comparison against Fortran).

All four pass every gate and match the Fortran outputs to machine precision
(mean absolute error in `theta` of about 7e-16 K). The Gemini, GPT, and Qwen
translations produce byte-identical outputs to one another; Sonnet's differ in
the last bits.

### Reading the provenance notes

The JAX file headers, fix logs, and READMEs cite `_2sd_exp_JDdata/<model>/jax/`
as the verbatim source of each translation. That names the April 2026 working
folder in the project these archives were copied from; the same code is what
sits in each archive's `jax/`. The `_compare_results` reports and plots refer to
the models by their short names (Claude, GPT, Gemini, Qwen); those are these
same four translations, analysed in April before the re-validation.

The reports and fix logs also compare digits against "earlier archived" runs:
those are runs of the working project that are not part of this paper and are
not included here.

## Reference data caveat

The reference inputs in `data/exp2_jd/fortran_io/` record `seed = 0` in
`meta.txt`, while `generate_fortran_inputs_jd.py` declares `SEED = 42`. The
on-disk data is the authoritative reference the translations were validated
against; regenerating from the script may not reproduce it bit for bit. The JD
`kessler.F90` needs `gfortran -ffree-line-length-none`, which the Makefile sets.

## Running the pipeline

`ORCHESTRATOR.md` here is a symlink to the framework's orchestrator. It is
meant to be executed by an AI coding agent (Claude Code, Gemini CLI, or
OpenAI Codex) started in this directory, with a person approving each stage;
see the top-level `README.md`.

## Running on another system

Two conda environments are needed, built from the pinned recipes in
`../LAFT/envs/`: `ts` for the Stage 0 tree-sitter parse and `jax-validate`
(JAX 0.6.2, the version every comparison in `translations/*/driver/results`
was produced with) for everything else, including `bridge_test/`. See
`../LAFT/docs/README.md` §Environments.

The PBS job scripts referenced from `config/project.toml` live in
`../LAFT/pbsJobs/` and are written for NCAR's Derecho and Casper. See
`../LAFT/pbsJobs/README.md` for what to change.
