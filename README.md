# LAFT — LLM-Assisted Fortran Translation

[![DOI](https://zenodo.org/badge/1355304974.svg)](https://doi.org/10.5281/zenodo.22696825)

Code and data for the LAFT paper 1: translating the Kessler warm-rain
microphysics scheme from Fortran to JAX with four large language models, and
validating every translation against the Fortran reference with a
model-agnostic pipeline.

The repository has two parts:

| Folder | What it is | Start with |
|---|---|---|
| [`LAFT/`](LAFT/) | The framework: Fortran parsing, bridge generation, multi-pass prompt authoring, lint, semantic audit, runtime validation, driver comparison, and GPU profiling. Project-agnostic; every project-specific value comes from a `config/project.toml`. | [`LAFT/docs/README.md`](LAFT/docs/README.md) for the overview, [`LAFT/ORCHESTRATOR.md`](LAFT/ORCHESTRATOR.md) to run it |
| [`kessler/`](kessler/) | The paper-1 project: the Kessler Fortran reference and data, the pipeline snapshot, the bridge tests, the four LLM translations with their full validation records, and the cross-model comparison outputs. | [`kessler/README.md`](kessler/README.md) |

## The four translations

| Archive | Model | Fortran comparison |
|---|---|---|
| `kessler/translations/claude-sonnet46-jd/` | Claude Sonnet 4.6 | all pass, machine precision |
| `kessler/translations/gemini31pro-jd/` | Gemini 3.1 Pro | all pass, machine precision |
| `kessler/translations/gpt54thinking-jd/` | GPT-5.4 Thinking | all pass, machine precision |
| `kessler/translations/qwen25-32b-jd/` | Qwen2.5-32B (Coder-Instruct, via vLLM) | all pass, machine precision |

Each archive holds the JAX code, the prompts, and every gate's output (lint,
semantic audit, runtime validation, bridge test suite, driver run, and the
comparison against the Fortran outputs). See `kessler/README.md` for how the
archives were produced and how to read their provenance notes.

## Setup

Two conda environments, built from the pinned recipes in `LAFT/envs/`:

```bash
conda env create -f LAFT/envs/ts.yml            # tree-sitter: Stage 0 parsing only
conda env create -f LAFT/envs/jax-validate.yml  # JAX 0.6.2: everything else
```

`jax-validate` installs the CUDA 12 build of JAX for NVIDIA GPUs. For a
CPU-only machine change `jax[cuda12]==0.6.2` to `jax==0.6.2` in the recipe;
the Fortran comparisons still pass, only the GPU timings cannot be reproduced.
Details, including the optional vLLM environment for locally hosted models,
are in [`LAFT/docs/README.md` §Environments](LAFT/docs/README.md#environments).

## Running the pipeline

**The pipeline is driven by an AI coding agent, not by a human at a shell.**
`ORCHESTRATOR.md` and the per-stage workflow documents are written as
instructions to an agent such as Claude Code, the Gemini CLI, or OpenAI Codex:
the agent reads the orchestrator, runs the phase scripts and submits the PBS
jobs in order, authors or repairs the JAX translation from the generated
prompts, reads each gate's output, and stops for human approval between
stages. A person supervises and approves; the agent executes.

```bash
cd kessler          # the project root: config/project.toml lives here
claude              # or your agent of choice, started in this directory
# then ask it: "Follow ORCHESTRATOR.md and run the pipeline for this project."
```

The individual phase scripts are ordinary Python and can also be run by hand
in the environments above (Stage 0 in `ts`, everything after in
`jax-validate`), but the workflow documents assume an agent is the reader.

The batch jobs in `LAFT/pbsJobs/` are written for NCAR's Derecho and Casper
systems (PBS Pro, NVIDIA A100). `LAFT/pbsJobs/README.md` lists what to change
on another cluster.

## Reproducing the paper's numbers

- **Accuracy** (`kessler/_compare_results/outputs/reports/performance_comparison_2026-04-08.md`,
  `plots/numerical_accuracy.png`): rerun any archive's driver and comparison
  through the pipeline; the mean absolute error in `theta` is about 7e-16 K
  for all four models.
- **The Qwen translation itself** (`kessler/translations/qwen25-32b-jd/`):
  `LAFT/pbsJobs/pbs_qwen25_translate.sh` re-runs Qwen2.5-Coder-32B-Instruct
  through vLLM with the paper's settings on the same prompts (see
  `LAFT/workflow_translator/qwen/qwen.md` §Paper-1 job). Sampling at
  temperature 0.1 is not bit-reproducible, so expect the configuration, not
  the archived bytes. The Claude, GPT, and Gemini translations were authored
  in-context by their coding agents and have no batch job.
- **GPU scalability and profiling** (`kessler/_compare_results/outputs/plots/`,
  `reports/*_nsys_report_*.md`): produced on A100 GPUs; the reference data
  and the JAX code are in the repository, the timings depend on the hardware.
  The Fortran-vs-JAX scalability figure is regenerated with the two
  `compare_scalability_*.sh` jobs in `LAFT/pbsJobs/` (serial Fortran on one
  CPU core, the four translations on one GPU) followed by
  `kessler/_compare_results/plot_Fortran_vs_LLMs_scalability.py`; the
  paper's timing JSONs are archived under `kessler/_compare_results/outputs/scalability/`.
  The Nsight Systems reports come from `compare_profiling_LLMs.sh` in `nsys`
  mode, which runs `kessler/_compare_results/LLMx_jd_profiling_run.py`.
- **The OpenACC GPU reference** (`kessler/data/exp2_jd/acc_src/`): the
  manually accelerated Fortran used in the paper's scalability comparison,
  with its driver, Makefile, and profiling runner, so its timings can be
  reproduced and compared with any JAX translation. See its `README.md`.

## Citation

If you use this code or data, please cite the paper and the repository. The
machine-readable metadata is in [`CITATION.cff`](CITATION.cff) (GitHub's
"Cite this repository" button reads it).

> Gagne, D. J., Linck, I., Dennis, J., Schreck, J., & Stengel, K. (2026).
> Evaluating LLMs for Translating Legacy Fortran Atmospheric Physics to
> GPU-Accelerated JAX. Preprint submitted to arXiv; manuscript submitted to
> the Journal of Advances in Modeling Earth Systems (JAMES). Code and data:
> https://doi.org/10.5281/zenodo.22696825

```bibtex
@article{gagne2026laft,
  title   = {Evaluating {LLMs} for Translating Legacy {Fortran} Atmospheric
             Physics to {GPU}-Accelerated {JAX}},
  author  = {Gagne, David J. and Linck, Iris and Dennis, John and
             Schreck, John and Stengel, Karen},
  year    = {2026},
  note    = {Preprint submitted to arXiv; manuscript submitted to JAMES},
  url     = {https://github.com/NCAR/LAFT},
  doi     = {10.5281/zenodo.22696825}
}
```

The Zenodo DOI above is the concept DOI and always resolves to the latest
release. The arXiv identifier and journal DOI will be added here and in
`CITATION.cff` once assigned.

## License

Apache License 2.0, see [`LICENSE`](LICENSE).
