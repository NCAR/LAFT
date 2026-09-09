# LAFT — LLM-Assisted Fortran Translation framework

Canonical home of the framework code and jobs. Project-specific material
(Fortran sources, `data/`, `out/`, `translations/`, reference trees, reports)
stays in a per-project directory that sits beside `LAFT/`; in this repository
that project is `../kessler/`, the Kessler microphysics experiment of paper 1.

**Start at [`ORCHESTRATOR.md`](ORCHESTRATOR.md)** — it is the pipeline entry
point: stage order, entry gates, and the mandatory approval pause between
stages. It is written **for an AI coding agent** (Claude Code, Gemini CLI, or
OpenAI Codex): the agent, started in a project root, reads it and drives the
pipeline, with a human approving each stage transition. It is a **single
shared document, symlinked into each project root**
(`kessler/ORCHESTRATOR.md` → this file): `cd`
into the project you want to translate and run it from there — the project is
selected entirely by the working directory's `config/project.toml`, never named
in the orchestrator. See its
[*One shared orchestrator, run per project*](ORCHESTRATOR.md#one-shared-orchestrator-run-per-project)
section for details. For a framework overview and the full documentation map,
see [`docs/README.md`](docs/README.md).

**Setup:** two conda environments, built from the pinned recipes in
[`envs/`](envs/) — `ts` (tree-sitter, Stage 0 parsing only) and
`jax-validate` (JAX 0.6.2, everything else):

```bash
conda env create -f LAFT/envs/ts.yml
conda env create -f LAFT/envs/jax-validate.yml
```

See [`docs/README.md` §Environments](docs/README.md#environments) for what
each stage needs, the CPU-only variant, and the optional vLLM environment for
locally hosted models.

## Contents

| Folder | Role |
|---|---|
| `tools/` | Shared housekeeping: the `clean_*/copy_*` scripts (archive/clean per-LLM `out/` artifacts) |
| `workflow_frontend/` | Stage 0 — the base representation: phases 01–02 (`FRONTEND_WORKFLOW.md`, `phase01_01_ts_parse.py`, `phase02_01..04`) parse the Fortran and build the packets/index every workflow reads |
| `workflow_bridge/` | Bridge workflow (`BRIDGE_WORKFLOW.md`, `phase03_make_bridge.py`, `run_bridge_tests.py` gate) — generates the Fortran↔JAX bridge and verifies it before translation |
| `workflow_translator/` | Authoring workflow — the 4-pass translation (`TRANSLATE_WORKFLOW.md`, state machine, `phase04_01_make_prompts.py`, `phase04_02_make_wrappers.py`, `phase04_04_completeness_check.py` (Step 2.5 scaffold stop), `phase05_01b_semantic_audit.py` + `audit_gate.py` (the Step 3.5 gate runtime validation cannot bypass), `prompt_policies/`, and one self-contained folder per external translator — `qwen/` holds that model's module doc + vLLM drivers) |
| `validation/` | Phase-05 validators **shared by translator + profiler** (`VALIDATION_MANUAL.md`, `phase05_01_lint_translation.py`, `phase05_02_runtime_validate.py`) |
| `workflow_profiler/` | Profiling workflow (`PROFILE_WORKFLOW.md`, `PROFILER_MANUAL.md`, diagnostics, HLO/trace parsers, tests) |
| `pbsJobs/` | PBS job scripts: driver (CPU/GPU), runtime validation, value comparison, nsys/trace profiling, and the paper-1 Qwen2.5 translation via local vLLM. Canonical copies live here; account, queue, and conda env are set directly in each script's `#PBS` headers (not templated from `project.toml`), so a new project edits the headers after copying |
| `envs/` | Pinned conda recipes: `ts.yml` (tree-sitter frontend) and `jax-validate.yml` (everything else); see `docs/README.md` §Environments |
| `config/` | Everything config: `framework_config.py` (the loader every workflow imports), `project.template.toml` (blank per-project config). Each project keeps its own real `config/project.toml`; the worked instance is `../kessler/config/project.toml` |
| `docs/` | Framework docs: `README.md` (overview + doc map, incl. generic-vs-project-specific), `architecture.md` (mermaid), `NEW_PROJECT_CHECKLIST.md`. Per-stage references live in the stage folders |

## Starting a new project

Copy `config/project.template.toml` into `<your-project>/config/project.toml`
and follow `docs/NEW_PROJECT_CHECKLIST.md`. Pipeline tools are run from the
project root, e.g. `python workflow_frontend/phase01_01_ts_parse.py` (in the
`ts` env; every later stage runs in `jax-validate`).
