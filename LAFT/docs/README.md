# LAFT — LLM-Assisted Fortran Translation Framework

**Purpose:** translate legacy Fortran scientific code into a modern target
language (JAX today, extensible beyond) using LLMs, then verify the result
numerically against the Fortran and make it hardware-efficient on GPU.

**Target platform:** NCAR Derecho (NVIDIA A100).

> **This file is the documentation map and framework overview.**
> To *run* the pipeline, start at [`../ORCHESTRATOR.md`](../ORCHESTRATOR.md) —
> it is the entry point and owns stage sequencing. Each stage's own playbook is
> authoritative for that stage's steps. The orchestrator and the playbooks are
> instructions to an **AI coding agent** (Claude Code, Gemini CLI, or OpenAI
> Codex) that executes the pipeline under human approval; they are not meant
> to be followed by hand.

The framework is **project-agnostic** and **LLM-agnostic**: every
project-specific value (source files, driver, comparison schema, PBS jobs,
translation author) lives in `config/project.toml`, loaded by
`config/framework_config.py`. The case study in this repository is the
Kessler microphysics scheme (`../kessler/`). Below, `{proc}` marks
per-procedure artifacts.

---

## Contents

1. [What the pipeline does](#what-the-pipeline-does)
2. [The four stages](#the-four-stages)
3. [Environments](#environments)
4. [Framework layout](#framework-layout)
5. [Project layout](#project-layout)
6. [Generated artifacts](#generated-artifacts)
7. [Multi-LLM translations](#multi-llm-translations)
8. [Testing strategy](#testing-strategy)
9. [Generic vs project-specific](#generic-vs-project-specific)
10. [Documentation map](#documentation-map)

---

## What the pipeline does

```
Fortran Code (CPU)  →  [LAFT pipeline]  →  Target-language Code (GPU)
   ↓                                              ↓
Column-major                                 Row-major
DO loops                                     lax.while_loop / vmap
INTENT(INOUT)                                Pure functions
```

Key properties:

- **Automated translation** — Fortran → target language via LLM, either
  in-context or a locally-hosted model on PBS, selected by `[translator]` in
  `config/project.toml`.
- **Bridge layer** — handles Fortran ↔ target-language memory layout
  conversion on the device (PATH C, batched D2H), is gated by its own test
  suite *before* translation starts, and exposes two calling contracts per
  array procedure: host-facing `{proc}_bridge` and device-resident
  `{proc}_bridge_device` (choice per project, recorded in
  `[driver].contracts`; see `NEW_PROJECT_CHECKLIST.md` §6).
- **Multi-pass authoring** — a 4-pass prompt chain rather than a single prompt.
- **Numeric verification** — completeness (a scaffolded procedure is a
  hard stop, no automatic retry) → lint → semantic audit (fidelity to the
  Fortran; gated — runtime validation cannot be submitted until it is clean)
  → runtime → driver → comparison against the Fortran reference, each step
  blocking the next.
- **Hardware profiling** — an optional profile → diagnose → fix → re-validate
  loop that runs until the diagnoser reports `production_ready`.
- **Multi-LLM** — independent, comparable translation runs per LLM.

## The four stages

Sequencing, entry gates, hand-off rules, and the mandatory user-approval pause
between stages are all defined in [`../ORCHESTRATOR.md`](../ORCHESTRATOR.md).
Summary only:

| # | Stage | Playbook | Reference |
|---|-------|----------|-----------|
| 0 | Frontend | `workflow_frontend/FRONTEND_WORKFLOW.md` | `workflow_frontend/FRONTEND_REFERENCE.md` |
| 1 | Bridge | `workflow_bridge/BRIDGE_WORKFLOW.md` | `workflow_bridge/BRIDGE_REFERENCE.md` |
| 2 | Translate | `workflow_translator/TRANSLATE_WORKFLOW.md` | `workflow_translator/TRANSLATE_REFERENCE.md` |
| 3 | Profile *(optional)* | `workflow_profiler/PROFILE_WORKFLOW.md` | `workflow_profiler/PROFILER_MANUAL.md` |

Phase-05 validation is **shared** by stages 2 and 3 — it lives in
`validation/`, documented by `validation/VALIDATION_MANUAL.md`.

## Environments

The pipeline runs in two conda environments, plus an optional third for
locally hosted open-weight models. They are **built from the pinned files in
`envs/`**, not downloaded: a conda environment is not portable (it bakes in
its absolute prefix and CUDA-linked binaries built for one machine), while the
recipe is a few lines and reproduces the exact package set anywhere.

| Environment | File | Used by | Why it is separate |
|---|---|---|---|
| `ts` | `envs/ts.yml` | Stage 0 only: `phase01_01_ts_parse.py`, `phase02_02_deps_JSON.py` | tree-sitter's Fortran grammar needs the pinned pair `tree-sitter 0.20.4` + `tree-sitter-languages 1.10.2`, which conflicts with current tree-sitter releases |
| `jax-validate` | `envs/jax-validate.yml` | Stages 1-3: bridge generation and tests, lint, runtime validation, driver, comparison, profiler; every `jax_*` PBS job | `jax 0.6.2` fixes the numerics every archived comparison was produced with and is part of the XLA cache key |
| `qwen-vllm` | `envs/qwen-vllm.yml` | `pbsJobs/pbs_qwen25_translate.sh` only — re-running the paper's Qwen2.5-Coder-32B translation through vLLM | vLLM 0.15.1 / Python 3.11, the environment of the April 2026 run; pulls its own torch and CUDA wheels (several GB); not needed for the in-context agents |

Build the first two on any machine with conda:

```bash
conda env create -f LAFT/envs/ts.yml
conda env create -f LAFT/envs/jax-validate.yml
```

`jax-validate` installs the CUDA 12 build of JAX for NVIDIA GPUs (the paper's
runs used A100s). On a CPU-only machine edit the file to `jax==0.6.2`; the
Fortran comparisons still pass at machine precision, only the GPU timings in
`kessler/_compare_results/` cannot be reproduced. Create the vLLM environment
(`conda env create -f LAFT/envs/qwen-vllm.yml`) on a GPU compute node — on
NCAR systems through a batch job — because the pip download is large and the
environment must be created on a node that matches the GPUs it will run on.

Which environment a step needs is stated in each stage's workflow document and
in the `conda activate` line of its PBS job. The remaining Python tools
(`phase02_01`, `phase02_03`, `phase02_04`, the prompt generator, lint, the
completeness check and the semantic audit) run in either environment.

## Framework layout

`LAFT/` holds framework code only; no project data. See
[`../README.md`](../README.md) for the per-folder role table.

```
LAFT/
├── ORCHESTRATOR.md         # pipeline entry point (stage order + approval gates)
├── config/                 # framework_config.py loader, project.template.toml
├── workflow_frontend/      # stage 0 — phases 01–02: parse → packets + index
├── workflow_bridge/        # stage 1 — phase 03: bridge generation + test gate
├── workflow_translator/    # stage 2 — phase 04 authoring, prompt_policies/,
│                           #   semantic audit, per-external-translator folders
├── validation/             # phase 05 validators shared by stages 2 and 3
├── workflow_profiler/      # stage 3 — profile → diagnose → fix loop
├── tools/                  # archive/clean housekeeping scripts
├── pbsJobs/                # canonical PBS job scripts (edit #PBS headers per project)
└── docs/                   # this file + architecture, checklist, bridge evolution,
                            #   contract lifecycle, context windows
```

Each stage folder holds a `*_WORKFLOW.md` (the procedural playbook an agent
follows in-context) and at most one companion reference/manual doc.

## Project layout

A project directory instantiates the framework. Generic shape:

```
<project>/
├── config/project.toml     # THE project instance definition
├── data/
│   ├── src/                # Fortran source ([source].fortran_files)
│   └── fortran_io/         # reference Fortran I/O data
├── <reference>/            # reference Fortran driver tree ([reference].dir):
│                           #   the column-model driver the target driver mirrors,
│                           #   lookup tables, captured ground-truth output
├── out/                    # all generated pipeline output (see below)
├── translations/           # per-LLM result copies
├── _compare_results/       # cross-LLM comparison (per-project material)
├── bridge_test/            # pytest bridge suite
├── reports/                # dated analysis & campaign reports
└── docs/, memory/          # project-specific notes
```

The four hand-made inputs a new project must supply (Fortran source, reference
data, the target-language driver, the comparison script) are enumerated in
[`NEW_PROJECT_CHECKLIST.md`](NEW_PROJECT_CHECKLIST.md).

## Generated artifacts

Everything the pipeline writes lands under `out/`:

```
out/
├── packets/                # phase 01–02 metadata: {proc}_merged.json, _ALL_deps.json
├── phase1_index.json       # index of all parsed items
├── procedures|modules|programs|units/   # phase 01 extractions
├── module_dependencies.json
├── bridge/{proc}_bridge.py # bridge layer (layout conversion, GPU-copy optimized)
├── prompts/{proc}_passN.md # generated multi-pass prompts (NEVER edit — see below)
├── wrappers/               # reference wrapper skeletons; never written to out/jax/
├── jax/                    # active translation: {proc}.py, {proc}_passN.py,
│                           #   {proc}.validated.py snapshots, workflow_state.json
├── lint/{proc}_lint.json   # lint results + _summary.json
├── validation/{proc}_runtime.json
├── issues/                 # fix_log.md, semantic_audit/*.json + _summary.json (gate input),
│                           #   semantic_audit_waivers.json (justified exceptions)
├── driver/                 # hand-ported driver ([driver].script) + run outputs
├── profiled/               # profiler working copy, profile_state.json, iteration_N/
└── reports/
    ├── bridge/             #   {proc}_analysis.txt, bridge_test_results.json, bridge_report.md,
    │                       #   direction_symmetry.json
    ├── translation/        #   translation_report.md, pass_stats.json (frozen Table 1),
    │                       #   fix_stats.jsonl (Table 2), completeness_check.json + completeness_report.md,
    │                       #   semantic_audit_report.md
    └── profile/            #   profile_report.md
```

**Prompts are generated, not authored.** Edit the policy files in
`workflow_translator/prompt_policies/` and regenerate — never edit
`out/prompts/*.md` directly. See
[`../workflow_translator/TRANSLATE_REFERENCE.md`](../workflow_translator/TRANSLATE_REFERENCE.md).

### Translated code shape

Array procedures get a two-layer structure:

```python
def {proc}_core(...):
    """Pure computation (JIT-compatible)"""
    # returns: updated arrays + error flag

def {proc}(...):
    """Wrapper (handles error messages)"""
    # calls core; returns updated arrays + error message + flag
```

Scalar procedures get no `_core` split: plain Python if `jax_required=false`,
`jnp`-safe scalar code (no own `@jax.jit`) if `jax_required=true`.

## Multi-LLM translations

The frontend packets and the bridge are **LLM-independent** — built once,
reused by every translator. Archiving a finished run and resetting `out/` for
the next LLM is covered in ORCHESTRATOR.md §Switching LLMs
(`tools/copy_AI_results.sh` → `tools/clean_AI_results.sh`).

Each LLM gets its own directory under `translations/` (driven by
`[llm].copy_targets`), all with the same shape:

```
translations/<llm>/
├── jax/                 # {proc}.py, {proc}_passN.py intermediates, {proc}.validated.py,
│                        #   workflow_state.json
├── prompts/             # prompts used for this run
├── lint/results/        # lint results
├── validation/results/  # runtime validation results
├── issues/              # fix_log.md + semantic_audit/
├── driver/results/      # run outputs compared against the Fortran reference
├── profiled/            # profiler stage artifacts, when stage 3 was run
└── reports/             # translation/, bridge/, profile/ (+ analysis/generation_log.jsonl
                         #   for the vLLM translators)
```

Keep the `{proc}_passN.py` intermediates: they are what makes a design
decision attributable to a pass. Per-LLM comparison outputs live in the
project's `_compare_results/` (per-project material, not framework code).

**Agentic in-context translators can read the workspace.** A CLI agent
(Gemini CLI, Codex, Claude Code) running Stage 2 has read access to every
earlier archive under `translations/`; nothing in the workflow forbids it
from reusing one. For an independent-translation claim, run such agents in
a checkout without `translations/` visible (or move the archives out for
the run). The vLLM path is immune — the model only ever sees the prompt.

## Testing strategy

Four levels, increasing in strength:

| Level | Tool | Checks |
|---|---|---|
| 0 — static (login node, no JAX) | `workflow_translator/phase04_04_completeness_check.py`, `phase05_01b_semantic_audit.py`, `audit_gate.py` | translated-not-summarised (scaffold phrases, dead/absent callees, silent stubs ⇒ STOP); fidelity to the Fortran (calls invoked, update terms, table-read base, LUT columns, zero-forever, direction masks …); the gate that runtime validation requires |
| 1 — smoke | `validation/phase05_02_runtime_validate.py` | imports, wrapper callable, core JIT-compiles, outputs finite, shapes match |
| 2 — unit | `bridge_test/` (pytest) | layout conversion, bridge vs direct call, contract-2 ≡ contract-1 bit-identity, vertical symmetry of direction-parameterised loops, physics sanity ranges |
| 3 — accuracy | the project's `[comparison].script` | numeric agreement with the Fortran reference; per-variable stats + top-level PASS/FAIL |

Level 3 is the authoritative gate. Its schema — variables, phenology gates,
tolerance rules — is per-project, defined in `[comparison]` in
`config/project.toml`; the framework owns only the PASS/FAIL contract. Details
in [`../validation/VALIDATION_MANUAL.md`](../validation/VALIDATION_MANUAL.md).

## Generic vs project-specific

The LAFT pipeline is project-agnostic: all project-specific values and
logic live in `config/project.toml`, loaded by `config/framework_config.py`
(template: `config/project.template.toml`; worked instance:
`../kessler/config/project.toml`).

Per project you supply: the Fortran source (one file or several —
`[source].fortran_files`), a reference driver tree with a captured reference
output, a hand-ported JAX driver that reads `LAFT_DRIVER_MODE`, the
`[comparison]` output schema, the per-project `bridge_test/` suite with its
`TESTING_GUIDE`, and optionally a project prompt-policy file and the
semantic-audit configuration (`[semantic_audit]`: `lut_indices` /
`level_conventions`, `prognostic_vars`, `[[table_readers]]`). Everything else — parsing, dependency analysis, bridge
generation (both contracts), prompt assembly, lint, semantic audit, runtime
validation, the comparison engine and the profiler — transfers unchanged.
Instantiation steps: `NEW_PROJECT_CHECKLIST.md`.

## Documentation map

**Entry point**

| Doc | Covers |
|---|---|
| [`../ORCHESTRATOR.md`](../ORCHESTRATOR.md) | Stage order, entry gates, approval pauses, switching LLMs |
| [`../README.md`](../README.md) | What lives in each LAFT folder; provenance |
| This file | Framework overview, layouts, artifacts, doc map |

**Per stage** — one playbook + at most one reference doc each

| Stage | Playbook (procedure) | Reference (deep-dive) |
|---|---|---|
| Frontend | `../workflow_frontend/FRONTEND_WORKFLOW.md` | `../workflow_frontend/FRONTEND_REFERENCE.md` |
| Bridge | `../workflow_bridge/BRIDGE_WORKFLOW.md` | `../workflow_bridge/BRIDGE_REFERENCE.md` |
| Translate | `../workflow_translator/TRANSLATE_WORKFLOW.md` | `../workflow_translator/TRANSLATE_REFERENCE.md` |
| Validation *(shared)* | — | `../validation/VALIDATION_MANUAL.md` |
| Profile | `../workflow_profiler/PROFILE_WORKFLOW.md` | `../workflow_profiler/PROFILER_MANUAL.md` (+ `headroom_patterns.md`, Step-7 signature → hypothesis table) |

**Framework-level**

| Doc | Covers |
|---|---|
| [`architecture.md`](architecture.md) | Stage architecture and multi-LLM fan-out (mermaid diagrams) |
| [`NEW_PROJECT_CHECKLIST.md`](NEW_PROJECT_CHECKLIST.md) | Instantiating LAFT for a new Fortran code |

Change history lives in git (dated docs above record decisions; there is no
separate change log).

---

**Framework:** LAFT — LLM-Assisted Fortran Translation Framework
**Case study:** Kessler microphysics (`../kessler/`)
**Platform:** NCAR Derecho (NVIDIA A100 GPUs)
