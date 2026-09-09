# LAFT Orchestrator — run the Fortran→JAX pipeline for this project

You are orchestrating the LAFT pipeline for one project. The pipeline is four
stages, each with its own authoritative playbook. **This document is the entry
point**: it gives the stage order, the hand-off gate between each, and how to
run the **whole pipeline** or **just one stage**. It only *sequences* the
stages — always follow each stage's own playbook for the actual steps, resume
handling, and abort gates. Codebase-agnostic: every project value comes from
`config/project.toml`.

**Working directory: the project root** (the directory containing `config/`).
Run everything from here.

## One shared orchestrator, run per project

There is exactly **one** orchestrator document — this file, `LAFT/ORCHESTRATOR.md`.
Each project does **not** get its own copy. Instead every project root holds a
symlink that points back here:

```
kessler/ORCHESTRATOR.md → ../LAFT/ORCHESTRATOR.md
```

So `kessler/ORCHESTRATOR.md` and `LAFT/ORCHESTRATOR.md` are the *same file*.
Edit it once in `LAFT/` and every project sees the change — there is nothing to
copy or keep in sync. (This requires each project to stay a sibling of
`LAFT/`, so the relative `../LAFT/` link resolves.)

**How to run it:** `cd` into the project folder you want to translate — the one
containing that project's `config/` — and follow `./ORCHESTRATOR.md` from there.
You are reading the shared framework document, but working inside one project.

**How the orchestrator knows which project it is translating:** it does *not*
name a project anywhere. This document is codebase-agnostic; the project's
identity comes entirely from the **working directory you launched from**. Every
project-specific value — Fortran sources, output paths, the target LLM — is read
from `config/project.toml` in that current directory, and every path below
(`out/`, `tools/`, `translations/`) is relative to it. Run from
`kessler/` and you translate Kessler; run from another project directory and
you translate that scheme. Same instructions, different `config/project.toml`.

## Approval gate between every stage (MANDATORY)

**Never advance from one stage to the next on your own — not even when the user
asked to "run the whole pipeline."** Each stage is a checkpoint. After you
finish a stage:

1. Confirm the stage's gate is green (see the stage table).
2. **Report to the user**: what ran, the gate result (e.g. bridge gate `PASS`,
   comparison `ALL PASS`), and what the *next* stage will do.
3. **Stop and wait for the user's explicit go-ahead.** Do not start the next
   stage until they approve.

"Run the whole pipeline" means run the stages *in order with these approval
pauses between them* — it is **not** authorization to run all four
back-to-back. If a stage's gate is **not** green, stop and report regardless;
never paper over a failed gate to keep moving. (This is the cross-stage gate;
each stage's own playbook still governs pauses *within* that stage, e.g. PBS
jobs.)

## The four stages

| # | Stage | Playbook | Does | Green when |
|---|-------|----------|------|-----------|
| 0 | **Frontend** | `workflow_frontend/FRONTEND_WORKFLOW.md` | parse Fortran → packets/index every stage reads | `out/phase1_index.json` + `out/packets/*_merged.json` exist |
| 1 | **Bridge** | `workflow_bridge/BRIDGE_WORKFLOW.md` | generate + gate the Fortran↔JAX bridge | `out/reports/bridge/bridge_test_results.json` → `"status": "PASS"` |
| 2 | **Translate** | `workflow_translator/TRANSLATE_WORKFLOW.md` | author (4 passes) → **completeness check (scaffold = STOP, human decision)** → validate (lint→**semantic audit (gated)**→runtime→driver→comparison) | comparison `ALL PASS` **in every driver mode declared by `[driver].contracts`** (TRANSLATE_WORKFLOW §5.0), `out/reports/translation/translation_report.md` written **with both statistics tables** (below) |
| 3 | **Profile** *(optional)* | `workflow_profiler/PROFILE_WORKFLOW.md` | GPU-efficiency profile→diagnose→fix loop | diagnosis `production_ready` |

A stage hands off to the next only when its gate is green. Stage 3 is optional:
run it when you want GPU efficiency; skip it if numerically-correct JAX is enough.

## Step 0 — Decide what to run

Pick the scenario that matches, then run only the stages it lists:

| Scenario | Run |
|----------|-----|
| **Brand-new project** (no `out/packets/`) | 0 → 1 → 2 → (3) |
| **New LLM, existing project** (packets + bridge already built) | archive+clean (§Switching LLMs) → 2 → (3). Frontend + bridge are LLM-independent — do **not** re-run them. |
| **Fortran source changed** | 0 → 1 → 2 → (3) — packets are stale, rebuild from source |
| **Generator changed** (phase02/03/04 logic) | re-run from the earliest changed stage forward |
| **Re-run one stage** | jump to it after confirming its entry gate (below) |
| **Just profile a finished translation** | 3 only (translator must be complete) |

## Run the full pipeline

Run the stages in order, **pausing for user approval between each** (see
"Approval gate between every stage" above). Do not start a stage until the
previous stage's gate is green **and** the user has approved continuing.

1. **Frontend** — follow `workflow_frontend/FRONTEND_WORKFLOW.md`.
   → gate green? report + **get approval** → then:
2. **Bridge** — follow `workflow_bridge/BRIDGE_WORKFLOW.md`.
   → gate green? report + **get approval** → then:
3. **Translate** — follow `workflow_translator/TRANSLATE_WORKFLOW.md`. (It
   regenerates the phase-04 prompts/wrappers, translates, then runs the
   phase-05 validation gate; it also re-runs the bridge suite against the real
   translation at its Step 4.5.)
   → gate green? report + **get approval** → then:
4. **Profile** *(optional)* — follow `workflow_profiler/PROFILE_WORKFLOW.md`.
   → then, and only then, **archive** (below).

A failed gate at any stage ends the run — stop and report; do not continue.
Inside Stage 2, a **scaffolded procedure** (TRANSLATE_WORKFLOW §2.5,
`status = aborted_scaffold`) is such a stop: it is a model/context limit, not a
code defect — never re-run the same prompt automatically; report the evidence
and wait for the user's decision (split / change model or prompt policy /
archive as a failure experiment).

## Stage 2 must produce two statistics tables

The translation report is not complete without them, and one of them can only
be captured at a moment that has already passed by the time the report is
written. Enforced in `TRANSLATE_WORKFLOW.md` §2e, §6 and §8:

| | Table 1 — four-pass authorship | Table 2 — validation fixes |
|---|---|---|
| captured | **once**, §2e, after every procedure is authored and **before** any fix | appended per fix attempt, §6 |
| source | `out/reports/translation/pass_stats.json` (+ `.sha256`) | `out/reports/translation/fix_stats.jsonl` |
| mutability | **IMMUTABLE** — `freeze` refuses to overwrite; `render` verifies the digest and prints an integrity warning if it changed | cumulative, re-rendered after every fix |

Both carry the same columns — procedure (grouped under its module, with
subtotals), passes, Fortran lines, JAX lines, sizes, tokens, wall-clock.

**Why Table 1 must be frozen before validation:** fixing edits
`out/jax/{proc}.py` in place. Once that happens the untouched four-pass output
no longer exists on disk and its line counts cannot be recovered. A late freeze
does not fail loudly — it quietly records repaired code as if the translator had
produced it, which is precisely the measurement the table exists to make. So
Table 1 and Table 2 *should* disagree; that disagreement is the finding.

Modules are grouping headers only. Nothing is generated for a module, so a
module has no line count, token count or duration of its own — only the sum of
its procedures'.

## Archiving: always the LAST step

`tools/copy_AI_results.sh <llm>` snapshots `out/` into `translations/<llm>/`.
**Run it only after every stage you intend to run has succeeded** — in
practice, after Profile if you are profiling, after Translate if you are not.

This ordering matters because the copy script archives `out/profiled/` **only
"if present"**. Archiving before the profiler runs is therefore silent: it
succeeds, produces an archive with no profiler artifacts, and reports nothing
wrong. `tools/clean_AI_results.sh` then verifies "everything is archived" —
which is true of what exists at that moment — and clears the iteration
directories. The profile results for that LLM are then gone, and the only way
back is to re-stage and re-profile.

So:

| If you are... | Archive when |
|---|---|
| translating only | comparison `ALL PASS` + translation report written |
| translating and profiling | diagnosis `production_ready` + profile report written |
| profiling an already-archived translation | re-run the copy afterwards so `translations/<llm>/profiled/` is populated |

Only archive early if you deliberately want a translation-only snapshot, and
say so — then re-run the copy after profiling to pick up `out/profiled/`.

## Run a single stage

Follow only that stage's playbook, after confirming its **entry gate**:

| Stage | Entry gate (must hold before you start) |
|-------|------------------------------------------|
| Frontend | `config/project.toml` exists with `[source].fortran_files` |
| Bridge | frontend outputs present (packets + `phase1_index.json`) |
| Translate | bridge gate is `PASS` (`out/reports/bridge/bridge_test_results.json`); if `[driver].contracts` declares contract 2, the project driver implements the `LAFT_DRIVER_MODE` switch (TRANSLATE_WORKFLOW §5.0 fail-fast) |
| Profile | translator complete (validated `.validated.py` snapshot, comparison green) |

If an entry gate is not met, run the earlier stage(s) first.

## Switching LLMs (re-translation)

The frontend packets and the bridge are **LLM-independent** — built once,
reused by every translator. To translate the same project with a different LLM:

1. Archive the current LLM's results: `tools/copy_AI_results.sh <llm>`.
   **Only if that LLM's run is finished** — including its profile, if you
   profiled it (see §Archiving: always the LAST step). Archiving mid-run
   snapshots an incomplete `out/`, and step 2's clean then makes it permanent.
2. Reset `out/` for the next run: `tools/clean_AI_results.sh` (preserves the
   frontend packets, the bridge + its reports, and the hand-authored
   driver/comparison scripts — it verifies everything is archived first).
3. Set `[translator]` in `config/project.toml` for the new LLM (or
   `source = "in_context"`).
4. Run **stage 2** (Translate) — its Step 1.0 regenerates the prompts that
   clean removed — then optionally **stage 3**.

Do **not** re-run frontend or bridge unless the Fortran source or a generator
changed.

## At a glance

Each `⏸` is a mandatory stop for the user's approval before the next stage:

```
config/project.toml
      │
      ▼
[0] FRONTEND ─packets─⏸─► [1] BRIDGE ─PASS─⏸─► [2] TRANSLATE ─ALL PASS─⏸─► [3] PROFILE (opt)
 parse/extract            gen + gate           author + validate           GPU efficiency
 FRONTEND_WORKFLOW.md     BRIDGE_WORKFLOW.md   TRANSLATE_WORKFLOW.md        PROFILE_WORKFLOW.md
                                                       │                          │
                                                       └──── ARCHIVE LAST ────────┘
                                                    tools/copy_AI_results.sh <llm>
                                              (after 2 if not profiling, after 3 if profiling)
```

**Pointers:** each stage's playbook is authoritative for its own steps.
Shared validators: `validation/VALIDATION_MANUAL.md`. Profiler reference
manual: `workflow_profiler/PROFILER_MANUAL.md`. Config schema:
`config/project.toml` (+ `LAFT/config/project.template.toml`).
