# Translator Reference — Prompt Engineering

> **Where this fits:** stage 2 of the pipeline. `../ORCHESTRATOR.md` owns stage
> sequencing; `TRANSLATE_WORKFLOW.md` is the authoritative procedure to follow.
> This document is the deep-dive reference for *why* the prompt chain is shaped
> the way it is, and how to change it.

**Last Updated:** July 2026 (pass-count classes + external translators; examples
from the Kessler case study still illustrate the mechanics)

---

## Table of Contents

1. [The Problem: Why a Single Prompt Fails](#1-the-problem-why-a-single-prompt-fails)
2. [The Solution: Multi-Pass Iterative Refinement](#2-the-solution-multi-pass-iterative-refinement)
3. [Policy File System](#3-policy-file-system)
4. [The 4-Pass Design](#4-the-4-pass-design)
5. [File Connections — Inputs and Outputs](#5-file-connections--inputs-and-outputs)
6. [How to Run](#6-how-to-run)
7. [Adding a New Physics Scheme](#7-adding-a-new-physics-scheme)
8. [Design Decisions and Trade-offs](#8-design-decisions-and-trade-offs)

---

## 1. The Problem: Why a Single Prompt Fails

### Context window ≠ attention

A large language model can technically receive a 100k-token prompt. But **receiving** and **following** are different things. Research and empirical testing consistently show the "lost in the middle" effect: rules and examples buried in the middle of a long prompt receive significantly less attention than those near the beginning or end.

The original single-pass prompt for `kessler_run` was **1345 lines (~13k tokens)**. That single file contained:

| Section | ~Lines | Position |
|---|---|---|
| Task header + hard constraints | 20 | Top ✅ |
| Layout policy | 50 | Early ✅ |
| Fortran intrinsics map | 40 | Early ✅ |
| dtype rules | 50 | Middle ⚠️ |
| Return type rules | 50 | Middle ⚠️ |
| JIT boundary rules | 265 | Middle ❌ |
| Vectorization rules | 200 | Middle ❌ |
| Control flow rules | 140 | Middle ❌ |
| Annotation rules | 70 | Late ⚠️ |
| Self-check checklist | 65 | Late ✅ |
| Fortran source | ~200 | Bottom ✅ |

Rules in the **middle** — exactly where JIT safety, vectorization, and dtype rules live — were the most frequently violated in practice. This is not a coincidence.

### The 200-line guideline

For persistent instruction files (like `ORCHESTRATOR.md`), the practical sweet spot for reliable adherence is **under 200 lines**. For one-shot translation prompts the limit is less strict, but the principle holds: a model asked to hold 13 competing rule sets in mind at once will miss some.

---

## 2. The Solution: Multi-Pass Iterative Refinement

**Technique:** Chain-of-prompts iterative refinement. Each pass has one focused job and receives only the policy rules relevant to that job.

```
Pass 1 ──► [First draft code]
              │
              ▼
Pass 2 ──► [JIT-safe code]
              │
              ▼
Pass 3 ──► [Vectorized code]
              │
              ▼
Pass 4 ──► [Final code: typed + annotated + self-checked]
```

Each pass injects the output of the previous pass via a `<<<PREVIOUS_CODE>>>` placeholder in the prompt template. The model reads: "Here is an existing translation — apply these specific rules to it."

**Why this works better than a single pass:**

- Each pass has ≤ 350 lines of policy — well within the reliable adherence zone
- The model focuses on one concern at a time rather than juggling 13
- Errors from Pass 1 that survive to Pass 2 are caught by a model that has read nothing but JIT rules — it is maximally primed to find them
- Intermediate outputs (`kessler_run_pass{1-4}.py`) are saved, making it easy to diagnose which pass introduced a problem

**Pass size comparison:**

| | Old single-pass | New per-pass |
|---|---|---|
| Policy lines | ~1080 | 150–350 |
| Total prompt lines | ~1300 | 535 / 273 / 322 / 197 |
| Token count | ~13k | ~5k each |

---

## 3. Policy File System

### Location

```
workflow_translator/prompt_policies/
├── jax_x64_header.md          # Mandatory imports block (os.environ + jax config)
├── layout_policy.md           # Fortran (ncol,nz) → JAX (nz,ncol) bridge layout rules
├── fortran_intrinsics_map.md  # Fortran intrinsics → JAX equivalents table
├── jax_dtype_rules.md         # float64 consistency rules (dtype= everywhere)
├── return_type_rules.md       # What to return, how to count return values
├── jax_jit_boundary_rules.md  # JIT/tracer rules, static-int, forbidden patterns
├── jax_vectorization_rules.md # Priority ladder: jnp → vmap → fori_loop
├── jax_control_flow_rules.md  # lax.fori_loop, lax.while_loop, state tuples
├── code_annotation_rules.md   # Inline [JAX-VEC], [JAX-FORI], [PY-IF] tags
├── smell_test.md              # Self-check checklist (run before outputting)
├── io_policy.md               # I/O handling: no I/O in _core, wrapper only
├── error_policy.md            # errflg/errmsg pattern: signal in core, string in wrapper
└── MODULE_VARIABLE_POLICY.md  # Fortran MODULE vars treated as INOUT parameters
```

### Design principles

**1. Framework-generic, not scheme-specific.**
All policy files use descriptive generic variable names (`field`, `tracer`, `i_start`, `i_end`, `coeff`) rather than Kessler-specific names (`qr`, `theta`, `lyr_surf`). This prevents:
- The LLM pattern-matching examples onto the wrong scheme's variables
- Answer contamination (copying Kessler formulas into non-Kessler translations)
- Confusion when a new scheme happens to share a variable name with the examples

**2. Embedded fallbacks in the Python script.**
Every policy file has a corresponding string constant in `workflow_translator/phase04_01_make_prompts.py`. The `_load_policy(filename, fallback)` function loads from `workflow_translator/prompt_policies/` first; if the file is missing it falls back to the embedded constant. This means the pipeline never silently breaks if a file is moved.

**3. Edit policy without touching Python.**
To update a rule (e.g., add a new Fortran intrinsic, clarify a JIT rule), edit the `.md` file in `workflow_translator/prompt_policies/` and re-run `phase04_01_make_prompts.py`. No Python code changes needed.

**4. `SCALAR_ONLY_POLICY_BLOCK` is the exception.**
This policy block remains embedded in the Python script because it uses `.format(proc_name=proc_name)` substitution that is incompatible with file-based loading.

---

## 4. The 4-Pass Design

### Pass 1 — First Draft

**Goal:** Produce a structurally correct first translation.

**Policies loaded:**
- `jax_x64_header.md` — mandatory imports
- `layout_policy.md` — array layout (critical for indexing correctness)
- `fortran_intrinsics_map.md` — so `MAX/MIN/DIM/MERGE` are translated correctly
- `return_type_rules.md` — correct signature and return values
- `MODULE_VARIABLE_POLICY.md` — only if the procedure uses Fortran MODULE variables
- `io_policy.md` — only if the procedure has I/O
- `error_policy.md` — only if the procedure has `errmsg`/`errflg`

**Also includes:** procedure-specific signature section + full Fortran source.

**Output:** A complete Python file with the two-layer structure (`_core` + wrapper), possibly with JIT/tracer violations and unoptimized loops — that is acceptable at this stage.

---

### Pass 2 — JIT Safety

**Goal:** Fix every JIT/tracer boundary violation.

**Policies loaded:**
- `jax_jit_boundary_rules.md` — the full rule set (256 lines), including:
  - FORBIDDEN patterns (Python `if`/`for`/`while` on JAX values)
  - Tracer infection rule
  - Static integer parameters rule
  - Correct replacements (`jnp.where`, `lax.cond`)
  - `lax` loop state type/shape invariant

**Input:** `<<<PREVIOUS_CODE>>>` — the Pass 1 output.

**Output:** Same code with all JIT violations fixed. Second line of file updated to `# JIT boundary check: FIXED — <description>` or `# JIT boundary check: PASSED`.

---

### Pass 3 — Vectorization & Control Flow

**Goal:** Optimize loops and fix control flow patterns.

**Policies loaded:**
- `jax_vectorization_rules.md` — nested loop consistency, priority ladder, vmap pattern, lax.scan
- `jax_control_flow_rules.md` — fori_loop/while_loop patterns, state management, scoping traps

**Input:** `<<<PREVIOUS_CODE>>>` — the Pass 2 output.

**Output:** Independent loops replaced with `jnp` broadcast or `jax.vmap`; no Python loops wrapping `lax.*` primitives; state tuples correct.

---

### Pass 4 — dtype, Annotations & Self-check

**Goal:** Finalize numerical correctness markers and verify the full checklist.

**Policies loaded:**
- `jax_dtype_rules.md` — `dtype=jnp.float64` on every allocation
- `code_annotation_rules.md` — `# [JAX-VEC]`, `# [JAX-FORI]`, `# [PY-IF]` etc. on every loop/conditional
- `smell_test.md` — exhaustive self-check checklist Claude must verify before outputting

**Input:** `<<<PREVIOUS_CODE>>>` — the Pass 3 output.

**Output:** Final production-ready Python file.

---

### Pass-count classes (scalar procedures)

Not every procedure runs all four passes. `out/packets/_ALL_deps.json` carries
two flags per procedure that set the pass count (see TRANSLATE_WORKFLOW.md §1b):

| `scalar_only` | `jax_required` | Passes | Style |
|---|---|---|---|
| `true` | `false` | **1** | Plain Python — no JAX at all (e.g. `kessler_init`). Only `_pass1.md` is generated. |
| `true` | `true` | **2** | JAX-scalar — called from inside a `@jax.jit` context, so it must use `jnp` ops and `jnp.where`, but gets no `_core` split and no own `@jax.jit`. Pass 1 drafts, pass 2 fixes JIT-boundary violations. |
| `false` | (always `true`) | **4** | Full JAX — `_core` + wrapper, the complete chain above. |

---

## 5. File Connections — Inputs and Outputs

```
out/packets/{proc}_merged.json          ← Phase 3 output: Fortran analysis packet
         │
         ▼
workflow_translator/phase04_01_make_prompts.py        ← Prompt generator
         │  reads: workflow_translator/prompt_policies/*.md
         │  reads: out/packets/{proc}_merged.json
         ▼
out/prompts/{proc}_pass1.md             ← Complete prompt (no placeholder)
out/prompts/{proc}_pass2.md             ← Template (contains <<<PREVIOUS_CODE>>>)
out/prompts/{proc}_pass3.md             ← Template (contains <<<PREVIOUS_CODE>>>)
out/prompts/{proc}_pass4.md             ← Template (contains <<<PREVIOUS_CODE>>>)
         │
         ▼
translation author                      ← in-context agent (Claude/GPT/Gemini)
         │                                or external batch driver selected by
         │                                [translator] in config/project.toml
         │                                (e.g. workflow_translator/qwen/qwen_API_translate.py)
         │  pass1: injects nothing
         │  pass2: injects out/jax/{proc}_pass1.py
         │  pass3: injects out/jax/{proc}_pass2.py
         │  pass4: injects out/jax/{proc}_pass3.py
         ▼
out/jax/{proc}_pass1.py                 ← Intermediate: first draft
out/jax/{proc}_pass2.py                 ← Intermediate: JIT-safe
out/jax/{proc}_pass3.py                 ← Intermediate: vectorized
out/jax/{proc}.py                       ← Final output (= pass4 result)
         │
         ▼
validation/phase05_01_lint_translation.py    ← Static analysis
validation/phase05_02_runtime_validate.py    ← Runtime comparison vs Fortran reference
```

### Claude / manual workflow

When using Claude (web or IDE), paste the prompt files directly in sequence:

```
1. Paste kessler_run_pass1.md          → copy the generated code
2. Paste kessler_run_pass2.md,
   replace <<<PREVIOUS_CODE>>> with
   the Pass 1 output                   → copy the refined code
3. Repeat for pass3, pass4
4. Save final output to out/jax/kessler_run.py
```

---

## 6. How to Run

### Step 1: Generate prompts

```bash
python workflow_translator/phase04_01_make_prompts.py
```

Reads `out/packets/*_merged.json`, writes all pass prompt files to `out/prompts/`.

### Step 2: Run translation

**In-context** (default): follow `workflow_translator/TRANSLATE_WORKFLOW.md`
Step 2 — the driving agent reads each prompt and authors the passes itself.

**External model** (when `[translator]` is set in `config/project.toml`, e.g.
the paper's Qwen2.5-Coder-32B via vLLM): one PBS job loads the model once and
batches every pass stage across all procedures —

```bash
qsub pbsJobs/pbs_qwen25_translate.sh                          # all procedures
qsub -v PROCS=proc_a,proc_b pbsJobs/pbs_qwen25_translate.sh   # re-roll a subset
```

Each external translator is one self-contained folder —
`workflow_translator/<model>/` with the module doc `<model>.md` and the
batch driver — plus its PBS jobs in `pbsJobs/`.

### Step 3: Validate

```bash
python validation/phase05_01_lint_translation.py
python validation/phase05_02_runtime_validate.py
```

### Updating a policy rule

```bash
# Edit the relevant policy file (never the generated out/prompts/* files)
vim workflow_translator/prompt_policies/jax_jit_boundary_rules.md

# Regenerate all prompts
python workflow_translator/phase04_01_make_prompts.py

# Re-run translation (in-context Step 2, or resubmit the [translator] PBS job)
```

---

## 7. Adding a New Physics Scheme

The prompt system is scheme-agnostic. Policy files use generic variable names (`field`, `tracer`, `i_start`, `i_end`) to avoid anchoring on Kessler-specific names.

To translate a new scheme:

1. **Run the analysis pipeline** (phases 1–3) to produce `out/packets/{proc}_merged.json`
2. **Run the prompt generator** — `phase04_01_make_prompts.py` picks up all `*_merged.json` files automatically and generates the pass prompts for each
3. **Pick the translation author** — in-context needs nothing extra; for an external model set `[translator]` in `config/project.toml` (see `../docs/NEW_PROJECT_CHECKLIST.md` §4)
4. **No policy changes needed** unless the new scheme introduces a translation pattern not covered by the existing rules (then add a project policy file via `[prompts].project_policies`)

Full setup checklist: `../docs/NEW_PROJECT_CHECKLIST.md`.

---

## 8. Design Decisions and Trade-offs

### Why not a single system prompt + multi-turn conversation?

Multi-turn conversation would be ideal but is not available in the Qwen/vLLM offline inference setup. The chain-of-prompts approach achieves the same progressive refinement effect using separate inference calls.

### Why keep embedded fallbacks in the Python script?

Robustness. If `workflow_translator/prompt_policies/` is missing or a file is renamed, the pipeline falls back to the embedded constant and continues working. The fallbacks also serve as documentation: the Python source is a readable record of every policy rule even without the `.md` files.

### Why save intermediate outputs?

Debugging. When the final output has a bug, the intermediates reveal which pass introduced it:
- Bug in `_pass1.py` → structural issue (signature, layout, intrinsics)
- Bug introduced between `_pass1` and `_pass2` → unlikely (pass 2 only fixes, shouldn't break)
- Bug in `_pass3.py` → vectorization or control flow regression
- Bug in final but not `_pass3` → dtype or annotation issue

### Why not re-run all 4 passes on failure?

Cost and time. Passes 1–4 together take ~4× the inference time of a single pass. If only pass 3 fails (vectorization), re-running passes 1–2 wastes GPU time. The `--start-pass` flag exists for this reason.

### Why descriptively generic names over single-letter placeholders?

Single-letter names (`a`, `b`, `x1`) are unreadable and reduce the educational value of examples. Descriptive generic names (`field`, `tracer`, `coeff`, `i_start`) are readable while being clearly illustrative and non-scheme-specific.

### Token budget per pass

Prompts sit at ~5k tokens of template plus the injected previous-pass code (kessler_run pass 3, the widest, measured ~14k prompt tokens). For the locally-served translator (qwen) the output cap and the engine window are **not fixed constants** — `workflow_translator/vllm_budget.py` sizes them per procedure and per pass:

| Constant | Value | Meaning |
|---|---|---|
| `CODE_RATIO` | 1.8 | expected JAX size = Fortran source × ratio (measured 1.37–1.65 on kessler) |
| `SAFETY` | 2.0 | output cap = expected code × safety |
| `OUT_FLOOR` | 3,072 | minimum cap — tiny procedures still get room |
| `OUT_CEILING` | 40,960 | maximum cap (was 16,384 until 2026-08-14; raised for a ~2,000-line orchestrator procedure in another LAFT project) |
| `MARGIN` | 512 | reserved for the chat template |

Per request `max_tokens = min(clamp(source × 1.8 × 2.0, floor, ceiling), window − prompt − margin)`, computed against the prompt actually sent, so the window cannot be exceeded. The engine window itself is chosen before model load by `plan_engine_context()` — the smallest window (rounded to 1024) that fits the widest pass plus its output budget, kept far below the trained maximum to preserve KV-cache concurrency. Every budget is logged to `out/reports/analysis/generation_log.jsonl` (`prompt_tokens`, `max_tokens`, `finish_reason`, `hit_cap`), so any truncation is attributable. Passing `--max-tokens` / `--max-model-len` pins either half and disables that part of the adaptation.

Three limits interact — context window, output cap, and the model's own *output horizon* (how much code it will emit before it scaffolds), which no token setting moves. The agentic in-context translators (Claude Code, Gemini CLI, Codex) are not bound by the per-request cap at all.
