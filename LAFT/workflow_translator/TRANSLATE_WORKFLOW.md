
# Translator Workflow — Generic Fortran→JAX/Python Translation

You are translating Fortran physics procedures into JAX/Python by executing a
pre-built chain of per-procedure prompts.  This document is **codebase-agnostic**:
the procedure list, pass counts, and translation order are all derived at runtime
from `out/packets/_ALL_deps.json`.

Hand-off gate: only enter this workflow when the bridge workflow
(`workflow_bridge/BRIDGE_WORKFLOW.md`) is complete —
`out/reports/bridge/bridge_test_results.json` exists with `"status": "PASS"`.
If it is missing or FAIL, run the bridge workflow first.

---

## Project configuration

Project-specific values live in `config/project.toml` (see `[hpc]`, `[driver]`,
`[comparison]`). The values below are the framework conventions plus the
config keys that supply the project-specific pieces:

```
PROMPTS_DIR   = out/prompts/
JAX_OUT_DIR   = out/jax/
PROMPTS_TOOL  = python workflow_translator/phase04_01_make_prompts.py
WRAPPERS_TOOL = python workflow_translator/phase04_02_make_wrappers.py
COMPLETE_TOOL = python workflow_translator/phase04_04_completeness_check.py   # §2.5 scaffold stop
LINT_TOOL     = python validation/phase05_01_lint_translation.py
AUDIT_TOOL    = python workflow_translator/phase05_01b_semantic_audit.py
GATE_TOOL     = python workflow_translator/audit_gate.py                      # §3.5 → §4
PBS_RUNTIME   = qsub pbsJobs/jax_gpu_runtimevalid.sh
BRIDGE_SUITE  = python workflow_bridge/run_bridge_tests.py --require-dependent
PBS_DRIVER    = qsub <[hpc].driver_job>                  # GPU; default: pbsJobs/jax_driver_gpu.sh
PBS_COMPARE   = qsub pbsJobs/jax_gpu_compvalues.sh
STATE_FILE    = out/jax/workflow_state.json
STATE_HELPER  = python workflow_translator/translate_workflow_state.py
```

---

## Step 0 — Check for a resume checkpoint (MANDATORY first action)

Read `out/jax/workflow_state.json`.

- **File exists:** you are resuming an in-progress run.  Announce the current
  `status`, the `llm` field, and the `next_steps` list.  Continue from there.
  Do not restart from scratch unless the user explicitly asks.  If the user
  wants a fresh run, tell them to delete `out/jax/workflow_state.json` and
  clear stale artifacts in `out/jax/`, `out/lint/`, `out/validation/`,
  `out/driver/`.
- **File absent:** fresh run; create the file as soon as the first artifact is
  produced (see §State file format).

---

## Step 1 — Build the translation plan from the dependency graph

### 1.0 — (Re)generate the translation inputs (phase 04) — fresh runs only

On a **fresh run** (Step 0 found no `workflow_state.json`), regenerate the
prompts and reference wrappers before translating anything:

```bash
python workflow_translator/phase04_01_make_prompts.py    # -> out/prompts/{proc}_pass*.md
python workflow_translator/phase04_02_make_wrappers.py   # -> out/wrappers/{proc}.py
```

Both are deterministic (derived from `out/packets/` + `prompt_policies/`) and
idempotent, so re-running is safe. This step is **required**, not optional:
`clean_AI_results.sh` wipes `out/prompts/` between LLMs, so a re-translation
starts with no prompts — the translation loop would otherwise skip every
procedure. Regenerating here also guarantees the prompts/wrappers reflect the
current generator (e.g. the module-var signature rule) rather than a stale
copy. Edit `prompt_policies/` — never the generated `out/prompts/*.md` — and
re-run `phase04_01` to change prompt content.

On a **resume** (Step 0 found a checkpoint), skip 1.0 — the prompts already
exist and you are continuing mid-run.

Read `out/packets/_ALL_deps.json`.  Each entry has:

```json
{
  "proc_name":   "kessler_run",
  "scalar_only": false,
  "calls":       ["..."],   // only internal procedure names
  ...
}
```

### 1a — Resolve translation order

Perform a topological sort on the `calls` graph: a procedure must appear in
the translation list **after** every procedure it calls.

Algorithm (Kahn's):
1. Build `in_degree[proc] = number of other procs that proc calls` (i.e.
   the number of dependencies it has).
2. Seed the queue with all procs where `in_degree == 0` (no dependencies).
3. Pop a proc, append to order.  For every other proc that lists this proc in
   its `calls`, decrement its in_degree; enqueue it when it reaches 0.
4. If `len(order) != len(all_procs)`, a cycle exists — report it and abort.

### 1b — Determine pass count per procedure

`_ALL_deps.json` carries two orthogonal flags per procedure:

- `scalar_only`: procedure has no array arguments
- `jax_required`: procedure is called (directly or transitively) from within a
  `@jax.jit` context, so it must use JAX-safe operations even if it is scalar

| `scalar_only` | `jax_required` | Passes | Translation style |
|---------------|---------------|--------|-------------------|
| `true` | `false` | **1** | Plain Python — no JAX at all. Safe to run and validate on CPU only. |
| `true` | `true` | **2** | JAX-scalar — use `jnp` ops and `jnp.where`; no `_core`; no `@jax.jit`. Pass 1 drafts it, pass 2 fixes any JIT-boundary violations. |
| `false` | (always true) | **4** | Full JAX — `{proc}_core` + `{proc}` wrapper, `@jax.jit`, vmap/scan. |

**Why scalar-JAX needs special treatment:** a scalar procedure called from inside
another procedure's `_core` receives JAX-traced scalar values.  Using a plain Python `if` on
a traced value raises `ConcretizationTypeError` at JIT compile time.  The
procedure does not need its own `@jax.jit` or a `_core` split — the JIT
boundary lives in the caller — but it must use `jnp.where` and `jnp` math
throughout.

Verify: a pass-1 prompt file exists at `out/prompts/{proc}_pass1.md` for every
procedure in the list — step 1.0 just generated them. If any is still missing,
**stop and diagnose** (a prompt should never be absent after 1.0; likely a
packet/policy problem), rather than silently skipping the procedure.

### 1c — Print the plan before starting

Print the full translation order with pass count and call list so the user can
verify before any code is written.

---

## Step 2 — Per-procedure translation loop

**Translation source (check once, before the loop):** read `[translator]`
from `config/project.toml`.

- `source = "in_context"`, or the section is absent: **you** author every
  pass, exactly as described in §2a below.
- Any other `source`: the initial translation is authored by an external
  model. Do **not** run §2a yourself — read the module doc at
  `[translator].module` and follow it to produce each procedure's files, then
  rejoin this workflow at §2b (header check) and §2d (state update), using
  `[translator].llm_label` as the LLM name everywhere the state file, headers,
  and report require one. Everything outside initial authorship — the plan,
  lint, semantic audit, PBS validation, the fix loop, the report — stays in
  this document, and **every repair is yours as the driving agent**: the
  external model is never re-invoked to fix its own output.

Repeat for each `{proc}` in topological order:

### 2a — Run the passes

**Scalar-only (1 pass)**

1. Read `out/prompts/{proc}_pass1.md` in full (it contains the Fortran source).
2. Produce a plain-Python translation — no `import jax`, no `_core` function.
3. Write to `out/jax/{proc}.py`.

**Array procedure (4 passes)**

Pass 1 — first draft:
1. Read `out/prompts/{proc}_pass1.md`.
2. Generate a first-draft JAX translation.
3. Write to `out/jax/{proc}_pass1.py`.

Pass 2 — JIT safety:
1. Read `out/prompts/{proc}_pass2.md`.
2. Substitute the content of `out/jax/{proc}_pass1.py` for `<<<PREVIOUS_CODE>>>`.
3. Fix every JIT/tracer violation.  Write to `out/jax/{proc}_pass2.py`.

Pass 3 — vectorization & control flow:
1. Read `out/prompts/{proc}_pass3.md`.
2. Substitute `out/jax/{proc}_pass2.py` for `<<<PREVIOUS_CODE>>>`.
3. Write to `out/jax/{proc}_pass3.py`.

Pass 4 — dtype, annotations & self-check:
1. Read `out/prompts/{proc}_pass4.md`.
2. Substitute `out/jax/{proc}_pass3.py` for `<<<PREVIOUS_CODE>>>`.
3. Write the **final** result to `out/jax/{proc}.py` (bare name, not `_pass4`).

### 2b — LLM version header (MANDATORY on every output file)

Every file you write — intermediates and finals — must begin with:

```python
# ---------------------------------------------------------------------------
# Translated by: <model name and exact model ID>
# Pass: <1 | 2 | 3 | 4 | final>
# ---------------------------------------------------------------------------
```

Use the model ID from your system prompt.  Place this before all imports.

### 2c — Hard rules (mirrors lint checks)

- Array procedures (`scalar_only=false`) MUST have both `{proc}_core(...)` and `{proc}(...)`.
- Scalar-Python procedures (`scalar_only=true, jax_required=false`) MUST NOT have `_core` and MUST NOT `import jax` or `jax.numpy`.
- Scalar-JAX procedures (`scalar_only=true, jax_required=true`) MUST NOT have `_core`, MUST NOT add `@jax.jit`, but MUST use `jnp` operations and MUST NOT use Python `if`/`else` on any argument.
- No `import numpy` and no bare `np.` anywhere.
- No `swapaxes(` in `out/jax/` files (bridge handles layout).
- `_core` MUST NOT accept string parameters (`errmsg`, `scheme_name`).
- `_core` MUST contain no `print(`, `input(`, or `open(`.
- Mandatory JAX header (array procs only):
  ```python
  import os
  os.environ["JAX_ENABLE_X64"] = "1"
  import jax
  jax.config.update("jax_enable_x64", True)
  ```
  The `os.environ` line MUST come before `import jax`.

### 2d — Update workflow state after each procedure

After writing the final `out/jax/{proc}.py`:

```bash
python workflow_translator/translate_workflow_state.py \
  --status generated_code \
  --llm "<LLM name and exact model ID>" \
  --complete-step "generated out/jax/{proc}.py"
python workflow_translator/phase04_04_completeness_check.py --proc {proc}     # Step 2.5 — scaffold stop
```

The second command is the per-procedure **completeness check (§2.5)**. If it
prints `SCAFFOLD DETECTED` the translation is **stopped** — do not author the
next procedure, do not re-run the pass; read §2.5.

**Run this immediately after each procedure, not batched at the end.** The
script stamps the wall-clock time of every completed step into
`step_timestamps`, and a procedure's authoring duration is the gap between
consecutive `generated out/jax/<proc>.py` stamps. Batching the calls collapses
those gaps and destroys the per-procedure timing in Table 1 (§2e). The
timestamp is recorded by the script, never reported by the agent.

### 2.5 — Completeness check: scaffold = HARD STOP (no retry)

```bash
python workflow_translator/phase04_04_completeness_check.py --proc {proc}   # after each procedure (§2d)
python workflow_translator/phase04_04_completeness_check.py                 # whole set, once the loop ends
```

Checks that each procedure was **translated, not summarised**: scaffold
admission phrases ("would go here", "remaining physics", …) and callees that
appear only commented-out are **FAIL**; callees absent altogether, an extreme
JAX/Fortran size ratio, or a scalar-only misclassification are **WARN**
(listed for you to judge). Results: `out/reports/translation/completeness_check.json`
+ the durable `completeness_report.md` (per-procedure run: `_<proc>` suffix).

**Any FAIL stops the translation.** The script writes `status =
"aborted_scaffold"` (+ `scaffold_abort` evidence) to `workflow_state.json`,
exits 2, and every downstream gate (`audit_gate.py` → runtime PBS) stays closed.
A scaffold is a model/context limit — the output horizon (how much code
the model emits before it scaffolds) — not a code defect:
re-running the same prompt reproduces it and would loop the fix budget away.
**No automatic re-translation. Report to the user and wait**: the options
(split the procedure, change model or prompt policy, archive the run as a
failure experiment via `tools/copy_AI_results.sh`) are the user's. Only after
that decision does the run resume at §2.5 for the affected procedure(s).
Scaffold FAILs do **not** count toward the fix-attempt budget (§6) — they are
not fixes.

### 2e — Freeze the four-pass statistics (ONCE, after the loop)

When **every** procedure in the plan has been authored — and **before** any
validation or fix — freeze the four-pass statistics:

```bash
python workflow_translator/phase04_03_translation_stats.py freeze
```

Writes `out/reports/translation/pass_stats.json` (+ a `.sha256` digest): per
procedure, its parent module, pass count, Fortran and JAX line counts, prompt
and output sizes, tokens where an API reported them, and wall-clock.

**This file is immutable and the command refuses to overwrite it.** It is the
only record of the untouched four-pass output; once fixing starts, the JAX
files change and that record cannot be reconstructed. Freezing late — after a
fix — silently records repaired code as if the translator had produced it.

`--refreeze` exists solely for a genuine re-run after
`tools/clean_AI_results.sh`. Never use it to "update" the numbers: post-
authorship changes belong in Table 2.

Add `"froze four-pass statistics"` to `completed_steps`.

---

## Step 3 — Lint all procedures

After all procedures are written, run lint:

```bash
python validation/phase05_01_lint_translation.py
```

Results land in `out/lint/{proc}_lint.json` and `out/lint/_summary.json`.
The summary's `"score"` must be **100.0** and `"overall": "PASS"`.

If lint fails: apply a minimal surgical fix directly to `out/jax/{proc}.py`
(the final file, not a pass intermediate).  Do not re-run any pass.  Rerun
lint until it passes.  Each fix cycle counts toward the fix-attempt budget (§6).

---

## Step 3.5 — Semantic audit (before runtime validation) — GATED

Lint checks structure; this step checks **fidelity to the Fortran**. It is a
**hard precondition of Step 4**: `translate_workflow_state.py` refuses to record
a runtime-validation job, and `pbsJobs/jax_gpu_runtimevalid.sh` refuses to run,
unless `workflow_translator/audit_gate.py` reports the gate OPEN — the
completeness check (§2.5) is present, fresh and not FAIL, the status is not
`aborted_scaffold`, and `out/issues/semantic_audit/_summary.json` exists, is
**newer than every `out/jax/<proc>.py`**, and has `total_fail == 0` (waived
findings do not count).
Every fix to a translation therefore re-closes the gate until the audit is
re-run. Two parts:

**3.5a — automated checker (always run):**

```bash
python workflow_translator/phase05_01b_semantic_audit.py
python workflow_translator/audit_gate.py        # prints OPEN / the reason it is closed
```

When the gate is open, record it:

```bash
python workflow_translator/translate_workflow_state.py \
  --status audit_passed \
  --llm "<LLM name and exact model ID>" \
  --complete-step "semantic audit passed (<n> FAIL, <m> WARN, <k> waived)"
```

Deterministic translation-vs-Fortran comparisons per procedure: **calls-invoked
(FAIL)** — every procedure the Fortran calls must be invoked in the translation
body (`X(`, `X_core(` or `X_value(`; imports don't count; waive deliberately
unported optional paths with a justification); **labelled-blocks (WARN)** —
every Fortran `name: if/do/…` construct must be mentioned somewhere in the
translation (block-not-ported detector); zero-forever variables (Fortran
computes them, translation only zeroes them — chained assignments are split),
`intent(out)` arguments without a fresh entry value, and numeric-literal
set-diffs; with `[semantic_audit].prognostic_vars` set, **update-terms (FAIL)**
— every term of the Fortran `x = x ± (…)` update of a prognostic variable must
appear in the translation's update of `x` (a computed rate that is never
applied; `"fortran:python"` entries alias renamed locals); with
`[[semantic_audit.table_readers]]` blocks, **table-reads (FAIL)** — the
translation's lookup-table reader is executed on the real file (numpy only)
and its column base slot, column count, leading-token skip and loop nesting
are checked against the Fortran read statement; **direction-masks (WARN,
self-selecting)** — for Fortran loops with an identifier stride (`do k = A, B,
-kdir`) the translation's `(x - y) * kdir >= 0` masks must have the orientation
the stride implies (the exact test is the vertical-symmetry bridge test,
`workflow_bridge/test_direction_symmetry.py`, in projects whose `bridge_test/` carries it, at Step 4.5). Projects
that enable them in `[semantic_audit].enabled_checks` (config/project.toml)
additionally get lookup-table column-index checking (invented columns are FAIL
and gate the run; Fortran-only columns are WARN — possibly unported optional
paths) and level-convention detection. Findings land in
`out/issues/semantic_audit/<proc>.json` (+ `_summary.json`, the gate input) and
the durable **`out/reports/translation/semantic_audit_report.md`** (header with
verdict/waived counts, which checks ran vs. n/a for this project, per-procedure
table, findings table) — pasted into the Step 8 report. Justified exceptions go
in `out/issues/semantic_audit_waivers.json`.

**3.5b — agent formula audit (run when 3.5a warns, or whenever a diagnosis
reveals the translation *paraphrased* rather than ported):**

Read the Fortran's variable-semantics declarations first (whatever block the
scheme uses to document its working variables), then diff each process section
formula-by-formula against the translation. Record every discrepancy as an
OPEN entry in `out/issues/fix_log.md` (file, error, evidence line numbers),
fix in batch, and re-validate once. Do NOT fix symptom-by-symptom once
paraphrasing is established — each invented formula found statically saves a
full validation cycle.

## Step 4 — PBS runtime validation

Submit after lint **and the semantic audit (Step 3.5)** are green — the state
tool and the PBS script both refuse otherwise (`audit_gate.py`):

```bash
qsub pbsJobs/jax_gpu_runtimevalid.sh
```

Record the returned job id immediately:

```bash
python workflow_translator/translate_workflow_state.py \
  --status waiting_for_runtime_validation \
  --llm "<LLM name and exact model ID>" \
  --job-kind runtime \
  --job-id <job_id> \
  --complete-step "lint passed"
```

Then check status:

```bash
qstat <job_id>
qstat -u "$USER"
```

**Rule — verify freshness before trusting results (MANDATORY).** A PBS job
leaving `qstat` means only that it *stopped* — not that it *succeeded*.  A
crashed job leaves previous result files in place.  After every PBS job:

1. Read the `out/jobs/<name>.o<job_id>` log and confirm no Python traceback, OOM, or
   scheduler error.  The log is the authoritative pass/fail signal.
2. Confirm result files are freshly written (mtime at or after job finish time).
   If files look unchanged, wait and re-check.

Runtime results land in `out/validation/{proc}_runtime.json`.  For each
procedure pass criteria:
- `imported=true`, `wrapper_ok=true`, `core_ok=true` (all procs)
- `wrapper_finite=true` (all procs)
- `core_finite=true` (array procs)

Scalar-only procedures produce `core_ok` and `core_finite` as `null` — that is
expected and not a failure.

If the job is queued for a long time, **pause here**.  Write the state file
with `status="waiting_for_runtime_validation"`, report the job id, and stop.

---

## Step 4.5 — Bridge suite (CPU PBS job)

After runtime validation passes, run the bridge suite against the real
translation — unit-level bridge↔kernel equivalence *before* paying for the
end-to-end driver job:

```bash
qsub -v TEST_SCRIPT=workflow_bridge/run_bridge_tests.py,TEST_ARGS=--require-dependent pbsJobs/jax_cpu_test.sh
```

The suite executes JAX, so it goes through PBS like every other JAX step
(login-node rule): `pbsJobs/jax_cpu_test.sh` runs
`python workflow_bridge/run_bridge_tests.py --require-dependent` on a CPU
compute node (develop queue, a few minutes including the queue wait; no new
workflow status). When the job leaves the queue read its log in `out/jobs/`.
It executes the project's whole `bridge_test/` folder — including
the translation-DEPENDENT tests (bridge vs direct-kernel call, error
propagation, module-var INOUT values), which are skipped during the bridge
workflow and only become active now that `out/jax/` holds a real
translation. `--require-dependent` makes those tests part of the gate:
skipped or absent dependent tests FAIL it.

Pass criterion: exit code 0 — equivalently
`out/reports/bridge/bridge_test_results.json` has `"status": "PASS"` with
`translation_dependent_tests.state = "passed"`. The JSON and
`bridge_report.md` are rewritten each run, so the report always reflects the
current translation.

On failure, enter the Step 6 fix loop. Attribution guide:
- Only dependent tests fail, layout tests green → the defect is in the
  **translation** (`out/jax/{proc}.py`) — the bridge was already verified
  translation-independently. A bridge-vs-direct mismatch means argument
  order or return-tuple order in the translation's signature; an error-path
  failure usually means the translation dropped or reworded a Fortran error
  branch.
- Layout tests fail too → regenerate the bridge (see
  `workflow_bridge/BRIDGE_WORKFLOW.md`, "If the test gate fails") before
  blaming the translation.
- If the project has no translation-dependent bridge tests yet, write them
  first (per-project code — see the BRIDGE_WORKFLOW.md new-project
  checklist); do not skip this step.

---

## Step 5 — PBS driver and comparison (end-to-end)

### 5.0 — Declared contracts define the mode matrix

Read `[driver].contracts` from `config/project.toml` (absent key = `[1]`,
host-only). It is the project's bridge-contract decision record — see the
template for the lifecycle (`[1, 2]` = undecided, exercise both; one value =
pinned). Each declared contract maps to a driver mode:

| contract | `LAFT_DRIVER_MODE` | bridge entry driven |
|---|---|---|
| 1 | `host` (also the default when the variable is unset) | `<proc>_bridge` |
| 2 | `device_resident` | `<proc>_bridge_device` |

`LAFT_DRIVER_MODE` is a frozen framework convention (name and both values);
the driver script itself stays per-project code (reference implementation:
`kessler/out/driver/kessler_jax_driver.py`).

**Fail fast:** if contract 2 is declared, confirm the project driver
actually implements the switch (`grep LAFT_DRIVER_MODE <[driver].script>`)
**before** submitting anything; if it does not, stop and report — the driver
needs the mode added first. Projects with no driver at all skip Step 5 as a
whole, exactly as before; the declaration records intent for when one exists.

**The driver + comparison pair below runs ONCE PER declared contract**, and
the gate is comparison `ALL PASS` in **every** declared mode. Run the modes
sequentially (each pair overwrites `[driver].output_file` and the comparison
artifacts — verify freshness per mode as always) and name the mode in every
state-file step you record.

### 5.1 — Driver and comparison, per declared mode

After the bridge suite passes (Step 4.5), for each declared mode:

```bash
qsub -v LAFT_DRIVER_MODE=<mode> <[hpc].driver_job>
                          # the project driver job from config/project.toml —
                          # runs on GPU (default: pbsJobs/jax_driver_gpu.sh);
                          # plain qsub (variable unset) = host mode
```

Record the job id:

```bash
python workflow_translator/translate_workflow_state.py \
  --status waiting_for_driver \
  --llm "<LLM name and exact model ID>" \
  --job-kind driver \
  --job-id <job_id> \
  --complete-step "runtime validation + bridge suite passed"
```

After the driver job completes and its log is traceback-free:

```bash
qsub pbsJobs/jax_gpu_compvalues.sh   # runs the project's [comparison].script from config
```

Record:

```bash
python workflow_translator/translate_workflow_state.py \
  --status waiting_for_comparison \
  --llm "<LLM name and exact model ID>" \
  --job-kind comparison \
  --job-id <job_id> \
  --complete-step "driver passed"
```

The comparison report at `out/driver/compare_results_fortran_jax.txt` is
authoritative.  Pass criterion: `ALL PASS` — the check set and tolerances are
defined by `[comparison]` in `config/project.toml` (physics-level criteria; a
single-precision Fortran reference vs a float64 translation does not agree to
machine epsilon).

**Step 5 is green only when comparison reads `ALL PASS` in every mode
declared by `[driver].contracts` (§5.0).** With more than one mode declared,
loop back to §5.1 for the next mode; the host run stays the reference
debugging path when a device-resident mode fails (both modes drive the same
translation and the two bridge entries are bit-identical by construction, so
a device-mode-only failure points at the driver's resident state handling,
not the kernel).

---

## Step 6 — Fix loop and abort gate

**Rule: fix the final `out/jax/{proc}.py` directly.  Do NOT re-run any pass.**

Re-running a pass with the same prompt and same `<<<PREVIOUS_CODE>>>` reproduces
the same bug.  Patch the generated file in place and re-validate.

**Maximum fix attempts: 5 per (file, error)** — five tries at fixing the *same*
error in the *same* file, at any stage (lint, runtime, bridge suite, driver,
comparison).
There is no cap on the total number of distinct fixes in a run; the budget
exists to stop unproductive loops on one stubborn error.  Track in
`workflow_state.json`:

```json
"fix_attempts": [
  {
    "n": 1,
    "timestamp": "YYYY-MM-DDTHH:MM:SS",
    "proc":          "kessler_run",
    "failed_stage":  "runtime",
    "error_summary": "ConcretizationTypeError at line 88",
    "file_edited":   "out/jax/kessler_run.py",
    "change_summary": "replaced if-on-array with jnp.where"
  }
],
"fix_attempts_used": 1,
"fix_attempts_max": 5
```

**Also append one line per fix attempt to
`out/reports/translation/fix_stats.jsonl`** — this is what Table 2 is rendered
from, and it is cumulative: every attempt gets a line, including failed ones.

```json
{"n": 1, "at": "YYYY-MM-DDTHH:MM:SS", "proc": "kessler_run",
 "module": "kessler", "failed_stage": "runtime",
 "error_summary": "ConcretizationTypeError at line 88",
 "file_edited": "out/jax/kessler_run.py",
 "change_summary": "if-on-array -> jnp.where",
 "duration_s": 900, "tokens": null}
```

`duration_s` is the elapsed time you spent on that attempt; use `null` if you
genuinely cannot bound it. `tokens` is `null` for in-context repair — there is
no API call to report a count, and a guess in that column would be worse than
an honest `n/a`. Do **not** touch `pass_stats.json` when fixing: Table 1 records
what the four-pass translation produced, Table 2 records what happened after.

When any single (file, error) pair reaches 5 failed attempts and validators
still fail:
- Set `status="aborted_max_fix_attempts"`.
- Still write the final report (§8).
- Tell the user which procedure and error remain unresolved, and recommend
  re-running the relevant pass with the error injected as new context.

---

## Step 7 — Snapshot validated translations

Once comparison is green, snapshot every final translation:

```bash
for f in out/jax/*.py; do
  base=$(basename "$f" .py)
  cp "$f" "out/jax/${base}.validated.py"
done
```

Add `"snapshotted out/jax/*.validated.py"` to `completed_steps`.

---

## Step 8 — Write the results report

Write `out/reports/translation/translation_report.md` (consolidated
reports tree: `out/reports/<workflow>/` — bridge and profile reports live
under their own subfolders the same way). Required sections:

- **Header**: run date, LLM name + exact model ID, workflow file path, final
  status (`complete` or `aborted_max_fix_attempts`), total `fix_attempts_used`.
- **Translation plan**: the topological order table generated in Step 1c, with
  pass counts.
- **Translation statistics (MANDATORY, both tables)** — generate them, do not
  hand-type them:

  ```bash
  python workflow_translator/phase04_03_translation_stats.py render \
    --out out/reports/translation/_stats_tables.md
  ```

  Paste the output into the report verbatim. It emits:
  - **Table 1 — four-pass translation statistics (FROZEN).** One row per
    procedure, grouped under its parent module with per-module subtotals and a
    grand total: passes, Fortran lines, JAX lines, JAX/Fortran ratio, prompt and
    output size, tokens, wall-clock. **Never edit this table**, and never
    re-freeze to make it agree with the current files — it is the record of the
    initial output, and it is expected to disagree with post-fix code.
  - **Table 2 — validation-phase fix statistics (LIVE).** Same shape, one row
    per *fixed* procedure: fix count, which stages failed, JAX lines frozen →
    now with the delta, fix tokens, fix time, last fixed. Re-render after every
    fix so the report always reflects the current state.

  Modules appear as grouping headers with subtotals only. A module is never
  translated as a unit — no file is generated for one — so it has no line
  count, token count or duration of its own; its numbers are strictly the sum
  of its procedures'.

  Columns show `n/a` where no honest source exists (tokens under
  `source = "in_context"`; wall-clock when state timestamps are missing). Never
  substitute an estimate for a measurement.

- **Completeness + semantic audit (MANDATORY)**: paste verbatim the header and
  per-procedure tables of `out/reports/translation/completeness_report.md`
  (§2.5) and `out/reports/translation/semantic_audit_report.md` (§3.5a), plus
  the FAIL/WARN findings rows and the waivers used. A run that stopped on
  scaffold (`aborted_scaffold`) reports the completeness table as its
  primary evidence.
- **Results table**: one row per procedure — lint score, runtime status
  (wrapper_ok / core_ok, finite), comparison MAE/RMSE (if reached).
- **Files generated**: list each final file with one line on what it does.
- **Fix log** (MANDATORY when `fix_attempts_used > 0`): numbered table matching
  `workflow_state.json → fix_attempts`.
- **What worked / what was tricky**: short paragraph for future runs.

Write this report even on a clean first-attempt success — it is the durable
record.  When the workflow aborts, the report is the *only* artifact explaining
why.

---

## State file format

```json
{
  "workflow": "workflow_translator/TRANSLATE_WORKFLOW.md",
  "status":   "waiting_for_runtime_validation",
  "llm":      "Claude Sonnet 4.6 (claude-sonnet-4-6)",
  "updated_at": "YYYY-MM-DDTHH:MM:SS",
  "translation_plan": [
    {"proc": "kessler_init", "scalar_only": true,  "pass_count": 1, "calls": []},
    {"proc": "kessler_run",  "scalar_only": false, "pass_count": 4, "calls": []},
    "..."
  ],
  "completed_steps": [
    "generated out/jax/kessler_init.py",
    "generated out/jax/kessler_run.py",
    "..."
  ],
  "pbs_runtime_validation": {
    "job_id": "1234567.desched1",
    "job_name": "runtimevalidate",
    "expected_log": "runtimevalidate.o1234567",
    "submit_command": "qsub pbsJobs/jax_gpu_runtimevalid.sh",
    "status_command": "qstat 1234567.desched1"
  },
  "pbs_driver":     null,
  "pbs_comparison": null,
  "fix_attempts": [],
  "fix_attempts_used": 0,
  "fix_attempts_max": 5,
  "next_steps": [
    "check qstat 1234567.desched1",
    "read runtimevalidate.o1234567",
    "read out/validation/*_runtime.json"
  ]
}
```

Recognized `status` values:

| Status | Meaning |
|--------|---------|
| `generated_code` | Code written; completeness check (§2.5) / lint not yet run |
| `aborted_scaffold` | **STOPPED** — a procedure was scaffolded (§2.5); human decision required, no retry; downstream gates closed |
| `lint_failed` | Fix code; re-run lint |
| `lint_passed` | Run the semantic audit (Step 3.5) |
| `audit_passed` | Submit PBS runtime validation (gate `audit_gate.py` must be OPEN) |
| `waiting_for_runtime_validation` | Waiting for PBS job |
| `runtime_validation_failed` | Fix code; re-run lint + PBS runtime |
| `runtime_validation_passed` | Run bridge suite (Step 4.5); if it passes, submit PBS driver |
| `waiting_for_driver` | Waiting for PBS driver job |
| `driver_failed` | Fix code/env; re-run lint + PBS runtime + bridge suite + driver |
| `driver_passed` | Submit PBS comparison |
| `waiting_for_comparison` | Waiting for PBS comparison job |
| `comparison_failed` | Fix numerical/layout bug; re-run full validation chain |
| `comparison_passed` | Snapshot + write report |
| `complete` | Report written |
| `aborted_max_fix_attempts` | Hit fix budget; report written |

---

## Quick command reference

```bash
# (Re)generate prompts + wrappers on a fresh run (Step 1.0)
python workflow_translator/phase04_01_make_prompts.py
python workflow_translator/phase04_02_make_wrappers.py

# Generate translation plan (Step 1)
python -c "
import json; from pathlib import Path; from collections import deque
d = json.loads(Path('out/packets/_ALL_deps.json').read_text())
name_map = {e['proc_name'].lower(): e['proc_name'] for e in d}
graph = {e['proc_name'].lower(): [c.lower() for c in e['calls']] for e in d}
in_degree = {n: 0 for n in graph}
for caller, callees in graph.items():
    for callee in callees:
        if callee in in_degree:
            in_degree[caller] += 1
q = deque(n for n in graph if in_degree[n] == 0); order = []
while q:
    n = q.popleft(); order.append(n)
    for caller, callees in graph.items():
        if n in callees:
            in_degree[caller] -= 1
            if in_degree[caller] == 0: q.append(caller)
for i, pn in enumerate(order, 1):
    e = next(x for x in d if x['proc_name'].lower() == pn)
    print(f'{i:2}. {e[\"proc_name\"]:40s} passes={1 if e[\"scalar_only\"] else 4}')
"

# Completeness / scaffold stop (§2.5 — per procedure and whole set; FAIL = STOP, no retry)
python workflow_translator/phase04_04_completeness_check.py --proc <proc>
python workflow_translator/phase04_04_completeness_check.py

# Run lint
python validation/phase05_01_lint_translation.py

# Semantic audit + gate (§3.5 — login node; MUST be open before Step 4)
python workflow_translator/phase05_01b_semantic_audit.py
python workflow_translator/audit_gate.py

# Submit and record PBS jobs
qsub pbsJobs/jax_gpu_runtimevalid.sh
# → record job id with translate_workflow_state.py (see §4)

# Bridge suite gate (§4.5 — CPU PBS job, after runtime validation)
qsub -v TEST_SCRIPT=workflow_bridge/run_bridge_tests.py,TEST_ARGS=--require-dependent pbsJobs/jax_cpu_test.sh

# Inspect results
cat out/jax/workflow_state.json
cat out/lint/_summary.json
cat out/validation/{proc}_runtime.json
tail -n 120 out/jobs/runtimevalidate.o<job_number>
cat out/driver/compare_results_fortran_jax.txt
```
