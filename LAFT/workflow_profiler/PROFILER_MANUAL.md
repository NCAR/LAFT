# workflow_profiler — Structure, Methodology & Operating Manual

A reference on what `workflow_profiler/` is, how it is organized, what it
measures and why, how it decides, and what it consumes and produces. This is
the **manual** (descriptive architecture + methodology); the **playbook** you
execute step-by-step is `PROFILE_WORKFLOW.md`. This document is
codebase-agnostic: every project value comes from `config/project.toml` —
chiefly `[profiler]` (`target_proc`, `staged_code`, `out_dir`,
`ncol`/`ncol_check`, `inputs_script`, `profile_trace_job`, `profile_nsys_job`)
plus `[hpc]` for the re-validation jobs. `<target_proc>` below stands for the
profiled procedure.

**Working directory for everything below:** the project root (the directory
containing `config/`). All PBS jobs are submitted from there.

---

## 0. Where this fits

| Question | Document |
|---|---|
| Which pipeline stage runs when, and what gates each hand-off? | `../ORCHESTRATOR.md` (pipeline entry point; mandatory user-approval pause between stages) |
| What exact steps do I execute, in order, to run this stage? | `PROFILE_WORKFLOW.md` (the authoritative playbook: resume handling, per-step commands, status values, fix budget) |
| What is this stage, what does it measure, and why are the checks built this way? | **This document** |

This manual does not restate pipeline sequencing or the step-by-step
procedure; it cross-references them. Where the two disagree on an operating
rule, `PROFILE_WORKFLOW.md` wins.

---

## 1. What it is

`workflow_profiler/` is the **profiling stage** of the Fortran→JAX
modernization pipeline (stage 3, optional; it runs after the translator stage
has produced a numerically correct translation). It takes that verified
translation and makes it **hardware-efficient on a GPU**:

```
  workflow_translator/   →   workflow_profiler/   →   final JAX code
  (correctness)              (GPU efficiency)         <[profiler].out_dir>/<target_proc>.py
```

It runs an iterative **profile → diagnose → fix** loop: profile the JAX code on
a GPU node, diagnose its structural/hardware problems, patch the code, and
repeat until the diagnoser reports `production_ready = true`. It then re-runs
the translator's validators to confirm the performance fixes did not break
numerics.

Only the array-heavy procedure named by `[profiler].target_proc` is profiled.
Scalar setup procedures (no GPU kernels) are staged through unchanged.

### Premise

The dominant performance defects in LLM-generated array code — serialized
batch loops, scattered memory writes, host–device round trips — are
**structural**: they are directly observable in the compiled program and its
execution trace, *without* reference timings and *without* a hand-optimized
baseline. The profiler therefore measures the structure of the compiled XLA
program, diagnoses defects against scale-invariant criteria, and drives the
same LLM that produced the translation through a bounded repair loop,
re-establishing numerical correctness after every repair.

The stage is strictly separated from correctness validation. It accepts only
translations that have already passed all four validators (static lint,
runtime validation, end-to-end driver, and Fortran-vs-JAX numerical comparison
at machine epsilon), and it operates on a staged working copy while an
immutable snapshot of the validated code is retained for recovery and for
byte-level change detection.

---

## 2. Directory layout

### `workflow_profiler/` — the stage's own files (shared, symlinked from `LAFT/`)

| File | Role |
|---|---|
| `PROFILE_WORKFLOW.md` | The authoritative assistant-facing **playbook**. Every operating rule (hand-off gate, iteration loop, re-validation, abort gate) lives here. |
| `PROFILER_MANUAL.md` | This document — structure, methodology, architecture. |
| `headroom_patterns.md` | Signature → hypothesis → candidate-fix reference for the post-convergence headroom analysis (PROFILE_WORKFLOW.md Step 7). Never gates. |
| `diagnose.py` | **Diagnostic engine.** Single gating input: the trace-pass metrics JSON (trace + compiled-HLO counts). Emits a prioritized issue list with LLM-actionable fixes. Optional `--nsys-json` attaches informational hardware corroboration. |
| `profile_trace_run.py` | Inner runner for the **jax.profiler.trace design pass** — the loop's ONLY profiling job. In-process; traces at two ncol values, dumps + parses the compiled HLO, writes the trace metrics and code_info. |
| `profile_run.py` | Inner runner for the **nsys hardware pass** — the mandatory one-time post-convergence characterization. Wrapped by `nsys profile` in its PBS job. |
| `profiler_inputs.py` | Driver-agnostic input dispatcher: handles staged-code loading and forwards `capture_state`/`tile_state` to the project's `[profiler].inputs_script`. See §7. |
| `parse_jax_trace.py` | Parser that turns a raw Perfetto trace into structural metrics (`while_iterations`, XLA event counts, the two-ncol scaling helper). Importable + CLI. |
| `parse_hlo.py` | Text-level parser of compiled-HLO dumps into instruction counts (fusion/scatter/gather/while, with in-fusion and fusion-root splits). Importable + CLI. |
| `test_diagnose.py` + `test_fixtures/` | Regression tests and real-HLO fixtures pinning the diagnoser's gating semantics. |
| `profile_workflow_state.py` | Helper that maintains the resume state file `<[profiler].out_dir>/profile_state.json`. |

### `pbsJobs/` — the batch jobs this stage submits (per-project)

| Script (config key) | Purpose |
|---|---|
| `<[profiler].profile_trace_job>` | **Profile — design pass (THE loop job).** Runs `profile_trace_run.py` (jax.profiler.trace + compiled HLO). |
| `<[profiler].profile_nsys_job>` | **Mandatory post-convergence hardware characterization.** Runs `profile_run.py` under `nsys`, once, for the headroom analysis and report. |
| `pbsJobs/jax_gpu_runtimevalid.sh` | Re-validation step 1 — runtime/import check. |
| `<[hpc].driver_job>` | Re-validation step 2 — end-to-end driver. |
| `pbsJobs/jax_gpu_compvalues.sh` | Re-validation step 3 — Fortran-vs-JAX numerical comparison. |

### `<[profiler].out_dir>/` (e.g. `out/profiled/`) — all artifacts this stage produces

```
out/profiled/
  <target_proc>.py           # the WORKING COPY the profiler edits ([profiler].staged_code)
  <init_proc>.py             # staged copy of any scalar init code (unchanged)
  <inputs_script basename>   # hand-authored capture_state/tile_state (project infra)
  profile_state.json         # resume state (status, iteration, job ids, fix log)
  iteration_1/               # per-iteration profiling artifacts (see §7)
  iteration_2/
  ...
```

The final results report is written to `out/reports/profile/profile_report.md`
(the consolidated reports tree).

---

## 3. The two-lens profiling model

The profiler looks through **two complementary lenses**. They answer different
questions and must not share a process — nsys instrumentation perturbs the
trace and vice versa. What distinguishes them operationally is **when they
run**: the design lens gates every loop iteration; the hardware lens runs once
after convergence (mandatory), as informational corroboration for the gates
and as the input to the headroom analysis in the report.

| Lens | PBS job | Layer | When | Answers |
|---|---|---|---|---|
| **Design / structure** | `<[profiler].profile_trace_job>` | XLA / HLO graph | **Every iteration (gates the loop)** | Did the translation vectorize each loop? `while_iterations` (+ two-ncol scaling), compiled-HLO scatter/gather/fusion counts, dispatch overhead, D2H copies |
| **Hardware** | `<[profiler].profile_nsys_job>` | Device / CUDA | **Once, post-convergence (mandatory; informational to the gates)** | Device kernel timing, CUDA-graph sync idle, memcpy cost — feeds the headroom analysis |

The single most important defect signal — `while_iterations` — comes from the
design lens: it counts XLA `while.<N>` body executions directly (per call),
and the in-process second-ncol trace measures `while_iterations_scaling`,
which separates legitimate physics subcycling (scaling ≈ 1) from a sequential
column loop (scaling ≈ ncol ratio — the textbook JAX anti-pattern). The
compiled-HLO dump supplies the scatter/gather instruction counts that the
fusion check gates on. Everything that gates `production_ready` is measured by
this one job; the hardware lens corroborates, never gates.

---

## 4. Measurement approach

A **single GPU batch job per iteration** acquires every signal the diagnosis
consumes; no parameter sweep across jobs is required. The job injects the
staged file as the project's translation module, imports the procedure's
**bridge call** (the production entry point, so that for an orchestrator-style
translation the trace covers host orchestration *and* all internal jitted
stage cores), loads the captured driver state tiled to `[profiler].ncol`, and
JIT-warms it so that compilation and autotuning fall outside every measurement
window.

**Inputs are real, not synthetic.** One realistic bridge call is captured from
the project's own driver and expanded to `ncol` columns by the project's
`[profiler].inputs_script` (see §7). Because the capture is cached and the
tiling is seeded, every translation in a cross-model comparison is profiled on
byte-identical inputs.

The job then acquires two complementary views of the same program:

1. **The compiled program.** The process runs with `--xla_dump_to
   … --xla_dump_hlo_as_text`, and every module XLA compiles during the
   main-ncol warmup is concatenated into `<target_proc>_hlo.txt` — the
   compiler's own statement of program structure after fusion, free of runtime
   capture noise. (A single-function `lower().compile().as_text()` dump is not
   used: an orchestrator-style target is not itself one jittable function.
   `parse_hlo`'s counters are line-based, so the aggregate metrics remain
   valid across the concatenation.) The dump is parsed into instruction counts
   for four categories: `fusion`, `scatter`, `gather`, and `while`. The parser
   additionally records *where* each instruction occurs: at entry level,
   inside a fused computation body, or as the root of a fusion. This
   positional distinction carries the diagnosis (§9).

2. **The executed program.** After the warm-up call, `n_calls` calls
   (default 10) are executed inside `jax.profiler.trace`, which records host
   and device activity at the XLA layer in Perfetto format. A second,
   single-call trace is then acquired at `[profiler].ncol_check` for the
   loop-scaling check below (set `ncol_check = 0` to disable).

From the trace, the parser extracts:

- **`while_iterations`** — the principal design signal. XLA emits one trace
  event per execution of each HLO `while` body; the parser counts these events
  directly and normalizes by the number of traced calls. A translation that
  vectorized the column loop executes its physics subcycle once per call,
  batched across all columns; a translation that serialized columns executes
  the body approximately once per column. The loop structure is thus
  *measured*, not inferred from kernel-launch heuristics.
- **`while_iterations_scaling`** — the ratio of per-call while-body executions
  between the two column counts (`ncol` vs `ncol_check`). An absolute
  iteration count alone cannot distinguish legitimate physics subcycling
  (several CFL-limited substeps inside a vectorized loop) from partial column
  serialization; the scaling can. A ratio of ≈1 means the iteration count is
  independent of the number of columns (vectorized — any absolute count above
  1 is the physics' substep count); a ratio tracking the column ratio
  identifies a sequential column loop. The check is structural rather than
  temporal, so a single call at the second size suffices. The metric is
  None-safe: a fully vectorized translation with no while-loop at all has
  `while_iterations` None at both sizes, and the no-while case is itself
  conclusive.
- **Per-call timing distributions** — medians over the traced calls of the
  end-to-end host span (outermost `jit` event), the device execution span
  (`GpuExecutable::ExecuteThunks`), and their difference (dispatch overhead).
  The subtraction is computed strictly within each call's own time window —
  each end-to-end span is paired with the device spans nested inside it —
  never across calls, which precludes the inconsistent-boundary artifacts of
  pairing independent maxima. **Timing is reported for context only; no
  efficiency verdict depends on it.**
- **Device-to-host copy events** (`MemcpyD2H`) per call, and an aggregate XLA
  device-event count used only as a weak complexity indicator.

The runner records the JAX version, device kind, and XLA flags alongside the
metrics for reproducibility, and the trace parser **degrades loudly**: if an
XLA version bump renames the events it matches, the affected metrics become
explicit `None`s accompanied by warnings, never silent zeros.

---

## 5. The iteration loop

Each iteration is three phases. Iterations are numbered from `1`; per-iteration
artifacts live in `<[profiler].out_dir>/iteration_<N>/`.

```
        ┌───────────────────────────────────────────────┐
        │  Phase A — PROFILE   (1 PBS job, async)        │
        │    jax_trace design pass (trace + HLO)         │
        └───────────────────────┬───────────────────────┘
                                │
        ┌───────────────────────▼───────────────────────┐
        │  Phase B — DIAGNOSE  (local, fast)             │
        │    diagnose.py → <target_proc>_diagnosis.json  │
        └───────────────────────┬───────────────────────┘
                                │
              production_ready? │
            ┌───────── no ──────┴────── yes ─────────┐
            ▼                                        ▼
  ┌─────────────────────┐              ┌───────────────────────────┐
  │ Phase C — FIX       │              │ Final validation re-check │
  │  edit staged copy   │              │  audit→runtime→driver→cmp   │
  │  N += 1, loop to A  │              └─────────────┬─────────────┘
  └─────────────────────┘                            ▼
                                            write profile_report.md
```

A hard **abort gate** stops the loop after **5 fix attempts** on the same
failing check if the code still fails — the workflow then writes the report
with `status = aborted_max_fix_attempts`.

---

## 6. Running it — what the stage requires

The step-by-step procedure, with every command and status transition, is in
`PROFILE_WORKFLOW.md`. This section records only the invariants a reader of the
architecture needs.

**Hand-off gate (entry condition).** The profiler must start from a
*known-correct* translation: the translator stage complete, lint score 100,
runtime validation green, driver green, comparison at MAE ≈ machine epsilon
(`ALL PASS`), and the permanent snapshot `out/jax/<target_proc>.validated.py`
present. If any of these fail, the stage must not start — hardware fixes on a
numerically broken translation would mask the bug.

**Staging invariant.** `out/jax/<target_proc>.validated.py` is never edited; it
stays as a permanent recovery snapshot and as the baseline for byte-level
change detection. The loop edits **only** `[profiler].staged_code`, and
`out/jax/` stays untouched until the final re-validated result is copied back.

**Re-validation invariant.** Any iteration that changed the staged code must
pass the full validator chain (completeness → lint → semantic audit gate → runtime → driver → comparison) before
the run may be declared complete; the three batch steps are a strict
sequential pipeline. If no fix was applied, the staged file is byte-identical
to the validated translation (`staged_code_sha256` unchanged) and
re-validation is skipped as redundant.

**Reporting invariant.** `out/reports/profile/profile_report.md` is written on
every terminal outcome — clean first-iteration success, converged-after-fixes,
or abort. It is the durable record.

**Mandatory post-convergence pass.** After convergence, the nsys job is
submitted once and attached via `diagnose.py --nsys-json`; it populates the
informational `hardware_corroboration` block without affecting any check.
Its numbers feed the headroom analysis (PROFILE_WORKFLOW.md Step 7 and
`headroom_patterns.md`) — the cost ladder and device-idle fraction that
quantify what optimization opportunity remains beyond `production_ready`.

---

## 7. Inputs and outputs

### Inputs (consumed)

| Input | Source | Used by |
|---|---|---|
| `out/jax/<target_proc>.validated.py` | Translator stage | Staged → `[profiler].staged_code` |
| Any scalar init translation | Translator stage | Staged unchanged into `[profiler].out_dir` |
| Translator validator results (`out/lint`, `out/validation`, driver) | Translator stage | Hand-off gate check |
| **Real driver state, tiled to ncol** | Captured + tiled by `[profiler].inputs_script` (dispatched via `profiler_inputs.py`) | Both profiling passes |

The profiler does **not** use synthetic inputs. `profiler_inputs.py` is a
thin, driver-agnostic dispatcher — it handles staged-code loading and forwards
`capture_state`/`tile_state` to `[profiler].inputs_script`, a hand-authored
per-project file (like `[driver].script`) living under the project's own
`out/profiled/`. It captures one realistic bridge call from the driver and
expands it to `[profiler].ncol` (and `[profiler].ncol_check`) columns. See
`PROFILE_WORKFLOW.md` for the `capture_state`/`tile_state` contract.

### Per-iteration outputs — `<[profiler].out_dir>/iteration_<N>/`

| File | Producer | Consumed by |
|---|---|---|
| `<target_proc>_trace.json` | trace job (loop) | `diagnose.py --trace-json` (the single gating input; includes the `hlo` block) |
| `<target_proc>_code_info.json` | trace job (loop) | `diagnose.py --code-info` |
| `<target_proc>_hlo.txt` | trace job (loop) | parsed into the `hlo` block; manual inspection |
| `jax_trace/plugins/profile/<ts>/<host>.trace.json.gz` | trace job (loop) | Manual inspection (ui.perfetto.dev) |
| `jax_trace_check/...` | trace job (loop) | the second-ncol scaling measurement |
| `<target_proc>_diagnosis.json` | `diagnose.py` | Phase C / the fix LLM |
| `<target_proc>.nsys-rep` | nsys job (mandatory, post-convergence) | Manual inspection (`nsys-ui`) |
| `<target_proc>.sqlite` | nsys job | nsys queries |
| `<target_proc>_cuda_{gpu_kern,api}_sum.json` | nsys job | merged into `<target_proc>_stats.json` |
| `<target_proc>_stats.json` | nsys job | `diagnose.py --nsys-json` (informational `hardware_corroboration` only) |

### Stage-level outputs — `<[profiler].out_dir>/`

| File | Description |
|---|---|
| `<target_proc>.py` | Final hardware-efficient JAX translation (the deliverable) |
| `<init_proc>.py` | Verified init code, passed through unchanged |
| `profile_state.json` | Resume state — status, iteration, job ids, fix log |
| `out/reports/profile/profile_report.md` | Final results report |

### The diagnosis JSON

`<target_proc>_diagnosis.json` has this shape:

```json
{
  "production_ready": false,
  "worst_severity": "critical",
  "summary": {
    "trace_metrics": { "while_iterations": ..., "while_iterations_scaling": ...,
                       "hlo": { "ops": { "scatter": 0, "gather": 4, "...": 0 }, "...": 0 },
                       "gpu_exec_ms": ..., "dispatch_overhead_ms": ... },
    "code_info":     { "ncol": ..., "jax_op_count": ..., "output_array_count": ... },
    "issue_count":   2,
    "advisory_count": 0
  },
  "hardware_corroboration": null,
  "actionable_issues": [ { "metric": "...", "severity": "...", "message": "...", "fix": "..." } ],
  "advisories":        [ ... ],
  "confirmations":     [ ... ],
  "llm_feedback":      "Prioritized block — paste straight into the fix prompt."
}
```

`hardware_corroboration` is `null` in the loop; it is populated only by the
optional post-convergence re-run with `--nsys-json`, and never affects any
check. `advisories` flag ambiguity (non-gating); only `actionable_issues`
drive a fix.

---

## 8. The PBS jobs

The GPU jobs run on the cluster named in each `pbsJobs/*.sh` (queue, account,
walltime, and conda env are set inside those scripts, per `[hpc]`).

| Job | Submit with | Writes |
|---|---|---|
| `<[profiler].profile_trace_job>` (the loop job) | `qsub -v ITERATION=<N> ...` | `<target_proc>_trace.json`, `<target_proc>_code_info.json`, `<target_proc>_hlo.txt`, raw Perfetto traces |
| `<[profiler].profile_nsys_job>` (optional, post-convergence) | `qsub -v ITERATION=<N_final> ...` | nsys-rep, sqlite, `*_stats.json` |
| `pbsJobs/jax_gpu_runtimevalid.sh` | `qsub ...` | `out/validation/*_runtime.json` |
| `<[hpc].driver_job>` | `qsub ...` | driver output |
| `pbsJobs/jax_gpu_compvalues.sh` | `qsub ...` | comparison output |

Optional override `qsub -v` keys: `NCOL` (default `[profiler].ncol`),
`NCOL_CHECK` (trace scaling check, `[profiler].ncol_check`, 0 disables),
`N_CALLS` (traced calls, default 10), `CODE_PATH`.

**Ordering rules:**
- The loop has one profile job per iteration; nothing to coordinate.
- The three re-validation jobs are a strict pipeline — submit one, wait for it
  to leave `qstat`, verify its output, then submit the next. Optionally harden
  with `-W depend=afterok:<jobid>`.

A job leaving `qstat` only means it stopped: read its `out/jobs/<name>.o<job_id>` log for
tracebacks/OOM and confirm the result files are freshly written before
trusting them.

---

## 9. Diagnosis — the checks and why they are shaped this way

`diagnose.py` runs three checks against the **single** trace-pass metrics
JSON. Severities are ordered CRITICAL > HIGH > MEDIUM > LOW; two further
classes — ADVISORY and OK — never affect the verdict. The program is declared
`production_ready` exactly when no actionable (CRITICAL–LOW) issue remains.
Every metric is normalized by the right denominator (or made scale-relative by
the two-ncol measurement), so one profiling job per iteration is enough (no
multi-`ncol` job sweep needed).

| Check | Key metric | Normalized by | Catches | Severity |
|---|---|---|---|---|
| **Column loop** | `while_iterations` + `while_iterations_scaling`; `xla_event_count` as weak proxy | `ncol` / `ncol_ratio` | `lax.fori_loop` over columns instead of `jax.vmap` | **CRITICAL** (ambiguous mid-band: `advisory`) |
| **Fusion quality** | HLO scatter total (gates); unfused gathers (advisory); `ops_per_fusion` (report-only) | absolute HLO counts | Subset indexing (`.at[idx].set(...)`), fragmented fusion | **HIGH** scatter; `advisory` gather; OK characterization for `ops_per_fusion` |
| **D2H copies** | trace MemcpyD2H events per call | `output_array_count + 4` | Scalar readbacks inside JIT (`int(x)`, `.item()`) | **LOW** |

### Column-loop structure (CRITICAL)

With two-size data, a while-iteration scaling ≤ 1.5 confirms column
parallelism; a scaling ≥ 75% of the column ratio convicts a sequential column
loop, with the prescribed repair being restructuring to `jax.vmap` (or an
equivalent batched formulation) with the subcycling `while` inside the
per-column function. With single-size data only, the absolute count is used
(≥ 0.9 × ncol convicts; ≤ 2 confirms), and the ambiguous middle band is
deliberately an advisory: at one problem size it is indistinguishable from
healthy subcycling.

### Fusion quality (HIGH)

The check is deliberately **asymmetric** between the two irregular-access
instructions, reflecting their cost asymmetry. Any `scatter` in the compiled
HLO — counted in total, because XLA wraps `.at[indices].set(...)` in an input
fusion whose *root* is still the scatter, executing its irregular writes on
every call — gates at HIGH, with the repair being full-array computation
masked by `jnp.where`. `gather` never gates: an *unfused* gather (entry-level
or fusion-root) is surfaced as an advisory, while gathers *interior* to fusion
bodies are reported as confirmations — they are indexed reads the compiler
successfully absorbed into larger kernels, and are the expected compilation
product of `jnp.clip`/`jnp.where` stencil boundary handling. An
instructions-per-fusion statistic characterizes kernel packing but is
report-only: the observed cross-model spread is fully explained by the gating
scatter check, so no independent threshold is imposed.

### Host–device traffic (LOW)

Device-to-host copy events per call are compared against the expected readback
volume (output-array count plus a small scalar allowance, counted statically
from the source). A ratio above 3× indicates scalar readbacks inside the JIT
boundary — typically `int(...)`/`float(...)` on traced values — with the
repair being to move host conversions outside the compiled region.

### Design principles

1. **No fixed magic numbers.** Every threshold is a ratio or a structural
   zero/one (zero scatters; scaling ≈ 1 vs ≈ the column ratio), so the
   criteria transfer unchanged across models, schemes, and problem sizes — and
   one profiling iteration is sufficient, since the signals are
   scale-invariant or resolved by the in-job two-size check.
2. **Ambiguity never gates.** Signals that cannot name a fixable defect are
   advisories, preventing the bounded repair loop from spending its budget on
   possibly healthy code.
3. **Agreement between instruments guards the green light.** The strongest
   pass ("no loop at all") fires only when the trace shows no while-body
   executions **and** the compiled HLO contains zero `while` instructions, so
   a trace-parser regression cannot convert every diagnosis into a false pass.

Synchronization metrics (`sync_calls`, `sync_avg_us`, `sync_to_compute_ratio`)
appear only in the optional post-convergence `hardware_corroboration` block —
**reported, never scored**: their root cause is XLA's `while_loop` dispatch
model, not a translation defect, so they do not gate `production_ready`.

The diagnoser emits machine-readable issues (severity, message, repair
instruction, evidence) plus a consolidated natural-language `llm_feedback`
block constructed for direct inclusion in the repairing LLM's prompt.

---

## 10. The repair loop and correctness preservation

When the diagnosis is not clean, the LLM applies the **highest-severity repair
first**, directly to the staged working copy. The repair is sized to the
defect — a sequential column loop requires a genuine restructure, not a
symptomatic patch — but scoped to the diagnosed issue, so the next profile can
attribute cause and effect. Every edit preserves the LLM-version header and
bumps the `Iteration:` line, and is logged both in `profile_state.json →
fix_attempts` and in `out/issues/fix_log.md`.

Because structural profiling is **blind to numerical regressions by
construction**, every repair is followed by the framework's full validation
chain — lint, runtime validation, end-to-end driver, and tolerance comparison
against the Fortran reference, executed as sequentially gated batch jobs —
before the run may be declared complete. A repair therefore converges only if
it simultaneously **profiles clean and validates clean**.

The loop is bounded at **five repair attempts per (file, failing check)**;
validation failures count against the same budget. On exhaustion the run
aborts with a durable report rather than iterating indefinitely.

In the **zero-repair case**, a cryptographic hash comparison against the staged
baseline certifies that the profiled artifact is byte-identical to the
validated translation, so the original validation results transfer verbatim
and re-validation is skipped.

All state transitions (staging, job submissions, diagnoses, repairs,
re-validations) are journaled to the persistent state file, making the loop
resumable across scheduler queue delays and assistant sessions.

---

## 11. Typical repair patterns

The three checks map onto a small, recurring set of defects. These are the
patterns the loop actually encounters, independent of scheme or model:

| Diagnosed signal | Underlying defect | Repair |
|---|---|---|
| `while_iterations_scaling` ≈ column ratio | The translation kept the Fortran column loop as `lax.fori_loop`/`scan` over the column axis | Restructure to whole-array ops or `jax.vmap`, with any physics subcycle *inside* the per-column function |
| Nonzero HLO `scatter` | Subset indexing — `x.at[idx].set(...)` on a conditional subset | Compute the full array and select with `jnp.where` (masked whole-array form) |
| D2H copies ≫ output arrays | `int(...)`, `float(...)`, `.item()`, or Python `if` on a traced value inside the jitted region | Hoist host conversions and host-side branching out of the compiled region; keep the jitted body pure math |
| Unfused gather (advisory) | Stencil indexing the compiler could not absorb | Usually benign; investigate only if it accompanies a gating issue |

Repair policy detail for the first three lives in
`workflow_translator/prompt_policies/jax_vectorization_rules.md` and
`orchestration_boundary_rules.md`.

### What a converged run looks like

A translation that needs no repair shows a recognizable signal shape, and it is
worth knowing so the loop is not "fixed" into a worse state:

- `while_iterations` small and **constant** across the two column counts
  (scaling ≈ 1) — or no `while` at all in both trace and HLO. A masked
  whole-array `lax.while_loop` formulation is the healthy form: it compiles to
  a single XLA `while` whose iteration count is column-independent, and its
  absolute count is the physics' substep count, not a defect.
- **Zero** scatters in the compiled HLO; any gathers present are *interior* to
  fusion bodies (the expected product of clipped/masked stencil indexing).
- Device-to-host traffic matching one readback of the output arrays.
- A high instructions-per-fusion figure — report-only, but a fragmented value
  is normally the shadow of a scatter the gating check already caught.

In that case the loop terminates at iteration 1, the hash check certifies the
artifact unchanged, re-validation is skipped, and the report is still written.

---

## 12. State, resume, and the abort gate

`<[profiler].out_dir>/profile_state.json` is the **resume backbone**. PBS jobs
can sit in the queue across assistant sessions, so the state file is updated at
every checkpoint (stage, submit, pause, diagnose, fix, re-validate, report).

At the start of any invocation:
- **State file exists** → resume: report `status`, `iteration`, last
  `completed_steps`, `next_steps`, and continue. Do not restart at iteration 1.
- **No state file** → fresh run: verify the hand-off gate and stage.

The recognized status values are tabulated in `PROFILE_WORKFLOW.md`.

**Abort gate:** the loop allows at most **5 fix attempts** on the same failing
check. On exhaustion the workflow stops, sets
`status = aborted_max_fix_attempts`, and still writes `profile_report.md` —
that report is then the only artifact explaining why.

---

## 13. End state

A successful run ends with:
- `<[profiler].out_dir>/profile_state.json` → `status = "complete"`
- the latest `<target_proc>_diagnosis.json` → `"production_ready": true`
- the final validation re-check green (lint 100, runtime 0-failures, driver
  outputs present, comparison MAE ≈ machine epsilon + `ALL PASS`)
- `[profiler].staged_code` — the final hardware-efficient translation
- `out/reports/profile/profile_report.md` — the durable results report

---

## 14. Pointers

- **Pipeline entry point / stage sequencing:** `../ORCHESTRATOR.md`.
- **Authoritative playbook for this stage:**
  `workflow_profiler/PROFILE_WORKFLOW.md` (every operating rule, the
  quick-command reference, recognized status values).
- **Upstream stages:** `workflow_bridge/BRIDGE_WORKFLOW.md`,
  `workflow_translator/TRANSLATE_WORKFLOW.md`.
- **Shared validators:** `validation/VALIDATION_MANUAL.md`.
