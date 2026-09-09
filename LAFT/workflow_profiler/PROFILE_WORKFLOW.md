# Profiler Workflow — GPU structure profiling and fix loop

You are profiling the project's translated JAX code on the GPU, diagnosing
structural problems (sequential column loops, missing vectorization, poor
fusion, excess device-to-host traffic), and applying targeted fixes until the
diagnosis is `production_ready`. `production_ready` is a **floor** ("no
pathological structure"), not a ceiling: the campaign always ends with a
mandatory nsys hardware pass (Step 6) and a headroom analysis (Step 7) that
quantify what optimization opportunity remains and rank it in the report.
This document is **codebase-agnostic**: every
project value comes from `config/project.toml` — chiefly the `[profiler]`
section (target procedure, staged code path, ncol points) plus `[hpc]` and
`[driver]` for the re-validation chain. Phase C (the fix) is always an
in-context edit; this document has no PBS-job fix path.

**Working directory: the project root** (the directory containing `config/`).
All PBS jobs are submitted from there.

For Kessler the target is `kessler_run`. For a scheme whose main routine is
a host-side orchestrator over multiple internal jitted stage cores, the
profiled call is still the procedure's **bridge call** (the production
entry), so the trace covers the host orchestration and all stage kernels
together.

Hand-off gate: only enter this workflow when the translation workflow
(`workflow_translator/TRANSLATE_WORKFLOW.md`) is complete — lint 100%,
runtime validation, driver, and comparison all green.

---

## Inputs: real driver state, tiled to ncol

The profiler does not use synthetic inputs. `workflow_profiler/profiler_inputs.py`
is a thin, driver-agnostic dispatcher — it only handles staged-code loading
(`load_staged_bridge`) and forwards `capture_state`/`tile_state` to
**`[profiler].inputs_script`**, a hand-authored per-project file (like
`[driver].script` / `[comparison].script`) living under the project's own
`out/profiled/`. *How* to capture one realistic bridge call from a driver,
and *how* to expand it to ncol columns, is driver-shaped, not
framework-shaped: a single-column time-stepping driver (capture at a
simulated minute, replicate one column to ncol) and a single-shot
real-multi-column driver (capture the one call, cyclically tile the real
columns to ncol) need genuinely different logic, not a config flag bolted
onto one "generic" implementation. `inputs_script` must define exactly:

```python
def capture_state(force: bool = False) -> dict: ...   # one call's kwargs, cached
def tile_state(state: dict, ncol: int, seed: int = 42) -> dict: ...
```

The driver file itself is never modified by either function. See an
existing project's `inputs_script` (`kessler/out/profiled/kessler_profiler_inputs.py`)
for a full example before writing a new one.

If the driver or its translation changes in a way that alters the state,
delete the cache (rebuilt on the next run), or re-capture explicitly:
`python workflow_profiler/profiler_inputs.py --force`.

## Why two ncol values

`diagnose.py` classifies structure by how metrics **scale** with ncol, not by
absolute values. `while_iterations` measured at `[profiler].ncol` and
`[profiler].ncol_check`: scaling ~1 means sequential loops are over
levels/time (vectorized over columns — fine); scaling ~ncol-ratio means a
hidden sequential column loop (production blocker). One ncol point cannot
distinguish these.

---

## Step 0 — Check for a resume checkpoint (MANDATORY first action)

Read `<[profiler].out_dir>/profile_state.json`.

- **File exists:** resume from its `status`, `iteration`, and `next_steps`.
  Do not restart unless the user explicitly asks.
- **File absent:** fresh run; create it at first staging (state helper below).

State helper (records status, PBS job ids, expected logs, next steps):

```bash
python workflow_profiler/profile_workflow_state.py --status <status> [options]
```

## Step 1 — Stage the working copy (iteration 1)

Copy the validated translation to the staged path and record its hash:

```bash
cp out/jax/<target_proc>.py <[profiler].staged_code>
python workflow_profiler/profile_workflow_state.py \
  --status staged --iteration 1 --hash-staged-code \
  --complete-step "staged <[profiler].staged_code> for iteration 1"
```

The loop edits ONLY the staged copy. `out/jax/` stays untouched until the
final re-validated fix is copied back (Step 5).

## Step 2 — Phase A: the profiling job (one PBS job per iteration)

```bash
qsub -v ITERATION=<N> pbsJobs/jax_gpu_profile_trace.sh
python workflow_profiler/profile_workflow_state.py \
  --status waiting_for_profile --iteration <N> \
  --job-kind profile_trace --job-id <job_id> \
  --complete-step "submitted jax_trace profile iteration <N>"
```

The job runs the staged code inside `jax.profiler.trace` at both ncol points,
collects the compiled HLO of every jitted module (XLA dump), and writes to
`<out_dir>/iteration_<N>/`:

- `<proc>_trace.json` — trace metrics + `hlo` block + `while_iterations_scaling`
- `<proc>_code_info.json` — static counts
- `<proc>_hlo.txt` — concatenated optimized-HLO modules
- `jax_trace*/.../*.trace.json.gz` — raw Perfetto traces

**PBS freshness rule (MANDATORY):** a job leaving `qstat` only means it
stopped. Read the `out/jobs/<name>.o<job_id>` log for tracebacks/OOM and confirm the result
files are freshly written before trusting them.

## Step 3 — Phase B: diagnose (local, no PBS)

```bash
python workflow_profiler/diagnose.py \
  --trace-json <it_dir>/<proc>_trace.json \
  --code-info  <it_dir>/<proc>_code_info.json \
  --out        <it_dir>/<proc>_diagnosis.json
```

`diagnose.py` is the single source of truth: every gate reads its
`production_ready` and `checks` output. Do not re-derive **verdicts** from
the raw trace by hand — interpreting the trace for remaining headroom is
required, but it happens in Step 7, after the gates. Record the outcome:

```bash
python workflow_profiler/profile_workflow_state.py \
  --status diagnose_passed|diagnose_failed --iteration <N> \
  --complete-step "diagnose iteration <N>: <verdict>"
```

- `production_ready: true` → Step 5 (re-validation, only needed if the loop
  changed the staged code; the recorded `staged_code_sha256` tells you),
  then Step 6 (nsys pass) and Step 7 (headroom analysis + report).
- `production_ready: false` → Step 4.

## Step 4 — Phase C: fix loop (in-context edit)

Apply a **minimal, targeted** fix to the staged copy for the specific check
that failed (the diagnosis names the failing metric and its evidence). Then:

```bash
python workflow_profiler/profile_workflow_state.py \
  --status staged --iteration <N+1> --bump-fix-attempts \
  --complete-step "fix iteration <N>: <one-line change summary>" \
  --complete-step "staged <[profiler].staged_code> for iteration <N+1>"
```

Also append the structured entry to `fix_attempts` in the state JSON (proc,
failed check, evidence, file, change summary) and log the cycle in
`out/issues/fix_log.md`. Return to Step 2 with ITERATION=<N+1>.

**Fix budget: 5 attempts per (file, check)** — five tries at the same failing
check. On exhaustion: set `status=aborted_max_fix_attempts`, still write the
report (Step 6), and tell the user what remains.

Common fix patterns: replace a `lax.fori_loop`/`scan` over the column axis
with vectorized ops or `vmap`; hoist device-to-host conversions out of the
per-call path; split host-side branching from jitted math (see
`workflow_translator/prompt_policies/jax_vectorization_rules.md` and
`orchestration_boundary_rules.md`).

## Step 5 — Phase D: re-validate after any substantial fix (MANDATORY)

A profiler fix is a code change to the translation: before re-profiling or
declaring victory, copy the staged file back and run the full chain —
completeness → lint → semantic audit (gate) → runtime → driver → comparison
(the runtime PBS script and the state tool refuse to run while
`workflow_translator/audit_gate.py` is closed, i.e. until the audit has been
re-run on the changed file with 0 unwaived FAIL — TRANSLATE_WORKFLOW §3.5):

```bash
cp <[profiler].staged_code> out/jax/<target_proc>.py
python workflow_translator/phase04_04_completeness_check.py   # Step 2.5 stop rule (scaffold = STOP)
python validation/phase05_01_lint_translation.py          # expect 100%
python workflow_translator/phase05_01b_semantic_audit.py  # Step 3.5 — a code change re-closes the gate
python workflow_translator/audit_gate.py                  # must print OPEN, else runtime is REFUSED
qsub pbsJobs/jax_gpu_runtimevalid.sh                 # expect all procs pass
qsub <[hpc].driver_job>                              # expect driver output verified
qsub pbsJobs/jax_gpu_compvalues.sh                   # expect ALL PASS
```

Record each with `--job-kind revalidation_runtime|revalidation_driver|
revalidation_comparison`. If the staged code is hash-identical to the
validated baseline (`staged_code_sha256`), re-validation may be skipped.
Either way, continue to Step 6.

## Step 6 — Phase E: nsys hardware pass (MANDATORY, once per converged code)

The trace job measures the program JAX dispatches; nsys measures what the
hardware actually does — device kernel times, launch and memcpy counts, and
the idle gaps between them. The headroom analysis (Step 7) needs those
numbers, so this pass is part of every campaign: run it once, after the
loop converges (after re-validation whenever the loop changed code).

```bash
qsub -v ITERATION=<final_N> pbsJobs/jax_gpu_profile_nsys.sh
python workflow_profiler/profile_workflow_state.py \
  --status waiting_for_nsys --iteration <final_N> \
  --job-kind profile --job-id <job_id> \
  --complete-step "submitted nsys hardware pass iteration <final_N>"
```

The PBS freshness rule (Step 2) applies. Then attach the stats to the final
diagnosis and record the outcome:

```bash
python workflow_profiler/diagnose.py \
  --trace-json <it_dir>/<proc>_trace.json \
  --code-info  <it_dir>/<proc>_code_info.json \
  --nsys-json  <it_dir>/<proc>_stats.json \
  --out        <it_dir>/<proc>_diagnosis.json
python workflow_profiler/profile_workflow_state.py \
  --status nsys_passed --iteration <final_N> \
  --complete-step "nsys pass iteration <final_N>: stats attached"
```

The `hardware_corroboration` block stays informational — it never affects
any check or `production_ready`. Verdicts gate; hardware numbers feed the
headroom analysis.

## Step 7 — Headroom analysis + profile report

Before writing the report, quantify what remains, reading the trace, the
nsys stats, and the staged **code** together (verdicts still come only from
`diagnose.py`; this step interprets, it never gates).

Build the **cost ladder** at `[profiler].ncol`, per profiled call:

| layer | source |
|---|---|
| end-to-end call | timing median from the trace pass |
| data movement (transposes + H2D + D2H) | nsys memcpy sums; bridge phase timings if measured |
| host dispatch residual | end-to-end − data movement − device kernel time |
| device kernel time | nsys `cuda_gpu_kern_sum` total ÷ profiled calls |

plus the **device-idle fraction** (1 − device-busy ÷ wall) and, per call:
kernel launches, D2D memcpys, jitted entry points invoked, and arguments
per jitted call. For each of the top cost layers, name the code construct
responsible (file:line) — a number without its cause is not a finding.

The ladder also *hints* at the bridge-contract choice — when data movement
dominates device kernel time, contract 1 cannot win end-to-end — but the
verdict on `[driver].contracts` belongs to the project's own benchmark, not
to this report: state the hint at most as prose, emit no
`recommended_contract` output, and never fold it into `production_ready`
(kernel efficiency and contract choice are independent axes).

Then apply `workflow_profiler/headroom_patterns.md` (signature → hypothesis
→ candidate fix) and write **3–5 ranked recommendations**, each stating:
the mechanism, the expected saving **bounded by the layer it lives in**
(never claim more than the layer contains), the validation cost (a code fix
re-triggers the full completeness → lint → semantic audit (gate) → runtime → driver → comparison chain), and the
cheapest measurement that would confirm or kill it.

Write `out/reports/profile/profile_report.md` (consolidated reports tree:
`out/reports/<workflow>/`): run date, LLM name + exact
model ID, iterations table (per iteration: while_iterations + scaling,
xla_event_count, d2h copies, timing medians, verdict), fixes applied,
final diagnosis, the nsys hardware numbers, the cost ladder, and the ranked
headroom recommendations.
Write the report even for a clean first-pass `production_ready` — a clean
pass with a 90% device-idle fraction is exactly what the ladder exists to
expose.

---

## Recognized status values

| Status | Meaning |
|--------|---------|
| `staged` | Working copy staged; submit profile job |
| `waiting_for_profile` | PBS trace job queued/running |
| `profile_passed` | Artifacts verified; run diagnose |
| `diagnose_failed` | Apply a fix (Step 4) |
| `diagnose_passed` | production_ready; re-validate if code changed, then nsys pass |
| `waiting_for_revalidation_runtime` / `_driver` / `_comparison` | Chain gates |
| `revalidation_passed` | Copy-back done; run the nsys pass (Step 6) |
| `waiting_for_nsys` | nsys hardware-pass PBS job queued/running |
| `nsys_passed` | Hardware stats attached; headroom analysis + report (Step 7) |
| `complete` | Report written |
| `aborted_max_fix_attempts` | Budget hit; report written |

## Quick command reference

```bash
# state capture (usually automatic; --force to re-capture)
python workflow_profiler/profiler_inputs.py --ensure

# profile iteration N (from the project root)
qsub -v ITERATION=N pbsJobs/jax_gpu_profile_trace.sh

# diagnose
python workflow_profiler/diagnose.py \
  --trace-json out/profiled/iteration_N/<proc>_trace.json \
  --code-info  out/profiled/iteration_N/<proc>_code_info.json \
  --out        out/profiled/iteration_N/<proc>_diagnosis.json

# nsys hardware pass (mandatory, once, post-convergence)
qsub -v ITERATION=<final_N> pbsJobs/jax_gpu_profile_nsys.sh
# ... then re-run diagnose with --nsys-json out/profiled/iteration_<final_N>/<proc>_stats.json

# inspect state
cat out/profiled/profile_state.json
```

Historical note: the report-only threshold intuitions in `diagnose.py` were
calibrated on the Kessler scheme at ncol=1000. Treat a new scheme's first
runs as its own baseline rather than comparing against Kessler numbers.
