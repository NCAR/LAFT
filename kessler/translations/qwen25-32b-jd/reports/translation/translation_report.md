# Translation report — laft-kessler, Qwen2.5-32B — 2026-04 JD-data translation REUSED under the current bridge

| | |
|---|---|
| run date | 2026-09-04 |
| LLM | Qwen2.5-32B (file header "Translated by: Qwen2.5-32B"; the 2026-04 script's default model path is `Qwen2.5-Coder-32B-Instruct`, vLLM offline inference, `max_model_len` 24576). The copied file already carries the 2026-04 run's own validation fix ("Fixed by: Claude Opus 4.7 — fix attempt 1: subcycle time_counter ordering") |
| translator mode | `source = "in_context"` — **no re-translation**: `out/jax/kessler_init.py` and `kessler_run.py` are verbatim copies of `_2sd_exp_JDdata/qwen/jax/` (2026-04 JD-data run) with the mandated version header; the old `wrappers.py` was not copied (bridges are regenerated) |
| constraint | loop/vectorization **structure FROZEN** (the paper compares per-model design): interface-level repairs only; a structural fix would have been a HARD STOP for the user |
| workflow | `workflow_translator/TRANSLATE_WORKFLOW.md` |
| driving agent | Claude Fable 5.1 (`claude-fable-5-1`) |
| final status | **`complete`** — comparison ALL PASS, host mode (`[driver].contracts = [1]`) |
| `fix_attempts_used` | **1** (runtime, one-level harness guard — a static guard, not structure) |

## 1. Translation plan (Step 1c)

| # | procedure | scalar_only | jax_required | passes | calls |
|---|---|---|---|---|---|
| 1 | kessler_init | yes | no | 1 (plain Python) | — |
| 2 | kessler_run | no | yes | 4 (full JAX) | — |

Passes are those of the 2026-04 run; nothing was regenerated in this run.

## 2. What this run is

Fourth and last reuse of the 2026-04 JD-data experiment under the current LAFT pipeline,
after Sonnet 4.6 (`translations/claude-sonnet46-jd/`, 1 guard fix), Gemini 3.1 Pro
(`translations/gemini31pro-jd/`, 0 fixes) and GPT-5.4 Thinking
(`translations/gpt54thinking-jd/`, 1 guard fix); `out/` was cleaned with
`clean_AI_results.sh` before this run. The Qwen translation was produced under the
pre-path-C bridge; here it goes through regenerated prompts/wrappers/bridges, the §2.5
completeness check, lint, the gated semantic audit, GPU runtime validation, the bridge suite
with translation-dependent tests, and the driver + comparison — without altering the model's
design.

Design (unchanged, from the file): **serial `lax.fori_loop` over columns** (the only one of
the four JD translations without `jax.vmap`); inside each column the level loops are
vectorised as `.at[klevs].set(...)` scatter updates on the full (nz, ncol) arrays and on
per-level workspaces (`r`, `rhalf`, `velqr`, `sed`, `pc`) threaded through the loop carry;
per-column CFL `dt0` via `jnp.min(jnp.where(...))` over the interior range; `lax.while_loop`
sub-cycle carrying `dt0`, `time_counter`, `precl_acc`; no `dt <= 0` branch of its own (a
nonpositive `dt` surfaces through the CFL check as `errflg = 1`); the sub-cycle is not skipped
on `errflg`. This is the 2026-04 design as validated then; see the fix log for the two
"noted, not changed" points.

## Translation statistics

### Table 1 — Four-pass translation statistics (FROZEN)

Frozen at **2026-09-04T10:48:32** · translator **in_context** (Qwen2.5-32B (Qwen2.5-Coder-32B-Instruct via vLLM; 2026-04 JD-data run, reused)) · 2 procedures.

**This table is immutable.** It records the initial four-pass output before any validation fix. Do not update it — later repairs belong in Table 2.

> Token columns are `n/a`: the translation was authored in-context, so no API reported token counts. Prompt and output **characters** are exact and stand in for size.

> Wall-clock source: elapsed between per-procedure state updates (includes pauses).

| module / procedure | passes | Fortran lines (code) | JAX lines (code) | JAX/F | prompt chars | JAX chars | prompt tok | output tok | wall-clock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **kessler** | | | | | | | | | |
| &nbsp;&nbsp;kessler_init | 1 | 17 (12) | 39 (27) | 2.29x | 7,807 | 1,454 | n/a | n/a | n/a |
| &nbsp;&nbsp;kessler_run | 4 | 210 (129) | 121 (76) | 0.58x | 85,360 | 8,818 | n/a | n/a | 0s |
| &nbsp;&nbsp;*subtotal (2 procs)* | | *227* | *160* | | *93,167* | *10,272* | *n/a* | *n/a* | *n/a* |
| **TOTAL (2 procs)** | | **227** | **160** | | **93,167** | **10,272** | **n/a** | **n/a** | **n/a** |

### Table 2 — Validation-phase fix statistics (LIVE)

1 fix attempt(s) across 1 procedure(s). Re-rendered on every fix; the JAX-line column shows the frozen four-pass value versus the current file.

| module / procedure | fixes | failed stage(s) | Fortran lines | JAX lines frozen → now | fix tokens | fix time | last fixed |
|---|---:|---|---:|---:|---:|---:|---|
| **kessler** | | | | | | | |
| &nbsp;&nbsp;kessler_run | 1 | runtime | 210 | 121 → 123 (+2) | n/a | 3.0m | 2026-09-04T10:50:26 |

Wall-clock in Table 1 is meaningless for a copied translation (the "0s" is the interval
between the two copy timestamps); tokens are `n/a` (in-context / copy).

## 4. Completeness (§2.5) and semantic audit (§3.5)

# Translation completeness report (TRANSLATE_WORKFLOW Step 2.5)

| | |
|---|---|
| run | 2026-09-04T10:50:26.747821 |
| scope | all procedures |
| verdict | **PASS** — PASS 2, WARN 0, FAIL 0, MISSING 0 |
| rule | FAIL (scaffold phrase / commented-out callee) = translation STOPPED, `status aborted_scaffold`, no automatic retry — human decision |

| procedure | Fortran lines | JAX code lines | ratio | verdict | why |
|---|---|---|---|---|---|
| kessler_init | 13 | 23 | 1.769 | PASS |  |
| kessler_run | 173 | 78 | 0.451 | PASS |  |

Machine-readable: `out/reports/translation/completeness_check.json`

# Semantic audit report (TRANSLATE_WORKFLOW Step 3.5a)

| | |
|---|---|
| run | 2026-09-04T10:50:26.877612 |
| translator | Qwen2.5-32B (Qwen2.5-Coder-32B-Instruct via vLLM; 2026-04 JD-data run, reused) |
| procedures audited | 2 |
| verdict | **PASS** — 0 FAIL, 0 WARN, 1 waived |
| waivers file | `out/issues/semantic_audit_waivers.json` |
| gate | `workflow_translator/audit_gate.py` reads `out/issues/semantic_audit/_summary.json`; runtime validation (Step 4) requires 0 FAIL and this report newer than every `out/jax/<proc>.py` |

## Checks

| check | severity | status in this project | meaning |
|---|---|---|---|
| lut-indices | FAIL | n/a — `lut_indices` not in `[semantic_audit].enabled_checks` | translation uses LUT columns the Fortran never uses |
| lut-indices-unported | WARN | n/a — `lut_indices` not in `[semantic_audit].enabled_checks` | Fortran LUT columns the translation never fetches |
| calls-invoked | FAIL | ran | a procedure the Fortran calls is never invoked in the translation body (X( / X_core( / X_value( ) |
| labelled-blocks | WARN | ran | a Fortran labelled block (name: if/do/…) is not mentioned anywhere in the translation |
| update-terms | FAIL | n/a — `[semantic_audit].prognostic_vars` not set | a term of the Fortran `x = x ± (…)` prognostic update is absent from the translation's update of x |
| update-terms-extra | WARN | n/a — `[semantic_audit].prognostic_vars` not set | the translation's update of x has terms the Fortran's does not |
| direction-masks | WARN | ran | heuristic: `(x - y) * dir >= 0` mask orientation vs the Fortran `do k = A, B, ±dir` loop (only procedures with identifier-stride loops; exact check = bridge_test/test_direction_symmetry.py) |
| table-reads | FAIL | n/a — `[semantic_audit].table_readers` not set (no lookup tables) | lookup-table reader executed on the real file: column base slot, column count, leading-token skip and loop nesting must match the Fortran read statement |
| zero-forever | WARN | ran | Fortran computes it, translation only ever assigns zero (chained assignments split) |
| intent-out | WARN | ran | intent(out) argument never freshly assigned |
| missing-consts | WARN | ran | Fortran numeric literals absent from the translation |
| invented-consts | INFO | ran | translation literals absent from the Fortran |
| conventions | INFO | n/a — `level_conventions` not in `[semantic_audit].enabled_checks` | level/index convention claims for manual cross-check |

## Per procedure

| procedure | FAIL | WARN | waived | INFO |
|---|---|---|---|---|
| kessler_init | 0 | 0 | 0 | 0 |
| kessler_run | 0 | 0 | 1 | 1 |

## Findings (FAIL / WARN / waived)

| procedure | check | severity | name | detail |
|---|---|---|---|---|
| kessler_run | intent-out | WAIVED | `relhum` | Fortran intent(out) arg is only ever copied from its incoming value (or never assigned) — risk of accumulation across calls |

Machine-readable findings: `out/issues/semantic_audit/<proc>.json`, summary `_summary.json`.

## 5. Results

| procedure | lint | runtime GPU (7313968 → 7313978 after fix) | bridge suite CPU (7313969 → 7313979) | driver (7313985, host) | comparison (7313992) |
|---|---|---|---|---|---|
| kessler_init | 9/9 | imported, wrapper_ok, core_ok, wrapper_finite | — | init errflg 0 | — |
| kessler_run | 14/14 | **FAIL → PASS**: imported, wrapper_ok, core_ok, wrapper_finite, core_finite | 17/17 both rounds (translation-dependent 8/8) | PASS, errflg 0, all outputs finite | **ALL PASS** |

Comparison vs Fortran (128 cols × 56 levels, dt = 60 s, seed 0; `compare_fortran_jax.json` 2026-09-04 10:52):

| variable | MAE | RMSE | max abs err | rel MAE | within tol |
|---|---:|---:|---:|---:|---|
| theta (K) | 7.296e-16 | 6.440e-15 | 5.684e-14 | 2.54e-18 | yes |
| qv | 3.685e-19 | 1.869e-18 | 6.245e-17 | 3.12e-17 | yes |
| qc | 2.194e-19 | 9.559e-19 | 2.860e-17 | 3.39e-16 | yes |
| qr | 2.429e-18 | 3.624e-18 | 2.082e-17 | 1.53e-16 | yes |
| precl | 1.821e-20 | 2.835e-20 | 1.084e-19 | 1.30e-16 | yes |
| relhum (%) | 7.583e-15 | 1.786e-14 | 4.121e-13 | 9.51e-17 | yes |

Identical to the digit with the Gemini 3.1 Pro and GPT-5.4 reuses and the earlier
archived set; worst ULP 4221 on `qc` as for every model. Physics sanity 6/6
on both sides. Bitwise: 98.7 % of theta values bit-identical to Fortran.

## 6. Files generated

| file | what it is |
|---|---|
| `out/jax/kessler_init.py` | verbatim 2026-04 scalar-only init (+ version header) |
| `out/jax/kessler_run.py` | verbatim 2026-04 `kessler_run_core` (jit, static `ncol nz lyr_surf lyr_toa`) + host wrapper, **two static guard lines added by fix attempt 1**; header `# Pass: fix-attempt-1 (runtime, 2026-09-04) on final` |
| `out/jax/*.validated.py` | Step 7 snapshot after the comparison passed |
| `out/validation/*_runtime.json` | Step 4 results (PBS 7313978) |
| `out/reports/bridge/bridge_test_results.{json,xml}`, `bridge_report.md` | Step 4.5 (PBS 7313979) |
| `out/driver/*` | driver outputs (PBS 7313985) and comparison (PBS 7313992) |
| `out/reports/translation/pass_stats.json` (+ `.sha256`), `fix_stats.jsonl`, `_stats_tables.md` | Tables 1 and 2 |
| `out/reports/translation/completeness_report.md`, `semantic_audit_report.md` | §2.5 / §3.5 |
| `out/issues/fix_log.md` | gate trail + fix narrative |
| `out/jobs/73139{68,69,78,79,85,92}.desched1.OU` | PBS logs |

## 7. Fix log (`workflow_state.json → fix_attempts`)

| n | proc | failed stage | error | change |
|---|---|---|---|---|
| 1 | kessler_run | runtime (7313968) | `ValueError: zero-size array to reduction operation min which has no identity` — the harness feeds every integer scalar `DUMMY_N = 4`, so `lyr_surf = lyr_toa = 4` (one-level range) and the interior range of the CFL `jnp.min` is empty. Fortran's CFL loop is simply empty then (`dt0` stays `dt`). Real grids (bridge suite, driver) never hit it. Predicted from the kernel before the job; the 2026-04 harness evidently did not use a degenerate range (its record shows `core_ok = True`). | Static Python guard `if lyr_surf_idx != lyr_toa_idx:` around the two CFL min-reductions (initial `dt0` and the sub-cycle recompute); resolved at trace time, so the multi-level graph is unchanged. Serial column loop and level scatters untouched. Full chain re-run: lint 100 %, audit 0/0, runtime PASS, bridge 17/17, driver PASS, comparison ALL PASS. |

Noted, not changed (the user's call): (a) no `dt <= 0` branch in the core — a nonpositive `dt`
is flagged through the CFL check and the sub-cycle still executes one step before its
condition closes; the wrapper picks the Fortran message from the sign of `dt`; the bridge
test for this path passes numerically. (b) The sub-cycle is not skipped when `errflg != 0`
(no `lax.cond`); the flag is only reported. Both are the 2026-04 design as validated then.

## 8. What worked / what was tricky

- **The only defect was the harness's degenerate level range**, the same pitfall fresh
  translations in the working project hit in August 2026: vectorised sweeps
  over `lyr_surf..lyr_toa` are not empty-safe the way a Fortran `do` loop is. A static guard
  on the static level bounds is the whole fix.
- **Design outlier, same numbers.** This is the one JD translation with a serial column
  loop; it still reproduces the Fortran to the same digits as the three `vmap` designs.
  Performance, not accuracy, is where the design shows (Stage 3 / `_compare_results`).
- **Scheduler note:** `qstat <id>` again reported "Unknown Job Id" for running develop-queue
  jobs; the `out/jobs/<id>.desched1.OU` logs are the evidence.

**Next (Stage gate — the user decides):** archive under a NEW name; Stage 3 (profile) optional.
