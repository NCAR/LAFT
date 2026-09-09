# Translation report — laft-kessler, GPT-5.4 Thinking — 2026-04 JD-data translation REUSED under the current bridge

| | |
|---|---|
| run date | 2026-09-04 |
| LLM | GPT-5.4 Thinking (file docstrings: "Translated by GPT-5.4 Thinking"; no run log records an API model id — label per `[translator].llm_label`) |
| translator mode | `source = "in_context"` — **no re-translation**: `out/jax/kessler_init.py` and `kessler_run.py` are verbatim copies of `_2sd_exp_JDdata/gpt/jax/` (2026-04 JD-data run) with the mandated version header; the old `wrappers.py` was not copied (bridges are regenerated) |
| constraint | loop/vectorization **structure FROZEN** (the paper compares per-model design): interface-level repairs only; a structural fix would have been a HARD STOP for the user |
| workflow | `workflow_translator/TRANSLATE_WORKFLOW.md` |
| driving agent | Claude Fable 5.1 (`claude-fable-5-1`) |
| final status | **`complete`** — comparison ALL PASS, host mode (`[driver].contracts = [1]`) |
| `fix_attempts_used` | **1** (bridge suite, error-flag precedence — a guard, not structure; the same defect and fix as the Sonnet 4.6 reuse) |

## 1. Translation plan (Step 1c)

| # | procedure | scalar_only | jax_required | passes | calls |
|---|---|---|---|---|---|
| 1 | kessler_init | yes | no | 1 (plain Python) | — |
| 2 | kessler_run | no | yes | 4 (full JAX) | — |

Passes are those of the 2026-04 run; nothing was regenerated in this run.

## 2. What this run is

Third reuse of the 2026-04 JD-data experiment under the current LAFT pipeline, after Sonnet
4.6 (`translations/claude-sonnet46-jd/`, 1 guard fix) and Gemini 3.1 Pro
(`translations/gemini31pro-jd/`, 0 fixes); `out/` was cleaned with `clean_AI_results.sh`
before this run. The GPT translation was produced under the pre-path-C bridge; here it goes
through regenerated prompts/wrappers/bridges, the §2.5 completeness check, lint, the gated
semantic audit, GPU runtime validation, the bridge suite with translation-dependent tests,
and the driver + comparison — without altering the model's design.

Design (unchanged, from the file): `jax.vmap` over columns (`in_axes=1`) of a per-column
kernel; active level range as an index array (`level_indices`, `interior_indices`) with
gathers for the vectorised formulas and `.at[level_indices].set` scatters back to full-length
workspaces (`r`, `rhalf`, `pc`, `velqr`, `sed_local`, `relhum_c`, and the prognostic columns
in every sub-cycle step); per-column CFL `dt0` via `min(concat([dt], candidates))`;
`lax.while_loop` sub-cycle, skipped via `lax.cond` when the column's error flag is set;
`jnp.max` flag reduction. Structurally the same layout as the Sonnet 4.6 JD translation
(which also explains the identical defect).

## Translation statistics

### Table 1 — Four-pass translation statistics (FROZEN)

Frozen at **2026-09-04T10:38:32** · translator **in_context** (GPT-5.4 Thinking (2026-04 JD-data run, reused)) · 2 procedures.

**This table is immutable.** It records the initial four-pass output before any validation fix. Do not update it — later repairs belong in Table 2.

> Token columns are `n/a`: the translation was authored in-context, so no API reported token counts. Prompt and output **characters** are exact and stand in for size.

> Wall-clock source: elapsed between per-procedure state updates (includes pauses).

| module / procedure | passes | Fortran lines (code) | JAX lines (code) | JAX/F | prompt chars | JAX chars | prompt tok | output tok | wall-clock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **kessler** | | | | | | | | | |
| &nbsp;&nbsp;kessler_init | 1 | 17 (12) | 28 (17) | 1.65x | 7,807 | 936 | n/a | n/a | n/a |
| &nbsp;&nbsp;kessler_run | 4 | 210 (129) | 347 (271) | 1.65x | 85,360 | 20,221 | n/a | n/a | 0s |
| &nbsp;&nbsp;*subtotal (2 procs)* | | *227* | *375* | | *93,167* | *21,157* | *n/a* | *n/a* | *n/a* |
| **TOTAL (2 procs)** | | **227** | **375** | | **93,167** | **21,157** | **n/a** | **n/a** | **n/a** |

### Table 2 — Validation-phase fix statistics (LIVE)

1 fix attempt(s) across 1 procedure(s). Re-rendered on every fix; the JAX-line column shows the frozen four-pass value versus the current file.

| module / procedure | fixes | failed stage(s) | Fortran lines | JAX lines frozen → now | fix tokens | fix time | last fixed |
|---|---:|---|---:|---:|---:|---:|---|
| **kessler** | | | | | | | |
| &nbsp;&nbsp;kessler_run | 1 | bridge_suite | 210 | 347 → 351 (+4) | n/a | 2.0m | 2026-09-04T10:39:58 |

Wall-clock in Table 1 is meaningless for a copied translation (the "0s" is the interval
between the two copy timestamps); tokens are `n/a` (in-context / copy).

## 4. Completeness (§2.5) and semantic audit (§3.5)

# Translation completeness report (TRANSLATE_WORKFLOW Step 2.5)

| | |
|---|---|
| run | 2026-09-04T10:39:58.778496 |
| scope | all procedures |
| verdict | **PASS** — PASS 2, WARN 0, FAIL 0, MISSING 0 |
| rule | FAIL (scaffold phrase / commented-out callee) = translation STOPPED, `status aborted_scaffold`, no automatic retry — human decision |

| procedure | Fortran lines | JAX code lines | ratio | verdict | why |
|---|---|---|---|---|---|
| kessler_init | 13 | 13 | 1.0 | PASS |  |
| kessler_run | 173 | 269 | 1.555 | PASS |  |

Machine-readable: `out/reports/translation/completeness_check.json`

# Semantic audit report (TRANSLATE_WORKFLOW Step 3.5a)

| | |
|---|---|
| run | 2026-09-04T10:39:58.890661 |
| translator | GPT-5.4 Thinking (2026-04 JD-data run, reused) |
| procedures audited | 2 |
| verdict | **PASS** — 0 FAIL, 1 WARN, 1 waived |
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
| kessler_run | 0 | 1 | 1 | 1 |

## Findings (FAIL / WARN / waived)

| procedure | check | severity | name | detail |
|---|---|---|---|---|
| kessler_run | zero-forever | WARN | `relhum` | translation only ever assigns a zero constant, but the Fortran computes this variable |
| kessler_run | zero-forever | WAIVED | `precl` | translation only ever assigns a zero constant, but the Fortran computes this variable |

Machine-readable findings: `out/issues/semantic_audit/<proc>.json`, summary `_summary.json`.

The `zero-forever relhum` WARN is a checker false positive: `relhum_c` is
`jnp.zeros(nz).at[level_indices].set(qv/qvs*100)` — the scatter design the freeze protects.
Not waived, no action.

## 5. Results

| procedure | lint | runtime GPU (7313881 → 7313896 after fix) | bridge suite CPU (7313882 → 7313897) | driver (7313908, host) | comparison (7313910) |
|---|---|---|---|---|---|
| kessler_init | 9/9 | imported, wrapper_ok, core_ok, wrapper_finite | — | init errflg 0 | — |
| kessler_run | 14/14 | imported, wrapper_ok, core_ok, wrapper_finite, core_finite | 16/17 → **17/17** (translation-dependent 8/8) | PASS, errflg 0, all outputs finite | **ALL PASS** |

Comparison vs Fortran (128 cols × 56 levels, dt = 60 s, seed 0; `compare_fortran_jax.json` 2026-09-04 10:41):

| variable | MAE | RMSE | max abs err | rel MAE | within tol |
|---|---:|---:|---:|---:|---|
| theta (K) | 7.296e-16 | 6.440e-15 | 5.684e-14 | 2.54e-18 | yes |
| qv | 3.685e-19 | 1.869e-18 | 6.245e-17 | 3.12e-17 | yes |
| qc | 2.194e-19 | 9.559e-19 | 2.860e-17 | 3.39e-16 | yes |
| qr | 2.429e-18 | 3.624e-18 | 2.082e-17 | 1.53e-16 | yes |
| precl | 1.821e-20 | 2.835e-20 | 1.084e-19 | 1.30e-16 | yes |
| relhum (%) | 7.583e-15 | 1.786e-14 | 4.121e-13 | 9.51e-17 | yes |

Identical to the digit with the Gemini 3.1 Pro reuse and the earlier archived
set; worst ULP 4221 on `qc` as for every model (scheme cancellation). Physics sanity
6/6 on both sides. Bitwise: 98.7 % of theta values bit-identical to Fortran.

## 6. Files generated

| file | what it is |
|---|---|
| `out/jax/kessler_init.py` | verbatim 2026-04 scalar-only init (+ version header) |
| `out/jax/kessler_run.py` | verbatim 2026-04 `kessler_run_core` (jit, static `ncol nz lyr_surf lyr_toa`) + host wrapper, **one guard line changed by fix attempt 1**; header `# Pass: fix-attempt-1 (bridge suite, 2026-09-04) on final` |
| `out/jax/*.validated.py` | Step 7 snapshot after the comparison passed |
| `out/validation/*_runtime.json` | Step 4 results (PBS 7313896) |
| `out/reports/bridge/bridge_test_results.{json,xml}`, `bridge_report.md` | Step 4.5 (PBS 7313897) |
| `out/driver/*` | driver outputs (PBS 7313908) and comparison (PBS 7313910) |
| `out/reports/translation/pass_stats.json` (+ `.sha256`), `fix_stats.jsonl`, `_stats_tables.md` | Tables 1 and 2 |
| `out/reports/translation/completeness_report.md`, `semantic_audit_report.md` | §2.5 / §3.5 |
| `out/issues/fix_log.md` | gate trail + fix narrative |
| `out/jobs/73138{81,82,96,97}.desched1.OU`, `73139{08,10}.desched1.OU` | PBS logs |

## 7. Fix log (`workflow_state.json → fix_attempts`)

| n | proc | failed stage | error | change |
|---|---|---|---|---|
| 1 | kessler_run | bridge suite (7313882) | `test_error_handling_negative_dt`: `errflg = 2` instead of `1` for `dt = -60`. Top-level `errflg = where(dt <= 0, 1, 0)` correct; per-column `col_errflg = where(dt0 < 1e-12, 2, errflg)` overwrote it (`dt0 = min(dt, …)` negative). Fortran `return`s at the `dt <= 0` check first. Bug in the 2026-04 original (verbatim-copy check passed), masked then by the old bridge's wrapper-side guard the path-C bridge does not run. Predicted from the kernel before the job; the gate confirmed it. | Re-nested the `jnp.where` so a non-zero `errflg` is kept and code 2 is set only when `errflg == 0`. One guard, no structural change. Full chain re-run: completeness PASS, lint 100 %, audit 0 FAIL / 1 WARN, runtime PASS, bridge 17/17, driver PASS, comparison ALL PASS. |

Noted, not changed (the user's call): `errflg = 2` for bad time splitting where the Fortran uses
`1` for both branches (only `errmsg` differs). Untested; left minimal.

## 8. What worked / what was tricky

- **Same defect, same one-line fix as the Sonnet 4.6 JD translation.** Two of the three JD
  translations re-validated today (Sonnet, GPT) share the level-index scatter layout and the
  `lax.cond` skip, and both put the bad-time-split flag after the dt flag without preserving
  precedence. Gemini's outer-`lax.cond` bypass avoided it. The translation-dependent bridge
  tests (path C, no wrapper guard) are what surface this class of defect.
- **Accuracy unchanged by the fix**, as expected for an error-path guard: metrics identical to
  the Gemini reuse to the digit.
- **Scheduler note:** `qstat <id>` again reported "Unknown Job Id" for running develop-queue
  jobs; the `out/jobs/<id>.desched1.OU` logs are the evidence.

**Next (Stage gate — the user decides):** archive under a NEW name (do not overwrite
an existing archive); Stage 3 (profile) optional.
