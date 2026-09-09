# Translation report — laft-kessler, Claude Sonnet 4.6 (claude-sonnet-4-6) — 2026-04 JD-data translation REUSED under the current bridge

| | |
|---|---|
| run date | 2026-09-03 (Steps 1–3.5) / 2026-09-04 (Steps 4–8, after Derecho came back) |
| LLM | Claude Sonnet 4.6 (`claude-sonnet-4-6`) — authored in-context by Claude Code in the 2026-04 run (file header "Translated by Claude Sonnet 4.6") |
| translator mode | `source = "in_context"` — **no re-translation**: `out/jax/kessler_init.py` and `kessler_run.py` are verbatim copies of `_2sd_exp_JDdata/claude/jax/` (2026-04 JD-data run) with the mandated version header; the old `wrappers.py` was not copied (bridges are regenerated) |
| constraint | loop/vectorization **structure FROZEN** (the paper compares per-model design): interface-level repairs only; a structural fix would have been a HARD STOP for the user |
| workflow | `workflow_translator/TRANSLATE_WORKFLOW.md` |
| driving agent | Claude Fable 5.1 (`claude-fable-5-1`) |
| final status | **`complete`** — comparison ALL PASS, host mode (`[driver].contracts = [1]`) |
| `fix_attempts_used` | **1** (bridge suite, error-flag precedence — a guard, not structure) |

## 1. Translation plan (Step 1c)

| # | procedure | scalar_only | jax_required | passes | calls |
|---|---|---|---|---|---|
| 1 | kessler_init | yes | no | 1 (plain Python) | — |
| 2 | kessler_run | no | yes | 4 (full JAX) | — |

Passes are those of the 2026-04 run; nothing was regenerated in this run.

## 2. What this run is

The 2026-04 Sonnet 4.6 translation of the JD-data Kessler source was produced under the
pre-path-C bridge (wrapper-side guards, no device-resident entry). This run re-enters it
through the current LAFT pipeline — regenerated prompts/wrappers/bridges, the §2.5
completeness check, lint, the gated semantic audit, GPU runtime validation, the bridge
suite with translation-dependent tests, and the driver + comparison — to obtain a
like-for-like validation record next to the earlier archived runs of the working
project without altering the model's design.

Design (unchanged, from the file): `jax.vmap` over columns (`in_axes=1`) of a per-column
kernel; active level range as an index array (`level_indices`) with
`jnp.zeros(nz).at[level_indices].set(...)` scatter writes for the level-constant arrays;
per-column CFL `dt0`; sub-cycle as `lax.while_loop`, skipped via `lax.cond` when the
column's error flag is set; column flags reduced with `jnp.max`.

## Translation statistics

### Table 1 — Four-pass translation statistics (FROZEN)

Frozen at **2026-09-03T10:07:06** · translator **in_context** (Claude Sonnet 4.6 (claude-sonnet-4-6)) · 2 procedures.

**This table is immutable.** It records the initial four-pass output before any validation fix. Do not update it — later repairs belong in Table 2.

> Token columns are `n/a`: the translation was authored in-context, so no API reported token counts. Prompt and output **characters** are exact and stand in for size.

> Wall-clock source: elapsed between per-procedure state updates (includes pauses).

| module / procedure | passes | Fortran lines (code) | JAX lines (code) | JAX/F | prompt chars | JAX chars | prompt tok | output tok | wall-clock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **kessler** | | | | | | | | | |
| &nbsp;&nbsp;kessler_init | 1 | 17 (12) | 42 (28) | 2.47x | 7,807 | 1,383 | n/a | n/a | n/a |
| &nbsp;&nbsp;kessler_run | 4 | 210 (129) | 309 (222) | 1.47x | 85,360 | 13,684 | n/a | n/a | 0s |
| &nbsp;&nbsp;*subtotal (2 procs)* | | *227* | *351* | | *93,167* | *15,067* | *n/a* | *n/a* | *n/a* |
| **TOTAL (2 procs)** | | **227** | **351** | | **93,167** | **15,067** | **n/a** | **n/a** | **n/a** |

### Table 2 — Validation-phase fix statistics (LIVE)

1 fix attempt(s) across 1 procedure(s). Re-rendered on every fix; the JAX-line column shows the frozen four-pass value versus the current file.

| module / procedure | fixes | failed stage(s) | Fortran lines | JAX lines frozen → now | fix tokens | fix time | last fixed |
|---|---:|---|---:|---:|---:|---:|---|
| **kessler** | | | | | | | |
| &nbsp;&nbsp;kessler_run | 1 | bridge_suite | 210 | 309 → 313 (+4) | n/a | 4.5m | 2026-09-04T10:13:31 |

Wall-clock in Table 1 is meaningless for a copied translation (the "0s" is the interval
between the two copy timestamps); tokens are `n/a` (in-context / copy).

## 4. Completeness (§2.5) and semantic audit (§3.5)

# Translation completeness report (TRANSLATE_WORKFLOW Step 2.5)

| | |
|---|---|
| run | 2026-09-04T10:13:59.632129 |
| scope | all procedures |
| verdict | **PASS** — PASS 2, WARN 0, FAIL 0, MISSING 0 |
| rule | FAIL (scaffold phrase / commented-out callee) = translation STOPPED, `status aborted_scaffold`, no automatic retry — human decision |

| procedure | Fortran lines | JAX code lines | ratio | verdict | why |
|---|---|---|---|---|---|
| kessler_init | 13 | 24 | 1.846 | PASS |  |
| kessler_run | 173 | 220 | 1.272 | PASS |  |

Machine-readable: `out/reports/translation/completeness_check.json`

# Semantic audit report (TRANSLATE_WORKFLOW Step 3.5a)

| | |
|---|---|
| run | 2026-09-04T10:13:59.857332 |
| translator | Claude Sonnet 4.6 (claude-sonnet-4-6) |
| procedures audited | 2 |
| verdict | **PASS** — 0 FAIL, 4 WARN, 2 waived |
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
| kessler_run | 0 | 4 | 2 | 1 |

## Findings (FAIL / WARN / waived)

| procedure | check | severity | name | detail |
|---|---|---|---|---|
| kessler_run | zero-forever | WARN | `pc` | translation only ever assigns a zero constant, but the Fortran computes this variable |
| kessler_run | zero-forever | WARN | `r` | translation only ever assigns a zero constant, but the Fortran computes this variable |
| kessler_run | zero-forever | WARN | `rhalf` | translation only ever assigns a zero constant, but the Fortran computes this variable |
| kessler_run | zero-forever | WARN | `velqr` | translation only ever assigns a zero constant, but the Fortran computes this variable |
| kessler_run | intent-out | WAIVED | `precl` | Fortran intent(out) arg is only ever copied from its incoming value (or never assigned) — risk of accumulation across calls |
| kessler_run | intent-out | WAIVED | `relhum` | Fortran intent(out) arg is only ever copied from its incoming value (or never assigned) — risk of accumulation across calls |

Machine-readable findings: `out/issues/semantic_audit/<proc>.json`, summary `_summary.json`.

The four `zero-forever` WARNs are checker false positives: each variable is
`jnp.zeros(nz).at[level_indices].set(<formula>)` — the scatter design the freeze protects.
Not waived, no action. The two `intent-out` waivers (`precl`, `relhum`) are the project's
standing waivers (`out/issues/semantic_audit_waivers.json`).

## 5. Results

| procedure | lint | runtime GPU (7313640 → 7313717 after fix) | bridge suite CPU (7313663 → 7313718) | driver (7313723, host) | comparison (7313728) |
|---|---|---|---|---|---|
| kessler_init | 9/9 | imported, wrapper_ok, core_ok, wrapper_finite | — | init errflg 0 | — |
| kessler_run | 13/13 | imported, wrapper_ok, core_ok, wrapper_finite, core_finite | 16/17 → **17/17** (translation-dependent 8/8) | PASS, errflg 0, all outputs finite, 128/128 columns precipitating | **ALL PASS** |

Comparison vs Fortran (128 cols × 56 levels, dt = 60 s, seed 0; `compare_fortran_jax.json` 2026-09-04 10:16):

| variable | MAE | RMSE | max abs err | rel MAE | within tol |
|---|---:|---:|---:|---:|---|
| theta (K) | 7.851e-16 | 6.680e-15 | 5.684e-14 | 2.73e-18 | yes |
| qv | 4.029e-19 | 1.904e-18 | 6.245e-17 | 3.41e-17 | yes |
| qc | 2.194e-19 | 9.559e-19 | 2.860e-17 | 3.39e-16 | yes |
| qr | 2.431e-18 | 3.624e-18 | 2.082e-17 | 1.53e-16 | yes |
| precl | 1.863e-20 | 2.875e-20 | 1.084e-19 | 1.33e-16 | yes |
| relhum (%) | 7.660e-15 | 1.810e-14 | 4.121e-13 | 9.60e-17 | yes |

Same round-off class as the earlier archived runs (theta max abs err 5.68e-14 and worst ULP
4221 on `qc` identical to the archived set — the scheme's cancellation, not a model
property); MAEs differ from the earlier archived values in the last digits
(theta 7.85e-16 vs 7.30e-16), i.e. a different but equally valid summation order. Physics
sanity 6/6 on both sides. Bitwise: 98.6 % of theta values bit-identical to Fortran.

Casper CPU smoke check (2026-09-03, ad-hoc job 5818522, not a workflow step): the same
validator on CPU had already shown import/trace/finite for both procedures before the
Derecho GPU job — consistent with 7313640.

## 6. Files generated

| file | what it is |
|---|---|
| `out/jax/kessler_init.py` | verbatim 2026-04 scalar-only init (+ version header) |
| `out/jax/kessler_run.py` | verbatim 2026-04 `kessler_run_core` (jit, static `ncol nz lyr_surf lyr_toa`) + host wrapper, **one guard line changed by fix attempt 1**; header `# Pass: fix-attempt-1 (bridge suite, 2026-09-04) on final` |
| `out/jax/*.validated.py` | Step 7 snapshot after the comparison passed |
| `out/validation/*_runtime.json` | Step 4 results (PBS 7313717) |
| `out/reports/bridge/bridge_test_results.{json,xml}`, `bridge_report.md` | Step 4.5 (PBS 7313718) |
| `out/driver/*` | driver outputs (PBS 7313723) and comparison (PBS 7313728) |
| `out/reports/translation/pass_stats.json` (+ `.sha256`), `fix_stats.jsonl`, `_stats_tables.md` | Tables 1 and 2 |
| `out/reports/translation/completeness_report.md`, `semantic_audit_report.md` | §2.5 / §3.5 |
| `out/issues/fix_log.md` | gate trail + fix narrative |
| `out/jobs/73136{40,63}.desched1.OU`, `73137{17,18,23,28}.desched1.OU` | PBS logs |

## 7. Fix log (`workflow_state.json → fix_attempts`)

| n | proc | failed stage | error | change |
|---|---|---|---|---|
| 1 | kessler_run | bridge suite (7313663) | `test_error_handling_negative_dt`: `errflg = 2` instead of `1` for `dt = -60`. The top-level `errflg = where(dt <= 0, 1, 0)` was correct, but the per-column `col_errflg = where(dt0 < 1e-12, 2, errflg)` overwrote it (`dt0 = min(dt, …)` is negative when `dt` is). The Fortran `return`s at the `dt <= 0` check before the `dt0` check runs. The bug is in the 2026-04 original (verbatim-copy check passed); it never surfaced because the 2026-04 bridge ran a wrapper-side `dt <= 0` guard that the current path-C bridge deliberately does not. | Re-nested the `jnp.where` so a non-zero `errflg` is kept and the bad-time-split code (2) is set only when `errflg == 0`. One guard, no structural change. Full chain re-run: completeness PASS, lint 100 %, audit 0 FAIL / 4 WARN, runtime PASS, bridge 17/17, driver PASS, comparison ALL PASS. |

Noted, not changed (the user's call): the translation encodes the bad-time-split case as
`errflg = 2` where the Fortran uses `1` for both error branches (only `errmsg` differs).
The wrapper maps 2 → the right message; the numeric value deviates from Fortran. Untested;
left as is to keep the repair minimal.

## 8. What worked / what was tricky

- **Reuse works through the current pipeline with one guard fix.** A five-month-old
  translation, made under a different bridge, imports, traces, runs finite on the A100 and
  matches the Fortran at round-off without any change to its `vmap` / scatter / while-loop
  design. The only defect was an error-flag precedence that the old wrapper masked — exactly
  the class of defect the translation-dependent bridge tests exist to catch.
- **Wrapper-side guards hide core-side bugs.** The path-C bridge calls the core directly, so
  every Fortran early `return` must be reproduced numerically inside the core; a
  `jnp.where` chain has to preserve the Fortran's branch order (first error wins).
- **Scheduler quirks:** `qstat <id>` reported "Unknown Job Id" for every job in this run
  while it was running or seconds after submission; the `out/jobs/<id>.desched1.OU` log is
  the only reliable signal (the develop queue turned each job around in ~1 min).
- **Freeze discipline:** the audit's four `zero-forever` WARNs point straight at the scatter
  design; under a frozen structure they are documented as false positives and left alone
  rather than "fixed" into a masked-`where` form.

**Next (Stage gate — the user decides):** archive under a NEW name (do not overwrite
an existing archive), e.g. `translations/claude-sonnet46-jd/` — add it to
`[llm].copy_targets`, then `tools/copy_AI_results.sh <name>`; Stage 3 (profile) optional.
