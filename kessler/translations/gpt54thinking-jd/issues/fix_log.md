# Fix log — laft-kessler, GPT-5.4 Thinking JD-data translation REUSED under the current bridge, 2026-09-04

Driving agent: Claude Fable 5.1 (claude-fable-5-1). Translator: `source = "in_context"` — NO
re-translation: `out/jax/kessler_init.py` + `kessler_run.py` are verbatim copies of
`_2sd_exp_JDdata/gpt/jax/` (2026-04 JD-data run; file docstrings say "Translated by GPT-5.4
Thinking") with the mandated version header. Loop/vectorization structure FROZEN —
interface-level repairs only; structural fix = HARD STOP. Third JD reuse of the day, after
Sonnet 4.6 (`translations/claude-sonnet46-jd/`, 1 guard fix) and Gemini 3.1 Pro
(`translations/gemini31pro-jd/`, 0 fixes); out/ was cleaned with
`clean_AI_results.sh gemini31pro-jd` before this run.

## Stage 2 gate trail

| step | result | evidence |
|---|---|---|
| Step 1.0 | prompts + wrappers regenerated | `out/prompts/`, `out/wrappers/` |
| §2a | verbatim copy + header (diff of non-comment lines vs source: identical) | `out/jax/*.py` |
| §2.5 completeness | PASS 2/2 | `out/reports/translation/completeness_report.md` |
| §2e | four-pass stats frozen (tokens n/a; wall-clock meaningless for a copy) | `pass_stats.json` |
| Step 3 lint | 23/23, 100 % | `out/lint/_summary.json` |
| Step 3.5 audit | 0 FAIL, 1 WARN (`zero-forever relhum` — `jnp.zeros(nz).at[level_indices].set(...)` scatter, checker false positive; not waived) | `semantic_audit_report.md` |
| Step 4 runtime (GPU) | PASS — PBS **7313881** (2026-09-04): both procs imported/wrapper_ok/core_ok/wrapper_finite, `kessler_run` core_finite | `out/validation/*_runtime.json`, `out/jobs/7313881.desched1.OU` |
| Step 4.5 bridge suite (CPU PBS) | **FAIL 16/17** — PBS **7313882**: `test_error_handling_negative_dt` (dependent test; layout tests green → translation defect) | `out/jobs/7313882.desched1.OU` |

## Fix attempt 1 — `kessler_run.py`, bridge suite, error-flag precedence

**Same defect as the Sonnet 4.6 reuse** (see `translations/claude-sonnet46-jd/issues/fix_log.md`).
`kessler_run_bridge(dt=-60, …)` returned `errflg = 2`; test and Fortran expect `1`. The
top-level `errflg = where(dt <= 0, 1, 0)` is right, but the per-column
`col_errflg = where(dt0 < 1e-12, 2, errflg)` overwrote it (`dt0 = min(dt, …)` is negative when
`dt` is). The Fortran (`out/modules/kessler.F90` l.152–156) `return`s at the `dt <= 0` check
before the `dt0` check (l.204–208). Bug is in the 2026-04 original (verbatim-copy check
passed); masked then by the old bridge's wrapper-side `dt <= 0` guard, which the path-C bridge
deliberately does not run. Predicted from reading the kernel before the job; the gate confirmed it.

**Change (interface-level guard, NOT structural).** One `jnp.where` re-nested:
`col_errflg = where(errflg != 0, errflg, where(dt0 < 1e-12, 2, 0))`. The `vmap` over columns,
the `.at[level_indices].set` scatter design, the `lax.while_loop` sub-cycle and the `lax.cond`
skip are untouched. Header now `# Pass: fix-attempt-1 (bridge suite, 2026-09-04) on final`.

**Noted, not changed:** as with Sonnet, `errflg = 2` for bad time splitting where the Fortran
uses 1 for both branches (only errmsg differs). The GPT design is otherwise near-identical to
the Sonnet one (same level-index scatter layout, same `lax.cond` skip, same flag reduction).

### Re-validation after fix attempt 1

| step | result | evidence |
|---|---|---|
| §2.5 / Step 3 / Step 3.5 re-run on the edited file | completeness PASS 2/2; lint 23/23; audit 0 FAIL, 1 WARN; gate OPEN | reports regenerated 10:39:58 |
| Step 4 runtime (GPU) | PASS — PBS **7313896** | `out/validation/*_runtime.json`, `out/jobs/7313896.desched1.OU` |
| Step 4.5 bridge suite (CPU PBS) | **PASS 17/17** (8/8 dependent) — PBS **7313897** | `out/reports/bridge/bridge_test_results.json`, `out/jobs/7313897.desched1.OU` |
| Step 5 driver (host mode, `contracts = [1]`) | PASS — PBS **7313908** (10:41): init errflg 0, run errflg 0 | `out/driver/kessler_driver.json` |
| Step 5 comparison | **ALL PASS** — PBS **7313910** (10:41): theta MAE 7.30e-16 K, identical to the digit with the Gemini reuse and the earlier archived set; physics 6/6 both sides; worst ULP 4221 on `qc` | `out/driver/compare_results_fortran_jax.txt`, `compare_fortran_jax.json` |
| Step 7 snapshot | `out/jax/*.validated.py` | |

Result: Stage 2 `complete`, `fix_attempts_used = 1` (interface-level guard; structure frozen and untouched).
