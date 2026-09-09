# Fix log — laft-kessler, Claude Sonnet 4.6 (claude-sonnet-4-6) JD-data translation REUSED under the current bridge, 2026-09-03/04

Driving agent: Claude Fable 5.1 (claude-fable-5-1). Translator: `source = "in_context"` — NO
re-translation: `out/jax/kessler_init.py` + `kessler_run.py` are verbatim copies of
`_2sd_exp_JDdata/claude/jax/` (2026-04 run) with the mandated version header. The
loop/vectorization structure is FROZEN (paper compares per-model design); only
interface-level / guard repairs are allowed, a structural fix is a HARD STOP for the user.

## Stage 2 gate trail

| step | result | evidence |
|---|---|---|
| §2.5 completeness | PASS 2/2 | `out/reports/translation/completeness_report.md` |
| Step 3 lint | 22/22, 100 % | `out/lint/_summary.json` |
| Step 3.5 audit | 0 FAIL, 4 WARN (zero-forever false positives on `pc, r, rhalf, velqr` — scatter design the freeze protects; not waived) | `out/reports/translation/semantic_audit_report.md` |
| Step 4 runtime (GPU) | PASS — PBS **7313640** (2026-09-04, deg0081): both procs imported/wrapper_ok/core_ok/wrapper_finite; `kessler_run` core_finite | `out/validation/*_runtime.json`, `out/jobs/7313640.desched1.OU` |
| Step 4.5 bridge suite (CPU PBS) | **FAIL 16/17** — PBS **7313663**: `test_error_handling_negative_dt` (dependent test), layout tests all green → translation defect | `out/jobs/7313663.desched1.OU`, `out/reports/bridge/bridge_test_results.json` (overwritten by the re-run) |

## Fix attempt 1 — `kessler_run.py`, bridge suite, error-flag precedence

**Symptom.** `kessler_run_bridge(dt=-60, …)` returned `errflg = 2`; the test (and the
Fortran) expect `1`.

**Cause (audit before fix).** The translation computes the top-level flag
`errflg = where(dt <= 0, 1, 0)` correctly, but inside the per-column kernel
`col_errflg = where(dt0 < 1e-12, 2, errflg)` — and `dt0 = min(dt, cfl…)` is negative when
`dt` is, so the bad-time-split branch overwrote the nonpositive-dt flag. In the Fortran
(`out/modules/kessler.F90` l.152–156) the `dt <= 0` check `return`s before the `dt0`
check (l.204–208) is ever reached, so `1` must win. Verbatim-copy check: the reused file
is identical (comments aside) to `_2sd_exp_JDdata/claude/jax/kessler_run.py` — the bug
is in the 2026-04 original; it never surfaced because the 2026-04 bridge had a wrapper
guard for `dt <= 0` that the current path-C bridge deliberately does not run (see the
test's docstring).

**Change (interface-level guard, NOT structural).** One `jnp.where` re-nested:

```python
col_errflg = jnp.where(errflg != 0, errflg,
                       jnp.where(dt0 < 1.0e-12, 2, 0))
```

Everything else — the `vmap` over columns, the `.at[level_indices].set` scatter design,
the masked `lax.while_loop` sub-cycle, `lax.cond` skip — is untouched. Header now reads
`# Pass: fix-attempt-1 (bridge suite, 2026-09-04) on final`.

**Noted, not changed (the user's call).** The translation uses `errflg = 2` for the
bad-time-split case where the Fortran uses `1` for both (only `errmsg` differs). The
wrapper maps 2 → "bad time splitting" so callers see the right message; the numeric
value deviates from Fortran. Not covered by any test; left as is to keep the repair
minimal.

**Re-validation.** Step 4 (GPU runtime) and Step 4.5 (bridge suite) re-run after the fix —
see the rows appended below.

### Re-validation after fix attempt 1 (2026-09-04)

| step | result | evidence |
|---|---|---|
| §2.5 / Step 3 / Step 3.5 re-run on the edited file | completeness PASS 2/2; lint 22/22 (100 %); audit 0 FAIL, 4 WARN (same four zero-forever false positives); gate OPEN | reports regenerated 10:13:59 |
| Step 4 runtime (GPU) | PASS — PBS **7313717**: both procs imported/wrapper_ok/core_ok/wrapper_finite, `kessler_run` core_finite | `out/validation/*_runtime.json` (10:14), `out/jobs/7313717.desched1.OU` |
| Step 4.5 bridge suite (CPU PBS) | **PASS 17/17** (8/8 translation-dependent) — PBS **7313718** | `out/reports/bridge/bridge_test_results.json` (10:14), `out/jobs/7313718.desched1.OU` |
| Step 5 driver (host mode, `contracts = [1]`) | see below | |
| Step 5 driver (host mode, `contracts = [1]`) | PASS — PBS **7313723** (2026-09-04 10:15): init errflg 0, run errflg 0, all outputs finite, 128/128 columns precipitating | `out/driver/kessler_driver.json`, `out/jobs/7313723.desched1.OU` |
| Step 5 comparison | **ALL PASS** — PBS **7313728** (10:16): theta MAE 7.85e-16 K, all six variables within tolerance; physics sanity 6/6 Fortran and JAX; worst ULP 4221 on `qc` (the scheme's cancellation, same as every archived model) | `out/driver/compare_results_fortran_jax.txt`, `compare_fortran_jax.json` |
| Step 7 snapshot | `out/jax/*.validated.py` | |

Result: Stage 2 `complete`, `fix_attempts_used = 1` (interface-level guard; structure frozen and untouched).
