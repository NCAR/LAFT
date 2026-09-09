# Fix log — laft-kessler, Qwen2.5-32B JD-data translation REUSED under the current bridge, 2026-09-04

Driving agent: Claude Fable 5.1 (claude-fable-5-1). Translator: `source = "in_context"` — NO
re-translation: `out/jax/kessler_init.py` + `kessler_run.py` are verbatim copies of
`_2sd_exp_JDdata/qwen/jax/` (2026-04 JD-data run: "Translated by: Qwen2.5-32B", script default
`Qwen2.5-Coder-32B-Instruct` via vLLM offline inference; the file already carries that run's
own validation fix "Fixed by: Claude Opus 4.7 — fix attempt 1: subcycle time_counter
ordering") with the mandated version header. Loop/vectorization structure FROZEN —
interface-level repairs only; structural fix = HARD STOP. Fourth and last JD reuse of the
day, after Sonnet 4.6 (1 guard fix), Gemini 3.1 Pro (0), GPT-5.4 Thinking (1, same guard);
out/ cleaned with `clean_AI_results.sh gpt54thinking-jd` before this run.

## Stage 2 gate trail

| step | result | evidence |
|---|---|---|
| Step 1.0 | prompts + wrappers regenerated | `out/prompts/`, `out/wrappers/` |
| §2a | verbatim copy + header (diff of non-comment lines vs source: identical) | `out/jax/*.py` |
| §2.5 completeness | PASS 2/2 | `out/reports/translation/completeness_report.md` |
| §2e | four-pass stats frozen (tokens n/a; wall-clock meaningless for a copy) | `pass_stats.json` |
| Step 3 lint | 23/23, 100 % | `out/lint/_summary.json` |
| Step 3.5 audit | 0 FAIL, 0 WARN | `semantic_audit_report.md` |
| Step 4 runtime (GPU) | **FAIL** — PBS **7313968**: `kessler_run` wrapper_ok=False / core_ok=False, `ValueError: zero-size array to reduction operation min which has no identity` | `out/jobs/7313968.desched1.OU` |
| Step 4.5 bridge suite (CPU PBS) | **PASS 17/17** (8/8 dependent) with real multi-level inputs — PBS **7313969** | `out/jobs/7313969.desched1.OU` |

## Fix attempt 1 — `kessler_run.py`, runtime, one-level harness guard

**Symptom.** The runtime harness feeds every integer scalar `DUMMY_N = 4`, so
`lyr_surf = lyr_toa = 4`: a ONE-level range. The interior range
`jnp.arange(lyr_surf_idx, lyr_toa_idx, lyr_step)` is then empty and the CFL step
`dt0 = jnp.min(jnp.where(... velqr[klevs] ...))` reduces over zero elements — JAX refuses
(no identity for `min`). The bridge suite, driver and comparison all use real multi-level
grids and never hit it (bridge 17/17 in the same round). The Fortran tolerates the case: its
CFL `do` loop is empty, so `dt0` keeps the value `dt`.

**Predicted before the job** from the kernel (same harness pitfall as fresh translations
in the working project hit in August 2026). The 2026-04 run's
own `kessler_run_runtime.json` shows `core_ok = True` — the harness of that time evidently
did not use a degenerate level range.

**Change (interface-level guard, NOT structural).** `lyr_surf`/`lyr_toa` are static under
jit, so a Python-level `if lyr_surf_idx != lyr_toa_idx:` now wraps the two CFL
min-reductions (initial `dt0` and the recompute in the sub-cycle body). With a one-level
range `dt0` is left unchanged — exactly the Fortran's empty loop. On every multi-level grid
the traced graph is identical to before (the guard is resolved at trace time), so the
driver-grid numerics cannot change. The serial column `lax.fori_loop`, the vectorised
`.at[klevs].set` level updates and the `lax.while_loop` sub-cycle are untouched. Header now
`# Pass: fix-attempt-1 (runtime, 2026-09-04) on final`.

**Noted, not changed.** The core has no `dt <= 0` branch of its own — a nonpositive `dt`
surfaces through the CFL check (`dt0 = dt < 1e-12 → errflg = 1`), and the sub-cycle still
runs one step before its condition closes; the wrapper then picks the Fortran message from
the sign of `dt`. The bridge test for that path passes numerically. Also: the sub-cycle does
not stop on `errflg` (no `lax.cond` skip) — the flag is only reported. Both are the 2026-04
design as validated then; left as is.

### Re-validation after fix attempt 1

| step | result | evidence |
|---|---|---|
| §2.5 / Step 3 / Step 3.5 re-run on the edited file | completeness PASS 2/2; lint 23/23; audit 0 FAIL, 0 WARN; gate OPEN | reports regenerated 10:50:26 |
| Step 4 runtime (GPU) | PASS — PBS **7313978**: both procs ok, `kessler_run` core_finite | `out/validation/*_runtime.json`, `out/jobs/7313978.desched1.OU` |
| Step 4.5 bridge suite (CPU PBS) | **PASS 17/17** — PBS **7313979** | `out/reports/bridge/bridge_test_results.json` |
| Step 5 driver (host mode, `contracts = [1]`) | PASS — PBS **7313985** (10:51): init errflg 0, run errflg 0 | `out/driver/kessler_driver.json` |
| Step 5 comparison | **ALL PASS** — PBS **7313992** (10:52): theta MAE 7.30e-16 K, identical to the digit with the Gemini and GPT reuses and the earlier archived set; physics 6/6 both sides; worst ULP 4221 on `qc` | `out/driver/compare_results_fortran_jax.txt`, `compare_fortran_jax.json` |
| Step 7 snapshot | `out/jax/*.validated.py` | |

Result: Stage 2 `complete`, `fix_attempts_used = 1` (interface-level static guard; structure frozen and untouched).
