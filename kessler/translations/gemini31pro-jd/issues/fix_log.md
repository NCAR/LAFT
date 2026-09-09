# Fix log — laft-kessler, Gemini 3.1 Pro JD-data translation REUSED under the current bridge, 2026-09-04

Driving agent: Claude Fable 5.1 (claude-fable-5-1). Translator: `source = "in_context"` — NO
re-translation: `out/jax/kessler_init.py` + `kessler_run.py` are verbatim copies of
`_2sd_exp_JDdata/gemini/jax/` (2026-04 JD-data run; file docstrings say "Gemini 3.1 PRO",
authored through the Gemini CLI code agent) with the mandated version header. Loop/
vectorization structure FROZEN — interface-level repairs only; structural fix = HARD STOP.
Preceded by the Sonnet 4.6 reuse (archived `translations/claude-sonnet46-jd/`, out/ cleaned
with `clean_AI_results.sh claude-sonnet46-jd` before this run).

## Stage 2 gate trail

| step | result | evidence |
|---|---|---|
| Step 1.0 | prompts + wrappers regenerated | `out/prompts/`, `out/wrappers/` |
| §2a | verbatim copy + header (diff of non-comment lines vs source: identical) | `out/jax/*.py` |
| §2.5 completeness | PASS 2/2 | `out/reports/translation/completeness_report.md` |
| §2e | four-pass stats frozen (tokens n/a; wall-clock meaningless for a copy) | `pass_stats.json` |
| Step 3 lint | 23/23, 100 % | `out/lint/_summary.json` |
| Step 3.5 audit | 0 FAIL, 3 WARN (`intent-out` on `errflg`, `errmsg`, `scheme_name` — the wrapper passes them through / sets them host-side; not waived) | `semantic_audit_report.md` |
| Step 4 runtime (GPU) | PASS — PBS **7313832** (2026-09-04 10:32): both procs imported/wrapper_ok/core_ok/wrapper_finite, `kessler_run` core_finite | `out/validation/*_runtime.json`, `out/jobs/7313832.desched1.OU` |
| Step 4.5 bridge suite (CPU PBS) | **PASS 17/17** (8/8 translation-dependent) first try — PBS **7313833** | `out/reports/bridge/bridge_test_results.json`, `out/jobs/7313833.desched1.OU` |
| Step 5 driver (host mode, `contracts = [1]`) | PASS — PBS **7313840** (10:33): init errflg 0, run errflg 0, outputs finite | `out/driver/kessler_driver.json`, `out/jobs/7313840.desched1.OU` |
| Step 5 comparison | **ALL PASS** — PBS **7313848** (10:34): theta MAE 7.30e-16 K (identical to the digit with the earlier archived metrics), all six within tolerance; physics sanity 6/6 both sides; worst ULP 4221 on `qc` (scheme cancellation) | `out/driver/compare_results_fortran_jax.txt`, `compare_fortran_jax.json` |
| Step 7 snapshot | `out/jax/*.validated.py` | |

Result: Stage 2 `complete`, **`fix_attempts_used = 0`** — the 2026-04 Gemini translation passed
every gate of the current pipeline unmodified. Its `dt <= 0` branch is an outer `lax.cond`
bypass that sets `errflg = 1` inside the core, so the path-C bridge (no wrapper guard) still
reproduces the Fortran early return — the very defect the Sonnet 4.6 reuse needed a fix for.
