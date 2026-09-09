# translations/qwen25-32b-jd — laft-kessler, the 2026-04 Qwen2.5-32B JD-data translation re-validated under the current bridge, 2026-09-04

Archive of `out/` after Stage 2 (translate/validate) only — no Stage 3 profile — made with
`tools/copy_AI_results.sh qwen25-32b-jd` (`__pycache__` dropped).

| | |
|---|---|
| model | Qwen2.5-32B (file header; the 2026-04 script's default is `Qwen2.5-Coder-32B-Instruct`, vLLM offline inference). The file already carried the 2026-04 run's own validation fix ("Fixed by: Claude Opus 4.7 — subcycle time_counter ordering") |
| what this is | **NOT a new translation.** `jax/kessler_init.py` + `kessler_run.py` are verbatim copies of `_2sd_exp_JDdata/qwen/jax/` (the 2026-04 JD-data run, pre-path-C bridge) with the mandated version header, pushed through the current LAFT pipeline with the loop/vectorization **structure frozen** (interface-level repairs only) |
| driving agent | Claude Fable 5.1 (gates, 1 guard fix) |
| contract | 1 (host), `[driver].contracts = [1]` |
| Stage 2 | completeness PASS · lint 100 % (23/23) · audit 0 FAIL / 0 WARN · runtime **FAIL → PASS** (PBS 7313968 → 7313978) · bridge suite **17/17** both rounds (7313969, 7313979) · driver PASS (7313985) · comparison **ALL PASS** (7313992, theta MAE 7.30e-16 K) · `fix_attempts_used = 1` |
| the fix | one-level harness guard: the runtime harness's `lyr_surf = lyr_toa = 4` makes the interior CFL range empty and `jnp.min` has no identity; a static `if lyr_surf_idx != lyr_toa_idx:` around the two CFL min-reductions leaves `dt0` unchanged there, exactly the Fortran's empty loop. Trace-time only — multi-level graphs unchanged |
| design | **serial `lax.fori_loop` over columns** (the one JD translation without `vmap`); vectorised level updates as `.at[klevs].set` scatters on the full arrays + per-level workspaces in the loop carry; `min(where)` CFL `dt0`; `lax.while_loop` sub-cycle; no `dt <= 0` branch of its own (flagged via the CFL check), sub-cycle not skipped on `errflg` |

Read first: `reports/translation/translation_report.md`, then `issues/fix_log.md`.
Siblings (same recipe, same day): `translations/claude-sonnet46-jd/` (1 fix),
`translations/gemini31pro-jd/` (0), `translations/gpt54thinking-jd/` (1).
