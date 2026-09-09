# translations/gemini31pro-jd — laft-kessler, the 2026-04 Gemini 3.1 Pro JD-data translation re-validated under the current bridge, 2026-09-04

Archive of `out/` after Stage 2 (translate/validate) only — no Stage 3 profile — made with
`tools/copy_AI_results.sh gemini31pro-jd` (`__pycache__` dropped).

| | |
|---|---|
| model | Gemini 3.1 Pro via the Gemini CLI code agent (file docstrings: "Gemini 3.1 PRO"; no API model id) |
| what this is | **NOT a new translation.** `jax/kessler_init.py` + `kessler_run.py` are verbatim copies of `_2sd_exp_JDdata/gemini/jax/` (the 2026-04 JD-data run, pre-path-C bridge) with the mandated version header, pushed through the current LAFT pipeline with the loop/vectorization **structure frozen** |
| driving agent | Claude Fable 5.1 (gates only) |
| contract | 1 (host), `[driver].contracts = [1]` |
| Stage 2 | completeness PASS · lint 100 % (23/23) · audit 0 FAIL / 3 WARN (intent-out on errflg/errmsg/scheme_name, wrapper-side) · runtime PASS (PBS 7313832) · bridge suite **17/17** first try (7313833) · driver PASS (7313840) · comparison **ALL PASS** (7313848, theta MAE 7.30e-16 K, identical to the earlier archived digits) · **`fix_attempts_used = 0`** |
| design | outer `lax.cond` on `dt <= 0` inside the core (errflg 1); `jax.vmap` over columns of a full-(nz,) per-column kernel with a `valid_k` mask and clipped `k_next` gathers; masked-`min` CFL `dt0`; `lax.while_loop` sub-cycle stopping on the column flag; TOA row via `jnp.where`; no `.at[].set` scatters |

Read first: `reports/translation/translation_report.md`; `issues/fix_log.md` is the gate trail.
Sibling reuse: `translations/claude-sonnet46-jd/` (same recipe, 1 guard fix).
