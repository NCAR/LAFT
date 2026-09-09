# translations/claude-sonnet46-jd — laft-kessler, the 2026-04 Claude Sonnet 4.6 JD-data translation re-validated under the current bridge, 2026-09-04

Archive of `out/` after Stage 2 (translate/validate) only — no Stage 3 profile — made with
`tools/copy_AI_results.sh claude-sonnet46-jd` (`__pycache__` dropped).

| | |
|---|---|
| model | Claude Sonnet 4.6 (`claude-sonnet-4-6`), authored in-context by Claude Code in the 2026-04 run (file header "Translated by Claude Sonnet 4.6") |
| what this is | **NOT a new translation.** `jax/kessler_init.py` + `kessler_run.py` are verbatim copies of `_2sd_exp_JDdata/claude/jax/` (the 2026-04 JD-data run, pre-path-C bridge) with the mandated version header, pushed through the current LAFT pipeline with the loop/vectorization **structure frozen** (interface-level repairs only) |
| driving agent | Claude Fable 5.1 (gates, 1 guard fix) |
| contract | 1 (host), `[driver].contracts = [1]` |
| Stage 2 | completeness PASS · lint 100 % (22/22) · audit 0 FAIL / 4 WARN (zero-forever false positives on the scatter design, not waived) · runtime PASS (PBS 7313717) · bridge suite **17/17** (dep 8/8, PBS 7313718; 16/17 before the fix, PBS 7313663) · driver PASS (7313723) · comparison **ALL PASS** (7313728, theta MAE 7.85e-16 K) · `fix_attempts_used = 1` |
| the fix | error-flag precedence: the per-column bad-time-split check (`errflg = 2`) overwrote the nonpositive-dt flag (`1`); the Fortran returns at the dt check first. One re-nested `jnp.where`; bug is in the 2026-04 original, masked then by a wrapper-side guard the path-C bridge no longer runs |
| design | `jax.vmap` over columns (`in_axes=1`) of a per-column kernel; index-array level range with `jnp.zeros(nz).at[level_indices].set(...)` scatters for level-constant arrays; per-column CFL `dt0`; `lax.while_loop` sub-cycle skipped via `lax.cond` on the column flag; `jnp.max` flag reduction |

Read first: `reports/translation/translation_report.md`, then `issues/fix_log.md` (gate trail +
fix narrative). Not changed, noted: the translation encodes bad time splitting as `errflg = 2`
where the Fortran uses 1 for both branches (only the message differs).
