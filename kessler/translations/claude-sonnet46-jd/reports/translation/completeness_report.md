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
