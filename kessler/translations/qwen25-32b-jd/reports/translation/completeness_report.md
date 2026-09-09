# Translation completeness report (TRANSLATE_WORKFLOW Step 2.5)

| | |
|---|---|
| run | 2026-09-04T10:50:26.747821 |
| scope | all procedures |
| verdict | **PASS** — PASS 2, WARN 0, FAIL 0, MISSING 0 |
| rule | FAIL (scaffold phrase / commented-out callee) = translation STOPPED, `status aborted_scaffold`, no automatic retry — human decision |

| procedure | Fortran lines | JAX code lines | ratio | verdict | why |
|---|---|---|---|---|---|
| kessler_init | 13 | 23 | 1.769 | PASS |  |
| kessler_run | 173 | 78 | 0.451 | PASS |  |

Machine-readable: `out/reports/translation/completeness_check.json`
