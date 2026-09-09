# Semantic audit report (TRANSLATE_WORKFLOW Step 3.5a)

| | |
|---|---|
| run | 2026-09-04T10:50:26.877612 |
| translator | Qwen2.5-32B (Qwen2.5-Coder-32B-Instruct via vLLM; 2026-04 JD-data run, reused) |
| procedures audited | 2 |
| verdict | **PASS** — 0 FAIL, 0 WARN, 1 waived |
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
| kessler_run | 0 | 0 | 1 | 1 |

## Findings (FAIL / WARN / waived)

| procedure | check | severity | name | detail |
|---|---|---|---|---|
| kessler_run | intent-out | WAIVED | `relhum` | Fortran intent(out) arg is only ever copied from its incoming value (or never assigned) — risk of accumulation across calls |

Machine-readable findings: `out/issues/semantic_audit/<proc>.json`, summary `_summary.json`.
