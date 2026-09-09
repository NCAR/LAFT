## Translation statistics

### Table 1 — Four-pass translation statistics (FROZEN)

Frozen at **2026-09-04T10:38:32** · translator **in_context** (GPT-5.4 Thinking (2026-04 JD-data run, reused)) · 2 procedures.

**This table is immutable.** It records the initial four-pass output before any validation fix. Do not update it — later repairs belong in Table 2.

> Token columns are `n/a`: the translation was authored in-context, so no API reported token counts. Prompt and output **characters** are exact and stand in for size.

> Wall-clock source: elapsed between per-procedure state updates (includes pauses).

| module / procedure | passes | Fortran lines (code) | JAX lines (code) | JAX/F | prompt chars | JAX chars | prompt tok | output tok | wall-clock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **kessler** | | | | | | | | | |
| &nbsp;&nbsp;kessler_init | 1 | 17 (12) | 28 (17) | 1.65x | 7,807 | 936 | n/a | n/a | n/a |
| &nbsp;&nbsp;kessler_run | 4 | 210 (129) | 347 (271) | 1.65x | 85,360 | 20,221 | n/a | n/a | 0s |
| &nbsp;&nbsp;*subtotal (2 procs)* | | *227* | *375* | | *93,167* | *21,157* | *n/a* | *n/a* | *n/a* |
| **TOTAL (2 procs)** | | **227** | **375** | | **93,167** | **21,157** | **n/a** | **n/a** | **n/a** |

### Table 2 — Validation-phase fix statistics (LIVE)

1 fix attempt(s) across 1 procedure(s). Re-rendered on every fix; the JAX-line column shows the frozen four-pass value versus the current file.

| module / procedure | fixes | failed stage(s) | Fortran lines | JAX lines frozen → now | fix tokens | fix time | last fixed |
|---|---:|---|---:|---:|---:|---:|---|
| **kessler** | | | | | | | |
| &nbsp;&nbsp;kessler_run | 1 | bridge_suite | 210 | 347 → 351 (+4) | n/a | 2.0m | 2026-09-04T10:39:58 |
