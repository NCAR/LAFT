## Translation statistics

### Table 1 — Four-pass translation statistics (FROZEN)

Frozen at **2026-09-04T10:31:29** · translator **in_context** (Gemini 3.1 Pro (via Gemini CLI code agent; 2026-04 JD-data run, reused)) · 2 procedures.

**This table is immutable.** It records the initial four-pass output before any validation fix. Do not update it — later repairs belong in Table 2.

> Token columns are `n/a`: the translation was authored in-context, so no API reported token counts. Prompt and output **characters** are exact and stand in for size.

> Wall-clock source: elapsed between per-procedure state updates (includes pauses).

| module / procedure | passes | Fortran lines (code) | JAX lines (code) | JAX/F | prompt chars | JAX chars | prompt tok | output tok | wall-clock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **kessler** | | | | | | | | | |
| &nbsp;&nbsp;kessler_init | 1 | 17 (12) | 30 (19) | 1.76x | 7,807 | 1,028 | n/a | n/a | n/a |
| &nbsp;&nbsp;kessler_run | 4 | 210 (129) | 218 (137) | 1.04x | 85,360 | 12,175 | n/a | n/a | 0s |
| &nbsp;&nbsp;*subtotal (2 procs)* | | *227* | *248* | | *93,167* | *13,203* | *n/a* | *n/a* | *n/a* |
| **TOTAL (2 procs)** | | **227** | **248** | | **93,167** | **13,203** | **n/a** | **n/a** | **n/a** |

### Table 2 — Validation-phase fix statistics (LIVE)

No fixes recorded: every procedure passed validation as authored. Table 1 is the complete record.
