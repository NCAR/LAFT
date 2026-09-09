## Translation statistics

### Table 1 — Four-pass translation statistics (FROZEN)

Frozen at **2026-09-04T10:48:32** · translator **in_context** (Qwen2.5-32B (Qwen2.5-Coder-32B-Instruct via vLLM; 2026-04 JD-data run, reused)) · 2 procedures.

**This table is immutable.** It records the initial four-pass output before any validation fix. Do not update it — later repairs belong in Table 2.

> Token columns are `n/a`: the translation was authored in-context, so no API reported token counts. Prompt and output **characters** are exact and stand in for size.

> Wall-clock source: elapsed between per-procedure state updates (includes pauses).

| module / procedure | passes | Fortran lines (code) | JAX lines (code) | JAX/F | prompt chars | JAX chars | prompt tok | output tok | wall-clock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **kessler** | | | | | | | | | |
| &nbsp;&nbsp;kessler_init | 1 | 17 (12) | 39 (27) | 2.29x | 7,807 | 1,454 | n/a | n/a | n/a |
| &nbsp;&nbsp;kessler_run | 4 | 210 (129) | 121 (76) | 0.58x | 85,360 | 8,818 | n/a | n/a | 0s |
| &nbsp;&nbsp;*subtotal (2 procs)* | | *227* | *160* | | *93,167* | *10,272* | *n/a* | *n/a* | *n/a* |
| **TOTAL (2 procs)** | | **227** | **160** | | **93,167** | **10,272** | **n/a** | **n/a** | **n/a** |

### Table 2 — Validation-phase fix statistics (LIVE)

1 fix attempt(s) across 1 procedure(s). Re-rendered on every fix; the JAX-line column shows the frozen four-pass value versus the current file.

| module / procedure | fixes | failed stage(s) | Fortran lines | JAX lines frozen → now | fix tokens | fix time | last fixed |
|---|---:|---|---:|---:|---:|---:|---|
| **kessler** | | | | | | | |
| &nbsp;&nbsp;kessler_run | 1 | runtime | 210 | 121 → 123 (+2) | n/a | 3.0m | 2026-09-04T10:50:26 |
