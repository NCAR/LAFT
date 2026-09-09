# nsys Profiling Report — JAX Claude & GPT `kessler_run_core` (JD Inputs)

**Date:** 2026-04-28  
**Source:** `claude.nsys-rep` · `gpt.nsys-rep` — SQLite extracted with nsys 2025.6.1 (nvhpc/26.1)  
**Platform:** NVIDIA A100-SXM4-40GB · CUDA 13.0 · Derecho  
**Inputs:** `ncol=1000`, `nz=56`, `dt=60 s` — JD driver inputs (Box-Muller, seed=42, `qr = arr × 0.01`)  
**Calls:** 1 warmup + 6 profiled calls  
**Script:** `LLMx_jd_profiling_run.py --mode nsys --model <claude|gpt> --ncol 1000`  
**PBS:** `jax_gpu_profile.sh` (MODE=nsys)

> Claude and GPT produce **nearly identical nsys profiles** — same kernel types, same graph
> structure, same timing budget. They are documented together; differences are noted where relevant.

---

## 1. CUDA Kernel Summary

### Claude

| Kernel | Count (total) | Per call | Avg (µs) | Total (µs) |
|--------|:-------------:|:--------:|:--------:|:----------:|
| `input_reduce_fusion` | 84 | 12 | 3.8 | 315.8 |
| `wrapped_broadcast` | 1 | — | 3.5 | 3.5 |
| **Total (per call)** | | **12** | | **52.6 µs** |

### GPT

| Kernel | Count (total) | Per call | Avg (µs) | Total (µs) |
|--------|:-------------:|:--------:|:--------:|:----------:|
| `input_reduce_fusion` | 84 | 12 | 3.8 | 321.0 |
| `wrapped_broadcast` | 1 | — | 3.5 | 3.5 |
| **Total (per call)** | | **12** | | **53.5 µs** |

`wrapped_broadcast` fires once at XLA process start (initialization). The physics work is
entirely carried by `input_reduce_fusion`, which fires 12 times per call — matching the
11 while-loop subcycle body iterations + 1 exit-condition check.

Both models produce only **2 unique kernel types**, compared to Gemini's 5. The absence of
`loop_multiply_fusion`, `loop_convert_fusion`, and `input_reduce_select_fusion_1` reflects
a less decomposed XLA computation graph — likely because Claude and GPT use `lax.fori_loop`
for the vertical (nz) loop rather than Gemini's fully vectorized column-parallel approach.

---

## 2. CUDA Graph Structure

Both models produce **13 CUDA graphs per call** (91 total / 7 calls), matching Gemini's structure:

| Graph slot | Count | Interpretation |
|:----------:|:-----:|----------------|
| 1st graph | 1 | Pre-while initialization |
| Middle graphs | 11 | While-loop body — 1 graph per physics subcycle |
| Last graph | 1 | Post-while finalization |

**11 physics subcycles** confirmed (seed=42, same as Gemini).

---

## 3. Per-Call Execution Timing

### Claude

| Call | Type | GPU span (µs) | Graph exec (µs) | Inter-graph gap (µs) |
|:----:|------|:-------------:|:---------------:|:--------------------:|
| 1 | warmup | 1765.4 | 1345.6 | 419.8 |
| 2 | profiled | 1737.9 | 1342.8 | 395.0 |
| 3 | profiled | 1726.8 | 1341.5 | 385.3 |
| 4 | profiled | 1717.1 | 1341.7 | 375.4 |
| 5 | profiled | 1727.9 | 1342.5 | 385.4 |
| 6 | profiled | 1729.6 | 1342.9 | 386.8 |
| 7 | profiled | 1719.5 | 1341.8 | 377.8 |
| **Avg (calls 2–7)** | | **1726.5 µs** | **1342.2 µs** | **384.3 µs** |

### GPT

| Call | Type | GPU span (µs) | Graph exec (µs) | Inter-graph gap (µs) |
|:----:|------|:-------------:|:---------------:|:--------------------:|
| 1 | warmup | 1752.2 | 1343.2 | 408.9 |
| 2 | profiled | 1748.3 | 1339.8 | 408.5 |
| 3 | profiled | 1715.8 | 1339.9 | 375.9 |
| 4 | profiled | 1711.7 | 1339.8 | 371.9 |
| 5 | profiled | 1716.6 | 1340.4 | 376.2 |
| 6 | profiled | 1697.5 | 1339.9 | 357.7 |
| 7 | profiled | 1703.9 | 1339.7 | 364.2 |
| **Avg (calls 2–7)** | | **1715.6 µs** | **1339.9 µs** | **375.7 µs** |

- **GPU span**: first kernel start to last kernel end.
- **Graph exec**: sum of all 13 CUDA graph execution durations per call.
- **Inter-graph gap**: GPU span − graph exec = GPU idle while CPU dispatches next graph.

---

## 4. Time Budget Breakdown (avg profiled call)

### Claude

| Layer | Time (µs) | % of GPU span |
|-------|:---------:|:-------------:|
| CUDA kernel compute (actual math) | 52.6 | 3.0% |
| Intra-graph GPU idle (sync within graphs) | 1289.6 | 74.7% |
| Inter-graph GPU idle (CPU dispatch latency) | 384.3 | 22.3% |
| **Total GPU span** | **1726.5** | 100% |

### GPT

| Layer | Time (µs) | % of GPU span |
|-------|:---------:|:-------------:|
| CUDA kernel compute (actual math) | 53.5 | 3.1% |
| Intra-graph GPU idle (sync within graphs) | 1286.4 | 75.0% |
| Inter-graph GPU idle (CPU dispatch latency) | 375.7 | 21.9% |
| **Total GPU span** | **1715.6** | 100% |

The GPU is executing actual compute for only **~3% of the GPU-side timeline.**  
The dominant cost (75%) is intra-graph idle — GPU waiting within each CUDA graph for
synchronization barriers to clear before the next kernel can launch.

**cuStreamSynchronize events** — syncType=2:

| Model | Total events | Per call | Avg duration |
|-------|:------------:|:--------:|:------------:|
| Claude | 226 | 32.3 | 44.8 µs |
| GPT | 226 | 32.3 | 44.8 µs |
| Gemini | 233 | 33.3 | **16.2 µs** |

Claude and GPT have the same sync event count as Gemini but each sync takes **2.8× longer**
(44.8 µs vs 16.2 µs), directly explaining the 3.7× larger intra-graph idle time.

---

## 5. Comparison: Claude / GPT vs Gemini vs ACC Fortran

| Metric | ACC Fortran | JAX Gemini | JAX Claude | JAX GPT |
|--------|:-----------:|:----------:|:----------:|:-------:|
| Physics subcycles | 10 | 11 | 11 | 11 |
| Unique kernel types | 17 | 5 | **2** | **2** |
| Kernel launches / call | 98 | 15 | 12 | 12 |
| **Kernel compute / call** | 0.610 ms | 0.059 ms | **0.053 ms** | **0.054 ms** |
| Intra-graph idle / call | ~0 ms | 0.351 ms | **1.290 ms** | **1.286 ms** |
| Inter-graph gap / call | ~0 ms | 0.355 ms | 0.384 ms | 0.376 ms |
| **Total GPU span / call** | **0.610 ms** | 0.765 ms | **1.727 ms** | **1.716 ms** |
| GPU compute utilization | ~100% | 7.7% | **3.0%** | **3.1%** |
| cuStreamSync avg | — | 16.2 µs | 44.8 µs | 44.8 µs |

**Key findings:**

- Claude and GPT kernel compute (~53 µs) is **slightly faster than Gemini's** (59 µs) and
  **11× faster than ACC Fortran's** (610 µs). XLA fusion is working well at the kernel level.
- However, intra-graph idle (~1289 µs) is **3.7× larger than Gemini's** (351 µs), making
  Claude/GPT **2.25× slower** than Gemini end-to-end and **2.83× slower** than ACC Fortran.
- The bottleneck is not GPU compute — it is synchronization overhead within CUDA graphs, driven
  by cuStreamSynchronize events that take 2.8× longer to resolve than in Gemini.

---

## 6. Why Claude / GPT Have More Intra-Graph Idle Than Gemini

The likely cause is how the subcycle while-loop body is compiled:

- **Gemini** translates the vertical (nz) loop with fully vectorized JAX operations (`vmap` or
  broadcasting over the level dimension). XLA compiles this into a compact fused kernel
  (`input_reduce_fusion`) with minimal synchronization barriers within each graph. Each
  cuStreamSynchronize resolves quickly (~16 µs) because the GPU has little work queued before
  the sync point.

- **Claude / GPT** use `lax.fori_loop` for the vertical loop. XLA compiles this as a sequential
  loop with synchronization checkpoints between iterations, producing more dependencies within
  each CUDA graph. Although the final kernel count is similar (12 `input_reduce_fusion` vs 12
  for Gemini), the graph execution includes more internal barriers — each sync event must wait
  longer (~45 µs) for the GPU to drain in-flight work before the next kernel can proceed.

This is consistent with the jax_trace findings (see `jax_trace` outputs), where Claude/GPT's
Perfetto timelines show per-level sequential dispatch patterns inside each CUDA graph, versus
Gemini's single-pass vectorized dispatch.

---

## 7. Comparison with ACC Fortran at ncol=1000

| Metric | ACC Fortran | JAX Claude | JAX GPT |
|--------|:-----------:|:----------:|:-------:|
| GPU kernel exec / call | 0.610 ms | 0.053 ms | 0.054 ms |
| Total GPU span / call | 0.610 ms | 1.727 ms | 1.716 ms |
| Speedup (kernel only) | 1× | **11.5×** | **11.3×** |
| Speedup (end-to-end GPU) | 1× | 0.35× (ACC faster) | 0.36× (ACC faster) |

Claude and GPT's XLA kernels execute the physics **~11× faster** than ACC Fortran.
However, the ~1.3 ms of intra-graph idle completely erases this advantage — ACC Fortran's
continuous GPU stream (no intra-kernel barriers) finishes in 0.61 ms while Claude/GPT take
~1.72 ms end-to-end.

At larger ncol (see §7.4–7.5 of `gemini_vs_acc_nsys_report_2026-04-27.md`), the kernel
compute time grows and sync overhead becomes relatively cheaper:
- At ncol=100,000: Claude 62.9 ms vs ACC GPU exec 54.4 ms (ACC still faster, ~15%)
- At ncol=1,000,000: Claude 632 ms vs ACC GPU exec 751 ms (**Claude 16% faster**)

The break-even for Claude/GPT vs ACC GPU exec is around ncol=500,000–1,000,000.
