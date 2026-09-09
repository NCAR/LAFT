# nsys Profiling Report — JAX Gemini `kessler_run_core` (Correct JD Inputs)

**Date:** 2026-04-24  
**Source:** `jax_gpu_jd_nsys.o5960321` · `gemini.nsys-rep` exported to SQLite with nsys 2025.6.1 (nvhpc/26.1)  
**Platform:** NVIDIA A100-SXM4-40GB · CUDA 13.0 · Derecho  
**Inputs:** `ncol=1000`, `nz=56`, `dt=60 s`, `qr = arr * 0.01` (Box-Muller, seed=42)  
**Calls:** 1 warmup + 6 profiled calls  
**Script:** `LLMx_jd_profiling_run.py --mode nsys --model gemini --ncol 1000`  

> **Previous nsys run (old_data/gemini.sqlite, 2026-04-08) used wrong inputs:**
> `qr=0.0001` (100× smaller) and `nz=72` via a stale inline PBS heredoc. That run produced
> only 1 physics subcycle (trivial CFL) and is invalid for comparison against ACC Fortran.
> This run uses the correct JD inputs and produces 11 physics subcycles.

---

## 1. CUDA Kernel Summary

| Kernel | Count (total) | Per call | Avg (µs) | Total (ms) |
|--------|:-------------:|:--------:|:--------:|:----------:|
| `input_reduce_fusion` | 84 | 12 | 3.7 | 0.314 |
| `input_reduce_select_fusion_1` | 7 | 1 | 7.6 | 0.054 |
| `loop_multiply_fusion` | 7 | 1 | 4.0 | 0.028 |
| `loop_convert_fusion` | 7 | 1 | 2.7 | 0.019 |
| `wrapped_broadcast` | 1 | — | 3.4 | 0.003 |
| **Total (per call)** | | **16** | | **0.059 ms** |

`wrapped_broadcast` fires once at process start (XLA initialization). The remaining 4 kernel
types fire every call in the same sequence:

```
loop_convert_fusion (×1)
  input_reduce_fusion (×12)          ← while_loop: 11 body iterations + 1 exit check
loop_multiply_fusion (×1)
input_reduce_select_fusion_1 (×1)
```

---

## 2. Physics Subcycle Count

Each call launches **13 CUDA graphs** in sequence:

| Graph slot | Count | GPU exec (µs) | Interpretation |
|:----------:|:-----:|:-------------:|----------------|
| 1st graph | 1 | ~50 | Pre-while initialization |
| Middle graphs | 11 | ~31 each | While-loop body — 1 graph per physics subcycle |
| Last graph | 1 | ~17 | Post-while finalization (precipitation accumulation) |

**11 physics subcycles confirmed**, consistent with the jax_trace finding from 2026-04-08.
The `input_reduce_fusion` kernel runs 12 times per call (11 true condition checks + 1 false
exit check), which is the while-loop CFL condition reduction.

---

## 3. Per-Call Execution Timing

| Call | Type | GPU span (µs) | CudaGraph GPU exec (µs) | Inter-graph gap (µs) |
|:----:|------|:-------------:|:------------------------:|:--------------------:|
| 1 | warmup | 780.7 | 413.1 | 367.6 |
| 2 | profiled | 778.1 | 410.8 | 367.3 |
| 3 | profiled | 757.3 | 410.1 | 347.2 |
| 4 | profiled | 765.7 | 410.4 | 355.3 |
| 5 | profiled | 746.0 | 410.5 | 335.5 |
| 6 | profiled | 783.0 | 410.3 | 372.7 |
| 7 | profiled | 758.9 | 410.1 | 348.8 |
| **Avg (calls 2–7)** | | **764.8 µs** | **410.4 µs** | **354.5 µs** |

- **GPU span**: time from first GPU operation to last GPU operation within the call (measured from CUPTI kernel timestamps).
- **CudaGraph GPU exec**: sum of all 13 graph execution durations (includes GPU idle within each graph).
- **Inter-graph gap**: GPU span − CudaGraph GPU exec = GPU idle time while the CPU dispatches the next graph.

---

## 4. Time Budget Breakdown (avg profiled call)

| Layer | Time (µs) | % of GPU span |
|-------|:---------:|:-------------:|
| CUDA kernel execution (actual math) | 59 | 7.7% |
| Intra-graph GPU idle (sync within graphs) | 351 | 45.9% |
| Inter-graph GPU idle (CPU dispatch latency) | 355 | 46.4% |
| **Total GPU span** | **765** | 100% |

The GPU is running actual compute for **only 7.7% of the GPU-side timeline.**  
The remaining 92.3% is synchronization and dispatch overhead.

**Stream synchronization events:** 233 type-2 (`cuStreamSynchronize`) across all 7 calls
= 33 per call, avg 16.2 µs each.

---

## 5. Comparison: Correct vs Wrong Inputs

| | Old nsys (2026-04-08) | New nsys (2026-04-24) |
|--|:---------------------:|:---------------------:|
| Input `qr` | `0.0001` (hardcoded) | `arr * 0.01` (Box-Muller, seed=42) |
| `nz` | 72 | 56 |
| Physics subcycles | **1** (trivial CFL) | **11** (correct) |
| Avg GPU span / call | ~810 µs | **765 µs** |
| CUDA kernel compute / call | ~28 µs | **59 µs** |

Despite 11× more subcycles, the wall time is nearly identical (~810 µs vs ~765 µs). This
confirms that GPU computation is not the bottleneck — sync and dispatch overhead
dominate, and that overhead scales weakly with subcycle count because it is incurred
per CUDA graph launch (of which there are 13 either way for the pre/post + loop structure).

---

## 6. Why JAX Has 1 More Subcycle Than ACC Fortran

From `subcycle_analysis_2026-04-24.md` Section 3 — both drivers use the same Box-Muller formula:

```
arr ~ N(mean=1, std=0.1)
qr = arr * 0.01
```

But with different seeds:
- Fortran uses `seed = [1, 2, ..., seed_size]`
- JAX uses `seed = 42`

The number of subcycles is determined by the maximum `qr` across all 1000 columns:

```
dt0 = 0.8 * min_k( dz_k / velqr(i,k) )    where velqr ∝ qr^0.1364
n_subcycles = ceil( dt / dt0 )
```

Higher `qr` → faster fall velocity → smaller `dt0` → more subcycles needed. With 1000 columns
each drawing from N(0.01, 0.001), the extreme maximum value drawn depends on the seed.
Fortran's seed happens to draw a slightly smaller `max(qr)` → `dt0 ≈ 6.0 s` → 10 subcycles.
JAX's seed=42 draws a slightly larger `max(qr)` → `dt0 ≈ 5.5 s` → 11 subcycles.

This is normal statistical variation — both seeds sample the same distribution, they just land
on different tails. It is not a bug or algorithmic difference. Using a shared input file
(generate once, both sides load the same array) eliminates this entirely.

---

## 7. Comparison: JAX Gemini (nsys) vs ACC Fortran

From `subcycle_analysis_2026-04-24.md` (ACC Fortran nsys, same ncol=1000, `qr~0.01`):

| Metric | ACC Fortran | JAX Gemini |
|--------|:-----------:|:----------:|
| Physics subcycles | 10 | 11 |
| Total GPU-side time / call | **0.610 ms** | 0.765 ms |
| Actual kernel compute / call | 0.610 ms | **0.059 ms** |
| Sync + dispatch overhead | ~0 ms | 0.706 ms |
| GPU utilization | ~100% | **7.7%** |

JAX Gemini's XLA kernels execute the physics **10× faster** than ACC Fortran
(0.059 ms vs 0.610 ms for ~11 vs 10 subcycles of equivalent physics). However,
0.706 ms of synchronization and dispatch overhead completely erases this advantage,
leaving JAX 1.25× slower end-to-end on the GPU side.

> The root cause is XLA's current dispatch model for `lax.while_loop`: each loop
> body is submitted as a separate CUDA graph, requiring CPU intervention between
> each of the 11 subcycle iterations. ACC Fortran's OpenMP target offload runs the
> entire while-loop body in a single continuous GPU kernel without CPU round-trips.
