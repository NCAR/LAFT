# Kessler Run — Scalability Comparison: All Implementations

**Generated:** 2026-04-28  
**Grid:** nz=56, dt=60 s  
**Runtimes:** median ± std (ms) for JAX/Fortran; mean (ms) for OpenACC wall-clock; single nsys run for OpenACC GPU exec  
**Reps:** 5 timed runs per ncol after 1 warmup

---

## Runtime Table (ms)

| ncol | Fortran<br>CPU serial | OpenACC<br>wall-clock† | OpenACC<br>GPU exec‡ | Claude<br>JAX/GPU | GPT<br>JAX/GPU | Gemini<br>JAX/GPU | Qwen<br>JAX/GPU |
|-----:|---------------------:|----------------------:|---------------------:|------------------:|---------------:|------------------:|----------------:|
| 50 | 2.419 ± 0.011 | 337.241 | 0.218 | 1.250 ± 0.094 | 1.091 ± 0.029 | 0.738 ± 0.029 | 25.824 ± 0.035 |
| 100 | 4.819 ± 0.023 | 342.081 | 0.283 | 1.264 ± 0.020 | 1.304 ± 0.025 | 0.742 ± 0.023 | 51.229 ± 0.022 |
| 200 | 9.708 ± 0.014 | 344.941 | 0.330 | 1.544 ± 0.024 | 1.542 ± 0.022 | 0.882 ± 0.020 | 109.822 ± 0.075 |
| 500 | 24.652 ± 0.013 | 354.650 | 0.366 | 1.641 ± 0.024 | 1.647 ± 0.018 | 0.880 ± 0.040 | 285.878 ± 0.088 |
| 1,000 | 49.669 ± 0.007 | 351.442 | 0.470 | 1.920 ± 0.014 | 1.919 ± 0.027 | 1.005 ± 0.024 | 581.351 ± 0.048 |
| 2,000 | 99.370 ± 0.009 | 355.178 | 0.871 | 2.693 ± 0.023 | 2.595 ± 0.024 | 1.254 ± 0.026 | 1163.156 ± 0.307 |
| 5,000 | 248.778 ± 0.027 | 384.158 | 1.453 | 4.495 ± 0.018 | 4.459 ± 0.020 | 1.982 ± 0.025 | 2908.149 ± 0.991 |
| 10,000 | 496.194 ± 0.025 | 396.487 | 2.957 | 7.316 ± 0.054 | 7.259 ± 0.028 | 3.050 ± 0.037 | 6271.749 ± 0.561 |
| 100,000 | 4985.835 ± 0.167 | 618.970 ⚠ | 54.377 | 62.869 ± 1.175 | 60.959 ± 1.287 | 16.302 ± 0.050 | N/A |
| 1,000,000 | 54546.955 ± 22.473 | 3069.902 ⚠ | 751.161 | 632.059 ± 0.060 | 619.743 ± 0.184 | 152.637 ± 0.143 | N/A |

---

## Gemini JAX vs OpenACC GPU Exec (kernel-only, apples-to-apples)

Speedup = OpenACC GPU exec / Gemini JAX. Values > 1 mean OpenACC kernel is faster; values < 1 mean Gemini is faster.

| ncol | OpenACC GPU exec (ms) | Gemini JAX (ms) | Speedup | Winner |
|-----:|---------------------:|----------------:|--------:|:-------|
| 50 | 0.218 | 0.738 | 0.30× | **OpenACC 3.4×** |
| 100 | 0.283 | 0.742 | 0.38× | **OpenACC 2.6×** |
| 200 | 0.330 | 0.882 | 0.37× | **OpenACC 2.7×** |
| 500 | 0.366 | 0.880 | 0.42× | **OpenACC 2.4×** |
| 1,000 | 0.470 | 1.005 | 0.47× | **OpenACC 2.1×** |
| 2,000 | 0.871 | 1.254 | 0.69× | **OpenACC 1.4×** |
| 5,000 | 1.453 | 1.982 | 0.73× | **OpenACC 1.4×** |
| 10,000 | 2.957 | 3.050 | 0.97× | ~equal |
| 100,000 | 54.377 | 16.302 | 3.34× | **Gemini 3.3×** |
| 1,000,000 | 751.161 | 152.637 | 4.92× | **Gemini 4.9×** |

**Crossover at ncol ≈ 10,000.** OpenACC compiled kernels are faster at small ncols; Gemini JAX scales significantly better above 10,000 columns, reaching ~5× faster at 1M. JAX's vmap parallelizes efficiently across large column counts, while OpenACC kernel performance degrades at 1M likely due to memory bandwidth saturation.

---

## Column Notes

**† OpenACC wall-clock** — full binary process time: CPU array allocation + Box-Muller RNG + H2D transfer + GPU kernels + D2H transfer. Compiled ahead-of-time with nvhpc (no per-ncol JIT cost). The large baseline (~337 ms even at ncol=50) reflects GPU context initialization and fixed binary overhead.

⚠ At ncol ≥ 100,000 the wall-clock is dominated by H2D/D2H transfers (~448 MB per array at 1M cols). **Do not compare against JAX cached-execute times**, which pre-stage arrays on GPU and pay no transfer cost.

**‡ OpenACC GPU exec** — pure CUDA kernel time from `nsys stats cuda_gpu_kern_sum` (single nsys-profiled run). This is the apples-to-apples comparison against JAX cached-execute times, as both measure only the physics kernel without host-device transfers.

**JAX/GPU columns** — JIT-compiled, arrays resident on GPU. Timing excludes H2D transfer (input arrays pre-staged before the timed loop). Values are median ± std over 5 reps.

**Qwen N/A** — Qwen's sequential (non-vmap) implementation makes JIT compilation at ncol > 10,000 infeasible within job walltime.

---

## Sources

| Data | Source |
|------|--------|
| Fortran CPU serial | `_compare_results/outputs/scalability/fortran_scalability_results.json` |
| JAX LLM timings | `_compare_results/outputs/scalability/LLMs_scalability_results.json` |
| OpenACC wall-clock | `acc_kessler_profiling.py --mode ncol_sweep` in `data/exp2_jd/acc_src/` (steady-state wall-clock table) |
| OpenACC GPU exec | same sweep, nsys pass (`cuda_gpu_kern_sum`); the OpenACC source and runner are in `data/exp2_jd/acc_src/` |
