# Bridge Reference — Reading the Bridge Analysis Reports

> **Where this fits:** stage 1 of the pipeline. `../ORCHESTRATOR.md` owns stage
> sequencing; `BRIDGE_WORKFLOW.md` is the authoritative procedure to follow.
> This document is the deep-dive reference for interpreting what the bridge
> stage emits.

Examples come from the Kessler case study; the same reports are produced per
procedure in every project.
Platform: NCAR Derecho (NVIDIA A100 GPUs)

## Overview

This guide explains how to interpret the analysis reports generated for each procedure.
Individual reports are in `out/reports/bridge/<procedure>_analysis.txt`.

---

## Report Structure

Each analysis report contains:

1. **Bridge Strategy** - The approach used (device-side layout conversion + JIT)
2. **Loop Swap Safety Analysis** - Per-loop analysis of optimization opportunities
3. **GPU Efficiency Warning** - Hardware-specific performance guidance
4. **Performance Expectations** - Memory usage estimates

---

## Section 1: Bridge Strategy

```
BRIDGE STRATEGY:
  ✓ Device-side layout conversion (PATH C)
  ✓ Pure H2D/D2H — arrays ship to the device unchanged
  ✓ Axis reversal INSIDE the jitted device wrapper (fused by XLA)
  ✓ ONE batched jax.device_get for all outputs (single sync)
  ✓ ALWAYS use JIT compilation
```

**What this means:**
- Host arrays cross the bridge **as-is** (no host-side transpose or copy)
- The permutation between Fortran `(ncol, nz)` index order and the kernel's
  `(nz, ncol)` layout is real data movement, but it runs **on the device**,
  inside the jitted region — XLA fuses it into the consuming kernel
- Outputs come back the same way: reversed in-jit, then pure D2H, so the
  bridge still hands the driver standard C-order NumPy `(ncol, nz)` arrays
- All device-resident outputs (arrays and numeric scalars) are fetched in
  **one batched `jax.device_get`** — a single stream sync per bridge call
  instead of one blocking round-trip per output (per-array `np.array(...)`
  cost ~16 syncs/call on Kessler at ncol=1000; the batch collapses them)
- The bridge's public signature and returns are unchanged by this strategy
- JIT compilation makes subsequent calls ~1000x faster

**Why we do this:**
- **Correctness:** The axis reversal preserves Fortran semantics exactly
  (bit-identical to the host-transpose strategy it replaced); the batched
  output fetch is also bit-identical to the per-array path it replaced
  (verified on CPU and A100)
- **Performance:** The device permutes at HBM bandwidth (~1.5 TB/s) instead
  of host memcpy speed (~2 GB/s measured); the host transpose was the
  dominant bridge cost at large problem sizes. The batched `device_get`
  collapses one blocking round-trip per output into a single sync with
  overlapped transfers — measured on Kessler (A100, 2026-08-11, same-session
  A/B, 5-rep medians): **1.34× end-to-end at ncol=1e4** (13.54→10.08 ms) and
  **1.17× at ncol=1e6** (1.71→1.46 s, −247 ms/call); evidence in the
  project's `out/reports/bridge/perf_ab_batched_d2h_results.json`
- **Simplicity:** No conditional logic, same path every time
- **Portability:** Works on CPU and GPU without code changes (the transpose
  is pure data movement — no FP ops, numerics unchanged on any backend)

---

## Section 2: Loop Swap Safety Analysis

This section analyzes **individual nested DO loops** to identify optimization opportunities.

### Example: PROBABLY_SAFE Loop

```
Loop 1 (Lines 81-195):
  DO col = ...
    DO klev = ...
      f5 = 4093.0 * lv / cpair(col,klev)
      ...
    END DO
  END DO

  Safety: PROBABLY_SAFE
  Reasons:
    • No obvious unsafe patterns detected
  Warnings:
    ⚠️  Manual verification still required

  💡 OPTIMIZATION OPPORTUNITY (NOT IMPLEMENTED):
     Could swap to: DO klev=...; DO col=...
     - Would eliminate transpose operations
     - Would save memory copies
     - BUT: Requires manual verification
     - DECISION: Safety > Performance (not implemented)
```

**What "PROBABLY_SAFE" means:**
- The analyzer found **no obvious problems** (no dependencies, reductions, or I/O)
- The loop **could potentially** be swapped without breaking correctness
- **Manual verification by domain expert is still required**
- The bridge does **NOT** swap loops automatically (safety first)

### Example: DEFINITELY_UNSAFE Loop

```
Loop 2 (Lines 120-135):
  DO i = ...
    DO j = ...
      total = total + array(i,j)  ← Reduction detected
    END DO
  END DO

  Safety: DEFINITELY_UNSAFE
  Reasons:
    • Reduction/accumulation operations detected
  Warnings:
    ⚠️  Floating-point order would change

  ✗ CANNOT SWAP:
     Loop swapping would produce incorrect results
```

**What "DEFINITELY_UNSAFE" means:**
- The analyzer detected **reduction operations** (sum, accumulate, total)
- Swapping loops would change **floating-point arithmetic order**
- Different order = different roundoff errors = **wrong physics results**
- Transpose is **mandatory** for correctness

**Why order matters:**
```fortran
! Original order:
sum = (a + b) + (c + d)  = 10.0000001

! Swapped order:
sum = (a + c) + (b + d)  = 10.0000003  ← Different!
```

### Example: UNKNOWN Safety

```
Loop 3 (Lines 88-94):
  DO i = ...
    DO j = ...
      result(i,j) = compute_physics(i, j, state)
    END DO
  END DO

  Safety: UNKNOWN
  Reasons:
    • Function calls detected
  Warnings:
    ⚠️  Cannot analyze function internals

  ⚠️  CANNOT VERIFY:
     Function might use global state or have dependencies
     Manual inspection of compute_physics required
```

**What "UNKNOWN" means:**
- The analyzer **cannot determine** if swapping is safe
- Function calls, I/O, or global state detected
- Would require analyzing called functions (not implemented)
- Treated as **unsafe** (conservative approach)

### Detected Patterns

| Pattern | Classification | Example |
|---------|---------------|---------|
| Array dependencies | DEFINITELY_UNSAFE | `a(i,j) = a(i-1,j)` |
| Reductions | DEFINITELY_UNSAFE | `sum = sum + a(i,j)` |
| Function calls | UNKNOWN | `CALL compute(...)` |
| I/O operations | UNKNOWN | `WRITE(*,*) i, j` |
| Module state | UNKNOWN | `USE physics_module` |
| Independent ops | PROBABLY_SAFE | `c(i,j) = a(i,j) * b(i,j)` |

---

## Section 3: GPU Efficiency Warning (Derecho A100)

### Hardware Specifications

**Platform:** NCAR Derecho Supercomputer
**GPU:** NVIDIA A100 (Ampere architecture)
- **Memory:** 40 GB or 80 GB HBM2e
- **Memory Bandwidth:** 1.2 TB/s (peak)
- **Compute:** 9.7 TFLOPS (FP64), 19.5 TFLOPS (with tensor cores)
- **Interconnect:** PCIe Gen4 (~50 GB/s effective bandwidth)

### Performance Characteristics

**A100 Overhead Breakdown:**
```
Kernel launch:        ~15μs
PCIe setup:           ~30μs
Data transfer:        Depends on array size
```

### Example Warning (Small Arrays)

```
GPU EFFICIENCY WARNING:
  ⚠️  Arrays are very small (~78.1 KB)
  ⚠️  Derecho A100 GPU overhead (~47μs: 15μs launch + 30μs setup + 2μs transfer)
      likely exceeds compute time
  ⚠️  Recommendation: Use CPU execution for this workload
```

**What this means:**
- Your arrays are **too small** to benefit from GPU acceleration
- GPU overhead (kernel launch + PCIe transfer) dominates actual computation
- **CPU would be faster** for this problem size

**Why small arrays are inefficient on GPU:**

```
GPU Timeline (78 KB arrays):
├─ Kernel launch:     15μs  ┐
├─ PCIe transfer in:   1μs  │
├─ Computation:      0.2μs  ├─ Overhead: 47μs
├─ PCIe transfer out:  1μs  │  Compute: 0.2μs
└─ Total:            ~47μs  ┘  Efficiency: 0.4%!

CPU Timeline (78 KB arrays):
├─ Computation:      10μs  ← Direct compute, no overhead
└─ Total:            10μs     Efficiency: 100%

Result: CPU is 4-5x faster for small arrays!
```

### When to Use GPU vs CPU

| Array Size | GPU Overhead | Recommendation |
|------------|--------------|----------------|
| < 100 KB | Dominates | ✅ **Use CPU** |
| 100 KB - 1 MB | Significant | ⚠️  Profile first |
| 1 MB - 10 MB | Moderate | 🟡 GPU may help |
| > 10 MB | Negligible | ✅ **Use GPU** |

**For atmospheric models:**
- **Small test cases** (ncol=10, nz=72): ~6 KB → Use CPU
- **Column physics** (ncol=100, nz=72): ~60 KB → Use CPU  
- **Regional domains** (ncol=10,000, nz=72): ~6 MB → GPU beneficial
- **Global models** (ncol=100,000, nz=72): ~60 MB → GPU strongly recommended

---

## Section 4: Performance Expectations

```
PERFORMANCE EXPECTATIONS:
  cpair: ~7.6 MB (typical)
  rair: ~7.6 MB (typical)
  ...
```

**Important:** These are **estimates**, not actual runtime sizes!

**How estimates are calculated:**
- Assumes `typical_n = 1000` for each dimension
- For 2D array: `1000 × 1000 × 8 bytes = 7.6 MB`
- For your actual problem (ncol=100, nz=72): `100 × 72 × 8 bytes = ~56 KB`

**Why we can't show exact sizes:**
- Array dimensions are **runtime values** (not known at analysis time)
- The analyzer only knows arrays are 2D (from type information)
- Actual sizes depend on problem configuration

**To get accurate estimates:**
- Update `typical_n` parameter for your domain
- For atmospheric models, use `ncol×nz` from your typical run configuration

---

## Using This Information

### For Computational Scientists

1. **Review loop safety analysis** for your procedures
2. **Check GPU efficiency warnings** for your typical problem sizes
3. **Understand why** the bridge converts layout on the device (correctness first, at HBM bandwidth)
4. **Don't worry** about optimization - the bridge handles it

### For Performance Engineers

1. **Identify PROBABLY_SAFE loops** for potential hand-optimization
2. **Profile actual runs** to validate GPU efficiency predictions
3. **Consider problem size** when choosing CPU vs GPU execution
4. **Verify physics** if you decide to manually swap loops

### For Code Developers

1. **The bridge always works correctly** - no manual intervention needed
2. **Reports are educational** - they explain what's happening
3. **Safety first** - bridge never applies unsafe optimizations
4. **Performance is good** - JIT + device-side layout conversion; no host permute cost

---

## Key Takeaways

✅ **Bridge is conservative** - Layout conversion is always applied, device-side (guaranteed correct)
✅ **Reports are informative** - Explain loop-by-loop optimization opportunities  
✅ **A100-specific** - Performance analysis tailored to Derecho hardware
✅ **No action required** - Bridge works correctly automatically

🎯 **Bottom line:** Trust the bridge, use the reports to understand your code!

---

## Questions?

- **Bridge not working?** Check that JAX translations are in `out/jax/`
- **Wrong performance?** Verify your array sizes match typical use case
- **Want to optimize?** Manually verify PROBABLY_SAFE loops, then modify Fortran source

Generated by: phase03_make_bridge.py
Platform: NCAR Derecho (NVIDIA A100)
