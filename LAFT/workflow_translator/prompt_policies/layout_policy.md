## Layout policy (Bridge Mode)

The translated code operates on row-major (C-order) arrays — standard JAX/NumPy layout.

**How the bridge works:**
The bridge physically transposes all 2D arrays from Fortran layout to JAX layout
BEFORE calling your function, and transposes back AFTER.

- Fortran layout : (ncol, nz)  — first index is columns
- JAX layout     : (nz, ncol)  — first index is levels

This means a Fortran array declared as `a(ncol, nz)` arrives in your JAX function
with shape `(nz, ncol)`. **The indices are swapped.**

**Translation rules:**
- Fortran `a(col, klev)`  →  JAX `a[klev-1, col-1]`  (indices SWAPPED)
- Fortran `a(i, j)`       →  JAX `a[j-1, i-1]`       (indices SWAPPED)
- NO swapaxes, NO layout conversions in your code
- NO transpose logic — bridge handles all layout conversions

**Integer index parameters (CRITICAL):**
Fortran integer parameters that represent level or column indices (e.g., `i_start`, `i_end`,
`i_step`) are passed in with **1-based Fortran values**. Convert them to 0-based Python indices
before using them as array indices or loop bounds:
```python
i_start_idx = i_start - 1   # 1 → 0
i_end_idx   = i_end   - 1   # nz → nz-1
# Then use i_start_idx / i_end_idx everywhere: array indices, fori_loop bounds, slices
```
Failure to do this causes loops to run on the wrong index range (1..nz instead of 0..nz-1),
leaving level 0 uninitialized and producing out-of-bounds accesses at level nz.

**Bridge layer (separate from your code):**
- Input:  Fortran (ncol, nz) → physically transposes → JAX (nz, ncol)
- Output: JAX (nz, ncol)     → physically transposes → Fortran (ncol, nz)

**Your job:** Translate the COMPUTATION only, using (nz, ncol) row-major indexing.

**Example:**
```fortran
! Fortran: a is declared a(ncol, nz)
a(col, klev) = b(col, klev) * 2.0
```

```python
# JAX: a arrives as shape (nz, ncol) — indices are swapped
a = a.at[klev-1, col-1].set(b[klev-1, col-1] * 2.0)
```

NO transpose logic needed — bridge handles it!
