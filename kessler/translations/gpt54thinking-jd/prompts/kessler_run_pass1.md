# Fortran → Python/JAX Translation — Pass 1: First Draft

Translate the Fortran procedure below into Python with a two-layer structure:
- `kessler_run_core(...)`: JAX-safe pure compute (no I/O, no side effects), decorated with `@functools.partial(jax.jit, static_argnames=(...))`
- `kessler_run(...)`: Python wrapper

## Hard constraints
- NO NumPy: do not import or use `numpy` at all.
- NO in-place mutation. If Fortran writes to an argument, return the updated value(s).
- The core must be `jax.jit`-compatible: no Python loops over arrays, no host conversions inside core.
- Do not invent globals. Only use what's in the Fortran unit.
- This two-layer jitted contract applies to EVERY procedure with array
  arguments — including a scheme's top-level main routine and model-facing
  wrappers. "It mostly orchestrates other calls" is NOT an exemption (see
  Orchestration Boundary Rules below).

## Output format
Return Python code only (no explanations). Include:
1) Imports (os.environ line first, then functools, then jax)
2) Module-level docstring
3) `kessler_run_core(...)` with `@functools.partial(jax.jit, static_argnames=(...))`
4) `kessler_run(...)` wrapper

## Required imports (MANDATORY - every generated file must start with this)
```python
import os
os.environ["JAX_ENABLE_X64"] = "1"

import functools
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
```

The `os.environ` line MUST come before `import jax`. Without this, float64 silently truncates to float32.
`import functools` is required for the `@functools.partial(jax.jit, ...)` decorator on `_core`.

## Orchestration Boundary Rules (CRITICAL)

These rules define which procedures may be translated as host-side (plain
Python) layers and which MUST receive the full JAX two-layer contract
(`<proc>_core` decorated with `@jax.jit` + wrapper). They exist to close a
specific failure mode: a scheme's top-level routine being emitted as
host-side Python loops that call jitted helpers — code that produces correct
numbers but serializes the grid on CPU, is invisible to XLA-level profiling,
and destroys GPU throughput.

---

### The invariant

> **Any procedure whose signature carries grid/state arrays (a non-scalar
> procedure) gets the full JAX contract — jitted `_core` + wrapper — no
> matter where it sits in the call graph.**

Position grants no exemption. In particular, ALL of the following are
non-scalar COMPUTE procedures, even though they "orchestrate" other
procedures:

- the scheme's **main/top-level routine** (the one the driver calls per step),
- **model-facing wrappers** that marshal fields between a host model layout
  and the scheme (unit conversions, array packing/unpacking ARE grid
  compute),
- any intermediate routine that loops over columns, levels, categories, or
  any other axis of the physics state while doing numeric work.

If a routine both sequences calls AND does per-element numeric work, it is a
COMPUTE procedure. "It mostly just calls other things" is not an exemption —
the per-element work must move inside the JIT boundary (vectorized `jnp`
ops, `jax.vmap`, `lax.fori_loop`/`lax.scan` — see the VECTORIZATION PRIORITY
LADDER in the vectorization rules).

---

### What host-side layers are allowed to be

A procedure may be translated as a host-side plain-Python layer ONLY if it
is **scalar-only**: no grid/state arrays in its signature. Typical
legitimate host-side layers:

- configuration readers (namelist / config-file parsing),
- init-time lookup-table file loaders (`has_io=True`; NumPy allowed, and
  vectorized parsing such as `np.loadtxt`/`np.frombuffer` is preferred over
  element loops),
- init sequencing that sets scalar module constants.

Everything else compiles.

---

### Forbidden in ANY translated procedure (host code included)

1. **A Python `for`/`while` loop that writes array elements**
   (`x[i] = ...`, `x[i] += ...`, `x[i, k] = ...`). Element-wise mutation
   over a grid axis is compute; it belongs inside a jitted `_core` as a
   vectorized/whole-array operation.

2. **A Python `for`/`while` loop wrapping any JAX construct**
   (`jnp.*`, `lax.*`, `jax.*`, or a call to any `<proc>_core`). This is
   Level 5 of the VECTORIZATION PRIORITY LADDER — never acceptable. A
   Python loop over columns calling a jitted helper per column serializes
   the grid exactly like a `lax.fori_loop` over columns, with kernel-launch
   overhead added on top.

The one exception for pattern 1: scalar-only I/O loaders (`has_io=True`)
filling arrays from file records — and even there, prefer vectorized
parsing.

These two patterns are enforced mechanically by the lint (they are hard
FAILs, not warnings), and the dependency phase refuses to classify any
non-scalar procedure as host-side. Do not attempt to work around either
gate; if a routine seems impossible to jit, that is a design problem to
raise, not a classification to relax.

---

### Why this matters (the P3 lesson)

In an earlier translation, the scheme's ~2000-line main routine was
classified as a host-side "orchestration" layer. Every per-column/per-level
physics loop ran as Python `for` loops over NumPy arrays, calling dozens of
tiny jitted kernels per element. The result validated bit-for-bit against
Fortran — and was structurally incapable of GPU batching: XLA-level
profiling could not even see the defect, because the loops never compiled.
The fix required a full rewrite. These rules make that classification
impossible from the start.

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

## Fortran Intrinsics → JAX Mapping (MANDATORY)

Fortran has intrinsic functions that do NOT map directly to Python builtins.
Use the JAX equivalents below. Mistranslating these causes silent numerical errors.

| Fortran intrinsic         | JAX equivalent                                      | Notes |
|---------------------------|-----------------------------------------------------|-------|
| `MAX(a, b)`               | `jnp.maximum(a, b)`                                 | element-wise; not `max()` |
| `MIN(a, b)`               | `jnp.minimum(a, b)`                                 | element-wise; not `min()` |
| `ABS(x)`                  | `jnp.abs(x)`                                        | |
| `EXP(x)`                  | `jnp.exp(x)`                                        | |
| `SQRT(x)`                 | `jnp.sqrt(x)`                                       | |
| `LOG(x)`                  | `jnp.log(x)`                                        | natural log |
| `MOD(a, b)`               | `jnp.mod(a, b)`                                     | sign follows dividend |
| `MODULO(a, b)`            | `a % b`                                             | result always same sign as b |
| `DIM(a, b)`               | `jnp.maximum(a - b, 0.0)`                           | positive difference; **NOT** `a - b` |
| `MERGE(t, f, mask)`       | `jnp.where(mask, t, f)`                             | |
| `NINT(x)`                 | `jnp.round(x).astype(jnp.int32)`                   | nearest integer |
| `INT(x)`                  | `x.astype(jnp.int32)`                               | truncation toward zero |
| `REAL(x)` / `DBLE(x)`    | `x.astype(jnp.float64)`                             | |
| `SIGN(a, b)`              | `jnp.sign(b) * jnp.abs(a)`                         | magnitude of a, sign of b |
| `SUM(a)` / `SUM(a, dim)` | `jnp.sum(a)` / `jnp.sum(a, axis=dim-1)`            | axis is 0-based |
| `MAXVAL(a)`               | `jnp.max(a)`                                        | |
| `MINVAL(a)`               | `jnp.min(a)`                                        | |
| `ANY(mask)`               | `jnp.any(mask)`                                     | |
| `ALL(mask)`               | `jnp.all(mask)`                                     | |
| `MATMUL(a, b)`            | `jnp.matmul(a, b)`                                  | |
| `TRANSPOSE(a)`            | `jnp.transpose(a)` or `a.T`                         | |
| `RESHAPE(a, shape)`       | `jnp.reshape(a, shape)`                             | C-order by default |

**Common trap — `DIM(a, b)`:**
```fortran
! Fortran
result = MIN(dt0 * ... * DIM(sat_val, field) / ..., tracer)
```
```python
# ✅ Correct
result = jnp.minimum(dt0 * ... * jnp.maximum(sat_val - field, 0.0) / ..., tracer)
# ❌ Wrong — DIM is NOT simple subtraction
result = jnp.minimum(dt0 * ... * (sat_val - field) / ..., tracer)
```

## Return Type Rules (CRITICAL)

**Rule 1: Match the return count to the signature**

If signature policy says:
```
Return updated values: theta, field_a, field_b, field_c, accum, scalar_out, scheme_name, errmsg, errflg
```

Then BOTH wrapper and core must return EXACTLY these values:
```python
def proc_core(...):
    return theta, field_a, field_b, field_c, accum, scalar_out, scheme_name, errmsg, errflg  # ✅ 9 returns

def proc(...):
    return theta, field_a, field_b, field_c, accum, scalar_out, scheme_name, errmsg, errflg  # ✅ 9 returns
```

**Rule 2: Scalar outputs are RETURNED, not passed as parameters**

```python
# ❌ Wrong
def proc(n, result):      # result is an output, should not be a parameter
    result = compute(n)
    return None

# ✅ Correct
def proc(n):
    result = compute(n)
    return result
```

**Rule 3: Arrays modified in-place must be returned**

Fortran INOUT arrays must be returned (JAX does not allow mutation):
```python
def proc(a, n):
    a = a.at[0].set(n)    # functional update
    return a              # ✅ return the new array
```

**Rule 4: Core returns ONLY JAX-compatible types**

Core can return:   ✅ JAX arrays, ✅ Python numbers, ✅ JAX scalars
Core CANNOT return: ❌ Strings, ❌ None (if signature says return something)

Wrapper can return strings; core cannot.


======================================================================
MODULE VARIABLE POLICY (CRITICAL - READ CAREFULLY)
======================================================================

This procedure uses MODULE variables from `kessler`:
  - lv
  - pref
  - rhoqr

## MODULE VARIABLE POLICY (INOUT Pattern)

**CRITICAL: MODULE variables are INOUT parameters, not hidden state.**

When a Fortran procedure uses MODULE variables, treat them as regular INOUT parameters:

### Pattern

**Fortran:**
```fortran
MODULE my_module
  REAL :: lv, pref, rhoqr
END MODULE

SUBROUTINE init(lv_in, pref_in, rhoqr_in, errmsg, errflg)
  USE my_module
  lv = lv_in
  pref = pref_in / 100.0  ! Convert Pa to hPa
  rhoqr = rhoqr_in
END SUBROUTINE

SUBROUTINE compute(ncol, theta, ...)
  USE my_module
  ! Uses lv, pref, rhoqr for computation
END SUBROUTINE
```

**JAX Translation:**
```python
# INIT procedure: MODULE vars as INOUT parameters
def init_core(lv_in, pref_in, rhoqr_in, errmsg, errflg, lv, pref, rhoqr):
    '''
    MODULE VARIABLES (from my_module):
        lv: MODULE variable (INOUT)
        pref: MODULE variable (INOUT)
        rhoqr: MODULE variable (INOUT)
    '''
    lv     = jnp.asarray(lv_in,    dtype=jnp.float64)
    pref   = jnp.asarray(pref_in,  dtype=jnp.float64) / 100.0
    rhoqr  = jnp.asarray(rhoqr_in, dtype=jnp.float64)
    errflg = jnp.asarray(0, dtype=jnp.int32)
    return errflg, lv, pref, rhoqr

def init(lv_in, pref_in, rhoqr_in, errmsg, errflg, lv, pref, rhoqr):
    '''
    MODULE VARIABLES (from my_module):
        lv: MODULE variable (INOUT)
        pref: MODULE variable (INOUT)
        rhoqr: MODULE variable (INOUT)
    '''
    errflg, lv, pref, rhoqr = init_core(lv_in, pref_in, rhoqr_in, errmsg, errflg, lv, pref, rhoqr)
    errmsg = ""
    return errmsg, errflg, lv, pref, rhoqr


# COMPUTE procedure: MODULE vars as INOUT parameters
def compute_core(ncol, theta, ..., lv, pref, rhoqr):
    '''
    MODULE VARIABLES (from my_module):
        lv: MODULE variable (INOUT)
        pref: MODULE variable (INOUT)
        rhoqr: MODULE variable (INOUT)
    '''
    # Use MODULE vars in computation — they are just regular parameters.
    # MODULE vars typically do not change in COMPUTE procedures,
    # but we still return them for consistency.
    return theta_out, ..., lv, pref, rhoqr

def compute(ncol, theta, ..., lv, pref, rhoqr):
    '''
    MODULE VARIABLES (from my_module):
        lv: MODULE variable (INOUT)
        pref: MODULE variable (INOUT)
        rhoqr: MODULE variable (INOUT)
    '''
    theta_out, ..., lv, pref, rhoqr = compute_core(ncol, theta, ..., lv, pref, rhoqr)
    return theta_out, ..., lv, pref, rhoqr
```

### Key Rules

1. **MODULE vars appear in BOTH wrapper AND core signatures**
2. **MODULE vars are ALWAYS returned** (even if unchanged)
3. **Treat MODULE vars like any other INOUT parameter**
4. **No hidden state** - driver manages MODULE vars explicitly
5. **The MODULE variable list in the prompt is exhaustive — never add parameters.**
   Do NOT add a parameter for a variable that appears only in Fortran comments
   or commented-out code (dead code) — e.g. a disabled lookup-table branch.
   The bridge layer is generated from the same list, so an invented parameter
   makes the bridge call fail with a signature mismatch. If commented-out code
   references a module variable, omit it: translate only the live code path.

### Driver Flow (for reference)

```python
# INIT: Get initial MODULE vars
lv, pref, rhoqr, errmsg, errflg = init_bridge(lv_in, pref_in, rhoqr_in, "", 0)

# COMPUTE: Pass and receive MODULE vars
theta, ..., lv, pref, rhoqr = compute_bridge(..., lv, pref, rhoqr)
```

The bridge layer handles layout conversion, but MODULE vars flow through as regular INOUT parameters.

======================================================================


## Signature policy (Bridge Mode)
- Wrapper signature MUST be: `kessler_run(ncol, nz, dt, lyr_surf, lyr_toa, cpair, rair, rho, z, pk, theta, qv, qc, qr, precl, relhum, scheme_name, errmsg, errflg, lv, pref, rhoqr)`
- Core signature MUST be: `kessler_run_core(ncol, nz, dt, lyr_surf, lyr_toa, cpair, rair, rho, z, pk, theta, qv, qc, qr, precl, relhum, scheme_name, errmsg, errflg, lv, pref, rhoqr)`
- Use these signatures EXACTLY — do not add, drop, or reorder parameters.
- Do NOT add parameters for variables that appear only in Fortran comments
  or commented-out code (dead code). The bridge layer is generated from the
  signature above; an invented parameter breaks the bridge call.

**MODULE VARIABLES (from kessler):**
  - `lv`: MODULE variable (INOUT)
  - `pref`: MODULE variable (INOUT)
  - `rhoqr`: MODULE variable (INOUT)

MODULE variables are INOUT parameters - include in signature and return them!

- Return updated values: `theta, qv, qc, qr, precl, relhum, scheme_name, errmsg, errflg, lv, pref, rhoqr`

**Important:** Do NOT include dropped scalar output-only flags in parameters.

## Error handling policy (MANDATORY)
Many Fortran routines signal errors like this:
- `WRITE(errmsg, ...)`
- `errflg = 1`
- `RETURN`
Rules:
- `<PROC>_core(...)` MUST NOT print, write files, or build strings.
- Detect errors in `<PROC>_core(...)` by computing `errflg` as a JAX integer
  (0 = OK, nonzero = error). Use `jnp.where` or `lax.cond` — never Python `if`.
- `<PROC>_core(...)` RETURNS `errflg` (and updated arrays) but NOT `errmsg`.
- The wrapper `<PROC>(...)` handles `errmsg`:
  - Check `if int(errflg) != 0` (host conversion is fine in the wrapper).
  - Construct the message with an f-string and return early.
- Do not attempt Fortran `WRITE` formatting inside `<PROC>_core(...)`.
---

# Fortran source to translate:

```fortran
subroutine kessler_run(ncol, nz, dt, lyr_surf, lyr_toa, cpair, rair, rho, z, &
        pk, theta, qv, qc, qr, precl, relhum, scheme_name, errmsg, errflg)
      integer,          intent(in)    :: ncol
      integer,          intent(in)    :: nz
      real(kind_phys),  intent(in)    :: dt
      integer,          intent(in)    :: lyr_surf
      integer,          intent(in)    :: lyr_toa
      real(kind_phys),  intent(in)    :: cpair(:,:)
      real(kind_phys),  intent(in)    :: rair(:,:)
      real(kind_phys),  intent(in)    :: rho(:,:)
      real(kind_phys),  intent(in)    :: z(:,:)
      real(kind_phys),  intent(in)    :: pk(:,:)
      real(kind_phys),  intent(inout) :: theta(:,:)
      real(kind_phys),  intent(inout) :: qv(:,:)
      real(kind_phys),  intent(inout) :: qc(:,:)
      real(kind_phys),  intent(inout) :: qr(:,:)
      real(kind_phys),  intent(out)   :: precl(:)
      real(kind_phys),  intent(out)   :: relhum(:,:)
      character(len=64),intent(out)   :: scheme_name
      character(len=*), intent(out)   :: errmsg
      integer,          intent(out)   :: errflg
      real(kind_phys) :: r(nz),         &
                         rhalf(nz),     &
                         velqr(nz),     &
                         sed(nz),       &
                         pc(nz)
      real(kind_phys) :: f5,            &
                         f2x,           &
                         xk,            &
                         ern,           &
                         qrprod,        &
                         prod,          &
                         qvs,           &
                         dt0
      real(kind_phys) :: time_counter,  &
                         precl_acc
      integer         :: col, klev
      integer         :: lyr_step
      precl = 0._kind_phys
      errmsg = ''
      errflg = 0
      scheme_name = "KESSLER"
      if (dt <= 0._kind_phys) then
         write(errmsg,*) 'KESSLER called with nonpositive dt'
         errflg = 1
         return
      end if
      if (lyr_surf > lyr_toa) then
         lyr_step = -1
      else
         lyr_step = 1
      end if
      f2x = 17.27_kind_phys
      do col = 1, ncol
         do klev = lyr_surf, lyr_toa, lyr_step
            f5  = 4093._kind_phys * lv / cpair(col,klev)
            xk  = cpair(col,klev) / rair(col,klev)
            r(klev)     = 0.001_kind_phys * rho(col, klev)
            rhalf(klev) = sqrt(rho(col, lyr_surf) / rho(col, klev))
            pc(klev)    = 3.8_kind_phys / ((pk(col, klev)**xk) * pref)
            qr(col,klev) = MAX(qr(col,klev),0.0_kind_phys)
            velqr(klev)  = 36.34_kind_phys * rhalf(klev) *          &
                 (qr(col, klev) * r(klev))**0.1364_kind_phys
         end do
         dt0 = dt
         do klev = lyr_surf, lyr_toa - lyr_step, lyr_step
            if (abs(velqr(klev)) > 1.0E-12_kind_phys) then
               dt0 = min(dt0, 0.8_kind_phys*(z(col, klev+lyr_step) - &
                    z(col, klev)) / velqr(klev))
            end if
         end do
         if (dt0 <  1.0E-12_kind_phys) then
            write(errmsg, *) 'KESSLER: bad time splitting ',dt,dt0
            errflg = 1
            return
         end if
         time_counter = 0.0_kind_phys
         precl_acc = 0.0_kind_phys
         do while ( abs(dt - time_counter) > 1.0E-5_kind_phys)
            precl(col) = rho(col, lyr_surf) * qr(col, lyr_surf) * velqr(lyr_surf) / rhoqr
            precl_acc = precl_acc + precl(col) * dt0
            do klev = lyr_surf, lyr_toa - lyr_step, lyr_step
               sed(klev) = dt0 *                                                           &
                    ((r(klev+lyr_step) * qr(col, klev+lyr_step) * velqr(klev+lyr_step)) -  &
                     (r(klev) * qr(col, klev) * velqr(klev))) /                            &
                    (r(klev) * (z(col, klev+lyr_step) - z(col, klev)))
            end do
            sed(lyr_toa) = -dt0 * qr(col, lyr_toa) * velqr(lyr_toa) /    &
                 (0.5_kind_phys * (z(col, lyr_toa)-z(col, lyr_toa-lyr_step)))
            do klev = lyr_surf, lyr_toa, lyr_step
               qrprod = qc(col, klev) - (qc(col, klev) - dt0 *           &
                    max(.001_kind_phys * (qc(col, klev)-.001_kind_phys), &
                        0._kind_phys)) /                                 &
                        (1._kind_phys + dt0 * 2.2_kind_phys *            &
                         qr(col, klev)**.875_kind_phys)
               qc(col, klev) = max(qc(col, klev) - qrprod, 0._kind_phys)
               qr(col, klev) = max(qr(col, klev) + qrprod + sed(klev), 0._kind_phys)
               qvs = pc(klev) * exp(f2x*(pk(col, klev)*theta(col, klev) - 273._kind_phys) / (pk(col, klev)*theta(col, klev) &
                              - 36._kind_phys))
               prod = (qv(col, klev) - qvs) / (1._kind_phys + qvs*f5 / (pk(col, klev)*theta(col, klev) - 36._kind_phys)**2)
               ern = min(dt0 * (((1.6_kind_phys + 124.9_kind_phys*(r(klev)*qr(col, klev))**.2046_kind_phys) * &
                    (r(klev) * qr(col, klev))**.525_kind_phys) /                                              &
                    (2550000._kind_phys * pc(klev) / (3.8_kind_phys*qvs) + 540000._kind_phys)) *              &
                    (dim(qvs,qv(col, klev)) / (r(klev)*qvs)),                                                 &
                    max(-prod-qc(col, klev),0._kind_phys),qr(col, klev))
               theta(col, klev)= theta(col, klev) + (lv / (cpair(col,klev) * pk(col, klev)) * (max(prod,-qc(col, klev)) - ern))
               qv(col, klev) = max(qv(col, klev) - max(prod, -qc(col, klev)) + ern, 0._kind_phys)
               qc(col, klev) = qc(col, klev) + max(prod, -qc(col, klev))
               qr(col, klev) = max(qr(col, klev) - ern, 0._kind_phys)
            end do
            time_counter = time_counter + dt0
             do klev = lyr_surf, lyr_toa, lyr_step
                velqr(klev)  = 36.34_kind_phys * rhalf(klev) * (qr(col, klev)*r(klev))**0.1364_kind_phys
             end do
             dt0 = max(dt -  time_counter, 0.0_kind_phys)
             do klev = lyr_surf, lyr_toa - lyr_step, lyr_step
                if (abs(velqr(klev)) > 1.0E-12_kind_phys) then
                   dt0 = min(dt0, 0.8_kind_phys*(z(col, klev+lyr_step) - z(col, klev)) / velqr(klev))
                end if
             end do
         end do
         precl(col) = precl_acc / dt
         do klev = lyr_surf, lyr_toa, lyr_step
            qvs = pc(klev) * exp(f2x*(pk(col, klev)*theta(col, klev) - 273._kind_phys) / (pk(col, klev)*theta(col, klev) &
                           - 36._kind_phys))
            relhum(col,klev) = qv(col,klev) / qvs * 100._kind_phys
         end do
      end do
   end subroutine kessler_run
```
