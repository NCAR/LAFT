# Fortran → Python Translation Task (SCALAR-ONLY)

Translate a single Fortran procedure into a **plain Python function**.

This procedure contains NO array arguments — only scalars — and is NOT called
from within any JAX/JIT context, so plain Python is safe and optimal.
**Do NOT generate a `kessler_init_core` function. Do NOT use JAX.**

## Hard constraints (must follow)
- Generate ONE function only: `kessler_init(...)` — no `_core`.
- Use plain Python arithmetic and Python types (float, int, str).
- Do NOT import jax, jax.numpy, or numpy.
- Do not invent globals. Only use what's in the Fortran unit.

## SCALAR-ONLY PROCEDURE — Plain Python, No JAX

This procedure operates **only on scalar values** (no arrays).
Generating a JAX `_core` with `jax.jit` would cause unnecessary GPU dispatch
overhead for what are just a few scalar assignments.

### Rules

1. **Do NOT generate a `_core` function.**  Generate a SINGLE plain Python function only.
2. **Do NOT import JAX** unless absolutely needed (e.g. `jnp.asarray` for a scalar result
   that must interop with a JAX array downstream — even then, prefer plain Python floats).
3. **Do NOT use `jnp.asarray`, `jnp.float64`, or any `jnp.*` call** for scalar assignments.
   Just use plain Python arithmetic:
   ```python
   lv    = float(lv_in)
   pref  = float(pref_in) / 100.0
   rhoqr = float(rhoqr_in)
   errflg = 0
   errmsg = ""
   ```
4. **String outputs** (`errmsg`) are plain Python strings — assign directly.
5. **Module variables** are plain Python floats passed in and returned — no JAX wrappers.

### Output format

Return Python code only (no explanations). Include:
1) No JAX imports (or minimal if genuinely needed)
2) Module-level docstring marking this as SCALAR-ONLY
3) A SINGLE function `kessler_init(...)` — NO `kessler_init_core`
4) Return the updated values as plain Python scalars


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
- Wrapper signature MUST be: `kessler_init(lv_in, pref_in, rhoqr_in, errmsg, errflg, lv, pref, rhoqr)`
- Core signature MUST be: `kessler_init_core(lv_in, pref_in, rhoqr_in, errmsg, errflg, lv, pref, rhoqr)`
- Use these signatures EXACTLY — do not add, drop, or reorder parameters.
- Do NOT add parameters for variables that appear only in Fortran comments
  or commented-out code (dead code). The bridge layer is generated from the
  signature above; an invented parameter breaks the bridge call.

**MODULE VARIABLES (from kessler):**
  - `lv`: MODULE variable (INOUT)
  - `pref`: MODULE variable (INOUT)
  - `rhoqr`: MODULE variable (INOUT)

MODULE variables are INOUT parameters - include in signature and return them!

- Return updated values: `errmsg, errflg, lv, pref, rhoqr`

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
subroutine kessler_init(lv_in, pref_in, rhoqr_in, errmsg, errflg)
      real(kind_phys),    intent(in)  :: lv_in
      real(kind_phys),    intent(in)  :: pref_in
      real(kind_phys),    intent(in)  :: rhoqr_in
      character(len=512), intent(out) :: errmsg
      integer,            intent(out) :: errflg
      errmsg = ''
      errflg = 0
      lv    = lv_in
      pref  = pref_in/100._kind_phys
      rhoqr = rhoqr_in
   end subroutine kessler_init
```
