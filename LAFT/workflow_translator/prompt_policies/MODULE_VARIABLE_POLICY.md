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
