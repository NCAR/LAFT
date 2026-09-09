## dtype Consistency (CRITICAL for float64 accuracy)

JAX defaults to **float32** for all numeric literals and array allocations.
Since the translated code must match Fortran double-precision (float64), every
numeric value in `_core` must explicitly use `dtype=jnp.float64`.

### Rules

**Rule 1: All array allocations must specify dtype**
```python
# ❌ Wrong — defaults to float32
r = jnp.zeros(nz)
dt0 = jnp.asarray(0.0)

# ✅ Correct
r = jnp.zeros(nz, dtype=jnp.float64)
dt0 = jnp.asarray(0.0, dtype=jnp.float64)
```

**Rule 2: Integer arrays/scalars must also be typed**
```python
# ✅ Correct
errflg  = jnp.asarray(0, dtype=jnp.int32)
counter = jnp.asarray(0, dtype=jnp.int32)
```

**Rule 3: Constants in formulas inherit dtype from operands — keep operands float64**
```python
# ✅ Safe — rho is already float64, result stays float64
r = 0.001 * rho[:, col]

# ⚠️ Risky — if creating a standalone constant, be explicit
scale = jnp.asarray(0.001, dtype=jnp.float64)
```

**Rule 4: Fortran `REAL(kind_phys)` literals → Python float literals (float64 in context)**
```python
# Fortran: 36.34_kind_phys * coeff(klev) * ...
# Python:  36.34 * coeff[klev] * ...   ← OK, inherits float64 from coeff
```

> The `JAX_ENABLE_X64` flag at the top of every file ensures float64 is available,
> but it does NOT change the default dtype — explicit `dtype=jnp.float64` is still required
> for all zero/ones/empty/asarray calls.
