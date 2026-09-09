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
