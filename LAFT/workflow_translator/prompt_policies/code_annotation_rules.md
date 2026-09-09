## MANDATORY Inline Annotation Tags
Every loop, conditional, and non-trivial assignment in `_core` MUST carry an
inline comment tag so that Python/JAX mixing bugs are immediately visible.
### Tag vocabulary
| Tag | Meaning | Allowed in `_core`? |
|---|---|---|
| `# [JAX-VEC]` | Vectorized `jnp` array op — no loop | ✅ |
| `# [JAX-VMAP]` | `jax.vmap` over a batch dimension | ✅ |
| `# [JAX-FORI]` | Sequential `lax.fori_loop` | ✅ |
| `# [JAX-WHILE]` | `lax.while_loop` | ✅ |
| `# [JAX-SCAN]` | `lax.scan` | ✅ |
| `# [JAX-COND]` | `lax.cond` for branching | ✅ |
| `# [JAX-WHERE]` | `jnp.where` for element-wise conditional | ✅ |
| `# [JAX]` | Any other JAX/jnp scalar op (assignment, cast, …) | ✅ |
| `# [STATIC-INT]` | Value derived purely from static_argnames int params | ✅ stays Python, never pass through jnp |
| `# [PY]` | Plain Python op — value is NOT a JAX tracer | ✅ only if value never touches jnp |
| `# [PY-IF]` | Python `if` / `elif` / `else` | ✅ on `[STATIC-INT]` values; ❌ on JAX tracers |
| `# [PY-FOR]` | Python `for` loop | ❌ FORBIDDEN in `_core` |
| `# [PY-WHILE]` | Python `while` loop | ❌ FORBIDDEN in `_core` |
**Rule: any `[PY-IF]`, `[PY-FOR]`, or `[PY-WHILE]` tag inside `_core` is a bug.**
These are allowed only in the wrapper.
### Where to place tags
- On the **same line** as the statement (end-of-line comment).
- For multi-line `lax.*` constructs, tag the **opening line**.
- For `jnp.where`, tag the line where `jnp.where` appears.
### Example
```python
def proc_core(ncol, nz, dt, rho, pres, ...):
    field = scale * rho                               # [JAX-VEC]  field(nz,ncol)  # illustrative
    b   = jnp.sqrt(rho[surf_idx] / rho)              # [JAX-VEC]
    vel = jnp.where(                                 # [JAX-WHERE]
        tracer > 0,
        coeff * b * (tracer * field) ** exp_val,
        jnp.zeros_like(tracer))
    def one_col(rho_c, tracer_c, vel_c):             # [JAX-VMAP]  per-column fn
        dt0 = jnp.minimum(dt, ...)                   # [JAX]
        def body(i, state):                          # [JAX-FORI]
            tracer_c, acc = state
            ...
            return tracer_c, acc
        tracer_c, acc = lax.fori_loop(0, nz, body, (tracer_c, jnp.asarray(0.0, dtype=jnp.float64)))
        return tracer_c, acc
    tracer, acc = jax.vmap(one_col, in_axes=(1,1,1), out_axes=(1,0))(rho, tracer, vel)
    return ..., q, acc
def proc(...):
    if dt <= 0.0:                                    # [PY-IF]  ✅ wrapper only
        return ..., "nonpositive dt", 1
    return proc_core(...)
```
**Wrong — mixing Python control flow with JAX values in `_core`:**
```python
# ❌
if dt0 < 1e-12:          # [PY-IF]  ← tag exposes the bug immediately
    errflg = 1
for col in range(ncol):  # [PY-FOR] ← tag exposes the bug immediately
    ...
```