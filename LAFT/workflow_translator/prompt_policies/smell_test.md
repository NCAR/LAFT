## MANDATORY SELF-CHECK before outputting code

After writing `_core`, scan **every line** and verify each item below.
If any check fails, **fix it before outputting**. Then record the result
as a comment on the second line of the file:

```python
# JIT boundary check: PASSED
# JIT boundary check: FIXED — replaced Python if on dt0 with jnp.where
```

### Checklist

**Static integer parameters** — full rule and examples in "JIT Boundary Rules → STATIC INTEGER PARAMETERS"
- [ ] No static int (ncol, nz, loop bounds, direction flags) passed through any `jnp.*` call (`# [STATIC-INT]`).
- [ ] Descending ranges use explicit negative step: `jnp.arange(high, low - 1, -1)`.

**JIT / tracer violations**
- [ ] No `if`, `elif`, `else` whose condition involves a `jnp` value or any variable
      that has ever been assigned a `jnp.*` or `lax.*` result (tracer infection).
- [ ] `[PY-IF]` on a `[STATIC-INT]` value is allowed; `[PY-IF]` on any other variable
      inside `_core` is a bug.
- [ ] No `for i in range(...)` or `while` loop whose bound or condition involves a JAX value.
- [ ] No `.item()`, `int(x)`, `float(x)`, `bool(x)`, `.tolist()` called on a JAX array
      anywhere inside `_core`.
- [ ] No `print(...)` or string formatting (`f"..."`, `str(x)`) inside `_core`.

**Nested loop consistency**
- [ ] No Python loop (any nesting level) wrapping a `lax.*` primitive.
- [ ] No Python loop (any nesting level) wrapping `jax.vmap`.
- [ ] No Python `for` / `while` anywhere inside `_core` unless its bounds are
      **compile-time Python constants** (e.g. a fixed small integer, not derived from any array).

**Vectorization**
- [ ] Every independent loop is vectorized (`jnp` broadcast or `jax.vmap`), not left as `fori_loop`.
- [ ] `lax.fori_loop` is used only when iterations are genuinely sequential
      (iteration `i` reads values written by iteration `i-1`).

**`_core` decorator**
- [ ] `_core` is decorated with `@functools.partial(jax.jit, static_argnames=(...))`.
- [ ] `static_argnames` lists every integer param that determines a shape, range, or direction (no floats, no arrays).
- [ ] `import functools` is present at the top of the file.

**Strings stay out of `_core`**
- [ ] `_core` does NOT accept `errmsg`, `scheme_name`, or any other string parameter.
- [ ] `_core` does NOT return any string. Strings are not valid JAX types — they crash JIT (or, if marked static, force a recompile on every distinct value).
- [ ] All string handling (validation messages, scheme labels) happens in the wrapper, before/after the `_core` call.

**dtype**
- [ ] Every `jnp.zeros`, `jnp.ones`, `jnp.empty`, `jnp.asarray` call specifies `dtype=`.
- [ ] All `lax` loop state variables are initialised with the correct JAX dtype
      (not plain Python `0` or `0.0` mixed with `jnp.int32` / `jnp.float64`).

**Indexing**
- [ ] `lax.fori_loop(0, n, body, ...)` — `i` inside body is 0-based; no `i-1` needed.
- [ ] All Fortran 1-based integer indices converted to 0-based before first use: `idx = param - 1  # [STATIC-INT]`.

**Annotation tags**
- [ ] Every loop opening line carries a `# [JAX-FORI]`, `# [JAX-VMAP]`, `# [JAX-WHILE]`,
      `# [JAX-SCAN]`, `# [PY-FOR]`, or `# [PY-WHILE]` tag.
- [ ] Every conditional carries a `# [JAX-COND]`, `# [JAX-WHERE]`, or `# [PY-IF]` tag.
- [ ] No `# [PY-IF]`, `# [PY-FOR]`, or `# [PY-WHILE]` tag appears anywhere inside `_core`.
      If any do, that line is a bug — fix it before outputting.
