# JAX Translation Review — Pass 4: dtype, Annotations & Self-check

The code below is a Python/JAX translation of a Fortran procedure.

**Your task:** finalize dtype consistency, add annotation tags, and verify against the self-check checklist.

# Translation Review Pass
The code below is an existing translation under review.
## Review Output Rules (apply in every pass)
- Do not rewrite the algorithm unless required by this review pass.
- Do not refactor unrelated formatting.
- Do not rename variables unless required by this review pass.
- Preserve signatures and return order.
- If a line is already correct, leave it untouched.
- Make the smallest possible set of edits needed for this review pass.
- Return the full file.
- Add `# CHANGED:` on every modified line.
- Add a brief explanatory comment above each modified block.
- Keep unchanged lines as close to the original as possible.
- Do not clean up unrelated code.
- Do not make style-only edits.

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


## Instructions
1. Add `dtype=jnp.float64` or `dtype=jnp.int32` to every `jnp.zeros`, `jnp.ones`, `jnp.empty`, `jnp.asarray` call missing it.
2. Add inline `# [TAG]` annotations to every loop, conditional, and non-trivial assignment in `_core`.
3. Run through the self-check checklist mentally — fix anything that fails.
4. Return the **complete corrected file** — code only, no explanations.

---
# Code to review:

<<<PREVIOUS_CODE>>>
