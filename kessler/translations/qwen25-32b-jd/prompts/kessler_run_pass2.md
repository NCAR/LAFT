# JAX Translation Review — Pass 2: JIT Safety

The code below is a Python/JAX translation of a Fortran procedure.

**Your task:** fix every JIT/tracer boundary violation found in the code.

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

## JIT Boundary Rules (CRITICAL)

The `_core` function MUST be decorated with `@functools.partial(jax.jit, static_argnames=(...))`.
This ensures it is JIT-compiled and GPU-accelerated regardless of how it is called
(bridge, driver, tests). Never rely on the caller to JIT the core.

### MANDATORY decorator on every `_core` function

```python
@functools.partial(jax.jit, static_argnames=('ncol', 'nz', ...))
def proc_core(ncol, nz, ...):
    ...
```

**What goes in `static_argnames`:** every **integer** parameter whose value
determines an array shape, loop range length, or iteration direction (see
STATIC INTEGER PARAMETERS section below).

Floats and arrays must NOT be static.

### NEVER static: per-call-varying scalars

A timestep/iteration counter (`it`, `kount`, `itimestep`, `istep`) or a
simulation-time value must NOT be in `static_argnames`, even though it is an
integer scalar. A static arg is baked into the compiled binary, so a value
that changes every call recompiles the kernel every call — for a large
`_core` that is tens of seconds *per step* and turns a minutes-long run into
hours (observed on p3_main: ~18 s/step, 15 of 90 simulated minutes inside a
1-hour walltime). Branch on such values with `jnp.where`
(`jnp.where(it <= 1, init_path, normal_path)`), never with Python `if`.
Rule of thumb: static = constant for the whole run (shapes, category counts,
config flags); traced = anything the driver loop changes.

**Why:** JAX must know all array shapes at compile time. Static args are baked into
the compiled binary. The function is recompiled each time a static arg value changes;
shape/config statics change never (one compile per run), which is exactly why
per-call-varying scalars must stay traced.

### Strings MUST NOT be passed to `_core`

`errmsg`, `scheme_name`, and any other string parameter belong in the **wrapper
only**. Do NOT pass them through to `_core`.

**Why this is the rule, not just a preference:**
- Strings are not valid JAX types. If `_core` accepts them, they crash JIT
  with `TypeError: ... type <class 'str'> ...` unless marked static.
- Marking them static "works" but every distinct string value (e.g. each
  different error message) becomes a separate JIT cache entry and triggers
  recompilation. That's wasted work — `_core` does no string computation.
- `_core` is a pure numerical kernel. Diagnostic text is control-flow, and
  control-flow lives in the wrapper.
- `_core` MUST NOT return strings either (XLA cannot represent them).

**Correct pattern — strings stay in the wrapper:**

```python
@functools.partial(jax.jit, static_argnames=('ncol', 'nz', 'lyr_surf', 'lyr_toa'))
def kessler_run_core(ncol, nz, dt, lyr_surf, lyr_toa, cpair, rair, rho, z, pk,
                     theta, qv, qc, qr, precl, relhum, errflg, lv, pref, rhoqr):
    # pure compute — no strings in or out
    ...
    return theta, qv, qc, qr, precl, relhum, errflg, lv, pref, rhoqr


def kessler_run(ncol, nz, dt, lyr_surf, lyr_toa, ...,
                scheme_name, errmsg, errflg, ...):
    # wrapper handles strings + validation
    if dt <= 0.0:
        return ..., scheme_name, "KESSLER nonpositive dt", 1, ...
    out = kessler_run_core(ncol, nz, dt, lyr_surf, lyr_toa, ...,
                           theta, qv, qc, qr, ..., errflg, lv, pref, rhoqr)
    return *out_arrays, scheme_name, "", out_errflg, ...
```

The wrapper calls the decorated core directly — no extra `jax.jit` needed.

---

### Two worlds that must never mix

| World | Runs in | Sees JAX values as |
|---|---|---|
| **Trace-time** (Python) | First JIT call | Abstract tracers — shape/dtype only, NO actual values |
| **Runtime** (XLA binary) | Every subsequent call | Real data — but Python no longer exists |

Every JIT bug is a **trace-time / runtime boundary violation**: Python code that
assumes it can inspect or branch on a JAX value that only exists at runtime.

---

### FORBIDDEN patterns inside `_core` (will silently miscompute or crash)

```python
# ❌ FORBIDDEN — Python if/elif/else on any JAX scalar or array
if errflg != 0: ...
if dt0 < 1e-12: ...
if field[k] > 0: ...

# ❌ FORBIDDEN — wrapping static_argnames integers in jnp.where
# Any integer that determines a loop range or array shape must stay Python.
# Using jnp.where infects the result with tracer status, making it unusable
# in Python if/range/arange — regardless of the variable name in the source.
step_dir = jnp.where(i_start > i_end, -1, 1)      # ← step_dir is now a tracer!
# Fix: use plain Python
step_dir = 1 if i_start <= i_end else -1           # ✅ [STATIC-INT]

# ❌ FORBIDDEN — Python for loop whose range depends on a JAX value
for i in range(int(n)): ...          # int() extracts from tracer → breaks trace
for i in range(i_start, i_end): ...  # safe only if i_start/i_end are plain Python ints
                                     # becomes forbidden the moment they touch jnp

# ❌ FORBIDDEN — Python while loop on a JAX condition
while time_counter < dt: ...         # dt / time_counter are JAX scalars

# ❌ FORBIDDEN — host value extraction inside _core
n = int(x)                           # breaks tracing
v = float(arr[i])                    # breaks tracing
s = x.item()                         # breaks tracing
s = x.tolist()                       # breaks tracing

# ❌ FORBIDDEN — string construction on JAX values inside _core
errmsg = f"invalid dt={dt0}"         # dt0 is a tracer — cannot format
print(f"field = {field[0]}")        # also forbidden: print inside _core
```

---

### TRACER INFECTION RULE

Any Python variable that is assigned the result of a `jnp.*` or `lax.*` operation
becomes a **JAX tracer**. From that point on it must be treated as a JAX value —
you can no longer use it in Python control flow.

```python
dt0 = dt                             # dt0 is a plain Python float — safe in if/while
dt0 = jnp.minimum(dt0, cfl_dt)      # dt0 is NOW a tracer — infected!
if dt0 < 1e-12: ...                  # ❌ FORBIDDEN — dt0 was infected by jnp.minimum

# The fix: use jnp.where or lax.cond from this point forward
dt0 = jnp.minimum(dt0, cfl_dt)
errflg = jnp.where(dt0 < 1e-12,
                   jnp.asarray(1, dtype=jnp.int32),
                   errflg)           # ✅ CORRECT
```

Infection is **permanent and transitive**: if `a` is a tracer and you compute
`b = a + 1.0`, then `b` is also a tracer. When in doubt, assume it is infected.

---

### STATIC INTEGER PARAMETERS — never infect them

Some integer scalar parameters must remain **plain Python integers** inside `_core`.
The bridge passes them as `static_argnames` to `jax.jit`. XLA must know all array
shapes at compile time, so any integer that **determines a shape, range length, or
iteration direction** must stay Python — regardless of its name in the Fortran source.

**How to recognise them in any physics code:**
- Array dimension sizes: anything like `n`, `ncol`, `nz`, `nlev`, `nx`, `ny`, …
- Loop range boundaries: start/end level or index integers (e.g. `kbot`, `ktop`,
  `istart`, `iend`, `i_start`, `i_end`, `k1`, `k2`, …)
- Loop direction flags: integers that select ascending vs. descending iteration
- Any integer whose value controls how many elements `jnp.arange` or a slice produces

**Rule: never pass any of the above through `jnp.*`** — that converts it into a
tracer, making its value unknown at trace time and breaking every `jnp.arange`,
slice, or shape expression that depends on it.

The examples below use generic names (`i_start`, `i_end`) but the rule applies
to equivalent variables in **any** physics scheme:

```python
# ❌ WRONG — infects the derived value; its value is now unknown at trace time
step_dir = jnp.where(i_start > i_end, -1, 1)   # any physics: same mistake

# ✅ CORRECT — i_start/i_end are static Python ints; keep step_dir Python too
step_dir = 1 if i_start <= i_end else -1        # [STATIC-INT]
```

Once infected, every downstream use breaks:
```python
# ❌ jnp.where requires both branches to have the SAME shape.
#    JAX evaluates both at trace time regardless of condition.
#    If the two arange() calls produce different lengths → crash.
indices = jnp.where(step_dir == 1,
    jnp.arange(i_start, i_end + 1),   # shape (N,)
    jnp.arange(i_end,   i_start + 1)) # shape (0,) ← CRASH when i_end > i_start

# ✅ step_dir is a Python int → [PY-IF] is legal; branches can have different sizes
if step_dir == 1:                      # [PY-IF] ✅ legal on STATIC-INT
    indices = jnp.arange(i_start, i_end + 1)
else:
    indices = jnp.arange(i_start, i_end - 1, -1)  # step=-1 !
```

**Static int values are perfectly usable inside JAX operations** — JAX treats them
as compile-time constants. The one-way rule is:

| Direction | OK? |
|---|---|
| Python int → argument to `jnp.*` | ✅ JAX uses it as a constant |
| JAX tracer → determines array shape | ❌ shape unknown at trace time |

Annotate static-int derivations with `# [STATIC-INT]`:
```python
# Generic pattern — applies to any physics variable names
i_start_idx = i_start - 1               # [STATIC-INT]
i_end_idx   = i_end   - 1               # [STATIC-INT]
step_dir    = 1 if i_start <= i_end else -1  # [STATIC-INT]
nlevs       = abs(i_end - i_start) + 1  # [STATIC-INT]
```

**Descending ranges:** `jnp.arange(a, b)` is **empty** when `a >= b`.
For a descending range always pass an explicit negative step:
```python
jnp.arange(i_end_idx, i_start_idx - 1, -1)    # ✅ descending, correct length
```

---

### Correct replacements for every forbidden pattern

```python
# Python if on JAX value  →  jnp.where (simple) or lax.cond (complex)
# ❌
if field > 0:
    vel = 36.34 * coeff * field ** 0.1364  # illustrative formula
else:
    vel = 0.0
# ✅
vel = jnp.where(field > 0,
                36.34 * coeff * field ** 0.1364,  # illustrative formula
                jnp.asarray(0.0, dtype=jnp.float64))

# Condition on a STATIC INT  →  [PY-IF], NOT jnp.where.
# See "STATIC INTEGER PARAMETERS" section above for the full rule, examples, and descending-range fix.

# Python if with large branches  →  lax.cond (see "In-JIT error signalling" below for full pattern)
# Python while on JAX condition   →  lax.while_loop (see JAX Control Flow Rules)
# Python for over dynamic range   →  lax.fori_loop or vectorized ops (see JAX Control Flow Rules)
```

---

### Python `if` is allowed in the **wrapper only**, on plain Python values

```python
# ✅ CORRECT — wrapper receives dt as a plain Python float before any jnp op
def my_proc(ncol, nz, dt, ...):
    if dt <= 0.0:                            # dt is still plain Python here
        errmsg = f"nonpositive dt={dt}"
        errflg = 1
        return ..., errmsg, errflg
    return my_proc_core(ncol, nz, dt, ...)
```

---

### In-JIT error signalling with lax.cond

When an error condition can only be detected inside the JIT boundary
(e.g. a CFL violation computed from JAX arrays), signal it via `errflg`:

```python
# ✅ Inside _core: set errflg with lax.cond, never build strings
errflg = lax.cond(
    dt0 < jnp.asarray(1.0e-12, dtype=jnp.float64),
    lambda _: jnp.asarray(1, dtype=jnp.int32),
    lambda _: errflg,
    operand=None,
)
# Return errflg; the wrapper will build the error string if errflg != 0.
```

---

### State type/shape invariant for lax loops

JAX requires that every iteration of `lax.fori_loop` / `lax.while_loop` / `lax.scan`
returns the **exact same dtypes and shapes** as the initial state.

```python
# ❌ WRONG — Python int 0 mixed with jnp.int32 operations
counter = 0
def body(i, counter):
    return counter + jnp.asarray(1, dtype=jnp.int32)   # type conflict

# ✅ CORRECT — initialise with the same type you will compute with
counter = jnp.asarray(0, dtype=jnp.int32)
def body(i, counter):
    return counter + jnp.asarray(1, dtype=jnp.int32)
```

---

> **Body scoping rule:** See "CRITICAL: State Management in lax loops" in JAX Control Flow Rules
> for the full explanation of the Python `UnboundLocalError` trap and Option A / Option B fixes.
```


## Instructions
1. Read the code and apply the rules above.
2. Fix every violation: forbidden Python `if`/`for`/`while` on JAX values, tracer infection, static-int misuse, host value extraction (`.item()`, `int()`, `float()`), strings inside `_core`, etc.
3. On the second line of the file add: `# JIT boundary check: FIXED — <what you changed>` if fixes were made, or `# JIT boundary check: PASSED` if none were needed.
4. Return the **complete corrected file** — code only, no explanations.

---
# Code to review:

<<<PREVIOUS_CODE>>>
