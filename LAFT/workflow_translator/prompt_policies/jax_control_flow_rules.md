## JAX Control Flow Rules

### Regular FOR loops → lax.fori_loop (when sequential or not yet vectorized)

```fortran
DO i = 1, n
  a(i) = b(i) * 2.0
END DO
```

```python
# lax.fori_loop(start, stop, body, init_state)
# i ranges over [start, stop) — i.e. 0-based, stop is EXCLUSIVE
def loop_body(i, state):
    a, b = state
    a = a.at[i].set(b[i] * 2.0)   # i is already 0-based; no -1 needed
    return a, b

a, b = lax.fori_loop(0, n, loop_body, (a, b))
```

⚠️ If the Fortran loop starts at 1, start the fori_loop at 0 and use `i` directly
(0-based). Never write `a.at[i-1]` inside a loop that starts at 0 — that accesses
`a[-1]` on the first iteration.

### WHILE loops → lax.while_loop

```fortran
DO WHILE (x < 10.0)
  x = x + 1.0
END DO
```

```python
def cond(state):
    x, = state
    return x < 10.0

def body(state):
    x, = state
    return (x + 1.0,)

(x,) = lax.while_loop(cond, body, (x,))
```

### IF statements → jnp.where or lax.cond

```python
# Simple conditional (element-wise, no side effects)
result = jnp.where(condition, value_if_true, value_if_false)

# Complex conditional (large branches, multiple outputs)
def true_fn(state):
    return ...   # true branch

def false_fn(state):
    return ...   # false branch

state = lax.cond(condition, true_fn, false_fn, state)
```

### CRITICAL: State Management in lax loops

ALL variables used or modified inside a loop body MUST be in the state tuple:

```python
# ❌ WRONG — accumulator not in state; causes UnboundLocalError
def body(i, state):
    a, = state
    accumulator = accumulator + a[i]   # accumulator not defined!
    return (a,)

# ✅ CORRECT — ALL touched variables are in state
def body(i, state):
    a, accumulator = state
    accumulator = accumulator + a[i]
    return a, accumulator              # return EVERYTHING, even unchanged vars

a, accumulator = lax.fori_loop(0, n, body, (a, jnp.asarray(0.0, dtype=jnp.float64)))
```

**"When in doubt, ADD IT TO STATE."**
Extra state variables are cheap; missing ones cause hard-to-debug runtime errors.

### Variables declared in outer scope — NOT automatically captured in lax loop bodies

JAX loop body functions (for `lax.fori_loop`, `lax.while_loop`, `lax.scan`) can **read**
outer-scope compile-time constants (e.g. `nz`, `dt`, physical constants) but **cannot write**
to outer-scope arrays. If you write to an outer-scope variable, Python raises `UnboundLocalError`.

```python
# ❌ WRONG — coeff_arr is from outer scope; JAX does NOT capture it automatically for writing
def body(i, state):
    field, = state
    field = field.at[i].set(field[i] + coeff_arr[i] * 2.0)   # coeff_arr is not in state!
    return (field,)

# ✅ CORRECT — pass coeff_arr through state (read-only arrays also benefit from being in state)
def body(i, state):
    field, coeff_arr = state
    field = field.at[i].set(field[i] + coeff_arr[i] * 2.0)
    return field, coeff_arr                              # coeff_arr unchanged but must be returned

field, _ = lax.fori_loop(0, nz, body, (field, coeff_arr))
```

**Python scoping trap — ANY assignment triggers local scope (CRITICAL):**
If a variable name appears on the LEFT side of ANY assignment anywhere inside a function
(including `.at[].set()`, tuple unpacking, or plain `=`), Python marks it as a *local
variable* for the ENTIRE function scope — even lines that appear before the assignment.
If the variable isn't assigned locally before it is first READ, Python raises
`UnboundLocalError` — even if the variable exists in an outer scope.

```python
# ❌ TRIGGER 1 — .at[].set() assignment
workspace = jnp.zeros(nz)            # outer scope

def subcycle_body(state):
    q, = state
    workspace = workspace.at[0].set(q[0])   # UnboundLocalError!
    # Python sees 'workspace =' → treats it as local → not yet assigned on this line

# ❌ TRIGGER 2 — tuple unpacking from lax.while_loop return value (same trap, different syntax)
# velqr defined in outer scope (process_one_col); run_subcycle receives args without it.
def run_subcycle(args):
    theta_c, qv_c, qc_c, qr_c, ... = args
    # Python sees 'velqr' on the left of the unpacking below → marks it local for ALL of run_subcycle
    theta_c, qv_c, qc_c, qr_c, velqr, ... = lax.while_loop(
        cond, body,
        (theta_c, qv_c, qc_c, qr_c, velqr, ...),  # ❌ UnboundLocalError: reads velqr before assignment
    )
```

**Fix for TRIGGER 1:**
```python
# ✅ OPTION A — carry in state (if value is needed from the previous iteration)
def subcycle_body(state):
    q, workspace = state
    workspace = workspace.at[0].set(q[0])
    return (q, workspace)

# ✅ OPTION B — reinitialize inside body (if fully recomputed each iteration)
def subcycle_body(state):
    q, = state
    workspace = jnp.zeros(nz)               # fresh init — local variable
    workspace = workspace.at[0].set(q[0])
    return (q,)
```

**Fix for TRIGGER 2 — pass the variable through `lax.cond` args so it has a local binding before `lax.while_loop`:**
```python
# ✅ CORRECT — velqr passed through lax.cond args so it is unpacked (local) before while_loop
def run_subcycle(args):
    theta_c, qv_c, qc_c, qr_c, ..., velqr = args   # local binding established here
    theta_c, qv_c, qc_c, qr_c, velqr, ... = lax.while_loop(
        cond, body,
        (theta_c, qv_c, qc_c, qr_c, velqr, ...),   # ✅ velqr is already local
    )

# Both lax.cond branches must accept and return the same structure:
def skip_subcycle(args):
    theta_c, qv_c, qc_c, qr_c, ..., velqr = args   # unpack but don't use
    return theta_c, qv_c, qc_c, qr_c, ...

lax.cond(condition, skip_subcycle, run_subcycle,
         (theta_c, qv_c, qc_c, qr_c, ..., velqr))  # ✅ velqr in args tuple
```

**Decision guide:**
- Array needs its value from the previous iteration? → **Option A** (add to state tuple)
- Array recomputed from scratch each iteration? → **Option B** (init inside body)
- Array from outer scope needed inside `lax.cond` branch that also unpacks it from `lax.while_loop`? → pass through `lax.cond` args

It is perfectly fine to have large state tuples:
```python
# 10 variables — JAX handles this efficiently
state = (time, accum, dt0, f1, f2, f3, f4, vel, coeff_arr, tmp)  # illustrative — variable names from your Fortran source will differ
```
