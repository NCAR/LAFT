## JAX Vectorization & Performance Guidelines

---

### NESTED LOOP CONSISTENCY RULE (CRITICAL)

Every level of a nested loop MUST live in the **same world** (JAX or Python).
You CANNOT mix a Python outer loop with a JAX inner construct, or vice versa.

```python
# ❌ WRONG — Python outer loop + lax.fori_loop inner
for col in range(ncol):              # Python loop runs at TRACE TIME
    result = lax.fori_loop(0, nz, body, init)
# XLA sees ncol COPIES of the inner graph baked in at compile time.
# Shape is locked; no column-level parallelism.

# ❌ WRONG — Python outer loop + vectorized jnp inner
for col in range(ncol):              # still Python — same problem
    r = r.at[:, col].set(0.001 * rho[:, col])

# ❌ WRONG — lax.fori_loop outer + Python for inner
def outer_body(col, state):
    for klev in range(nz):           # Python for inside a JAX loop body
        ...                          # nz hardcoded at trace time; not JIT-safe

# ✅ CORRECT — both levels in JAX world
r = jax.vmap(lambda rho_c: 0.001 * rho_c, in_axes=1, out_axes=1)(rho)
# or, if sequential across levels:
r, _ = lax.fori_loop(0, nz, body, (r, rho))
# or, if both independent:
r = 0.001 * rho   # shape (nz, ncol) — single op, no loops at all
```

**Why mixed nesting is wrong:**
- The Python loop unrolls `ncol` copies of the inner JAX graph at trace time,
  bloating compile time and locking the compiled binary to a specific `ncol`.
- It destroys column-level parallelism: XLA cannot schedule columns concurrently.
- It may produce correct numbers for one input shape, then silently break for another.

---

### VECTORIZATION PRIORITY LADDER

When translating a Fortran loop (or nested loops), work down this ladder and stop
at the **first level that applies**:

**Level 1 — FULL VECTORIZATION** ✅ best
All iterations independent across ALL dimensions?
→ Single `jnp` operation on the full array. No loops at all.
```python
# Fortran: do col=1,ncol / do klev=1,nz / r(col,klev) = 0.001*rho(col,klev)
r = 0.001 * rho    # shape (nz, ncol) — done in one line
```

**Level 2 — vmap over columns + vectorized levels** ✅ very good
Column iterations independent; level ops also independent?
→ `jax.vmap` over columns, vectorized `jnp` inside for levels.
```python
def one_col(rho_c):      # shape (nz,)
    return 0.001 * rho_c
r = jax.vmap(one_col, in_axes=1, out_axes=1)(rho)
```

**Level 3 — vmap over columns + lax.fori_loop over levels** ✅ good
Column iterations independent; level loop is sequential (data dependency between levels)?
→ `jax.vmap` over columns, `lax.fori_loop` inside for levels.

**Level 4 — lax.fori_loop inside lax.fori_loop** ⚠️ acceptable
Both loops sequential (each iteration depends on the previous)?
→ Nested `lax.fori_loop`. Both loops are in JAX world.

**Level 5 — Python loop wrapping ANY JAX construct** ❌ NEVER ACCEPTABLE
```python
for col in range(ncol): lax.fori_loop(...)   # WRONG
for col in range(ncol): jnp.something(...)   # WRONG
```

---

### Level loop: prefer full-array vectorization over subset indexing

When the Fortran level loop runs over an *active range* (e.g. `do klev = lyr_surf, lyr_toa - lyr_step, lyr_step`) AND a separate scalar statement special-cases one end (e.g. `sed(lyr_toa) = ...`), there are two ways to vectorize. Prefer **Pattern B** — it produces a single fused XLA kernel with no scalar fix-up branch.

The signature of Pattern B is the triple **`jnp.arange(nz) + jnp.clip + jnp.where`**.

#### ❌ Pattern A — subset index + scalar special-case (less GPU-idiomatic)

```python
# Vectorize the active range only
k        = jnp.arange(lyr_surf_idx, lyr_toa_idx)              # [STATIC-INT bounds]
k_next   = k + lyr_step
sed_body = dt0 * (r[k_next]*qr[k_next]*vel[k_next]
                - r[k]     *qr[k]     *vel[k])     / (...)    # illustrative
sed      = jnp.zeros(nz, dtype=jnp.float64).at[k].set(sed_body)
# Separate scalar statement for the boundary level:
sed = sed.at[lyr_toa_idx].set(-dt0 * qr[lyr_toa_idx] * vel[lyr_toa_idx] / (...))
```

Two kernels — one vector op plus a single-element scatter. The boundary branch breaks fusion.

#### ✅ Pattern B — full-array index + clip + where (preferred)

```python
k        = jnp.arange(nz)                                     # ALL levels at once
k_next   = jnp.clip(k + lyr_step, 0, nz - 1)                  # neighbor stays in-bounds
sed_body = dt0 * (r[k_next]*qr[k_next]*vel[k_next]
                - r[k]     *qr[k]     *vel[k])     / (...)    # illustrative
sed_toa  = -dt0 * qr[k] * vel[k] / (...)                      # TOA formula, evaluated everywhere
is_toa   = (k == lyr_toa_idx)
sed      = jnp.where(is_toa, sed_toa, sed_body)               # boundary absorbed inline
```

One fused kernel; every level computed in parallel; no scalar fix-up. The `jnp.arange(nz) + jnp.clip + jnp.where` triple is the indicator of this pattern.

#### When to use which

- Boundary special-case in the Fortran loop (`sed(toa) = ...`, `sed(surf) = ...`)? → **Pattern B**.
- Active range covers the whole array AND no boundary special-case? → either works; Pattern B is still slightly preferred because it avoids `[STATIC-INT]` arange bounds.
- Need to skip levels entirely (not just adjust them)? → Pattern B with `jnp.where(active_mask, body, prev_value)` so inactive levels pass through unchanged.

#### Safety note — Pattern B requires the formula to be safe everywhere

`jnp.where` evaluates **both** branches at every level before selecting. The "default" expression therefore runs even at boundary levels where it would be invalid in Fortran. Before adopting Pattern B, check that the body formula is safe at every index in `[0, nz)`:

- ❌ **Division by a quantity that is zero outside the active range** (e.g. `(z[k_next] - z[k])` becomes zero after `jnp.clip` collapses `k_next` onto `k` at a boundary). Result: a NaN/Inf is computed and *then* masked out — but on some hardware/compilers the NaN can poison subsequent reductions (`jnp.min`, `jnp.sum`).
- ❌ **`jnp.log`, `jnp.sqrt`, fractional powers** of expressions that can go non-positive at boundary indices.
- ❌ **Indexing offsets larger than `±1`** that `jnp.clip` cannot fully repair.

When the body is unsafe at boundaries, sanitise the operands *before* the body so the masked-out value is finite:

```python
# Replace any boundary-unsafe operand with a safe placeholder; jnp.where masks the result anyway.
dz_safe  = jnp.where(is_toa, jnp.asarray(1.0, dtype=jnp.float64), z[k_next] - z[k])
sed_body = dt0 * (r[k_next]*qr[k_next]*vel[k_next] - r[k]*qr[k]*vel[k]) / (r[k] * dz_safe)
sed      = jnp.where(is_toa, sed_toa, sed_body)              # sed_body is finite everywhere
```

Rule of thumb: if the Fortran loop bounds *avoid* a boundary level on purpose, the body is probably unsafe there — sanitise the operand, don't just trust the outer `jnp.where`.

#### Safety note 2 — levels entirely OUTSIDE the active range (MANDATORY mask)

The boundary level is not the only hazard. `jnp.arange(nz)` evaluates the body at
**every** index of the array — including levels the Fortran loop never visits. Do
NOT assume the active range `lyr_surf..lyr_toa` spans the whole level axis: the
runtime validator (and any caller with a partial active range) passes arrays
**taller** than the active range. At the far end of such an array, `jnp.clip`
collapses `k_next` onto `k` exactly as it does at the TOA boundary, so the same
zero-denominator / Inf appears at indices a TOA-only sanitisation never covers —
and the Inf flows into the state arrays, failing the validator's finiteness check.

Whenever the active range is given by start/end index parameters, derive a static
in-range mask and apply it in **three** places:

```python
lo     = min(lyr_surf_idx, lyr_toa_idx)               # [STATIC-INT]
hi     = max(lyr_surf_idx, lyr_toa_idx)               # [STATIC-INT]
in_rng = (k >= lo) & (k <= hi)                        # levels the Fortran loop visits

# 1) sanitise the unsafe operand outside the range too (not just at the boundary)
dz_safe  = jnp.where(is_toa | ~in_rng, jnp.asarray(1.0, dtype=jnp.float64),
                     z[k_next] - z[k])
# 2) zero (or neutralise) the computed term outside the range
sed      = jnp.where(is_toa, sed_toa, jnp.where(in_rng, sed_body, 0.0))
# 3) pass out-of-range levels through UNCHANGED in every state update,
#    exactly matching the Fortran loop bounds
theta_new = jnp.where(in_rng_2d, theta_updated, theta_old)
```

When the active range does cover the whole axis (the production case), `in_rng`
is all-True and XLA folds the masks away — there is no cost. Omitting the mask is
the bug: it is invisible on production grids and only surfaces on validator-style
inputs where `nz_array > (hi - lo + 1)`.

#### Why this matters

Pattern A reads like the Fortran source, but it forces XLA to emit a scalar update for each boundary statement and prevents the level dimension from being a single kernel. Pattern B turns "active-range body + boundary statement" into one masked broadcast — which is what the GPU wants.

---

### Prefer vectorized ops over lax.fori_loop for independent iterations

→ Follow **Level 1–2 of the VECTORIZATION PRIORITY LADDER** above: single `jnp` broadcast or `jax.vmap`. Reserve `lax.fori_loop` for loops where iteration `i` genuinely depends on iteration `i-1`.

---

### Column loop: principle and the current recommended technique

**Principle.** When columns are independent, the column loop MUST be vectorised
using the **best available JAX primitive for the target backend**. A sequential
column loop (Python `for col in range(ncol)` *or* `lax.fori_loop(0, ncol, ...)`)
is **never an acceptable fallback** — both serialise the columns and destroy GPU
throughput.

**Current recommended technique:** `jax.vmap` over the column axis. The rest of
this section is written around `vmap` because it is today's right answer for
column-level parallelism in the kernels in scope (Kessler, RRTMG, CLUBB,
single-column physics tendencies).

**If a different JAX primitive is more appropriate for the kernel** — e.g.
`pjit` / `shard_map` for multi-device parallelism, fused `lax.scan` over a
column-axis with carried state, or a future JAX primitive purpose-built for
column-wise vectorisation — use it instead and justify the choice in a
comment. Do not silently substitute it for `vmap` without explanation.

The non-negotiable is **column-level parallelism**, not the specific spelling.
Sequential column iteration is the bug; the primitive is the technique.

#### Why `lax.fori_loop` over the column axis is a serialization bug

Replacing `vmap` with `lax.fori_loop(0, ncol, body, ...)` serialises the columns
inside the JIT binary — every column runs one after another on a single GPU
thread. The code compiles, runs, and produces correct numbers, but throughput
drops by 10×–100× depending on `ncol`. Do **not** treat `lax.fori_loop` as an
acceptable fallback for the column dimension.

#### When are columns independent?

Columns are independent if and only if the body of `do col = 1, ncol` reads and writes only *its own column slice* of every 2-D array. Concretely:

- ✅ Every 2-D array access inside the body uses `col` (or the equivalent column index) as one of its indices.
- ✅ No 2-D array is indexed at `col ± 1`, `col ± k`, or any other column.
- ✅ Reductions across columns (`SUM(field, dim=1)`, etc.) happen *outside* the loop, not inside.
- ✅ Per-column local arrays (e.g. `r(nz)`, `velqr(nz)`, `pc(nz)` in Kessler) are reset on every column iteration.

If all four hold, the columns are independent → **use `jax.vmap`** (or the
better column-parallel primitive, if you have one and can justify it).

Things that **do NOT** create column coupling (even though they may look like they do):

- A `do while` **subcycle** inside the column body — that's a per-column inner loop. `vmap` it like any other body; the `lax.while_loop` runs inside `vmap` independently for each column.
- A `lax.fori_loop` **over levels** inside the column body — levels are sequential *within* a column, but each column still runs independently. `vmap` over columns, `fori_loop` over levels is the canonical pattern.
- **INOUT 2-D arrays** like `theta`, `qv`, `qc`, `qr` — they are written across iterations, but each iteration touches a different column slice. This is exactly the case `vmap` was built for.

#### ❌ Anti-pattern — `lax.fori_loop` over the column axis

```python
# ❌ WRONG — columns processed sequentially inside the JIT binary; no GPU parallelism.
def col_body(col, state):
    theta, qv, qc, qr, precl, ... = state
    # ... per-column work, possibly with a lax.while_loop subcycle inside ...
    theta = theta.at[:, col].set(...)
    qv    = qv.at[:, col].set(...)
    return theta, qv, qc, qr, precl, ...

theta, qv, qc, qr, precl, ... = lax.fori_loop(
    0, ncol, col_body,
    (theta, qv, qc, qr, precl, ...),
)
```

The slicing pattern (`.at[:, col].set(...)`) is the unmistakable signal of this bug: it reveals that the loop body is touching one column at a time. Every time you find yourself writing `.at[:, col]` inside a `lax.fori_loop` over `col`, replace the whole construct with `vmap`.

#### ✅ Correct pattern — `vmap` over columns; anything inside the body

```python
# ✅ CORRECT — all columns batched in parallel; body can contain any per-column construct
def process_one_col(theta_c, qv_c, qc_c, qr_c, ..., dt, lyr_surf_idx, lyr_toa_idx, ...):
    # 1-D per-column arrays — shape (nz,). No `col` index needed.
    # Inside here you may freely use:
    #   - vectorized jnp ops over the level axis
    #   - lax.fori_loop over levels (sequential level dependencies are fine)
    #   - lax.while_loop for per-column subcycling
    #   - lax.cond for per-column branching on errflg
    ...
    return theta_c, qv_c, qc_c, qr_c, ..., precl_c, errflg_c

theta, qv, qc, qr, ..., precl, errflg = jax.vmap(
    process_one_col,
    in_axes=(1, 1, 1, 1, ..., None, None, None),   # 1 for (nz, ncol); None for scalars/static ints
    out_axes=(1, 1, 1, 1, ..., 0, 0),              # 1 for (nz, ncol); 0 for per-column scalars
)(theta, qv, qc, qr, ..., dt, lyr_surf_idx, lyr_toa_idx, ...)
```

Per-column subcycling, per-column error flags, per-column reductions — all of these go **inside** `process_one_col`, not outside it. `vmap` runs the whole inner story once per column, in parallel.

#### Decision rule

> If a Fortran loop is `do col = 1, ncol` AND the body satisfies the four independence checks above, **`jax.vmap` is mandatory**. `lax.fori_loop(0, ncol, ...)` is permitted ONLY when column `col` reads or writes data from column `col ± k` (genuine cross-column dependency) — which essentially never happens in column-physics schemes like Kessler, RRTMG, CLUBB, or the CAM/CESM single-column physics tendencies.

---

### Use jax.vmap to eliminate the column loop

The outer column loop (`do col = 1, ncol`) processes each column independently —
the ideal use case for `jax.vmap`.

```python
# ✅ vmap pattern — define logic for ONE column, vmap over all columns
# (variable names are illustrative — use the actual names from your Fortran source)
def process_one_col(a_c, b_c, rho_c, z_c, tracer_c, out_c):
    # All arrays are 1D (shape: nz) — no col index needed
    field_c = scale * rho_c                              # (nz,)  # illustrative
    derived = coeff * field_c * tracer_c ** exp_val      # (nz,)  # illustrative — use actual variable names from your Fortran source
    scalar_out = rho_c[surf_idx] * tracer_c[surf_idx] / norm_val
    return out_c, tracer_c, scalar_out

# in_axes=1    → slice the column axis (axis 1 of (nz, ncol)) for each 2-D input
# in_axes=None → broadcast scalars shared across all columns (e.g. dt, physical constants)
# out_axes=1   → stack (nz,) outputs back into (nz, ncol)
# out_axes=0   → stack scalar outputs into (ncol,)
out, q, scalar = jax.vmap(
    process_one_col,
    in_axes=(1, 1, 1, 1, 1, 1),
    out_axes=(1, 1, 0),
)(a, b, rho, z, q, out)
```

**Rules for vmap axes:**
- `in_axes=1`    for every `(nz, ncol)` input array
- `in_axes=None` for scalars shared across all columns (`const_a`, `dt`, `const_b`, …)
- `out_axes=1`   for `(nz,)` outputs that expand back to `(nz, ncol)`
- `out_axes=0`   for scalar outputs that expand to `(ncol,)`

---

### Common `vmap` pitfalls (must avoid)

These are silent-miscompute or hard-crash bugs that recur across translations.
They all stem from the same misunderstanding: under `vmap`, the column axis is
vectorised away **only for arrays passed as explicit arguments with the right
`in_axes`**. Closure-captured arrays and "scratch" arrays do not get this
treatment automatically.

#### Pitfall 1 — closure-captured per-column arrays (silent miscompute)

Any array that has a column dimension MUST enter the vmapped function as an
explicit argument with the correct `in_axes`. **Closure-capturing it from the
enclosing scope is wrong** — the captured array keeps its full `(nz, ncol)`
shape inside every vmap iteration, and indexing it with a fixed column index
makes every iteration see the same column.

```python
# ❌ WRONG — cpair, rho, pk are closure-captured 2-D arrays of shape (nz, ncol).
#    Inside process_one_col under vmap, they are NOT mapped — their full shape
#    is preserved. cpair[:, 0] always picks column 0, regardless of which column
#    the current vmap iteration is processing. Every column gets column-0's data.
def kessler_run_core(..., cpair, rho, pk, ...):
    def process_one_col(theta_c, qv_c, ...):
        f5  = 4093.0 * lv / cpair[:, 0]   # ← always column 0!
        r_c = 0.001 * rho[:, 0]           # ← always column 0!
        pc_c = 3.8 / ((pk[:, 0] ** xk) * pref)
        ...
    vmap(process_one_col, in_axes=(1, 1, ..., None))(theta, qv, ...)

# ✅ CORRECT — pass the per-column slice as an explicit vmap argument.
#    in_axes=1 maps each over the ncol axis, so inside the body cpair_c, rho_c,
#    pk_c are the current column's 1-D vertical profiles, shape (nz,).
def kessler_run_core(..., cpair, rho, pk, ...):
    def process_one_col(theta_c, qv_c, ..., cpair_c, rho_c, pk_c):
        f5  = 4093.0 * lv / cpair_c
        r_c = 0.001 * rho_c
        pc_c = 3.8 / ((pk_c ** xk) * pref)
        ...
    vmap(process_one_col,
         in_axes=(1, 1, ..., 1, 1, 1),
         out_axes=(1, 1, ..., 0))(theta, qv, ..., cpair, rho, pk)
```

**The signature of this bug** is any expression of the form `<array>[:, 0]`,
`<array>[k, 0]`, or `<array>[idx, 0]` *inside* the vmapped function, when
`<array>` is closure-captured. The literal `0` for the column index is the
giveaway — it means the author forgot to thread the column slice through
`vmap`.

**Closure capture is only safe** for arrays without a batch dimension:
- 1-D vertical-only fields (`z` of shape `(nz,)`)
- scalars (`dt`, `lv`, `pref`, `rhoqr`)
- module constants (`f2x`)
- static integers (`ncol`, `nz`, `lyr_surf`, `lyr_toa`)

#### Pitfall 2 — inconsistent batch sizes across `vmap` arguments (hard crash)

Every `vmap` argument that is mapped on a real axis must have **the same size
on its mapped axis**. Mixing `(nz, ncol)` arrays mapped on axis 1 (size
`ncol`) with `(nz,)` arrays mapped on axis 0 (size `nz`) gives:

```
ValueError: vmap got inconsistent sizes for array axes to be mapped:
  * most axes (M of them) had size ncol;
  * some axes (K of them) had size nz
```

This typically happens when "scratch" or "workspace" arrays are created with
shape `(nz,)` outside the vmap call and then threaded through it as if they
were per-column inputs:

```python
# ❌ WRONG — r/rhalf/velqr/sed/pc are scratch, length nz, mapped on axis 0.
#    But theta/qv/qc/qr have shape (nz, ncol) and are mapped on axis 1 (size ncol).
#    nz ≠ ncol → vmap errors.
r     = jnp.zeros(nz, dtype=jnp.float64)
rhalf = jnp.zeros(nz, dtype=jnp.float64)
velqr = jnp.zeros(nz, dtype=jnp.float64)
...
vmap(process_one_col,
     in_axes=(1, 1, 1, 1, ..., 0, 0, 0, 0, 0))(theta, qv, qc, qr, ..., r, rhalf, velqr, sed, pc)

# ✅ CORRECT — declare per-column scratch INSIDE process_one_col; never thread
#    through vmap. They are local to a single column anyway.
def process_one_col(theta_c, qv_c, qc_c, qr_c, ...):
    r_c     = jnp.zeros(nz, dtype=jnp.float64)
    rhalf_c = jnp.zeros(nz, dtype=jnp.float64)
    velqr_c = jnp.zeros(nz, dtype=jnp.float64)
    sed_c   = jnp.zeros(nz, dtype=jnp.float64)
    pc_c    = jnp.zeros(nz, dtype=jnp.float64)
    ...
```

**Rule of thumb:** if a value is computed afresh inside the per-column body,
it does not belong in the `vmap` signature. Pass through `vmap` only the
INOUT per-column arrays the caller actually owns.

#### Pitfall 3 — Python LHS-assignment shadows the enclosing scope (UnboundLocalError)

In Python, **any assignment to a name inside a function makes that name local
to that function for its entire body**, including reads that appear textually
*before* the assignment. This is a Python scoping rule, not a JAX one — but
it bites hardest inside `vmap`-mapped functions where the author wants to
update a per-column flag they thought was captured from the outer scope.

```python
# ❌ WRONG — UnboundLocalError on errflg.
def kessler_run_core(..., errflg, ...):
    def process_one_col(theta_c, qv_c, ...):
        # The `errflg = ...` line below makes `errflg` local to process_one_col.
        # That turns the RHS read into an unassigned-local read → UnboundLocalError.
        errflg = jnp.where(dt0 < 1.0e-12, 1, errflg)  # ← raises at call time
        return ...

    theta, qv, ... = jax.vmap(process_one_col, in_axes=(1, 1, ...))(theta, qv, ...)
```

**The signature of this bug** is any assignment of the form `X = ...X...`
inside a function nested under `vmap`, where `X` is not in that function's
parameter list. The fix is **never** `nonlocal` or `global` — those interact
badly with JAX tracing and will silently miscompile or fail to vectorise.
The fix is to **thread the state through the vmap boundary explicitly**:

```python
# ✅ CORRECT — pass errflg through process_one_col as a per-column scalar.
def kessler_run_core(..., errflg, ...):
    # Broadcast a scalar errflg to one entry per column.
    errflg_in = jnp.broadcast_to(errflg, (ncol,))

    def process_one_col(theta_c, qv_c, ..., errflg_c):
        errflg_c = jnp.where(dt0 < 1.0e-12, 1, errflg_c)
        ...
        return theta_c, qv_c, ..., errflg_c

    theta, qv, ..., errflg_out = jax.vmap(
        process_one_col,
        in_axes=(1, 1, ..., 0),
        out_axes=(1, 1, ..., 0),
    )(theta, qv, ..., errflg_in)

    # Collapse the per-column flag back to a scalar (use the reduction that
    # matches the Fortran semantics — usually jnp.max for an error flag,
    # jnp.any for a boolean, etc.).
    errflg = jnp.max(errflg_out)
    return ..., errflg
```

The same pattern applies to **any** mutating scalar or per-column flag inside
a vmapped function: accumulators, status codes, "we did this branch" markers.
Pass in, return out, reduce after vmap if the caller wants a scalar.

Notes that follow from the same rule:
- Do **not** use `nonlocal` or `global` to "fix" this. They break under JAX
  tracing.
- Do **not** wrap the inner read in `lax.cond` thinking it dodges the
  local-binding rule. Python scope is decided at function-definition time,
  not at trace time.
- The check is mechanical: scan every nested function (especially the body
  of any `vmap`'d function), and for every name that gets assigned, verify
  it is **either** a parameter of that function **or** never read on a RHS
  before its first assignment.

#### Mandatory `vmap` self-check (do this before finalising the call)

Before writing the `vmap` invocation, list every array referenced inside the
vmapped function and classify each one:

| Class | Where it should enter | Mapped axis |
|---|---|---|
| Per-column INOUT field (`theta`, `qv`, …) shape `(nz, ncol)` | explicit `vmap` arg | `in_axes=1` (the `ncol` axis) |
| Per-column INOUT scalar (`precl`, `errflg` per col) shape `(ncol,)` | explicit `vmap` arg | `in_axes=0` |
| Per-column INPUT field (`cpair`, `rair`, `rho`, `pk`) shape `(nz, ncol)` | explicit `vmap` arg | `in_axes=1` |
| Vertical-only or 1-D shared field (`z` shape `(nz,)`) | closure capture | n/a |
| Scalar / constant (`dt`, `lv`, `pref`) | closure capture or `in_axes=None` | n/a |
| Static int (`ncol`, `nz`, `lyr_surf`, `lyr_toa`) | closure capture | n/a |
| Per-column scratch (`r`, `rhalf`, `velqr`, `sed`, `pc`) | **declare inside body** | not in vmap signature |

Then verify:
1. Every per-column array appears as a `vmap` argument with the correct
   `in_axes`. None are closure-captured.
2. Every mapped argument has the **same size** on its mapped axis (`ncol`).
3. No scratch / workspace arrays are passed through `vmap`.
4. No `<array>[:, 0]` / `<array>[k, 0]` indexing remains inside the body
   (that's the closure-capture-with-fixed-column bug).
5. For every name assigned inside the vmapped function, that name is **either**
   a parameter of the function **or** never read on a RHS before its first
   assignment (the Pitfall 3 Python-scope rule). Per-column flags (`errflg`
   etc.) MUST be vmap parameters with `in_axes=0`, not closure-captured.

If any of those checks fail, the vmap call is wrong — fix it before writing
the rest.

---

### Use lax.scan instead of while_loop for bounded subcycling

If a `DO WHILE` loop has a **known maximum iteration count**, prefer `lax.scan`
over `lax.while_loop`. `lax.scan` supports `jax.grad` through the loop and gives
XLA better optimisation opportunities.

```python
# ✅ lax.scan for subcycling (max iterations known at trace time)
#
# IMPORTANT: max_steps must be a plain Python int, computed BEFORE entering _core.
# Do NOT call int(jnp.ceil(...)) inside _core — that extracts a value from a tracer.
# Pass max_steps as a static Python int argument, or compute it in the wrapper.

def subcycle_step(carry, _):
    time, dt0, accum, f1, f2, f3, theta, vel = carry
    # ... one subcycle step ...
    return new_carry, None

# max_steps is a Python int (computed in wrapper or passed as static arg)
final_carry, _ = lax.scan(subcycle_step, init_carry, None, length=max_steps)

# Use lax.while_loop only when the iteration count is truly dynamic and unbounded.
```

---

### Summary: when to use each primitive

| Fortran pattern | Recommended JAX | Fallback |
|---|---|---|
| Independent DO loop over levels | Vectorized `jnp` ops | `lax.fori_loop` |
| Independent DO loop over columns | `jax.vmap` (today's recommended technique; a better column-parallel JAX primitive may substitute *with justification*) | — (`lax.fori_loop` or Python `for` over the column axis is a serialization bug — see "Column loop: principle and the current recommended technique" above) |
| Nested independent loops | Full `jnp` broadcast or `vmap` + vectorized | — |
| DO WHILE with bounded iterations | `lax.scan` | `lax.while_loop` |
| DO WHILE with truly unbounded iterations | `lax.while_loop` | — |
| Sequential DO loop (each iter depends on previous) | `lax.fori_loop` or `lax.scan` | — |
| Simple IF/ELSE on arrays | `jnp.where` | `lax.cond` |
| IF/ELSE with large branches or side effects | `lax.cond` | — |
| **Python loop wrapping any JAX construct** | **❌ NEVER** | — |
