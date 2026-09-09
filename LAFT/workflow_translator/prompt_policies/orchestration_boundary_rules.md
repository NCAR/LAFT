## Orchestration Boundary Rules (CRITICAL)

These rules define which procedures may be translated as host-side (plain
Python) layers and which MUST receive the full JAX two-layer contract
(`<proc>_core` decorated with `@jax.jit` + wrapper). They exist to close a
specific failure mode: a scheme's top-level routine being emitted as
host-side Python loops that call jitted helpers — code that produces correct
numbers but serializes the grid on CPU, is invisible to XLA-level profiling,
and destroys GPU throughput.

---

### The invariant

> **Any procedure whose signature carries grid/state arrays (a non-scalar
> procedure) gets the full JAX contract — jitted `_core` + wrapper — no
> matter where it sits in the call graph.**

Position grants no exemption. In particular, ALL of the following are
non-scalar COMPUTE procedures, even though they "orchestrate" other
procedures:

- the scheme's **main/top-level routine** (the one the driver calls per step),
- **model-facing wrappers** that marshal fields between a host model layout
  and the scheme (unit conversions, array packing/unpacking ARE grid
  compute),
- any intermediate routine that loops over columns, levels, categories, or
  any other axis of the physics state while doing numeric work.

If a routine both sequences calls AND does per-element numeric work, it is a
COMPUTE procedure. "It mostly just calls other things" is not an exemption —
the per-element work must move inside the JIT boundary (vectorized `jnp`
ops, `jax.vmap`, `lax.fori_loop`/`lax.scan` — see the VECTORIZATION PRIORITY
LADDER in the vectorization rules).

---

### What host-side layers are allowed to be

A procedure may be translated as a host-side plain-Python layer ONLY if it
is **scalar-only**: no grid/state arrays in its signature. Typical
legitimate host-side layers:

- configuration readers (namelist / config-file parsing),
- init-time lookup-table file loaders (`has_io=True`; NumPy allowed, and
  vectorized parsing such as `np.loadtxt`/`np.frombuffer` is preferred over
  element loops),
- init sequencing that sets scalar module constants.

Everything else compiles.

---

### Forbidden in ANY translated procedure (host code included)

1. **A Python `for`/`while` loop that writes array elements**
   (`x[i] = ...`, `x[i] += ...`, `x[i, k] = ...`). Element-wise mutation
   over a grid axis is compute; it belongs inside a jitted `_core` as a
   vectorized/whole-array operation.

2. **A Python `for`/`while` loop wrapping any JAX construct**
   (`jnp.*`, `lax.*`, `jax.*`, or a call to any `<proc>_core`). This is
   Level 5 of the VECTORIZATION PRIORITY LADDER — never acceptable. A
   Python loop over columns calling a jitted helper per column serializes
   the grid exactly like a `lax.fori_loop` over columns, with kernel-launch
   overhead added on top.

The one exception for pattern 1: scalar-only I/O loaders (`has_io=True`)
filling arrays from file records — and even there, prefer vectorized
parsing.

These two patterns are enforced mechanically by the lint (they are hard
FAILs, not warnings), and the dependency phase refuses to classify any
non-scalar procedure as host-side. Do not attempt to work around either
gate; if a routine seems impossible to jit, that is a design problem to
raise, not a classification to relax.

---

### Why this matters (a lesson from an earlier translation)

In an earlier translation, the scheme's ~2000-line main routine was
classified as a host-side "orchestration" layer. Every per-column/per-level
physics loop ran as Python `for` loops over NumPy arrays, calling dozens of
tiny jitted kernels per element. The result validated bit-for-bit against
Fortran — and was structurally incapable of GPU batching: XLA-level
profiling could not even see the defect, because the loops never compiled.
The fix required a full rewrite. These rules make that classification
impossible from the start.
