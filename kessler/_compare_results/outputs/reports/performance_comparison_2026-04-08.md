# Comparative Performance Analysis: Claude vs GPT vs Gemini vs Qwen
## Fortran → JAX Translation of Kessler Microphysics

*Generated: 2026-04-08*

Dimensions compared:
1. Runtime validation (JIT pass/fail)
2. Lint score (static JAX idiom checks)
3. Numerical accuracy vs Fortran reference
4. Physics sanity checks
5. Code complexity (line counts)
6. JAX primitive usage
7. Per-model design analysis
8. Vectorization strategy

---

## 1. Runtime Validation Status

### kessler_init

| Check | Claude | GPT | Gemini | Qwen |
| --- | --- | --- | --- | --- |
| Module imports OK | ✓ | ✓ | ✓ | ✓ |
| Wrapper callable | ✓ | ✓ | ✓ | ✓ |
| Core JIT callable | ✓ | ✓ | ✓ | ✓ |
| Outputs finite | ✓ | ✓ | ✓ | ✓ |

### kessler_run

| Check | Claude | GPT | Gemini | Qwen |
| --- | --- | --- | --- | --- |
| Module imports OK | ✓ | ✓ | ✓ | ✓ |
| Wrapper callable | ✓ | ✓ | ✓ | ✓ |
| Core JIT callable | ✓ | ✓ | ✓ | ✓ |
| Outputs finite | ✓ | ✓ | ✓ | ✓ |


---

## 2. Lint Score

| Model | kessler_init | kessler_run | Total score |
| --- | --- | --- | --- |
| Claude | 8/8 | 10/10 | **100%** |
| GPT | 8/8 | 10/10 | **100%** |
| Gemini | 8/8 | 10/10 | **100%** |
| Qwen | 8/8 | 10/10 | **100%** |

> Lint checks: uses `jax`/`jnp`, no raw numpy, has wrapper+core, no I/O in core, correct signature, no `swapaxes`, bridge metadata present.

---

## 3. Numerical Accuracy (JAX vs Fortran reference)

| Variable | Claude MAE | GPT MAE | Gemini MAE | Qwen MAE | Unit |
| --- | --- | --- | --- | --- | --- |
| theta | 7.851e-16 | 7.296e-16 | 7.296e-16 | 1.074e-01 | K |
| qv | 4.029e-19 | 3.685e-19 | 3.685e-19 | 4.888e-05 | kg/kg |
| qc | 2.194e-19 | 2.194e-19 | 2.194e-19 | 7.962e-05 | kg/kg |
| qr | 2.431e-18 | 2.429e-18 | 2.429e-18 | 2.588e-04 | kg/kg |
| precl | 1.863e-20 | 1.821e-20 | 1.821e-20 | 1.121e-05 | m/s |
| relhum | 7.660e-15 | 7.583e-15 | 7.583e-15 | 2.661e-01 | % |

---

## 4. Physics Sanity Checks

| Check | Claude | GPT | Gemini | Qwen |
| --- | --- | --- | --- | --- |
| theta in 200–400 K | ✓ | ✓ | ✓ | ✓ |
| qv ≥ 0 | ✓ | ✓ | ✓ | ✓ |
| qc ≥ 0 | ✓ | ✓ | ✓ | ✓ |
| qr ≥ 0 | ✓ | ✓ | ✓ | ✓ |
| precl ≥ 0 | ✓ | ✓ | ✓ | ✓ |
| relhum ≥ 0 | ✓ | ✓ | ✓ | ✓ |
| **Score** | **6/6** | **6/6** | **6/6** | **6/6** |

---

## 5. Code Complexity (line counts)

| File | Claude | GPT | Gemini | Qwen |
| --- | --- | --- | --- | --- |
| kessler_run.py | 302 | 340 | 211 | 114 |
| kessler_init.py | 35 | 21 | 23 | 32 |
| **Total** | **337** | **361** | **234** | **146** |

---

## 6. JAX Primitive Usage (kessler_run.py)

| Primitive | Claude | GPT | Gemini | Qwen |
| --- | --- | --- | --- | --- |
| `jax.vmap` | 2 | 3 | 1 | 0 |
| `lax.fori_loop` | 0 | 1 | 0 | 1 |
| `lax.while_loop` | 1 | 3 | 1 | 1 |
| `lax.cond` | 2 | 1 | 2 | 0 |
| `jnp.where` | 4 | 4 | 8 | 3 |
| `jnp.arange` | 2 | 2 | 1 | 7 |
| `jnp.clip` | 0 | 0 | 1 | 0 |
| `jnp.min` | 2 | 2 | 2 | 5 |
| Vectorized level ops (`jnp.arange`, `jnp.clip`, `jnp.min`) — total | 4 | 4 | 4 | 12 |

---

## 7. Per-Model Design Analysis

### kessler_init

All models correctly translated `kessler_init` as **scalar Python with no JAX primitives**. This is the right call: the procedure only copies three scalar constants (`lv`, `pref`, `rhoqr`) with a single unit conversion (`Pa → hPa`), so there is nothing to vectorize or JIT-compile.

> **TOA (Top Of Atmosphere)**: the uppermost vertical level in the atmospheric column (`lyr_toa`). The sedimentation loop computes a flux from level `k` to level `k+1`, but at the TOA there is no level above — making boundary handling necessary. Claude, GPT, and Qwen treat it as an explicit special case; Gemini handles it implicitly via `jnp.clip` (clamps the `k+1` index to `[0, nz-1]`) and `jnp.where` (masks the TOA contribution inline), eliminating the special case entirely.

### kessler_run — Primitive Choices per Design Aspect

| Design Aspect | Claude | GPT | Gemini | Qwen |
| --- | --- | --- | --- | --- |
| Column loop | `jax.vmap` **[good]** | `jax.vmap` **[good]** | `jax.vmap` **[best]** | `lax.fori_loop (sequential)` **[poor]** |
| Level vec. | `jnp.arange(surf→toa) subset` **[good]** | `jnp.arange(surf→toa) subset` **[good]** | `jnp.arange(nz) full array` **[best]** | `jnp.arange(surf→toa) subset in fori_loop` **[ok]** |
| TOA boundary | `explicit lyr_toa_idx scalar index` **[ok]** | `explicit lyr_toa_idx scalar index` **[ok]** | `jnp.clip + jnp.where inline masking` **[best]** | `explicit .at[lyr_toa_idx].set()` **[ok]** |
| CFL reduction | `jnp.min(jnp.concatenate([dt, cands]))` **[ok]** | `jnp.min(jnp.concatenate([dt, cands]))` **[ok]** | `jnp.min(safe_dt) — direct, no concat` **[best]** | `jnp.min(jnp.where(...))` **[ok]** |
| Error branch | `lax.cond per column` **[good]** | `lax.cond (multiple calls)` **[good]** | `lax.cond top-level bypass (dt<=0)` **[good]** | `jnp.where propagation, no lax.cond skip` **[ok]** |
| Subcycle loop | `lax.while_loop` **[good]** | `lax.while_loop (multiple)` **[good]** | `lax.while_loop` **[good]** | `lax.while_loop` **[good]** |

### Design Analysis

**Claude** — Uses `jax.vmap` for column parallelism (all columns batched into one parallel kernel) and a subset `jnp.arange(surf→toa)` for level vectorization (only active levels included). The TOA boundary is handled as a separate scalar index, and the CFL reduction prepends `dt` to the candidate array before calling `jnp.min`. Error guarding uses a `lax.cond` per column to skip subcycling cleanly when `errflg != 0`. The design is correct and GPU-friendly; the main gap relative to Gemini is the subset-level pattern which requires explicit boundary special-casing.

**GPT** — Structurally mirrors Claude in every key design dimension: `jax.vmap` for columns, subset `jnp.arange` for levels, `lax.cond`-guarded subcycling, and the same concat-then-min CFL pattern. GPT applies `lax.cond` and `lax.while_loop` at multiple call sites within the subcycle body for finer-grained in-JIT branching, but this does not change the overall vectorization quality.

**Gemini** — Most GPU-idiomatic translation. The defining pattern is `jnp.arange(nz)` over **all** `nz` levels (not a subset), combined with `jnp.clip` to prevent out-of-bounds on the `k+1` offset and `jnp.where` to mask the TOA level inline. This eliminates all boundary special-casing and allows the level dimension to compile into a single fused XLA kernel with no sequential structure. The CFL reduction is a direct `jnp.min(safe_dt)` — no concatenation needed. `lax.cond` at the top of `kessler_run_core` bypasses the entire computation when `dt <= 0`, keeping the JIT graph clean. The three-op signature (`jnp.arange + jnp.clip + jnp.min`) is the hallmark of this full-array pattern.

**Qwen** — Uses `lax.fori_loop` to iterate over columns sequentially instead of `jax.vmap`. This is the most significant GPU efficiency gap: `vmap` batches all columns into a single parallel kernel, while `fori_loop` processes them one at a time inside the JIT-compiled function, preventing the XLA compiler from exploiting inter-column parallelism. Within each column, levels are correctly vectorized with a subset `jnp.arange`, and `lax.while_loop` drives subcycling. Error propagation uses `jnp.where` rather than `lax.cond`, so the subcycle loop still executes even when an error flag is set (no early exit).

### Overall Primitive Quality Summary

| Model | Column primitive | Level primitive | Boundary | Assessment |
| --- | --- | --- | --- | --- |
| Claude | `jax.vmap` | subset `jnp.arange` | scalar index | Good — subset pattern adds minor boundary overhead |
| GPT | `jax.vmap` | subset `jnp.arange` | scalar index | Good — same quality as Claude, heavier lax.cond usage |
| Gemini | `jax.vmap` | full `jnp.arange(nz)` + clip | `jnp.clip`+`jnp.where` | **Best — single fused kernel, no boundary special-casing** |
| Qwen | `lax.fori_loop` | subset `jnp.arange` | scalar index | Poor — sequential columns eliminate GPU parallelism |


---

## 8. Vectorization Strategy

| Aspect | Claude | GPT | Gemini | Qwen |
| --- | --- | --- | --- | --- |
| Column loop | jax.vmap — all columns batched in parallel | jax.vmap — all columns batched in parallel | jax.vmap — all columns batched in parallel | lax.fori_loop — columns processed sequentially (no vmap) |
| Level loop | Subset indexing: jnp.arange(lyr_surf_idx, lyr_toa_idx+1) — active levels only | Subset indexing: jnp.arange(lyr_surf_idx, lyr_toa_idx+1) — active levels only | Full-array: jnp.arange(nz) over ALL levels + jnp.clip for safe k+1 offset + jnp.where masking of TOA boundary | Subset indexing: jnp.arange(lyr_surf_idx, lyr_toa_idx, lyr_step) inside each column iteration |
| CFL reduction | jnp.min over concatenated [dt, per-level CFL candidates] | jnp.min over per-level CFL candidates | jnp.min(safe_dt) — direct scalar reduction from full-level array | jnp.min(jnp.where(...)) — scalar reduction within column loop |
| Subcycle loop | lax.while_loop | lax.while_loop (multiple — one per loop nest) | lax.while_loop | lax.while_loop |
| Error handling | lax.cond — skips subcycling if errflg != 0 | lax.cond — multiple uses for in-JIT error branching | lax.cond — top-level bypass if dt <= 0 | jnp.where for errflg propagation |

### Why Gemini's strategy is most GPU-idiomatic

All models received the same prompt with the hint *'prefer vectorized jnp array operations over explicit loops wherever iterations are independent'*. They applied it differently:

- **Claude / GPT**: Vectorize levels using a *subset index array* `jnp.arange(lyr_surf_idx, lyr_toa_idx+1)` — only active levels are included. The TOA boundary must be handled separately. Columns are parallelized via `jax.vmap`.

- **Gemini**: Vectorize levels using the *full index array* `jnp.arange(nz)` over all `nz` levels at once. `jnp.clip` prevents out-of-bounds when computing `k+1` at the top boundary; `jnp.where` masks the TOA level inline. No boundary special-casing is needed. Columns are parallelized via `jax.vmap`.
  The three-op signature `jnp.arange + jnp.clip + jnp.min` is the indicator of this pattern — it produces a single fused XLA kernel per loop body, with no sequential structure for the level dimension.

- **Qwen**: Levels are vectorized (subset `jnp.arange`), but columns are processed **sequentially** via `lax.fori_loop` instead of `jax.vmap`. This is the most significant GPU efficiency gap: `vmap` batches all columns into a single parallel kernel; `fori_loop` iterates them one at a time inside the JIT-compiled function.

### Summary

| Model | Overall strategy | GPU parallelism |
| --- | --- | --- |

| Claude | vmap + subset-level vectorization + lax.cond error guard | High |
| GPT | vmap + subset-level vectorization + heavy lax.cond/while_loop usage | High |
| Gemini | vmap + full-array level vectorization (jnp.arange/clip/where) — most GPU-idiomatic | High |
| Qwen | fori_loop columns (sequential) + subset-level vectorization — least GPU-parallel | Low |


---

*Data sources: `translations/*/lint/results/_summary.json`, `translations/*/validation/results/*.json`, `translations/*/reports/compare_results_fortran_jax.txt`, direct file inspection of `translations/*/jax/kessler_run.py`.*