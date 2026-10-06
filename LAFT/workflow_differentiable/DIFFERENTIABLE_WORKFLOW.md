# Differentiable Workflow — learnable-parameter, reverse-mode-differentiable variant

You are deriving a **differentiable variant** of a validated translation: the
same scheme, with its physics constants exposed as learnable parameters,
reverse-mode AD (`jax.grad`) working end to end, and an optional soft-clamp
mode so gradients also flow through active limiters. In its default ("hard")
mode the variant must reproduce the validated translation, and through it the
Fortran, to round-off. That is what makes its gradients the gradients *of the
validated scheme*, not of a look-alike.

This document is **codebase-agnostic**: every project value comes from the
`[differentiable]` section of `config/project.toml`. Stage 4 is optional and
runs only when `[differentiable].enabled = true`.

**Working directory: the project root** (the directory containing `config/`).

Worked instance: `kessler/differentiable/` (`kessler_run_diff.py`,
`gate_adapter.py`, `tests/`, `reports/`).

## Where it lives, and what it never touches

The variant is a **separate artifact**, in the project's `differentiable/`
folder (`[differentiable].module`), next to its own tests and gate reports:

```
<project>/differentiable/
├── <proc>_diff.py         # the differentiable implementation        ([differentiable].module)
├── gate_adapter.py        # project hooks for the gate               ([differentiable].adapter)
├── tests/                 # fast pytest pins, separate from bridge_test/
├── reports/               # grad_gate_results.json + grad_gate_report.md ([differentiable].report_dir)
└── README.md
```

It **never edits** `out/jax/`, the `.validated.py` snapshots, the
`translations/<llm>/` archives, `bridge_test/` or the bridge. The validated
translation stays the frozen measurement the paper reports; the variant is
derived from it and gated against it. Because it lives outside `out/`,
`tools/clean_AI_results.sh` does not touch it and `copy_AI_results.sh` does not
need to archive it.

## Entry gate

- Stage 2 is complete: comparison `ALL PASS` and a `.validated.py` snapshot of
  `[differentiable].target_proc` exists (`[differentiable].reference`).
- `[differentiable].enabled = true`, with `target_proc`, `module`, `adapter`,
  `reference` set (`framework_config.Config.differentiable` refuses otherwise).

## Step 1 — Author the variant (in-context edit)

Start from the validated translation (`[differentiable].reference`) and apply
the transformations below. Use the shared primitives in
`workflow_differentiable/diff_primitives.py`; do not re-implement them inline.

| # | Construct in the validated translation | Replace with | Why |
|---|---|---|---|
| T1 | `lax.while_loop` with a data-dependent trip count (sub-cycling, iteration to convergence) | `bounded_while(cond, body, init, n_max=..., loop=...)` | JAX cannot reverse-differentiate a dynamic-trip-count loop. `loop="scan"` is a fixed-length masked scan that **freezes** converged state (a zero-length step is often not a no-op); `loop="while"` keeps the fast forward path. Both enforce `n_max`. |
| T2 | `x ** p` with `0 < p < 1` where `x` can be 0 | `safe_pow(x, p)` | `p*x**(p-1)` is infinite at 0, so reverse mode yields NaN. |
| T3 | `jnp.where(mask, a / b, c)` where `b` can be 0 off-mask | `jnp.where(mask, safe_div(a, b, mask), c)` | The unselected branch's infinite partial times a zero cotangent is NaN. |
| T4 | Literal constants in process formulas | Fields of a `NamedTuple` params pytree (defaults = the Fortran literals, exactly) plus `**param_overrides` keywords | Learnable parameters; `jax.grad` w.r.t. the pytree gives every sensitivity at once. |
| T5 | `max`/`min`/positivity limiters in process updates | `C = make_clamps(clamp, soft_width)`, then `C.relu` / `C.max` / `C.min` | `clamp="hard"` is exact; `clamp="soft"` is softplus-smoothed with width in the clamped quantity's units. |

Rules:

1. **Hard mode must be the validated scheme.** Keep the floating-point
   association of the Fortran where it is cheap to (`(0.8*dz)/v`, not
   `0.8*(dz/v)`); G1 is a round-off gate.
2. **Parameters are physical constants only.** Unit conversions, numerical
   safety factors (CFL fractions), convergence tolerances and thresholds that
   exist only to avoid a division by zero stay literals or become *keyword
   options*, not parameters. Module variables already arrive as arguments.
   Keep the Fortran literal even where it is a rounded derived value (e.g. the
   Kessler `4093 ≈ 17.27 × 237`), so the default stays exact. Document the
   relation in the field's comment.
3. **Soft clamps apply to process limiters, not to control flow.** Loop
   conditions, sub-step selection (CFL `min`) and error-flag logic stay hard:
   they decide *how* the scheme integrates, not *what* it computes. The sub-step
   count is piecewise constant, so its jumps carry no gradient. Offer
   `stop_dt0_grad`-style switches if the sub-step length should not either.
4. **The width has units.** Choose one per clamped quantity, and say which in
   the module docstring. If clamps act on quantities with different units, give
   each its own width.
5. **Error flags are data.** Strings stay out of the jitted core, as in the
   translation. Add a flag value for "iteration budget exhausted" so a too-small
   `n_max` can never pass silently.
6. **`n_max` is sized from the problem.** The scan always runs `n_max`
   iterations: the cost is linear in it, and so is reverse-mode memory, even
   with checkpointing. Report the sub-steps actually used, and set `n_max` in
   the config with headroom over the reference case.

## Step 2 — Write the adapter

`[differentiable].adapter` is a small hand-authored module, the analogue of
`[profiler].inputs_script`, so the gate stays project-agnostic. It must define:

```python
def load_case() -> dict                                   # reference inputs (+ loss scales)
def default_params() -> pytree                            # the params NamedTuple defaults
def input_fields() -> list[str]                           # inputs whose gradients are checked
def output_fields() -> list[str]
def run(case, params, *, inputs=None, clamp, soft_width, loop, n_max) -> dict
def run_reference(case, reference_path) -> dict           # the validated translation
def fortran_reference() -> dict | None                    # optional: Fortran ground truth
def loss(outputs, case) -> scalar                         # touches every output, O(1) each
```

Use the same reference inputs Stage 2 compared against, so G1 is a direct
continuation of the Stage 2 comparison.

## Step 3 — Run the gate

```bash
python ../LAFT/workflow_differentiable/phase06_01_grad_gate.py
```

| Gate | Checks | Tolerance |
|---|---|---|
| G1 equivalence | hard mode vs the validated translation and vs Fortran (max \|diff\| / max \|ref\| per field); scan vs while; `errflg == 0` (so `n_max` suffices) | `tol_reference` |
| G2 finite | `jax.grad` of the loss w.r.t. every parameter and every listed input is finite, per clamp mode | — |
| G3 ad | reverse mode · v = forward mode (`jax.jvp`) along a random direction, through the scan **and** the independent while path | `tol_ad` |
| G4 fd | reverse mode vs central finite differences for every parameter (error in `p·dL/dp`) and along a random input direction | `tol_fd` |
| G5 soft | soft → hard convergence is monotone for every field over `soft_width_sweep` | — |

Writes `<report_dir>/grad_gate_results.json` (`"status": "PASS" | "FAIL"`) and
`grad_gate_report.md`. Exit 0 = PASS. It also reports, without gating, the
parameters whose gradient is exactly zero in each mode (the limiters' dead
zones that soft clamps exist to open) and forward and gradient wall-clock.

Then run the project's fast pins:

```bash
python -m pytest differentiable/tests -v
python -m pytest ../LAFT/workflow_differentiable/test_diff_primitives.py -v
```

**Green when:** `grad_gate_results.json` → `"status": "PASS"` and both pytest
suites pass.

## Reading a failure

| Symptom | Usual cause |
|---|---|
| G1 fails at ~1e-16 · (sub-steps) | Floating-point association differs from the reference (Rule 1). Usually harmless, but the gate is round-off; match the order. |
| G1 fails with `errflg = 3` | `n_max` too small for the reference case. Raise it in the config. |
| G1 fails at O(1) in one field | A transformation changed the scheme (e.g. a clamp converted outside the hard path, or a constant default not equal to the literal). |
| G2 NaN | A remaining fractional power or in-`where` division (T2/T3), or a `sqrt`/`log` at 0. |
| G3 scan ≠ while | The scan does not freeze converged state (T1), or `n_max` truncates one path. |
| G4 fails for one parameter in hard mode only | A kink of a hard limiter lies within ±`fd_rel_step` of the default. Confirm with soft mode and a smaller step before suspecting the gradient. |
| G5 not monotone | A soft clamp is applied to control flow (Rule 3), or the width's units are wrong for that clamp (Rule 4). |
