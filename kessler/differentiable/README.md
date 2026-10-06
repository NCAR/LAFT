# differentiable — reverse-mode-differentiable Kessler with learnable constants

LAFT Stage 4 (optional) for this project; the playbook is
[`../../LAFT/workflow_differentiable/DIFFERENTIABLE_WORKFLOW.md`](../../LAFT/workflow_differentiable/DIFFERENTIABLE_WORKFLOW.md),
configured by `[differentiable]` in `../config/project.toml`.

`kessler_run_diff.py` is a variant of the validated `kessler_run` translations
in which:

- **every physics constant is a learnable parameter.** These are the 17 fields of
  `KesslerParams`, and their defaults are the `kessler.F90` literals.
- **`jax.grad` works.** The CFL sub-cycling runs as a fixed-length masked
  `scan`, and the fractional powers and CFL divisions have NaN-free forms.
- **soft clamps are optional.** With `clamp="soft"`, every max/min/positivity
  limiter becomes a softplus of width `soft_width` [kg/kg], so gradients also
  pass through limiters that are active.

With the defaults it **is** the validated scheme. It matches the Gemini
translation and the Fortran outputs to ≤ 3e-15 (scaled max error) on the
128 × 56 reference case.

This folder is separate from `bridge_test/`, `out/` and `translations/` on
purpose: the validated translations stay frozen, and nothing here modifies them.

| File | What it is |
|---|---|
| `kessler_run_diff.py` | The differentiable implementation (`kessler_run_diff`, `KesslerParams`, `KesslerOutputs`) |
| `gate_adapter.py` | Kessler hooks for the generic gate (reference case, reference run, Fortran truth, loss) |
| `tests/` | Fast pytest pins: equivalence, flags, parameter kwargs, gradients, soft clamps |
| `reports/` | Latest gate output: `grad_gate_results.json` (PASS/FAIL) and `grad_gate_report.md` |

## Usage

Arrays use the bridge layout `(nz, ncol)`. Level 0 is `lyr_surf`; a reversed
column (`lyr_surf=nz, lyr_toa=1`) is flipped internally. `pref` is in **hPa**,
the value `kessler_init` stores.

```python
from kessler_run_diff import kessler_run_diff, KesslerParams, DEFAULT_PARAMS

out = kessler_run_diff(dt, cpair, rair, rho, z, pk, theta, qv, qc, qr,
                       lv=2.5e6, pref=1000.0, rhoqr=1000.0)          # = the Fortran answer
out.theta, out.qv, out.qc, out.qr, out.precl, out.relhum, out.errflg, out.n_substeps

# constants as keyword arguments ...
out = kessler_run_diff(..., accretion_coeff=3.0, fallspeed_exp=0.15)

# ... or as a pytree, which is what you differentiate / optimise
def loss(p):
    o = kessler_run_diff(..., params=p, clamp="soft", soft_width=1e-6)
    return ((o.precl - precl_obs) ** 2).mean()

grads = jax.grad(loss)(DEFAULT_PARAMS)      # a KesslerParams of dL/dp, one per constant
```

| Option | Default | Meaning |
|---|---|---|
| `params` | `DEFAULT_PARAMS` | `KesslerParams` pytree of constants; keyword overrides are applied on top |
| `clamp` | `"hard"` | `"hard"` = exact limiters; `"soft"` = softplus limiters |
| `soft_width` | `1e-6` | softplus width [kg/kg]; can be traced, e.g. annealed during training |
| `loop` | `"scan"` | `"scan"` supports reverse mode; `"while"` is the faster forward-only path (forward-mode AD still works) |
| `n_max` | `32` | sub-step budget. The reference case needs 7–9 at dt = 60 s. Cost and reverse-mode memory scale with it |
| `cfl_safety` | `0.8` | CFL fraction, a numerical option rather than a parameter |
| `stop_dt0_grad` | `False` | `True` treats the sub-step length as a numerical choice, with no gradient through it |

`errflg`: `0` = ok, `1` = dt ≤ 0, `2` = bad time splitting, `3` = `n_max` too small.
Flags 1 and 2 are the Fortran ones; flag 3 means the scan stopped before
reaching dt. It is never silent.

## Parameters

| Group | Fields (default) |
|---|---|
| Autoconversion, KW78 2.13a | `autoconv_rate` (1e-3 s⁻¹), `autoconv_threshold` (1e-3 kg/kg) |
| Accretion, KW78 2.13b | `accretion_coeff` (2.2), `accretion_exp` (0.875) |
| Fall speed, KW78 2.15 | `fallspeed_coeff` (36.34), `fallspeed_exp` (0.1364) |
| Rain evaporation, KW78 2.14 | `vent_a` (1.6), `vent_b` (124.9), `vent_exp` (0.2046), `evap_exp` (0.525), `evap_denom_c` (2.55e6), `evap_denom_d` (5.4e5) |
| Tetens q_sat, KW78 2.11 | `tetens_e0` (3.8 hPa), `tetens_a` (17.27), `tetens_t0` (273 K), `tetens_t1` (36 K) |
| Condensation, DK83 A13–14 | `cond_coeff` (4093; ≈ `tetens_a·(t0−t1)`, kept as the Fortran literal) |

These are not parameters:
- the 0.001 density unit conversion;
- the 0.8 CFL fraction and the 1e-12 / 1e-5 thresholds, which are numerical
  controls;
- `lv`, `pref` and `rhoqr`, which are already arguments and can be
  differentiated like any input.

## Soft clamps: what they change

Accuracy against hard mode, from the gate's G5 table (max |soft − hard| / max |hard|):

| `soft_width` | θ | q_v | q_c | q_r | precl | relhum |
|---|---|---|---|---|---|---|
| 1e-6 (default) | 3e-5 | 2e-4 | 1e-3 | 2e-4 | 1e-4 | 8e-2 |
| 1e-7 | 5e-6 | 3e-5 | 1e-4 | 2e-5 | 9e-6 | 8e-3 |
| 1e-8 | 6e-7 | 4e-6 | 1e-5 | 2e-6 | 9e-7 | 8e-4 |

- **Convergence is linear in the width.** Relative humidity is the most
  sensitive output: where `q_v` sits on its positivity floor, the softplus
  offset (≤ width·ln 2) is divided by a small `q_sat`.
- **Mixing ratios stay ≥ 0.** Water is conserved only to O(width × sub-steps).
- **The gradient flows on the clamped side, attenuated by `sigmoid(x / width)`.**
  For example, with the autoconversion threshold above every `q_c`, the hard
  gradient w.r.t. the threshold is exactly 0 and the soft one is not.
- **A larger width gives smoother, more informative gradients and a less
  faithful forward model.** Annealing the width during training is the usual
  compromise.
- **Sub-step selection and error logic always stay hard.** The sub-step count
  is piecewise constant, so its jumps carry no gradient.

## Running

From the project root (`kessler/`):

```bash
python ../LAFT/workflow_differentiable/phase06_01_grad_gate.py   # Stage 4 gate → reports/
python -m pytest differentiable/tests -v
python -m pytest ../LAFT/workflow_differentiable/test_diff_primitives.py -v
```

The committed `reports/` were produced on CPU with JAX 0.6.2 (the
`jax-validate` pin). They show the gate status `PASS`. Reverse mode agrees with
forward mode to ~4e-15, and with finite differences to ≤ 1e-6, for every
parameter in both clamp modes.

Cost at the reference case (CPU, 128 × 56): forward 2.4 ms (`while`) and
3.7 ms (`scan`, n_max = 16); gradient of all 17 constants 17 ms (hard) and
30 ms (soft).
