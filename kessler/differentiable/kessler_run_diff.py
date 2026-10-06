"""
kessler_run_diff — differentiable variant of the validated Kessler translation.

Derived from the validated JAX translations of kessler.F90 (translations/
*/jax/kessler_run.validated.py; the per-column vmap + whole-column vector form
of the Gemini 3.1 Pro translation, the fastest of the four). Three changes,
each switchable, none of which alters the default answer:

  * Physics constants are LEARNABLE PARAMETERS. Every literal in the Fortran
    process formulas is a field of ``KesslerParams`` (defaults = the Fortran
    literals). Pass a ``params`` pytree and/or individual keyword overrides
    (``accretion_coeff=3.0``); ``jax.grad`` w.r.t. ``params`` gives the
    sensitivity to every constant at once.
  * Reverse-mode AD. The CFL sub-cycling ``while_loop`` becomes a fixed-length
    masked ``scan`` (``loop="scan"``, budget ``n_max``); fractional powers and
    CFL divisions use NaN-free forms. ``loop="while"`` keeps the fast
    forward-only path.
  * Optional SOFT CLAMPS. ``clamp="soft"`` replaces every max/min/positivity
    limiter in the process updates with a softplus-smoothed version of width
    ``soft_width`` [kg/kg], so gradients also flow through active limiters.

With the defaults (``clamp="hard"``) the result matches the validated
translation and the Fortran reference to round-off — enforced by the gate
(LAFT/workflow_differentiable/phase06_01_grad_gate.py) and by
differentiable/tests/.

Layout follows the bridge convention: 2D arrays are (nz, ncol), level index 0
is ``lyr_surf`` (1-based ``lyr_surf``/``lyr_toa`` as in Fortran; reversed
columns are flipped internally). ``pref`` is in hPa, i.e. the value
``kessler_init`` stores (pref_in / 100), exactly as the translations' core.

What is NOT a parameter, and why:
  * 0.001 (kg/m^3 -> g/cm^3): a unit conversion.
  * the CFL safety factor 0.8, the 1e-12 fall-speed and 1e-5 s convergence
    thresholds: numerical controls (``cfl_safety`` is a keyword option).
  * lv, pref, rhoqr: already arguments (module variables), differentiable
    like any other input.
The sub-step COUNT is piecewise constant in the inputs, so its jumps carry no
gradient; the sub-step LENGTH does (set ``stop_dt0_grad=True`` to treat it as
a pure numerical choice).
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_LAFT = Path(__file__).resolve().parents[2] / "LAFT" / "workflow_differentiable"
if str(_LAFT) not in sys.path:
    sys.path.insert(0, str(_LAFT))
from diff_primitives import bounded_while, make_clamps, safe_div, safe_pow  # noqa: E402


class KesslerParams(NamedTuple):
    """Kessler (1969) / Klemp & Wilhelmson (1978) constants; defaults = kessler.F90."""

    # Autoconversion, KW78 Eq. 2.13a: k1 * max(qc - a, 0)
    autoconv_rate: float = 0.001          # k1 [1/s]
    autoconv_threshold: float = 0.001     # a [kg/kg]
    # Accretion (collection of cloud by rain), KW78 Eq. 2.13b: k2 * qc * qr**e
    accretion_coeff: float = 2.2          # k2 [1/s]
    accretion_exp: float = 0.875
    # Rain terminal velocity, KW78 Eq. 2.15: c * rhalf * (rho_gcc * qr)**e [m/s]
    fallspeed_coeff: float = 36.34
    fallspeed_exp: float = 0.1364
    # Rain evaporation, KW78 Eq. 2.14 / DK83 A8-A9:
    #   (a + b*(rho qr)**e1) * (rho qr)**e2 / (c * pc/(e0 qvs) + d)
    vent_a: float = 1.6
    vent_b: float = 124.9
    vent_exp: float = 0.2046
    evap_exp: float = 0.525
    evap_denom_c: float = 2550000.0
    evap_denom_d: float = 540000.0
    # Tetens saturation mixing ratio, KW78 Eq. 2.11:
    #   qvs = (e0 / p_hPa) * exp(a * (T - t0) / (T - t1))
    tetens_e0: float = 3.8                # [hPa], ~ eps * 6.11 hPa
    tetens_a: float = 17.27
    tetens_t0: float = 273.0              # [K]
    tetens_t1: float = 36.0               # [K]
    # Condensation-rate coefficient, DK83 A13-A14 (= tetens_a * (t0 - t1) in
    # theory, 4093 as a literal in the Fortran; kept separate so the default
    # stays bit-identical).
    cond_coeff: float = 4093.0


DEFAULT_PARAMS = KesslerParams()


class KesslerOutputs(NamedTuple):
    theta: jax.Array      # (nz, ncol) potential temperature [K]
    qv: jax.Array         # (nz, ncol) [kg/kg]
    qc: jax.Array         # (nz, ncol) [kg/kg]
    qr: jax.Array         # (nz, ncol) [kg/kg]
    precl: jax.Array      # (ncol,) precipitation rate [m_water/s]
    relhum: jax.Array     # (nz, ncol) [%]
    errflg: jax.Array     # () int32: 0 ok, 1 dt <= 0, 2 bad time splitting,
                          #           3 sub-step budget n_max exhausted
    n_substeps: jax.Array  # (ncol,) int32 sub-steps actually taken


def kessler_run_diff(dt, cpair, rair, rho, z, pk, theta, qv, qc, qr, lv, pref, rhoqr, *,
                     params: KesslerParams | None = None,
                     clamp: str = "hard", soft_width=1e-6,
                     loop: str = "scan", n_max: int = 32,
                     lyr_surf: int = 1, lyr_toa: int | None = None,
                     cfl_safety: float = 0.8, stop_dt0_grad: bool = False,
                     **param_overrides) -> KesslerOutputs:
    """Differentiable kessler_run. Returns ``KesslerOutputs`` (see module docstring).

    Learnable constants: ``params`` (a ``KesslerParams`` pytree) and/or keyword
    overrides of its fields, e.g. ``kessler_run_diff(..., accretion_coeff=3.0)``.
    Gradients: ``jax.grad(lambda p: loss(kessler_run_diff(..., params=p)))(DEFAULT_PARAMS)``.
    """
    params = DEFAULT_PARAMS if params is None else params
    if param_overrides:
        unknown = set(param_overrides) - set(KesslerParams._fields)
        if unknown:
            raise TypeError(f"unknown Kessler parameter(s): {sorted(unknown)}; "
                            f"valid: {list(KesslerParams._fields)}")
        params = params._replace(**param_overrides)
    nz = theta.shape[0]
    lyr_toa = nz if lyr_toa is None else lyr_toa
    if {lyr_surf, lyr_toa} != {1, nz}:
        raise ValueError(f"lyr_surf/lyr_toa must span the full column (1 and {nz}), "
                         f"got {lyr_surf}/{lyr_toa}")
    return _kessler_core(dt, cpair, rair, rho, z, pk, theta, qv, qc, qr, lv, pref, rhoqr,
                         params, soft_width, cfl_safety,
                         clamp=clamp, loop=loop, n_max=int(n_max),
                         flip=lyr_surf > lyr_toa, stop_dt0_grad=bool(stop_dt0_grad))


@functools.partial(jax.jit, static_argnames=("clamp", "loop", "n_max", "flip", "stop_dt0_grad"))
def _kessler_core(dt, cpair, rair, rho, z, pk, theta, qv, qc, qr, lv, pref, rhoqr,
                  params, soft_width, cfl_safety, *, clamp, loop, n_max, flip, stop_dt0_grad):
    f64 = jnp.float64
    fields = [jnp.asarray(a, f64) for a in (cpair, rair, rho, z, pk, theta, qv, qc, qr)]
    if flip:  # internal convention: index 0 = surface
        fields = [a[::-1] for a in fields]
    dt, lv, pref, rhoqr = (jnp.asarray(s, f64) for s in (dt, lv, pref, rhoqr))
    P = jax.tree_util.tree_map(lambda v: jnp.asarray(v, f64), params)
    C = make_clamps(clamp, jnp.asarray(soft_width, f64))
    cfl = jnp.asarray(cfl_safety, f64)
    err_dt = jnp.where(dt <= 0.0, 1, 0).astype(jnp.int32)

    def column(cp_c, rd_c, rho_c, z_c, pk_c, th_c, qv_c, qc_c, qr_c):
        # ---- level constants (Fortran first parallel region) ----
        f5 = P.cond_coeff * lv / cp_c
        xk = cp_c / rd_c
        r = 0.001 * rho_c
        rhalf = jnp.sqrt(rho_c[0] / rho_c)
        pc = P.tetens_e0 / ((pk_c ** xk) * pref)
        qr_c = C.relu(qr_c)

        def fallspeed(qr_):
            return P.fallspeed_coeff * rhalf * safe_pow(qr_ * r, P.fallspeed_exp)

        dz = z_c[1:] - z_c[:-1]
        dz_top = 0.5 * (z_c[-1] - z_c[-2])

        def cfl_dt0(velqr, cap):
            v = velqr[:-1]
            m = jnp.abs(v) > 1.0e-12
            cand = jnp.where(m, safe_div(cfl * dz, v, m), cap)
            d = jnp.minimum(cap, jnp.min(cand))
            return jax.lax.stop_gradient(d) if stop_dt0_grad else d

        velqr = fallspeed(qr_c)
        dt0 = cfl_dt0(velqr, dt)
        err = jnp.where(err_dt != 0, err_dt, jnp.where(dt0 < 1.0e-12, 2, 0)).astype(jnp.int32)

        # ---- CFL sub-cycling (Fortran do-while) ----
        def cond_fn(s):
            return (jnp.abs(dt - s[6]) > 1.0e-5) & (err == 0)

        def body_fn(s):
            th, qv_, qc_, qr_, velqr_, acc, tc, d0 = s
            acc = acc + rho_c[0] * qr_[0] * velqr_[0] / rhoqr * d0

            flux = r * qr_ * velqr_
            sed = jnp.concatenate([
                d0 * (flux[1:] - flux[:-1]) / (r[:-1] * dz),
                (-d0 * qr_[-1] * velqr_[-1] / dz_top)[None],
            ])

            # Autoconversion + accretion (semi-implicit), KW78 2.13a,b
            auto = P.autoconv_rate * C.relu(qc_ - P.autoconv_threshold)
            qrprod = qc_ - (qc_ - d0 * auto) / (
                1.0 + d0 * P.accretion_coeff * safe_pow(qr_, P.accretion_exp))
            qc_ = C.relu(qc_ - qrprod)
            qr_ = C.relu(qr_ + qrprod + sed)

            # Saturation mixing ratio (Tetens) and condensation rate, DK83 A13-A14
            T = pk_c * th
            qvs = pc * jnp.exp(P.tetens_a * (T - P.tetens_t0) / (T - P.tetens_t1))
            prod = (qv_ - qvs) / (1.0 + qvs * f5 / (T - P.tetens_t1) ** 2)

            # Rain evaporation, KW78 2.14 / DK83 A8-A9
            rq = r * qr_
            evap_rate = ((P.vent_a + P.vent_b * safe_pow(rq, P.vent_exp))
                         * safe_pow(rq, P.evap_exp)) / (
                P.evap_denom_c * pc / (P.tetens_e0 * qvs) + P.evap_denom_d)
            evap = d0 * evap_rate * (C.relu(qvs - qv_) / (r * qvs))
            ern = C.min(C.min(evap, C.relu(-prod - qc_)), qr_)

            # Saturation adjustment, DK83 A1-A4
            cond = C.max(prod, -qc_)
            th = th + lv / (cp_c * pk_c) * (cond - ern)
            qv_ = C.relu(qv_ - cond + ern)
            qc_ = qc_ + cond
            qr_ = C.relu(qr_ - ern)

            tc = tc + d0
            velqr_ = fallspeed(qr_)
            d0 = cfl_dt0(velqr_, jnp.maximum(dt - tc, 0.0))
            return th, qv_, qc_, qr_, velqr_, acc, tc, d0

        zero = jnp.asarray(0.0, f64)
        init = (th_c, qv_c, qc_c, qr_c, velqr, zero, zero, dt0)
        final, n_sub, exhausted = bounded_while(cond_fn, body_fn, init, n_max=n_max, loop=loop)
        th, qv_, qc_, qr_, _, acc, _, _ = final

        ok = err == 0
        precl = jnp.where(ok, acc / jnp.where(ok, dt, 1.0), 0.0)
        T = pk_c * th
        qvs = pc * jnp.exp(P.tetens_a * (T - P.tetens_t0) / (T - P.tetens_t1))
        relhum = jnp.where(ok, qv_ / qvs * 100.0, 0.0)
        err = jnp.where(ok & exhausted, 3, err).astype(jnp.int32)
        return th, qv_, qc_, qr_, precl, relhum, err, n_sub

    th, qv_, qc_, qr_, precl, relhum, errs, n_sub = jax.vmap(
        column, in_axes=(1,) * 9, out_axes=(1, 1, 1, 1, 0, 1, 0, 0))(*fields)
    if flip:
        th, qv_, qc_, qr_, relhum = (a[::-1] for a in (th, qv_, qc_, qr_, relhum))
    return KesslerOutputs(th, qv_, qc_, qr_, precl, relhum, jnp.max(errs), n_sub)

