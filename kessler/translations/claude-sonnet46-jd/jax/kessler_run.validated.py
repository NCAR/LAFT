# ---------------------------------------------------------------------------
# Translated by: Claude Sonnet 4.6 (claude-sonnet-4-6)
# Pass: fix-attempt-1 (bridge suite, 2026-09-04) on final
# ---------------------------------------------------------------------------
import os
# JIT boundary check: FIXED — removed scheme_name/errmsg from core signature; added inline annotation tags  # CHANGED: updated self-check status for pass 4
os.environ["JAX_ENABLE_X64"] = "1"

import functools
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

"""
JAX translation of the Fortran procedure kessler_run (Pass 1).

Bridge mode:
- 2D Fortran arrays declared (ncol, nz) arrive as JAX arrays with shape (nz, ncol).
- Fortran a(col, klev) corresponds to a[klev-1, col-1], or a_col[klev-1] inside vmap.
- MODULE variables (lv, pref, rhoqr) are explicit INOUT parameters.

Translated by Claude Sonnet 4.6
"""


# CHANGED: removed scheme_name and errmsg from static_argnames — strings must not appear in core at all
@functools.partial(jax.jit, static_argnames=("ncol", "nz", "lyr_surf", "lyr_toa"))
def kessler_run_core(
    ncol, nz, dt, lyr_surf, lyr_toa,
    cpair, rair, rho, z, pk,
    theta, qv, qc, qr, precl, relhum,
    errflg,  # CHANGED: removed scheme_name, errmsg — strings belong in wrapper only (self-check: core params rule)
    lv, pref, rhoqr,
):
    """
    Pure JAX compute kernel for kessler_run.

    MODULE VARIABLES (from kessler):
        lv: MODULE variable (INOUT)
        pref: MODULE variable (INOUT)
        rhoqr: MODULE variable (INOUT)

    All 2D arrays have shape (nz, ncol) — bridge has already transposed from Fortran layout.
    """
    dt    = jnp.asarray(dt,    dtype=jnp.float64)  # [JAX]
    lv    = jnp.asarray(lv,    dtype=jnp.float64)  # [JAX]
    pref  = jnp.asarray(pref,  dtype=jnp.float64)  # [JAX]
    rhoqr = jnp.asarray(rhoqr, dtype=jnp.float64)  # [JAX]
    cpair = jnp.asarray(cpair, dtype=jnp.float64)  # [JAX]
    rair  = jnp.asarray(rair,  dtype=jnp.float64)  # [JAX]
    rho   = jnp.asarray(rho,   dtype=jnp.float64)  # [JAX]
    z     = jnp.asarray(z,     dtype=jnp.float64)  # [JAX]
    pk    = jnp.asarray(pk,    dtype=jnp.float64)  # [JAX]
    theta = jnp.asarray(theta, dtype=jnp.float64)  # [JAX]
    qv    = jnp.asarray(qv,    dtype=jnp.float64)  # [JAX]
    qc    = jnp.asarray(qc,    dtype=jnp.float64)  # [JAX]
    qr    = jnp.asarray(qr,    dtype=jnp.float64)  # [JAX]

    # Static index setup (lyr_surf / lyr_toa are static argnames)
    lyr_surf_idx = lyr_surf - 1   # [STATIC-INT] 1-based Fortran → 0-based Python
    lyr_toa_idx  = lyr_toa  - 1   # [STATIC-INT]
    lyr_step     = 1 if lyr_surf <= lyr_toa else -1  # [STATIC-INT]

    if lyr_step == 1:  # [PY-IF] safe: lyr_step is [STATIC-INT]
        level_indices = jnp.arange(lyr_surf_idx, lyr_toa_idx + 1, dtype=jnp.int32)  # [JAX]
    else:
        level_indices = jnp.arange(lyr_surf_idx, lyr_toa_idx - 1, -1, dtype=jnp.int32)  # [JAX]

    interior_indices = level_indices[:-1]           # [JAX] all active levels except lyr_toa
    next_indices     = interior_indices + lyr_step   # [JAX] next level in traversal direction

    f2x = 17.27  # [PY] constant for saturation mixing ratio

    # Error check: dt must be positive
    errflg = jnp.where(  # [JAX-WHERE]
        dt <= 0.0,
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )

    # -----------------------------------------------------------------------
    # Per-column kernel (vectorized over columns via jax.vmap)
    # Inputs are 1D column slices of shape (nz,)
    # -----------------------------------------------------------------------
    def process_column(theta_c, qv_c, qc_c, qr_c, cpair_c, rair_c, rho_c, z_c, pk_c):  # [JAX-VMAP]

        # Level-constant arrays (computed once per column)
        xk    = cpair_c[level_indices] / rair_c[level_indices]             # [JAX-VEC] 1/kappa = cp/R
        r     = jnp.zeros(nz, dtype=jnp.float64).at[level_indices].set(   # [JAX-VEC]
            0.001 * rho_c[level_indices]                                    # density in g/cm^3
        )
        rhalf = jnp.zeros(nz, dtype=jnp.float64).at[level_indices].set(   # [JAX-VEC]
            jnp.sqrt(rho_c[lyr_surf_idx] / rho_c[level_indices])
        )
        pc    = jnp.zeros(nz, dtype=jnp.float64).at[level_indices].set(   # [JAX-VEC]
            3.8 / ((pk_c[level_indices] ** xk) * pref)                     # 3.8 hPa / pressure
        )

        # Clamp qr to zero (avoid floating-point exception in velqr exponent)
        qr_c  = qr_c.at[level_indices].set(jnp.maximum(qr_c[level_indices], 0.0))  # [JAX-VEC]

        # Initial terminal fall speed (KW78 Eq 2.15)
        velqr = jnp.zeros(nz, dtype=jnp.float64).at[level_indices].set(   # [JAX-VEC]
            36.34 * rhalf[level_indices]
            * (qr_c[level_indices] * r[level_indices]) ** 0.1364
        )

        # CFL-limited initial time step
        dz_int    = z_c[next_indices] - z_c[interior_indices]  # [JAX-VEC]
        cfl_cands = jnp.where(  # [JAX-WHERE]
            jnp.abs(velqr[interior_indices]) > 1.0e-12,
            0.8 * dz_int / velqr[interior_indices],
            dt,
        )
        dt0 = jnp.min(jnp.concatenate([jnp.asarray([dt], dtype=jnp.float64), cfl_cands]))  # [JAX]

        col_errflg = jnp.where(  # [JAX-WHERE]  # CHANGED (fix-attempt-1, 2026-09-04): dt<=0 (errflg=1) takes precedence — Fortran returns at the dt check before the dt0 check
            errflg != jnp.asarray(0, dtype=jnp.int32),
            errflg,
            jnp.where(  # [JAX-WHERE]
                dt0 < 1.0e-12,
                jnp.asarray(2, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            ),
        )

        # -------------------------------------------------------------------
        # Subcycling while loop
        # State: (theta_c, qv_c, qc_c, qr_c, velqr_, precl_acc, time_counter, dt0)
        # -------------------------------------------------------------------
        def cond_fn(state):  # [JAX-WHILE]
            _, _, _, _, _, _, time_counter, _ = state
            return jnp.abs(dt - time_counter) > 1.0e-5  # [JAX]

        def body_fn(state):  # [JAX-WHILE]
            theta_c, qv_c, qc_c, qr_c, velqr_, precl_acc, time_counter, dt0_ = state

            # Precipitation rate at surface (m_water/s)
            precl_now = rho_c[lyr_surf_idx] * qr_c[lyr_surf_idx] * velqr_[lyr_surf_idx] / rhoqr  # [JAX]
            precl_acc = precl_acc + precl_now * dt0_  # [JAX]

            # Mass-weighted sedimentation (upstream differencing)
            sed_int = (  # [JAX-VEC]
                dt0_ * (
                    r[next_indices] * qr_c[next_indices] * velqr_[next_indices]
                    - r[interior_indices] * qr_c[interior_indices] * velqr_[interior_indices]
                ) / (r[interior_indices] * dz_int)
            )
            top_dz  = 0.5 * (z_c[lyr_toa_idx] - z_c[lyr_toa_idx - lyr_step])  # [JAX]
            sed_top = -dt0_ * qr_c[lyr_toa_idx] * velqr_[lyr_toa_idx] / top_dz  # [JAX]
            sed = (  # [JAX-VEC]
                jnp.zeros(nz, dtype=jnp.float64)
                .at[interior_indices].set(sed_int)
                .at[lyr_toa_idx].set(sed_top)
            )

            # Gather active levels
            qc_a    = qc_c[level_indices]     # [JAX-VEC]
            qr_a    = qr_c[level_indices]     # [JAX-VEC]
            theta_a = theta_c[level_indices]  # [JAX-VEC]
            qv_a    = qv_c[level_indices]     # [JAX-VEC]
            pk_a    = pk_c[level_indices]     # [JAX-VEC]
            cp_a    = cpair_c[level_indices]  # [JAX-VEC]
            r_a     = r[level_indices]        # [JAX-VEC]
            pc_a    = pc[level_indices]       # [JAX-VEC]
            sed_a   = sed[level_indices]      # [JAX-VEC]

            # Autoconversion and collection (KW78 Eq 2.13a,b)
            qrprod = qc_a - (  # [JAX-VEC]
                qc_a - dt0_ * jnp.maximum(0.001 * (qc_a - 0.001), 0.0)
            ) / (1.0 + dt0_ * 2.2 * qr_a ** 0.875)
            qc_new = jnp.maximum(qc_a - qrprod, 0.0)  # [JAX-VEC]
            qr_new = jnp.maximum(qr_a + qrprod + sed_a, 0.0)  # [JAX-VEC]

            # Saturation vapor mixing ratio: Teten's formula (KW78 Eq 2.11)
            t_abs = pk_a * theta_a  # [JAX-VEC]
            qvs   = pc_a * jnp.exp(f2x * (t_abs - 273.0) / (t_abs - 36.0))  # [JAX-VEC]

            # Condensation rate (DK83 A13-A14)
            f5   = 4093.0 * lv / cp_a  # [JAX-VEC]
            prod = (qv_a - qvs) / (1.0 + qvs * f5 / (t_abs - 36.0) ** 2.0)  # [JAX-VEC]

            # Evaporation rate (KW78 Eq 2.14, DK83 A8-A9)
            # DIM(qvs, qv) = max(qvs - qv, 0)
            evap = (  # [JAX-VEC]
                dt0_
                * ((1.6 + 124.9 * (r_a * qr_new) ** 0.2046) * (r_a * qr_new) ** 0.525)
                / (2550000.0 * pc_a / (3.8 * qvs) + 540000.0)
                * jnp.maximum(qvs - qv_a, 0.0) / (r_a * qvs)
            )
            ern = jnp.minimum(evap, jnp.minimum(jnp.maximum(-prod - qc_new, 0.0), qr_new))  # [JAX-VEC]

            # Saturation adjustment (DK83 A1-A4, KW78 Eq 3.10)
            cond    = jnp.maximum(prod, -qc_new)  # [JAX-VEC]
            theta_n = theta_a + (lv / (cp_a * pk_a)) * (cond - ern)  # [JAX-VEC]
            qv_n    = jnp.maximum(qv_a - cond + ern, 0.0)  # [JAX-VEC]
            qc_n    = qc_new + cond  # [JAX-VEC]
            qr_n    = jnp.maximum(qr_new - ern, 0.0)  # [JAX-VEC]

            theta_c = theta_c.at[level_indices].set(theta_n)  # [JAX-VEC]
            qv_c    = qv_c.at[level_indices].set(qv_n)        # [JAX-VEC]
            qc_c    = qc_c.at[level_indices].set(qc_n)        # [JAX-VEC]
            qr_c    = qr_c.at[level_indices].set(qr_n)        # [JAX-VEC]

            time_counter = time_counter + dt0_  # [JAX]

            # Recalculate terminal velocity
            velqr_ = velqr_.at[level_indices].set(  # [JAX-VEC]
                36.34 * rhalf[level_indices]
                * (qr_c[level_indices] * r[level_indices]) ** 0.1364
            )

            # Recompute CFL-limited dt0
            dt0_new = jnp.maximum(dt - time_counter, 0.0)  # [JAX]
            cfl_new = jnp.where(  # [JAX-WHERE]
                jnp.abs(velqr_[interior_indices]) > 1.0e-12,
                0.8 * dz_int / velqr_[interior_indices],
                dt0_new,
            )
            dt0_new = jnp.min(jnp.concatenate([jnp.asarray([dt0_new], dtype=jnp.float64), cfl_new]))  # [JAX]

            return theta_c, qv_c, qc_c, qr_c, velqr_, precl_acc, time_counter, dt0_new

        # -------------------------------------------------------------------
        # lax.cond: skip if errflg != 0, otherwise run subcycling
        # -------------------------------------------------------------------
        def run_subcycle(op):  # [JAX-COND]
            theta_c, qv_c, qc_c, qr_c, velqr_, dt0_ = op
            init = (
                theta_c, qv_c, qc_c, qr_c, velqr_,
                jnp.asarray(0.0, dtype=jnp.float64),   # precl_acc
                jnp.asarray(0.0, dtype=jnp.float64),   # time_counter
                dt0_,
            )
            theta_c, qv_c, qc_c, qr_c, velqr_, precl_acc, _, _ = lax.while_loop(  # [JAX-WHILE]
                cond_fn, body_fn, init
            )
            precl_c = precl_acc / dt  # [JAX]

            # Diagnostic: relative humidity (percent)
            t_abs    = pk_c[level_indices] * theta_c[level_indices]  # [JAX-VEC]
            qvs      = pc[level_indices] * jnp.exp(f2x * (t_abs - 273.0) / (t_abs - 36.0))  # [JAX-VEC]
            relhum_c = jnp.zeros(nz, dtype=jnp.float64).at[level_indices].set(  # [JAX-VEC]
                qv_c[level_indices] / qvs * 100.0
            )
            return theta_c, qv_c, qc_c, qr_c, precl_c, relhum_c, col_errflg

        def skip_subcycle(op):  # [JAX-COND]
            theta_c, qv_c, qc_c, qr_c, velqr_, dt0_ = op
            return (
                theta_c, qv_c, qc_c, qr_c,
                jnp.asarray(0.0, dtype=jnp.float64),
                jnp.zeros(nz, dtype=jnp.float64),
                col_errflg,
            )

        return lax.cond(  # [JAX-COND]
            col_errflg != jnp.asarray(0, dtype=jnp.int32),
            skip_subcycle,
            run_subcycle,
            (theta_c, qv_c, qc_c, qr_c, velqr, dt0),
        )

    # Vectorize over columns (axis 1 of all 2D arrays)
    theta, qv, qc, qr, precl, relhum, col_errflgs = jax.vmap(  # [JAX-VMAP]
        process_column,
        in_axes=(1, 1, 1, 1, 1, 1, 1, 1, 1),
        out_axes=(1, 1, 1, 1, 0, 1, 0),
    )(theta, qv, qc, qr, cpair, rair, rho, z, pk)

    errflg = jnp.max(col_errflgs)  # [JAX]

    return theta, qv, qc, qr, precl, relhum, errflg, lv, pref, rhoqr


def kessler_run(
    ncol, nz, dt, lyr_surf, lyr_toa,
    cpair, rair, rho, z, pk,
    theta, qv, qc, qr, precl, relhum,
    scheme_name, errmsg, errflg,
    lv, pref, rhoqr,
):
    """
    Python wrapper for kessler_run_core.

    MODULE VARIABLES (from kessler):
        lv: MODULE variable (INOUT)
        pref: MODULE variable (INOUT)
        rhoqr: MODULE variable (INOUT)
    """
    # CHANGED: removed scheme_name, errmsg from call — strings not accepted by core
    theta, qv, qc, qr, precl, relhum, errflg, lv, pref, rhoqr = kessler_run_core(
        ncol, nz, dt, lyr_surf, lyr_toa,
        cpair, rair, rho, z, pk,
        theta, qv, qc, qr, precl, relhum,
        errflg,
        lv, pref, rhoqr,
    )

    scheme_name = "KESSLER"

    if int(errflg) == 1:
        errmsg = "KESSLER called with nonpositive dt"
    elif int(errflg) == 2:
        errmsg = f"KESSLER: bad time splitting {float(dt)}"
    else:
        errmsg = ""

    return theta, qv, qc, qr, precl, relhum, scheme_name, errmsg, errflg, lv, pref, rhoqr
