# ---------------------------------------------------------------------------
# Translated by: GPT-5.4 Thinking
# Pass: fix-attempt-1 (bridge suite, 2026-09-04) on final
# ---------------------------------------------------------------------------
import os
# CHANGED: updated self-check status for pass 4 after dtype/tag review.
# JIT boundary check: FIXED — added inline tags, completed dtype annotations, retained required wrapper/core signatures from prior pass
os.environ["JAX_ENABLE_X64"] = "1"

import functools
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

"""
JAX translation of the Fortran procedure `kessler_run`.

Bridge mode:
- 2D Fortran arrays with shape (ncol, nz) arrive here as JAX arrays with shape (nz, ncol).
- Fortran indexing a(col, klev) becomes JAX indexing a[klev - 1, col - 1].
- MODULE variables (lv, pref, rhoqr) are explicit INOUT parameters.
"""


@functools.partial(jax.jit, static_argnames=("ncol", "nz", "lyr_surf", "lyr_toa"))
def kessler_run_core(
    ncol,
    nz,
    dt,
    lyr_surf,
    lyr_toa,
    cpair,
    rair,
    rho,
    z,
    pk,
    theta,
    qv,
    qc,
    qr,
    precl,
    relhum,
    errflg,
    lv,
    pref,
    rhoqr,
):
    """
    MODULE VARIABLES (from kessler):
        lv: MODULE variable (INOUT)
        pref: MODULE variable (INOUT)
        rhoqr: MODULE variable (INOUT)
    """
    # CHANGED: add inline JAX tags to casts for dtype consistency visibility.
    dt = jnp.asarray(dt, dtype=jnp.float64)  # [JAX]  # CHANGED:
    cpair = jnp.asarray(cpair, dtype=jnp.float64)  # [JAX]  # CHANGED:
    rair = jnp.asarray(rair, dtype=jnp.float64)  # [JAX]  # CHANGED:
    rho = jnp.asarray(rho, dtype=jnp.float64)  # [JAX]  # CHANGED:
    z = jnp.asarray(z, dtype=jnp.float64)  # [JAX]  # CHANGED:
    pk = jnp.asarray(pk, dtype=jnp.float64)  # [JAX]  # CHANGED:
    theta = jnp.asarray(theta, dtype=jnp.float64)  # [JAX]  # CHANGED:
    qv = jnp.asarray(qv, dtype=jnp.float64)  # [JAX]  # CHANGED:
    qc = jnp.asarray(qc, dtype=jnp.float64)  # [JAX]  # CHANGED:
    qr = jnp.asarray(qr, dtype=jnp.float64)  # [JAX]  # CHANGED:
    lv = jnp.asarray(lv, dtype=jnp.float64)  # [JAX]  # CHANGED:
    pref = jnp.asarray(pref, dtype=jnp.float64)  # [JAX]  # CHANGED:
    rhoqr = jnp.asarray(rhoqr, dtype=jnp.float64)  # [JAX]  # CHANGED:

    precl = jnp.zeros((ncol,), dtype=jnp.float64)  # [JAX]  # CHANGED:
    relhum = jnp.zeros((nz, ncol), dtype=jnp.float64)  # [JAX]  # CHANGED:

    lyr_surf_idx = lyr_surf - 1  # [STATIC-INT]
    lyr_toa_idx = lyr_toa - 1  # [STATIC-INT]
    lyr_step = 1 if lyr_surf <= lyr_toa else -1  # [STATIC-INT]
    nlev = abs(lyr_toa_idx - lyr_surf_idx) + 1  # [STATIC-INT]

    # Brief comment above each modified block:
    # CHANGED: annotate static Python branch and vectorized level-index construction.
    if lyr_step == 1:  # [PY-IF]  # CHANGED: static-int branch for ascending level order
        level_indices = jnp.arange(lyr_surf_idx, lyr_toa_idx + 1, dtype=jnp.int32)  # [JAX]  # CHANGED: vectorized active-level indexing
    else:
        level_indices = jnp.arange(lyr_surf_idx, lyr_toa_idx - 1, -1, dtype=jnp.int32)  # [JAX]  # CHANGED: vectorized active-level indexing

    interior_indices = level_indices[:-1]  # [JAX]  # CHANGED: vectorized interior-level indexing for CFL and sedimentation

    f2x = jnp.asarray(17.27, dtype=jnp.float64)  # [JAX]  # CHANGED:
    errflg = jnp.where(  # [JAX-WHERE]  # CHANGED:
        dt <= jnp.asarray(0.0, dtype=jnp.float64),  # CHANGED:
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )

    # Brief comment above each modified block:
    # CHANGED: annotate vmapped single-column kernel and all non-trivial assignments inside it.
    def process_one_col(theta_c, qv_c, qc_c, qr_c, cpair_c, rair_c, rho_c, z_c, pk_c):  # [JAX-VMAP]  # CHANGED: per-column kernel for jax.vmap
        r = jnp.zeros((nz,), dtype=jnp.float64)  # [JAX]  # CHANGED: per-column workspace stays local to vmapped column
        rhalf = jnp.zeros((nz,), dtype=jnp.float64)  # [JAX]  # CHANGED: per-column workspace stays local to vmapped column
        velqr = jnp.zeros((nz,), dtype=jnp.float64)  # [JAX]  # CHANGED: per-column workspace stays local to vmapped column
        pc = jnp.zeros((nz,), dtype=jnp.float64)  # [JAX]  # CHANGED: per-column workspace stays local to vmapped column

        rho_surf_col = rho_c[lyr_surf_idx]  # [JAX]  # CHANGED:

        # Brief comment above each modified block:
        # CHANGED: vectorize independent level initialization instead of looping with lax.fori_loop.
        qr_active = jnp.maximum(qr_c[level_indices], jnp.asarray(0.0, dtype=jnp.float64))  # [JAX-VEC]  # CHANGED: vectorized nonnegative rainwater clamp
        cpair_active = cpair_c[level_indices]  # [JAX-VEC]  # CHANGED: gather active levels once for vectorized formulas
        rair_active = rair_c[level_indices]  # [JAX-VEC]  # CHANGED: gather active levels once for vectorized formulas
        rho_active = rho_c[level_indices]  # [JAX-VEC]  # CHANGED: gather active levels once for vectorized formulas
        pk_active = pk_c[level_indices]  # [JAX-VEC]  # CHANGED: gather active levels once for vectorized formulas

        xk_active = cpair_active / rair_active  # [JAX-VEC]  # CHANGED: vectorized xk over active levels
        r_active = jnp.asarray(0.001, dtype=jnp.float64) * rho_active  # [JAX-VEC]  # CHANGED: vectorized density conversion over active levels
        rhalf_active = jnp.sqrt(rho_surf_col / rho_active)  # [JAX-VEC]  # CHANGED: vectorized rhalf over active levels
        pc_active = jnp.asarray(3.8, dtype=jnp.float64) / ((pk_active**xk_active) * pref)  # [JAX-VEC]  # CHANGED: vectorized pc over active levels
        velqr_active = jnp.asarray(36.34, dtype=jnp.float64) * rhalf_active * ((qr_active * r_active) ** jnp.asarray(0.1364, dtype=jnp.float64))  # [JAX-VEC]  # CHANGED: vectorized terminal velocity over active levels

        qr_c = qr_c.at[level_indices].set(qr_active)  # [JAX-VEC]  # CHANGED: scatter vectorized qr clamp back to full column
        r = r.at[level_indices].set(r_active)  # [JAX-VEC]  # CHANGED: scatter vectorized r back to full workspace
        rhalf = rhalf.at[level_indices].set(rhalf_active)  # [JAX-VEC]  # CHANGED: scatter vectorized rhalf back to full workspace
        pc = pc.at[level_indices].set(pc_active)  # [JAX-VEC]  # CHANGED: scatter vectorized pc back to full workspace
        velqr = velqr.at[level_indices].set(velqr_active)  # [JAX-VEC]  # CHANGED: scatter vectorized velqr back to full workspace

        # Brief comment above each modified block:
        # CHANGED: vectorize CFL candidate computation across interior levels.
        dz_interior = z_c[interior_indices + lyr_step] - z_c[interior_indices]  # [JAX-VEC]  # CHANGED: vectorized dz for CFL calculation
        dt0_cfl = jnp.asarray(0.8, dtype=jnp.float64) * dz_interior / velqr[interior_indices]  # [JAX-VEC]  # CHANGED: vectorized CFL timestep candidates
        dt0_candidates = jnp.where(  # [JAX-WHERE]  # CHANGED: vectorized masking of zero-velocity levels
            jnp.abs(velqr[interior_indices]) > jnp.asarray(1.0e-12, dtype=jnp.float64),
            dt0_cfl,
            dt,
        )
        dt0 = jnp.min(jnp.concatenate((jnp.asarray([dt], dtype=jnp.float64), dt0_candidates)))  # [JAX]  # CHANGED: vectorized minimum over timestep candidates
        col_errflg = jnp.where(  # [JAX-WHERE]  # CHANGED (fix-attempt-1, 2026-09-04): dt<=0 (errflg=1) takes precedence — Fortran returns at the dt check before the dt0 check
            errflg != jnp.asarray(0, dtype=jnp.int32),
            errflg,
            jnp.where(dt0 < jnp.asarray(1.0e-12, dtype=jnp.float64), jnp.asarray(2, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32)),  # [JAX-WHERE]
        )

        def skip_subcycle(args):
            theta_c, qv_c, qc_c, qr_c, precl_c, relhum_c, col_errflg, velqr, dt0 = args
            return theta_c, qv_c, qc_c, qr_c, precl_c, relhum_c, col_errflg

        def run_subcycle(args):
            theta_c, qv_c, qc_c, qr_c, precl_c, relhum_c, col_errflg, velqr, dt0 = args  # velqr and dt0 unpacked here to give them local bindings before lax.while_loop

            time_counter0 = jnp.asarray(0.0, dtype=jnp.float64)  # [JAX]  # CHANGED:
            precl_acc0 = jnp.asarray(0.0, dtype=jnp.float64)  # [JAX]  # CHANGED:

            def while_cond(loop_state):  # [JAX-WHILE]  # CHANGED:
                theta_c, qv_c, qc_c, qr_c, velqr, time_counter, precl_acc, dt0 = loop_state
                return jnp.abs(dt - time_counter) > jnp.asarray(1.0e-5, dtype=jnp.float64)  # [JAX]  # CHANGED:

            def while_body(loop_state):  # [JAX-WHILE]  # CHANGED:
                theta_c, qv_c, qc_c, qr_c, velqr, time_counter, precl_acc, dt0 = loop_state

                precl_c = rho_c[lyr_surf_idx] * qr_c[lyr_surf_idx] * velqr[lyr_surf_idx] / rhoqr  # [JAX]  # CHANGED:
                precl_acc = precl_acc + precl_c * dt0  # [JAX]  # CHANGED:

                sed_local = jnp.zeros((nz,), dtype=jnp.float64)  # [JAX]  # CHANGED:

                # Brief comment above each modified block:
                # CHANGED: vectorize sedimentation interior updates instead of using a level fori_loop.
                flux_up = r[interior_indices + lyr_step] * qr_c[interior_indices + lyr_step] * velqr[interior_indices + lyr_step]  # [JAX-VEC]  # CHANGED: vectorized upstream flux
                flux_dn = r[interior_indices] * qr_c[interior_indices] * velqr[interior_indices]  # [JAX-VEC]  # CHANGED: vectorized downstream flux
                sed_interior = dt0 * (flux_up - flux_dn) / (r[interior_indices] * dz_interior)  # [JAX-VEC]  # CHANGED: vectorized sedimentation tendency
                sed_local = sed_local.at[interior_indices].set(sed_interior)  # [JAX-VEC]  # CHANGED: scatter vectorized sedimentation into workspace

                top_dz = jnp.asarray(0.5, dtype=jnp.float64) * (z_c[lyr_toa_idx] - z_c[lyr_toa_idx - lyr_step])  # [JAX]  # CHANGED:
                sed_top = -dt0 * qr_c[lyr_toa_idx] * velqr[lyr_toa_idx] / top_dz  # [JAX]  # CHANGED:
                sed_local = sed_local.at[lyr_toa_idx].set(sed_top)  # [JAX]  # CHANGED:

                # Brief comment above each modified block:
                # CHANGED: vectorize level-local moisture adjustment; each active level is independent within one subcycle.
                qc_active = qc_c[level_indices]  # [JAX-VEC]  # CHANGED: gather active qc for vectorized adjustment
                qr_active = qr_c[level_indices]  # [JAX-VEC]  # CHANGED: gather active qr for vectorized adjustment
                theta_active = theta_c[level_indices]  # [JAX-VEC]  # CHANGED: gather active theta for vectorized adjustment
                qv_active = qv_c[level_indices]  # [JAX-VEC]  # CHANGED: gather active qv for vectorized adjustment
                pk_active = pk_c[level_indices]  # [JAX-VEC]  # CHANGED: gather active pk for vectorized adjustment
                cpair_active = cpair_c[level_indices]  # [JAX-VEC]  # CHANGED: gather active cpair for vectorized adjustment
                r_active = r[level_indices]  # [JAX-VEC]  # CHANGED: gather active r for vectorized adjustment
                pc_active = pc[level_indices]  # [JAX-VEC]  # CHANGED: gather active pc for vectorized adjustment
                sed_active = sed_local[level_indices]  # [JAX-VEC]  # CHANGED: gather active sedimentation tendency for vectorized adjustment

                qrprod = qc_active - (  # [JAX-VEC]  # CHANGED: vectorized autoconversion/collection tendency
                    (qc_active - dt0 * jnp.maximum(jnp.asarray(0.001, dtype=jnp.float64) * (qc_active - jnp.asarray(0.001, dtype=jnp.float64)), jnp.asarray(0.0, dtype=jnp.float64)))
                    / (jnp.asarray(1.0, dtype=jnp.float64) + dt0 * jnp.asarray(2.2, dtype=jnp.float64) * (qr_active**jnp.asarray(0.875, dtype=jnp.float64)))
                )
                qc_new = jnp.maximum(qc_active - qrprod, jnp.asarray(0.0, dtype=jnp.float64))  # [JAX-VEC]  # CHANGED: vectorized qc update
                qr_mid = jnp.maximum(qr_active + qrprod + sed_active, jnp.asarray(0.0, dtype=jnp.float64))  # [JAX-VEC]  # CHANGED: vectorized qr update after sedimentation

                t_abs = pk_active * theta_active  # [JAX-VEC]  # CHANGED: vectorized absolute temperature-like term
                qvs = pc_active * jnp.exp(f2x * (t_abs - jnp.asarray(273.0, dtype=jnp.float64)) / (t_abs - jnp.asarray(36.0, dtype=jnp.float64)))  # [JAX-VEC]  # CHANGED: vectorized saturation mixing ratio
                f5 = jnp.asarray(4093.0, dtype=jnp.float64) * lv / cpair_active  # [JAX-VEC]  # CHANGED: vectorized condensation-rate coefficient
                prod = (qv_active - qvs) / (jnp.asarray(1.0, dtype=jnp.float64) + qvs * f5 / ((t_abs - jnp.asarray(36.0, dtype=jnp.float64)) ** jnp.asarray(2.0, dtype=jnp.float64)))  # [JAX-VEC]  # CHANGED: vectorized condensation tendency

                evap_term = (  # [JAX-VEC]  # CHANGED: vectorized evaporation-rate candidate
                    dt0
                    * (
                        ((jnp.asarray(1.6, dtype=jnp.float64) + jnp.asarray(124.9, dtype=jnp.float64) * ((r_active * qr_mid) ** jnp.asarray(0.2046, dtype=jnp.float64))) * ((r_active * qr_mid) ** jnp.asarray(0.525, dtype=jnp.float64)))
                        / (jnp.asarray(2550000.0, dtype=jnp.float64) * pc_active / (jnp.asarray(3.8, dtype=jnp.float64) * qvs) + jnp.asarray(540000.0, dtype=jnp.float64))
                    )
                    * (jnp.maximum(qvs - qv_active, jnp.asarray(0.0, dtype=jnp.float64)) / (r_active * qvs))
                )
                ern = jnp.minimum(  # [JAX-VEC]  # CHANGED: vectorized bounded evaporation rate
                    evap_term,
                    jnp.minimum(jnp.maximum(-prod - qc_new, jnp.asarray(0.0, dtype=jnp.float64)), qr_mid),
                )

                cond = jnp.maximum(prod, -qc_new)  # [JAX-VEC]  # CHANGED: vectorized saturation adjustment limiter
                theta_new = theta_active + (lv / (cpair_active * pk_active)) * (cond - ern)  # [JAX-VEC]  # CHANGED: vectorized theta update
                qv_new = jnp.maximum(qv_active - cond + ern, jnp.asarray(0.0, dtype=jnp.float64))  # [JAX-VEC]  # CHANGED: vectorized qv update
                qc_fin = qc_new + cond  # [JAX-VEC]  # CHANGED: vectorized qc final update
                qr_fin = jnp.maximum(qr_mid - ern, jnp.asarray(0.0, dtype=jnp.float64))  # [JAX-VEC]  # CHANGED: vectorized qr final update

                theta_c = theta_c.at[level_indices].set(theta_new)  # [JAX-VEC]  # CHANGED: scatter vectorized theta back to full column
                qv_c = qv_c.at[level_indices].set(qv_new)  # [JAX-VEC]  # CHANGED: scatter vectorized qv back to full column
                qc_c = qc_c.at[level_indices].set(qc_fin)  # [JAX-VEC]  # CHANGED: scatter vectorized qc back to full column
                qr_c = qr_c.at[level_indices].set(qr_fin)  # [JAX-VEC]  # CHANGED: scatter vectorized qr back to full column

                time_counter = time_counter + dt0  # [JAX]  # CHANGED:

                # Brief comment above each modified block:
                # CHANGED: vectorize terminal-velocity recomputation after subcycle updates.
                velqr_active = jnp.asarray(36.34, dtype=jnp.float64) * rhalf[level_indices] * ((qr_c[level_indices] * r[level_indices]) ** jnp.asarray(0.1364, dtype=jnp.float64))  # [JAX-VEC]  # CHANGED: vectorized terminal velocity recomputation
                velqr = velqr.at[level_indices].set(velqr_active)  # [JAX-VEC]  # CHANGED: scatter vectorized velqr back to full workspace

                dt0_new = jnp.maximum(dt - time_counter, jnp.asarray(0.0, dtype=jnp.float64))  # [JAX]  # CHANGED:

                # Brief comment above each modified block:
                # CHANGED: vectorize recomputed CFL timestep candidates across interior levels.
                dt0_cfl_new = jnp.asarray(0.8, dtype=jnp.float64) * dz_interior / velqr[interior_indices]  # [JAX-VEC]  # CHANGED: vectorized recomputed CFL candidates
                dt0_candidates_new = jnp.where(  # [JAX-WHERE]  # CHANGED: vectorized masking of zero-velocity levels during recompute
                    jnp.abs(velqr[interior_indices]) > jnp.asarray(1.0e-12, dtype=jnp.float64),
                    dt0_cfl_new,
                    dt0_new,
                )
                dt0_new = jnp.min(jnp.concatenate((jnp.asarray([dt0_new], dtype=jnp.float64), dt0_candidates_new)))  # [JAX]  # CHANGED: vectorized minimum over recomputed timestep candidates

                return theta_c, qv_c, qc_c, qr_c, velqr, time_counter, precl_acc, dt0_new

            theta_c, qv_c, qc_c, qr_c, velqr, time_counter, precl_acc, dt0 = lax.while_loop(  # [JAX-WHILE]  # CHANGED:
                while_cond,
                while_body,
                (theta_c, qv_c, qc_c, qr_c, velqr, time_counter0, precl_acc0, dt0),
            )

            precl_c = precl_acc / dt  # [JAX]  # CHANGED:

            # Brief comment above each modified block:
            # CHANGED: vectorize the relative-humidity diagnostic over active levels.
            t_abs = pk_c[level_indices] * theta_c[level_indices]  # [JAX-VEC]  # CHANGED: gather active temperature state for vectorized diagnostic
            qvs = pc[level_indices] * jnp.exp(f2x * (t_abs - jnp.asarray(273.0, dtype=jnp.float64)) / (t_abs - jnp.asarray(36.0, dtype=jnp.float64)))  # [JAX-VEC]  # CHANGED: vectorized saturation mixing ratio for diagnostic
            relhum_c = jnp.zeros((nz,), dtype=jnp.float64)  # [JAX]  # CHANGED: initialize per-column diagnostic workspace
            relhum_c = relhum_c.at[level_indices].set(qv_c[level_indices] / qvs * jnp.asarray(100.0, dtype=jnp.float64))  # [JAX-VEC]  # CHANGED: scatter vectorized relative humidity diagnostic

            return theta_c, qv_c, qc_c, qr_c, precl_c, relhum_c, col_errflg

        return lax.cond(  # [JAX-COND]  # CHANGED:
            col_errflg != jnp.asarray(0, dtype=jnp.int32),
            skip_subcycle,
            run_subcycle,
            (
                theta_c,
                qv_c,
                qc_c,
                qr_c,
                jnp.asarray(0.0, dtype=jnp.float64),
                jnp.zeros((nz,), dtype=jnp.float64),
                col_errflg,
                velqr,   # passed so run_subcycle has a local binding before lax.while_loop
                dt0,     # same scoping trap: dt0 also appears on the left of the while_loop unpacking
            ),
        )

    theta, qv, qc, qr, precl, relhum, col_errflg = jax.vmap(  # [JAX-VMAP]  # CHANGED: eliminate the outer column loop with jax.vmap
        process_one_col,
        in_axes=(1, 1, 1, 1, 1, 1, 1, 1, 1),  # CHANGED: map over column axis for all 2-D fields
        out_axes=(1, 1, 1, 1, 0, 1, 0),  # CHANGED: stack scalar precipitation/error on axis 0 and fields on axis 1
    )(theta, qv, qc, qr, cpair, rair, rho, z, pk)
    errflg = jnp.max(col_errflg)  # [JAX]  # CHANGED: combine per-column error flags after vmapped execution

    return theta, qv, qc, qr, precl, relhum, errflg, lv, pref, rhoqr


def kessler_run(
    ncol,
    nz,
    dt,
    lyr_surf,
    lyr_toa,
    cpair,
    rair,
    rho,
    z,
    pk,
    theta,
    qv,
    qc,
    qr,
    precl,
    relhum,
    scheme_name,
    errmsg,
    errflg,
    lv,
    pref,
    rhoqr,
):
    """
    MODULE VARIABLES (from kessler):
        lv: MODULE variable (INOUT)
        pref: MODULE variable (INOUT)
        rhoqr: MODULE variable (INOUT)
    """
    theta, qv, qc, qr, precl, relhum, errflg, lv, pref, rhoqr = kessler_run_core(
        ncol,
        nz,
        dt,
        lyr_surf,
        lyr_toa,
        cpair,
        rair,
        rho,
        z,
        pk,
        theta,
        qv,
        qc,
        qr,
        precl,
        relhum,
        errflg,
        lv,
        pref,
        rhoqr,
    )

    scheme_name = "KESSLER"

    if int(errflg) == 1:
        errmsg = "KESSLER called with nonpositive dt"
    elif int(errflg) == 2:
        errmsg = f"KESSLER: bad time splitting {dt}"
    else:
        errmsg = ""

    return theta, qv, qc, qr, precl, relhum, scheme_name, errmsg, errflg, lv, pref, rhoqr