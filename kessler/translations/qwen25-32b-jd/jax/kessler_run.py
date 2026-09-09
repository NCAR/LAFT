# ---------------------------------------------------------------------------
# Translated by: Qwen2.5-32B (Qwen2.5-Coder-32B-Instruct via vLLM)
# Pass: fix-attempt-1 (runtime, 2026-09-04) on final
# ---------------------------------------------------------------------------
# JIT boundary check: PASSED
# Translated by: Qwen2.5-32B
# Fixed by: Claude Opus 4.7 (claude-opus-4-7) — fix attempt 1: subcycle time_counter ordering
import os
os.environ["JAX_ENABLE_X64"] = "1"

import functools
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

@functools.partial(jax.jit, static_argnames=('ncol', 'nz', 'lyr_surf', 'lyr_toa'))  # CHANGED: added lyr_surf, lyr_toa — needed as static for Python if/arange on loop direction
def kessler_run_core(ncol, nz, dt, lyr_surf, lyr_toa, cpair, rair, rho, z, pk, theta, qv, qc, qr, precl, relhum, errflg, lv, pref, rhoqr):  # CHANGED: removed scheme_name — string not allowed in JIT core
    lyr_surf_idx = lyr_surf - 1  # [STATIC-INT]
    lyr_toa_idx = lyr_toa - 1  # [STATIC-INT]

    lyr_step = 1 if lyr_surf <= lyr_toa else -1  # [STATIC-INT]  # CHANGED: Replaced Python if with static int assignment

    f2x = 17.27
    precl = jnp.zeros(ncol, dtype=jnp.float64)  # CHANGED: Added dtype
    errflg = jnp.asarray(0, dtype=jnp.int32)  # CHANGED: Added dtype

    def column_loop(col, state):
        theta, qv, qc, qr, precl, relhum, errflg, r, rhalf, velqr, sed, pc = state

        # Vectorized level loop
        klevs = jnp.arange(lyr_surf_idx, lyr_toa_idx + lyr_step, lyr_step)  # [JAX-VEC]
        f5 = 4093.0 * lv / cpair[klevs, col]
        xk = cpair[klevs, col] / rair[klevs, col]

        r = r.at[klevs].set(0.001 * rho[klevs, col])  # [JAX-VEC]
        rhalf = rhalf.at[klevs].set(jnp.sqrt(rho[lyr_surf_idx, col] / rho[klevs, col]))  # [JAX-VEC]
        pc = pc.at[klevs].set(3.8 / ((pk[klevs, col] ** xk) * pref))  # [JAX-VEC]

        qr = qr.at[klevs, col].set(jnp.maximum(qr[klevs, col], 0.0))  # [JAX-VEC]
        velqr = velqr.at[klevs].set(36.34 * rhalf[klevs] * (qr[klevs, col] * r[klevs]) ** 0.1364)  # [JAX-VEC]

        # Vectorized dt0 loop — CHANGED: reduce to scalar with jnp.min to avoid vector time_counter
        dt0 = dt
        klevs = jnp.arange(lyr_surf_idx, lyr_toa_idx, lyr_step)  # [JAX-VEC]
        if lyr_surf_idx != lyr_toa_idx:  # [PY-IF] CHANGED (fix-attempt-1, 2026-09-04): static one-level guard — the Fortran CFL loop is empty then (dt0 unchanged); jnp.min over the empty interior range has no identity (runtime harness DUMMY_N=4 → lyr_surf = lyr_toa)
            dt0 = jnp.min(jnp.where(jnp.abs(velqr[klevs]) > 1.0E-12, jnp.minimum(dt0, 0.8 * (z[klevs + lyr_step, col] - z[klevs, col]) / velqr[klevs]), dt0))  # [JAX-WHERE] CHANGED: jnp.min() to scalar

        errflg = jnp.where(dt0 < 1.0E-12, jnp.asarray(1, dtype=jnp.int32), errflg)  # [JAX-WHERE]

        time_counter = 0.0
        precl_acc = 0.0

        def subcycle_loop(state):
            theta, qv, qc, qr, precl, relhum, r, rhalf, velqr, sed, pc, precl_acc, dt0, time_counter = state  # CHANGED: added dt0 to state

            precl = precl.at[col].set(rho[lyr_surf_idx, col] * qr[lyr_surf_idx, col] * velqr[lyr_surf_idx] / rhoqr)  # [JAX-VEC]
            precl_acc += precl[col] * dt0  # [JAX]

            # Vectorized sed loop
            klevs = jnp.arange(lyr_surf_idx, lyr_toa_idx, lyr_step)  # [JAX-VEC]
            sed = sed.at[klevs].set(dt0 * ((r[klevs + lyr_step] * qr[klevs + lyr_step, col] * velqr[klevs + lyr_step]) - (r[klevs] * qr[klevs, col] * velqr[klevs])) / (r[klevs] * (z[klevs + lyr_step, col] - z[klevs, col])))  # [JAX-VEC]
            sed = sed.at[lyr_toa_idx].set(-dt0 * qr[lyr_toa_idx, col] * velqr[lyr_toa_idx] / (0.5 * (z[lyr_toa_idx, col] - z[lyr_toa_idx - lyr_step, col])))  # [JAX-VEC]

            # Vectorized adjustment loop
            klevs = jnp.arange(lyr_surf_idx, lyr_toa_idx + lyr_step, lyr_step)  # [JAX-VEC]
            qrprod = qc[klevs, col] - (qc[klevs, col] - dt0 * jnp.maximum(0.001 * (qc[klevs, col] - 0.001), 0.0)) / (1.0 + dt0 * 2.2 * qr[klevs, col] ** 0.875)  # [JAX-VEC]
            qc = qc.at[klevs, col].set(jnp.maximum(qc[klevs, col] - qrprod, 0.0))  # [JAX-VEC]
            qr = qr.at[klevs, col].set(jnp.maximum(qr[klevs, col] + qrprod + sed[klevs], 0.0))  # [JAX-VEC]

            qvs = pc[klevs] * jnp.exp(f2x * (pk[klevs, col] * theta[klevs, col] - 273.0) / (pk[klevs, col] * theta[klevs, col] - 36.0))  # [JAX-VEC]
            prod = (qv[klevs, col] - qvs) / (1.0 + qvs * f5 / (pk[klevs, col] * theta[klevs, col] - 36.0) ** 2)  # [JAX-VEC]
            ern = jnp.minimum(jnp.minimum(dt0 * (((1.6 + 124.9 * (r[klevs] * qr[klevs, col]) ** 0.2046) * (r[klevs] * qr[klevs, col]) ** 0.525) / (2550000.0 * pc[klevs] / (3.8 * qvs) + 540000.0)) * (jnp.maximum(qvs - qv[klevs, col], 0.0) / (r[klevs] * qvs)), jnp.maximum(-prod - qc[klevs, col], 0.0)), qr[klevs, col])  # [JAX-VEC] CHANGED: nested jnp.minimum — jnp.minimum is binary, not variadic like Fortran min()

            theta = theta.at[klevs, col].set(theta[klevs, col] + (lv / (cpair[klevs, col] * pk[klevs, col])) * (jnp.maximum(prod, -qc[klevs, col]) - ern))  # [JAX-VEC]
            qv = qv.at[klevs, col].set(jnp.maximum(qv[klevs, col] - jnp.maximum(prod, -qc[klevs, col]) + ern, 0.0))  # [JAX-VEC]
            qc = qc.at[klevs, col].set(qc[klevs, col] + jnp.maximum(prod, -qc[klevs, col]))  # [JAX-VEC]
            qr = qr.at[klevs, col].set(jnp.maximum(qr[klevs, col] - ern, 0.0))  # [JAX-VEC]

            # Vectorized velqr loop
            klevs = jnp.arange(lyr_surf_idx, lyr_toa_idx + lyr_step, lyr_step)  # [JAX-VEC]
            velqr = velqr.at[klevs].set(36.34 * rhalf[klevs] * (qr[klevs, col] * r[klevs]) ** 0.1364)  # [JAX-VEC]

            time_counter += dt0  # [JAX] FIX(claude-opus-4-7, attempt 1): accumulate the dt0 actually USED for this iteration's physics BEFORE recomputing dt0, matching Fortran kessler.F90 lines 316-350. Previously time_counter was incremented by the NEXT dt0, over-integrating the subcycle by ~one initial CFL substep.
            dt0 = jnp.maximum(dt - time_counter, 0.0)  # [JAX]
            klevs = jnp.arange(lyr_surf_idx, lyr_toa_idx, lyr_step)  # [JAX-VEC]
            if lyr_surf_idx != lyr_toa_idx:  # [PY-IF] CHANGED (fix-attempt-1, 2026-09-04): static one-level guard — the Fortran CFL loop is empty then (dt0 unchanged); jnp.min over the empty interior range has no identity (runtime harness DUMMY_N=4 → lyr_surf = lyr_toa)
                dt0 = jnp.min(jnp.where(jnp.abs(velqr[klevs]) > 1.0E-12, jnp.minimum(dt0, 0.8 * (z[klevs + lyr_step, col] - z[klevs, col]) / velqr[klevs]), dt0))  # [JAX-WHERE] CHANGED: jnp.min() to scalar

            return theta, qv, qc, qr, precl, relhum, r, rhalf, velqr, sed, pc, precl_acc, dt0, time_counter  # CHANGED: added dt0 to return

        theta, qv, qc, qr, precl, relhum, r, rhalf, velqr, sed, pc, precl_acc, dt0, time_counter = lax.while_loop(lambda state: jnp.abs(dt - state[-1]) > 1.0E-5, subcycle_loop, (theta, qv, qc, qr, precl, relhum, r, rhalf, velqr, sed, pc, precl_acc, dt0, time_counter))  # [JAX-WHILE] CHANGED: dt0 in state

        precl = precl.at[col].set(precl_acc / dt)  # [JAX-VEC]

        # Vectorized relhum loop
        klevs = jnp.arange(lyr_surf_idx, lyr_toa_idx + lyr_step, lyr_step)  # [JAX-VEC]
        qvs = pc[klevs] * jnp.exp(f2x * (pk[klevs, col] * theta[klevs, col] - 273.0) / (pk[klevs, col] * theta[klevs, col] - 36.0))  # [JAX-VEC]
        relhum = relhum.at[klevs, col].set(qv[klevs, col] / qvs * 100.0)  # [JAX-VEC]

        return theta, qv, qc, qr, precl, relhum, errflg, r, rhalf, velqr, sed, pc

    r = jnp.zeros(nz, dtype=jnp.float64)  # CHANGED: Added dtype
    rhalf = jnp.zeros(nz, dtype=jnp.float64)  # CHANGED: Added dtype
    velqr = jnp.zeros(nz, dtype=jnp.float64)  # CHANGED: Added dtype
    sed = jnp.zeros(nz, dtype=jnp.float64)  # CHANGED: Added dtype
    pc = jnp.zeros(nz, dtype=jnp.float64)  # CHANGED: Added dtype

    theta, qv, qc, qr, precl, relhum, errflg, r, rhalf, velqr, sed, pc = lax.fori_loop(0, ncol, column_loop, (theta, qv, qc, qr, precl, relhum, errflg, r, rhalf, velqr, sed, pc))  # [JAX-FORI]

    return theta, qv, qc, qr, precl, relhum, errflg, lv, pref, rhoqr  # CHANGED: removed scheme_name from return

def kessler_run(ncol, nz, dt, lyr_surf, lyr_toa, cpair, rair, rho, z, pk, theta, qv, qc, qr, precl, relhum, scheme_name, errmsg, errflg, lv, pref, rhoqr):
    # CHANGED: removed scheme_name from call and unpacking
    theta, qv, qc, qr, precl, relhum, errflg, lv, pref, rhoqr = kessler_run_core(ncol, nz, dt, lyr_surf, lyr_toa, cpair, rair, rho, z, pk, theta, qv, qc, qr, precl, relhum, errflg, lv, pref, rhoqr)
    scheme_name = "KESSLER"  # CHANGED: set scheme_name in wrapper, not core
    if int(errflg) != 0:  # [PY-IF]
        errmsg = 'KESSLER called with nonpositive dt' if dt <= 0.0 else 'KESSLER: bad time splitting {},{}'.format(dt, precl[0])
        return theta, qv, qc, qr, precl, relhum, scheme_name, errmsg, errflg, lv, pref, rhoqr
    return theta, qv, qc, qr, precl, relhum, scheme_name, errmsg, errflg, lv, pref, rhoqr
