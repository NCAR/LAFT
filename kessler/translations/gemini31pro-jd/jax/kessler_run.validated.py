# ---------------------------------------------------------------------------
# Translated by: Gemini 3.1 Pro (via Gemini CLI code agent)
# Pass: final
# ---------------------------------------------------------------------------
import os
# JIT boundary check: FIXED — Added inline JIT boundary tags and verified all dtypes and static indices.
os.environ["JAX_ENABLE_X64"] = "1"

import functools
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

"""
Python/JAX translation of the Kessler microphysics scheme (kessler_run).
Transled by Gemini 3.1 PRO
"""

# Add static indices that control iteration or shapes
@functools.partial(jax.jit, static_argnames=('ncol', 'nz', 'lyr_surf', 'lyr_toa'))
def kessler_run_core(ncol, nz, dt, lyr_surf, lyr_toa, cpair, rair, rho, z, pk, theta, qv, qc, qr, precl, relhum, errflg, lv, pref, rhoqr):  # CHANGED: removed scheme_name, errmsg — strings not allowed in JIT core
    """
    JAX-safe pure compute core for kessler_run.
    
    MODULE VARIABLES (from kessler):
        lv: MODULE variable (INOUT)
        pref: MODULE variable (INOUT)
        rhoqr: MODULE variable (INOUT)
    """
    # Enforce float64 constants for scalar parameters
    # CHANGED: Added [JAX] inline annotations for scalar operations
    dt_val = jnp.asarray(dt, dtype=jnp.float64)     # [JAX] # CHANGED:
    lv_val = jnp.asarray(lv, dtype=jnp.float64)     # [JAX] # CHANGED:
    pref_val = jnp.asarray(pref, dtype=jnp.float64) # [JAX] # CHANGED:
    rhoqr_val = jnp.asarray(rhoqr, dtype=jnp.float64) # [JAX] # CHANGED:
    
    # Convert static integers using plain Python to avoid tracer infection
    lyr_surf_idx = lyr_surf - 1  # [STATIC-INT]
    lyr_toa_idx = lyr_toa - 1    # [STATIC-INT]
    
    # Calculate static loop direction using Python if/else
    lyr_step = -1 if lyr_surf > lyr_toa else 1  # [STATIC-INT]

    def compute_all_columns(_):
        
        # Define logic for a single column (1D arrays of size nz)
        def process_column(cpair_c, rair_c, rho_c, z_c, pk_c, theta_c, qv_c, qc_c, qr_c):
            # CHANGED: Added inline JAX tags for all local array calculations
            f2x = jnp.asarray(17.27, dtype=jnp.float64)     # [JAX] # CHANGED:
            
            f5 = 4093.0 * lv_val / cpair_c                  # [JAX-VEC] # CHANGED:
            xk = cpair_c / rair_c                           # [JAX-VEC] # CHANGED:
            r = 0.001 * rho_c                               # [JAX-VEC] # CHANGED:
            rhalf = jnp.sqrt(rho_c[lyr_surf_idx] / rho_c)   # [JAX-VEC] # CHANGED:
            pc = 3.8 / ((pk_c ** xk) * pref_val)            # [JAX-VEC] # CHANGED:
            
            qr_c = jnp.maximum(qr_c, 0.0)                   # [JAX-VEC] # CHANGED:
            velqr = 36.34 * rhalf * (qr_c * r) ** 0.1364    # [JAX-VEC] # CHANGED:
            
            # Setup shifting indices for spatial differencing
            k_indices = jnp.arange(nz)                      # [JAX-VEC] # CHANGED:
            k_next = k_indices + lyr_step                   # [JAX-VEC] # CHANGED:
            valid_k = (k_indices != lyr_toa_idx)            # [JAX-VEC] # CHANGED:
            k_next_safe = jnp.clip(k_next, 0, nz - 1)       # [JAX-VEC] # CHANGED:
            dz = z_c[k_next_safe] - z_c                     # [JAX-VEC] # CHANGED:
            
            # Compute dz_toa once outside the subcycle loop to avoid recomputing every iteration
            dz_toa = z_c[lyr_toa_idx] - z_c[lyr_toa_idx - lyr_step]  # [JAX] # CHANGED:
            
            # CHANGED: defined vel_mask here (was NameError — only defined inside subcycle_body before)
            vel_mask = valid_k & (jnp.abs(velqr) > 1.0e-12)        # [JAX-VEC] # CHANGED:
            # Explicit dtype for float constant to prevent downcasting
            safe_dt = jnp.where(                                     # [JAX-WHERE] # CHANGED:
                vel_mask,
                0.8 * dz / jnp.where(vel_mask, velqr, jnp.asarray(1.0, dtype=jnp.float64)),
                dt_val
            )
            dt0 = jnp.min(safe_dt)                                   # [JAX] # CHANGED:
            
            errflg_col = jnp.where(dt0 < 1.0e-12, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32)) # [JAX-WHERE] # CHANGED:
            
            def subcycle_cond(state):
                tc, _, dt0_w, _, _, _, _, _, err_w = state
                return (jnp.abs(dt_val - tc) > 1.0e-5) & (err_w == 0) # [JAX] # CHANGED:
            
            def subcycle_body(state):
                tc, pa, d0, qr_w, qc_w, qv_w, th_w, vq_w, err_w = state
                
                # Precipitation rate
                precl_c = rho_c[lyr_surf_idx] * qr_w[lyr_surf_idx] * vq_w[lyr_surf_idx] / rhoqr_val # [JAX] # CHANGED:
                pa = pa + precl_c * d0                               # [JAX] # CHANGED:
                
                # Sedimentation
                flux = r * qr_w * vq_w                               # [JAX-VEC] # CHANGED:
                flux_next = flux[k_next_safe]                        # [JAX-VEC] # CHANGED:
                sed_main = d0 * (flux_next - flux) / (r * dz)        # [JAX-VEC] # CHANGED:
                
                # Use hoisted dz_toa
                sed_toa = -d0 * qr_w[lyr_toa_idx] * vq_w[lyr_toa_idx] / (0.5 * dz_toa) # [JAX] # CHANGED:
                sed = jnp.where(k_indices == lyr_toa_idx, sed_toa, sed_main)           # [JAX-WHERE] # CHANGED:
                
                # Adjustments
                qrprod = qc_w - (qc_w - d0 * jnp.maximum(0.001 * (qc_w - 0.001), 0.0)) / (1.0 + d0 * 2.2 * (qr_w ** 0.875)) # [JAX-VEC] # CHANGED:
                qc_w = jnp.maximum(qc_w - qrprod, 0.0)               # [JAX-VEC] # CHANGED:
                qr_w = jnp.maximum(qr_w + qrprod + sed, 0.0)         # [JAX-VEC] # CHANGED:
                
                qvs = pc * jnp.exp(f2x * (pk_c * th_w - 273.0) / (pk_c * th_w - 36.0)) # [JAX-VEC] # CHANGED:
                prod = (qv_w - qvs) / (1.0 + qvs * f5 / (pk_c * th_w - 36.0) ** 2)     # [JAX-VEC] # CHANGED:
                
                dim_qvs_qv = jnp.maximum(qvs - qv_w, 0.0)            # [JAX-VEC] # CHANGED:
                term1 = 1.6 + 124.9 * (r * qr_w) ** 0.2046           # [JAX-VEC] # CHANGED:
                term2 = (r * qr_w) ** 0.525                          # [JAX-VEC] # CHANGED:
                term3 = 2550000.0 * pc / (3.8 * qvs) + 540000.0      # [JAX-VEC] # CHANGED:
                term4 = dim_qvs_qv / (r * qvs)                       # [JAX-VEC] # CHANGED:
                
                ern = jnp.minimum(d0 * (term1 * term2 / term3) * term4, # [JAX-VEC] # CHANGED:
                                  jnp.minimum(jnp.maximum(-prod - qc_w, 0.0), qr_w))
                
                th_w = th_w + (lv_val / (cpair_c * pk_c)) * (jnp.maximum(prod, -qc_w) - ern) # [JAX-VEC] # CHANGED:
                qv_w = jnp.maximum(qv_w - jnp.maximum(prod, -qc_w) + ern, 0.0)               # [JAX-VEC] # CHANGED:
                qc_w = qc_w + jnp.maximum(prod, -qc_w)                                       # [JAX-VEC] # CHANGED:
                qr_w = jnp.maximum(qr_w - ern, 0.0)                                          # [JAX-VEC] # CHANGED:
                
                tc = tc + d0                                         # [JAX] # CHANGED:
                
                vq_w = 36.34 * rhalf * (qr_w * r) ** 0.1364          # [JAX-VEC] # CHANGED:
                
                d0_new = jnp.maximum(dt_val - tc, 0.0)               # [JAX] # CHANGED:
                v_mask = valid_k & (jnp.abs(vq_w) > 1.0e-12)         # [JAX-VEC] # CHANGED:
                
                # Ensure float64 constant in jnp.where to prevent downcasting inside while_loop
                safe_d0 = jnp.where(v_mask, 0.8 * dz / jnp.where(v_mask, vq_w, jnp.asarray(1.0, dtype=jnp.float64)), d0_new) # [JAX-WHERE] # CHANGED:
                d0_new = jnp.minimum(d0_new, jnp.min(safe_d0))       # [JAX] # CHANGED:
                
                return (tc, pa, d0_new, qr_w, qc_w, qv_w, th_w, vq_w, err_w)
            
            # CHANGED: Added tags to initialization and while_loop
            init_state = (                                           # [JAX] # CHANGED:
                jnp.asarray(0.0, dtype=jnp.float64), 
                jnp.asarray(0.0, dtype=jnp.float64), 
                dt0, qr_c, qc_c, qv_c, theta_c, velqr, errflg_col
            )
            
            final_state = lax.while_loop(subcycle_cond, subcycle_body, init_state) # [JAX-WHILE] # CHANGED:
            tc, pa, d0, qr_c_out, qc_c_out, qv_c_out, theta_c_out, velqr_out, errflg_col_out = final_state
            
            # Explicit dtype for float constant
            precl_c_out = jnp.where(errflg_col_out == 0, pa / dt_val, jnp.asarray(0.0, dtype=jnp.float64)) # [JAX-WHERE] # CHANGED:
            
            qvs_out = pc * jnp.exp(f2x * (pk_c * theta_c_out - 273.0) / (pk_c * theta_c_out - 36.0))       # [JAX-VEC] # CHANGED:
            relhum_c_out = qv_c_out / qvs_out * 100.0                                                      # [JAX-VEC] # CHANGED:
            
            return theta_c_out, qv_c_out, qc_c_out, qr_c_out, precl_c_out, relhum_c_out, errflg_col_out

        # Vectorize over columns
        # inputs are (nz, ncol) -> axis 1
        # out_axes sets (nz, ncol) for arrays and (ncol,) for scalars
        # CHANGED: Tagged vmap execution
        th_out, qv_out, qc_out, qr_out, pr_out, rh_out, errs = jax.vmap(     # [JAX-VMAP] # CHANGED:
            process_column, 
            in_axes=(1, 1, 1, 1, 1, 1, 1, 1, 1),
            out_axes=(1, 1, 1, 1, 0, 1, 0)
        )(cpair, rair, rho, z, pk, theta, qv, qc, qr)
        
        errflg_out = jnp.max(errs)                                           # [JAX] # CHANGED:
        return th_out, qv_out, qc_out, qr_out, pr_out, rh_out, errflg_out

    def bypass_computation(_):
        # CHANGED: Added tags to the bypass computations
        return (
            theta, qv, qc, qr, 
            jnp.zeros(ncol, dtype=jnp.float64),                              # [JAX] # CHANGED:
            relhum, 
            jnp.asarray(1, dtype=jnp.int32)                                  # [JAX] # CHANGED:
        )

    # Main branch: if dt <= 0, bypass computation entirely
    # CHANGED: Tagged lax.cond and dummy return variables
    th_out, qv_out, qc_out, qr_out, pr_out, rh_out, errflg_out = lax.cond(   # [JAX-COND] # CHANGED:
        dt_val <= 0.0,
        bypass_computation,
        compute_all_columns,
        operand=None
    )
    
    # CHANGED: removed scheme_dummy and errmsg_dummy — strings cannot be returned from JIT core
    return th_out, qv_out, qc_out, qr_out, pr_out, rh_out, errflg_out, lv_val, pref_val, rhoqr_val


def kessler_run(ncol, nz, dt, lyr_surf, lyr_toa, cpair, rair, rho, z, pk, theta, qv, qc, qr, precl, relhum, scheme_name, errmsg, errflg, lv, pref, rhoqr):
    """
    Python wrapper for kessler_run.
    
    MODULE VARIABLES (from kessler):
        lv: MODULE variable (INOUT)
        pref: MODULE variable (INOUT)
        rhoqr: MODULE variable (INOUT)
    """
    # CHANGED: removed scheme_name, errmsg from call; updated unpacking (no longer 12 values)
    th_out, qv_out, qc_out, qr_out, pr_out, rh_out, errflg_out, lv_out, pref_out, rhoqr_out = kessler_run_core(
        ncol, nz, dt, lyr_surf, lyr_toa, cpair, rair, rho, z, pk, theta, qv, qc, qr, precl, relhum, errflg, lv, pref, rhoqr
    )
    
    scheme_name_out = "KESSLER"
    errmsg_out = ""
    
    # Handle error messages on the host
    if int(errflg_out) != 0:
        if float(dt) <= 0.0:
            errmsg_out = "KESSLER called with nonpositive dt"
        else:
            errmsg_out = "KESSLER: bad time splitting"
            
    return th_out, qv_out, qc_out, qr_out, pr_out, rh_out, scheme_name_out, errmsg_out, errflg_out, lv_out, pref_out, rhoqr_out