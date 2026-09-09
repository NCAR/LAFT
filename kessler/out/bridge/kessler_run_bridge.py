"""
Bridge: kessler_run
Strategy: PATH C device-side layout conversion — pure H2D/D2H,
in-jit axis reversal fused by XLA; public contract unchanged

MODULE: kessler
MODULE vars: lv, pref, rhoqr
Procedure type: COMPUTE

MODULE_VARIABLE_POLICY (INOUT Pattern):
- MODULE vars passed as function parameters (INOUT)
- Driver manages MODULE state explicitly
- No hidden state dictionary
- INIT: Returns updated MODULE vars to driver
- COMPUTE: Returns updated MODULE vars to driver
"""

"""
Layout Conversion (PATH C: device-side)

STRATEGY:
- Ship host arrays to the device AS-IS (pure H2D, no host permute)
- Reverse axes INSIDE the jitted device wrapper — XLA fuses the
  permute into the kernel (HBM bandwidth, not host memcpy)
- Pure D2H on the way back: outputs already (ncol, nz) C-order,
  fetched in ONE batched jax.device_get (single stream sync)

Two bridge contracts per array-bearing procedure:
  1. {proc}_bridge        — host-facing: standard C-order NumPy arrays
                            in Fortran (ncol, nz) index order in/out
                            (H2D + jitted device wrapper + ONE batched D2H)
  2. {proc}_bridge_device — device-resident: the jitted wrapper itself;
                            jax.Arrays already on the device in/out, no
                            transfers. For callers that keep the model
                            state on the device across the per-step call
                            sequence (fetch once at the end).
"""

import functools

import numpy as np
import jax
import jax.numpy as jnp

def to_device(arr):
    """Host → device unchanged (pure H2D; no host-side permute)"""
    return jnp.asarray(arr, dtype=jnp.float64)

def to_host(arr):
    """Device → host NumPy for a SINGLE array (blocking D2H).

    The bridge body does NOT use this per-array path: outputs are
    fetched together in one batched jax.device_get (one sync total
    instead of one per array). Kept for tests and interactive use."""
    return np.array(arr, dtype=np.float64)

from out.jax.kessler_run import kessler_run_core

# Device-side layout wrapper (PATH C): arrays arrive in Fortran
# (ncol, nz) index order; axes are reversed INSIDE the jitted region
# so the permute runs on the device and XLA can fuse it into the
# kernel. CHARACTER args never cross this boundary (not JAX types).
# PUBLIC device-resident entry — see its docstring.
@functools.partial(jax.jit, static_argnames=('ncol', 'nz', 'lyr_surf', 'lyr_toa', 'errflg',))
def kessler_run_bridge_device(ncol, nz, dt, lyr_surf, lyr_toa, cpair, rair, rho, z, pk, theta, qv, qc, qr, precl, relhum, errflg, lv, pref, rhoqr):
    """
    DEVICE-RESIDENT bridge entry for kessler_run (contract 2, no transfers).

    Same arguments as kessler_run_bridge minus CHARACTER params, in the
    same order; array arguments are jax.Arrays ALREADY ON THE DEVICE in
    Fortran (ncol, nz) index order (the layout to_device(host) produces);
    integer/logical scalars are static Python values; MODULE vars are
    passed and returned (INOUT pattern). Returns the written non-CHARACTER
    params in argument order, then MODULE vars — as device arrays, left
    on the device (no device_get). Chain these entries across the
    scheme's own per-step phase sequence and fetch once at the end;
    the outputs are bit-identical to kessler_run_bridge's (same jitted code — the
    host bridge is this function wrapped in to_device / device_get).
    """
    # In-jit input reversal: reverse ALL axes on rank>=2
    # (covers rank 3+; rank 0/1 pass through)
    cpair = jnp.transpose(cpair)
    rair = jnp.transpose(rair)
    rho = jnp.transpose(rho)
    z = jnp.transpose(z)
    pk = jnp.transpose(pk)
    theta = jnp.transpose(theta)
    qv = jnp.transpose(qv)
    qc = jnp.transpose(qc)
    qr = jnp.transpose(qr)
    relhum = jnp.transpose(relhum)
    theta, qv, qc, qr, precl, relhum, errflg, lv, pref, rhoqr = kessler_run_core(ncol, nz, dt, lyr_surf, lyr_toa, cpair, rair, rho, z, pk, theta, qv, qc, qr, precl, relhum, errflg, lv, pref, rhoqr)
    # In-jit output reversal: back to Fortran (ncol, nz) order
    theta = jnp.transpose(theta)
    qv = jnp.transpose(qv)
    qc = jnp.transpose(qc)
    qr = jnp.transpose(qr)
    relhum = jnp.transpose(relhum)
    return theta, qv, qc, qr, precl, relhum, errflg, lv, pref, rhoqr

_kessler_run_device = kessler_run_bridge_device  # alias (private name kept for compatibility)

def kessler_run_bridge(ncol, nz, dt, lyr_surf, lyr_toa, cpair, rair, rho, z, pk, theta, qv, qc, qr, precl, relhum, scheme_name, errmsg, errflg, lv, pref, rhoqr):
    """
    Production HOST-FACING bridge for kessler_run (contract 1).

    Device-resident callers use kessler_run_bridge_device instead
    (contract 2: device arrays in/out, no transfers); this function is
    exactly to_device -> kessler_run_bridge_device -> batched device_get.

    Strategy (PATH C): ship arrays to the device unchanged (pure H2D),
    reverse axes inside the jitted device wrapper (fused by XLA), call
    kessler_run_core, reverse back in-jit, pure D2H. Inputs and outputs
    stay standard C-order NumPy in Fortran (ncol, nz) index order.

    CHARACTER args stay host-side: this path calls the jitted core
    directly, so wrapper-level string handling does not run here;
    CHARACTER inputs/outputs pass through the bridge unchanged.

    MODULE VARIABLES (from kessler):
      - lv (INOUT)
      - pref (INOUT)
      - rhoqr (INOUT)
    
    Procedure type: COMPUTE
    Pattern: MODULE vars passed as INOUT parameters
    """
    # Ship arrays to the device unchanged (pure H2D, no host permute)
    cpair_dev = to_device(cpair)
    rair_dev = to_device(rair)
    rho_dev = to_device(rho)
    z_dev = to_device(z)
    pk_dev = to_device(pk)
    theta_dev = to_device(theta)
    qv_dev = to_device(qv)
    qc_dev = to_device(qc)
    qr_dev = to_device(qr)
    precl_dev = to_device(precl)
    relhum_dev = to_device(relhum)

    # Compute — in-jit layout conversion + jitted core
    theta_out, qv_out, qc_out, qr_out, precl_out, relhum_out, errflg_out, lv, pref, rhoqr = kessler_run_bridge_device(ncol=ncol, nz=nz, dt=dt, lyr_surf=lyr_surf, lyr_toa=lyr_toa, cpair=cpair_dev, rair=rair_dev, rho=rho_dev, z=z_dev, pk=pk_dev, theta=theta_dev, qv=qv_dev, qc=qc_dev, qr=qr_dev, precl=precl_dev, relhum=relhum_dev, errflg=errflg, lv=lv, pref=pref, rhoqr=rhoqr)

    # Convert outputs — ONE batched D2H (single sync); layout already
    # Fortran (ncol, nz) C-order
    theta_host, qv_host, qc_host, qr_host, precl_host, relhum_host, errflg_host = jax.device_get((theta_out, qv_out, qc_out, qr_out, precl_out, relhum_out, errflg_out))
    theta_fortran = np.asarray(theta_host, dtype=np.float64)
    qv_fortran = np.asarray(qv_host, dtype=np.float64)
    qc_fortran = np.asarray(qc_host, dtype=np.float64)
    qr_fortran = np.asarray(qr_host, dtype=np.float64)
    precl_fortran = np.asarray(precl_host, dtype=np.float64)
    relhum_fortran = np.asarray(relhum_host, dtype=np.float64)
    errflg_fortran = np.asarray(errflg_host).item()

    return theta_fortran, qv_fortran, qc_fortran, qr_fortran, precl_fortran, relhum_fortran, scheme_name, errmsg, errflg_fortran, lv, pref, rhoqr
