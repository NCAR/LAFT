"""
Bridge: kessler_init
Strategy: PATH C device-side layout conversion — pure H2D/D2H,
in-jit axis reversal fused by XLA; public contract unchanged

MODULE: kessler
MODULE vars: lv, pref, rhoqr
Procedure type: INIT

MODULE_VARIABLE_POLICY (INOUT Pattern):
- MODULE vars passed as function parameters (INOUT)
- Driver manages MODULE state explicitly
- No hidden state dictionary
- INIT: Returns updated MODULE vars to driver
- COMPUTE: Returns updated MODULE vars to driver
"""

# No array conversions needed (all parameters are scalars)

from out.jax.kessler_init import kessler_init

def kessler_init_bridge(lv_in, pref_in, rhoqr_in, errmsg, errflg, lv, pref, rhoqr):
    """
    Production bridge for kessler_init (scalar-only).

    No array arguments — no layout conversion exists; calls the
    translated wrapper directly (strings and Python control flow
    stay host-side).

    MODULE VARIABLES (from kessler):
      - lv (INOUT)
      - pref (INOUT)
      - rhoqr (INOUT)
    
    Procedure type: INIT
    Pattern: MODULE vars passed as INOUT parameters
    """
    errmsg_out, errflg_out, lv, pref, rhoqr = kessler_init(lv_in, pref_in, rhoqr_in, errmsg, errflg, lv, pref, rhoqr)

    return errmsg_out, errflg_out, lv, pref, rhoqr
