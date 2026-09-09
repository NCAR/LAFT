# ---------------------------------------------------------------------------
# Translated by: Qwen2.5-32B (Qwen2.5-Coder-32B-Instruct via vLLM)
# Pass: final
# ---------------------------------------------------------------------------
"""
Translated by Qwen2.5-32B
SCALAR-ONLY PROCEDURE
This module contains a single function that operates on scalar values only.
No JAX or NumPy is used for scalar arithmetic.
"""

def kessler_init(lv_in, pref_in, rhoqr_in, errmsg, errflg, lv, pref, rhoqr):
    """
    Initialize physical constants for the Kessler scheme.

    Parameters:
    lv_in (float): Latent heat of vaporization, J/kg
    pref_in (float): Reference pressure, Pa
    rhoqr_in (float): Density of fresh liquid water, kg/m^3
    errmsg (str): Error message (output)
    errflg (int): Error flag (output)
    lv (float): Latent heat of vaporization (inout)
    pref (float): Reference pressure (inout)
    rhoqr (float): Density of fresh liquid water (inout)

    Returns:
    tuple: Updated errmsg, errflg, lv, pref, rhoqr
    """
    errmsg = ''
    errflg = 0

    lv = lv_in
    pref = pref_in / 100.0
    rhoqr = rhoqr_in

    return errmsg, errflg, lv, pref, rhoqr
