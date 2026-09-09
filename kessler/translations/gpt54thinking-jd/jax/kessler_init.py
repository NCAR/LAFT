# ---------------------------------------------------------------------------
# Translated by: GPT-5.4 Thinking
# Pass: final
# ---------------------------------------------------------------------------
"""
SCALAR-ONLY translation of the Fortran procedure kessler_init.

Translated by GPT-5.4 Thinking
"""

def kessler_init(lv_in, pref_in, rhoqr_in, errmsg, errflg, lv, pref, rhoqr):
    """
    MODULE VARIABLES (from kessler):
    lv: MODULE variable (INOUT)
    pref: MODULE variable (INOUT)
    rhoqr: MODULE variable (INOUT)
    """
    errmsg = ""
    errflg = 0

    lv = float(lv_in)
    pref = float(pref_in) / 100.0
    rhoqr = float(rhoqr_in)

    return errmsg, errflg, lv, pref, rhoqr
