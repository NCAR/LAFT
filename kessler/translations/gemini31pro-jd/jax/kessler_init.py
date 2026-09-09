# ---------------------------------------------------------------------------
# Translated by: Gemini 3.1 Pro (via Gemini CLI code agent)
# Pass: final
# ---------------------------------------------------------------------------
"""
SCALAR-ONLY PROCEDURE
Plain Python, No JAX.
Translated by Gemini 3.1 PRO
"""

def kessler_init(lv_in, pref_in, rhoqr_in, errmsg, errflg, lv, pref, rhoqr):
    """
    Python translation of the kessler_init Fortran procedure.
    
    MODULE VARIABLES (from kessler):
        lv: MODULE variable (INOUT)
        pref: MODULE variable (INOUT)
        rhoqr: MODULE variable (INOUT)
    """
    lv = float(lv_in)
    pref = float(pref_in) / 100.0
    rhoqr = float(rhoqr_in)
    
    errmsg = ""
    errflg = 0
    
    return errmsg, errflg, lv, pref, rhoqr