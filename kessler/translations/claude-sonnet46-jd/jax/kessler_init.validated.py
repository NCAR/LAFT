# ---------------------------------------------------------------------------
# Translated by: Claude Sonnet 4.6 (claude-sonnet-4-6)
# Pass: final
# ---------------------------------------------------------------------------
"""
SCALAR-ONLY translation of the Fortran procedure kessler_init.

No JAX — plain Python scalars only.
Translated by Claude Sonnet 4.6
"""


def kessler_init(lv_in, pref_in, rhoqr_in, errmsg, errflg, lv, pref, rhoqr):
    """
    Set physical constants to be consistent with calling model.

    Parameters
    ----------
    lv_in    : float  latent heat of vaporization, J/kg
    pref_in  : float  reference pressure, Pa
    rhoqr_in : float  density of fresh liquid water, kg/m^3
    errmsg   : str    error message (out)
    errflg   : int    error flag (out)
    lv       : float  MODULE variable (INOUT)
    pref     : float  MODULE variable (INOUT)
    rhoqr    : float  MODULE variable (INOUT)

    Returns
    -------
    errmsg, errflg, lv, pref, rhoqr
    """
    errmsg = ""
    errflg = 0

    lv    = float(lv_in)
    pref  = float(pref_in) / 100.0   # Pa → hPa
    rhoqr = float(rhoqr_in)

    return errmsg, errflg, lv, pref, rhoqr
