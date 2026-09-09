#!/usr/bin/env python3
"""
Translation-DEPENDENT tests for kessler_run_bridge.py.

These tests execute the real JAX translation (out/jax/kessler_run.py), so
they belong to the TRANSLATOR workflow's validation stage — run them after a
translation lands, not after bridge generation. If the translation (or the
bridge) is missing, the whole file skips with a clear reason.

The translation-INDEPENDENT layout/wiring tests live in
bridge_test/test_kessler_run_bridge_layout.py (bridge-workflow gate).

Run with:
    python -m pytest bridge_test/test_kessler_run_bridge.py -v

Layout contract (PATH C, device-side layout conversion):
  Fortran side : shape (ncol, nz) — any host layout accepted; outputs are
                 standard C-order NumPy (ncol, nz)
  Bridge       : pure H2D, in-jit axis reversal to (nz, ncol) around the
                 jitted core, in-jit reversal back, pure D2H
  JAX core side: shape (nz, ncol)

PATH C NOTE (flagged, 2026-08-11): the array-bearing bridge calls
kessler_run_core directly — the translated wrapper's host-side string
handling does not run, so bridge CHARACTER outputs (scheme_name, errmsg)
are passthrough of the inputs. errflg is numeric and propagates from the
core. Tests below assert this contract explicitly.
"""

import os
os.environ["JAX_ENABLE_X64"] = "1"
# Optional: avoid CUDA plugin noise on CPU-only nodes
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

import sys
import numpy as np
import pytest
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Skip guard: this suite needs the generated bridge AND a real translation.
if not (project_root / "out" / "bridge" / "kessler_run_bridge.py").exists():
    pytest.skip("out/bridge/kessler_run_bridge.py not found — run "
                "workflow_bridge/phase03_make_bridge.py first",
                allow_module_level=True)
if not (project_root / "out" / "jax" / "kessler_run.py").exists():
    pytest.skip("out/jax/kessler_run.py not found — run the translator "
                "workflow first (these tests execute the real translation)",
                allow_module_level=True)

# The layout test file plants a FAKE out.jax.kessler_run in sys.modules; pop
# any cached bridge/kernel modules so this file always imports the real ones,
# regardless of test-collection order.
for _mod in ("out.bridge.kessler_run_bridge", "out.jax.kessler_run"):
    sys.modules.pop(_mod, None)

import jax.numpy as jnp  # noqa: E402

from out.bridge.kessler_run_bridge import (  # noqa: E402
    kessler_run_bridge,
    to_device,
    to_host,
)
from out.jax.kessler_run import kessler_run  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fortran_2d(arr_ncol_nz):
    """
    Convert (ncol, nz) data into true Fortran layout:
    keep shape (ncol, nz) and make it F-contiguous.

    Simulates what a real Fortran model passes to the bridge.
    """
    return np.asfortranarray(arr_ncol_nz)


@pytest.fixture
def sample_inputs():
    """Sample atmospheric inputs in true Fortran layout: (ncol, nz) F-order."""
    ncol, nz = 10, 5

    z_row = np.tile(np.linspace(0, 5000, nz), (ncol, 1))  # (ncol, nz)

    return {
        'ncol'       : ncol,
        'nz'         : nz,
        'dt'         : 60.0,
        'lyr_surf'   : 1,
        'lyr_toa'    : nz,
        'cpair'      : make_fortran_2d(np.full((ncol, nz), 1004.0)),
        'rair'       : make_fortran_2d(np.full((ncol, nz), 287.0)),
        'rho'        : make_fortran_2d(np.full((ncol, nz), 1.2)),
        'z'          : make_fortran_2d(z_row),
        'pk'         : make_fortran_2d(np.full((ncol, nz), 1.0)),
        'theta'      : make_fortran_2d(np.full((ncol, nz), 300.0)),
        'qv'         : make_fortran_2d(np.full((ncol, nz), 0.010)),
        'qc'         : make_fortran_2d(np.full((ncol, nz), 0.001)),
        'qr'         : make_fortran_2d(np.full((ncol, nz), 0.0001)),
        'precl'      : np.zeros(ncol),
        'relhum'     : make_fortran_2d(np.zeros((ncol, nz))),
        'scheme_name': '',
        'errmsg'     : '',
        'errflg'     : 0,
        'lv'         : 2.5e6,
        'pref'       : 1000.0,
        'rhoqr'      : 1000.0,
    }


# ---------------------------------------------------------------------------
# Bridge functionality tests
# (layout-conversion tests moved to test_kessler_run_bridge_layout.py)
# ---------------------------------------------------------------------------

class TestBridgeFunctionality:

    def test_runs_successfully(self, sample_inputs):
        result = kessler_run_bridge(**sample_inputs)
        assert len(result) == 12

        (theta_out, qv_out, qc_out, qr_out, precl_out, relhum_out,
         scheme_name_out, errmsg_out, errflg_out,
         lv_out, pref_out, rhoqr_out) = result

        assert errflg_out == 0, f"Bridge returned error: {errmsg_out}"

        ncol, nz = sample_inputs['ncol'], sample_inputs['nz']
        assert theta_out.shape  == (ncol, nz)
        assert qv_out.shape     == (ncol, nz)
        assert precl_out.shape  == (ncol,)
        assert relhum_out.shape == (ncol, nz)

    def test_bridge_vs_direct_jax(self, sample_inputs):
        bridge_result = kessler_run_bridge(**sample_inputs)

        # Build JAX inputs manually (Fortran -> core layout: (nz, ncol))
        jax_inputs = {k: v for k, v in sample_inputs.items()}
        for key in ['cpair', 'rair', 'rho', 'z', 'pk',
                    'theta', 'qv', 'qc', 'qr', 'relhum']:
            jax_inputs[key] = jnp.asarray(
                np.ascontiguousarray(sample_inputs[key].T), dtype=jnp.float64)
        jax_inputs['precl'] = jnp.asarray(sample_inputs['precl'],
                                          dtype=jnp.float64)

        jax_result = kessler_run(**jax_inputs)

        # Convert JAX outputs back to Fortran (ncol, nz) for comparison
        jax_theta  = np.array(jax_result[0]).T
        jax_qv     = np.array(jax_result[1]).T
        jax_qc     = np.array(jax_result[2]).T
        jax_qr     = np.array(jax_result[3]).T
        jax_precl  = np.array(jax_result[4])
        jax_relhum = np.array(jax_result[5]).T

        np.testing.assert_array_almost_equal(bridge_result[0], jax_theta,  decimal=10)
        np.testing.assert_array_almost_equal(bridge_result[1], jax_qv,     decimal=10)
        np.testing.assert_array_almost_equal(bridge_result[2], jax_qc,     decimal=10)
        np.testing.assert_array_almost_equal(bridge_result[3], jax_qr,     decimal=10)
        np.testing.assert_array_almost_equal(bridge_result[4], jax_precl,  decimal=10)
        np.testing.assert_array_almost_equal(bridge_result[5], jax_relhum, decimal=10)

        # PATH C: bridge CHARACTER outputs are passthrough of the inputs
        # (the direct wrapper call still sets its own strings — not compared)
        assert bridge_result[6] == sample_inputs['scheme_name']
        assert bridge_result[7] == sample_inputs['errmsg']
        # errflg is numeric and must agree with the wrapper path
        assert bridge_result[8]  == jax_result[8]
        assert bridge_result[9]  == jax_result[9]
        assert bridge_result[10] == jax_result[10]
        assert bridge_result[11] == jax_result[11]

    def test_driver_layout_preserved(self, sample_inputs):
        """PATH C contract: outputs are (ncol, nz) standard C-order NumPy."""
        result = kessler_run_bridge(**sample_inputs)
        ncol, nz = sample_inputs['ncol'], sample_inputs['nz']

        (theta_out, qv_out, qc_out, qr_out, precl_out, relhum_out,
         _, _, _, _, _, _) = result

        for name, arr in [('theta',  theta_out),
                          ('qv',     qv_out),
                          ('qc',     qc_out),
                          ('qr',     qr_out),
                          ('relhum', relhum_out)]:
            assert arr.shape == (ncol, nz),   f"{name}: shape {arr.shape} != ({ncol},{nz})"
            assert arr.flags['C_CONTIGUOUS'], f"{name}: not C-contiguous"

        assert precl_out.shape == (ncol,), f"precl shape {precl_out.shape}"

    def test_physics_sanity(self, sample_inputs):
        result = kessler_run_bridge(**sample_inputs)
        (theta_out, qv_out, qc_out, qr_out, precl_out, relhum_out,
         _, _, errflg_out, _, _, _) = result

        assert errflg_out == 0
        assert np.all(theta_out > 200.0) and np.all(theta_out < 400.0)
        assert np.all(qv_out >= 0.0)
        assert np.all(qc_out >= 0.0)
        assert np.all(qr_out >= 0.0)
        assert np.all(precl_out >= 0.0)
        assert np.all(relhum_out >= 0.0) and np.all(relhum_out <= 100.0)

    def test_error_handling_negative_dt(self):
        """PATH C BEHAVIOR CHANGE (flagged, 2026-08-11): the bridge calls the
        core directly, so the wrapper's `if dt <= 0` guard and its errmsg
        ("KESSLER called with nonpositive dt") do NOT run on the bridge path.
        The error still surfaces NUMERICALLY: the core's own dt0 < 1e-12
        check raises errflg=1. errmsg is CHARACTER passthrough ('' here).
        Pre-path-C this test asserted the wrapper's errmsg text."""
        ncol, nz = 5, 3

        result = kessler_run_bridge(
            ncol=ncol, nz=nz, dt=-60.0, lyr_surf=1, lyr_toa=nz,
            cpair =make_fortran_2d(np.ones((ncol, nz))),
            rair  =make_fortran_2d(np.ones((ncol, nz))),
            rho   =make_fortran_2d(np.ones((ncol, nz))),
            z     =make_fortran_2d(np.ones((ncol, nz))),
            pk    =make_fortran_2d(np.ones((ncol, nz))),
            theta =make_fortran_2d(np.ones((ncol, nz))),
            qv    =make_fortran_2d(np.ones((ncol, nz))),
            qc    =make_fortran_2d(np.ones((ncol, nz))),
            qr    =make_fortran_2d(np.ones((ncol, nz))),
            precl =np.ones(ncol),
            relhum=make_fortran_2d(np.ones((ncol, nz))),
            scheme_name='', errmsg='', errflg=0,
            lv=2.5e6, pref=1000.0, rhoqr=1000.0,
        )

        assert len(result) == 12
        _, _, _, _, _, _, _, errmsg_out, errflg_out, _, _, _ = result
        assert errflg_out == 1, "core did not flag nonpositive dt numerically"
        assert errmsg_out == '', "PATH C: errmsg must be CHARACTER passthrough"

    def test_module_vars_inout_pattern(self, sample_inputs):
        sample_inputs['lv']    = 2.6e6
        sample_inputs['pref']  = 950.0
        sample_inputs['rhoqr'] = 999.0

        result = kessler_run_bridge(**sample_inputs)
        _, _, _, _, _, _, _, _, _, lv_out, pref_out, rhoqr_out = result

        assert lv_out    == sample_inputs['lv']
        assert pref_out  == sample_inputs['pref']
        assert rhoqr_out == sample_inputs['rhoqr']