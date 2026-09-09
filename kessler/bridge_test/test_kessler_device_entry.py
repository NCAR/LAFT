"""
Translation-DEPENDENT bridge tests — contract 2 (device-resident entry).

Since 2026-08-19 every array-bearing bridge exposes two entries built on the
same jitted code (LAFT/workflow_bridge/BRIDGE_WORKFLOW.md, "Two bridge
contracts"):

  kessler_run_bridge(...)          contract 1, host-facing (NumPy in/out,
                                   H2D + ONE batched D2H per call)
  kessler_run_bridge_device(...)   contract 2, device-resident (jax.Arrays
                                   already on the device in/out, NO transfers)

This file checks, for kessler_run (the project's only array-bearing
procedure — kessler_init is scalar-only and has no device entry):
  * the public device entry exists and the private alias points at it;
  * contract 2 on to_device(host inputs) returns, after jax.device_get,
    arrays BIT-IDENTICAL to contract 1 on the same host inputs, with the
    same output arity, shapes and dtypes;
  * contract 2 leaves its array outputs on the device (jax.Arrays);
  * MODULE vars (lv, pref, rhoqr) are threaded identically by both entries.

Needs the real translation in out/jax/ (it executes the kernel); skips with
a clear reason when it (or the bridge) is absent, exactly like
test_kessler_run_bridge.py. Runs in TRANSLATE_WORKFLOW Step 4.5 via
    python workflow_bridge/run_bridge_tests.py --require-dependent
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import sys
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

if not (project_root / "out" / "bridge" / "kessler_run_bridge.py").exists():
    pytest.skip("out/bridge/kessler_run_bridge.py not found — run "
                "workflow_bridge/phase03_make_bridge.py first",
                allow_module_level=True)
if not (project_root / "out" / "jax" / "kessler_run.py").exists():
    pytest.skip("out/jax/kessler_run.py not found — run the translator "
                "workflow first (these tests execute the real translation)",
                allow_module_level=True)

# The layout tests plant FAKE out.jax.* modules; always import the real ones.
sys.modules.pop("out.bridge.kessler_run_bridge", None)
sys.modules.pop("out.jax.kessler_run", None)

import out.bridge.kessler_run_bridge as B  # noqa: E402

NCOL, NZ = 10, 5
rng = np.random.default_rng(2026_08_19)


def _f2d(base, jitter):
    return np.ascontiguousarray(base + jitter * rng.standard_normal((NCOL, NZ)))


def host_inputs():
    """Physically plausible random fields, Fortran (ncol, nz) C-order."""
    z_row = np.tile(np.linspace(0.0, 5000.0, NZ), (NCOL, 1))
    return {
        "ncol": NCOL, "nz": NZ, "dt": 60.0, "lyr_surf": 1, "lyr_toa": NZ,
        "cpair": _f2d(1004.0, 0.0), "rair": _f2d(287.0, 0.0),
        "rho": _f2d(1.2, 0.01), "z": np.ascontiguousarray(z_row),
        "pk": _f2d(1.0, 0.001), "theta": _f2d(300.0, 1.0),
        "qv": np.abs(_f2d(0.010, 0.001)), "qc": np.abs(_f2d(0.001, 0.0002)),
        "qr": np.abs(_f2d(0.0001, 0.00002)),
        "precl": np.zeros(NCOL), "relhum": np.zeros((NCOL, NZ)),
        "lv": 2.5e6, "pref": 1000.0, "rhoqr": 1000.0,
    }


class TestDeviceEntryContract:
    def test_public_entry_and_alias(self):
        pub = getattr(B, "kessler_run_bridge_device", None)
        assert callable(pub), "kessler_run_bridge_device missing from the generated bridge"
        assert B._kessler_run_device is pub, "private alias must point at the public entry"

    def test_device_entry_bit_identical_to_host_bridge(self):
        h = host_inputs()
        # contract 1 — host NumPy in/out (strings pass through)
        r1 = B.kessler_run_bridge(scheme_name="", errmsg="", errflg=0, **h)
        (theta1, qv1, qc1, qr1, precl1, relhum1,
         _scheme, _errmsg, errflg1, lv1, pref1, rhoqr1) = r1

        # contract 2 — the same inputs shipped up front, outputs left on device
        d = {k: (B.to_device(v) if isinstance(v, np.ndarray) else v)
             for k, v in h.items()}
        r2 = B.kessler_run_bridge_device(errflg=0, **d)
        (theta2, qv2, qc2, qr2, precl2, relhum2, errflg2,
         lv2, pref2, rhoqr2) = r2

        for name, a1, a2 in (("theta", theta1, theta2), ("qv", qv1, qv2),
                             ("qc", qc1, qc2), ("qr", qr1, qr2),
                             ("precl", precl1, precl2), ("relhum", relhum1, relhum2)):
            assert isinstance(a2, jax.Array), f"{name}: contract 2 must leave outputs on the device"
            a2h = np.asarray(jax.device_get(a2))
            assert a2h.shape == np.asarray(a1).shape
            assert a2h.dtype == np.float64
            assert np.array_equal(a2h, np.asarray(a1)), \
                f"{name}: device entry differs from host bridge (not bit-identical)"

        assert int(np.asarray(jax.device_get(errflg2))) == errflg1 == 0
        # MODULE vars threaded identically (INOUT pattern, unchanged values)
        for m1, m2 in ((lv1, lv2), (pref1, pref2), (rhoqr1, rhoqr2)):
            assert float(np.asarray(jax.device_get(m2))) == float(np.asarray(m1))
