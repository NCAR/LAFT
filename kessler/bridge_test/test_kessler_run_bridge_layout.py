#!/usr/bin/env python3
"""
Translation-INDEPENDENT tests for out/bridge/kessler_run_bridge.py.

This is the bridge-workflow test gate (see workflow_bridge/BRIDGE_WORKFLOW.md):
it runs right after phase03_make_bridge.py generates the bridge, BEFORE any
LLM translation exists in out/jax/. To break the bridge's module-level
`from out.jax.kessler_run import kessler_run_core`, a pass-through FAKE core
is injected into sys.modules before the bridge is imported — so these tests
exercise exactly what phase03 is responsible for (argument wiring order,
in-jit axis reversal on exactly the rank>=2 variables, output layout,
module-var threading) with zero physics involved.

PATH C contract (device-side layout conversion): the bridge ships arrays to
the device unchanged (pure H2D), reverses axes INSIDE its jitted device
wrapper on both sides of the core call, and returns pure D2H. With a
pass-through core the bridge reduces to
    reverse -> identity -> reverse
so every array must come back bit-identical, original shape, standard
C-order (ncol, nz). The fake core is a PURE jit-traceable function — it runs
under jax.jit, so its array arguments are tracers: it records tracer
*shapes/dtypes* (trace-time only) and the concrete values of the jit-static
scalars; value-level wiring is proven by the roundtrip identity (every field
carries unique values, so a swapped argument cannot cancel out).

CHARACTER args (scheme_name, errmsg) must NOT reach the core — strings
cannot cross the jit boundary — and pass through the bridge unchanged in
their return slots.

Run with:
    python -m pytest bridge_test/test_kessler_run_bridge_layout.py -v

The translation-DEPENDENT suite (physics, bridge-vs-direct-JAX) lives in
bridge_test/test_kessler_run_bridge.py and belongs to the translator
workflow's validation stage.

Layout contract:
  Fortran side : shape (ncol, nz), standard C-order NumPy
  Bridge       : pure H2D, in-jit reversal to (nz, ncol) for the core,
                 in-jit reversal back, pure D2H
  JAX core side: shape (nz, ncol)
"""

import os
os.environ["JAX_ENABLE_X64"] = "1"

import jax
jax.config.update("jax_enable_x64", True)

import sys
import types
import numpy as np
import pytest
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Fake pass-through core, injected BEFORE the bridge import
# ---------------------------------------------------------------------------
# The generated bridge does `from out.jax.kessler_run import kessler_run_core`
# at module level. Planting a fake module under that name makes the bridge
# importable with out/jax/ empty (or holding any translation — the fake is
# always used here, so these tests stay deterministic regardless of out/jax
# state). Real modules possibly cached by the other test file are popped
# first so import order between the two files never matters.
#
# NOTE the fake is called INSIDE the bridge's jitted device wrapper, so its
# body executes at trace time (once per shape signature — jit caches compiled
# calls, so it does NOT re-run every bridge call). It must be pure and
# traceable: no branching on tracer values, no concrete-value capture of
# arrays. Trace-time recording of shapes/dtypes and of the concrete static
# scalars is safe and is all these tests need.

_traced = {}   # shapes/dtypes (arrays) + concrete values (statics) at trace


def _fake_kessler_run_core(ncol, nz, dt, lyr_surf, lyr_toa, cpair, rair, rho,
                           z, pk, theta, qv, qc, qr, precl, relhum, errflg,
                           lv, pref, rhoqr):
    """Pass-through stub: returns its inputs in the real core output order."""
    args = locals()
    _traced.clear()
    for name, val in args.items():
        if hasattr(val, "shape"):
            _traced[name] = {"shape": tuple(val.shape), "dtype": str(val.dtype)}
        else:
            _traced[name] = {"value": val}   # jit-static: concrete at trace
    return (theta, qv, qc, qr, precl, relhum, errflg, lv, pref, rhoqr)


for _mod in ("out.bridge.kessler_run_bridge", "out.jax.kessler_run"):
    sys.modules.pop(_mod, None)

_fake_module = types.ModuleType("out.jax.kessler_run")
_fake_module.kessler_run_core = _fake_kessler_run_core
sys.modules["out.jax.kessler_run"] = _fake_module

from out.bridge.kessler_run_bridge import (  # noqa: E402
    kessler_run_bridge,
    to_device,
    to_host,
)


def teardown_module(module):
    """Drop the fake and the bridge bound to it so later imports get real code."""
    sys.modules.pop("out.jax.kessler_run", None)
    sys.modules.pop("out.bridge.kessler_run_bridge", None)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

NCOL, NZ = 4, 3      # deliberately unequal: a missing axis reversal is a
                     # shape mismatch, not a silent pass

ARRAY_2D_FIELDS = ["cpair", "rair", "rho", "z", "pk",
                   "theta", "qv", "qc", "qr", "relhum"]
# (return-slot index, field name) for the 2-D outputs; precl (1-D) is slot 4
OUTPUT_2D_SLOTS = [(0, "theta"), (1, "qv"), (2, "qc"), (3, "qr"), (5, "relhum")]


def _distinct_2d(offset):
    """(ncol, nz) C-order array with every element unique across all fields —
    a swapped argument or missing reversal cannot go unnoticed."""
    base = np.arange(NCOL * NZ, dtype=np.float64).reshape(NCOL, NZ)
    return np.ascontiguousarray(base + 1000.0 * offset)


@pytest.fixture
def wiring_inputs():
    """Synthetic inputs where every field carries a unique value range.
    Arrays are standard C-order (ncol, nz) — the driver-side contract."""
    inputs = {
        'ncol'       : NCOL,
        'nz'         : NZ,
        'dt'         : 60.0,
        'lyr_surf'   : 1,
        'lyr_toa'    : NZ,
        'precl'      : np.arange(NCOL, dtype=np.float64) + 500.0,
        'scheme_name': 'kessler',
        'errmsg'     : '',
        'errflg'     : 0,
        'lv'         : 2.5e6,
        'pref'       : 1000.0,
        'rhoqr'      : 999.0,
    }
    for i, name in enumerate(ARRAY_2D_FIELDS):
        inputs[name] = _distinct_2d(i + 1)
    return inputs


# ---------------------------------------------------------------------------
# Transfer helper tests (pure helpers, no kernel at all)
# ---------------------------------------------------------------------------

class TestTransferHelpers:

    def test_to_device_no_host_permute(self):
        """to_device ships the array AS-IS: same shape, same values, float64 —
        no host-side transpose or reordering."""
        x = np.arange(12.0).reshape(4, 3) + 7.0
        d = to_device(x)
        assert tuple(d.shape) == (4, 3)
        assert str(d.dtype) == "float64"
        np.testing.assert_array_equal(np.asarray(d), x)

    def test_roundtrip_identity(self):
        """to_host(to_device(x)) is bit-exact, same shape, float64."""
        for x in (np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                  np.arange(12.0).reshape(4, 3) + 100.0):
            back = to_host(to_device(x))
            assert back.shape == x.shape
            assert back.dtype == np.float64
            np.testing.assert_array_equal(back, x)

    def test_to_host_c_order(self):
        """to_host returns standard C-order NumPy."""
        back = to_host(to_device(np.arange(12.0).reshape(4, 3)))
        assert back.flags['C_CONTIGUOUS']


# ---------------------------------------------------------------------------
# Bridge wiring tests (pass-through core inside the jitted device wrapper)
# ---------------------------------------------------------------------------

class TestBridgeWiring:

    def test_passthrough_roundtrip_identity(self, wiring_inputs):
        """With an identity core, every output must equal its input
        bit-exactly — proves each variable rides its own argument slot and
        the in-jit reverse/reverse pair is lossless."""
        result = kessler_run_bridge(**wiring_inputs)
        assert len(result) == 12

        for idx, name in OUTPUT_2D_SLOTS:
            np.testing.assert_array_equal(
                result[idx], wiring_inputs[name],
                err_msg=f"{name}: bridge output != input under identity core "
                        f"(argument wiring or reversal defect)")
        np.testing.assert_array_equal(result[4], wiring_inputs['precl'])

    def test_outputs_c_order_layout(self, wiring_inputs):
        """Outputs must come back (ncol, nz) standard C-order NumPy,
        precl (ncol,) — the PATH C driver-side contract."""
        result = kessler_run_bridge(**wiring_inputs)

        for idx, name in OUTPUT_2D_SLOTS:
            arr = result[idx]
            assert isinstance(arr, np.ndarray), f"{name}: not a NumPy array"
            assert arr.shape == (NCOL, NZ), f"{name}: shape {arr.shape}"
            assert arr.flags['C_CONTIGUOUS'], f"{name}: not C-contiguous"
        assert result[4].shape == (NCOL,), f"precl shape {result[4].shape}"

    def test_core_receives_reversed_layout(self, wiring_inputs):
        """Inside the jitted wrapper the core must see every 2-D array with
        REVERSED axes (nz, ncol) float64 — the in-jit input reversal happened
        — while precl (1-D) passes through with its shape unchanged."""
        kessler_run_bridge(**wiring_inputs)

        for name in ARRAY_2D_FIELDS:
            info = _traced[name]
            assert info["shape"] == (NZ, NCOL), (
                f"{name}: core saw shape {info['shape']}, expected ({NZ},{NCOL})")
            assert info["dtype"] == "float64", f"{name}: dtype {info['dtype']}"

        assert _traced['precl']["shape"] == (NCOL,), (
            f"precl: core saw {_traced['precl']['shape']}")

    def test_static_scalars_concrete_at_trace(self, wiring_inputs):
        """Shape/index integer scalars are jit statics: they must reach the
        core as concrete Python values (not tracers) with their input values."""
        kessler_run_bridge(**wiring_inputs)

        for name in ('ncol', 'nz', 'lyr_surf', 'lyr_toa', 'errflg'):
            info = _traced[name]
            assert "value" in info, f"{name}: reached the core as a tracer, " \
                                    f"expected a concrete jit-static"
            assert info["value"] == wiring_inputs[name], (
                f"{name}: core saw {info['value']!r}, "
                f"expected {wiring_inputs[name]!r}")

    def test_strings_stay_host_side(self, wiring_inputs):
        """CHARACTER args must NOT reach the core (strings cannot cross the
        jit boundary) and must pass through the bridge unchanged in their
        return slots (slots 6, 7); errflg (slot 8) is numeric passthrough."""
        result = kessler_run_bridge(**wiring_inputs)

        assert 'scheme_name' not in _traced
        assert 'errmsg' not in _traced
        assert result[6] == wiring_inputs['scheme_name']
        assert result[7] == wiring_inputs['errmsg']
        assert result[8] == wiring_inputs['errflg']
        assert isinstance(result[8], int), (
            f"errflg: expected Python int, got {type(result[8])}")

    def test_module_vars_threaded(self, wiring_inputs):
        """Module vars lv/pref/rhoqr must reach the core and come back in
        the last 3 return slots (INOUT pattern). They are traced (not
        static), so in-core they are tracers; value equality is proven by
        the passthrough return."""
        result = kessler_run_bridge(**wiring_inputs)

        for name in ('lv', 'pref', 'rhoqr'):
            assert name in _traced, f"{name}: never reached the core"
            assert "shape" in _traced[name], (
                f"{name}: reached the core as a static, expected traced")

        assert float(result[9])  == wiring_inputs['lv']
        assert float(result[10]) == wiring_inputs['pref']
        assert float(result[11]) == wiring_inputs['rhoqr']
