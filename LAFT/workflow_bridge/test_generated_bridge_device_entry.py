#!/usr/bin/env python3
"""
Translation-DEPENDENT bridge test — contract 2 (device-resident entry) for
EVERY array-bearing procedure — SHARED across projects (lives in
LAFT/workflow_bridge/, runs from any project root; no per-project copy).

Since 2026-08-19 every array-bearing bridge exposes two entries built on the
same jitted code (BRIDGE_WORKFLOW.md §Two bridge contracts):

  <proc>_bridge(...)          contract 1, host-facing (NumPy in/out, H2D +
                              ONE batched D2H per call)
  <proc>_bridge_device(...)   contract 2, device-resident (jax.Arrays already
                              on the device in/out, NO transfers)

For each array-bearing procedure this file checks, against the REAL
translation in out/jax/:
  * the translation exists, the public device entry exists and the private
    alias points at it;
  * contract 2 on to_device(host inputs) returns, after jax.device_get,
    values BIT-IDENTICAL to contract 1 on the same host inputs (same arity,
    shapes, dtypes; NaN-equal compare, since synthetic inputs may
    legitimately produce NaN in a kernel that is not the subject of the test);
  * contract 2 leaves array outputs on the device (jax.Arrays).

WHERE THE CONTRACT COMES FROM
-----------------------------
The procedure list, parameter metadata and module-variable list are the ones
test_generated_bridge_layout.py derives from the packets through the bridge
generator's own extraction — nothing is transcribed per project.

INPUTS — generic by default, per-project hook when the kernel needs it
-----------------------------------------------------------------------
By default a packet-derived synthetic builder (_kernel_inputs): integer
size arguments bound to the array extents, index arguments inside every
array, small positive real values. Bit-identity of the two entries holds
for any input, so this is enough for most kernels. A kernel that is not
defined on synthetic data (lookup-table paths, iteration counts that depend
on physical magnitudes) gets physically plausible inputs from an OPTIONAL
per-project hook:

    <project>/bridge_test/bridge_inputs.py
        def inputs(proc: str) -> dict | None

returning the full host-side kwargs for `<proc>_bridge` (params + module
vars + optional pass-through kwargs), or None to fall back to the generic
builder for that procedure.

OUT OF SCOPE (stays per project, BRIDGE_WORKFLOW checklist item 5)
------------------------------------------------------------------
A CHAIN test — procedures handing device-resident intermediates to each
other across a per-step phase sequence (init → run → final). The packets do
not say which procedures chain; add it by hand where it applies.

Skips (module-level, with reason) while out/jax/ holds no array-bearing
translation; becomes part of the gate at TRANSLATE_WORKFLOW Step 4.5 via
    python workflow_bridge/run_bridge_tests.py --require-dependent
A project with no array-bearing procedure at all records n/a as a PASS
(the gate forbids skips).

History: generalised 2026-10-09 from a per-project parameterised device-entry
test (2026-08-24); project-specific input logic moved behind the hook above.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import importlib
import inspect
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from test_generated_bridge_layout import (  # noqa: E402  (contract discovery; plants no fakes)
    CONTRACTS, ARRAY_PROCS, PROJECT_ROOT, _make_inputs, _import_line, _as_tuple, safe_py_name,
)

JAX_DIR = PROJECT_ROOT / "out" / "jax"


def _translation_file(proc) -> Path:
    return JAX_DIR / f"{_import_line(CONTRACTS[proc])[0]}.py"


if ARRAY_PROCS and not any(_translation_file(p).exists() for p in ARRAY_PROCS):
    pytest.skip("out/jax/ holds no array-bearing translation — run the translator "
                "workflow first (these tests execute the real kernels)",
                allow_module_level=True)


# ---------------------------------------------------------------------------
# Inputs: per-project hook, else the generic packet-derived builder
# ---------------------------------------------------------------------------

_HOOK = None          # None = not looked up yet; False = no hook in this project


def _project_hook():
    global _HOOK
    if _HOOK is None:
        hook_dir = PROJECT_ROOT / "bridge_test"
        if (hook_dir / "bridge_inputs.py").exists():
            if str(hook_dir) not in sys.path:
                sys.path.insert(0, str(hook_dir))
            _HOOK = importlib.import_module("bridge_inputs")
            assert callable(getattr(_HOOK, "inputs", None)), \
                "bridge_test/bridge_inputs.py must define inputs(proc) -> dict | None"
        else:
            _HOOK = False
    return _HOOK


def _kernel_inputs(contract, rng_seed=42):
    """Generic inputs for a REAL kernel: the layout builder's integer rule
    (sizes bound to extents, indices inside every array) with SMALL POSITIVE
    real values — arrays in [1e-3, 2e-3) plus a per-parameter offset, real
    scalars near 1. Bit-identity of the two entries holds for any input; the
    magnitudes only keep data-dependent loops short. Large random values
    made a Kessler translation's sub-stepping `lax.while_loop` run for
    minutes (claude-sonnet46-jd, 2026-10-09: dt ~ 1e2 and mixing ratios ~ 10
    kg/kg give an astronomical substep count) — this is the generic builder's
    limit, and the per-project hook is for what it cannot know."""
    inputs = _make_inputs(contract, rng_seed)          # ints, logicals, strings, mvars
    rng = np.random.default_rng(rng_seed)
    for i, p in enumerate(contract.params):
        n = safe_py_name(p.name)
        if p.dtype in ("character", "integer", "logical"):
            continue
        if p.rank == 0:
            inputs[n] = 1.0 + 0.01 * i
        else:
            inputs[n] = 1e-3 * (1.0 + rng.random(np.shape(inputs[n]))) + 1e-4 * i
    return inputs


def _inputs(proc):
    hook = _project_hook()
    if hook:
        got = hook.inputs(proc)
        if got is not None:
            return got
    return _kernel_inputs(CONTRACTS[proc])


# ---------------------------------------------------------------------------
# Real bridge + split into host / device kwargs
# ---------------------------------------------------------------------------

def _real_bridge(proc):
    """Import the bridge bound to the REAL translation (drop any planted fake)."""
    modname, _, _ = _import_line(CONTRACTS[proc])
    sys.modules.pop(f"out.bridge.{proc}_bridge", None)
    sys.modules.pop(f"out.jax.{modname}", None)
    # a real translation may import OTHER real translations (host-model
    # wrappers -> main routine): evict every fake the layout tests planted
    # (fakes are types.ModuleType objects without a __file__)
    for name in [m for m in sys.modules if m.startswith("out.jax.")]:
        if getattr(sys.modules[name], "__file__", None) is None:
            sys.modules.pop(name, None)
    return importlib.import_module(f"out.bridge.{proc}_bridge")


def _split(contract, inputs):
    """host kwargs (all params + mvars + optional pass-through kwargs) and
    device kwargs (minus CHARACTER, arrays uploaded)."""
    mod = _real_bridge(contract.proc)
    host = dict(inputs)
    dev = {}
    for p in contract.nonchar:
        n = contract.py[p.name]
        dev[n] = mod.to_device(inputs[n]) if p.rank > 0 else inputs[n]
    for v in contract.mvars:
        dev[v] = inputs[v]
    # optional pass-through kwargs (e.g. lookup tables): forward whatever the
    # device entry accepts that the host inputs carry
    dev_sig = inspect.signature(getattr(mod, f"{contract.proc}_bridge_device")).parameters
    for k, v in inputs.items():
        if k not in dev and k in dev_sig:
            dev[k] = v
    return mod, host, dev


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_array_bearing_procedures_known():
    """A project with no array-bearing procedure has nothing to check here —
    recorded as a pass (n/a), never a skip, so --require-dependent stays
    meaningful."""
    assert CONTRACTS, "no generated bridges with packets — run phase03 first"


@pytest.mark.parametrize("proc", ARRAY_PROCS)
class TestDeviceEntryContract:

    def test_translation_present(self, proc):
        f = _translation_file(proc)
        assert f.exists(), f"{f.relative_to(PROJECT_ROOT)} missing — translation incomplete"

    def test_public_entry_and_alias(self, proc):
        m = _real_bridge(proc)
        pub = getattr(m, f"{proc}_bridge_device", None)
        assert callable(pub), f"{proc}_bridge_device missing from the generated bridge"
        assert getattr(m, f"_{proc}_device") is pub, "private alias must point at the public entry"

    def test_device_entry_bit_identical_to_host_bridge(self, proc):
        c = CONTRACTS[proc]
        mod, host, dev = _split(c, _inputs(proc))
        r_host = _as_tuple(getattr(mod, f"{proc}_bridge")(**host))
        r_dev = _as_tuple(getattr(mod, f"{proc}_bridge_device")(**dev))
        # bridge returns = written params (incl. CHARACTER passthrough) + mvars;
        # device returns = written non-CHARACTER params + mvars
        assert len(r_host) == len(c.bridge_returns), \
            f"{proc}: host arity {len(r_host)} != {len(c.bridge_returns)}"
        assert len(r_dev) == len(c.core_returns), \
            f"{proc}: device arity {len(r_dev)} != {len(c.core_returns)}"
        host_by_name = dict(zip(c.bridge_returns, r_host))
        for name, d_val in zip(c.core_returns, r_dev):
            h_val = host_by_name[name]
            if name in c.mvars:
                np.testing.assert_array_equal(np.asarray(jax.device_get(d_val)), np.asarray(h_val),
                                              err_msg=f"{proc}: module var {name} differs")
                continue
            d_host = np.asarray(jax.device_get(d_val))
            h_arr = np.asarray(h_val)
            assert d_host.shape == h_arr.shape, f"{proc}.{name}: shape {d_host.shape} != {h_arr.shape}"
            if h_arr.ndim > 0:
                assert isinstance(d_val, jax.Array), \
                    f"{proc}.{name}: contract 2 must leave arrays on the device"
                assert d_host.dtype == np.float64
            assert np.array_equal(d_host, h_arr, equal_nan=True), \
                f"{proc}.{name}: device entry differs from host bridge (not bit-identical)"
