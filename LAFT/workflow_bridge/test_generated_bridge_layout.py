#!/usr/bin/env python3
"""
Translation-INDEPENDENT layout/wiring tests for EVERY generated bridge —
SHARED across projects (lives in LAFT/workflow_bridge/, runs from any
project root; no per-project copy, no per-project edit).

This is the bridge-workflow test gate (BRIDGE_WORKFLOW.md §Step 2). It runs
right after phase03_make_bridge.py, BEFORE any translation exists in
out/jax/: a pass-through FAKE kernel is planted in sys.modules under the
translated module's name, so each array-bearing bridge reduces to

    pure H2D  →  [in-jit reverse → identity → in-jit reverse]  →  batched D2H

and every array must come back bit-identical, original shape, standard
C-order. Any wiring defect (swapped argument, lost output slot, dropped
module var, a per-step scalar made jit-static, a CHARACTER arg leaking into
the jitted region) breaks that immediately. Scalar-only bridges are checked
to call the translated wrapper with every argument and pass its returns
straight back.

WHERE THE CONTRACT COMES FROM
-----------------------------
Not transcribed by hand: each procedure's parameter metadata (rank, dtype,
output-ness) and module-variable list are taken from the SAME source phase03
used to emit the bridge — `BridgeGenerator.extract_parameters()` /
`extract_module_info()` over out/phase1_index.json + out/packets/, plus
[bridge.optional_kwargs] and the per-step-scalar rule from config/project.toml.
The test then checks the generated file against that contract. A new
procedure, a renamed argument or a changed module-var list needs no test
edit — the contract follows the packets.

Because the contract and the bridge come from the same metadata, this file
proves the GENERATOR wrote what its own inputs say (wiring, reversal, slots,
statics) — it cannot catch a wrong packet. That is what the frontend stage
and the semantic audit are for.

History: generalised 2026-10-09 from a per-project parameterised layout test
(2026-08-24, PATH C + two contracts); the per-project hand-written layout file
it replaced (kessler/bridge_test/test_kessler_run_bridge_layout.py) is kept in
the repository history as the reference for the mutation check. Dimensions, integer values and counts that were
project-specific are derived or dropped here.

Run from a project root (jax-validate env):
    python -m pytest workflow_bridge/test_generated_bridge_layout.py -v
"""

import os
os.environ["JAX_ENABLE_X64"] = "1"

import importlib
import inspect
import re
import sys
import types
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402
import pytest  # noqa: E402

# This file lives in LAFT/workflow_bridge/ (symlinked into each project as
# <project>/workflow_bridge/). Resolve the real location for the generator
# and the shared config loader; the PROJECT is whatever the cwd says —
# framework_config walks up from the cwd to config/project.toml, exactly as
# phase03 and run_bridge_tests.py do.
_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent / "config")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import phase03_make_bridge as ph3  # noqa: E402  (the generator's own metadata)
from framework_config import get_config, safe_py_name  # noqa: E402

_cfg = get_config()
PROJECT_ROOT = _cfg.root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))   # for `import out.bridge.<proc>_bridge`

BRIDGE_DIR = Path(_cfg.bridge_dir)

# ---------------------------------------------------------------------------
# Array extents — every axis a DIFFERENT length, so a missing or extra axis
# reversal is a shape mismatch, never a silent pass. Rank r uses the first r.
# ---------------------------------------------------------------------------
_EXTENTS = (4, 5, 3, 2, 6, 7)


def _shape(rank: int) -> tuple:
    assert 1 <= rank <= len(_EXTENTS), f"rank {rank} unsupported by this test"
    return _EXTENTS[:rank]


# ---------------------------------------------------------------------------
# Contract discovery — one BridgeGenerator over the real packets
# ---------------------------------------------------------------------------

class Contract:
    """Everything the test needs to know about one generated bridge."""

    def __init__(self, gen, packet_file: Path):
        packet = gen.load_merged_packet(packet_file)
        self.orig_name = packet["proc_name"]
        self.proc = self.orig_name.lower()
        self.params = gen.extract_parameters(packet)
        _, mvars = gen.extract_module_info(packet)
        pnames = {p.name.lower() for p in self.params}
        mvars = [v for v in mvars if v.lower() not in pnames]
        self.optional = [safe_py_name(v) for v in
                         gen.optional_kwargs.get(self.proc, [])]
        opt_lower = {v.lower() for v in self.optional}
        self.mvars = [safe_py_name(v) for v in mvars if v.lower() not in opt_lower]
        self.has_arrays = any(p.rank > 0 for p in self.params)
        self.nonchar = [p for p in self.params if p.dtype != "character"]
        self.char = [safe_py_name(p.name) for p in self.params if p.dtype == "character"]
        self.statics = [safe_py_name(p.name) for p in self.params
                        if p.rank == 0 and p.dtype in ("integer", "logical")
                        and not _cfg.is_per_step_scalar(p.name)]
        self.per_step = [safe_py_name(p.name) for p in self.params
                         if p.rank == 0 and p.dtype in ("integer", "logical")
                         and _cfg.is_per_step_scalar(p.name)]
        # core positional order: non-CHARACTER params, then module vars
        self.core_positional = [safe_py_name(p.name) for p in self.nonchar] + self.mvars
        # what the core returns: written non-CHARACTER params, then module vars
        self.core_returns = ([safe_py_name(p.name) for p in self.nonchar if p.is_output]
                             + self.mvars)
        # what the bridge returns: written params INCLUDING character
        # passthroughs (arg order), then module vars
        self.bridge_returns = ([safe_py_name(p.name) for p in self.params if p.is_output]
                               + self.mvars)
        self.bridge_file = BRIDGE_DIR / f"{self.proc}_bridge.py"

    @property
    def py(self):
        return {p.name: safe_py_name(p.name) for p in self.params}


def _load_contracts():
    gen = ph3.BridgeGenerator(
        phase1_index=str(_cfg.phase1_index),
        packets_dir=str(_cfg.packets_dir),
        jax_dir=str(_cfg.jax_dir),
        bridge_dir=str(_cfg.bridge_dir),
        reports_dir=str(_cfg.bridge_reports_dir),
        optional_kwargs=_cfg.section("bridge").get("optional_kwargs", {}),
    )
    contracts = {}
    for pf in sorted(Path(_cfg.packets_dir).glob("*_merged.json")):
        c = Contract(gen, pf)
        if c.bridge_file.exists():
            contracts[c.proc] = c
    return contracts


CONTRACTS = _load_contracts()
ARRAY_PROCS = sorted(p for p, c in CONTRACTS.items() if c.has_arrays)
SCALAR_PROCS = sorted(p for p, c in CONTRACTS.items() if not c.has_arrays)


def _import_line(contract):
    """(module name, symbol, alias) of the bridge's `from out.jax.X import Y [as Z]`."""
    src = contract.bridge_file.read_text(encoding="utf-8")
    m = re.search(r"^from out\.jax\.(\w+) import (\w+)(?: as (\w+))?", src, re.M)
    assert m, f"{contract.proc}: no `from out.jax.* import *` line in the bridge"
    return m.group(1), m.group(2), m.group(3) or m.group(2)


# ---------------------------------------------------------------------------
# Input builder (unique values everywhere) and the fake kernels
# ---------------------------------------------------------------------------

# Integer arguments that mean "size of an axis" when the array declarations do
# not say so (assumed-shape `(:,:)`). Three sources, in this order: the
# project's [heuristics].dim_names, its [profiler].ncol_args ("scalars meaning
# the column count"), and these common grid-size names. A project whose size
# arguments have other names lists them in dim_names.
_SIZE_NAME_DEFAULTS = frozenset({
    "ncol", "ncols", "ncolumns", "nz", "nlev", "nlevs", "nlevels", "nk", "nkm1",
    "ni", "nj", "nx", "ny", "ncat", "n_icecat", "pcols", "pver", "pverp",
})
_NCOL_ARGS = frozenset(str(n).lower() for n in _cfg.section("profiler").get("ncol_args", []) or [])


def _is_size_name(name: str) -> bool:
    n = name.lower()
    return _cfg.is_dim_name(n) or n in _NCOL_ARGS or n in _SIZE_NAME_DEFAULTS


def _integer_scalar_values(contract) -> dict:
    """Values for the integer scalar arguments, consistent with the array
    extents so a REAL kernel (the device-entry test reuses this builder) does
    not index out of bounds:

      * an integer named in an array's declared dimensions (phase-1
        `dimensions`, e.g. `its:ite` -> its=1, ite=extent; `ncat` -> extent)
        gets that axis' extent;
      * an integer that names an axis size (_is_size_name: the project's
        [heuristics].dim_names, [profiler].ncol_args, or the common grid-size
        names in _SIZE_NAME_DEFAULTS) gets the widest array's extents in
        order of appearance — Fortran index order, so (ncol, nz) binds
        ncol -> axis 0, nz -> axis 1 (assumed-shape arrays declare `(:,:)`,
        so there is nothing better to bind to);
      * every other integer cycles through 1..min(extent) (index-like
        arguments stay inside every array; distinct within a procedure, so a
        swapped pair is still detectable).

    The layout tests never compute on these values; the rule only keeps
    arbitrary kernels executable on synthetic inputs."""
    ints = [p for p in contract.params if p.rank == 0 and p.dtype == "integer"]
    arrays = [p for p in contract.params if p.rank > 0]
    values = {}
    for a in arrays:
        shape = _shape(a.rank)
        for ax, tok in enumerate(a.dimensions or []):
            tok = str(tok).strip().lower()
            if ":" in tok:
                lo, _, hi = (t.strip() for t in tok.partition(":"))
                if lo.isidentifier():
                    values.setdefault(lo, 1)
                if hi.isidentifier():
                    values.setdefault(hi, shape[ax])
            elif tok.isidentifier():
                values.setdefault(tok, shape[ax])
    if arrays:
        widest = max(arrays, key=lambda a: a.rank)
        extents = list(_shape(widest.rank))
        min_extent = min(min(_shape(a.rank)) for a in arrays)
    else:
        extents, min_extent = [], 0
    k = 0
    for p in ints:
        if p.name.lower() in values:
            continue
        if extents and _is_size_name(p.name):
            values[p.name.lower()] = extents[min(k, len(extents) - 1)]
            k += 1
    j = 0
    for p in ints:
        if p.name.lower() in values:
            continue
        values[p.name.lower()] = (j % min_extent) + 1 if min_extent else 10 + j
        j += 1
    return values


def _make_inputs(contract, rng_seed=42):
    """One distinct value per parameter: arrays random float64 with a
    per-parameter offset, real scalars distinct per position, integer scalars
    per _integer_scalar_values — a swapped argument cannot cancel out. The
    layout tests never compute on these values (identity kernel); the
    device-entry test runs the real kernel on them, which is why the integer
    rule keeps indices inside the arrays."""
    rng = np.random.default_rng(rng_seed)
    int_values = _integer_scalar_values(contract)
    inputs = {}
    for i, p in enumerate(contract.params):
        n = safe_py_name(p.name)
        if p.dtype == "character":
            inputs[n] = f"str_{n}"
        elif p.rank == 0:
            if p.dtype == "integer":
                inputs[n] = int_values[p.name.lower()]
            elif p.dtype == "logical":
                inputs[n] = bool(i % 2)
            else:
                inputs[n] = 100.0 + i          # real scalar, traced
        else:
            inputs[n] = rng.random(_shape(p.rank)) + 10.0 * i
    for i, v in enumerate(contract.mvars):
        inputs[v] = 1000.0 + i
    return inputs


class _Trace:
    """What the fake core saw at trace time."""
    def __init__(self):
        self.args = {}      # name -> {"shape","dtype"} (tracer) or {"value"} (static)
        self.kwargs = {}    # optional kwargs -> shape/dtype or None


def _make_fake_core(contract, trace):
    """Pass-through core for array-bearing bridges. Called INSIDE the jitted
    device wrapper, so it must stay pure and traceable: it records tracer
    shapes/dtypes and the concrete jit-static values at trace time only."""
    def fake_core(*args, **kw):
        assert len(args) == len(contract.core_positional), (
            f"{contract.proc}: core called with {len(args)} positional args, "
            f"contract says {len(contract.core_positional)}")
        named = dict(zip(contract.core_positional, args))
        trace.args.clear()
        for name, val in named.items():
            if hasattr(val, "shape"):
                trace.args[name] = {"shape": tuple(val.shape), "dtype": str(val.dtype)}
            else:
                trace.args[name] = {"value": val}
        trace.kwargs.clear()
        for name, val in kw.items():
            trace.kwargs[name] = ({"shape": tuple(val.shape), "dtype": str(val.dtype)}
                                  if hasattr(val, "shape") else val)
        rets = tuple(named[n] for n in contract.core_returns)
        return rets if len(rets) != 1 else rets[0]
    return fake_core


def _make_fake_wrapper(contract, trace):
    """Scalar-only bridges call the translated WRAPPER directly (host code)."""
    def fake_wrapper(*args, **kw):
        names = [safe_py_name(p.name) for p in contract.params] + contract.mvars
        named = dict(zip(names, args))
        named.update(kw)
        trace.args.clear()
        trace.args.update({k: {"value": v} for k, v in named.items()})
        rets = tuple(named[n] for n in contract.bridge_returns)
        return rets if len(rets) != 1 else rets[0]
    return fake_wrapper


def _bridge_module(contract, fake_attr_value, symbol, modname):
    """Plant the fake translated module, (re)import the bridge, return it."""
    sys.modules.pop(f"out.bridge.{contract.proc}_bridge", None)
    sys.modules.pop(f"out.jax.{modname}", None)
    fake = types.ModuleType(f"out.jax.{modname}")
    setattr(fake, symbol, fake_attr_value)
    sys.modules[f"out.jax.{modname}"] = fake
    return importlib.import_module(f"out.bridge.{contract.proc}_bridge")


@pytest.fixture
def array_bridge(request):
    """(bridge module, contract, trace) for one array-bearing procedure."""
    contract = CONTRACTS[request.param]
    modname, symbol, _alias = _import_line(contract)
    assert symbol.endswith("_core"), f"{contract.proc}: array bridge must import *_core, got {symbol}"
    trace = _Trace()
    mod = _bridge_module(contract, _make_fake_core(contract, trace), symbol, modname)
    yield mod, contract, trace
    sys.modules.pop(f"out.jax.{modname}", None)
    sys.modules.pop(f"out.bridge.{contract.proc}_bridge", None)


@pytest.fixture
def scalar_bridge(request):
    contract = CONTRACTS[request.param]
    modname, symbol, _alias = _import_line(contract)
    trace = _Trace()
    mod = _bridge_module(contract, _make_fake_wrapper(contract, trace), symbol, modname)
    yield mod, contract, trace
    sys.modules.pop(f"out.jax.{modname}", None)
    sys.modules.pop(f"out.bridge.{contract.proc}_bridge", None)


def _as_tuple(r):
    if r is None:
        return ()          # bridges with no written params and no module vars return None
    return r if isinstance(r, tuple) else (r,)


# ---------------------------------------------------------------------------
# 0. Coverage self-checks
# ---------------------------------------------------------------------------

def test_bridges_found():
    assert CONTRACTS, (
        f"no generated bridges with packets under {BRIDGE_DIR} — run "
        "`python workflow_bridge/phase03_make_bridge.py` first")


def test_every_generated_bridge_has_a_contract():
    """Every out/bridge/*_bridge.py must map to a packet (and vice versa)."""
    on_disk = {f.name[:-len("_bridge.py")] for f in BRIDGE_DIR.glob("*_bridge.py")}
    assert on_disk == set(CONTRACTS), (
        f"bridges without packet: {on_disk - set(CONTRACTS)}; "
        f"packets without bridge: {set(CONTRACTS) - on_disk}")


def test_partition_array_vs_scalar():
    assert len(ARRAY_PROCS) + len(SCALAR_PROCS) == len(CONTRACTS)


# ---------------------------------------------------------------------------
# 1. Array-bearing bridges (PATH C + two contracts)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("array_bridge", ARRAY_PROCS, indirect=True)
class TestArrayBridge:

    def test_transfer_helpers(self, array_bridge):
        """to_device ships the array AS-IS (same shape, same values, float64 —
        no host-side permute); to_host(to_device(x)) is bit-exact, C-order."""
        mod, c, _ = array_bridge
        to_device, to_host = getattr(mod, "to_device", None), getattr(mod, "to_host", None)
        assert callable(to_device) and callable(to_host), f"{c.proc}: transfer helpers missing"
        for x in (np.arange(5.0) + 7.0, np.arange(12.0).reshape(4, 3) + 100.0):
            d = to_device(x)
            assert tuple(d.shape) == x.shape and str(d.dtype) == "float64"
            np.testing.assert_array_equal(np.asarray(d), x)
            back = to_host(d)
            assert isinstance(back, np.ndarray) and back.dtype == np.float64
            assert back.flags["C_CONTIGUOUS"]
            np.testing.assert_array_equal(back, x)

    def test_two_entries_and_alias(self, array_bridge):
        mod, c, _ = array_bridge
        host = getattr(mod, f"{c.proc}_bridge", None)
        dev = getattr(mod, f"{c.proc}_bridge_device", None)
        assert callable(host) and callable(dev), f"{c.proc}: missing a bridge entry"
        assert getattr(mod, f"_{c.proc}_device") is dev, "private alias must point at the public entry"
        # contract 2 signature = contract 1 minus CHARACTER params (+ optional kwargs)
        host_params = list(inspect.signature(host).parameters)
        dev_params = list(inspect.signature(dev).parameters)
        assert host_params == [c.py[p.name] for p in c.params] + c.mvars + c.optional
        assert dev_params == [c.py[p.name] for p in c.nonchar] + c.mvars + c.optional

    def test_static_argnames_match_rule(self, array_bridge):
        """Statics = integer/logical scalars minus per-step counters."""
        mod, c, _ = array_bridge
        src = c.bridge_file.read_text(encoding="utf-8")
        m = re.search(r"static_argnames=\((.*?)\)\)\n", src, re.S)
        declared = [s.strip("' ") for s in m.group(1).split(",") if s.strip("' ")] if m else []
        assert declared == c.statics, f"{c.proc}: statics {declared} != rule {c.statics}"
        for n in c.per_step:
            assert n not in declared, f"{c.proc}: per-step scalar {n} must not be jit-static"

    def test_passthrough_roundtrip_identity(self, array_bridge):
        mod, c, _ = array_bridge
        inputs = _make_inputs(c)
        result = _as_tuple(getattr(mod, f"{c.proc}_bridge")(**inputs))
        assert len(result) == len(c.bridge_returns), (
            f"{c.proc}: {len(result)} return slots, contract {len(c.bridge_returns)}")
        for idx, name in enumerate(c.bridge_returns):
            got, ref = result[idx], inputs[name]
            if isinstance(ref, str):
                assert got == ref, f"{c.proc} slot {idx} ({name}): CHARACTER not passed through"
            else:
                np.testing.assert_array_equal(
                    np.asarray(got), np.asarray(ref),
                    err_msg=f"{c.proc} slot {idx} ({name}): output != input under identity core")

    def test_outputs_c_order_and_dtype(self, array_bridge):
        mod, c, _ = array_bridge
        inputs = _make_inputs(c)
        result = _as_tuple(getattr(mod, f"{c.proc}_bridge")(**inputs))
        for idx, name in enumerate(c.bridge_returns):
            ref = inputs[name]
            if not isinstance(ref, np.ndarray):
                if name not in c.mvars and not isinstance(ref, str):
                    # phase03 emits `np.asarray(x_host).item()` for scalar
                    # outputs: a plain Python int/float/bool, never a 0-d
                    # NumPy scalar or a jax.Array (mutation check 2026-10-09:
                    # a dropped .item() leaked np.int64 past a weaker check).
                    got = result[idx]
                    assert isinstance(got, (int, float, bool)) and \
                        not isinstance(got, (np.generic, jax.Array)), (
                        f"{c.proc}.{name}: scalar output must be a host Python "
                        f"value (.item()), got {type(got).__name__}")
                continue
            arr = result[idx]
            assert isinstance(arr, np.ndarray), f"{c.proc}.{name}: not NumPy"
            assert arr.shape == ref.shape and arr.dtype == np.float64
            if ref.ndim >= 2:
                assert arr.flags["C_CONTIGUOUS"], f"{c.proc}.{name}: not C-contiguous"

    def test_core_sees_reversed_layout(self, array_bridge):
        mod, c, tr = array_bridge
        inputs = _make_inputs(c)
        getattr(mod, f"{c.proc}_bridge")(**inputs)
        for p in c.nonchar:
            n = c.py[p.name]
            if p.rank == 0:
                continue
            seen = tr.args[n]["shape"]
            want = tuple(reversed(_shape(p.rank)))
            assert seen == want, (
                f"{c.proc}.{n}: core saw {seen}, expected all axes reversed {want}")
            assert tr.args[n]["dtype"] == "float64"

    def test_static_scalars_concrete_and_others_traced(self, array_bridge):
        mod, c, tr = array_bridge
        inputs = _make_inputs(c)
        getattr(mod, f"{c.proc}_bridge")(**inputs)
        for n in c.statics:
            assert "value" in tr.args[n], f"{c.proc}.{n}: static reached the core as a tracer"
            assert tr.args[n]["value"] == inputs[n]
        for p in c.nonchar:
            n = c.py[p.name]
            if p.rank == 0 and n not in c.statics:
                assert "shape" in tr.args[n], (
                    f"{c.proc}.{n}: reached the core as a jit-static; real scalars and "
                    f"per-step counters must stay traced")
        for v in c.mvars:
            assert "shape" in tr.args[v], f"{c.proc}: module var {v} must be traced"

    def test_character_args_stay_host_side(self, array_bridge):
        """CHARACTER args never enter the jitted region and pass through the
        bridge unchanged in their return slots. A procedure with no CHARACTER
        args passes trivially (a pass, not a skip — the gate forbids skips)."""
        mod, c, tr = array_bridge
        if not c.char:
            return
        inputs = _make_inputs(c)
        result = _as_tuple(getattr(mod, f"{c.proc}_bridge")(**inputs))
        for n in c.char:
            assert n not in tr.args, f"{c.proc}.{n}: CHARACTER arg reached the jitted core"
            if n in c.bridge_returns:
                assert result[c.bridge_returns.index(n)] == inputs[n]

    def test_optional_kwargs_forwarded(self, array_bridge):
        mod, c, tr = array_bridge
        if not c.optional:
            return
        inputs = _make_inputs(c)
        first = c.optional[0]
        table = np.arange(6.0).reshape(2, 3)
        getattr(mod, f"{c.proc}_bridge")(**inputs, **{first: table})
        assert set(tr.kwargs) == set(c.optional), f"{c.proc}: optional kwargs not all forwarded"
        assert tr.kwargs[first]["shape"] == (2, 3)
        for n in c.optional[1:]:
            assert tr.kwargs[n] is None, f"{c.proc}.{n}: expected None (stub mode)"


# ---------------------------------------------------------------------------
# 2. Scalar-only bridges (direct call to the translated wrapper)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scalar_bridge", SCALAR_PROCS, indirect=True)
class TestScalarBridge:

    def test_no_device_entry_and_no_transfers(self, scalar_bridge):
        mod, c, _ = scalar_bridge
        assert not hasattr(mod, f"{c.proc}_bridge_device"), f"{c.proc}: scalar-only bridge must not emit a device entry"
        src = c.bridge_file.read_text(encoding="utf-8")
        assert "to_device(" not in src and "jax.jit" not in src

    def test_wrapper_called_and_returns_passed_through(self, scalar_bridge):
        mod, c, tr = scalar_bridge
        inputs = _make_inputs(c)
        result = _as_tuple(getattr(mod, f"{c.proc}_bridge")(**inputs))
        expect_names = [c.py[p.name] for p in c.params] + c.mvars
        assert set(tr.args) == set(expect_names), f"{c.proc}: wrapper did not receive every argument"
        for n in expect_names:
            assert tr.args[n]["value"] == inputs[n], f"{c.proc}.{n}: value changed on the way to the wrapper"
        assert len(result) == len(c.bridge_returns)
        for idx, name in enumerate(c.bridge_returns):
            assert result[idx] == inputs[name], f"{c.proc} slot {idx} ({name}): not passed through"
