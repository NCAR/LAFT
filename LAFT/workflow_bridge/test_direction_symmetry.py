"""
Translation-DEPENDENT bridge test — vertical symmetry of direction-parameterised
procedures (LAFT semantic-audit item 1b, 2026-08-25). Template shipped in
LAFT/workflow_bridge/; copied unchanged into <project>/bridge_test/ by the
NEW_PROJECT_CHECKLIST for suites that provide the test_all_bridges_layout.py
helpers imported below (the kessler suite does not). Runs with the rest of
the suite:

    python workflow_bridge/run_bridge_tests.py --require-dependent   (CPU PBS job)

What it checks
--------------
A Fortran procedure that loops `do k = A, B, ±dir` with an IDENTIFIER stride
(e.g. `do k = k_qxtop, k_qxbot, -kdir`) must give the same physics when the
column is flipped and the direction reversed: for every array argument indexed
by the loop variable, reverse that axis; map the two bound arguments
A' = nk+1-A, B' = nk+1-B; negate the direction argument; call the host bridge;
flip the array outputs back — they must equal the outputs of the unflipped
call. Any orientation bug (mask sign, roll direction, top/bottom flux
handling) breaks this equality; the 08-25 sedimentation bug would have.

Self-selecting — no per-project configuration
---------------------------------------------
Cases are derived from the Fortran (out/procedures/<proc>.F90) and the bridge
contract: a procedure qualifies iff it has an identifier-stride loop AND the
stride identifier and both bounds are integer scalar arguments AND every
written output is an array with a level axis or a real scalar (integer level
indices as outputs — find_top's k_qxtop — cannot be compared by a flip and are
reported as excluded). Zero cases is NOT a skip: the applicability test always
runs and records the result in out/reports/bridge/direction_symmetry.json, so
`--require-dependent` sees a passed test, not a skipped one.

Inputs come from the project's device-entry suite (`_inputs(proc)` in
test_all_bridges_device_entry.py — physically plausible column inputs where the
project defines them), falling back to the generic layout builder.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from test_all_bridges_layout import CONTRACTS, _make_inputs, _import_line  # noqa: E402

try:
    from test_all_bridges_device_entry import _inputs as _project_inputs, _real_bridge  # noqa: E402
except Exception:  # noqa: BLE001 — generic fallback when the project has no device-entry suite
    _project_inputs = None

    def _real_bridge(proc):
        import importlib
        return importlib.import_module(f"out.bridge.{proc}_bridge")

F90_DIR = project_root / "out" / "procedures"
REPORT = project_root / "out" / "reports" / "bridge" / "direction_symmetry.json"

_ID_STRIDE_LOOP = re.compile(
    r"^\s*do\s+(\w+)\s*=\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*,\s*([-+]?)\s*([A-Za-z_]\w*)\s*$",
    re.I | re.M)


def _fortran_text(proc):
    for p in F90_DIR.glob("*.F90"):
        if p.stem.lower() == proc.lower():
            src = p.read_text(encoding="utf-8", errors="replace")
            src = "\n".join(ln.split("!", 1)[0] for ln in src.splitlines())
            return re.sub(r"&\s*\n\s*&?", " ", src)
    return None


def _split_top(s):
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(s):
        depth += (ch == "(") - (ch == ")")
        if ch == "," and depth == 0:
            parts.append(s[start:i]); start = i + 1
    parts.append(s[start:])
    return parts


def _level_axis(f_txt, name, loopvar):
    """Fortran index position of the loop variable in references `name(...)`, or None."""
    for m in re.finditer(rf"\b{re.escape(name)}\s*\(([^()]*)\)", f_txt, re.I):
        for ax, idx in enumerate(_split_top(m.group(1))):
            if re.search(rf"\b{re.escape(loopvar)}\b", idx, re.I):
                return ax
    return None


def _discover():
    """[(proc, case)] + [(proc, reason)] for excluded identifier-stride procs."""
    cases, excluded = [], []
    for proc, c in sorted(CONTRACTS.items()):
        f_txt = _fortran_text(c.orig_name)
        if not f_txt:
            continue
        loops = _ID_STRIDE_LOOP.findall(f_txt)
        if not loops:
            continue
        var, a, b, sign, dirv = loops[0]
        by_name = {p.name.lower(): p for p in c.params}
        ints = {n for n, p in by_name.items() if p.rank == 0 and p.dtype == "integer"}
        if dirv.lower() not in ints:
            excluded.append((proc, f"direction `{dirv}` of `do {var}={a},{b},{sign}{dirv}` is not "
                                   "an integer scalar argument (local or module state)"))
            continue
        # Bounds: the loop's own bounds when they are arguments; otherwise (inner
        # loops over locals such as k_qxtop/k_qxbot) the integer arguments that
        # travel with the direction in some statement — `call find_top(..., ktop,
        # kbot, kdir)`. Order is irrelevant for the flip (both map to nk+1-x).
        bounds = {a.lower(), b.lower()} & ints
        if len(bounds) != 2:
            bounds = set()
            for stmt in re.findall(rf"^[^\n]*\b{re.escape(dirv)}\b[^\n]*$", f_txt, re.I | re.M):
                # executable statements only — the header and intent/type
                # declarations list every argument and would add e.g. liq_type
                if re.match(r"\s*(subroutine|function|integer|real|logical|character|"
                            r"double|type)\b", stmt, re.I) or "::" in stmt:
                    continue
                ids = {m.lower() for m in re.findall(r"[A-Za-z_]\w*", stmt)}
                bounds |= (ids & ints) - {dirv.lower()}
        if len(bounds) != 2:
            excluded.append((proc, f"could not identify exactly two integer bound arguments "
                                   f"travelling with `{dirv}` (found {sorted(bounds)})"))
            continue
        a, b = sorted(bounds)
        axes = {}
        for p in c.params:
            if p.rank > 0:
                ax = _level_axis(f_txt, p.name, var)
                if ax is not None:
                    axes[c.py[p.name]] = ax
        if not axes:
            excluded.append((proc, "no array argument is indexed by the loop variable"))
            continue
        bad = [c.py[p.name] for p in c.params
               if p.is_output and p.rank == 0 and p.dtype != "real"]
        if bad:
            excluded.append((proc, f"integer/logical scalar outputs {bad} are level indices or "
                                   "flags that a flip cannot map — not comparable"))
            continue
        cases.append((proc, {"loop": f"do {var} = {a}, {b}, {sign}{dirv}",
                             "start": c.py[by_name[a.lower()].name],
                             "end": c.py[by_name[b.lower()].name],
                             "dir": c.py[by_name[dirv.lower()].name],
                             "axes": axes}))
    return cases, excluded


CASES, EXCLUDED = _discover()


def _inputs(proc):
    if _project_inputs is not None:
        try:
            return _project_inputs(proc)
        except Exception:  # noqa: BLE001
            pass
    return _make_inputs(CONTRACTS[proc])


def _flip(inputs, case):
    out = dict(inputs)
    nk = None
    for name, ax in case["axes"].items():
        arr = np.asarray(inputs[name])
        nk = nk or arr.shape[ax]
        out[name] = np.flip(arr, axis=ax).copy()
    out[case["start"]] = nk + 1 - int(inputs[case["start"]])
    out[case["end"]] = nk + 1 - int(inputs[case["end"]])
    out[case["dir"]] = -int(inputs[case["dir"]])
    return out


def test_direction_symmetry_applicability():
    """Always runs: records which procedures the symmetry test covers (or n/a)."""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": "cases" if CASES else "n/a — no procedure with an identifier-stride "
                                        "level loop whose bounds and direction are arguments",
        "cases": {p: c for p, c in CASES},
        "excluded": {p: r for p, r in EXCLUDED},
    }, indent=2) + "\n", encoding="utf-8")
    assert True


if CASES:
    @pytest.mark.parametrize("proc", [p for p, _ in CASES], ids=[p for p, _ in CASES])
    def test_vertical_symmetry(proc):
        case = dict(CASES)[proc]
        c = CONTRACTS[proc]
        mod = _real_bridge(proc)
        fn = getattr(mod, f"{proc}_bridge")
        base = _inputs(proc)
        r0 = fn(**base)
        r1 = fn(**_flip(base, case))
        r0 = r0 if isinstance(r0, tuple) else (r0,)
        r1 = r1 if isinstance(r1, tuple) else (r1,)
        assert len(r0) == len(r1) == len(c.bridge_returns)
        worst = 0.0
        for name, a, b in zip(c.bridge_returns, r0, r1):
            a = np.asarray(a); b = np.asarray(b)
            if name in case["axes"]:
                b = np.flip(b, axis=case["axes"][name])
            assert a.shape == b.shape, f"{proc}.{name}: shape {a.shape} vs flipped {b.shape}"
            if a.dtype.kind in "fc":
                ok = np.allclose(a, b, rtol=1e-10, atol=0.0, equal_nan=True)
                if not ok:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        rel = np.nanmax(np.abs(a - b) / np.maximum(np.abs(a), 1e-300))
                    worst = max(worst, float(rel))
                assert ok, (f"{proc}.{name}: not symmetric under column flip + direction "
                            f"reversal ({case['loop']}); max rel diff {rel:.3e} — orientation "
                            "bug (mask sign / roll direction / boundary flux)")
            else:
                assert np.array_equal(a, b), f"{proc}.{name}: differs under flip"
