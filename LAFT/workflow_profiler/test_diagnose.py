#!/usr/bin/env python3
"""Regression tests for the single-source diagnoser stack.

Fixtures are trace-metrics + code_info dicts — the diagnoser's only
gating inputs. nsys JSON appears in exactly one test, pinning that it
is informational and cannot affect a verdict.

The two HLO fixtures under test_fixtures/ are REAL compiler output,
not synthesis:
  hlo_known_good_interior_gathers.txt  the converged production-ready
      translation's GPU dump (metadata-trimmed, counts verified
      identical) — 4 interior gathers from the prescribed
      jnp.clip/jnp.where boundary fix. Pins "the diagnoser must not
      condemn the code its own fix text produces."
  hlo_scatter_root.txt  real XLA output of a `.at[idx].set()` kernel at
      the kessler shape — the scatter lands as a fused_scatter ROOT.
      Pins "demoting gather did not weaken scatter detection."

History note: the tests for the nsys launch-signal conflict branch and
the capture_includes_warmup denominator coupling were deleted together
with the code they pinned (the single-source rewrite removed the nsys
gating path entirely, so there is nothing left to conflict with).

Severity note: unfused gather pins ADVISORY (never gates), per the
pass-2 amendment that superseded the earlier gather→HIGH instruction.

Run:  pytest workflow_profiler/test_diagnose.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from diagnose import diagnose  # noqa: E402
from parse_hlo import parse_hlo_metrics  # noqa: E402
from parse_jax_trace import compute_while_scaling, parse_trace  # noqa: E402

FIXTURES = HERE / "test_fixtures"
KNOWN_GOOD_HLO = parse_hlo_metrics(
    (FIXTURES / "hlo_known_good_interior_gathers.txt").read_text())
SCATTER_ROOT_HLO = parse_hlo_metrics(
    (FIXTURES / "hlo_scatter_root.txt").read_text())

NCOL = 1000


def make_trace(**overrides):
    """Healthy-vmap trace metrics, as profile_trace_run.py writes them."""
    t = {
        "while_iterations": 1,
        "while_iterations_raw": 1,
        "while_iterations_scaling": 1.0,
        "ncol_ratio": 4.0,
        "xla_event_count": 180,
        "unique_hlo_ops": 40,
        "d2h_copies": 12,
        "n_calls": 1,
        "ncol": NCOL,
        "hlo": KNOWN_GOOD_HLO,
    }
    t.update(overrides)
    return t


def make_code_info(**overrides):
    info = {
        "ncol": NCOL,
        "nz": 56,
        "n_calls": 1,
        "jax_op_count": 200,
        "output_array_count": 7,
    }
    info.update(overrides)
    return info


def by_metric(issues):
    return {i["metric"]: i for i in issues}


# ─────────────────────────────────────────────────────────────────────
# Report contract
# ─────────────────────────────────────────────────────────────────────

def test_report_top_level_shape_is_stable():
    """The top-level diagnosis JSON shape is a compatibility contract:
    downstream consumers and the report template read these keys."""
    report = diagnose(make_trace(), make_code_info())
    for key in ("production_ready", "worst_severity", "summary",
                "actionable_issues", "advisories", "confirmations",
                "llm_feedback"):
        assert key in report, key
    assert "code_info" in report["summary"]
    assert "issue_count" in report["summary"]


# ─────────────────────────────────────────────────────────────────────
# Column-loop structure
# ─────────────────────────────────────────────────────────────────────

def test_healthy_vmap_clean():
    """wi=1 with ncol-independent scaling and a clean HLO: clean bill."""
    report = diagnose(make_trace(), make_code_info())
    assert report["production_ready"] is True
    assert report["worst_severity"] == "ok"
    assert report["actionable_issues"] == []
    assert by_metric(report["confirmations"])["while_iterations"]


def test_sequential_single_ncol_critical():
    """wi ~ ncol with single-ncol data: the textbook defect, CRITICAL."""
    report = diagnose(
        make_trace(while_iterations=980, while_iterations_scaling=None,
                   ncol_ratio=None),
        make_code_info())
    assert report["production_ready"] is False
    issue = by_metric(report["actionable_issues"])["while_iterations"]
    assert issue["severity"] == "critical"
    assert "vmap" in issue["fix"]


def test_subcycling_single_ncol_is_advisory_not_gating():
    """12 legitimate substeps, single-ncol data: indistinguishable from
    partial sequentialization, so advisory — the fix loop must never
    burn attempts on it."""
    report = diagnose(
        make_trace(while_iterations=12, while_iterations_scaling=None,
                   ncol_ratio=None),
        make_code_info())
    assert report["production_ready"] is True
    adv = by_metric(report["advisories"])["while_iterations"]
    assert "subcycling" in adv["message"]
    assert report["summary"]["advisory_count"] >= 1


def test_subcycling_two_ncol_scaling_one_is_ok():
    """Two-ncol data, scaling ~1: ncol-independent iteration count is
    vectorized regardless of the absolute wi."""
    report = diagnose(make_trace(while_iterations=12), make_code_info())
    assert report["production_ready"] is True
    conf = by_metric(report["confirmations"])["while_iterations"]
    assert "subcycle" in conf["message"]


def test_sequential_two_ncol_scaling_is_critical():
    """Two-ncol data, scaling ~ ncol_ratio: sequential, CRITICAL."""
    report = diagnose(
        make_trace(while_iterations=1000, while_iterations_scaling=4.0),
        make_code_info())
    assert report["production_ready"] is False
    issue = by_metric(report["actionable_issues"])["while_iterations"]
    assert issue["severity"] == "critical"
    assert "scales" in issue["message"]


def test_no_while_loop_strongest_ok_requires_hlo_agreement():
    """wi=None + hlo.ops.while == 0: both instruments agree there is no
    loop — the strongest vectorization signal."""
    hlo = json.loads(json.dumps(KNOWN_GOOD_HLO))
    hlo["ops"]["while"] = 0
    report = diagnose(
        make_trace(while_iterations=None, while_iterations_scaling=None,
                   hlo=hlo),
        make_code_info())
    assert report["production_ready"] is True
    conf = by_metric(report["confirmations"])["while_iterations"]
    assert "both instruments agree" in conf["message"]
    assert conf["evidence"]["hlo_while_ops"] == 0


def test_wi_none_hlo_while_mismatch_is_advisory():
    """hlo.ops.while=3 but trace wi=None: a trace-parser regression
    (e.g. renamed while.<N> events) must NOT become a silent false OK —
    it surfaces as a mismatch advisory naming WHILE_RE."""
    hlo = json.loads(json.dumps(KNOWN_GOOD_HLO))
    hlo["ops"]["while"] = 3
    report = diagnose(
        make_trace(while_iterations=None, while_iterations_scaling=None,
                   hlo=hlo),
        make_code_info())
    adv = by_metric(report["advisories"])["while_iterations"]
    assert "mismatch" in adv["message"].lower()
    assert "WHILE_RE" in adv["message"]
    assert "while_iterations" not in by_metric(report["confirmations"])


def test_wi_none_without_hlo_block_is_advisory():
    """wi=None with no hlo block: cannot distinguish loop-free from
    parser failure — no strongest-OK."""
    t = make_trace(while_iterations=None, while_iterations_scaling=None)
    del t["hlo"]
    report = diagnose(t, make_code_info())
    assert "while_iterations" in by_metric(report["advisories"])
    assert "while_iterations" not in by_metric(report["confirmations"])


def test_xla_event_proxy_is_weak_advisory():
    """Device events scaling like ncol: a weak hint, advisory only."""
    report = diagnose(
        make_trace(xla_event_count=3 * NCOL), make_code_info())
    assert report["production_ready"] is True
    assert "xla_events_per_column" in by_metric(report["advisories"])


# ─────────────────────────────────────────────────────────────────────
# Fusion quality (HLO block)
# ─────────────────────────────────────────────────────────────────────

def test_scatter_in_hlo_gates_high():
    """Real .at[].set() HLO: the scatter is a fused_scatter ROOT and
    must gate HIGH on the TOTAL count — wrapping does not hide it."""
    report = diagnose(make_trace(hlo=SCATTER_ROOT_HLO), make_code_info())
    assert report["production_ready"] is False
    assert report["worst_severity"] == "high"
    issue = by_metric(report["actionable_issues"])["scatter_ops"]
    assert issue["evidence"]["scatter_total"] == 1
    assert issue["evidence"]["scatter_as_fusion_roots"] == 1


def test_unfused_gather_is_advisory_not_gating():
    """Unfused gathers (entry-level/fusion-root) are ADVISORY with full
    fix text — never gating. (The pass-2 amendment demoted gather from
    the originally specified HIGH: irregular reads are cheap relative
    to scatter's writes, and an unfused gather cannot always name a
    fixable defect.)"""
    hlo = json.loads(json.dumps(KNOWN_GOOD_HLO))
    hlo["ops"]["gather"] = 6  # 2 entry-level on top of the 4 interior
    report = diagnose(make_trace(hlo=hlo), make_code_info())
    assert report["production_ready"] is True
    adv = by_metric(report["advisories"])["gather_ops"]
    assert adv["evidence"]["gather_gating"] == 2
    assert "UNFUSED" in adv["fix"]


def test_clean_hlo_interior_gathers_confirm_ok():
    """The converged known-good dump: 4 interior gathers (the product
    of the prescribed clip/where fix) confirm OK with a count, scatter
    confirms zero — production_ready stays True."""
    report = diagnose(make_trace(), make_code_info())
    assert report["production_ready"] is True
    confs = by_metric(report["confirmations"])
    assert confs["gather_ops"]["evidence"]["gather_interior_absorbed"] == 4
    assert confs["scatter_ops"]["evidence"]["scatter_total"] == 0


def test_ops_per_fusion_real_fixtures():
    """HLO-grounded ops-per-fusion: arithmetic pinned against both real
    dumps, surfaced in the diagnosis as a REPORT-ONLY OK entry (the
    AST-based fusion_efficiency ratio is retired)."""
    # known-good dump: 424 instructions in fusion bodies / 24 fusions
    assert KNOWN_GOOD_HLO["ops_per_fusion"] == 17.67
    assert SCATTER_ROOT_HLO["ops_per_fusion"] == round(
        SCATTER_ROOT_HLO["instructions_in_fusion_bodies"]
        / SCATTER_ROOT_HLO["ops"]["fusion"], 2)

    report = diagnose(make_trace(), make_code_info())
    char = by_metric(report["confirmations"])["fusion_characterization"]
    assert char["evidence"]["ops_per_fusion"] == 17.67
    assert char["evidence"]["fusion_count"] == 24
    assert "Report-only" in char["message"]
    # the retired AST proxy must not appear anywhere
    everything = (report["actionable_issues"] + report["advisories"]
                  + report["confirmations"])
    assert "fusion_efficiency" not in by_metric(everything)


def test_no_fusion_hlo_ops_per_fusion_is_none():
    """Zero fusion ops: the metric is None (no division), the
    characterization reports the total instruction count, and nothing
    gates."""
    no_fusion_hlo = parse_hlo_metrics(
        "ENTRY %main.1 (a: f64[8]) -> f64[8] {\n"
        "  %a = f64[8]{0} parameter(0)\n"
        "  ROOT %m = f64[8]{0} multiply(%a, %a)\n"
        "}\n")
    assert no_fusion_hlo["ops_per_fusion"] is None
    report = diagnose(make_trace(hlo=no_fusion_hlo), make_code_info())
    assert report["production_ready"] is True
    char = by_metric(report["confirmations"])["fusion_characterization"]
    assert char["evidence"]["ops_per_fusion"] is None
    assert char["evidence"]["total_instructions"] == 2


def test_missing_hlo_block_is_advisory():
    """A pre-HLO trace JSON: scatter/gather detection cannot silently
    vanish — an advisory says to re-run the trace pass."""
    t = make_trace()
    del t["hlo"]
    report = diagnose(t, make_code_info())
    adv = by_metric(report["advisories"])["hlo_block"]
    assert "Re-run the trace pass" in adv["message"]


# ─────────────────────────────────────────────────────────────────────
# D2H copies (trace-sourced)
# ─────────────────────────────────────────────────────────────────────

def test_d2h_excess_fires_low_normalized_per_trace_call():
    """d2h is cumulative over the trace's own n_calls (exact window,
    no warmup): 400 events over 2 calls = 200/call >> budget -> LOW."""
    report = diagnose(
        make_trace(d2h_copies=400, n_calls=2), make_code_info())
    assert report["production_ready"] is False
    issue = by_metric(report["actionable_issues"])["d2h_copies"]
    assert issue["severity"] == "low"
    assert issue["evidence"]["d2h_per_call"] == 200.0


def test_d2h_normal_confirms_ok():
    """A copy count consistent with reading back the outputs once per
    call confirms OK and does not gate."""
    report = diagnose(make_trace(d2h_copies=12), make_code_info())
    assert report["production_ready"] is True
    conf = by_metric(report["confirmations"])["d2h_copies"]
    assert conf["evidence"]["d2h_per_call"] == 12.0


# ─────────────────────────────────────────────────────────────────────
# nsys is informational only
# ─────────────────────────────────────────────────────────────────────

def test_nsys_json_never_affects_verdict():
    """Absurd hardware numbers attach as hardware_corroboration and
    leave the verdict untouched — the single-source guarantee."""
    nsys = {
        "cuda_gpu_kern_sum": [
            {"name": "k", "instances": 10**6, "total_time_ns": 5_000_000}],
        "cuda_api_sum": [
            {"name": "cuStreamSynchronize", "num_calls": 100,
             "total_time_ns": 4_000_000}],
    }
    report = diagnose(make_trace(), make_code_info(), nsys_json=nsys)
    assert report["production_ready"] is True
    hw = report["hardware_corroboration"]
    assert hw["kernel_launches"] == 10**6
    assert hw["sync_avg_us"] == 40.0
    assert "informational" in hw["note"]


# ─────────────────────────────────────────────────────────────────────
# CLI exit codes
# ─────────────────────────────────────────────────────────────────────

def _run_cli(trace, code_info):
    with tempfile.TemporaryDirectory() as td:
        tp, cp = Path(td) / "trace.json", Path(td) / "code_info.json"
        tp.write_text(json.dumps(trace))
        cp.write_text(json.dumps(code_info))
        return subprocess.run(
            [sys.executable, str(HERE / "diagnose.py"),
             "--trace-json", str(tp), "--code-info", str(cp)],
            capture_output=True, text=True)


def test_advisory_only_run_exits_zero():
    """Advisories never gate: a run whose only findings are advisories
    must exit 0 so the orchestrator does not trigger a fix pass."""
    trace = make_trace(while_iterations=12, while_iterations_scaling=None,
                       ncol_ratio=None)
    proc = _run_cli(trace, make_code_info())
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["production_ready"] is True
    assert report["summary"]["advisory_count"] >= 1


def test_actionable_run_exits_one():
    proc = _run_cli(make_trace(hlo=SCATTER_ROOT_HLO), make_code_info())
    assert proc.returncode == 1


# ─────────────────────────────────────────────────────────────────────
# HLO parser (synthetic text fixture)
# ─────────────────────────────────────────────────────────────────────

SYNTHETIC_HLO = """HloModule jit_test, entry_computation_layout={(f64[56,1000]{1,0})->f64[56,1000]{1,0}}

%fused_scatter.2 (p0: f64[56,1000], p1: s32[10]) -> f64[56,1000] {
  %p0 = f64[56,1000]{1,0} parameter(0)
  %p1 = s32[10]{0} parameter(1)
  ROOT %scatter.3 = f64[56,1000]{1,0} scatter(%p0, %p1, %p1), update_window_dims={}
}

%fused_computation.1 (p2: f64[56,1000]) -> f64[56,1000] {
  %p2 = f64[56,1000]{1,0} parameter(0)
  %gather.7 = f64[10,1000]{1,0} gather(%p2, %p2), offset_dims={1}
  ROOT %multiply.9 = f64[56,1000]{1,0} multiply(%gather.7, %p2)
}

%body.5 (arg.1: (s32[], f64[56,1000])) -> (s32[], f64[56,1000]) {
  %arg.1 = (s32[], f64[56,1000]{1,0}) parameter(0)
  %gte.1 = s32[] get-tuple-element(%arg.1), index=0
  %gather.9 = f64[10,1000]{1,0} gather(%gte.1, %gte.1), offset_dims={1}
  ROOT %tuple.2 = (s32[], f64[56,1000]{1,0}) tuple(%gte.1, %gte.1)
}

ENTRY %main.1 (a: f64[56,1000]) -> f64[56,1000] {
  %a = f64[56,1000]{1,0} parameter(0)
  %c = f64[] constant(inf)
  %fusion.1 = f64[56,1000]{1,0} fusion(%a), kind=kInput, calls=%fused_scatter.2
  %while.11 = (s32[], f64[56,1000]{1,0}) while(%a), condition=%cond.4, body=%body.5
  ROOT %fusion.2 = f64[56,1000]{1,0} fusion(%a), kind=kLoop, calls=%fused_computation.1
}
"""


def test_parse_hlo_counts_synthetic():
    """Instruction-definition counting with the trap cases: 'scatter' in
    a computation NAME is not an op; tuple shapes parse; root-vs-interior
    split; while bodies are not fusion bodies."""
    m = parse_hlo_metrics(SYNTHETIC_HLO)
    assert m["ops"] == {"fusion": 2, "scatter": 1, "gather": 2, "while": 1}
    # the scatter is a fusion ROOT; the fused gather is interior; the
    # while-body gather is neither (regions are not fusion bodies)
    assert m["ops_as_fusion_roots"]["scatter"] == 1
    assert m["ops_as_fusion_roots"]["gather"] == 0
    assert m["ops_in_fusion_bodies"]["gather"] == 1
    assert m["ops_in_fusion_bodies"]["scatter"] == 1
    # 3 + 3 instructions in the two fused computations
    assert m["instructions_in_fusion_bodies"] == 6
    assert m["total_instructions"] == 15


def test_parse_hlo_real_fixtures():
    """The committed real fixtures keep their verified counts."""
    assert KNOWN_GOOD_HLO["ops"] == {"fusion": 24, "scatter": 0,
                                     "gather": 4, "while": 1}
    assert KNOWN_GOOD_HLO["ops_in_fusion_bodies"]["gather"] == 4
    assert SCATTER_ROOT_HLO["ops"]["scatter"] == 1
    assert SCATTER_ROOT_HLO["ops_as_fusion_roots"]["scatter"] == 1


# ─────────────────────────────────────────────────────────────────────
# Per-call timing pairing (Bug #3)
# ─────────────────────────────────────────────────────────────────────

def _x(name, ts, dur, pid=2):
    return {"ph": "X", "name": name, "ts": ts, "dur": dur, "pid": pid}


def _parse_events(events, n_calls):
    base = [{"ph": "M", "name": "process_name", "pid": 1,
             "args": {"name": "/device:GPU:0"}}]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "host.trace.json"
        p.write_text(json.dumps({"traceEvents": base + events}))
        return parse_trace(p, n_calls=n_calls)


def test_timing_dispatch_is_per_call_not_max_of_maxes():
    """REGRESSION PIN for the original bug: max e2e (call 0) and max gpu
    (call 1) come from different calls. The old independent-max logic
    yielded dispatch = 2.0 - 0.8 = 1.2 ms; per-call pairing must yield
    [1.7, 0.2] -> median 0.95 ms."""
    events = [
        _x("PjitFunction(kessler_run_core)", ts=0, dur=2000),
        _x("ExecuteThunks", ts=100, dur=300),
        _x("PjitFunction(kessler_run_core)", ts=5000, dur=1000),
        _x("ExecuteThunks", ts=5100, dur=800),
    ]
    m = _parse_events(events, n_calls=2)
    pc = m["timing"]["per_call"]
    assert pc["dispatch_overhead_ms"] == [1.7, 0.2]
    assert m["dispatch_overhead_ms"] == 0.95
    assert m["end_to_end_ms"] == 1.5
    assert m["gpu_exec_ms"] == 0.55
    old_logic_dispatch = round((2000 - 800) / 1000.0, 4)  # 1.2 ms
    assert m["dispatch_overhead_ms"] != old_logic_dispatch
    assert m["warnings"] == []


def test_timing_pairing_selects_nested_spans_only():
    """A gpu span outside every end-to-end window is ignored, and an
    inner jit(...) sub-span must not count as a second call."""
    events = [
        _x("PjitFunction(kessler_run_core)", ts=0, dur=1000),
        _x("jit(inner_subspan)", ts=50, dur=500),     # nested candidate
        _x("ExecuteThunks", ts=100, dur=200),          # nested -> paired
        _x("ExecuteThunks", ts=2000, dur=900),         # stray -> ignored
    ]
    m = _parse_events(events, n_calls=1)
    assert m["timing"]["per_call"]["gpu_exec_ms"] == [0.2]
    assert m["timing"]["end_to_end_ms"]["n"] == 1
    assert m["warnings"] == []


def test_timing_median_std_n():
    events = []
    for k, (e2e, gpu) in enumerate([(1000, 400), (2000, 400), (3000, 400)]):
        t0 = k * 10_000
        events += [_x("PjitFunction(f)", ts=t0, dur=e2e),
                   _x("ExecuteThunks", ts=t0 + 50, dur=gpu)]
    m = _parse_events(events, n_calls=3)
    stats = m["timing"]["end_to_end_ms"]
    assert (stats["median"], stats["std"], stats["n"]) == (2.0, 1.0, 3)
    assert m["timing"]["gpu_exec_ms"]["std"] == 0.0
    assert m["dispatch_overhead_ms"] == 1.6  # median of [0.6, 1.6, 2.6]


def test_timing_jit_pattern_matches():
    """Newer event naming ('jit(...)' instead of 'PjitFunction') pairs."""
    events = [_x("jit(kessler_run_core)", ts=0, dur=1000),
              _x("ExecuteThunks", ts=100, dur=400)]
    m = _parse_events(events, n_calls=1)
    assert m["end_to_end_ms"] == 1.0
    assert m["dispatch_overhead_ms"] == 0.6


def test_timing_no_match_warns_loudly():
    """No matching events -> metrics None AND a human-readable warning;
    a silent None propagating into the diagnosis is the failure mode."""
    m = _parse_events([_x("ExecuteThunks", ts=0, dur=400)], n_calls=1)
    assert m["end_to_end_ms"] is None and m["dispatch_overhead_ms"] is None
    assert any("end-to-end" in w and "None" in w for w in m["warnings"])

    m = _parse_events([_x("PjitFunction(f)", ts=0, dur=1000)], n_calls=1)
    assert m["gpu_exec_ms"] is None and m["dispatch_overhead_ms"] is None
    assert any("ExecuteThunks" in w for w in m["warnings"])


def test_derived_characterization_metrics():
    """events_per_while_iteration and mean_event_duration_us from a
    synthetic trace with known counts: 8 device events over 2 calls,
    4 while-body executions, gpu median 0.4 ms ->
    events/iteration = 8/4 = 2.0; mean event duration =
    0.4 ms * 1000 / (8/2 events per call) = 100 us."""
    events = [
        _x("PjitFunction(f)", ts=0, dur=1000),
        _x("ExecuteThunks", ts=100, dur=400),
        _x("PjitFunction(f)", ts=5000, dur=1000),
        _x("ExecuteThunks", ts=5100, dur=400),
    ]
    events += [_x("while.0", ts=200 + i, dur=5) for i in range(4)]
    events += [_x("fusion.1", ts=300 + i, dur=10, pid=1) for i in range(8)]
    m = _parse_events(events, n_calls=2)
    assert m["xla_event_count"] == 8
    assert m["while_iterations_raw"] == 4
    assert m["events_per_while_iteration"] == 2.0
    assert m["gpu_exec_ms"] == 0.4
    assert m["mean_event_duration_us"] == 100.0

    # loop-free trace: denominator clamps to 1
    no_while = [e for e in events if e["name"] != "while.0"]
    m2 = _parse_events(no_while, n_calls=2)
    assert m2["events_per_while_iteration"] == 8.0


def test_timing_pairing_count_mismatch_warns():
    """Fewer end-to-end spans than n_calls, and a call whose window has
    no nested gpu span, both warn (the bad call is excluded, not mixed)."""
    events = [
        _x("PjitFunction(f)", ts=0, dur=1000),
        _x("ExecuteThunks", ts=100, dur=300),
        _x("PjitFunction(f)", ts=5000, dur=1000),  # no gpu span inside
    ]
    m = _parse_events(events, n_calls=3)
    assert m["timing"]["end_to_end_ms"]["n"] == 1
    assert any("expected 3" in w for w in m["warnings"])
    assert any("call 1" in w and "excluded" in w for w in m["warnings"])


# ─────────────────────────────────────────────────────────────────────
# Trace parser helpers (live code, kept from the previous suite)
# ─────────────────────────────────────────────────────────────────────

def test_parse_trace_normalizes_by_n_calls():
    """while-body executions are cumulative over the traced calls;
    parse_trace divides by n_calls and keeps the raw count alongside."""
    events = [{"ph": "M", "name": "process_name", "pid": 1,
               "args": {"name": "/device:GPU:0"}}]
    events += [{"ph": "X", "name": "while.0", "pid": 2, "dur": 5.0}
               for _ in range(24)]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "host.trace.json"
        p.write_text(json.dumps({"traceEvents": events}))
        m1 = parse_trace(p)
        m2 = parse_trace(p, n_calls=2)
        p.write_text(json.dumps(
            {"traceEvents": [e for e in events if e["name"] != "while.0"]}))
        m3 = parse_trace(p, n_calls=2)
    assert (m1["while_iterations"], m1["while_iterations_raw"]) == (24, 24)
    assert (m2["while_iterations"], m2["while_iterations_raw"]) == (12, 24)
    assert m3["while_iterations"] is None and m3["while_loops"] == {}


def test_compute_while_scaling_none_safe():
    """The scaling division must not throw on None/zero wi (loop-free
    translations) and must order hi/lo by ncol."""
    assert compute_while_scaling(1000, None, 250, None) == (None, 4.0)
    assert compute_while_scaling(1000, 12, 250, None) == (None, 4.0)
    assert compute_while_scaling(1000, None, 250, 12) == (None, 4.0)
    assert compute_while_scaling(1000, 12, 250, 0) == (None, 4.0)
    assert compute_while_scaling(1000, 12, 250, 12) == (1.0, 4.0)
    assert compute_while_scaling(1000, 1000, 250, 250) == (4.0, 4.0)
    assert compute_while_scaling(250, 250, 1000, 1000) == (4.0, 4.0)


if __name__ == "__main__":
    # plain-python fallback when pytest is unavailable
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
