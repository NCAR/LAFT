#!/usr/bin/env python3
"""
parse_jax_trace.py
─────────────────────────────────────────────────────────────────────────
Parse a jax.profiler.trace (Perfetto/XLA) trace into the structural
metrics the profiler diagnoser consumes.

jax.profiler.trace operates at the XLA/HLO graph layer — it sees how the
translation structured each loop and how XLA fused the op graph. That is
the right lens for loop *design*; nsys (the other pass) sees what the
hardware did. The two passes are complementary, not competing.
See workflow_profiler/JAX_TRACE_PROFILING_PLAN.md.

The single most important signal here is `while_iterations`: jax_trace
counts XLA `while.<N>` body executions DIRECTLY (1 = vectorized column
loop, ncol = a sequential per-column loop — the textbook vectorization
defect). This replaces the old nsys launch-ratio heuristic; the retired
infer_while_iterations.py only *inferred* it.

INPUT
  A jax.profiler.trace output directory, OR the trace JSON inside it.
  jax.profiler.trace writes:
      <trace_dir>/plugins/profile/<timestamp>/<host>.trace.json.gz
      <trace_dir>/plugins/profile/<timestamp>/<host>.xplane.pb

OUTPUT (dict / kessler_run_trace.json)
  while_iterations       max XLA while-body execution count PER CALL
                         (AUTHORITATIVE; raw count / n_calls, since body
                         executions are cumulative over the traced calls)
  while_iterations_raw   the un-normalized cumulative count
  while_loops            {while.<N>: body_execution_count} (raw/cumulative)
  xla_event_count        GPU compute-stream events — an XLA-event aggregate
                         (fused ops, call.N, wrapped_*, while.* bodies).
                         A structural-complexity proxy; deliberately NOT
                         named `kernel_launches` — it is not comparable to
                         the nsys device-kernel count of that name.
  unique_hlo_ops         distinct HLO op names among those (the nsys
                         `unique_kernels` deduplicates by compiled CUDA
                         kernel symbol instead — a different identity)
  end_to_end_ms          MEDIAN per-call host end-to-end span
                         (PjitFunction/jit(...) outer span)
  gpu_exec_ms            MEDIAN per-call GpuExecutable::ExecuteThunks
                         time, paired WITHIN the same call's window
  dispatch_overhead_ms   MEDIAN per-call (e2e - gpu_exec), computed
                         strictly from same-call pairs — never the
                         difference of spans from different calls
  timing                 distribution block: {median, std, n} for each
                         of the three metrics plus per_call lists
  warnings               human-readable parse problems (event-name
                         patterns not matching, pairing failures) —
                         timing Nones are never silent
  events_per_while_iteration   xla_event_count / max(while_iterations_raw, 1)
                         — report-only characterization separating
                         per-step structure from loop count (a
                         sequential translation can have ordinary
                         per-step code; the defect is the outer loop)
  mean_event_duration_us gpu_exec_ms * 1000 / (xla_event_count per
                         call) — report-only; small values flag
                         launch-bound execution
  d2h_copies             MemcpyD2H events

  The three scalar timing fields are MEDIANS over the traced calls
  (they were single-sample values before); the names are kept so
  downstream consumers keep working. Per-call values live under
  timing.per_call.

Methodology mirrors the April 2026 Kessler trace-profiling notes
(kessler/_compare_results/outputs/profiling, §7).
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

# XLA HLO while instructions are named "while.0", "while.11", ... The
# capitalised "While" event is the host-side runtime thunk — not counted.
WHILE_RE = re.compile(r"^while\.\d+$", re.IGNORECASE)

# Host-side end-to-end span names, by JAX version. A tuple so a version
# bump that renames the event degrades LOUDLY (warnings + None metrics),
# not silently — extend this when a new naming appears.
E2E_PATTERNS = ("PjitFunction", "jit(")
GPU_EXEC_PATTERN = "ExecuteThunks"


def find_trace_json(path) -> Path:
    """Resolve a jax.profiler.trace output to its trace JSON file.

    Accepts either the trace JSON itself or the directory passed to
    jax.profiler.trace (in which case the file lives under
    plugins/profile/<timestamp>/). If several traces are present the
    newest by mtime wins, so a fresh pass beats any stale leftover.
    """
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"trace path not found: {path}")
    candidates = (sorted(path.glob("**/*.trace.json.gz"))
                  + sorted(path.glob("**/*.trace.json")))
    if not candidates:
        raise FileNotFoundError(
            f"no *.trace.json[.gz] under {path} "
            f"(expected <dir>/plugins/profile/<ts>/<host>.trace.json.gz)"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load(trace_json: Path) -> dict:
    opener = gzip.open if trace_json.suffix == ".gz" else open
    with opener(trace_json, "rt") as f:
        return json.load(f)


def compute_while_scaling(ncol_a, wi_a, ncol_b, wi_b):
    """(while_iterations_scaling, ncol_ratio) from two trace runs.

    scaling = wi(ncol_hi) / wi(ncol_lo). None-safe: a fully vectorized
    translation with no while-loop at all has wi None at both ncols —
    scaling is then None (the no-while case is itself conclusive; the
    diagnoser reads it as the strongest vectorization signal).
    """
    if ncol_a >= ncol_b:
        n_hi, n_lo, wi_hi, wi_lo = ncol_a, ncol_b, wi_a, wi_b
    else:
        n_hi, n_lo, wi_hi, wi_lo = ncol_b, ncol_a, wi_b, wi_a
    ncol_ratio = round(n_hi / n_lo, 3)
    scaling = (round(wi_hi / wi_lo, 3)
               if (wi_hi is not None and wi_lo) else None)
    return scaling, ncol_ratio


def parse_trace(trace_json, n_calls: int = 1) -> dict:
    """Parse a Perfetto trace JSON into the diagnoser-facing metric dict.

    n_calls: number of profiled calls inside the trace. Body-execution
    counts are cumulative over all traced calls, so while_iterations is
    normalized to a per-call figure (the raw count is kept alongside as
    while_iterations_raw). With the default n_calls=1 the two are equal,
    preserving the field's meaning for existing single-call traces.
    """
    trace_json = find_trace_json(trace_json)
    data = _load(trace_json)
    events = data.get("traceEvents", [])

    # --- map pids to names from the metadata (ph == "M") events ---------
    # The GPU device and the host CPU each get their own process; compute
    # events and while-loop bodies live on the GPU process.
    pid_name = {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            pid_name[e.get("pid")] = e.get("args", {}).get("name", "")
    gpu_pids = {p for p, n in pid_name.items()
                if "GPU" in n or "/device:" in n}

    x_events = [e for e in events if e.get("ph") == "X"]

    # --- per-call timing: pair spans within the same call ----------------
    # The old logic took max(dur) over PjitFunction events and max(dur)
    # over ExecuteThunks events INDEPENDENTLY; with n_calls > 1 the two
    # maxima can come from different calls, making the subtraction
    # (dispatch overhead) a difference of inconsistent boundaries — it
    # could even go negative. Here every metric is computed per call:
    # each outermost end-to-end span is paired with the ExecuteThunks
    # event(s) nested inside its time window, and dispatch = e2e - gpu
    # strictly within that pair. Calls that fail to pair are excluded
    # with a warning, never mixed.
    warnings = []

    def _span(e):
        return {"ts": float(e.get("ts", 0.0)), "dur": float(e.get("dur", 0.0))}

    e2e_spans = [_span(e) for e in x_events
                 if any(p in e.get("name", "") for p in E2E_PATTERNS)]
    # outermost only: drop candidates nested inside another candidate
    # (inner jit(...) sub-spans must not count as calls)
    def _nested_in(a, b):
        return (a is not b and b["ts"] <= a["ts"]
                and a["ts"] + a["dur"] <= b["ts"] + b["dur"])

    outer = [a for a in e2e_spans
             if not any(_nested_in(a, b) for b in e2e_spans)]
    outer.sort(key=lambda s: s["ts"])

    gpu_spans = [_span(e) for e in x_events
                 if GPU_EXEC_PATTERN in e.get("name", "")]

    per_e2e_us, per_gpu_us, per_disp_us = [], [], []
    if not outer:
        warnings.append(
            f"no end-to-end span events matched patterns {E2E_PATTERNS}; "
            f"all timing metrics set to None - check this JAX version's "
            f"trace event naming against E2E_PATTERNS")
    elif not gpu_spans:
        warnings.append(
            f"no '{GPU_EXEC_PATTERN}' device-execution events found; all "
            f"timing metrics set to None - check this JAX version's trace "
            f"event naming")
    else:
        for k, s in enumerate(outer):
            nested = [g for g in gpu_spans
                      if g["ts"] >= s["ts"]
                      and g["ts"] + g["dur"] <= s["ts"] + s["dur"]]
            if not nested:
                warnings.append(
                    f"call {k}: no {GPU_EXEC_PATTERN} span nested within "
                    f"the end-to-end window; call excluded from timing")
                continue
            e2e = s["dur"]
            gpu = sum(g["dur"] for g in nested)
            per_e2e_us.append(e2e)
            per_gpu_us.append(gpu)
            per_disp_us.append(e2e - gpu)

    if n_calls and len(outer) != n_calls:
        warnings.append(
            f"expected {n_calls} end-to-end span(s), found {len(outer)} - "
            f"timing pairing count mismatch")

    def _stats_ms(vals_us):
        """(median_ms, {median, std, n}) — std None below 2 samples."""
        if not vals_us:
            return None, {"median": None, "std": None, "n": 0}
        ms = [v / 1000.0 for v in vals_us]
        med = round(statistics.median(ms), 4)
        std = round(statistics.stdev(ms), 4) if len(ms) > 1 else None
        return med, {"median": med, "std": std, "n": len(ms)}

    e2e_med, e2e_stats = _stats_ms(per_e2e_us)
    gpu_med, gpu_stats = _stats_ms(per_gpu_us)
    disp_med, disp_stats = _stats_ms(per_disp_us)

    all_names = Counter(e.get("name", "") for e in x_events)

    # --- GPU compute-stream events --------------------------------------
    # The device process holds the actual kernel executions (fused ops,
    # transposes, memcpys). xla_event_count counts those minus the memory
    # copies — an XLA-event structural-complexity proxy, deliberately NOT
    # named (or comparable to) the nsys device-kernel `kernel_launches`.
    gpu_names = Counter(e.get("name", "") for e in x_events
                        if e.get("pid") in gpu_pids)
    kernel_names = Counter({n: c for n, c in gpu_names.items()
                            if not n.startswith("Memcpy")})

    # --- while-loop body executions — the design signal -----------------
    # XLA `while.<N>` instruction spans live on the HOST process (one ph=X
    # event per body execution), so scan all events, not just the device.
    while_loops = {n: c for n, c in all_names.items() if WHILE_RE.match(n)}
    wi_raw = max(while_loops.values()) if while_loops else None
    if wi_raw is not None and n_calls and n_calls > 1:
        while_iterations = round(wi_raw / n_calls)
    else:
        while_iterations = wi_raw

    # MemcpyD2H: count every event (host + device span), matching the
    # reference report's methodology.
    d2h = all_names.get("MemcpyD2H", 0)

    # --- derived characterization metrics (report-only, never gate) ----
    xec = sum(kernel_names.values())
    # per-step structure independent of loop count: a sequential
    # translation can have perfectly ordinary per-step code — this metric
    # shows that the defect is purely the outer loop.
    events_per_while_iteration = round(xec / max(wi_raw or 0, 1), 2)
    # average device time per XLA event; small values flag launch-bound
    # execution. Uses the per-call event count to match the per-call
    # gpu_exec_ms median.
    xec_per_call = (xec / n_calls) if n_calls else xec
    mean_event_duration_us = (
        round(gpu_med * 1000.0 / xec_per_call, 2)
        if (gpu_med is not None and xec_per_call) else None
    )

    return {
        "source_trace": str(trace_json),
        "while_iterations": while_iterations,
        "while_iterations_raw": wi_raw,
        "while_loops": dict(sorted(while_loops.items())),
        "xla_event_count": xec,
        "unique_hlo_ops": len(kernel_names),
        "events_per_while_iteration": events_per_while_iteration,
        "mean_event_duration_us": mean_event_duration_us,
        # scalar timing fields are MEDIANS over the per-call pairs (names
        # kept for downstream compatibility; see module docstring)
        "end_to_end_ms": e2e_med,
        "gpu_exec_ms": gpu_med,
        "dispatch_overhead_ms": disp_med,
        "timing": {
            "end_to_end_ms": e2e_stats,
            "gpu_exec_ms": gpu_stats,
            "dispatch_overhead_ms": disp_stats,
            "per_call": {
                "end_to_end_ms": [round(v / 1000.0, 4) for v in per_e2e_us],
                "gpu_exec_ms": [round(v / 1000.0, 4) for v in per_gpu_us],
                "dispatch_overhead_ms": [round(v / 1000.0, 4)
                                         for v in per_disp_us],
            },
        },
        "warnings": warnings,
        "d2h_copies": d2h,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Parse a jax.profiler.trace output into structural "
                    "metrics (while_iterations, XLA-layer counts)."
    )
    ap.add_argument("--trace", required=True,
                    help="jax.profiler.trace output directory, or the "
                         "*.trace.json[.gz] file inside it.")
    ap.add_argument("--n-calls", type=int, default=1,
                    help="Number of profiled calls inside the trace; "
                         "while_iterations is normalized per call "
                         "(default: 1).")
    ap.add_argument("--out", default=None,
                    help="Where to write the metrics JSON (default: stdout).")
    args = ap.parse_args()

    metrics = parse_trace(args.trace, n_calls=args.n_calls)
    out_text = json.dumps(metrics, indent=2)
    if args.out:
        Path(args.out).write_text(out_text + "\n")
        print(f"Trace metrics written to {args.out}")
        print(f"while_iterations = {metrics['while_iterations']}  "
              f"xla_event_count = {metrics['xla_event_count']}")
    else:
        print(out_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
