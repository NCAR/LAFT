#!/usr/bin/env python3
"""
diagnose.py
─────────────────────────────────────────────────────────────────────────
Structural diagnosis for LLM-translated Fortran -> JAX code.

SINGLE-SOURCE DESIGN: every check that gates `production_ready` reads
one input — the metrics JSON written by profile_trace_run.py
(kessler_run_trace.json), which combines the jax.profiler.trace pass
(while_iterations + two-ncol scaling, xla_event_count, d2h_copies) with
the compiled-HLO counts (`hlo` block: actual fusion / scatter / gather /
while instructions). Both instruments control their own capture window
exactly, so none of the nsys-era accidental complexity (warmup/autotune
contamination, capture flags, denominator coupling, trace-overrules-
launch conflict handling) exists here.

nsys is NOT an input to the loop diagnosis. An nsys stats JSON may be
attached via --nsys-json; it lands in an informational
`hardware_corroboration` block (kernel time, sync stats) that never
affects any check or `production_ready`. Its place in the workflow is a
one-time hardware characterization of the final converged code.

DESIGN PRINCIPLES
  1. Single profiling iteration. Scaling questions are answered by the
     trace runner's in-process two-ncol check, not a job sweep.
  2. Every metric is normalized by the RIGHT denominator:
        while_iterations  -> per call; classified by ncol-SCALING when
                             two-ncol data exists
        scatter           -> absolute HLO instruction count, total
                             (zero bar; GATES at HIGH)
        gather            -> unfused count (entry + fusion-root);
                             ADVISORY only, never gates
        ops_per_fusion    -> HLO instructions per fused kernel
                             (report-only characterization, no
                             threshold yet)
        d2h_copies        -> per call, relative to output-array count
  3. No hardcoded magic constants tied to one specific code.
     Thresholds are ratios or structural (== 0, ~1, ~ncol_ratio).
  4. Gate only on signals that name a defect. Ambiguity is an ADVISORY:
     surfaced, never gating — a fix loop must not burn attempts on a
     signal that cannot say what to fix.

INPUTS
  - trace metrics JSON (REQUIRED): profile_trace_run.py output.
  - code_info JSON: ncol, nz, n_calls, jax_op_count, output_array_count.
  - nsys stats JSON (OPTIONAL): hardware corroboration only.

This stage is Fortran-independent: it assesses the structure of the
compiled JAX code on its own. Comparison against a Fortran baseline is
a separate downstream phase.

OUTPUT
  - JSON report: prioritized issues, each with severity + fix text
  - exit code 0 = production ready, 1 = issues found
─────────────────────────────────────────────────────────────────────────
"""

import json
import sys
import argparse
from dataclasses import dataclass, field, asdict
from enum import Enum


# ─────────────────────────────────────────────────────────────────────
# Severity levels
# ─────────────────────────────────────────────────────────────────────
class Severity(str, Enum):
    CRITICAL = "critical"   # translation failed its purpose (e.g. fori_loop)
    HIGH     = "high"       # major perf loss (e.g. scatter fusion break)
    MEDIUM   = "medium"     # meaningful optimization opportunity
    LOW      = "low"        # minor (excess D2H copies after compute)
    ADVISORY = "advisory"   # ambiguous signal worth a look; does NOT gate
                            # production_ready (a fix loop must never burn
                            # attempts on a signal that cannot name a defect)
    OK       = "ok"         # confirmation, no action


@dataclass
class Issue:
    metric: str
    severity: Severity
    message: str
    fix: str = ""
    evidence: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────
# Individual diagnostic checks
# Each takes (trace_metrics, code_info) and returns list[Issue].
# ─────────────────────────────────────────────────────────────────────

def check_column_loop_structure(t, code_info):
    """
    CRITICAL check: is the column loop parallel (vmap) or sequential
    (fori_loop)?

    Signals, in order of authority:
      - no while-loop at all (wi None from a trace that ran): fully
        vectorized, no subcycling — the strongest vectorization signal.
      - while_iterations_scaling (two-ncol data): scaling ~1 means the
        iteration count is ncol-independent (vectorized; any absolute
        wi > 1 is the physics' subcycle count, NOT a column loop);
        scaling ~ncol_ratio means sequential.
      - absolute while_iterations (single-ncol data): structural
        thresholds; the ambiguous mid-band (2 < wi < 0.9*ncol) is an
        ADVISORY — indistinguishable from legitimate subcycling.
      - xla_event_count per call vs ncol: a WEAK advisory-only proxy
        (device-event totals scaling like ncol hint at per-column
        dispatch); never gates.
    """
    issues = []
    ncol = code_info["ncol"]
    wi = t.get("while_iterations")
    scaling = t.get("while_iterations_scaling")
    ncol_ratio = t.get("ncol_ratio")

    wi_evidence = {"while_iterations": wi, "ncol": ncol,
                   "while_iterations_scaling": scaling,
                   "ncol_ratio": ncol_ratio}

    if wi is None:
        # wi None has two possible causes: the translation truly has no
        # while-loop (strongest vectorization signal), or the trace
        # parser's while.<N> event-name regex stopped matching (e.g. a
        # JAX/XLA version bump renamed the events) — which would turn
        # EVERY diagnosis into a false OK. The hlo block counts while
        # instructions independently, so the strongest-OK fires only
        # when both instruments agree there is no loop.
        hlo_ops = (t.get("hlo") or {}).get("ops") or {}
        hlo_while = hlo_ops.get("while")
        if hlo_while:
            issues.append(Issue(
                metric="while_iterations",
                severity=Severity.ADVISORY,
                message=(
                    f"Trace-parser/HLO mismatch: the compiled HLO "
                    f"contains {hlo_while} while instruction(s), but the "
                    f"trace pass measured no while.<N> body executions "
                    f"(while_iterations=None). The trace parser's "
                    f"while-event matching may not match this JAX/XLA "
                    f"version's event naming - check "
                    f"parse_jax_trace.WHILE_RE against a raw trace "
                    f"before trusting any while_iterations result."
                ),
                evidence={"while_iterations": None,
                          "hlo_while_ops": hlo_while, "ncol": ncol},
            ))
        elif hlo_while is None:
            issues.append(Issue(
                metric="while_iterations",
                severity=Severity.ADVISORY,
                message=(
                    "while_iterations is None and no `hlo` block is "
                    "present to corroborate - cannot distinguish a "
                    "loop-free translation from a trace-parser failure. "
                    "Re-run the trace pass with the current "
                    "profile_trace_run.py."
                ),
                evidence={"while_iterations": None, "ncol": ncol},
            ))
        else:
            issues.append(Issue(
                metric="while_iterations",
                severity=Severity.OK,
                message=(
                    "No XLA while-loop: the trace pass measured no "
                    "while-body executions AND the compiled HLO contains "
                    "zero while instructions - the translation is fully "
                    "vectorized with no subcycling loop. Strongest "
                    "possible vectorization signal (both instruments "
                    "agree)."
                ),
                evidence={"while_iterations": None, "hlo_while_ops": 0,
                          "ncol": ncol},
            ))
    elif scaling is not None:
        # Two-ncol data present: classify by how wi SCALES, not by its
        # absolute value. Thresholds are structural ratios: ~1 means
        # ncol-independent, ~ncol_ratio means proportional to ncol.
        if scaling <= 1.5:
            issues.append(Issue(
                metric="while_iterations",
                severity=Severity.OK,
                message=(
                    f"while_iterations ({wi}) scales x{scaling} when ncol "
                    f"changes x{ncol_ratio} -> iteration count is "
                    f"ncol-independent: vmap column parallelism confirmed. "
                    f"An absolute count > 1 is the physics' subcycle "
                    f"count, not a column loop."
                ),
                evidence=wi_evidence,
            ))
        elif ncol_ratio and scaling >= 0.75 * ncol_ratio:
            issues.append(Issue(
                metric="while_iterations",
                severity=Severity.CRITICAL,
                message=(
                    f"while_iterations ({wi}) scales x{scaling} when ncol "
                    f"changes x{ncol_ratio} -> iteration count tracks "
                    f"problem size: the column loop is sequential."
                ),
                fix=(
                    "Replace lax.fori_loop over the column dimension with "
                    "jax.vmap. Columns in this physics are independent "
                    "(no inter-column data dependency), so they must be "
                    "batched into a single vectorized call. The while-loop "
                    "iteration count should then be independent of ncol."
                ),
                evidence=wi_evidence,
            ))
        else:
            issues.append(Issue(
                metric="while_iterations",
                severity=Severity.ADVISORY,
                message=(
                    f"while_iterations ({wi}) scales x{scaling} when ncol "
                    f"changes x{ncol_ratio} - neither ncol-independent "
                    f"(~1) nor proportional (~{ncol_ratio}). Partially "
                    f"sequential structure possible; inspect manually. "
                    f"Advisory only - not gating production_ready."
                ),
                evidence=wi_evidence,
            ))
    else:
        # Single-ncol data: structural thresholds.
        if wi >= ncol * 0.9:
            issues.append(Issue(
                metric="while_iterations",
                severity=Severity.CRITICAL,
                message=(
                    f"while_iterations ({wi}) approximately equals ncol "
                    f"({ncol}). The subcycling while-loop is running once "
                    f"per column -> the column loop is sequential."
                ),
                fix=(
                    "Replace lax.fori_loop over the column dimension with "
                    "jax.vmap. Columns in this physics are independent "
                    "(no inter-column data dependency), so they must be "
                    "batched into a single vectorized call. The while-loop "
                    "should then execute exactly once, vectorized across "
                    "all columns."
                ),
                evidence={"while_iterations": wi, "ncol": ncol},
            ))
        elif wi <= 2:
            issues.append(Issue(
                metric="while_iterations",
                severity=Severity.OK,
                message=(
                    f"while_iterations ({wi}) is independent of ncol "
                    f"({ncol}) -> vmap column parallelism confirmed."
                ),
                evidence={"while_iterations": wi, "ncol": ncol},
            ))
        else:
            # Mid-band: indistinguishable from legitimate physics
            # subcycling with single-ncol data, so it must not gate the
            # fix loop (which would burn every attempt on possibly
            # healthy code). The two-ncol scaling check disambiguates.
            issues.append(Issue(
                metric="while_iterations",
                severity=Severity.ADVISORY,
                message=(
                    f"while_iterations ({wi}) is neither ~1 nor ~ncol "
                    f"({ncol}). This is consistent with legitimate physics "
                    f"subcycling ({wi} substeps inside a vectorized loop) "
                    f"but cannot be distinguished from partial "
                    f"sequentialization at a single ncol. Advisory only - "
                    f"not gating production_ready. Run the trace pass's "
                    f"two-ncol scaling check (--ncol-check) to "
                    f"disambiguate."
                ),
                evidence={"while_iterations": wi, "ncol": ncol},
            ))

    # ---- weak structural proxy: XLA device events per call vs ncol ---
    # An XLA-event total that scales like ncol hints at per-column
    # dispatch. It aggregates a different layer than device kernels and
    # is contamination-free (the trace window is exact), but it is still
    # only a complexity proxy — advisory, never gating.
    xec = t.get("xla_event_count")
    n_calls = t.get("n_calls") or 1
    if xec is not None and ncol:
        events_per_call = xec / n_calls
        events_per_col = events_per_call / ncol
        if events_per_col > 1.0:
            issues.append(Issue(
                metric="xla_events_per_column",
                severity=Severity.ADVISORY,
                message=(
                    f"{events_per_col:.1f} XLA device events per column "
                    f"per call ({xec} events / {n_calls} call(s) / {ncol} "
                    f"columns). Device-event totals that scale like ncol "
                    f"are a weak hint of per-column dispatch. Advisory "
                    f"only - the while_iterations signals above are the "
                    f"authoritative structure measurement."
                ),
                evidence={"xla_event_count": xec, "n_calls": n_calls,
                          "ncol": ncol,
                          "events_per_column": round(events_per_col, 3)},
            ))

    return issues


def check_fusion_quality(t, code_info):
    """
    HIGH / advisory check: did XLA fuse operations, or is the chain
    broken by scatter / gather (subset indexing)?

    PRIMARY signal: actual scatter/gather instructions in the compiled
    HLO (`hlo` block; zero is the bar). The two are NOT symmetric:
      - scatter GATES (HIGH) on the TOTAL count. A scatter performs
        irregular writes; XLA wraps `.at[idx].set(...)` in an input
        fusion whose ROOT is the scatter — the cost exists wherever it
        appears.
      - gather NEVER gates. An unfused gather (entry-level or
        fusion-root: ops - ops_in_fusion_bodies + ops_as_fusion_roots)
        is an ADVISORY with full fix text. INTERIOR gathers absorbed
        mid-fusion are indexed reads folded into the kernel — the
        compiler succeeding, and what stencil indexing (q[k+/-1])
        legitimately compiles to; empirically the converged
        production-ready translation contains 4 of them, produced by
        the prescribed jnp.clip/jnp.where boundary fix itself. They get
        a counted OK confirmation.

    SECONDARY signal: ops_per_fusion — HLO instructions per fused
    kernel, measured directly from the compiled HLO. REPORT-ONLY (an OK
    characterization entry, never an advisory or gate): no threshold is
    set deliberately, because any bar chosen today is guesswork; after
    the four-model re-profiling with the new harness, the empirical
    spread (Gemini vs Claude/GPT vs Qwen) will show where a meaningful
    bar sits, and a threshold can be added then with data behind it.
    This retires the old AST-based fusion_efficiency proxy
    (unique_hlo_ops / jax_op_count), whose denominator varied with LLM
    coding style — operator arithmetic, method calls, and setup calls
    were miscounted — making it incomparable across translations
    (jax_op_count stays in code_info for reference).
    """
    issues = []
    hlo = t.get("hlo")
    if not hlo:
        issues.append(Issue(
            metric="hlo_block",
            severity=Severity.ADVISORY,
            message=(
                "No `hlo` block in the trace metrics - scatter/gather "
                "detection skipped. Re-run the trace pass with the "
                "current profile_trace_run.py (it dumps and parses the "
                "compiled HLO)."
            ),
        ))
    else:
        ops = hlo.get("ops", {})
        in_fusion = hlo.get("ops_in_fusion_bodies", {})
        roots = hlo.get("ops_as_fusion_roots", {})

        scatter_total = ops.get("scatter", 0)
        if scatter_total > 0:
            issues.append(Issue(
                metric="scatter_ops",
                severity=Severity.HIGH,
                message=(
                    f"{scatter_total} scatter instruction(s) in the "
                    f"compiled HLO "
                    f"({roots.get('scatter', 0)} as fusion roots, "
                    f"{in_fusion.get('scatter', 0)} inside fusion bodies). "
                    f"Each scatter corresponds to an "
                    f"`.at[indices].set(...)` operation; its irregular "
                    f"memory writes execute on every call and break the "
                    f"XLA fusion chain."
                ),
                fix=(
                    "Replace `array.at[subset_indices].set(new_values)` "
                    "with `jnp.where(mask, new_values, array)` over the "
                    "full array. This keeps memory access uniform and "
                    "lets XLA fuse the operation with its neighbors. "
                    "Target: zero scatter instructions."
                ),
                evidence={
                    "scatter_total": scatter_total,
                    "scatter_in_fusion_bodies": in_fusion.get("scatter", 0),
                    "scatter_as_fusion_roots": roots.get("scatter", 0),
                },
            ))

        gather_total = ops.get("gather", 0)
        gather_interior = (in_fusion.get("gather", 0)
                           - roots.get("gather", 0))
        gather_gating = gather_total - gather_interior
        if gather_gating > 0:
            # ADVISORY by design: gather must never gate production_ready.
            # Irregular reads are far cheaper than scatter's irregular
            # writes, and legitimate stencil indexing can leave a gather
            # unfused without naming a fixable defect.
            issues.append(Issue(
                metric="gather_ops",
                severity=Severity.ADVISORY,
                message=(
                    f"{gather_gating} gather instruction(s) at entry "
                    f"level or as fusion roots in the compiled HLO "
                    f"({gather_total} total, {gather_interior} interior/"
                    f"absorbed). Each corresponds to "
                    f"`array[subset_indices]` indexing whose irregular "
                    f"memory access XLA could not fold into a larger "
                    f"kernel."
                ),
                fix=(
                    "Replace subset indexing `jnp.arange(surf, toa)` + "
                    "`array[idx]` with full-array operations "
                    "`jnp.arange(nz)` and select the active region with "
                    "`jnp.where(mask, ...)`. Use `jnp.clip` for boundary "
                    "handling instead of explicit boundary indices. "
                    "Target: zero UNFUSED gathers (entry-level or "
                    "fusion-root). Interior gathers inside fusion bodies "
                    "are acceptable and expected from stencil indexing "
                    "(q[k+/-1]-style) - do not chase those."
                ),
                evidence={
                    "gather_gating": gather_gating,
                    "gather_total": gather_total,
                    "gather_interior_absorbed": gather_interior,
                    "gather_as_fusion_roots": roots.get("gather", 0),
                },
            ))
        elif gather_total > 0:
            issues.append(Issue(
                metric="gather_ops",
                severity=Severity.OK,
                message=(
                    f"{gather_total} gather instruction(s), all interior "
                    f"to fusion bodies - XLA absorbed them as indexed "
                    f"reads inside larger kernels (the expected result of "
                    f"jnp.clip/jnp.where boundary handling). No "
                    f"standalone gather cost."
                ),
                evidence={"gather_total": gather_total,
                          "gather_interior_absorbed": gather_interior},
            ))

        if (scatter_total == 0 and gather_gating <= 0):
            issues.append(Issue(
                metric="scatter_ops",
                severity=Severity.OK,
                message=(
                    "Zero scatter instructions and zero unabsorbed "
                    "gathers in the compiled HLO - no fusion-chain "
                    "breaks from subset indexing."
                ),
                evidence={"scatter_total": scatter_total,
                          "gather_total": gather_total},
            ))

        # ---- SECONDARY: ops-per-fusion characterization (REPORT-ONLY) --
        # How much work XLA packed into each fused kernel, measured from
        # the compiled HLO. Deliberately NO threshold and NO advisory.
        # The four-model re-profiling (2026-06-11, Kessler @ ncol=1000,
        # A100, jax 0.6.2) measured the spread: clean translations sit at
        # 18.08 (Claude), 18.08 (GPT), 21.91 (Gemini); the one defective
        # translation (Qwen, 14 scatters) sits at 11.0. The low outlier is
        # fully explained by scatter fragmentation — each scatter breaks
        # the fusion chain into more, smaller kernels — which the gating
        # scatter check above already catches. A threshold here would add
        # no independent signal on n=4, so this stays report-only. This
        # retires the AST-based fusion_efficiency proxy (unique_hlo_ops /
        # jax_op_count): its denominator varied with LLM coding style,
        # making it incomparable across translations.
        fusion_count = ops.get("fusion", 0)
        total_instr = hlo.get("total_instructions")
        in_fusion_total = hlo.get("instructions_in_fusion_bodies")
        ops_per_fusion = (
            round(in_fusion_total / fusion_count, 2)
            if fusion_count and in_fusion_total is not None else None
        )
        if ops_per_fusion is not None:
            char_msg = (
                f"Fusion characterization: {ops_per_fusion} HLO "
                f"instructions per fused kernel ({fusion_count} fusions, "
                f"{total_instr} total instructions). Higher = XLA packed "
                f"more work per kernel. Report-only - the measured "
                f"cross-model spread (clean 18-22 vs scatter-fragmented "
                f"11) is explained by the gating scatter check, so no "
                f"independent threshold is set."
            )
        else:
            char_msg = (
                f"Fusion characterization: no fusion kernels in the "
                f"compiled HLO ({total_instr} total instructions). "
                f"Report-only."
            )
        issues.append(Issue(
            metric="fusion_characterization",
            severity=Severity.OK,
            message=char_msg,
            evidence={
                "ops_per_fusion": ops_per_fusion,
                "fusion_count": fusion_count,
                "total_instructions": total_instr,
            },
        ))

    return issues


def check_d2h_copies(t, code_info):
    """
    LOW check: device-to-host copies, from the trace pass (whose capture
    window is exactly the profiled calls — no warmup in the count).

    Normalized per call, then compared against the OUTPUT-ARRAY COUNT.
    A correct translation copies back roughly: output arrays + a few
    scalars. Many multiples of that means scalar readbacks inside the
    JIT boundary (e.g. `int(traced_array)` inside a loop). The trace
    counts MemcpyD2H events on host and device spans, so the figure can
    double-count a physical copy; the 3x ratio bar absorbs that.
    """
    issues = []
    d2h = t.get("d2h_copies")
    if d2h is None:
        return issues

    n_calls = t.get("n_calls") or 1
    d2h_per_call = d2h / n_calls

    out_count = code_info.get("output_array_count") or 8
    expected_max = out_count + 4
    ratio = d2h_per_call / expected_max

    if ratio > 3.0:
        issues.append(Issue(
            metric="d2h_copies",
            severity=Severity.LOW,
            message=(
                f"{d2h_per_call:.1f} device-to-host copy events per call "
                f"({d2h} over {n_calls} traced call(s)) vs an expected "
                f"~{expected_max} (output arrays + scalars). Excess copies "
                f"indicate scalar readbacks inside the JIT boundary."
            ),
            fix=(
                "Move `int(...)` / `float(...)` conversions of traced "
                "arrays OUT of the @jax.jit function - do them in the "
                "Python wrapper after the compute call returns. Inside "
                "jit, use lax.cond / jnp.where instead of Python `if` on "
                "array values so nothing is read back mid-computation."
            ),
            evidence={"d2h_copies": d2h,
                      "d2h_per_call": round(d2h_per_call, 2),
                      "n_calls": n_calls, "expected_max": expected_max,
                      "ratio": round(ratio, 2)},
        ))
    else:
        issues.append(Issue(
            metric="d2h_copies",
            severity=Severity.OK,
            message=(
                f"{d2h_per_call:.1f} device-to-host copy events per call "
                f"({d2h} over {n_calls} traced call(s)) - consistent with "
                f"reading back {out_count} output arrays once after "
                f"compute."
            ),
            evidence={"d2h_copies": d2h,
                      "d2h_per_call": round(d2h_per_call, 2),
                      "n_calls": n_calls, "expected_max": expected_max},
        ))

    return issues


# ─────────────────────────────────────────────────────────────────────
# Optional nsys hardware corroboration (informational ONLY)
# ─────────────────────────────────────────────────────────────────────
def extract_hardware_corroboration(nsys_json):
    """
    Hardware-layer numbers from an nsys stats export. Attached to the
    report verbatim for human inspection / the final hardware
    characterization of converged code. NEVER read by any check and
    never affects production_ready: the nsys capture spans the whole
    process (warmup, autotuning), so its counts are not comparable to
    the trace pass's exact-window metrics.
    """
    kernels = (
        nsys_json.get("cuda_gpu_kern_sum")
        or nsys_json.get("kernels")
        or []
    )
    api = (
        nsys_json.get("cuda_api_sum")
        or nsys_json.get("cuda_api_summary")
        or []
    )

    kernel_time_ns = sum(int(k.get("total_time_ns", 0) or 0) for k in kernels)
    kernel_launches = sum(
        int(k.get("instances", k.get("count", 0)) or 0) for k in kernels
    )

    sync_calls = 0
    sync_time_ns = 0
    memcpy_d2h = 0
    for row in api:
        nm = row.get("name", "").lower()
        cnt = int(row.get("num_calls", row.get("count", 0)) or 0)
        if nm == "custreamsynchronize":
            sync_calls = cnt
            sync_time_ns = int(row.get("total_time_ns", 0) or 0)
        if "memcpy" in nm and "dtoh" in nm:
            memcpy_d2h += cnt

    return {
        "note": ("informational only - full-process nsys capture "
                 "(includes warmup/autotuning); not used by any check"),
        "gpu_kernel_time_ms": (
            round(kernel_time_ns / 1e6, 4) if kernel_time_ns else None
        ),
        "kernel_launches": kernel_launches,
        "unique_kernels": len(kernels),
        "memcpy_d2h": memcpy_d2h,
        "sync_calls": sync_calls,
        "sync_avg_us": (
            round(sync_time_ns / sync_calls / 1e3, 1) if sync_calls else None
        ),
        "sync_to_compute_ratio": (
            round(sync_time_ns / kernel_time_ns, 1) if kernel_time_ns
            else None
        ),
    }


# ─────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────
_SEVERITY_RANK = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
    Severity.ADVISORY: 4,
    Severity.OK: 5,
}


def diagnose(trace_metrics, code_info, nsys_json=None):
    """
    Run all checks against one trace-pass metrics dict and return a
    prioritized report dict.

    trace_metrics (REQUIRED): profile_trace_run.py output —
        while_iterations (+ _scaling / ncol_ratio), xla_event_count,
        unique_hlo_ops, d2h_copies, n_calls, and the `hlo` block.
    code_info expected keys:
        ncol, nz, n_calls, jax_op_count, output_array_count
    nsys_json (optional): attached as hardware_corroboration,
        informational only.
    """
    t = trace_metrics

    all_issues = []
    all_issues += check_column_loop_structure(t, code_info)
    all_issues += check_fusion_quality(t, code_info)
    all_issues += check_d2h_copies(t, code_info)

    # sort by severity (critical first)
    all_issues.sort(key=lambda i: _SEVERITY_RANK[i.severity])

    # Advisories are surfaced but never gate production_ready: they flag
    # ambiguity, not a named defect, and a fix loop must not burn its
    # attempts on them.
    actionable = [i for i in all_issues
                  if i.severity not in (Severity.OK, Severity.ADVISORY)]
    advisories = [i for i in all_issues if i.severity == Severity.ADVISORY]
    confirmations = [i for i in all_issues if i.severity == Severity.OK]

    worst = actionable[0].severity if actionable else Severity.OK
    production_ready = len(actionable) == 0

    report = {
        "production_ready": production_ready,
        "worst_severity": worst.value,
        "summary": {
            # The single gating instrument: trace + compiled-HLO metrics.
            "trace_metrics": trace_metrics,
            "code_info": code_info,
            "issue_count": len(actionable),
            "advisory_count": len(advisories),
        },
        # Optional, informational, never gating.
        "hardware_corroboration": (
            extract_hardware_corroboration(nsys_json) if nsys_json else None
        ),
        "actionable_issues": [asdict_issue(i) for i in actionable],
        "advisories": [asdict_issue(i) for i in advisories],
        "confirmations": [asdict_issue(i) for i in confirmations],
        # one consolidated block the LLM can be fed directly
        "llm_feedback": build_llm_feedback(actionable, advisories),
    }
    return report


def asdict_issue(issue: Issue):
    d = asdict(issue)
    d["severity"] = issue.severity.value
    return d


def build_llm_feedback(actionable, advisories=()):
    """
    Turn the actionable issues into a single ordered instruction block
    suitable for feeding straight back to the translating LLM.
    Critical issues first; only the top few to avoid lost-in-the-middle.
    Advisories are mentioned by count only — they are informational and
    must not read as fix instructions.
    """
    if not actionable:
        msg = ("No structural issues detected. The translation exploits "
               "the GPU correctly.")
        if advisories:
            msg += (f" ({len(advisories)} advisory note(s) in the report "
                    f"- informational, not gating.)")
        return msg

    lines = [
        "The translated JAX code has the following structural problems, "
        "in priority order. Fix the CRITICAL issue first, then re-profile "
        "before addressing lower-severity items.",
        "",
    ]
    # cap at 4 so the feedback prompt itself stays small/focused
    for n, issue in enumerate(actionable[:4], 1):
        lines.append(f"{n}. [{issue.severity.value.upper()}] "
                     f"{issue.metric}")
        lines.append(f"   Problem: {issue.message}")
        lines.append(f"   Fix: {issue.fix}")
        lines.append("")
    if len(actionable) > 4:
        lines.append(f"({len(actionable) - 4} additional lower-severity "
                     f"item(s) deferred until after re-profiling.)")
    if advisories:
        lines.append(f"({len(advisories)} advisory note(s) in the report "
                     f"- informational, not gating.)")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Structural diagnosis for LLM-translated Fortran->JAX "
                    "code (single input: the trace-pass metrics JSON)."
    )
    ap.add_argument("--trace-json", required=True,
                    help="Path to kessler_run_trace.json from "
                         "profile_trace_run.py (trace metrics + hlo "
                         "block). The single gating input.")
    ap.add_argument("--code-info", required=True,
                    help="Path to JSON with ncol, nz, n_calls, "
                         "jax_op_count, output_array_count.")
    ap.add_argument("--nsys-json", default=None,
                    help="Optional nsys stats JSON; attached as an "
                         "informational hardware_corroboration block. "
                         "Never affects checks or production_ready.")
    ap.add_argument("--out", default=None,
                    help="Where to write the JSON report "
                         "(default: stdout).")
    args = ap.parse_args()

    with open(args.trace_json) as f:
        trace_metrics = json.load(f)
    with open(args.code_info) as f:
        code_info = json.load(f)

    nsys_json = None
    if args.nsys_json:
        with open(args.nsys_json) as f:
            nsys_json = json.load(f)

    report = diagnose(trace_metrics, code_info, nsys_json)

    out_text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out_text)
        print(f"Report written to {args.out}")
        print(f"production_ready = {report['production_ready']} "
              f"(worst: {report['worst_severity']})")
    else:
        print(out_text)

    # exit code: 0 = clean, 1 = issues found  -> lets the orchestrator
    # decide whether to trigger an LLM refinement pass
    sys.exit(0 if report["production_ready"] else 1)


if __name__ == "__main__":
    main()
