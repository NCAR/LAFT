#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# phase04_04_completeness_check.py (moved from validation/ 2026-08-25: it is
# TRANSLATE_WORKFLOW Step 2.5, run BEFORE lint) — flag translations that were SCAFFOLDED / STUBBED
#                         instead of fully translated.
#
# Motivation: lint checks structure (has _core, @jax.jit, wrapper) and passes a
# scaffold; the semantic audit catches faithfulness but runs later and heavier.
# Nothing checked COMPLETENESS — "did the model translate the whole procedure,
# or summarise it?" This is the cheap first gate for that, run after translation
# and before lint.
#
# Signal hierarchy (tuned on a run where the main orchestrator was a full stub
# and the init routine partial, while several *compact* procs were complete):
#
#   FAIL (high confidence — block the procedure):
#     - scaffold-phrase scan: precise phrases a model emits when it punts
#       (e.g. "for the sake of this draft", "would go here", "remaining physics").
#     - dead calls: a procedure in this proc's `calls` list that appears ONLY
#       commented-out in the translation (model punted on a dependency, e.g.
#       a commented `# qvs = qv_sat(...)`).
#
#   WARN (advisory — trigger a look, never an auto-fail):
#     - extreme-low size ratio with NO honest scaffold marker: could be silent
#       truncation, OR legitimate de-duplication of repetitive Fortran (LUT
#       interpolation). Size ratio is deliberately NOT a fail — it false-positives
#       on faithfully-compacted code.
#     - scalar-only misclassification: frontend marked the proc scalar_only but
#       its Fortran declares arrays (this is what made a model drop an init
#       routine's table work "to obey the SCALAR-ONLY constraint").
#
# Writes out/reports/translation/completeness_check.json and prints a table.
#
# STOP RULE (TRANSLATE_WORKFLOW.md Step 2.5, 2026-08-25): any FAIL is a hard
# stop — workflow_state.json → status "aborted_scaffold" with the evidence,
# exit code 2, and every downstream gate (audit_gate.py → runtime PBS) stays
# closed. No automatic re-translation: a scaffold is an output-horizon /
# capability signal, re-running the same prompt reproduces it; a human decides.
# `--proc <name>` runs the check for one procedure right after it is authored;
# `--no-abort` reports without touching the state file. Exit 1 = MISSING only.
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from framework_config import get_config, init as init_config, add_config_arg  # noqa: E402

# Precise scaffold-admission phrases (case-insensitive). Kept tight on purpose —
# these are distinctive enough not to appear in genuine translated code.
SCAFFOLD_PHRASES = [
    "would go here",
    "for the sake of this draft",
    "represent the logic flow",
    "remaining physics",
    "in a real scenario",
    "omitted as it",
    "placeholder for",          # 2026-08-25: "placeholder for <proc>_core"
    "this stub returns",        # 2026-08-25: seen in a scaffold: "This stub returns all INOUT parameters unchanged"
    "parameters unchanged",
    "placeholder for actual",
    "placeholder or that",
    "would be numpy loads",
    "would be calls to other",
    "massive orchestration routine",
    "logic must be vectorized across",
    "the actual file reading",
    "represent the logic",
]

# size-ratio WARN only at an extreme, and only for sizeable procedures
RATIO_WARN_BELOW = 0.10
RATIO_WARN_MIN_FLINES = 200

ARRAY_DECL = [re.compile(r"::\s*\w+\s*\("), re.compile(r"\bdimension\s*\(", re.IGNORECASE)]


def code_lines(text: str) -> int:
    """Non-blank, non-comment lines (rough measure of real translated content)."""
    n = 0
    for ln in text.splitlines():
        c = ln.split("#", 1)[0].strip()
        if c and not c.startswith('"""') and not c.startswith("'''"):
            n += 1
    return n


def scan_scaffold(text: str) -> list[str]:
    low = text.lower()
    return [p for p in SCAFFOLD_PHRASES if p in low]


def _call_pattern(callee: str) -> re.Pattern:
    """A live call to `callee` in a translation may be spelled `callee(`,
    `callee_core(`, `callee_value(` or a private `_callee_core(` /
    `_callee_value(` (the bridge-contract core splits). Comments are excluded
    by the caller. Case-insensitive: Fortran names are case-insensitive and
    translations keep the Fortran spelling."""
    return re.compile(rf"(?<![\w])_?{re.escape(callee)}(?:_core|_value)?\s*\(", re.I)


def scan_calls(text: str, calls: list[str], self_name: str) -> tuple[list[str], list[str]]:
    """Return (dead, absent): `dead` = callee appears ONLY commented-out
    (model punted on a dependency — FAIL); `absent` = callee appears nowhere,
    live or commented (inlined legitimately, or silently dropped — WARN,
    listed for the human; the semantic audit's call check makes it strict)."""
    lines = text.splitlines()
    dead, absent = [], []
    for c in calls:
        if not c or c.lower() == self_name.lower():
            continue
        pat = _call_pattern(c)
        live = commented = False
        for ln in lines:
            code, sep, comment = ln.partition("#")
            if pat.search(code):
                live = True
            if sep and pat.search(comment):
                commented = True
        if commented and not live:
            dead.append(c)
        elif not live:
            absent.append(c)
    return dead, absent


def scan_dead_calls(text: str, calls: list[str], self_name: str) -> list[str]:
    """Backward-compatible wrapper: only the FAIL-level dead calls."""
    return scan_calls(text, calls, self_name)[0]


def abort_on_scaffold(state_path: Path, failed: list[dict], json_path: Path) -> None:
    """Hard stop (TRANSLATE_WORKFLOW.md Step 2.5): record status
    `aborted_scaffold` with the evidence. No automatic re-translation — a
    scaffold is a model/context limit, re-running the same prompt reproduces
    it; a human decides (split the procedure, change model or prompt policy,
    or archive as a failure experiment)."""
    state = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except ValueError:
            state = {}
    now = datetime.now().isoformat(timespec="seconds")
    state["status"] = "aborted_scaffold"
    state["updated_at"] = now
    state["scaffold_abort"] = {
        "at": now,
        "procedures": [{"proc": r["proc"], "scaffold_phrases": r.get("scaffold_phrases", []),
                        "dead_calls": r.get("dead_calls", []),
                        "fortran_lines": r.get("fortran_lines"),
                        "jax_code_lines": r.get("jax_code_lines"), "ratio": r.get("ratio")}
                       for r in failed],
        "evidence": str(json_path),
    }
    state["next_steps"] = [
        "HUMAN DECISION REQUIRED — translation stopped on scaffold "
        f"({', '.join(r['proc'] for r in failed)}); see scaffold_abort + evidence JSON",
        "options: split the procedure / change model or prompt policy / archive as a "
        "failure experiment (tools/copy_AI_results.sh) — do NOT re-run the same prompt",
        "after the decision: fix or re-author, then re-run "
        "workflow_translator/phase04_04_completeness_check.py and continue at Step 2.5",
    ]
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def fortran_has_arrays(src: str) -> bool:
    return any(rx.search(src) for rx in ARRAY_DECL)


def main(only_proc: str | None = None, no_abort: bool = False) -> int:
    cfg = get_config()
    jax_dir = cfg.jax_dir
    packets_dir = cfg.packets_dir
    out_dir = cfg.reports_dir / "translation"
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = cfg.jax_dir / "workflow_state.json"

    deps = {e["proc_name"]: e for e in
            json.loads((cfg.all_deps_file).read_text(encoding="utf-8"))}
    if only_proc:
        match = {k: v for k, v in deps.items() if k.lower() == only_proc.lower()}
        if not match:
            print(f"ERROR: procedure {only_proc!r} not in {cfg.all_deps_file}")
            return 2
        deps = match

    results = []
    for name, e in sorted(deps.items()):
        jf = jax_dir / f"{name}.py"
        pk = packets_dir / f"{name}_merged.json"
        if not jf.exists():
            results.append({"proc": name, "verdict": "MISSING",
                            "note": "no translation file"})
            continue
        text = jf.read_text(encoding="utf-8")
        src = ""
        if pk.exists():
            src = json.loads(pk.read_text(encoding="utf-8")) \
                .get("phase2_packet", {}).get("fortran_source", "")
        f_lines = len([l for l in src.splitlines() if l.strip()])
        j_lines = len(text.splitlines())
        j_code = code_lines(text)
        ratio = round(j_code / f_lines, 3) if f_lines else None

        scaffold = scan_scaffold(text)
        dead, absent = scan_calls(text, e.get("calls", []) or [], name)
        scalar_mis = bool(e.get("scalar_only")) and fortran_has_arrays(src)
        low_ratio = (ratio is not None and f_lines >= RATIO_WARN_MIN_FLINES
                     and ratio < RATIO_WARN_BELOW and not scaffold)

        # Silent stub (2026-08-25 specimen: 2877 → 172 code lines,
        # 30 of 35 callees absent, no admission phrase, no commented call): a
        # sizeable procedure that is both extremely compact AND has lost most of
        # its call graph is a scaffold even when it says nothing.
        calls = [c for c in (e.get("calls", []) or []) if c and c.lower() != name.lower()]
        silent_stub = (ratio is not None and f_lines >= RATIO_WARN_MIN_FLINES
                       and ratio < RATIO_WARN_BELOW and len(calls) >= 3
                       and len(absent) >= max(3, (len(calls) + 1) // 2))
        if scaffold or dead or silent_stub:
            verdict = "FAIL"
        elif scalar_mis or low_ratio or absent:
            verdict = "WARN"
        else:
            verdict = "PASS"

        results.append({
            "proc": name, "fortran_lines": f_lines, "jax_lines": j_lines,
            "jax_code_lines": j_code, "ratio": ratio,
            "scaffold_phrases": scaffold, "dead_calls": dead, "absent_calls": absent,
            "silent_stub": silent_stub,
            "scalar_misclassified": scalar_mis, "low_ratio_advisory": low_ratio,
            "verdict": verdict,
        })

    order = {"FAIL": 0, "MISSING": 1, "WARN": 2, "PASS": 3}
    results.sort(key=lambda r: (order.get(r["verdict"], 9), r["proc"]))
    n_fail = sum(r["verdict"] == "FAIL" for r in results)
    n_warn = sum(r["verdict"] == "WARN" for r in results)
    n_miss = sum(r["verdict"] == "MISSING" for r in results)
    n_pass = sum(r["verdict"] == "PASS" for r in results)

    summary = {
        "stage": "completeness_check",
        "timestamp": datetime.now().isoformat(),
        "total": len(results),
        "pass": n_pass, "warn": n_warn, "fail": n_fail, "missing": n_miss,
        "status": "FAIL" if (n_fail or n_miss) else ("WARN" if n_warn else "PASS"),
        "results": results,
    }
    out_json = out_dir / ("completeness_check.json" if not only_proc
                          else f"completeness_check_{only_proc}.json")
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # --- table ---
    print("=" * 78)
    print("  TRANSLATION COMPLETENESS CHECK")
    print("=" * 78)
    print(f"  {'proc':<30} {'F':>5} {'Jcode':>6} {'ratio':>6}  {'verdict':<7} why")
    print(f"  {'-'*30} {'-'*5} {'-'*6} {'-'*6}  {'-'*7} {'-'*20}")
    for r in results:
        if r["verdict"] == "PASS":
            continue
        why = []
        if r.get("scaffold_phrases"): why.append(f"scaffold:{r['scaffold_phrases'][:2]}")
        if r.get("dead_calls"): why.append(f"dead_calls:{r['dead_calls']}")
        if r.get("silent_stub"): why.append(f"SILENT STUB: ratio {r['ratio']} with {len(r['absent_calls'])} callees absent")
        if r.get("absent_calls"): why.append(f"absent_calls:{r['absent_calls']} (inlined or dropped?)")
        if r.get("scalar_misclassified"): why.append("scalar-only-but-arrays")
        if r.get("low_ratio_advisory"): why.append("very-compact (verify)")
        if r["verdict"] == "MISSING": why.append("no translation")
        print(f"  {r['proc']:<30} {r.get('fortran_lines','?'):>5} "
              f"{r.get('jax_code_lines','?'):>6} {str(r.get('ratio','?')):>6}  "
              f"{r['verdict']:<7} {'; '.join(why)}")
    print(f"\n  PASS {n_pass}   WARN {n_warn}   FAIL {n_fail}   MISSING {n_miss}"
          f"   → status {summary['status']}")
    print(f"  JSON: {out_json}")

    # Durable markdown twin of the table (0d, 2026-08-25) — pasted into Step 8's
    # "Completeness / semantic audit" section; the JSON stays the gate input.
    out_md = out_dir / (out_json.stem + ".md").replace("completeness_check", "completeness_report")
    md = ["# Translation completeness report (TRANSLATE_WORKFLOW Step 2.5)", "",
          "| | |", "|---|---|",
          f"| run | {summary['timestamp']} |",
          f"| scope | {'procedure ' + only_proc if only_proc else 'all procedures'} |",
          f"| verdict | **{summary['status']}** — PASS {n_pass}, WARN {n_warn}, FAIL {n_fail}, MISSING {n_miss} |",
          "| rule | FAIL (scaffold phrase / commented-out callee) = translation STOPPED, "
          "`status aborted_scaffold`, no automatic retry — human decision |",
          "", "| procedure | Fortran lines | JAX code lines | ratio | verdict | why |",
          "|---|---|---|---|---|---|"]
    for r in results:
        why = []
        if r.get("scaffold_phrases"): why.append(f"scaffold phrases {r['scaffold_phrases']}")
        if r.get("dead_calls"): why.append(f"commented-out calls {r['dead_calls']}")
        if r.get("absent_calls"): why.append(f"absent calls {r['absent_calls']} (inlined or dropped?)")
        if r.get("scalar_misclassified"): why.append("scalar-only flag but Fortran declares arrays")
        if r.get("low_ratio_advisory"): why.append("very compact — verify")
        if r["verdict"] == "MISSING": why.append("no translation file")
        md.append(f"| {r['proc']} | {r.get('fortran_lines','')} | {r.get('jax_code_lines','')} | "
                  f"{r.get('ratio','')} | {r['verdict']} | {'; '.join(why).replace('|', '/')} |")
    md += ["", f"Machine-readable: `{out_json.relative_to(cfg.root) if out_json.is_absolute() else out_json}`", ""]
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"  Report: {out_md}")

    failed = [r for r in results if r["verdict"] == "FAIL"]
    if failed:
        if not no_abort:
            abort_on_scaffold(state_path, failed, out_json)
        print("\n" + "!" * 78)
        print("  SCAFFOLD DETECTED — TRANSLATION STOPPED (status: aborted_scaffold)")
        for r in failed:
            ev = []
            if r.get("scaffold_phrases"): ev.append(f"phrases {r['scaffold_phrases']}")
            if r.get("dead_calls"): ev.append(f"commented-out calls {r['dead_calls']}")
            if r.get("silent_stub"): ev.append(f"silent stub: {len(r['absent_calls'])} callees absent at ratio {r['ratio']}")
            print(f"    {r['proc']}: {'; '.join(ev)}  "
                  f"(Fortran {r.get('fortran_lines')} lines → JAX {r.get('jax_code_lines')} code lines)")
        print("  This is a model/context limit (output horizon), not a code defect.")
        print("  Do NOT re-run the same prompt — it reproduces the scaffold.")
        print("  HUMAN DECISION REQUIRED: split the procedure, change model or prompt")
        print("  policy, or archive as a failure experiment. Downstream gates stay closed.")
        print("!" * 78)
        return 2
    return 1 if n_miss else 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Flag scaffolded/stubbed translations (TRANSLATE_WORKFLOW Step 2.5). "
                    "Any FAIL stops the translation: workflow_state.json → aborted_scaffold, "
                    "exit 2, human decision required.")
    add_config_arg(ap)
    ap.add_argument("--proc", help="check ONE procedure (per-procedure Step 2.5 run, "
                                   "right after its final pass lands)")
    ap.add_argument("--no-abort", action="store_true",
                    help="report only; do not write aborted_scaffold to workflow_state.json")
    a = ap.parse_args()
    init_config(a.config)
    raise SystemExit(main(a.proc, a.no_abort))
