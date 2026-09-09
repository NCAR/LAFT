#!/usr/bin/env python3
"""
Bridge workflow test-gate runner — runs the project's bridge_test/ suite and
writes the durable gate artifacts:

    out/reports/bridge/bridge_test_results.json   (top-level PASS/FAIL status)
    out/reports/bridge/bridge_report.md           (generation + analysis + gate)

Gate rule (translation-independent): every layout/wiring test
(bridge_test/test_*_layout.py) must run and pass — no skips among them. The
translation-DEPENDENT suite is executed in the same pytest run but only
reported: skipped while no translation exists (normal during the bridge
workflow), pass/fail once the translator workflow has produced out/jax/.

With --require-dependent (translator workflow, Step 4.5) the gate
additionally requires every translation-dependent test to run and PASS —
skipped or absent dependent tests then FAIL the gate.

Exit code: 0 when the gate PASSes, 1 otherwise — usable as a scripted gate.

Run from the project root:
    python workflow_bridge/run_bridge_tests.py                      # bridge workflow gate
    python workflow_bridge/run_bridge_tests.py --require-dependent  # translator validation
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

# workflow_bridge/ is a per-project SYMLINK to LAFT/workflow_bridge/ — resolve
# back to the sibling LAFT/config/ only to import framework_config; per-project paths come
# from get_config(), which walks up from the cwd (same convention as
# phase03_make_bridge.py / workflow_profiler/).
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "config"))

from framework_config import get_config, init as init_config, add_config_arg  # noqa: E402


def run_pytest(project_root: Path, junit_xml: Path) -> int:
    """Run the whole bridge_test/ suite; dependent files self-skip as needed."""
    cmd = [sys.executable, "-m", "pytest", "bridge_test/", "-v",
           f"--junitxml={junit_xml}"]
    print(f"▶ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(project_root))
    return proc.returncode


def parse_junit(junit_xml: Path) -> list[dict]:
    """Flatten junit XML into [{file, test, outcome, message}]."""
    tests = []
    root = ET.parse(junit_xml).getroot()
    for case in root.iter("testcase"):
        outcome, message = "passed", ""
        for tag in ("failure", "error", "skipped"):
            node = case.find(tag)
            if node is not None:
                outcome = tag if tag != "failure" else "failed"
                message = (node.get("message") or "").strip()
                break
        tests.append({
            "file": case.get("classname", ""),   # e.g. bridge_test.test_x_layout
            "test": f"{case.get('classname', '').split('.')[-1]}::{case.get('name', '')}",
            "outcome": outcome,
            "message": message,
        })
    return tests


def collect_analysis(reports_dir: Path) -> list[dict]:
    """Summarize phase03 *_analysis.txt files (safety line + GPU warning)."""
    summaries = []
    for f in sorted(reports_dir.glob("*_analysis.txt")):
        text = f.read_text(encoding="utf-8")
        safety = next((ln.strip() for ln in text.splitlines()
                       if ln.strip().startswith("Safety:")), "Safety: ?")
        summaries.append({
            "file": f.name,
            "safety": safety,
            "gpu_warning": "GPU EFFICIENCY WARNING" in text,
        })
    return summaries


def main() -> int:
    ap = argparse.ArgumentParser(description="Bridge test-gate runner + report")
    add_config_arg(ap)
    ap.add_argument(
        "--require-dependent", action="store_true",
        help="Also require the translation-dependent suite to run and pass "
             "(translator workflow, Step 4.5). Skipped/absent dependent "
             "tests then fail the gate.")
    args = ap.parse_args()
    init_config(args.config)

    cfg = get_config()
    project_root = cfg.root
    reports_dir = cfg.bridge_reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    test_dir = project_root / "bridge_test"
    if not test_dir.is_dir():
        print(f"❌ {test_dir} not found — this project has no bridge tests yet "
              "(see BRIDGE_WORKFLOW.md, 'New project checklist').")
        return 1

    junit_xml = reports_dir / "bridge_test_results.xml"
    exit_code = run_pytest(project_root, junit_xml)
    if not junit_xml.exists():
        print("❌ pytest produced no junit XML — collection crashed; see output above.")
        return 1

    tests = parse_junit(junit_xml)
    layout = [t for t in tests if "_layout" in t["file"]]
    dependent = [t for t in tests if "_layout" not in t["file"]]

    def count(rows, outcome):
        return sum(1 for t in rows if t["outcome"] == outcome)

    dependent_state = ("not_present" if not dependent else
                       "skipped" if count(dependent, "skipped") == len(dependent) else
                       "passed" if count(dependent, "passed") == len(dependent) else
                       "failed")

    gate_pass = (
        len(layout) > 0
        and count(layout, "passed") == len(layout)          # all layout tests green
        and count(tests, "failed") == 0
        and count(tests, "error") == 0                      # nothing anywhere broke
    )
    if args.require_dependent and dependent_state != "passed":
        gate_pass = False
    status = "PASS" if gate_pass else "FAIL"

    gate_rule = ("all bridge_test/test_*_layout.py tests pass, zero skips "
                 "among them, no failures/errors anywhere")
    if args.require_dependent:
        gate_rule += ("; AND every translation-dependent test runs and "
                      "passes (--require-dependent)")

    analysis = collect_analysis(reports_dir)
    bridges = sorted(p.name for p in cfg.bridge_dir.glob("*_bridge.py"))
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result = {
        "status": status,                       # gate verdict, top-level (PASS/FAIL)
        "date": now,
        "pytest_exit_code": exit_code,
        "require_dependent": args.require_dependent,
        "gate_rule": gate_rule,
        "bridges": bridges,
        "layout_tests": {"total": len(layout),
                         "passed": count(layout, "passed"),
                         "skipped": count(layout, "skipped")},
        "translation_dependent_tests": {"total": len(dependent),
                                        "passed": count(dependent, "passed"),
                                        "failed": count(dependent, "failed"),
                                        "skipped": count(dependent, "skipped"),
                                        "state": dependent_state},
        "analysis_reports": analysis,
        "tests": tests,
    }
    json_path = reports_dir / "bridge_test_results.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Bridge workflow report",
        "",
        f"- **Date**: {now}",
        f"- **Gate status**: **{status}**",
        f"- Gate rule: {result['gate_rule']}",
        "",
        "## Generated bridges",
        "",
    ]
    md += [f"- `out/bridge/{b}`" for b in bridges] or ["- (none found)"]
    md += ["", "## Phase 03 analysis reports", ""]
    if analysis:
        for a in analysis:
            gpu = " — ⚠️ GPU efficiency warning" if a["gpu_warning"] else ""
            md.append(f"- `{a['file']}`: {a['safety']}{gpu}")
    else:
        md.append("- (none found — run workflow_bridge/phase03_make_bridge.py)")
    md += ["", "## Test gate (translation-independent)", "",
           "| Test | Outcome |", "|------|---------|"]
    md += [f"| `{t['test']}` | {t['outcome']} |" for t in layout]
    md += ["", "## Translation-dependent suite (informational)", "",
           f"State: **{dependent_state}**",
           ""]
    if dependent:
        md += ["| Test | Outcome |", "|------|---------|"]
        md += [f"| `{t['test']}` | {t['outcome']}"
               + (f" — {t['message']}" if t["outcome"] == "skipped" and t["message"] else "")
               + " |" for t in dependent]
        md += [""]
    md += ["These tests belong to the translator workflow's validation stage; "
           "`skipped` is the normal state before a translation exists.",
           "",
           f"Machine-readable verdict: `{json_path.relative_to(project_root) if json_path.is_absolute() else json_path}`",
           ""]
    md_path = reports_dir / "bridge_report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"\n{'✅' if gate_pass else '❌'} Bridge test gate: {status}")
    print(f"   JSON:   {json_path}")
    print(f"   Report: {md_path}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
