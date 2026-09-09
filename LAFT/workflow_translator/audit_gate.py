#!/usr/bin/env python3
"""
Semantic-audit gate (TRANSLATE_WORKFLOW.md Step 3.5 → Step 4).

Runtime validation must not be submitted before the semantic audit
(phase05_01b_semantic_audit.py) has been run on the CURRENT translations and
has no unwaived FAIL. This module is the single place that decides that, and
it is consulted by:

  * translate_workflow_state.py  — refuses `--job-kind runtime` / status
                                    `waiting_for_runtime_validation`
  * pbsJobs/jax_gpu_runtimevalid.sh — hard stop before the validator runs
  * `python workflow_translator/audit_gate.py` — manual check (exit 0 = open)

Rule (all must hold):
  0. workflow_state.json status is not `aborted_scaffold`, and
     out/reports/translation/completeness_check.json (Step 2.5) exists, is
     newer than every final translation, and is not FAIL;
  1. out/issues/semantic_audit/_summary.json exists;
  2. it is newer than every final translation out/jax/<proc>.py
     (pass intermediates and .validated snapshots are ignored);
  3. its total_fail is 0 (the audit already excludes WAIVED findings).

Why: on 2026-08-24 (in an earlier LAFT project) the audit was skipped between lint and runtime
validation because nothing enforced it — four semantic defects then cost four
driver/comparison PBS cycles.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from framework_config import get_config  # noqa: E402

_CFG = get_config()
PROJECT_ROOT = _CFG.root
JAX_DIR = PROJECT_ROOT / _CFG.jax_dir
SUMMARY = PROJECT_ROOT / _CFG.issues_dir / "semantic_audit" / "_summary.json"
AUDIT_CMD = "python workflow_translator/phase05_01b_semantic_audit.py"


def _final_translations() -> list[Path]:
    return [p for p in JAX_DIR.glob("*.py")
            if p.stem != "__init__"
            and not p.stem.endswith(".validated")
            and not re.search(r"_pass\d+$", p.stem)]


STATE = JAX_DIR / "workflow_state.json"
COMPLETENESS = PROJECT_ROOT / _CFG.reports_dir / "translation" / "completeness_check.json"
COMPLETENESS_CMD = "python workflow_translator/phase04_04_completeness_check.py"


def check_audit_gate() -> tuple[bool, str]:
    """Return (open, reason). `open` is True iff runtime validation may run."""
    finals = _final_translations()
    if not finals:
        return False, f"no final translations in {JAX_DIR}"
    newest = max(finals, key=lambda p: p.stat().st_mtime)
    # Step 2.5 — scaffold stop and completeness check come BEFORE the audit
    if STATE.exists():
        try:
            st = json.loads(STATE.read_text(encoding="utf-8"))
        except ValueError:
            st = {}
        if st.get("status") == "aborted_scaffold":
            procs = [p["proc"] for p in st.get("scaffold_abort", {}).get("procedures", [])]
            return False, (f"translation is STOPPED on scaffold ({', '.join(procs)}) — "
                           "human decision required (workflow_state.json → scaffold_abort); "
                           f"after fixing/re-authoring re-run `{COMPLETENESS_CMD}`")
    if not COMPLETENESS.exists():
        return False, (f"completeness check has not been run: {COMPLETENESS} missing — "
                       f"run `{COMPLETENESS_CMD}` (Step 2.5) first")
    if COMPLETENESS.stat().st_mtime < newest.stat().st_mtime:
        return False, (f"completeness check is STALE: {newest.name} was modified after "
                       f"{COMPLETENESS.name} — re-run `{COMPLETENESS_CMD}`")
    try:
        comp = json.loads(COMPLETENESS.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"cannot read {COMPLETENESS}: {exc}"
    if comp.get("status") == "FAIL":
        return False, (f"completeness check FAILED ({comp.get('fail', '?')} scaffolded, "
                       f"{comp.get('missing', '?')} missing) — human decision required")
    if not SUMMARY.exists():
        return False, (f"semantic audit has not been run: {SUMMARY} missing "
                       f"— run `{AUDIT_CMD}` (Step 3.5) first")
    if SUMMARY.stat().st_mtime < newest.stat().st_mtime:
        return False, (f"semantic audit is STALE: {newest.name} was modified after "
                       f"{SUMMARY.name} — re-run `{AUDIT_CMD}`")
    try:
        summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"cannot read {SUMMARY}: {exc}"
    n_fail = int(summary.get("total_fail", 0))
    if n_fail:
        return False, (f"semantic audit has {n_fail} unwaived FAIL finding(s) — fix "
                       f"the translation (or add a justified waiver to "
                       f"out/issues/semantic_audit_waivers.json) and re-run the audit")
    report_md = PROJECT_ROOT / _CFG.reports_dir / "translation" / "semantic_audit_report.md"
    if not report_md.exists() or report_md.stat().st_mtime < newest.stat().st_mtime:
        return False, (f"semantic audit report {report_md.name} is missing or stale — "
                       f"re-run `{AUDIT_CMD}` (it writes the report with the summary)")
    n_warn = int(summary.get("total_warn", 0))
    n_procs = len(summary.get("procs", []))
    return True, (f"semantic audit gate OPEN: {n_procs} procs, 0 FAIL, {n_warn} WARN "
                  f"({SUMMARY.name} at {summary.get('timestamp', '?')})")


def main() -> int:
    ok, reason = check_audit_gate()
    print(("✅ " if ok else "❌ ") + reason)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
