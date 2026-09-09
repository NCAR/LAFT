#!/usr/bin/env python3
"""Create and update translation workflow resume state.

This helper writes the persistent checkpoint used by
`workflow_translator/TRANSLATE_WORKFLOW.md`.

Why this exists
---------------
Runtime validation is submitted through PBS with
`qsub pbsJobs/jax_gpu_runtimevalid.sh`. A PBS job may sit in the queue longer
than the active assistant session. Without a durable checkpoint, a later resume
has to guess whether code generation, lint, runtime validation, driver output,
or comparison was the next step.

The checkpoint lives at:

    out/jax/workflow_state.json

On resume, read that JSON first and continue from its `status` and
`next_steps`. Do not restart the translation unless explicitly requested.

Common usage
------------
After generated code is written:

    python workflow_translator/translate_workflow_state.py \
      --status generated_code \
      --llm "OpenAI GPT-5 Codex (gpt-5)" \
      --complete-step "generated out/jax/<proc>.py"

After lint passes and the semantic audit (Step 3.5) is clean:

    python workflow_translator/translate_workflow_state.py \
      --status audit_passed \
      --complete-step "lint passed" \
      --complete-step "semantic audit passed (0 FAIL, N WARN)"

After the PBS runtime job is submitted (REFUSED unless the audit gate is open —
workflow_translator/audit_gate.py: _summary.json present, newer than every
out/jax/<proc>.py, total_fail == 0):

    python workflow_translator/translate_workflow_state.py \
      --status waiting_for_runtime_validation \
      --llm "OpenAI GPT-5 Codex (gpt-5)" \
      --job-kind runtime \
      --job-id 1234567.desched1 \
      --complete-step "runtime validation submitted"

The `--job-id` form automatically records the expected PBS log file
`runtimevalidate.o<job_number>` and the runtime-validation follow-up steps.

After runtime validation passes and the PBS driver job is submitted:

    python workflow_translator/translate_workflow_state.py \
      --status waiting_for_driver \
      --job-kind driver \
      --job-id 1234568.desched1 \
      --complete-step "runtime validation passed"

The driver job form records `<[hpc].driver_job_name>.o<job_number>` and the
comparison submission step (both from config/project.toml).

After the driver passes and the comparison PBS job is submitted:

    python workflow_translator/translate_workflow_state.py \
      --status waiting_for_comparison \
      --job-kind comparison \
      --job-id 1234569.desched1 \
      --complete-step "driver passed"

The comparison job form records `comparevalues.o<job_number>` and the final
report follow-up step.

After a later step completes:

    python workflow_translator/translate_workflow_state.py \
      --status driver_passed \
      --complete-step "driver passed" \
      --next-step "run the [comparison].script comparison (qsub pbsJobs/jax_gpu_compvalues.sh)"

Recognized status values are documented in TRANSLATE_WORKFLOW.md. This script
does not validate status names so the workflow can evolve without changing the
helper.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from framework_config import get_config  # noqa: E402

_CFG = get_config()

STATE_PATH = _CFG.jax_dir / "workflow_state.json"
"""Persistent resume-state file written by this helper."""

WORKFLOW_PATH = "workflow_translator/TRANSLATE_WORKFLOW.md"
"""Workflow document that defines status semantics and next-step rules."""

_DRIVER_SUMMARY = _CFG.section("driver").get("summary_json", "")
_DRIVER_OUTPUT = _CFG.section("driver").get("output_file", "")
_DRIVER_JOB = _CFG.section("hpc").get("driver_job", "")
_DRIVER_JOB_NAME = _CFG.section("hpc").get("driver_job_name", "driver")
_COMPARE_DIR = Path(_DRIVER_OUTPUT).parent if _DRIVER_OUTPUT else Path("out/driver")
_TRANSLATE_JOB = _CFG.section("translator").get("pbs_translate_job", "")
_TRANSLATE_JOB_NAME = _CFG.section("translator").get(
    "pbs_translate_job_name", "translate"
)


def _load_state() -> dict:
    """Load existing workflow state or return the default skeleton."""
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {
        "workflow": WORKFLOW_PATH,
        "status": "not_started",
        "llm": None,
        "updated_at": None,
        "completed_steps": [],
        "pbs_runtime_validation": None,
        "required_result_files": [
            str(_CFG.lint_dir / "_summary.json"),
            f"{_CFG.validation_dir}/<proc>_runtime.json (one per procedure)",
            _DRIVER_SUMMARY,
            str(_COMPARE_DIR / "compare_results_fortran_jax.txt"),
            str(_COMPARE_DIR / "comparison.json"),
        ],
        "next_steps": [],
    }


def _write_state(state: dict) -> None:
    """Write state JSON with a fresh timestamp, creating its parent directory."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _job_number(job_id: str) -> str:
    """Return the numeric PBS job id prefix used in `runtimevalidate.o<id>`."""
    return job_id.split(".", 1)[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Update out/jax/workflow_state.json so TRANSLATE_WORKFLOW.md can "
            "resume after PBS queue pauses or later validation steps."
        ),
        epilog=(
            "Example: python workflow_translator/translate_workflow_state.py "
            "--status waiting_for_runtime_validation "
            "--job-id 1234567.desched1 --complete-step 'lint passed'"
        ),
    )
    parser.add_argument(
        "--status",
        required=True,
        help=(
            "Workflow status to record, e.g. generated_code, lint_passed, "
            "audit_passed, waiting_for_runtime_validation (refused unless the "
            "semantic-audit gate is open), runtime_validation_passed, "
            "waiting_for_driver, driver_passed, waiting_for_comparison, "
            "comparison_passed, or complete."
        ),
    )
    parser.add_argument(
        "--llm",
        help='LLM name and exact model id, e.g. "OpenAI GPT-5 Codex (gpt-5)".',
    )
    parser.add_argument(
        "--job-kind",
        choices=(
            "translation",
            "runtime",
            "driver",
            "comparison",
        ),
        default="runtime",
        help=(
            "PBS job type to record when --job-id is provided. "
            "'translation' records the external translator job "
            "([translator].pbs_translate_job in config/project.toml); "
            "'runtime' records pbsJobs/jax_gpu_runtimevalid.sh; 'driver' records "
            "the project driver job ([hpc].driver_job in config/project.toml); "
            "'comparison' records pbsJobs/jax_gpu_compvalues.sh."
        ),
    )
    parser.add_argument(
        "--job-id",
        help=(
            "PBS job id returned by qsub. When set, records the qstat command, "
            "expected PBS log, and resume steps for --job-kind."
        ),
    )
    parser.add_argument(
        "--complete-step",
        action="append",
        default=[],
        help=(
            "Completed step to append. Can be supplied multiple times. Duplicate "
            "entries are ignored."
        ),
    )
    parser.add_argument(
        "--next-step",
        action="append",
        default=[],
        help=(
            "Next step to record. Can be supplied multiple times. Ignored when "
            "--job-id is provided because PBS resume steps are generated."
        ),
    )
    args = parser.parse_args()

    # Step 3.5 gate: runtime validation may only be recorded/submitted once the
    # semantic audit has run on the current translations with 0 unwaived FAIL
    # (workflow_translator/audit_gate.py — same rule the PBS script enforces).
    if (args.job_kind == "runtime" and args.job_id) or \
            args.status == "waiting_for_runtime_validation":
        from audit_gate import check_audit_gate
        ok, reason = check_audit_gate()
        if not ok:
            print(f"❌ REFUSED: {reason}", file=sys.stderr)
            print("   Record `--status audit_passed` after a clean audit "
                  "(TRANSLATE_WORKFLOW.md Step 3.5), then submit runtime validation.",
                  file=sys.stderr)
            return 1

    state = _load_state()
    state["workflow"] = WORKFLOW_PATH
    state["status"] = args.status
    if args.llm:
        state["llm"] = args.llm

    completed_steps = state.setdefault("completed_steps", [])
    # Timestamp every step as it lands. This is what makes per-procedure
    # wall-clock possible for in-context translation (a procedure's duration is
    # the gap between consecutive "generated out/jax/<proc>.py" stamps), and it
    # costs nothing: Step 2d already calls this script once per procedure.
    # Stamped HERE, by the script, so the figure is a recording rather than
    # something the driving agent reports about itself.
    step_timestamps = state.setdefault("step_timestamps", [])
    now = datetime.now().isoformat(timespec="seconds")
    for step in args.complete_step:
        if step not in completed_steps:
            completed_steps.append(step)
            step_timestamps.append({"step": step, "at": now})

    if args.job_id:
        job_num = _job_number(args.job_id)
        if args.job_kind == "translation":
            state["pbs_translation"] = {
                "job_id": args.job_id,
                "job_name": _TRANSLATE_JOB_NAME,
                "expected_log": f"{_TRANSLATE_JOB_NAME}.o{job_num}",
                "submit_command": f"qsub {_TRANSLATE_JOB}",
                "status_command": f"qstat {args.job_id}",
            }
            state["next_steps"] = [
                f"check qstat {args.job_id}",
                f"read {_TRANSLATE_JOB_NAME}.o{job_num} (must end DONE n/n, no FAILED)",
                f"verify finals in {_CFG.jax_dir}/ are freshly written with headers",
                "run lint (TRANSLATE_WORKFLOW.md Step 3)",
            ]
        elif args.job_kind == "runtime":
            state["pbs_runtime_validation"] = {
                "job_id": args.job_id,
                "job_name": "runtimevalidate",
                "expected_log": f"runtimevalidate.o{job_num}",
                "submit_command": "qsub pbsJobs/jax_gpu_runtimevalid.sh",
                "status_command": f"qstat {args.job_id}",
            }
            state["next_steps"] = [
                f"check qstat {args.job_id}",
                f"read runtimevalidate.o{job_num}",
                f"read {_CFG.validation_dir}/<proc>_runtime.json for each procedure",
                f"submit {_DRIVER_JOB}",
            ]
        elif args.job_kind == "driver":
            state["pbs_driver"] = {
                "job_id": args.job_id,
                "job_name": _DRIVER_JOB_NAME,
                "expected_log": f"{_DRIVER_JOB_NAME}.o{job_num}",
                "submit_command": f"qsub {_DRIVER_JOB}",
                "status_command": f"qstat {args.job_id}",
            }
            state["next_steps"] = [
                f"check qstat {args.job_id}",
                f"read {_DRIVER_JOB_NAME}.o{job_num}",
                f"verify {_DRIVER_OUTPUT} is freshly written",
                f"read {_DRIVER_SUMMARY} (convenience; PBS log is authoritative)",
                "submit pbsJobs/jax_gpu_compvalues.sh",
            ]
        else:
            state["pbs_comparison"] = {
                "job_id": args.job_id,
                "job_name": "comparevalues",
                "expected_log": f"comparevalues.o{job_num}",
                "submit_command": "qsub pbsJobs/jax_gpu_compvalues.sh",
                "status_command": f"qstat {args.job_id}",
            }
            state["next_steps"] = [
                f"check qstat {args.job_id}",
                f"read comparevalues.o{job_num}",
                f"read {_COMPARE_DIR / 'compare_results_fortran_jax.txt'}",
                f"read {_COMPARE_DIR / 'comparison.json'} (convenience; check invariants + status)",
                f"write {_CFG.jax_dir / 'translation_report.md'}",
            ]
    elif args.next_step:
        state["next_steps"] = args.next_step

    _write_state(state)
    print(STATE_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
