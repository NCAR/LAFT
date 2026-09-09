#!/usr/bin/env python3
"""Create and update profiling workflow resume state.

This helper writes the persistent checkpoint used by
`workflow_profiler/PROFILE_WORKFLOW.md`.

Why this exists
---------------
The profiling loop submits one PBS job per iteration:

    profile_trace (jax.profiler.trace + HLO)  ->  diagnose (local)
                                              ->  fix (in-context)

(the `profile` job kind — the nsys hardware pass — is the OPTIONAL
post-convergence characterization, run once for the report, not part of
the loop), and the final re-validation chain reuses the translator's
three PBS gates:

    revalidation_runtime  ->  revalidation_driver  ->  revalidation_comparison

A PBS job may sit in the queue longer than the active assistant session.
Without a durable checkpoint, a later resume has to guess whether profile,
diagnose, fix, or re-validation was the next step, and which iteration of
the fix loop is in progress.

The checkpoint lives at:

    out/profiled/profile_state.json

On resume, read that JSON first and continue from its `status`,
`iteration`, and `next_steps`. Do not restart the loop unless explicitly
requested.

Common usage
------------
After files are staged into out/profiled/ for iteration 1 (--hash-staged-code
records the baseline hash of the validated translation):

    python workflow_profiler/profile_workflow_state.py \
      --status staged \
      --llm "Claude Opus 4.7 (claude-opus-4-7)" \
      --iteration 1 \
      --hash-staged-code \
      --complete-step "staged <[profiler].staged_code> for iteration 1"

Phase A submits ONE PBS job — the jax.profiler.trace design pass:

    python workflow_profiler/profile_workflow_state.py \
      --status waiting_for_profile \
      --iteration 1 \
      --job-kind profile_trace \
      --job-id 1234568.desched1 \
      --complete-step "submitted jax_trace profile iteration 1"

The `--job-id` form automatically records the expected PBS log file
and the diagnose follow-up step.

After diagnose runs:

    python workflow_profiler/profile_workflow_state.py \
      --status diagnose_passed \
      --iteration 1 \
      --complete-step "diagnose iteration 1: production_ready"

After a fix is applied (the assistant is also expected to append to the
`fix_attempts` array in the JSON directly; this helper only bumps the
status and the counter):

    python workflow_profiler/profile_workflow_state.py \
      --status staged \
      --iteration 2 \
      --bump-fix-attempts \
      --complete-step "fix iteration 1: replaced fori_loop with vmap" \
      --complete-step "staged <[profiler].staged_code> for iteration 2"

During the final re-validation chain (one --job-id per step, gating on
qstat between submissions):

    python workflow_profiler/profile_workflow_state.py \
      --status waiting_for_revalidation_runtime \
      --iteration 2 \
      --job-kind revalidation_runtime \
      --job-id 1234568.desched1 \
      --complete-step "submitted revalidation runtime"

    python workflow_profiler/profile_workflow_state.py \
      --status waiting_for_revalidation_driver \
      --job-kind revalidation_driver \
      --job-id 1234569.desched1 \
      --complete-step "revalidation runtime passed"

    python workflow_profiler/profile_workflow_state.py \
      --status waiting_for_revalidation_comparison \
      --job-kind revalidation_comparison \
      --job-id 1234570.desched1 \
      --complete-step "revalidation driver passed"

Recognized status values are documented in PROFILE_WORKFLOW.md. This script
does not validate status names so the workflow can evolve without changing
the helper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from framework_config import get_config  # noqa: E402

_CFG = get_config()
_PROF = _CFG.section("profiler")
_PROC = _PROF.get("target_proc", "target")
_OUT = Path(_PROF.get("out_dir", "out/profiled"))
_STAGED = Path(_PROF.get("staged_code", f"out/profiled/{_PROC}.py"))
_DRIVER_JOB = _CFG.section("hpc").get("driver_job", "")
_DRIVER_JOB_NAME = _CFG.section("hpc").get("driver_job_name", "driver")
_DRIVER_OUTPUT = _CFG.section("driver").get("output_file", "")
_DRIVER_SUMMARY = _CFG.section("driver").get("summary_json", "")
_COMPARE_DIR = Path(_DRIVER_OUTPUT).parent if _DRIVER_OUTPUT else Path("out/driver")

STATE_PATH = _OUT / "profile_state.json"
"""Persistent resume-state file written by this helper."""

WORKFLOW_PATH = "workflow_profiler/PROFILE_WORKFLOW.md"
"""Workflow document that defines status semantics and next-step rules."""

FIX_ATTEMPTS_MAX = 5
"""Mirror of PROFILE_WORKFLOW.md fix-attempt budget."""


def _load_state() -> dict:
    """Load existing profiling state or return the default skeleton."""
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {
        "workflow": WORKFLOW_PATH,
        "status": "not_started",
        "llm": None,
        "updated_at": None,
        "iteration": 0,
        "staged_code_sha256": None,
        "completed_steps": [],
        "pbs_profile": None,
        "pbs_profile_trace": None,
        "pbs_revalidation_runtime": None,
        "pbs_revalidation_driver": None,
        "pbs_revalidation_comparison": None,
        "required_result_files": [],
        "next_steps": [],
        "fix_attempts": [],
        "fix_attempts_used": 0,
        "fix_attempts_max": FIX_ATTEMPTS_MAX,
    }


def _write_state(state: dict) -> None:
    """Write state JSON with a fresh timestamp, creating its parent directory."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _job_number(job_id: str) -> str:
    """Return the numeric PBS job id prefix used in `<name>.o<id>`."""
    return job_id.split(".", 1)[0]


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file's contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _iteration_dir(iteration: int) -> str:
    """Path (relative) to the per-iteration artifact directory."""
    return str(_OUT / f"iteration_{iteration}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Update out/profiled/profile_state.json so PROFILE_WORKFLOW.md "
            "can resume after PBS queue pauses or fix-loop iterations."
        ),
        epilog=(
            "Example: python workflow_profiler/profile_workflow_state.py "
            "--status waiting_for_profile --iteration 1 "
            "--job-kind profile --job-id 1234567.desched1 "
            "--complete-step 'submitted profile iteration 1'"
        ),
    )
    parser.add_argument(
        "--status",
        required=True,
        help=(
            "Workflow status to record, e.g. staged, waiting_for_profile, "
            "profile_passed, diagnose_failed, diagnose_passed, "
            "waiting_for_revalidation_runtime, "
            "waiting_for_revalidation_driver, "
            "waiting_for_revalidation_comparison, "
            "revalidation_passed, complete, aborted_max_fix_attempts."
        ),
    )
    parser.add_argument(
        "--llm",
        help='LLM name and exact model id, e.g. "Claude Opus 4.7 (claude-opus-4-7)".',
    )
    parser.add_argument(
        "--iteration",
        type=int,
        help=(
            "Current profile/diagnose/fix iteration number (1-based). If "
            "omitted, the iteration field is left unchanged."
        ),
    )
    parser.add_argument(
        "--job-kind",
        choices=(
            "profile",
            "profile_trace",
            "revalidation_runtime",
            "revalidation_driver",
            "revalidation_comparison",
        ),
        default="profile_trace",
        help=(
            "PBS job type to record when --job-id is provided. "
            "'profile_trace' records pbsJobs/jax_gpu_profile_trace.sh — the "
            "iteration loop's single profiling job (jax.profiler.trace + "
            "compiled HLO); 'profile' records pbsJobs/jax_gpu_profile_nsys.sh "
            "— the MANDATORY once-post-convergence hardware pass, not "
            "part of the iteration loop; the three "
            "'revalidation_*' kinds record the translator's runtime, "
            "driver, and comparison PBS scripts."
        ),
    )
    parser.add_argument(
        "--job-id",
        help=(
            "PBS job id returned by qsub. When set, records the qstat "
            "command, expected PBS log, required result files, and resume "
            "steps for --job-kind."
        ),
    )
    parser.add_argument(
        "--complete-step",
        action="append",
        default=[],
        help=(
            "Completed step to append. Can be supplied multiple times. "
            "Duplicate entries are ignored."
        ),
    )
    parser.add_argument(
        "--next-step",
        action="append",
        default=[],
        help=(
            "Next step to record. Can be supplied multiple times. Ignored "
            "when --job-id is provided because PBS resume steps are generated."
        ),
    )
    parser.add_argument(
        "--bump-fix-attempts",
        action="store_true",
        help=(
            "Increment fix_attempts_used by 1. Use after applying a fix in "
            "response to a failed diagnose or failed re-validation. The "
            "assistant is separately responsible for appending the new entry "
            "to the fix_attempts array (richer schema than this flag covers)."
        ),
    )
    parser.add_argument(
        "--hash-staged-code",
        action="store_true",
        help=(
            "Record a SHA-256 of the staged code ([profiler].staged_code) "
            "into staged_code_sha256. Use once, at iteration-1 staging: it "
            "captures the validated translation as the baseline so the final "
            "re-validation step can be skipped when the profiler changed "
            "nothing."
        ),
    )
    args = parser.parse_args()

    state = _load_state()
    state["workflow"] = WORKFLOW_PATH
    state["status"] = args.status
    if args.llm:
        state["llm"] = args.llm
    if args.iteration is not None:
        state["iteration"] = args.iteration

    completed_steps = state.setdefault("completed_steps", [])
    for step in args.complete_step:
        if step not in completed_steps:
            completed_steps.append(step)

    if args.bump_fix_attempts:
        state["fix_attempts_used"] = int(state.get("fix_attempts_used", 0)) + 1
        state.setdefault("fix_attempts_max", FIX_ATTEMPTS_MAX)

    if args.hash_staged_code:
        code_path = _STAGED
        if code_path.is_file():
            state["staged_code_sha256"] = _sha256(code_path)
        else:
            print(f"WARNING: {code_path} not found; staged_code_sha256 not recorded")

    iteration = int(state.get("iteration") or 0)

    if args.job_id:
        job_num = _job_number(args.job_id)
        if args.job_kind == "profile_trace":
            # The iteration loop's single profiling job: writes everything
            # diagnose.py needs (trace metrics + hlo block + code_info).
            it_dir = _iteration_dir(iteration)
            state["pbs_profile_trace"] = {
                "job_id": args.job_id,
                "job_name": "jax_trace",
                "expected_log": f"out/jobs/jax_trace.o{job_num}",
                "submit_command": f"qsub -v ITERATION={iteration} pbsJobs/jax_gpu_profile_trace.sh",
                "status_command": f"qstat {args.job_id}",
                "iteration": iteration,
                "iteration_dir": it_dir,
            }
            state["required_result_files"] = [
                f"{it_dir}/{_PROC}_trace.json",
                f"{it_dir}/{_PROC}_code_info.json",
            ]
            state["next_steps"] = [
                f"check qstat {args.job_id}",
                f"read out/jobs/jax_trace.o{job_num}",
                f"verify {it_dir}/{_PROC}_trace.json and "
                f"{it_dir}/{_PROC}_code_info.json exist",
                (
                    "run python workflow_profiler/diagnose.py "
                    f"--trace-json {it_dir}/{_PROC}_trace.json "
                    f"--code-info {it_dir}/{_PROC}_code_info.json "
                    f"--out {it_dir}/{_PROC}_diagnosis.json"
                ),
            ]
        elif args.job_kind == "profile":
            # OPTIONAL post-convergence hardware characterization (nsys).
            # Not part of the iteration loop; run once for the report.
            it_dir = _iteration_dir(iteration)
            state["pbs_profile"] = {
                "job_id": args.job_id,
                "job_name": "jax_profile",
                "expected_log": f"out/jobs/jax_profile.o{job_num}",
                "submit_command": f"qsub -v ITERATION={iteration} pbsJobs/jax_gpu_profile_nsys.sh",
                "status_command": f"qstat {args.job_id}",
                "iteration": iteration,
                "iteration_dir": it_dir,
            }
            state["required_result_files"] = [
                f"{it_dir}/{_PROC}_stats.json",
            ]
            state["next_steps"] = [
                f"check qstat {args.job_id}",
                f"read out/jobs/jax_profile.o{job_num}",
                f"verify {it_dir}/{_PROC}_stats.json exists",
                (
                    "optionally attach hardware corroboration: "
                    "python workflow_profiler/diagnose.py "
                    f"--trace-json {it_dir}/{_PROC}_trace.json "
                    f"--code-info {it_dir}/{_PROC}_code_info.json "
                    f"--nsys-json {it_dir}/{_PROC}_stats.json "
                    f"--out {it_dir}/{_PROC}_diagnosis.json"
                ),
                "cite the hardware numbers in out/reports/profile/profile_report.md",
            ]
        elif args.job_kind == "revalidation_runtime":
            state["pbs_revalidation_runtime"] = {
                "job_id": args.job_id,
                "job_name": "runtimevalidate",
                "expected_log": f"out/jobs/runtimevalidate.o{job_num}",
                "submit_command": "qsub pbsJobs/jax_gpu_runtimevalid.sh",
                "status_command": f"qstat {args.job_id}",
                "iteration": iteration,
            }
            state["required_result_files"] = [
                f"{_CFG.validation_dir}/<proc>_runtime.json (one per procedure)",
            ]
            state["next_steps"] = [
                f"check qstat {args.job_id}",
                f"read out/jobs/runtimevalidate.o{job_num}",
                f"read {_CFG.validation_dir}/<proc>_runtime.json for each procedure",
                f"if runtime passed, submit {_DRIVER_JOB}",
            ]
        elif args.job_kind == "revalidation_driver":
            state["pbs_revalidation_driver"] = {
                "job_id": args.job_id,
                "job_name": _DRIVER_JOB_NAME,
                "expected_log": f"out/jobs/{_DRIVER_JOB_NAME}.o{job_num}",
                "submit_command": f"qsub {_DRIVER_JOB}",
                "status_command": f"qstat {args.job_id}",
                "iteration": iteration,
            }
            state["required_result_files"] = [
                _DRIVER_OUTPUT,
                _DRIVER_SUMMARY,
            ]
            state["next_steps"] = [
                f"check qstat {args.job_id}",
                f"read out/jobs/{_DRIVER_JOB_NAME}.o{job_num}",
                f"verify {_DRIVER_OUTPUT} is freshly written",
                "if driver passed, submit pbsJobs/jax_gpu_compvalues.sh",
            ]
        else:  # revalidation_comparison
            state["pbs_revalidation_comparison"] = {
                "job_id": args.job_id,
                "job_name": "comparevalues",
                "expected_log": f"out/jobs/comparevalues.o{job_num}",
                "submit_command": "qsub pbsJobs/jax_gpu_compvalues.sh",
                "status_command": f"qstat {args.job_id}",
                "iteration": iteration,
            }
            state["required_result_files"] = [
                str(_COMPARE_DIR / "compare_results_fortran_jax.txt"),
            ]
            state["next_steps"] = [
                f"check qstat {args.job_id}",
                f"read out/jobs/comparevalues.o{job_num}",
                f"read {_COMPARE_DIR / 'compare_results_fortran_jax.txt'}",
                "if comparison passed, write out/reports/profile/profile_report.md",
            ]
    elif args.next_step:
        state["next_steps"] = args.next_step

    _write_state(state)
    print(STATE_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
