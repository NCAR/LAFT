# Translator module — Qwen (external initial translation via vLLM on PBS)

This module replaces **only §2a** of `workflow_translator/TRANSLATE_WORKFLOW.md`
when `config/project.toml` has `[translator] source = "qwen"`. Everything
else — the plan (Step 1), header rule (§2b), hard rules (§2c), state updates
(§2d), lint, semantic audit, PBS validation, the fix loop, and the report —
comes from the main workflow document. **Repair is always the driving agent's:
Qwen is never re-invoked to fix its own output.**

## The paper-1 configuration

The paper's Qwen archive, `kessler/translations/qwen25-32b-jd/`, was authored
in April 2026 with `Qwen2.5-Coder-32B-Instruct` through vLLM offline inference,
on the same four-pass prompts every model received. That configuration is
shipped as **`pbsJobs/pbs_qwen25_translate.sh`**, which runs this module's
driver (`qwen_API_translate.py`) with these settings fixed:

| setting | value |
|---|---|
| model | `Qwen2.5-Coder-32B-Instruct`, bfloat16, local weights |
| context window | `max_model_len` 24,576 (the model's native window is 32,768) |
| output cap | 16,384 tokens per pass, fixed |
| sampling | temperature 0.1; top-p, top-k, min-p, repetition penalty at vLLM defaults; unseeded |
| reasoning / RoPE scaling | none (Qwen2.5 has no thinking mode) |
| GPUs | 2 (tensor parallel; ~65 GB of BF16 weights on A100-40GB) |
| environment | `qwen-vllm` — `envs/qwen-vllm.yml`, vLLM 0.15.1 / Python 3.11 |

Sampling at temperature 0.1 is not bit-reproducible, so a rerun reproduces
the configuration, not the archived files; the archived files are what the
paper validated. (The archive itself was re-validated through the pipeline
with `[translator] source = "in_context"` — see `kessler/config/project.toml`.)

## Configuration

`[translator]` in `config/project.toml`:

```toml
source                 = "qwen"
module                 = "workflow_translator/qwen/qwen.md"
llm_label              = "Qwen2.5-32B (Qwen2.5-Coder-32B-Instruct via vLLM)"
pbs_translate_job      = "pbsJobs/pbs_qwen25_translate.sh"
pbs_translate_job_name = "qwen25_translate"     # #PBS -N (log = <name>.o<id>)
```

The driver stamps `llm_label` into every output file's version header.

## How the initial translation runs

One PBS job authors everything: it loads the model once and batches each pass
stage across all procedures via vLLM (procedures are independent, so
translation order does not matter for authoring — topological order still
governs the Step 1 plan and the report).

```bash
qsub pbsJobs/pbs_qwen25_translate.sh                           # all procedures
qsub -v PROCS=proc_a,proc_b pbsJobs/pbs_qwen25_translate.sh    # re-roll subset
qsub -v PROCS=all,SKIP_EXISTING=1 pbsJobs/pbs_qwen25_translate.sh  # resume
qsub -v MODEL=/path/to/Qwen2.5-Coder-32B-Instruct,QWEN_ENV=qwen-vllm ...  # paths
```

Record the job id immediately:

```bash
python workflow_translator/translate_workflow_state.py \
  --status waiting_for_translation \
  --llm "Qwen2.5-32B (Qwen2.5-Coder-32B-Instruct via vLLM)" \
  --job-kind translation \
  --job-id <job_id>
```

The job writes intermediates (`out/jax/{proc}_pass{n}.py`) and finals
(`out/jax/{proc}.py`) with the LLM version header already stamped. Expected
duration: model load a few minutes + one batched generation per pass stage;
budget well under an hour for Kessler (walltime 2 h); log is
`qwen25_translate.o<job_id>`. Every request's budget and `finish_reason` is
logged to `out/reports/analysis/generation_log.jsonl`.

## After the job leaves the queue (freshness rule applies)

1. Read `qwen25_translate.o<job_id>`: it must end with
   `DONE: <n>/<n> procedures translated` and no `FAILED:` lines, no traceback.
   The driver exits non-zero if any procedure failed a pass (empty code block
   or truncated generation) — those procedures' files are stale or absent.
2. Confirm every final `out/jax/{proc}.py` in the plan is freshly written
   (mtime at or after job finish) and starts with the version header.
3. On success: set `status="generated_code"` and rejoin the main workflow at
   Step 3 (lint). On per-procedure failures: re-roll just those procedures
   with `qsub -v PROCS=<failed,...>`; if the same procedure fails authoring
   twice, escalate to the user (the workflow does not edit prompt files).

## Extra status values (state file)

- `waiting_for_translation` — translation job submitted; finals not verified.
- `translation_failed` — job crashed or a procedure failed both authoring
  attempts. Environmental causes (model path, GPU memory, vLLM env): resubmit.
  Prompt-content causes: escalate.

## Free re-roll on first lint failure

If the **first** lint run after the initial translation fails on some
procedure and `workflow_state.json` has `free_retranslation_used: false` (or
absent), you may re-submit that procedure's translation once
(`qsub -v PROCS=<proc>`) instead of hand-fixing — a re-roll of the source, not
a fix, so it does **not** count against the 5-attempt budget. Set
`free_retranslation_used: true`. Afterwards every failure — including a second
lint failure on the same procedure — is repaired by direct edit per the main
workflow's Step 6.

## Fix-time header rule

When you patch a Qwen-authored file, keep
`# Translated by: Qwen2.5-32B (Qwen2.5-Coder-32B-Instruct via vLLM)` (Qwen
remains the original author) and set `# Pass: fix-attempt-<N>` per the main
workflow.

## Other Qwen models

The driver is model-generic: it reads the context window from the model
directory's `config.json` and exposes the sampling, window, seed, RoPE-scaling
and reasoning-mode (`--enable-thinking`, for the Qwen3 family) options on its
command line (`python workflow_translator/qwen/qwen_API_translate.py --help`).
To run another model, copy `pbs_qwen25_translate.sh`, change the model path,
GPU count and flags, and set `llm_label` accordingly; the paper-1 job itself
keeps its settings fixed so that it stays a faithful record.
