"""
qwen_API_translate.py
=====================
Project-generic Fortran -> JAX translation driver using a local Qwen model via
vLLM offline inference (the paper-1 job, pbsJobs/pbs_qwen25_translate.sh, runs
it with Qwen2.5-Coder-32B-Instruct). Codebase-agnostic and batch-oriented:

- The procedure list and per-procedure pass counts come from
  out/packets/_ALL_deps.json (same rules as TRANSLATE_WORKFLOW.md Step 1b):
      scalar_only & !jax_required -> 1 pass   (plain Python)
      scalar_only &  jax_required -> 2 passes (JAX-scalar)
      !scalar_only                -> 4 passes (full JAX, _core + wrapper)
- The model is loaded ONCE; each pass stage is submitted to vLLM as a single
  batched chat call across all procedures still in flight (procedures are
  independent — translation order does not matter for authoring).
- Pass N>1 prompts get the previous pass's code substituted for
  <<<PREVIOUS_CODE>>>.
- Intermediates land at out/jax/{proc}_pass{n}.py, finals at out/jax/{proc}.py,
  every file topped with the mandatory LLM version header.

Qwen3-family specifics (no-ops for Qwen2.5):
- Qwen3 is a hybrid-reasoning model whose chat template emits <think>...</think>
  blocks by default. Thinking is disabled at the template level
  (chat_template_kwargs={"enable_thinking": False}); extract_python_code() also
  strips any <think> block that slips through, so both layers must fail before
  reasoning text can reach a .py file.
- The context window is read from the model dir's config.json
  (max_position_embeddings): 40960 for Qwen3-32B (the advertised 131072 needs
  YaRN rope_scaling, not set in that dir), 262144 for Qwen3.8-27B (default
  since 2026-08-26). vllm_budget sizes --max-model-len / --max-tokens from it —
  a pass-4 prompt plus its <<<PREVIOUS_CODE>>> must fit in the remainder.

Typical use (inside pbsJobs/pbs_qwen25_translate.sh):
    python workflow_translator/qwen/qwen_API_translate.py --procs all
    python workflow_translator/qwen/qwen_API_translate.py --procs G_of_mu,qv_sat
    python workflow_translator/qwen/qwen_API_translate.py --procs all --dry-run
"""

import os
import argparse
import json
import re
import sys
import time
from pathlib import Path

# Shared adaptive token budgeting (workflow_translator/vllm_budget.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import vllm_budget

# Run from the project root (the PBS jobs cd to $PBS_O_WORKDIR before calling
# this script) — all out/ paths below are resolved against the cwd.
PROJECT_ROOT = Path.cwd()

PROMPTS_DIR = PROJECT_ROOT / "out" / "prompts"
OUTPUT_DIR  = PROJECT_ROOT / "out" / "jax"
DEPS_JSON   = PROJECT_ROOT / "out" / "packets" / "_ALL_deps.json"

# Per-generation record (one JSON object per line), appended for EVERY pass —
# successful or not. Same record the earlier vLLM driver kept: finish_reason and
# the output-token count are what separate "the model stopped of its own
# accord" from "it hit a cap", which no output file can show after the fact.
GEN_LOG = PROJECT_ROOT / "out" / "reports" / "analysis" / "generation_log.jsonl"
RAW_DIR = PROJECT_ROOT / "out" / "reports" / "analysis" / "raw_responses"   # thinking mode only

PLACEHOLDER = "<<<PREVIOUS_CODE>>>"

DEFAULT_MODEL = os.path.expandvars("/glade/work/$USER/models/Qwen2.5-Coder-32B-Instruct")


def _llm_label() -> str:
    """[translator].llm_label from config/project.toml — the version header
    must name the model that actually ran (job 7259421 stamped the old
    hard-coded "Qwen3-32B" label on a Qwen3.8-27B run)."""
    try:
        import tomllib
        cfg = tomllib.loads((PROJECT_ROOT / "config" / "project.toml").read_text())
        label = cfg.get("translator", {}).get("llm_label")
        if label:
            return label
    except Exception as exc:  # noqa: BLE001 — fall back, but say so
        print(f"  NOTE: could not read [translator].llm_label ({exc})")
    return "Qwen (label unset — set [translator].llm_label)"


LLM_LABEL = _llm_label()

SYSTEM_PROMPT = (
    "You are an expert scientific computing engineer specialising in "
    "translating Fortran code to Python/JAX. "
    "You produce only clean, correct Python code with no extra explanation."
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Translate Fortran procedures to JAX with Qwen3 (batched pass chain)"
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--procs", default="all",
                   help='"all" or comma-separated procedure names')
    p.add_argument("--skip-existing", action="store_true",
                   help="skip procedures whose final out/jax/{proc}.py already exists")
    # Budgets default to adaptive (sized from each procedure's Fortran source
    # — see workflow_translator/vllm_budget.py). Passing either explicitly
    # pins it to a fixed value and disables that half of the adaptation.
    p.add_argument("--max-tokens", type=int, default=None,
                   help="fixed output cap (clipped to the window room); default: "
                        "thinking on -> everything the window leaves after the "
                        "prompt; thinking off -> adaptive per procedure/pass")
    p.add_argument("--max-model-len", type=int, default=None,
                   help="engine context window; default: the model's native "
                        "maximum (config.json max_position_embeddings)")
    p.add_argument("--adaptive-window", action="store_true",
                   help="size the window from the plan instead (the pre-2026-08-27 "
                        "default; can truncate verbose models — kessler job 7259857)")
    p.add_argument("--temperature", type=float, default=None,
                   help="default: 0.6 with thinking on (Qwen thinking-mode "
                        "recommendation, with top_p 0.95 / top_k 20 unless given), "
                        "0.1 with thinking off (the archived Qwen3-32B runs)")
    # Sampling controls below default to None = vLLM's own defaults (top_p 1.0,
    # top_k off, min_p 0.0, repetition_penalty 1.0, unseeded) — leaving them
    # unset reproduces the pre-flag behavior exactly. --seed makes the sampling
    # RNG deterministic per request (repeat runs redraw the same samples; note
    # TP>1 batching can still flip near-tie tokens in rare cases). The
    # temperature=0.7/top-p=0.8/top-k=20/min-p=0 combination is Qwen's
    # documented recommendation for non-thinking mode.
    p.add_argument("--seed", type=int, default=None,
                   help="sampling seed (per request); default: unseeded")
    p.add_argument("--top-p", type=float, default=None,
                   help="nucleus sampling threshold; default: vLLM's 1.0")
    p.add_argument("--top-k", type=int, default=None,
                   help="top-k sampling cutoff; default: off")
    p.add_argument("--min-p", type=float, default=None,
                   help="min-p sampling threshold; default: 0.0")
    p.add_argument("--repetition-penalty", type=float, default=None,
                   help="penalise repeated tokens (>1.0 discourages); "
                        "default: 1.0 (off)")
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument("--rope-scaling", default=None, metavar="JSON",
                   help="rope_scaling dict passed to the engine via hf_overrides, "
                        "e.g. '{\"rope_type\": \"yarn\", \"factor\": 4.0, "
                        "\"original_max_position_embeddings\": 32768}'. Qwen3's "
                        "native window is 40960; the officially documented YaRN "
                        "x4 recipe extends it to 131072 for prompts that cannot "
                        "otherwise fit. Default: off (native context).")
    p.add_argument("--enable-thinking", action="store_true",
                   help="Qwen3 reasoning mode: the model reasons (tens of thousands "
                        "of tokens) before the answer; the driver keeps only the "
                        "text after </think> and saves the full response under "
                        "out/reports/analysis/raw_responses/. Default: off.")
    p.add_argument("--dry-run", action="store_true",
                   help="print the translation plan and exit (no model load)")
    return p.parse_args()


def pass_count(entry: dict) -> int:
    if not entry["scalar_only"]:
        return 4
    return 2 if entry.get("jax_required") else 1


def build_plan(args) -> list[dict]:
    """One item per procedure: name, pass count, prompt paths."""
    deps = json.loads(DEPS_JSON.read_text(encoding="utf-8"))
    wanted = None
    if args.procs != "all":
        wanted = {w.strip().lower() for w in args.procs.split(",") if w.strip()}
        known = {e["proc_name"].lower() for e in deps}
        unknown = wanted - known
        if unknown:
            sys.exit(f"ERROR: unknown procedure(s): {sorted(unknown)}")

    plan = []
    for e in deps:
        name = e["proc_name"]
        if wanted is not None and name.lower() not in wanted:
            continue
        final = OUTPUT_DIR / f"{name}.py"
        if args.skip_existing and final.exists():
            print(f"  skip (final exists): {name}")
            continue
        n = pass_count(e)
        prompts = [PROMPTS_DIR / f"{name}_pass{i}.md" for i in range(1, n + 1)]
        missing = [p.name for p in prompts if not p.exists()]
        if missing:
            print(f"  WARNING: skipping {name} — missing prompt(s): {missing}")
            continue
        plan.append({"proc": name, "passes": n, "prompts": prompts, "code": None,
                     "failed": None})
    return plan


def extract_python_code(response_text: str, reasoning: bool = False) -> str:
    """Code block from a response; `reasoning=True` when thinking mode is on.

    Qwen3.5+/3.8 chat templates end the generation prompt with "<think>\n", so
    the generated text carries the reasoning FIRST and only a closing </think>
    (no opening tag). Anything before the last </think> is reasoning — it is
    full of draft ```python snippets, and picking the first of them handed a
    7-line fragment to pass 2 (Kessler, 2026-08-27). So:
    keep only what follows the last </think>; under reasoning mode a response
    with no </think> at all was cut off mid-thought and yields NO code (the
    caller records it as a failed generation and keeps the raw text).
    """
    if "</think>" in response_text:
        response_text = response_text.rsplit("</think>", 1)[1]
    elif reasoning:
        return ""
    # Second line of defence: an opening+closing pair in the answer part.
    response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
    match = re.search(r"```python\s*\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip()


HEADER_RE = re.compile(
    r"^# -+\n# Translated by:.*?\n# Pass:.*?\n# -+\n",
    re.DOTALL | re.MULTILINE,
)


def ensure_header(code: str, pass_label: str) -> str:
    header = (
        "# ---------------------------------------------------------------------------\n"
        f"# Translated by: {LLM_LABEL}\n"
        f"# Pass: {pass_label}\n"
        "# ---------------------------------------------------------------------------\n"
    )
    if HEADER_RE.match(code):
        return HEADER_RE.sub(header, code, count=1)
    return header + code


def write_output(item: dict, stage: int, code: str):
    final = stage == item["passes"]
    if final:
        path = OUTPUT_DIR / f"{item['proc']}.py"
        label = "final"
    else:
        path = OUTPUT_DIR / f"{item['proc']}_pass{stage}.py"
        label = str(stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ensure_header(code, label) + "\n", encoding="utf-8")
    return path


def main():
    args = parse_args()

    print("=" * 70)
    print(f"  Qwen3 Fortran -> JAX translation   ({LLM_LABEL})")
    print(f"  model    : {args.model}")
    print(f"  thinking : {'on' if args.enable_thinking else 'off'}")
    # Only what deviates from vLLM defaults; every generation's full resolved
    # sampling config is also written to the generation log.
    # Sampling defaults follow the mode (Qwen's documented recommendations;
    # thinking mode must not be run near-greedy). Explicit flags always win.
    if args.temperature is None:
        args.temperature = 0.6 if args.enable_thinking else 0.1
    if args.enable_thinking:
        if args.top_p is None:
            args.top_p = 0.95
        if args.top_k is None:
            args.top_k = 20
    sampling_kwargs = {k: v for k, v in [
        ("seed", args.seed),
        ("top_p", args.top_p),
        ("top_k", args.top_k),
        ("min_p", args.min_p),
        ("repetition_penalty", args.repetition_penalty),
    ] if v is not None}
    print(f"  sampling : temperature={args.temperature}"
          + "".join(f", {k}={v}" for k, v in sampling_kwargs.items()))
    print("=" * 70)

    plan = build_plan(args)
    if not plan:
        sys.exit("ERROR: nothing to translate.")

    total_gens = sum(i["passes"] for i in plan)
    print(f"\nPlan: {len(plan)} procedures, {total_gens} generations")
    for i in plan:
        print(f"  {i['proc']:42s} passes={i['passes']}")
    # ---- adaptive token budgeting (before the model is loaded) -------------
    # max_model_len is an engine-level setting fixed at construction, so the
    # plan must be measured first; per-request output caps are then applied
    # per pass below.
    counter = vllm_budget.TokenCounter(args.model)
    if not counter.exact:
        print(f"  NOTE: no tokenizer ({counter.reason}) — budgets are estimated.")
    model_max = vllm_budget.model_max_len(args.model)
    budget_plan = vllm_budget.plan_engine_context(plan, counter, model_max)
    print()
    print(budget_plan.report())
    # {(proc, stage): expected code tokens} — the per-request sizing reference
    code_ref = {(b.proc, b.stage): b.ref_tokens for b in budget_plan.budgets}

    # Window policy (2026-08-27): the model's native maximum, always. The
    # adaptive plan was sized for code-only answers and truncated Qwen3.8
    # twice (verbose code, then reasoning); the KV cache on 4x A100-40GB holds
    # ~1.1M tokens, so the full 262144 window costs nothing. A ~2,000-line
    # procedure (~25k tokens of code) is the sizing case for future runs.
    if args.max_model_len:
        engine_len = min(args.max_model_len, model_max)
        print(f"  window: --max-model-len {args.max_model_len} "
              f"(model max {model_max}) -> {engine_len}")
    elif args.adaptive_window:
        engine_len = budget_plan.max_model_len
        print(f"  window: adaptive {engine_len} (--adaptive-window)")
    else:
        engine_len = model_max
        print(f"  window: model native maximum {engine_len}")

    if args.dry_run:
        print("\n--dry-run: exiting before model load.")
        return

    from vllm import LLM, SamplingParams
    t0 = time.perf_counter()
    engine_kwargs = {}
    if args.rope_scaling:
        rope = json.loads(args.rope_scaling)
        engine_kwargs["hf_overrides"] = {"rope_scaling": rope}
        # vLLM derives its window cap from config.json's max_position_embeddings
        # even when rope_scaling is overridden (it rejected max_model_len 90112
        # against the native 40960 with the override in place), so the extended
        # length YaRN provides must be stated explicitly too.
        if "factor" in rope and "original_max_position_embeddings" in rope:
            extended = int(rope["factor"] * rope["original_max_position_embeddings"])
            engine_kwargs["hf_overrides"]["max_position_embeddings"] = extended
        print(f"hf_overrides: {engine_kwargs['hf_overrides']}")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=engine_len,
        tensor_parallel_size=args.tensor_parallel_size,
        **engine_kwargs,
    )
    chat_kwargs = {"enable_thinking": bool(args.enable_thinking)}
    print(f"\nModel loaded in {time.perf_counter() - t0:.0f} s\n")

    max_passes = max(i["passes"] for i in plan)
    for stage in range(1, max_passes + 1):
        batch = [i for i in plan if i["passes"] >= stage and not i["failed"]]
        if not batch:
            break
        print(f"--- Pass {stage}: {len(batch)} procedures " + "-" * 40)

        conversations, sampling_list, sized, sized_meta = [], [], [], []
        for item in batch:
            template = item["prompts"][stage - 1].read_text(encoding="utf-8")
            if stage == 1:
                prompt = template
            elif PLACEHOLDER in template:
                prompt = template.replace(PLACEHOLDER, item["code"])
            else:
                print(f"  WARNING: {item['proc']} pass{stage} prompt has no "
                      f"{PLACEHOLDER!r} — using template as-is.")
                prompt = template

            # Size this request against the prompt actually being sent, so the
            # window cannot be exceeded rather than being found exceeded.
            prompt_tokens = counter.count(prompt)
            room = engine_len - prompt_tokens - vllm_budget.MARGIN
            if args.max_tokens:
                budget = min(args.max_tokens, room)      # never exceed the window
            elif args.enable_thinking:
                # Reasoning length is unpredictable (kessler pass 1: 38k in one
                # draw, >62k in another) — give the answer everything the
                # window leaves. A runaway costs at most one window.
                budget = room
            else:
                budget, _ = vllm_budget.request_max_tokens(
                    prompt, code_ref.get((item["proc"], stage), 0),
                    engine_len, counter)
            if budget < vllm_budget.OUT_FLOOR // 4:
                # The prompt itself nearly fills the window: any generation
                # would be truncated, so fail loudly instead of burning a slot.
                item["failed"] = (f"pass{stage}: prompt {prompt_tokens} tokens leaves "
                                  f"only {budget} for output (window {engine_len})")
                print(f"  FAIL {item['proc']}: {item['failed']}")
                continue

            conversations.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])
            sampling_list.append(SamplingParams(temperature=args.temperature,
                                                max_tokens=budget,
                                                **sampling_kwargs))
            sized.append(item)
            sized_meta.append({"prompt_tokens": prompt_tokens, "max_tokens": budget})
            print(f"    {item['proc']:<34} prompt={prompt_tokens:>6}  max_out={budget:>6}")

        batch = sized
        if not batch:
            continue

        t0 = time.perf_counter()
        outputs = llm.chat(conversations, sampling_params=sampling_list,
                           chat_template_kwargs=chat_kwargs)
        dt = time.perf_counter() - t0
        n_tok = sum(len(o.outputs[0].token_ids) for o in outputs)
        print(f"  batch generated: {n_tok} tokens in {dt:.0f} s "
              f"({n_tok / dt:.1f} tok/s aggregate)")

        for item, out, sp, meta in zip(batch, outputs, sampling_list, sized_meta):
            text = out.outputs[0].text
            finish = out.outputs[0].finish_reason
            n_out = len(out.outputs[0].token_ids)
            code = extract_python_code(text, reasoning=bool(args.enable_thinking))
            if args.enable_thinking:
                # Keep the full reasoning + answer: it is evidence (what the
                # model considered, how it settled the design) that no .py
                # file preserves, and it is cheap.
                try:
                    RAW_DIR.mkdir(parents=True, exist_ok=True)
                    (RAW_DIR / f"{item['proc']}_pass{stage}.raw.txt").write_text(
                        text, encoding="utf-8")
                except Exception as exc:                   # never fail a run on logging
                    print(f"  NOTE: could not save raw response: {exc}")

            # Record every generation, successful or not. `finish_reason ==
            # "stop"` with n_out well under the cap is the signature of the
            # model ending on its own; "length" means it was cut off.
            try:
                GEN_LOG.parent.mkdir(parents=True, exist_ok=True)
                with GEN_LOG.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "proc": item["proc"],
                        "pass": stage,
                        "finish_reason": finish,
                        "output_tokens": n_out,
                        "max_tokens": meta["max_tokens"],
                        "prompt_tokens": meta["prompt_tokens"],
                        "engine_max_model_len": engine_len,
                        "hit_cap": n_out >= sp.max_tokens,
                        "code_chars": len(code),
                        "ok": bool(code) and finish == "stop",
                        "model": args.model,
                        # Resolved sampling config; None = vLLM default. A
                        # replicate experiment is uninterpretable without this
                        # on the record.
                        "sampling": {"temperature": args.temperature,
                                     "seed": args.seed,
                                     "top_p": args.top_p,
                                     "top_k": args.top_k,
                                     "min_p": args.min_p,
                                     "repetition_penalty": args.repetition_penalty,
                                     "enable_thinking": bool(args.enable_thinking)},
                    }) + "\n")
            except Exception as exc:                       # never fail a run on logging
                print(f"  NOTE: could not append to {GEN_LOG.name}: {exc}")
            if not code or finish != "stop":
                # Preserve the raw text: a failed generation is still evidence
                # (E13 lost three 40k-token cap-runs this way — sizes survived
                # in the log, the text did not).
                try:
                    fp = OUTPUT_DIR / f"{item['proc']}_pass{stage}.FAILED.txt"
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    fp.write_text(text, encoding="utf-8")
                except Exception as exc:
                    print(f"  NOTE: could not save failed text: {exc}")
                detail = f"finish_reason={finish}, {n_out} output tokens, {len(code)} chars"
                if finish == "length":
                    # Distinguish "hit its own cap" from "filled the window" —
                    # the fixes differ (sampling/repetition vs. a bigger window).
                    cause = ("output cap reached — check for repetition, the expected "
                             "code is far smaller" if n_out >= sp.max_tokens
                             else "context window exhausted")
                    detail += (f"; cap was {sp.max_tokens} of a {engine_len}-token "
                               f"window — {cause}")
                item["failed"] = f"pass{stage}: {detail}"
                print(f"  FAIL {item['proc']}: {item['failed']}")
                continue
            item["code"] = code
            path = write_output(item, stage, code)
            print(f"  ok   {item['proc']:42s} -> {path.name}  "
                  f"[finish={finish}, out={n_out}/{sp.max_tokens} tok]")

    done = [i for i in plan if not i["failed"]]
    failed = [i for i in plan if i["failed"]]
    print("\n" + "=" * 70)
    print(f"  DONE: {len(done)}/{len(plan)} procedures translated")
    for i in failed:
        print(f"  FAILED: {i['proc']} — {i['failed']}")
    print("=" * 70)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
