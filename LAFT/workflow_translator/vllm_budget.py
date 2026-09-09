"""
vllm_budget.py
==============
Adaptive token-budget planning shared by the external-translator drivers
(workflow_translator/<model>/<model>_API_translate.py).

Problem it solves
-----------------
The drivers used to hard-code two constants: an engine context window
(`--max-model-len`) and an output cap (`--max-tokens`), both picked by hand
per model. That is wrong in both directions:

- Too large a context window wastes KV cache. vLLM's KV pool is fixed, so
  concurrency = pool / max_model_len; reserving 131k for prompts of a few
  thousand tokens throttles a batched pass stage for no benefit (and inflates
  the GPU count the PBS job has to queue for).
- Too large an output cap lets a repetition loop burn the whole budget before
  stopping, turning "the model rambled" into an expensive `finish_reason=
  length` failure that looks like a context problem.

Both numbers are derivable from the actual prompts, which the driver already
reads while building its plan — before the model is loaded.

Two-level design
----------------
1. **Engine level (once, pre-load):** `plan_engine_context()` measures every
   prompt in the plan and returns the smallest `max_model_len` that fits the
   widest pass plus its output budget, clamped to what the model was trained
   for.
2. **Request level (per pass):** `request_max_tokens()` re-checks against the
   *actual* prompt and returns `min(budget, engine_len - prompt - margin)`, so
   exceeding the window is structurally impossible rather than detected after
   the fact. vLLM accepts a per-request `Sequence[SamplingParams]`, so each
   procedure in a batch can carry its own cap.

Sizing signal
-------------
A translation is bounded by the size of what is being translated:

- **Pass 1** — the prompt embeds the Fortran source in a ```fortran fence;
  the largest such block is the reference.
- **Pass N>1** — the previous pass's code is the reference. Later passes are
  refactors (vectorise, split _core/wrapper), not expansions.

`OUT_FACTOR` multiplies that reference, `OUT_FLOOR` guarantees small
procedures still get room, and every computed budget is logged so a
truncation is always attributable to a number we chose.

This module deliberately does NOT import vllm — it returns plain ints so it
can be unit-tested on a login node. If the tokenizer cannot be loaded it falls
back to a characters-per-token heuristic and says so; budgeting must never be
the reason a translation run dies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# How large the JAX translation of a procedure is, relative to its Fortran
# source. Measured 2026-07-22 over the archived Kessler translations of two
# independent models: 1.37-1.65, clustering at ~1.46 for the
# non-trivial procedure. Later passes rewrite that code (vectorise, split
# _core/wrapper) rather than expanding it, so this ratio holds for every pass
# — it is deliberately NOT fed back from the previous pass's budget, which
# would compound into a runaway window.
CODE_RATIO = 1.8      # expected code size = source x this (measured max 1.65)
SAFETY = 2.0          # output cap = expected code x this
OUT_FLOOR = 3072      # tokens; even a tiny procedure gets real room
OUT_CEILING = 40960   # tokens; sized for the largest legitimate artifact —
                      # a large orchestrator procedure's final is ~2,045 lines (~21-25k
                      # tokens), which the old kessler-era 16384 would have
                      # truncated. Matches the 40k budget earlier
                      # experiments established. Small procedures still get
                      # small caps via the adaptive path; the per-request
                      # window clamp keeps overflow impossible.
MARGIN = 512          # tokens reserved for chat template + safety
CHARS_PER_TOKEN = 3.5  # fallback only, when no tokenizer is available

_FORTRAN_FENCE = re.compile(r"```fortran\s*\n(.*?)```", re.DOTALL)


@dataclass
class Budget:
    """Per-(procedure, pass) sizing decision, kept for logging."""
    proc: str
    stage: int
    prompt_tokens: int
    ref_tokens: int
    out_tokens: int

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.out_tokens


@dataclass
class EnginePlan:
    """Result of pre-load planning."""
    max_model_len: int
    budgets: list[Budget] = field(default_factory=list)
    exact: bool = True          # False => tokenizer unavailable, estimates used
    capped: bool = False        # True  => need exceeded the model's window
    required: int = 0           # what was needed before clamping

    def report(self) -> str:
        lines = [
            f"  token budget: {'measured' if self.exact else 'ESTIMATED (no tokenizer)'}"
            f"  |  max_model_len = {self.max_model_len}",
            f"  {'proc':<28}{'pass':>5}{'prompt':>9}{'ref':>8}{'out':>8}{'total':>8}",
            "  " + "-" * 66,
        ]
        for b in self.budgets:
            lines.append(f"  {b.proc:<28}{b.stage:>5}{b.prompt_tokens:>9}"
                         f"{b.ref_tokens:>8}{b.out_tokens:>8}{b.total:>8}")
        if self.capped:
            lines.append(
                f"  WARNING: plan needs {self.required} tokens but the model window is "
                f"{self.max_model_len}.\n"
                "           A wide pass may stop with finish_reason=length. Options: raise\n"
                "           the model's window with YaRN rope scaling (hf_overrides), or\n"
                "           lower OUT_FACTOR if the budget is simply generous.")
        return "\n".join(lines)


class TokenCounter:
    """Counts tokens with the model's own tokenizer; degrades to a heuristic."""

    def __init__(self, model_path: str | None = None):
        self.tokenizer = None
        self.reason = ""
        if model_path:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            except Exception as exc:            # missing lib, bad path, ...
                self.reason = f"{type(exc).__name__}: {exc}"

    @property
    def exact(self) -> bool:
        return self.tokenizer is not None

    def count(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        return int(len(text) / CHARS_PER_TOKEN) + 1


def model_max_len(model_path: str, default: int = 32768) -> int:
    """
    The context window the model was trained for, from its own config.json
    (`max_position_embeddings`). Multimodal configs nest it under
    `text_config`. Returns `default` if the file is unreadable —
    budgeting must never be the reason a run dies.

    NOTE: exceeding this value requires rope scaling (e.g. YaRN via
    `hf_overrides`); simply raising --max-model-len past it is refused by vLLM
    because positions beyond training break RoPE.
    """
    import json
    from pathlib import Path
    try:
        cfg = json.loads((Path(model_path) / "config.json").read_text())
    except Exception:
        return default
    text_cfg = cfg.get("text_config", cfg)
    return int(text_cfg.get("max_position_embeddings")
               or cfg.get("max_position_embeddings")
               or default)


def fortran_source_tokens(prompt_text: str, counter: TokenCounter) -> int:
    """Tokens in the largest ```fortran block — the pass-1 sizing reference.

    Pass-1 prompts embed the procedure's Fortran source in a fenced block;
    smaller fortran fences also appear as inline policy examples, so the
    largest one is the source.
    """
    blocks = _FORTRAN_FENCE.findall(prompt_text)
    if not blocks:
        return 0
    return max(counter.count(b) for b in blocks)


def expected_code_tokens(source_tokens: int, ratio: float = CODE_RATIO) -> int:
    """Size of the translated code we expect from a source of this size."""
    return int(source_tokens * ratio)


def output_budget(code_tokens: int,
                  safety: float = SAFETY,
                  floor: int = OUT_FLOOR,
                  ceiling: int = OUT_CEILING) -> int:
    """Output cap for one generation, from the expected code size."""
    return max(floor, min(ceiling, int(code_tokens * safety)))


def plan_engine_context(items: list[dict],
                        counter: TokenCounter,
                        model_max_len: int,
                        *,
                        ratio: float = CODE_RATIO,
                        safety: float = SAFETY,
                        floor: int = OUT_FLOOR,
                        ceiling: int = OUT_CEILING,
                        margin: int = MARGIN,
                        round_to: int = 1024) -> EnginePlan:
    """
    Smallest engine context window that fits every pass in the plan.

    `items` are the driver's plan entries: {"proc": str, "passes": int,
    "prompts": [Path, ...]}.

    Passes >1 inject the previous pass's code, which does not exist yet. Its
    size is predicted from the Fortran source (`CODE_RATIO`) and held constant
    across passes — NOT taken from the previous pass's output budget, which
    would compound each pass and inflate the window without bound.
    """
    budgets: list[Budget] = []
    required = 0

    for item in items:
        code = 0
        for stage, prompt_path in enumerate(item["prompts"], start=1):
            template = prompt_path.read_text(encoding="utf-8")
            template_tokens = counter.count(template)

            if stage == 1:
                src = fortran_source_tokens(template, counter)
                # No fortran fence (unexpected): fall back to the whole
                # template, which over-estimates but never under-budgets.
                code = expected_code_tokens(src or template_tokens, ratio)
                prompt_tokens = template_tokens
            else:
                # <<<PREVIOUS_CODE>>> is replaced by the previous pass's code
                prompt_tokens = template_tokens + code

            out = output_budget(code, safety, floor, ceiling)
            budgets.append(Budget(item["proc"], stage, prompt_tokens, code, out))
            required = max(required, prompt_tokens + out + margin)

    want = ((required + round_to - 1) // round_to) * round_to
    max_model_len = min(want, model_max_len)
    return EnginePlan(max_model_len=max_model_len, budgets=budgets,
                      exact=counter.exact, capped=want > model_max_len,
                      required=required)


def request_max_tokens(prompt_text: str,
                       code_tokens: int,
                       engine_len: int,
                       counter: TokenCounter,
                       *,
                       safety: float = SAFETY,
                       floor: int = OUT_FLOOR,
                       ceiling: int = OUT_CEILING,
                       margin: int = MARGIN) -> tuple[int, int]:
    """
    Output cap for one real request, given the prompt that will be sent.

    Returns (max_tokens, prompt_tokens). `max_tokens` can come back <= 0 when
    the prompt alone nearly fills the window — the caller must treat that as a
    hard failure for that procedure rather than sending the request, since any
    generation would be truncated.
    """
    prompt_tokens = counter.count(prompt_text)
    room = engine_len - prompt_tokens - margin
    return min(output_budget(code_tokens, safety, floor, ceiling), room), prompt_tokens
