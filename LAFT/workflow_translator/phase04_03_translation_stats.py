#!/usr/bin/env python
"""
phase04_03_translation_stats.py — per-procedure translation statistics.

Produces the TWO tables required in the translation report (TRANSLATE_WORKFLOW.md
Step 8):

  Table 1  Four-pass authorship statistics   — FROZEN, never updated
  Table 2  Validation-phase fix statistics   — cumulative, re-rendered per fix

The two tables answer different questions and must never be merged. Table 1 is
"what did the initial four-pass translation cost", a fact about a run that has
already finished. Table 2 is "what did it then take to make it correct", which
keeps moving. Overwriting Table 1 with post-fix numbers would silently destroy
the only record of the untouched machine output.

IMMUTABILITY IS STRUCTURAL, NOT A PROMISE:
  - `freeze` writes out/reports/translation/pass_stats.json exactly once and
    REFUSES to overwrite it (--refreeze is the deliberate override, legitimate
    only after tools/clean_AI_results.sh has reset out/ for a new LLM).
  - it also writes pass_stats.sha256; `render` re-hashes and shouts if the
    frozen file was edited after the fact.
  - the report's Table 1 is rendered from that file verbatim — never hand-typed.

WHAT IS MEASURABLE DEPENDS ON THE TRANSLATOR (config [translator].source):
  - a local model via vLLM (qwen) writes out/reports/analysis/
    generation_log.jsonl, so prompt/output TOKENS are exact. Wall-clock is NOT
    per-procedure there: a pass is one concurrent llm.chat() batch, so time is
    apportioned across the batch by output tokens and labelled as an estimate.
  - "in_context" (the driving agent authors the passes) has no API call, so
    there are no token counts. Sizes in CHARACTERS are exact and stand in.
    Wall-clock IS exact and per-procedure, taken from the timestamps
    translate_workflow_state.py stamps after each procedure.

Columns with no honest source render as "n/a", never as an estimate dressed up
as a measurement.

Usage:
    python workflow_translator/phase04_03_translation_stats.py freeze
    python workflow_translator/phase04_03_translation_stats.py render
    python workflow_translator/phase04_03_translation_stats.py render --out report_fragment.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from framework_config import init as init_config, add_config_arg  # noqa: E402


NA = "n/a"


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _lines(path: Path) -> int | None:
    """Total physical lines, or None if the file does not exist."""
    if not path.is_file():
        return None
    return len(path.read_text(encoding="utf-8", errors="replace").splitlines())


def _code_lines(path: Path, comment_prefixes=("!", "#")) -> int | None:
    """Non-blank, non-comment lines — the size that matters for output horizons."""
    if not path.is_file():
        return None
    n = 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s or s.startswith(comment_prefixes):
            continue
        n += 1
    return n


def _chars(path: Path) -> int | None:
    if not path.is_file():
        return None
    return len(path.read_text(encoding="utf-8", errors="replace"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fmt(v, suffix: str = "") -> str:
    if v is None:
        return NA
    if isinstance(v, float):
        return f"{v:,.1f}{suffix}"
    return f"{v:,}{suffix}"


def _fmt_secs(v) -> str:
    if v is None:
        return NA
    if v < 90:
        return f"{v:.0f}s"
    if v < 5400:
        return f"{v / 60:.1f}m"
    return f"{v / 3600:.2f}h"


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def _load_plan(cfg) -> list[dict]:
    """Procedure list with parent module and pass count."""
    deps_path = cfg.packets_dir / "_ALL_deps.json"
    if not deps_path.is_file():
        raise SystemExit(f"missing {deps_path} — run the frontend (stage 0) first")
    all_deps = json.loads(deps_path.read_text(encoding="utf-8"))

    rows = []
    for d in all_deps:
        proc = d["proc_name"]
        merged = cfg.packets_dir / f"{proc}_merged.json"
        module = None
        if merged.is_file():
            module = json.loads(merged.read_text(encoding="utf-8")).get("parent_module")
        rows.append({
            "module": module or "(no module)",
            "proc": proc,
            # scalar-only procedures take the 1-pass route; everything else 4.
            "passes": 1 if d.get("scalar_only") else 4,
            "unit_file": d.get("unit_file"),
        })
    return rows


def _load_generation_log(cfg) -> dict[str, dict]:
    """
    Per-procedure token totals from a local-model run. Absent for in_context.
    Returns {} when there is no log.
    """
    log = cfg.reports_dir / "analysis" / "generation_log.jsonl"
    if not log.is_file():
        return {}

    agg: dict[str, dict] = defaultdict(
        lambda: {"prompt_tokens": 0, "output_tokens": 0, "generations": 0,
                 "wall_clock_s": 0.0, "has_time": False}
    )
    for line in log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        proc = rec.get("proc")
        if not proc:
            continue
        a = agg[proc]
        a["prompt_tokens"] += rec.get("prompt_tokens") or 0
        a["output_tokens"] += rec.get("output_tokens") or 0
        a["generations"] += 1
        # Written only once the vLLM drivers are extended to record it. Under
        # batching this is an apportioned share, not a measured duration.
        share = rec.get("batch_share_seconds")
        if share is not None:
            a["wall_clock_s"] += float(share)
            a["has_time"] = True
    return dict(agg)


def _load_state_timings(cfg) -> tuple[dict[str, float], bool]:
    """
    Per-procedure wall-clock for in_context runs, derived from the timestamps
    translate_workflow_state.py stamps on each completed step. A procedure's
    duration is the gap between its own "generated out/jax/<proc>.py" stamp and
    the previous one — i.e. real elapsed authoring time, INCLUDING any pause.
    """
    # Same location translate_workflow_state.py writes to (out/jax/), not out/.
    state_path = cfg.jax_dir / "workflow_state.json"
    if not state_path.is_file():
        return {}, False
    state = json.loads(state_path.read_text(encoding="utf-8"))
    stamps = state.get("step_timestamps") or []
    if not stamps:
        return {}, False

    gen: list[tuple[str, datetime]] = []
    started: datetime | None = None
    for entry in stamps:
        step, at = entry.get("step", ""), entry.get("at")
        if not at:
            continue
        try:
            ts = datetime.fromisoformat(at)
        except ValueError:
            continue
        if step.startswith("generated ") and step.endswith(".py"):
            proc = Path(step.split(" ", 1)[1]).stem
            gen.append((proc, ts))
        elif started is None:
            started = ts

    out: dict[str, float] = {}
    prev = started
    for proc, ts in gen:
        if prev is not None:
            delta = (ts - prev).total_seconds()
            if delta >= 0:
                out[proc] = delta
        prev = ts
    return out, bool(out)


# --------------------------------------------------------------------------
# freeze
# --------------------------------------------------------------------------

def cmd_freeze(cfg, args) -> int:
    out_dir = cfg.reports_dir / "translation"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "pass_stats.json"
    hash_path = out_dir / "pass_stats.sha256"

    if stats_path.is_file() and not args.refreeze:
        prev = json.loads(stats_path.read_text(encoding="utf-8"))
        print(f"REFUSING to overwrite {stats_path}", file=sys.stderr)
        print(f"  It was frozen at {prev.get('frozen_at')} and Table 1 is immutable "
              f"by design.", file=sys.stderr)
        print("  Post-authorship changes belong in Table 2 (fix_stats.jsonl).",
              file=sys.stderr)
        print("  Use --refreeze ONLY after tools/clean_AI_results.sh has reset out/ "
              "for a new LLM run.", file=sys.stderr)
        return 1

    translator = cfg.section("translator")   # {} when the section is absent
    source = translator.get("source", "in_context")
    llm_label = translator.get("llm_label", "unknown")

    plan = _load_plan(cfg)
    gen_log = _load_generation_log(cfg)
    state_times, have_state_times = _load_state_timings(cfg)

    rows = []
    for p in plan:
        proc = p["proc"]
        f_path = Path(p["unit_file"]) if p["unit_file"] else None
        jax_path = cfg.jax_dir / f"{proc}.py"

        prompt_chars = 0
        found_prompt = False
        for n in range(1, p["passes"] + 1):
            pc = _chars(cfg.prompts_dir / f"{proc}_pass{n}.md")
            if pc is not None:
                prompt_chars += pc
                found_prompt = True

        g = gen_log.get(proc)
        rows.append({
            "module": p["module"],
            "proc": proc,
            "passes": p["passes"],
            "fortran_lines": _lines(f_path) if f_path else None,
            "fortran_code_lines": _code_lines(f_path) if f_path else None,
            "jax_lines": _lines(jax_path),
            "jax_code_lines": _code_lines(jax_path),
            "prompt_chars": prompt_chars if found_prompt else None,
            "jax_chars": _chars(jax_path),
            "prompt_tokens": g["prompt_tokens"] if g else None,
            "output_tokens": g["output_tokens"] if g else None,
            "wall_clock_s": (g["wall_clock_s"] if g and g["has_time"]
                             else state_times.get(proc)),
        })

    missing = [r["proc"] for r in rows if r["jax_lines"] is None]
    if missing and not args.allow_incomplete:
        print(f"REFUSING to freeze: {len(missing)} of {len(rows)} procedures have no "
              f"out/jax/<proc>.py yet.", file=sys.stderr)
        print(f"  e.g. {', '.join(missing[:5])}", file=sys.stderr)
        print("  Freeze at the END of Step 2, once every procedure is authored — "
              "that is what makes Table 1 the record of the untouched four-pass "
              "output.", file=sys.stderr)
        print("  Use --allow-incomplete only for a deliberately partial run.",
              file=sys.stderr)
        return 1

    doc = {
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "table": "four-pass authorship statistics (IMMUTABLE)",
        "llm_label": llm_label,
        "translator_source": source,
        "token_data_available": bool(gen_log),
        "wall_clock_source": (
            "apportioned from vLLM batch time (estimate)" if any(
                r["wall_clock_s"] is not None for r in rows) and gen_log
            else "elapsed between per-procedure state updates (includes pauses)"
            if have_state_times else "unavailable"
        ),
        "procedures": len(rows),
        "rows": rows,
    }
    stats_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    hash_path.write_text(_sha256(stats_path) + "\n", encoding="utf-8")

    print(f"Froze four-pass statistics for {len(rows)} procedures -> {stats_path}")
    print(f"  translator: {source} ({llm_label})")
    print(f"  tokens: {'from generation_log.jsonl' if gen_log else 'n/a (in-context)'}")
    print(f"  wall-clock: {doc['wall_clock_source']}")
    print(f"  digest -> {hash_path}")
    return 0


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------

def _group(rows: list[dict]) -> dict[str, list[dict]]:
    by_mod: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_mod[r["module"]].append(r)
    return dict(sorted(by_mod.items()))


def _render_table1(doc: dict, tamper: str | None) -> list[str]:
    out: list[str] = []
    out.append("### Table 1 — Four-pass translation statistics (FROZEN)")
    out.append("")
    out.append(f"Frozen at **{doc['frozen_at']}** · translator "
               f"**{doc['translator_source']}** ({doc['llm_label']}) · "
               f"{doc['procedures']} procedures.")
    out.append("")
    out.append("**This table is immutable.** It records the initial four-pass "
               "output before any validation fix. Do not update it — later "
               "repairs belong in Table 2.")
    out.append("")
    if not doc.get("token_data_available"):
        out.append("> Token columns are `n/a`: the translation was authored "
                   "in-context, so no API reported token counts. Prompt and "
                   "output **characters** are exact and stand in for size.")
        out.append("")
    out.append(f"> Wall-clock source: {doc['wall_clock_source']}.")
    out.append("")
    if tamper:
        out.append(f"> ⚠️ **INTEGRITY WARNING:** {tamper}")
        out.append("")

    hdr = ("| module / procedure | passes | Fortran lines (code) | JAX lines (code) "
           "| JAX/F | prompt chars | JAX chars | prompt tok | output tok | wall-clock |")
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    out.append(hdr)
    out.append(sep)

    tot = defaultdict(float)
    for module, rows in _group(doc["rows"]).items():
        out.append(f"| **{module}** | | | | | | | | | |")
        sub = defaultdict(float)
        for r in sorted(rows, key=lambda x: x["proc"].lower()):
            ratio = (f"{r['jax_lines'] / r['fortran_lines']:.2f}x"
                     if r["jax_lines"] and r["fortran_lines"] else NA)
            out.append(
                f"| &nbsp;&nbsp;{r['proc']} | {r['passes']} "
                f"| {_fmt(r['fortran_lines'])} ({_fmt(r['fortran_code_lines'])}) "
                f"| {_fmt(r['jax_lines'])} ({_fmt(r['jax_code_lines'])}) "
                f"| {ratio} | {_fmt(r['prompt_chars'])} | {_fmt(r['jax_chars'])} "
                f"| {_fmt(r['prompt_tokens'])} | {_fmt(r['output_tokens'])} "
                f"| {_fmt_secs(r['wall_clock_s'])} |"
            )
            for k in ("fortran_lines", "jax_lines", "prompt_chars", "jax_chars",
                      "prompt_tokens", "output_tokens", "wall_clock_s"):
                if r[k]:
                    sub[k] += r[k]
                    tot[k] += r[k]
            sub["procs"] += 1
            tot["procs"] += 1
        out.append(
            f"| &nbsp;&nbsp;*subtotal ({int(sub['procs'])} procs)* | "
            f"| *{_fmt(int(sub['fortran_lines']))}* | *{_fmt(int(sub['jax_lines']))}* | "
            f"| *{_fmt(int(sub['prompt_chars']))}* | *{_fmt(int(sub['jax_chars']))}* "
            f"| *{_fmt(int(sub['prompt_tokens'])) if sub['prompt_tokens'] else NA}* "
            f"| *{_fmt(int(sub['output_tokens'])) if sub['output_tokens'] else NA}* "
            f"| *{_fmt_secs(sub['wall_clock_s']) if sub['wall_clock_s'] else NA}* |"
        )
    out.append(
        f"| **TOTAL ({int(tot['procs'])} procs)** | | **{_fmt(int(tot['fortran_lines']))}** "
        f"| **{_fmt(int(tot['jax_lines']))}** | | **{_fmt(int(tot['prompt_chars']))}** "
        f"| **{_fmt(int(tot['jax_chars']))}** "
        f"| **{_fmt(int(tot['prompt_tokens'])) if tot['prompt_tokens'] else NA}** "
        f"| **{_fmt(int(tot['output_tokens'])) if tot['output_tokens'] else NA}** "
        f"| **{_fmt_secs(tot['wall_clock_s']) if tot['wall_clock_s'] else NA}** |"
    )
    out.append("")
    return out


def _render_table2(cfg, doc: dict) -> list[str]:
    fix_path = cfg.reports_dir / "translation" / "fix_stats.jsonl"
    out: list[str] = []
    out.append("### Table 2 — Validation-phase fix statistics (LIVE)")
    out.append("")

    if not fix_path.is_file():
        out.append("No fixes recorded: every procedure passed validation as "
                   "authored. Table 1 is the complete record.")
        out.append("")
        return out

    recs = []
    for line in fix_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not recs:
        out.append("No fixes recorded.")
        out.append("")
        return out

    frozen = {r["proc"]: r for r in doc["rows"]}
    agg: dict[str, dict] = {}
    for r in recs:
        proc = r.get("proc")
        if not proc:
            continue
        a = agg.setdefault(proc, {
            "module": r.get("module") or frozen.get(proc, {}).get("module", "(unknown)"),
            "fixes": 0, "stages": set(), "duration_s": 0.0, "has_time": False,
            "tokens": 0, "has_tokens": False, "last": None,
        })
        a["fixes"] += 1
        if r.get("failed_stage"):
            a["stages"].add(r["failed_stage"])
        if r.get("duration_s") is not None:
            a["duration_s"] += float(r["duration_s"])
            a["has_time"] = True
        if r.get("tokens") is not None:
            a["tokens"] += int(r["tokens"])
            a["has_tokens"] = True
        if r.get("at"):
            a["last"] = max(a["last"] or r["at"], r["at"])

    out.append(f"{len(recs)} fix attempt(s) across {len(agg)} procedure(s). "
               "Re-rendered on every fix; the JAX-line column shows the frozen "
               "four-pass value versus the current file.")
    out.append("")
    out.append("| module / procedure | fixes | failed stage(s) | Fortran lines "
               "| JAX lines frozen → now | fix tokens | fix time | last fixed |")
    out.append("|---|---:|---|---:|---:|---:|---:|---|")

    by_mod: dict[str, list[str]] = defaultdict(list)
    for proc in sorted(agg, key=str.lower):
        by_mod[agg[proc]["module"]].append(proc)

    for module in sorted(by_mod):
        out.append(f"| **{module}** | | | | | | | |")
        for proc in by_mod[module]:
            a = agg[proc]
            fz = frozen.get(proc, {})
            now = _lines(cfg.jax_dir / f"{proc}.py")
            before = fz.get("jax_lines")
            delta = ""
            if before is not None and now is not None and now != before:
                delta = f" ({now - before:+d})"
            out.append(
                f"| &nbsp;&nbsp;{proc} | {a['fixes']} "
                f"| {', '.join(sorted(a['stages'])) or NA} "
                f"| {_fmt(fz.get('fortran_lines'))} "
                f"| {_fmt(before)} → {_fmt(now)}{delta} "
                f"| {_fmt(a['tokens']) if a['has_tokens'] else NA} "
                f"| {_fmt_secs(a['duration_s']) if a['has_time'] else NA} "
                f"| {a['last'] or NA} |"
            )
    out.append("")
    return out


def cmd_render(cfg, args) -> int:
    out_dir = cfg.reports_dir / "translation"
    stats_path = out_dir / "pass_stats.json"
    hash_path = out_dir / "pass_stats.sha256"

    if not stats_path.is_file():
        print(f"missing {stats_path} — run `freeze` at the end of Step 2 first",
              file=sys.stderr)
        return 1
    doc = json.loads(stats_path.read_text(encoding="utf-8"))

    tamper = None
    if hash_path.is_file():
        expect = hash_path.read_text(encoding="utf-8").strip()
        actual = _sha256(stats_path)
        if expect != actual:
            tamper = (f"pass_stats.json has been modified since it was frozen at "
                      f"{doc.get('frozen_at')} (sha256 {actual[:12]}… != recorded "
                      f"{expect[:12]}…). Table 1 is supposed to be immutable — "
                      f"treat these numbers as unreliable and investigate.")
            print(f"INTEGRITY WARNING: {tamper}", file=sys.stderr)

    lines: list[str] = []
    lines.append("## Translation statistics")
    lines.append("")
    lines += _render_table1(doc, tamper)
    lines += _render_table2(cfg, doc)
    text = "\n".join(lines)

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    add_config_arg(ap)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("freeze", help="write the immutable four-pass stats (once)")
    f.add_argument("--refreeze", action="store_true",
                   help="overwrite an existing freeze — only after clean_AI_results.sh")
    f.add_argument("--allow-incomplete", action="store_true",
                   help="freeze even though some procedures have no out/jax file")

    r = sub.add_parser("render", help="emit both markdown tables")
    r.add_argument("--out", help="write to this file instead of stdout")

    args = ap.parse_args()
    cfg = init_config(args.config)
    return cmd_freeze(cfg, args) if args.cmd == "freeze" else cmd_render(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
