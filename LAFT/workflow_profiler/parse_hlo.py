#!/usr/bin/env python3
"""
parse_hlo.py
─────────────────────────────────────────────────────────────────────────
Text-level parse of an XLA optimized-HLO dump into structural counts.

profile_trace_run.py dumps the compiled HLO of the translation
(`fn_jit.lower(...).compile().as_text()` -> kessler_run_hlo.txt) and
feeds the counts produced here into the metrics JSON as the `hlo`
block. The compiled HLO is the compiler's own statement of program
structure — actual scatter / gather / while / fusion instructions,
with none of the runtime-capture noise (warmup, autotuning kernels)
that contaminated the nsys kernel counts.

Deliberately a line-level text parse, NOT a full HLO parser. It relies
on the stable surface shape of `as_text()` output:

    HloModule jit_kessler_run_core, ...

    %fused_computation.1 (param_0.1: f64[56,1000]) -> f64[56,1000] {
      %param_0.1 = f64[56,1000]{1,0} parameter(0)
      ROOT %multiply.9 = f64[56,1000]{1,0} multiply(...)
    }

    ENTRY %main.123 (...) -> (...) {
      %fusion.1 = f64[56,1000]{1,0} fusion(...), kind=kLoop, calls=...
      %while.11 = (...) while(...), condition=..., body=...
    }

  - computations open at column 0 with a line ending in "{" and close
    at a column-0 "}",
  - an instruction definition is an indented
    "[ROOT ]%name = <shape> <op>(operands...)" line,
  - the op is the identifier directly before the operand-list "(".

Counting is on instruction DEFINITIONS only: "scatter" appearing inside
a fused computation's *name* (e.g. %fused_scatter.2) is not a scatter
instruction and is deliberately not counted.

`ops` counts ALL definitions — entry computation AND fusion bodies.
This matters: XLA usually wraps an `.at[idx].set(...)` scatter in an
input fusion, so the actual scatter(...) instruction lives in a
fused_scatter body, not at entry level; gating on entry-level counts
alone would silently miss it.

Position within a fusion matters too, in the other direction.
`ops_as_fusion_roots` counts category ops that are the ROOT of their
fused computation: a root scatter still performs its irregular writes
on every call (it is what the fusion kernel ultimately does), whereas
an INTERIOR gather absorbed mid-fusion is just indexed reads folded
into the kernel. Empirically, the converged production-ready Kessler
translation contains 4 interior in-fusion gathers — produced by the
prescribed jnp.clip/jnp.where boundary-handling fix itself — with zero
standalone gather kernels in nsys. Interior occurrences are therefore
evidence detail, not automatically defects. (Limitation: a
multi-output fusion has a tuple root, so its constituent ops are
reported as interior, not as roots. This errs in the benign direction:
scatter is unaffected because it gates on totals, and a gather hiding
in a multi-output fusion is by construction fused.)
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# The op categories the diagnoser gates on.
HLO_OP_CATEGORIES = ("fusion", "scatter", "gather", "while")

# indented "[ROOT ]%name = <rhs>" (instruction definition)
_INSTR_RE = re.compile(r"^\s+(?:ROOT\s+)?%?[\w.~-]+\s*=\s*(?P<rhs>.+)$")
# HLO op identifier directly before the operand list, e.g. "fusion(",
# "custom-call(", "dynamic-slice("
_OP_RE = re.compile(r"^([a-z][a-z0-9-]*)\(")


def _strip_tuple_shape(rhs: str) -> str:
    """Drop a leading tuple shape "(f64[..], s32[]) " from an rhs."""
    if not rhs.startswith("("):
        return rhs
    depth = 0
    for i, ch in enumerate(rhs):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return rhs[i + 1:].lstrip()
    return rhs


def _instruction_op(line: str):
    """Op name of an instruction-definition line, or None.

    "%x = f64[56,1000]{1,0} fusion(...)"        -> "fusion"
    "ROOT %y = (f64[..], f64[..]) tuple(...)"   -> "tuple"
    """
    m = _INSTR_RE.match(line)
    if not m:
        return None
    rhs = _strip_tuple_shape(m.group("rhs").lstrip())
    if not _OP_RE.match(rhs):
        # non-tuple shape token first: "f64[56,1000]{1,0} fusion(...)"
        parts = rhs.split(None, 1)
        if len(parts) < 2:
            return None
        rhs = parts[1]
    m = _OP_RE.match(rhs)
    return m.group(1) if m else None


def parse_hlo_metrics(hlo_text: str) -> dict:
    """Structural counts from an optimized-HLO text dump.

    Returns:
        total_instructions             all instruction definitions
        instructions_in_fusion_bodies  subset inside fused computations
        ops                            {fusion/scatter/gather/while: n}
                                       counted EVERYWHERE (entry + fusion
                                       bodies)
        ops_in_fusion_bodies           same categories, fusion bodies only
                                       (subset of ops)
        ops_as_fusion_roots            same categories, ROOT instructions
                                       of fused computations only (subset
                                       of ops_in_fusion_bodies) — a root
                                       scatter/gather is what its fusion
                                       kernel ultimately does; an interior
                                       one was absorbed
        ops_per_fusion                 instructions_in_fusion_bodies /
                                       ops.fusion — how much work XLA
                                       packed into each fused kernel
                                       (higher = better packing). None
                                       when there are no fusion ops.
                                       Report-only characterization;
                                       replaces the AST-based
                                       fusion-efficiency proxy whose
                                       jax_op_count denominator varied
                                       with LLM coding style
    """
    ops = {c: 0 for c in HLO_OP_CATEGORIES}
    ops_in_fusion = {c: 0 for c in HLO_OP_CATEGORIES}
    ops_as_roots = {c: 0 for c in HLO_OP_CATEGORIES}
    total = 0
    total_in_fusion = 0

    in_fusion_body = False
    for line in hlo_text.splitlines():
        if line and not line[0].isspace():
            # column-0 line: computation header, closing "}", or module
            # header. Fusion bodies are the computations XLA names
            # "fused_*" / "*_fused_*" (fused_computation.N, fused_scatter,
            # copy_horizontally_fused_computation, ...); regions (while
            # bodies/conditions), wrapped_* and command_buffer
            # computations are NOT fusion bodies.
            if line.rstrip().endswith("{"):
                tokens = line.split()
                name = tokens[1] if tokens[0] == "ENTRY" else tokens[0]
                in_fusion_body = "fused" in name.lstrip("%")
            elif line.startswith("}"):
                in_fusion_body = False
            continue

        op = _instruction_op(line)
        if op is None:
            continue
        total += 1
        if in_fusion_body:
            total_in_fusion += 1
        if op in ops:
            ops[op] += 1
            if in_fusion_body:
                ops_in_fusion[op] += 1
                if line.lstrip().startswith("ROOT "):
                    ops_as_roots[op] += 1

    return {
        "total_instructions": total,
        "instructions_in_fusion_bodies": total_in_fusion,
        "ops": ops,
        "ops_in_fusion_bodies": ops_in_fusion,
        "ops_as_fusion_roots": ops_as_roots,
        "ops_per_fusion": (round(total_in_fusion / ops["fusion"], 2)
                           if ops["fusion"] else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Parse an XLA optimized-HLO dump into structural "
                    "counts (fusion/scatter/gather/while instructions)."
    )
    ap.add_argument("--hlo", required=True,
                    help="Path to the HLO text dump (kessler_run_hlo.txt).")
    ap.add_argument("--out", default=None,
                    help="Where to write the counts JSON (default: stdout).")
    args = ap.parse_args()

    metrics = parse_hlo_metrics(Path(args.hlo).read_text())
    out_text = json.dumps(metrics, indent=2)
    if args.out:
        Path(args.out).write_text(out_text + "\n")
        print(f"HLO metrics written to {args.out}")
    else:
        print(out_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
