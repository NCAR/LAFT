# -----------------------------------------------------------
# Phase 2.3: Merge Fortran Packets and Dependency Data (BRIDGE MODE ONLY)
#
# This script merges Phase 2 output packets (*.json) with their corresponding
# dependency files (*_deps.json) in out/packets/. For each procedure:
#   - Loads the base packet and its dependency packet.
#   - Computes scalar output-only args (e.g., isok) for downstream wrapper/prompt rules.
#   - Combines them with a BRIDGE layout policy (no transpose policy inside compute code).
#   - NEW: Preserves parent_module and module_vars_used at top level for easy access
#   - Writes the merged result to out/packets/PROCNAME_merged.json.
#   - Skips procedures missing dependency files.
#
# Bridge Mode:
#   - Translated compute code assumes row-major layout (standard JAX/Python)
#   - Layout conversion Fortran column-major <-> row-major is handled by separate bridge code
#   - No swapaxes/aF policy is used anywhere in translation/wrappers
# -----------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

# framework_config lives in LAFT/config/; this script now lives in
# workflow_frontend/ (symlinked from LAFT/ into each project) — resolve back
# to LAFT/config to import it. Same convention as workflow_bridge/phase03_make_bridge.py.
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "config"))

from framework_config import get_config


# -------------------------
# Bridge-only layout policy
# -------------------------
LAYOUT_POLICY: Dict[str, Any] = {
    "mode": "bridge",
    "rule": (
        "Bridge layer handles Fortran column-major ↔ row-major conversion. "
        "Translated core + wrapper operate purely on row-major arrays. "
        "No transpose/swapaxes logic inside translated compute modules."
    ),
    "boundary_ops": None,
}


# -------------------------
# Scalar output arg helpers
# -------------------------
def _is_dim_name(name: str) -> bool:
    """Heuristic: dimension/size variables that should never be treated as outputs.

    Single source: [heuristics].dim_names in config/project.toml."""
    return get_config().is_dim_name(name)


def _is_scalar_output_flag(name: str) -> bool:
    """Heuristic: scalar outputs like Fortran INTENT(OUT) flags/status.

    Single source: [heuristics].scalar_flag_names in config/project.toml."""
    return get_config().is_scalar_output_flag(name)


def compute_scalar_out_args(args: List[str], writes_to_args: List[str]) -> List[str]:
    """
    Return scalar OUT-only args (e.g., isok) that should be returned (not passed)
    by wrappers in Bridge Mode.

    This does NOT attempt to identify array outputs; only scalar "flag-like" outputs.
    """
    writes = {w.lower() for w in (writes_to_args or [])}
    out: List[str] = []
    for a in (args or []):
        al = a.lower()
        if al in writes and (not _is_dim_name(a)) and _is_scalar_output_flag(a):
            out.append(a)
    return out


def main():
    """
    Phase 3: Merge Fortran Packets and Dependency Data (Bridge Mode ONLY)

    Merges Phase 2 output packets (*.json) with their corresponding dependency files (*_deps.json)
    in out/packets/. For each procedure:
      - Loads the base packet and its dependency packet.
      - Computes scalar output-only args (deps["scalar_out_args"]) for downstream phases.
      - Combines them with Bridge layout policy.
      - NEW: Preserves parent_module and module_vars_used at top level
      - Writes the merged result to out/packets/PROCNAME_merged.json.
      - Skips procedures missing dependency files.
    """
    packets_dir = get_config().packets_dir
    packets_dir.mkdir(parents=True, exist_ok=True)

    # Base packets are Phase2 outputs: <proc>.json (excluding deps + aggregate + merged)
    base_packets = sorted(
        p for p in packets_dir.glob("*.json")
        if not p.name.endswith("_deps.json")
        and not p.name.endswith("_merged.json")
        and p.name != "_ALL_deps.json"
    )

    if not base_packets:
        raise SystemExit(
            "No base packets found in out/packets/. Run Phase 2 first (phase2_extract.py)."
        )

    merged_count = 0
    skipped = 0

    # --- Pass 1: load all base+deps packets ---
    records = []
    for base_path in base_packets:
        proc = base_path.stem
        deps_path = packets_dir / f"{proc}_deps.json"

        if not deps_path.exists():
            print(f"SKIP: missing deps for {proc}: {deps_path}")
            skipped += 1
            continue

        base: Dict[str, Any] = json.loads(base_path.read_text(encoding="utf-8"))
        deps: Dict[str, Any] = json.loads(deps_path.read_text(encoding="utf-8"))
        records.append((proc, base, deps))

    # Module DECLARATION order (per module) — the canonical ordering for
    # every generated signature. All generated bridges/wrappers/prompts list
    # module vars in this order so their signatures line up; see
    # framework_config.packet_module_vars.
    decl_index: Dict[str, int] = {}
    modules_dir = get_config().modules_dir
    if modules_dir.exists():
        for jf in sorted(modules_dir.glob("*.json")):
            try:
                md = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            for i, v in enumerate(md.get("module_variables", [])):
                name = (v["name"] if isinstance(v, dict) else v).lower()
                decl_index.setdefault(name, i)

    def in_decl_order(names):
        # Declared vars first (declaration order); any unknown name last,
        # alphabetically, so the result is always deterministic.
        return sorted(names, key=lambda n: (decl_index.get(n.lower(), 10**9), n.lower()))

    # --- Transitive module-variable closure over the call graph ---
    # Module globals are threaded through call chains as explicit wrapper
    # parameters, so a procedure must accept every module variable that any
    # of its (direct or indirect) callees uses — even if its own source never
    # names it (e.g. sedimentation_* needs the LUT table `itab` because
    # proc_from_LUT_main2mom reads it). Close module_vars_used over `calls`.
    direct_mvu = {p.lower(): list(b.get("module_vars_used", []) or []) for p, b, d in records}
    call_graph = {p.lower(): [c.lower() for c in (d.get("calls", []) or [])] for p, b, d in records}
    closed_mvu = {k: {v.lower() for v in vals} for k, vals in direct_mvu.items()}
    changed = True
    while changed:
        changed = False
        for proc_l, callees in call_graph.items():
            for callee in callees:
                extra = closed_mvu.get(callee, set()) - closed_mvu[proc_l]
                if extra:
                    closed_mvu[proc_l] |= extra
                    changed = True

    # --- Pass 2: merge and write ---
    for proc, base, deps in records:

        # --- NEW: compute scalar output-only args for wrappers/prompts ---
        args = deps.get("args", []) or []
        writes_to_args = deps.get("writes_to_args", []) or []
        deps["scalar_out_args"] = compute_scalar_out_args(args, writes_to_args)

        # --- NEW: Extract module variable info from base packet for top-level access ---
        parent_module = base.get("parent_module", "")
        # Transitive closure, in module DECLARATION order — this is what every
        # generated signature threads (see framework_config.packet_module_vars).
        module_vars_used = in_decl_order(closed_mvu.get(proc.lower(), set()))
        inherited = sorted(closed_mvu.get(proc.lower(), set())
                           - {v.lower() for v in direct_mvu.get(proc.lower(), [])})
        if inherited:
            print(f"  {proc}: +{len(inherited)} module vars from callees: {', '.join(inherited)}")

        merged: Dict[str, Any] = {
            "proc_name": proc,
            "parent_module": parent_module,  # NEW: Top-level for easy access
            "module_vars_used": module_vars_used,  # closure, declaration order
            # The procedure's OWN direct references (declaration order),
            # informational: closure above is what signatures use.
            "module_vars_direct": in_decl_order(direct_mvu.get(proc.lower(), [])),
            "layout_policy": LAYOUT_POLICY,
            "phase2_packet": base,
            "deps": deps,
        }

        out_path = packets_dir / f"{proc}_merged.json"
        out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        merged_count += 1

    print(f"Merged: {merged_count} procedures into *_merged.json")
    if skipped:
        print(f"Skipped: {skipped} (missing *_deps.json)")


if __name__ == "__main__":
    import argparse
    from framework_config import init as init_config, add_config_arg

    ap = argparse.ArgumentParser(description="Phase 02.3: merge packets with dependency data")
    add_config_arg(ap)
    init_config(ap.parse_args().config)
    main()