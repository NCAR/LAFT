# workflow_translator/phase04_02_make_wrappers.py
# -----------------------------------------------------------
# Phase 4.2: Wrapper Generation and Index (UNIFIED)
#
# ENHANCED WITH MODULE_VARIABLE_POLICY SUPPORT
#
# This unified phase:
# 1. Generates wrapper modules (one per procedure)
# 2. Creates wrapper index for stable imports
#
# OUTPUT: out/wrappers/ — reference skeletons only. The LLM translations live
# in out/jax/, which this script NEVER writes to, so rerunning it at any point
# is safe and cannot overwrite translated code.
#
# PURE BRIDGE MODE: Policy A completely removed
# Simple pass-through wrappers, no layout conversion
#
# Key rules:
#   - Scalar output flags (e.g., isok/info/ierr) MUST NOT appear in wrapper signature
#   - But they SHOULD still be returned if they are written outputs
#   - MODULE variables MUST appear in both wrapper and core signatures
#   - MODULE variables MUST be returned even if unchanged
# -----------------------------------------------------------

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

# framework_config lives in LAFT/config/; this script now lives in a
# workflow folder symlinked from LAFT/ into each project — resolve back to
# LAFT/config to import it. Same convention as workflow_bridge/phase03_make_bridge.py.
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "config"))

from framework_config import get_config, packet_module_vars, safe_py_name


# ============================================================================
# SCALAR-ONLY DETECTION
# ============================================================================

def _is_scalar_only_proc(merged_packet: Dict[str, Any]) -> bool:
    """
    Returns True if this procedure has NO array arguments.

    Detects arrays by scanning the Fortran source for:
      - ':: varname(' pattern  (e.g. 'real :: a(n)' or 'real :: a(:)')
      - 'dimension(' attribute (e.g. 'real, dimension(n) :: a')

    character(len=...) is scalar — the '(' appears before '::', not after,
    so it is not matched.

    Scalar-only procedures should NOT get a JAX _core or JIT overhead.
    """
    fortran_source = merged_packet.get("phase2_packet", {}).get("fortran_source", "")
    if not fortran_source:
        return False  # no source → assume arrays to be safe

    array_patterns = [
        r"::\s*\w+\s*\(",      # :: varname( — array in declaration
        r"\bdimension\s*\(",   # dimension( attribute
    ]
    for pattern in array_patterns:
        if re.search(pattern, fortran_source, re.IGNORECASE):
            return False

    return True


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _as_list(x) -> List[str]:
    return x if isinstance(x, list) else []


def _is_dim_name(name: str) -> bool:
    """Heuristic: dimension/size variables that should never be outputs.

    Single source: [heuristics].dim_names in config/project.toml."""
    return get_config().is_dim_name(name)


def _is_scalar_output_flag(name: str) -> bool:
    """
    Heuristic: scalar flags that are typically output-only in Fortran APIs
    and should not be passed into Python wrappers.

    Single source: [heuristics].scalar_flag_names in config/project.toml.
    """
    return get_config().is_scalar_output_flag(name)


def _wrapper_signature_args(args: List[str], writes_to_args: Set[str]) -> List[str]:
    """
    Decide wrapper signature for Bridge mode.

    Rule:
      - If an arg is written-to AND is a scalar output flag, drop it from wrapper signature.
      - Keep everything else (including arrays and dimensions like n).
    """
    out: List[str] = []
    for a in args:
        al = a.lower()
        if (al in writes_to_args) and _is_scalar_output_flag(al) and (not _is_dim_name(al)):
            # output-only scalar flag => not an input param
            continue
        out.append(a)
    return out


def _wrapper_return_values_in_arg_order(args: List[str], writes_to_args: Set[str]) -> List[str]:
    """
    Return written-to args in the SAME order as the original argument list.

    This is the Bridge-safe canonical ordering:
      returns = [a for a in args if a is written-to and not a dimension-name]
    """
    out: List[str] = []
    for a in args:
        al = a.lower()
        if al in writes_to_args and (not _is_dim_name(al)):
            out.append(a)
    return out


def sanitize_module_stem(stem: str) -> str:
    """Python module name rules: letters, digits, underscore; must not start with digit"""
    s = re.sub(r"[^A-Za-z0-9_]+", "_", stem)
    if re.match(r"^\d", s):
        s = "_" + s
    return s or "unknown"


# ============================================================================
# MODULE VARIABLE SUPPORT
# ============================================================================

def extract_module_info(merged_packet: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Extract MODULE variable information from merged packet.
    
    Implements MODULE_VARIABLE_POLICY:
    - parent_module: Which MODULE this procedure belongs to
    - module_vars_used: Which MODULE variables this procedure uses
    
    Returns:
        (parent_module, module_vars_used)
    """
    parent_module = merged_packet.get("parent_module", "")
    # Shared rule (declaration-ordered direct refs minus config excludes) —
    # bridges/prompts/lint use the same one, so signatures always agree.
    module_vars_used = packet_module_vars(merged_packet)

    # A module var that is also a dummy argument is already in the signature —
    # keeping it would emit a duplicate parameter (a Python SyntaxError in the
    # generated wrapper). Mirrors the dedup in phase03_make_bridge.
    arg_names = {a.lower()
                 for a in (merged_packet.get("deps", {}) or {}).get("args", []) or []}
    module_vars_used = [v for v in (module_vars_used or [])
                        if v.lower() not in arg_names]

    return parent_module, module_vars_used


# ============================================================================
# WRAPPER GENERATION
# ============================================================================

def generate_wrapper(
    proc_name: str, 
    merged_packet: Dict[str, Any],
    parent_module: str = "",
    module_vars_used: List[str] = None
) -> str:
    """
    Generate wrapper for BRIDGE MODE (no layout conversion).
    Simple pass-through to _core.
    
    ENHANCED: MODULE_VARIABLE_POLICY support
    - Adds module vars to wrapper signature
    - Passes module vars to core
    - Returns module vars from core
    """
    module_vars_used = module_vars_used or []
    has_module_vars = bool(module_vars_used and parent_module)
    
    deps = merged_packet["deps"]
    args = _as_list(deps.get("args", []))
    writes_to_args_raw = _as_list(deps.get("writes_to_args", []))
    writes_to_args_set = {w.lower() for w in writes_to_args_raw}

    effects = deps.get("effects", {}) or {}
    has_io = bool(effects.get("has_io", False))

    # Wrapper signature excludes scalar output flags that are written outputs
    wrapper_args = _wrapper_signature_args(args, writes_to_args_set)
    
    # NEW: Add module vars to wrapper signature
    if has_module_vars:
        wrapper_args.extend(module_vars_used)

    # Return order must be stable and policy-compatible:    
    returns = _wrapper_return_values_in_arg_order(args, writes_to_args_set)
    
    # NEW: Add module vars to returns (even if unchanged)
    if has_module_vars:
        returns.extend(module_vars_used)

    # Emission point: Fortran names become Python identifiers here.
    wrapper_args_str = ", ".join(safe_py_name(a) for a in wrapper_args)
    core_call_args_str = ", ".join(safe_py_name(a) for a in wrapper_args)  # core called with the wrapper args

    # Docstring
    doc = '    """\n'
    doc += f"    Wrapper for {proc_name} (Bridge Mode).\n\n"
    doc += "    Bridge handles layout conversion separately.\n"
    doc += "    This wrapper passes through to _core.\n\n"
    
    if has_module_vars:
        doc += f"    MODULE VARIABLES (from {parent_module}):\n"
        for var in module_vars_used:
            doc += f"        {var}: MODULE variable (INOUT)\n"
        doc += "\n"
    
    doc += "    Parameters:\n"
    for a in wrapper_args:
        al = a.lower()
        
        # Check if this is a module variable
        if has_module_vars and al in [v.lower() for v in module_vars_used]:
            doc += f"        {a}: MODULE variable\n"
        else:
            role = "inout/out" if al in writes_to_args_set else "in"
            doc += f"        {a}: {role}\n"
        
    if returns:
        doc += "\n    Returns:\n"
        return_list = []
        for r in returns:
            rl = r.lower()
            if has_module_vars and rl in [v.lower() for v in module_vars_used]:
                return_list.append(f"{r} (MODULE variable)")
            else:
                return_list.append(r)
        doc += f"        {', '.join(return_list)}\n"

    if has_io:
        doc += "\n    Notes:\n"
        doc += "        This routine has I/O effects; I/O must remain in wrapper, not in _core.\n"
    
    if has_module_vars:
        doc += "\n    MODULE_VARIABLE_POLICY:\n"
        doc += "        Module variables are passed as INOUT parameters.\n"
        doc += "        Driver manages MODULE vars explicitly (no hidden state).\n"

    doc += '    """\n'

    code = f"def {proc_name}({wrapper_args_str}):\n"
    code += doc

    if has_io:
        code += "    # I/O is allowed here (wrapper), but must not appear in _core.\n"

    if returns:
        _rets = ", ".join(safe_py_name(r) for r in returns)
        code += f"    {_rets} = {proc_name}_core({core_call_args_str})\n"
        code += f"    return {_rets}\n"
    else:
        code += f"    return {proc_name}_core({core_call_args_str})\n"

    return code


def generate_scalar_only_module(proc_name: str, merged_packet: Dict[str, Any]) -> str:
    """
    Generate a plain Python module for scalar-only procedures.

    No _core, no JAX imports, no JIT overhead.
    The function is a straightforward Python callable.
    """
    deps = merged_packet["deps"]
    args = _as_list(deps.get("args", []))
    writes_to_args_raw = _as_list(deps.get("writes_to_args", []))
    writes_to_args_set = {w.lower() for w in writes_to_args_raw}

    parent_module, module_vars_used = extract_module_info(merged_packet)
    has_module_vars = bool(module_vars_used and parent_module)

    # Full signature: regular args (no scalar-flag filtering needed here)
    # + module vars
    fn_args = list(args)
    if has_module_vars:
        fn_args.extend(module_vars_used)

    # Returns: written args + module vars
    returns = _wrapper_return_values_in_arg_order(args, writes_to_args_set)
    if has_module_vars:
        returns.extend(module_vars_used)

    # Module header
    code = '"""\n'
    code += f"Wrapper module for {proc_name}\n"
    code += "Auto-generated by workflow_translator/phase04_02_make_wrappers.py\n\n"
    code += "SCALAR-ONLY procedure: No arrays — plain Python, no JAX overhead.\n"
    if has_module_vars:
        code += f"\nMODULE: {parent_module}\n"
        code += f"MODULE variables: {', '.join(module_vars_used)}\n"
        code += "\nMODULE_VARIABLE_POLICY (INOUT Pattern):\n"
        code += "- Module variables passed as INOUT parameters\n"
        code += "- Driver manages MODULE vars explicitly\n"
        code += "- No hidden state dictionary\n"
    code += '"""\n\n\n'

    # Single plain Python function — no _core
    fn_args_str = ", ".join(fn_args)
    code += f"def {proc_name}({fn_args_str}):\n"
    code += '    """\n'
    code += f"    Plain Python implementation of {proc_name}.\n"
    code += "    SCALAR-ONLY: no arrays, no JAX JIT overhead.\n\n"
    if has_module_vars:
        code += f"    MODULE VARIABLES (from {parent_module}):\n"
        for var in module_vars_used:
            code += f"        {var}: MODULE variable (INOUT)\n"
        code += "\n    IMPORTANT: Module variables MUST be returned!\n\n"
    code += "    TO BE FILLED BY LLM TRANSLATION\n"
    code += '    """\n'
    code += "    raise NotImplementedError('Fill this via LLM translation')\n"

    return code


def generate_wrapper_module(proc_name: str, merged_packet: Dict[str, Any]) -> str:
    """
    Generate complete wrapper module (bridge mode only).

    SCALAR-ONLY procedures get a plain Python function (no _core, no JAX).
    Array procedures get the standard _core + wrapper pattern.
    """
    if _is_scalar_only_proc(merged_packet):
        return generate_scalar_only_module(proc_name, merged_packet)

    deps = merged_packet["deps"]
    args = _as_list(deps.get("args", []))
    writes_to_args_raw = _as_list(deps.get("writes_to_args", []))
    writes_to_args_set = {w.lower() for w in writes_to_args_raw}

    # Extract module info
    parent_module, module_vars_used = extract_module_info(merged_packet)
    has_module_vars = bool(module_vars_used and parent_module)

    # Core signature: wrapper args + module vars
    core_args = _wrapper_signature_args(args, writes_to_args_set)

    if has_module_vars:
        core_args.extend(module_vars_used)

    # Module header
    code = '"""\n'
    code += f"Wrapper module for {proc_name}\n"
    code += "Auto-generated by workflow_translator/phase04_02_make_wrappers.py\n\n"
    code += "Bridge Mode: Layout conversion handled by bridge layer.\n"
    code += "This wrapper is a simple pass-through to _core.\n"

    if has_module_vars:
        code += f"\nMODULE: {parent_module}\n"
        code += f"MODULE variables: {', '.join(module_vars_used)}\n"
        code += "\nMODULE_VARIABLE_POLICY (INOUT Pattern):\n"
        code += "- Module variables passed as INOUT parameters\n"
        code += "- Driver manages MODULE vars explicitly\n"
        code += "- No hidden state dictionary\n"

    code += '"""\n\n'

    # Imports
    code += "import jax\n"
    code += "import jax.numpy as jnp\n"
    code += "from jax import lax\n\n\n"

    # Core function stub (to be filled by LLM translation)
    core_args_str = ", ".join(safe_py_name(a) for a in core_args)
    code += f"def {proc_name}_core({core_args_str}):\n"
    code += '    """\n'
    code += f"    Core computation for {proc_name}.\n"
    code += "    Pure JAX function (no I/O, jit-compatible).\n\n"
    code += "    Operates on row-major arrays (standard JAX layout).\n"
    code += "    Bridge handles Fortran column-major conversion.\n\n"

    if has_module_vars:
        code += f"    MODULE VARIABLES (from {parent_module}):\n"
        for var in module_vars_used:
            code += f"        {var}: MODULE variable\n"
        code += "\n"
        code += "    IMPORTANT: Module variables MUST be returned!\n\n"

    code += "    TO BE FILLED BY LLM TRANSLATION\n"
    code += '    """\n'
    code += "    raise NotImplementedError('Fill this via LLM translation')\n\n\n"

    # Wrapper
    code += generate_wrapper(proc_name, merged_packet, parent_module, module_vars_used)

    return code


# ============================================================================
# WRAPPER INDEX GENERATION
# ============================================================================

def generate_wrapper_index(wrappers_dir: Path) -> None:
    """
    Generate wrappers.py index file for stable imports.

    This creates a single import surface:
        from out.wrappers.wrappers import <function_name>
    """
    # Collect wrapper modules (exclude wrappers.py itself and __init__.py)
    py_files = sorted(
        p for p in wrappers_dir.glob("*.py")
        if p.name not in ("wrappers.py", "__init__.py")
    )

    if not py_files:
        print("⚠️  No wrapper files found, skipping index generation")
        return

    lines: list[str] = []
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append('"""')
    lines.append("Auto-generated wrapper index.")
    lines.append("")
    lines.append("This file provides a stable import surface:")
    lines.append("  from out.wrappers.wrappers import Foo, Bar, ...")
    lines.append("")
    lines.append("Do not edit by hand; regenerate via workflow_translator/phase04_02_make_wrappers.py")
    lines.append('"""')
    lines.append("")
    lines.append("# Re-export all wrapper callables found in out/wrappers/*.py")
    lines.append("")

    exported: list[str] = []

    for p in py_files:
        mod = sanitize_module_stem(p.stem)
        fn = p.stem
        core_fn = f"{p.stem}_core"

        # Always export wrapper
        lines.append(f"from {mod} import {fn}  # noqa: F401")
        exported.append(fn)

        # Export core if present (wrapper-only procedures may not have it)
        src = p.read_text(encoding="utf-8", errors="replace")
        if re.search(rf"(?m)^\s*def\s+{re.escape(core_fn)}\s*\(", src):
            lines.append(f"from {mod} import {core_fn}  # noqa: F401")
            exported.append(core_fn)
            
    lines.append("")
    lines.append("__all__ = [")
    for name in exported:
        lines.append(f'    "{name}",')
    lines.append("]")
    lines.append("")

    out_path = wrappers_dir / "wrappers.py"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    
    print(f"\n✅ Wrapper index: {out_path}")
    print(f"   Exported {len(exported)} functions from {len(py_files)} modules")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Generate wrappers and index for all merged packets"""
    cfg = get_config()
    packets_dir = cfg.packets_dir
    # Reference skeletons go to out/wrappers/ — NEVER out/jax/, where the LLM
    # translations live. Rerunning this phase is therefore always safe.
    wrappers_dir = cfg.wrappers_dir
    wrappers_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 4.2: Wrapper Generation (UNIFIED)")
    print("ENHANCED: MODULE_VARIABLE_POLICY Support")
    print("=" * 70)
    print("✅ Generating pass-through wrappers")
    print("   - No swapaxes")
    print("   - No aF convention")
    print("   - Bridge handles layout separately")
    print("   - MODULE variables as parameters")
    print("   - Policy A removed entirely")
    print(f"   - Output: {wrappers_dir}/ (reference only — out/jax/ is never touched)")
    print()

    merged_files = sorted(packets_dir.glob("*_merged.json"))
    if not merged_files:
        raise SystemExit(f"No merged packets found in {packets_dir}. Run Phase 02 first.")

    # Always regenerate __init__.py as a plain package marker.
    # Do NOT put imports here — stale imports break loading.
    # Importable symbols live in wrappers.py (generated below).
    init_py = wrappers_dir / "__init__.py"
    init_py.write_text("# Auto-generated package marker\n", encoding="utf-8")
    print(f"✅ Package init: {init_py}")

    # Generate wrapper modules
    wrote = 0
    for pkt_file in merged_files:
        packet = json.loads(pkt_file.read_text(encoding="utf-8"))
        proc_name = packet["proc_name"]
        
        # Extract module info for reporting
        parent_module, module_vars_used = extract_module_info(packet)

        wrapper_code = generate_wrapper_module(proc_name, packet)

        out_path = wrappers_dir / f"{proc_name}.py"
        out_path.write_text(wrapper_code, encoding="utf-8")

        is_scalar = _is_scalar_only_proc(packet)
        tag = "[SCALAR-ONLY]" if is_scalar else ""
        if module_vars_used:
            print(f"✅ {proc_name} → {out_path.name} {tag} (MODULE vars: {', '.join(module_vars_used)})")
        else:
            print(f"✅ {proc_name} → {out_path.name} {tag}")
        
        wrote += 1

    # Generate wrapper index
    generate_wrapper_index(wrappers_dir)

    print("\n" + "=" * 70)
    print(f"🎉 Generated {wrote} wrapper modules")
    print("=" * 70)
    print(f"   Location: {wrappers_dir}/ (reference skeletons — translations go in out/jax/)")
    print("   Mode: Bridge only (Policy A removed)")
    print("   MODULE_VARIABLE_POLICY: Enabled")
    print("\n📦 Import surface ready:")
    print("   from out.wrappers.wrappers import <function_name>")
    print("   from out.wrappers.<module> import <function_name>")
    print("\n📝 Next: LLM translations go to out/jax/{proc}.py, using these")
    print("   skeletons as the signature reference.")
    print("   Remember: Module variables MUST be in core signature and returns!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    from framework_config import init as init_config, add_config_arg

    ap = argparse.ArgumentParser(description="Phase 04.2: wrapper generation")
    add_config_arg(ap)
    init_config(ap.parse_args().config)
    main()