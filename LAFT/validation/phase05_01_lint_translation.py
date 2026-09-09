# -----------------------------------------------------------
# Phase 5.1: Lint and Validate Translated Python/JAX Code (BRIDGE MODE ONLY)
#
# This script lints translated modules in out/jax/ against the framework rules:
#   - JAX-only (no NumPy)
#   - Bridge mode only: compute code must be row-major and must NOT contain swapaxes/aF logic
#   - Two-layer structure for compute routines: <proc>_core + <proc>
#   - Scalar procs called from a @jax.jit context (jax_required=True in
#     out/packets/_ALL_deps.json) are SCALAR-JAX: jnp only, no _core, no @jax.jit
#   - Scalar procs with jax_required=False may follow EITHER contract (plain
#     Python, or scalar-JAX style — jnp scalar code is also valid eagerly);
#     lint holds the file to whichever contract it follows
#   - There is NO host-side contract for non-scalar procs: every procedure
#     with grid/state arrays (including the scheme's top-level main routine
#     and model-facing wrappers) is held to the full JAX two-layer contract,
#     and additionally must contain no host compute loops (Python for/while
#     writing array elements or wrapping JAX calls). See
#     workflow_translator/prompt_policies/orchestration_boundary_rules.md
#   - Wrapper signature must be a SUPERSET of the Fortran args (module-level
#     LUT tables and constants are passed explicitly per MODULE_VARIABLE_POLICY
#     and the packet's module_vars_used under-lists them); missing args fail
#   - Wrapper-only allowed for pure I/O routines (effects.has_io=True and no writes/calls)
#   - Core must contain no I/O; wrapper may contain I/O only when has_io=True
#   - Wrapper signature must exclude scalar output-only flags (e.g., isok) per generator policy
#   - Bridge/JAX signature consistency: the bridge is generated from packets
#     alone (phase03 never reads out/jax/), so the bridge's call must match
#     the translation's actual `def <proc>(...)` signature — same names, no
#     extras on either side, same positional order
#
# NEW: Saves lint results to out/lint/{proc}_lint.json for AI comparison
# -----------------------------------------------------------

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import List, Set, Dict, Any, Tuple
from datetime import datetime

# framework_config lives in LAFT/config/; this script now lives in
# validation/ (symlinked from LAFT/ into each project) — resolve back to
# LAFT/config to import it. Same convention as workflow_bridge/phase03_make_bridge.py.
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "config"))

from framework_config import get_config, packet_module_vars


def find_host_compute_loops(source: str) -> Tuple[List[int], List[int]]:
    """
    Detect host compute loops — the two mechanically-checkable signatures of
    grid work escaping the JIT boundary (orchestration_boundary_rules.md):

      1. subscript stores: a Python for/while loop whose body assigns to an
         array element (``x[i] = ...`` / ``x[i] += ...``). String-constant
         keys (``d["name"] = ...``) are excluded — that is dict bookkeeping,
         not grid mutation.
      2. JAX-in-host-loop: a Python for/while loop whose body references
         ``jnp.*`` / ``lax.*`` / ``jax.*`` or calls a ``*_core`` function —
         Level 5 of the VECTORIZATION PRIORITY LADDER (a Python loop
         wrapping any JAX construct).

    Returns (subscript_store_lines, jax_call_lines) — line numbers of the
    offending statements, empty lists when the file is clean. Raises nothing:
    a file that fails to parse returns ([], []) and is left to other checks.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], []

    def _is_str_key(sub: ast.Subscript) -> bool:
        return isinstance(sub.slice, ast.Constant) and isinstance(sub.slice.value, str)

    store_lines: List[int] = []
    jaxcall_lines: List[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.While)):
            continue
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign):
                if any(isinstance(t, ast.Subscript) and not _is_str_key(t)
                       for t in stmt.targets):
                    store_lines.append(stmt.lineno)
            elif isinstance(stmt, ast.AugAssign):
                if isinstance(stmt.target, ast.Subscript) and not _is_str_key(stmt.target):
                    store_lines.append(stmt.lineno)
            if isinstance(stmt, ast.Attribute) and isinstance(stmt.value, ast.Name) \
                    and stmt.value.id in ("jnp", "lax", "jax"):
                jaxcall_lines.append(stmt.lineno)
            elif isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Name) \
                    and stmt.func.id.endswith("_core"):
                jaxcall_lines.append(stmt.lineno)
    return sorted(set(store_lines)), sorted(set(jaxcall_lines))


def _is_scalar_only_proc(merged: Dict[str, Any]) -> bool:
    """
    Returns True if this procedure has NO array arguments.
    Scalar-only procedures must NOT have a _core or JAX imports.
    Mirrors the detection in phase04_01 and phase04_02.
    """
    fortran_source = merged.get("phase2_packet", {}).get("fortran_source", "")
    if not fortran_source:
        return False
    array_patterns = [
        r"::\s*\w+\s*\(",      # :: varname( — array in declaration
        r"\bdimension\s*\(",   # dimension( attribute
    ]
    for pattern in array_patterns:
        if re.search(pattern, fortran_source, re.IGNORECASE):
            return False
    return True


_JAX_REQUIRED_MAP: Dict[str, bool] = None


def _load_jax_required_map() -> Dict[str, bool]:
    """
    Load jax_required flags from out/packets/_ALL_deps.json (computed by
    phase02_02). Mirrors phase04_01: missing file or entry defaults to True.
    """
    global _JAX_REQUIRED_MAP
    if _JAX_REQUIRED_MAP is None:
        all_deps = get_config().all_deps_file
        if all_deps.exists():
            data = json.loads(all_deps.read_text(encoding="utf-8"))
            _JAX_REQUIRED_MAP = {
                e["proc_name"].lower(): e.get("jax_required", True)
                for e in data
                if e.get("proc_name")
            }
        else:
            print("⚠️  _ALL_deps.json not found — defaulting all procedures to jax_required=True")
            _JAX_REQUIRED_MAP = {}
    return _JAX_REQUIRED_MAP


def list_procs_from_jax_dir() -> List[str]:
    """
    Discover procedures from actual .py files in out/jax/, excluding __init__.py.
    Uses the file stem as the canonical proc name so casing matches the Python source.
    """
    skip = {"__init__"}
    return sorted(
        p.stem
        for p in get_config().jax_dir.glob("*.py")
        # *.validated.py are permanent snapshots saved by the workflows,
        # not procedures — linting them double-counts and crashes on the
        # missing packet. *_passN.py are multi-pass translation
        # intermediates (e.g. left behind by a PBS translator) —
        # only the bare final name is a procedure.
        if p.stem not in skip
        and not p.stem.endswith(".validated")
        and not re.search(r"_pass\d+$", p.stem)
    )


def list_procs_from_packets() -> List[str]:
    """
    Prefer merged packets if present; otherwise fall back to _ALL_deps.json.
    """
    cfg = get_config()
    merged_files = sorted(cfg.packets_dir.glob("*_merged.json"))
    if merged_files:
        return [p.name.replace("_merged.json", "") for p in merged_files]

    all_deps = cfg.all_deps_file
    if all_deps.exists():
        data = json.loads(all_deps.read_text(encoding="utf-8"))
        out: List[str] = []
        seen = set()
        for item in data:
            n = (item.get("proc_name") or "").strip()
            if n and n.lower() not in seen:
                seen.add(n.lower())
                out.append(n)
        return out

    raise SystemExit("Could not discover procedures: no *_merged.json and no _ALL_deps.json")


def extract_function_block(source: str, func_name: str) -> str:
    """
    Best-effort: extract the text region for 'def func_name(...):' until the next top-level 'def '.
    Not perfect, but enough for linting print()/input() presence in core vs wrapper.
    """
    m = re.search(rf"(?m)^\s*def\s+{re.escape(func_name)}\s*\(", source)
    if not m:
        return ""
    start = m.start()

    m2 = re.search(r"(?m)^\s*def\s+\w+\s*\(", source[m.end() :])
    if not m2:
        return source[start:]
    end = m.end() + m2.start()
    return source[start:end]


def _is_dim_name(name: str) -> bool:
    # Single source: [heuristics].dim_names in config/project.toml
    return get_config().is_dim_name(name)


def _is_scalar_output_flag(name: str) -> bool:
    # Single source: [heuristics].scalar_flag_names in config/project.toml
    return get_config().is_scalar_output_flag(name)


def expected_wrapper_args(
    args: List[str], 
    writes_to_args: Set[str],
    module_vars_used: List[str] = None  # NEW: MODULE variables
) -> Set[str]:
    """
    Mirror the wrapper generator policy:
    - Drop scalar output-only flags from wrapper signature (e.g., isok)
    - Add MODULE variables to signature (MODULE_VARIABLE_POLICY)
    """
    module_vars_used = module_vars_used or []
    
    scalar_out = {
        a.lower()
        for a in args
        if (a.lower() in writes_to_args)
        and (not _is_dim_name(a))
        and (_is_scalar_output_flag(a))
    }
    
    # Start with Fortran args (minus scalar output flags)
    expected = {a.lower() for a in args if a.lower() not in scalar_out}
    
    # Add MODULE variables
    expected.update(v.lower() for v in module_vars_used)
    
    return expected

_MODULE_VAR_NAMES: Set[str] = None


def _load_module_var_names() -> Set[str]:
    """All module-level variable names (lowercase) from out/modules/*.json."""
    global _MODULE_VAR_NAMES
    if _MODULE_VAR_NAMES is None:
        _MODULE_VAR_NAMES = set()
        for f in get_config().modules_dir.glob("*.json"):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                _MODULE_VAR_NAMES |= {
                    v["name"].lower() for v in d.get("module_variables", [])
                }
            except (json.JSONDecodeError, OSError, KeyError, TypeError):
                continue
    return _MODULE_VAR_NAMES


def _fortran_optional_args(fortran_src: str) -> Set[str]:
    """Names declared with the OPTIONAL attribute (lowercase)."""
    names: Set[str] = set()
    for line in fortran_src.lower().splitlines():
        line = line.split("!")[0]
        if "optional" in line and "::" in line:
            for nm in line.split("::", 1)[1].split(","):
                nm = nm.strip().split("(")[0].strip()
                if nm:
                    names.add(nm)
    return names


def parse_wrapper_args(source: str, func_name: str):
    """
    Parse the parameter names of `def func_name(...)`, robust to multi-line
    signatures, `# comments` inside the signature (which may contain parens),
    defaults, and annotations. Returns None if the def is not found or the
    parens never balance.
    """
    m = re.search(rf"(?m)^def\s+{re.escape(func_name)}\s*\(", source)
    if m is None:
        return None
    i = m.end() - 1  # at the opening '('
    depth = 0
    buf = []
    n = len(source)
    while i < n:
        ch = source[i]
        if ch == "#":
            while i < n and source[i] != "\n":
                i += 1
            continue
        if ch in "([{":
            depth += 1
            if depth == 1:
                i += 1
                continue  # don't record the outer '('
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                break
        buf.append(ch)
        i += 1
    else:
        return None
    if depth != 0:
        return None

    # split on top-level commas
    parts, cur, d2 = [], [], 0
    for ch in "".join(buf):
        if ch in "([{":
            d2 += 1
        elif ch in ")]}":
            d2 -= 1
        if ch == "," and d2 == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))

    out = []
    for p in parts:
        name = p.split("=")[0].split(":")[0].strip().lstrip("*")
        if name and name != "/":
            out.append(name)
    return out


def _parse_def_params(source: str, func_name_lower: str):
    """
    Return (ordered_param_names, defaulted_param_names) for the top-level
    `def` whose name matches func_name_lower case-insensitively. Uses ast,
    so it is robust to multi-line signatures and comments inside them.
    Returns (None, None) when the source does not parse or the def is absent.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None, None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.lower() == func_name_lower:
            pos = list(node.args.posonlyargs) + list(node.args.args)
            names = [a.arg for a in pos]
            defaulted: Set[str] = set()
            if node.args.defaults:
                defaulted = {a.arg for a in pos[len(pos) - len(node.args.defaults):]}
            names += [a.arg for a in node.args.kwonlyargs]
            defaulted |= {a.arg for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults)
                          if d is not None}
            return names, defaulted
    return None, None


def _parse_bridge_call(bridge_src: str, proc_lower: str, bridge_param_names: Set[str]):
    """
    Find the bridge's call to the translated JAX function and return
    (positional_names, keyword_names). Positional entries are the argument's
    base variable name — conversion suffixes (_jax/_compute/_fortran/_j) are
    stripped when the base is a bridge parameter — or None for expressions
    whose bound name cannot be determined statically. Returns (None, None)
    when the source does not parse or no such call exists.
    """
    try:
        tree = ast.parse(bridge_src)
    except SyntaxError:
        return None, None

    def _base_name(node) -> str:
        if not isinstance(node, ast.Name):
            return None
        nm = node.id.lower()
        for suffix in ("_jax", "_compute", "_fortran", "_j"):
            if nm.endswith(suffix) and nm[: -len(suffix)] in bridge_param_names:
                return nm[: -len(suffix)]
        return nm

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id.lower() == proc_lower:
            pos = [_base_name(a) for a in node.args]
            kw = [k.arg.lower() for k in node.keywords if k.arg]
            return pos, kw
    return None, None


def require_bridge_metadata(proc: str) -> bool:
    """Bridge-only lint: require bridge file to exist (case-insensitive)."""
    return _find_bridge(proc).exists()

def save_lint_results(proc: str, checks: List[tuple], metadata: Dict[str, Any]):
    """
    Save lint results to JSON file for AI comparison
    
    Args:
        proc: Procedure name
        checks: List of (check_name, passed, details) tuples
        metadata: Additional metadata about the lint run
    """
    lint_dir = get_config().lint_dir
    lint_dir.mkdir(parents=True, exist_ok=True)
    
    # Build check results
    check_results = []
    for item in checks:
        if len(item) == 2:
            name, passed = item
            details = None
        else:
            name, passed, details = item
        
        check_results.append({
            "name": name,
            "passed": passed,
            "details": details
        })
    
    # Calculate summary
    total = len(check_results)
    passed_count = sum(1 for c in check_results if c["passed"])
    failed_count = total - passed_count
    
    # Build full result
    result = {
        "proc": proc,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata,
        "checks": check_results,
        "summary": {
            "total_checks": total,
            "passed": passed_count,
            "failed": failed_count,
            "score": round(100 * passed_count / total, 1) if total > 0 else 0,
            "overall": "PASS" if failed_count == 0 else "FAIL"
        }
    }
    
    # Save to file
    output_file = lint_dir / f"{proc}_lint.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    return output_file


def _find_packet(proc: str) -> Path:
    """Return the merged packet path for proc, matching case-insensitively."""
    packets_dir = get_config().packets_dir
    exact = packets_dir / f"{proc}_merged.json"
    if exact.exists():
        return exact
    proc_lower = proc.lower()
    for p in packets_dir.glob("*_merged.json"):
        if p.name.replace("_merged.json", "").lower() == proc_lower:
            return p
    return exact  # return non-existent path so caller can report the error


def _find_bridge(proc: str) -> Path:
    """Return the bridge path for proc, matching case-insensitively."""
    bridge_dir = get_config().bridge_dir
    exact = bridge_dir / f"{proc}_bridge.py"
    if exact.exists():
        return exact
    proc_lower = proc.lower()
    for p in bridge_dir.glob("*_bridge.py"):
        if p.name.replace("_bridge.py", "").lower() == proc_lower:
            return p
    return exact


def main(proc: str) -> int:
    merged_path = _find_packet(proc)
    if not merged_path.exists():
        raise SystemExit(f"Missing packet for '{proc}'. Run Phase 3.1 merge first.")

    merged = json.loads(merged_path.read_text(encoding="utf-8"))
    deps = merged.get("deps", {}) or {}

    args = deps.get("args", []) or []
    writes_to_args = set(a.lower() for a in deps.get("writes_to_args", []) or [])

    # Same module-var rule the generators use (declaration-ordered direct
    # refs minus config excludes) — lint must expect the signature the
    # bridge/wrapper generators actually emit.
    module_vars_used = packet_module_vars(merged)

    effects = deps.get("effects") or merged.get("effects") or {}
    has_io = bool(effects.get("has_io", False))

    calls = deps.get("calls", []) or []
    wrapper_only = has_io and (len(writes_to_args) == 0) and (len(calls) == 0)
    scalar_only = _is_scalar_only_proc(merged)
    jax_required = _load_jax_required_map().get(proc.lower(), True)

    pyfile = get_config().jax_dir / f"{proc}.py"
    if not pyfile.exists():
        raise SystemExit(f"Missing {pyfile}. Paste model output there first.")

    s = pyfile.read_text(encoding="utf-8", errors="replace")

    checks = []

    # Enforce bridge metadata exists (bridge-only framework)
    check_val = require_bridge_metadata(proc)
    details = None if check_val else "Bridge metadata file missing"
    checks.append(("has bridge metadata", check_val, details))

    uses_jnp_import = "import jax.numpy as jnp" in s
    # Scalar procs with jax_required=False may follow EITHER contract:
    # plain Python, or scalar-JAX style (jnp scalar code is also valid when
    # called eagerly from host-side code). Hold the file to whichever
    # contract it follows.
    scalar_jax = scalar_only and (jax_required or uses_jnp_import)
    # NOTE: there is deliberately NO host-side contract for non-scalar procs.
    # Every procedure with grid/state arrays — including the scheme's
    # top-level main routine and model-facing wrappers — is held to the full
    # JAX two-layer contract below, regardless of jax_required in a stale
    # _ALL_deps.json. See orchestration_boundary_rules.md.

    if scalar_only and not scalar_jax:
        # ----------------------------------------------------------------
        # SCALAR-ONLY checks: plain Python, no JAX, no _core
        # ----------------------------------------------------------------
        has_jax = "import jax" in s
        checks.append(("no jax imports (scalar-only)", not has_jax,
                        "Found 'import jax' — scalar-only proc must be plain Python" if has_jax else None))

        has_numpy = "import numpy" in s or re.search(r"\bnp\.", s) is not None
        if has_io:
            checks.append(("numpy allowed (scalar I/O loader)", True,
                            "NumPy usage allowed — has_io=True, arrays are materialized host-side" if has_numpy else None))
        else:
            checks.append(("no numpy", not has_numpy, "Found NumPy usage" if has_numpy else None))

        has_wrapper = f"def {proc}(" in s
        checks.append(("has wrapper", has_wrapper,
                        None if has_wrapper else f"Missing 'def {proc}(' function"))

        has_core = f"def {proc}_core" in s
        checks.append(("no core (scalar-only)", not has_core,
                        f"Found 'def {proc}_core' — scalar-only procs must NOT have a _core (no GPU overhead)" if has_core else None))

        core_block = ""
        wrapper_block = extract_function_block(s, proc)

    elif scalar_jax:
        # ----------------------------------------------------------------
        # SCALAR-JAX checks: no array args, but callable from a @jax.jit
        # context (jax_required=True in _ALL_deps.json, or voluntarily
        # jnp-style). Args may be traced scalars, so the body must be
        # trace-safe — but the JIT boundary lives in the caller, so still
        # no _core and no @jax.jit.
        # Mirrors phase04_01 get_scalar_jax_prompt_header.
        # ----------------------------------------------------------------
        # Pure arithmetic (no branching, casts, or math.*) is trace-safe
        # even without jnp (e.g. G_of_mu is a single rational expression).
        pure_arith = re.search(
            r"(?m)^\s*(if|elif|else\b|while|for)\b|\bmath\.|\bint\(|\bfloat\(", s) is None
        checks.append(("uses jnp (scalar-jax)", uses_jnp_import or pure_arith,
                        None if uses_jnp_import else
                        ("pure arithmetic — trace-safe without jnp" if pure_arith else
                         "Missing 'import jax.numpy as jnp' and body has branching/casts — not trace-safe")))
        if not jax_required:
            checks.append(("jax optional (scalar-jax style)", True,
                            "jax_required=False but scalar-JAX style is also valid when called eagerly"))

        bare_jax_import = re.search(r"(?m)^\s*import\s+jax\s*(?:$|#)", s) is not None
        checks.append(("no bare 'import jax' (scalar-jax)", not bare_jax_import,
                        "Found bare 'import jax' — scalar-jax procs import only jax.numpy as jnp" if bare_jax_import else None))

        has_numpy = "import numpy" in s or re.search(r"\bnp\.", s) is not None
        checks.append(("no numpy", not has_numpy, "Found NumPy usage" if has_numpy else None))

        has_wrapper = f"def {proc}(" in s
        checks.append(("has wrapper", has_wrapper,
                        None if has_wrapper else f"Missing 'def {proc}(' function"))

        has_core = f"def {proc}_core" in s
        checks.append(("no core (scalar-jax)", not has_core,
                        f"Found 'def {proc}_core' — JIT boundary lives in the caller; scalar-jax procs must NOT have a _core" if has_core else None))

        # Match actual jit usage (decorator line or jax.jit(...) call), not
        # docstring mentions like "called from inside @jax.jit".
        has_jit = re.search(r"(?m)^\s*@(?:jax\.)?jit\b|jax\.jit\s*\(", s) is not None
        checks.append(("no @jax.jit (scalar-jax)", not has_jit,
                        "Found jax.jit — JIT boundary lives in the caller" if has_jit else None))

        sets_x64 = "JAX_ENABLE_X64" in s
        checks.append(("no JAX_ENABLE_X64 (scalar-jax)", not sets_x64,
                        "Sets JAX_ENABLE_X64 — the caller's top-level file handles that" if sets_x64 else None))

        core_block = ""
        wrapper_block = extract_function_block(s, proc)

    else:
        # ----------------------------------------------------------------
        # Array procedure checks: JAX required, _core required.
        # Applies to EVERY non-scalar proc — the scheme's main routine and
        # model-facing wrappers included (orchestration_boundary_rules.md).
        # ----------------------------------------------------------------
        store_lines, jaxcall_lines = find_host_compute_loops(s)
        checks.append(("no host compute loops (element stores)", not store_lines,
                        (f"{len(store_lines)} array-element store(s) inside Python for/while "
                         f"(first at line {store_lines[0]}) — grid mutation must live inside "
                         f"the jitted _core as vectorized ops"
                         if store_lines else None)))
        checks.append(("no Python loop wrapping JAX calls", not jaxcall_lines,
                        (f"{len(jaxcall_lines)} jnp/lax/jax/_core reference(s) inside Python "
                         f"for/while (first at line {jaxcall_lines[0]}) — Level 5 of the "
                         f"VECTORIZATION PRIORITY LADDER, never acceptable"
                         if jaxcall_lines else None)))

        checks.append(("uses jax", "import jax" in s,
                        None if "import jax" in s else "Missing 'import jax'"))
        checks.append(("uses jnp", "import jax.numpy as jnp" in s,
                        None if "import jax.numpy as jnp" in s else "Missing 'import jax.numpy as jnp'"))

        has_numpy = "import numpy" in s or re.search(r"\bnp\.", s) is not None
        checks.append(("no numpy", not has_numpy, "Found NumPy usage" if has_numpy else None))

        has_wrapper = f"def {proc}(" in s
        checks.append(("has wrapper", has_wrapper,
                        None if has_wrapper else f"Missing 'def {proc}(' function"))

        has_swapaxes = "swapaxes(" in s
        checks.append(("no swapaxes in module", not has_swapaxes,
                        "Found swapaxes() - layout conversion should be in bridge, not JAX" if has_swapaxes else None))

        if wrapper_only:
            checks.append(("core optional (wrapper-only I/O)", True, "Pure I/O routine - core not required"))
        else:
            has_core = f"def {proc}_core" in s
            checks.append(("has core", has_core,
                            None if has_core else f"Missing 'def {proc}_core' function"))

        core_block = extract_function_block(s, f"{proc}_core") if not wrapper_only else ""
        wrapper_block = extract_function_block(s, proc)

    # _core must NOT accept string parameters (per jax_jit_boundary_rules.md).
    # Strings are not valid JAX types — they must stay in the wrapper.
    if not scalar_only and not wrapper_only:
        phase1_path = get_config().phase1_index
        char_args: List[str] = []
        if phase1_path.exists():
            try:
                phase1 = json.loads(phase1_path.read_text(encoding="utf-8"))
                # phase1_index may be a dict with 'procedures' or a list — handle both
                proc_entries = phase1.get("procedures", phase1) if isinstance(phase1, dict) else phase1
                if isinstance(proc_entries, list):
                    for entry in proc_entries:
                        if isinstance(entry, dict) and entry.get("name", "").lower() == proc.lower():
                            meta = entry.get("arg_metadata", {}) or {}
                            char_args = [
                                name for name, info in meta.items()
                                if isinstance(info, dict) and info.get("dtype") == "character"
                            ]
                            break
            except (json.JSONDecodeError, OSError):
                pass

        core_sig_match = re.search(rf"def\s+{re.escape(proc)}_core\s*\(([^)]*)\)\s*:", s)
        if core_sig_match and char_args:
            core_arg_names = {
                a.strip().split("=")[0].strip().lower()
                for a in core_sig_match.group(1).split(",") if a.strip()
            }
            offending = sorted(set(name.lower() for name in char_args) & core_arg_names)
            checks.append((
                "core has no string params",
                not offending,
                f"_core accepts string param(s) {offending} — strings must stay in the wrapper "
                f"(see jax_jit_boundary_rules.md 'Strings MUST NOT be passed to _core')"
                if offending else None,
            ))

    # I/O checks — core_block is "" for scalar-only (no core), so core check trivially passes
    core_has_io = ("print(" in core_block) or ("input(" in core_block) or ("open(" in core_block)
    wrapper_has_io = ("print(" in wrapper_block) or ("input(" in wrapper_block) or ("open(" in wrapper_block)

    # Core must never do I/O
    checks.append(("core has no I/O", not core_has_io, "Core contains I/O operations (print/input/open)" if core_has_io else None))

    # Wrapper may do I/O only if has_io=True
    if has_io:
        checks.append(("wrapper I/O allowed (has_io)", True, "I/O allowed in wrapper (has_io=True)"))
    else:
        checks.append(("wrapper has no I/O", not wrapper_has_io, "Wrapper contains I/O but has_io=False" if wrapper_has_io else None))

    # Signature sanity (v3):
    #   FAIL only on a missing required Fortran arg or a param that is
    #   neither a Fortran arg nor a known module variable (invented name).
    #   Allowed with a note: omitted optional Fortran args, module vars not
    #   threaded (baked constants / returned instead), extra module vars
    #   threaded (vestigial from earlier packet data; bridges match wrappers
    #   and the runtime driver is the true arbiter).
    wrapper_args = parse_wrapper_args(s, proc)
    if wrapper_args is None:
        checks.append(("wrapper signature parsed", False, "Could not parse wrapper signature"))
    else:
        actual = {a.lower() for a in wrapper_args}
        scalar_out = {
            a.lower() for a in args
            if a.lower() in writes_to_args
            and not _is_dim_name(a)
            and _is_scalar_output_flag(a)
        }
        required_args = {a.lower() for a in args} - scalar_out
        mvu = {v.lower() for v in module_vars_used}
        fortran_src = merged.get("phase2_packet", {}).get("fortran_source", "") or ""
        optional_args = _fortran_optional_args(fortran_src)
        module_var_names = _load_module_var_names()

        missing_req = sorted(required_args - actual - optional_args)
        missing_opt = sorted((required_args & optional_args) - actual)
        missing_mvu = sorted(mvu - actual)
        extra = actual - required_args - mvu
        extra_mv = sorted(extra & module_var_names)
        extra_unknown = sorted(extra - module_var_names)

        sig_ok = not missing_req and not extra_unknown
        notes = []
        if missing_req:
            notes.append(f"missing required Fortran args: {missing_req}")
        if extra_unknown:
            notes.append(f"unknown extra params (not Fortran args or module vars): {extra_unknown}")
        if missing_opt:
            notes.append(f"omitted optional args (allowed): {missing_opt}")
        if missing_mvu:
            notes.append(f"module vars not threaded (allowed — baked or returned): {missing_mvu}")
        if extra_mv:
            notes.append(f"extra module vars threaded (allowed): {extra_mv}")
        checks.append(("wrapper args match expected", sig_ok, "; ".join(notes) or None))

    # Bridge/JAX signature consistency:
    # phase03 generates the bridge from packets alone and never reads
    # out/jax/*.py, so when the translation's signature diverges from the
    # packet-derived one (a parameter added for a module var only referenced
    # in Fortran comments, a dropped intent(out) arg, reordered params) the
    # bridge's call raises TypeError or silently misbinds values. Compare the
    # bridge's ACTUAL call to the JAX def — the bridge signature itself may
    # legitimately order params differently and reorder at the call site.
    bridge_path = _find_bridge(proc)
    if bridge_path.exists():
        bridge_src = bridge_path.read_text(encoding="utf-8", errors="replace")
        bridge_params, _ = _parse_def_params(bridge_src, f"{proc.lower()}_bridge")
        bridge_param_names = {p.lower() for p in (bridge_params or [])}
        # PATH C (device-side layout conversion): array-bearing bridges call
        # {proc}_core from their emitted jitted device wrapper — that call is
        # the real cross-file coupling to check. Scalar-only bridges still
        # call the translated wrapper {proc}. Try the core call first and
        # fall back, comparing against the matching def either way.
        callee = f"{proc.lower()}_core"
        call_pos, call_kw = _parse_bridge_call(bridge_src, callee, bridge_param_names)
        if call_pos is None:
            callee = proc.lower()
            call_pos, call_kw = _parse_bridge_call(bridge_src, callee, bridge_param_names)
        jax_params, jax_defaulted = _parse_def_params(s, callee)
        if call_pos is None or jax_params is None:
            unparsed = []
            if call_pos is None:
                unparsed.append(f"call to {proc.lower()}_core(...) or "
                                f"{proc.lower()}(...) in {bridge_path.name}")
            if jax_params is None:
                unparsed.append(f"def {callee} in {pyfile.name}")
            checks.append(("bridge call matches JAX signature", False,
                           "could not locate/parse: " + "; ".join(unparsed)))
        else:
            j_lower = [p.lower() for p in jax_params]
            j_def = {p.lower() for p in jax_defaulted}
            notes = []

            # Positional args bind to the leading JAX params in order:
            # each statically-known name must match the param it lands on.
            if len(call_pos) > len(j_lower):
                notes.append(f"bridge passes {len(call_pos)} positional args "
                             f"but JAX def has only {len(j_lower)} params")
            else:
                misbound = [
                    f"arg '{nm}' binds to param '{j_lower[i]}' (position {i + 1})"
                    for i, nm in enumerate(call_pos)
                    if nm is not None and nm != j_lower[i]
                ]
                if misbound:
                    notes.append("positional misbinding: " + "; ".join(misbound))

            # Keyword args must name real JAX params.
            unknown_kw = sorted(set(call_kw) - set(j_lower))
            if unknown_kw:
                notes.append(f"bridge passes keyword(s) JAX signature lacks: {unknown_kw}")

            # Every non-defaulted JAX param must be bound by the call.
            bound = set(j_lower[: len(call_pos)]) | set(call_kw)
            unbound = sorted(set(j_lower) - bound - j_def)
            if unbound:
                notes.append(f"JAX expects (no default) but bridge never passes: {unbound}")
            unbound_defaulted = sorted((set(j_lower) - bound) & j_def)
            if unbound_defaulted:
                notes.append("defaulted JAX params not passed by bridge (allowed): "
                             f"{unbound_defaulted}")

            bridge_call_ok = not notes or (
                len(notes) == 1 and notes[0].startswith("defaulted JAX params"))
            checks.append(("bridge call matches JAX signature", bridge_call_ok,
                           "; ".join(notes) or None))
    # A missing bridge file is already reported by the 'has bridge metadata' check.

    # Print results to console
    ok = True
    for item in checks:
        name = item[0]
        passed = item[1]
        print(f"{'[OK]' if passed else '[FAIL]'} {name}")
        ok = ok and passed

    # Save results to JSON
    metadata = {
        "jax_file": str(pyfile),
        "has_io": has_io,
        "wrapper_only": wrapper_only,
        "scalar_only": scalar_only,
        "scalar_jax": scalar_jax,
        "jax_required": jax_required,
        "args_count": len(args),
        "writes_count": len(writes_to_args)
    }
    
    output_file = save_lint_results(proc, checks, metadata)
    print(f"\n💾 Lint results saved to: {output_file}")

    return 0 if ok else 2


def generate_summary_report():
    """
    Generate summary report across all linted procedures
    """
    lint_dir = get_config().lint_dir
    if not lint_dir.exists():
        return
    
    lint_files = sorted(lint_dir.glob("*_lint.json"))
    if not lint_files:
        return
    
    all_results = []
    for f in lint_files:
        with open(f, 'r') as fp:
            all_results.append(json.load(fp))
    
    # Calculate aggregate stats
    total_procs = len(all_results)
    total_checks = sum(r["summary"]["total_checks"] for r in all_results)
    total_passed = sum(r["summary"]["passed"] for r in all_results)
    total_failed = sum(r["summary"]["failed"] for r in all_results)
    
    procs_passed = sum(1 for r in all_results if r["summary"]["overall"] == "PASS")
    procs_failed = total_procs - procs_passed
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "procedures": {
            "total": total_procs,
            "passed": procs_passed,
            "failed": procs_failed
        },
        "checks": {
            "total": total_checks,
            "passed": total_passed,
            "failed": total_failed,
            "score": round(100 * total_passed / total_checks, 1) if total_checks > 0 else 0
        },
        "details": [
            {
                "proc": r["proc"],
                "passed": r["summary"]["passed"],
                "failed": r["summary"]["failed"],
                "score": r["summary"]["score"],
                "overall": r["summary"]["overall"]
            }
            for r in all_results
        ]
    }
    
    # Save summary
    summary_file = lint_dir / "_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Summary report saved to: {summary_file}")
    
    # Print summary to console
    print("\n" + "=" * 70)
    print("LINT SUMMARY")
    print("=" * 70)
    print(f"Procedures: {procs_passed}/{total_procs} passed ({procs_failed} failed)")
    print(f"Checks:     {total_passed}/{total_checks} passed ({total_failed} failed)")
    print(f"Score:      {summary['checks']['score']}%")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    from framework_config import init as init_config, add_config_arg

    ap = argparse.ArgumentParser(description="Phase 05.1: lint JAX translations")
    add_config_arg(ap)
    init_config(ap.parse_args().config)

    procs = list_procs_from_jax_dir()
    print(f"Linting {len(procs)} procedures (discovered from {get_config().jax_dir}/)...")
    print("=" * 70)

    failures = 0
    for p in procs:
        print(f"\n=== Linting {p} ===")
        try:
            rc = main(p)
            if rc != 0:
                failures += 1
        except Exception as e:
            failures += 1
            print(f"[ERROR] {p}: {e}")
            import traceback
            traceback.print_exc()

    # Generate summary report
    generate_summary_report()

    if failures:
        raise SystemExit(f"\n❌ Lint finished with {failures} failures.")
    print("\n✅ Lint finished successfully.")