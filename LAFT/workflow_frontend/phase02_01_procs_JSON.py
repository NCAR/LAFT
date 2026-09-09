# -----------------------------------------------------------
# Phase 2_01: Fortran Procedure Packet Extraction
#
# This script reads all Fortran procedure files (*.F90, *.f90) from out/procedures/
# (produced by Phase 1), and for each procedure:
#   - Reads the Fortran source code.
#   - Infers the procedure name from the code.
#   - Extracts OpenMP directives present in the unit.
#   - Generates a "skeleton" version of the code, keeping control flow,
#     declarations, and OpenMP hints, but compressing expressions and call arguments.
#   - NEW: Detects which MODULE variables the procedure uses
#   - Packs this information into a JSON object.
#   - Writes the JSON packet to out/packets/PROCNAME.json, where PROCNAME is the
#     inferred procedure name.
# Prints a summary of how many packets were written.
#
# This script is part of a pipeline to process Fortran code for further analysis
# or translation, organizing each procedure into a structured JSON file with
# metadata and a simplified code skeleton.
# -----------------------------------------------------------

# workflow_frontend/phase02_01_procs_JSON.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import re

# framework_config lives in LAFT/config/; this script now lives in
# workflow_frontend/ (symlinked from LAFT/ into each project) — resolve back
# to LAFT/config to import it. Same convention as workflow_bridge/phase03_make_bridge.py.
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "config"))

from framework_config import get_config

def proc_json_path(dirpath: Path, stem: str) -> Path:
    return dirpath / f"{stem}.json"

def sanitize_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name)
    return name.strip("_") or "unknown"

def unique_json_path(dirpath: Path, stem: str) -> Path:
    p = dirpath / f"{stem}.json"
    if not p.exists():
        return p
    k = 2
    while True:
        p2 = dirpath / f"{stem}__{k}.json"
        if not p2.exists():
            return p2
        k += 1

def openmp_lines(text: str) -> List[str]:
    hits = []
    for i, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().lower().startswith("!$omp"):
            hits.append(f"{i}: {line.rstrip()}")
    return hits


def skeletonize_fortran(code: str) -> str:
    """
    Lightweight skeleton:
    - keep control flow + declarations
    - compress RHS expressions to <EXPR>
    - keep CALL targets, hide args
    - keep OpenMP directives (as parallel hints)
    """
    skel = []
    for line in code.splitlines():
        s = line.rstrip()

        # Keep OpenMP directives verbatim
        if s.lstrip().lower().startswith("!$omp"):
            skel.append(s)
            continue

        # Drop non-OpenMP comments
        if s.lstrip().startswith("!"):
            continue

        low = s.lstrip().lower()

        # Keep common control-flow lines
        if low.startswith((
            "do ", "end do",
            "if ", "end if", "else", "elseif",
            "select ", "case", "end select",
            "where", "end where",
            "forall", "end forall",
        )):
            skel.append(s)
            continue

        # Keep declarations (helps JAX interface mapping later)
        if "::" in s or low.startswith((
            "real", "integer", "logical", "type", "character",
            "parameter", "dimension", "implicit", "use", "module", "contains"
        )):
            skel.append(s)
            continue

        # CALL sites: keep target, hide args
        if low.startswith("call "):
            head = s.split("(")[0].rstrip() if "(" in s else s
            skel.append(f"{head}(<ARGS>)")
            continue

        # Assignment: keep LHS, hide RHS expression
        if "=" in s and "==" not in s and "/=" not in s and "<=" not in s and ">=" not in s:
            lhs = s.split("=", 1)[0].rstrip()
            skel.append(f"{lhs} = <EXPR>")
            continue

        # Keep short lines, hide long ones
        skel.append(s if len(s) < 100 else "<STMT>")

    return "\n".join(skel).strip() + "\n"


def infer_proc_name_from_unit(unit_text: str, fallback: str) -> str:
    """
    Phase 1 already names files well, but we still try to infer the procedure name
    from the first non-empty line to avoid surprises.
    """
    for line in unit_text.splitlines():
        s = line.strip()
        if not s or s.startswith("!"):
            continue
        low = s.lower()
        if low.startswith("subroutine"):
            # subroutine NAME(...)
            rest = s.split(None, 1)[1] if len(s.split(None, 1)) > 1 else ""
            name = rest.split("(", 1)[0].strip()
            return name or fallback
        if low.startswith("function"):
            rest = s.split(None, 1)[1] if len(s.split(None, 1)) > 1 else ""
            name = rest.split("(", 1)[0].strip()
            return name or fallback
        break
    return fallback


# ============================================================================
# NEW: MODULE VARIABLE DETECTION
# ============================================================================

def load_module_metadata() -> Dict[str, Dict[str, Any]]:
    """
    Load module metadata from Phase 01
    
    Returns:
        Dict mapping module_name -> {
            'variables': [list of variable names],
            'procedures': [list of procedure names]
        }
    """
    modules_dir = get_config().modules_dir
    module_info = {}

    if not modules_dir.exists():
        return module_info

    for json_file in modules_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                module_data = json.load(f)

            module_name = module_data.get('module_name', '')
            variables = module_data.get('module_variables', [])
            procedures = module_data.get('procedures', [])

            if module_name:
                # 'variables' preserves module DECLARATION order — the
                # canonical ordering for every generated signature
                # (bridge/wrapper/prompt), so keep it list-ordered.
                module_info[module_name] = {
                    'variables': [v['name'] for v in variables],
                    'parameters': parse_parameter_names(
                        module_data.get('source_file', '')),
                    'procedures': procedures
                }

        except Exception as e:
            print(f"⚠️  Warning: Could not load {json_file}: {e}")

    return module_info


def parse_parameter_names(source_path: str) -> set:
    """Names declared with the Fortran `parameter` attribute in a module.

    Compile-time constants (e.g. `integer, parameter :: isize = 50`) are
    baked into translations, never runtime state, so they must not become
    INOUT module-var parameters. The phase01 module JSON does not (yet)
    record the attribute, so it is re-derived here from the module source.
    Comments are stripped and continuation lines (&) joined before matching.
    """
    if not source_path:
        return set()
    try:
        text = Path(source_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()

    lines = [re.sub(r"!.*", "", ln) for ln in text.splitlines()]
    joined, buf = [], ""
    for ln in lines:
        s = (buf + " " + ln.strip().lstrip("&")).strip() if buf else ln.strip()
        buf = ""
        if s.endswith("&"):
            buf = s[:-1]
            continue
        joined.append(s)
    if buf:
        joined.append(buf)

    params = set()
    decl = re.compile(
        r"(?:integer|real|logical|character|complex|double\s+precision)"
        r"[^:!]*,\s*parameter\b[^:]*::(.*)$", re.I)
    for stmt in joined:
        m = decl.match(stmt)
        if not m:
            continue
        # split the entity list on top-level commas; name is the token
        # before '=' in each entry
        depth, entry, entries = 0, "", []
        for ch in m.group(1):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch == "," and depth == 0:
                entries.append(entry)
                entry = ""
            else:
                entry += ch
        entries.append(entry)
        for e in entries:
            name = e.split("=")[0].strip()
            if re.fullmatch(r"[A-Za-z]\w*", name):
                params.add(name.lower())
    return params


def infer_parent_module(proc_name: str, module_info: Dict[str, Dict[str, Any]]) -> str:
    """
    Infer which module this procedure belongs to
    
    Args:
        proc_name: Name of procedure
        module_info: Dict of module metadata
        
    Returns:
        Parent module name or empty string
    """
    proc_lower = proc_name.lower()
    
    for module_name, info in module_info.items():
        if proc_lower in [p.lower() for p in info['procedures']]:
            return module_name
    
    return ""


def detect_module_var_usage(
    proc_text: str,
    module_name: str,
    module_vars: List[str]
) -> List[str]:
    """
    Detect which module variables a procedure uses
    
    Args:
        proc_text: Procedure source code
        module_name: Parent module name
        module_vars: List of module variable names
        
    Returns:
        List of module variables used by this procedure
    """
    if not module_vars:
        return []
    
    # Convert to lowercase and strip Fortran comments (! to end of line) so
    # words inside comments (e.g. units like "g/kg" matching module var "g")
    # don't count as usage. Also strip declaration lines (containing ::) —
    # a module parameter used only as an array bound in a declaration
    # (e.g. dimension(n_args_r)) is not a runtime dependency: Python arrays
    # carry their own shape, so the wrapper never needs it as a parameter.
    source_lower = re.sub(r"!.*", "", proc_text.lower())
    source_lower = "\n".join(
        line for line in source_lower.splitlines() if "::" not in line
    )
    used_vars = []
    
    for var in module_vars:
        var_lower = var.lower()
        
        # Check if variable appears as a word (not part of another identifier)
        pattern = r'\b' + re.escape(var_lower) + r'\b'
        if re.search(pattern, source_lower):
            used_vars.append(var)
    
    return used_vars


# ============================================================================
# END NEW CODE
# ============================================================================


def main():
    """
    Phase 2.01: Fortran Unit Packet Extraction

    Reads all Fortran unit files (*.F90, *.f90) from out/procedures/ (produced by Phase 1).
    For each unit:
      - Reads the Fortran source code.
      - Infers the procedure name from the code.
      - Extracts OpenMP directives present in the unit.
      - Generates a "skeleton" version of the code, keeping control flow,
        declarations, and OpenMP hints, but compressing expressions and call arguments.
      - NEW: Detects which MODULE variables the procedure uses
      - Packs this information into a JSON object.
      - Writes the JSON packet to out/packets/PROCNAME.json, where PROCNAME is the
        inferred procedure name.
    Prints a summary of how many packets were written.

    This script is part of a pipeline to process Fortran code for further analysis
    or translation, organizing each procedure into a structured JSON file with
    metadata and a simplified code skeleton.
    """

    cfg = get_config()
    procedures_dir = cfg.procedures_dir
    packets_dir = cfg.packets_dir
    packets_dir.mkdir(parents=True, exist_ok=True)

    procedure_files = sorted(list(procedures_dir.glob("*.F90")) + list(procedures_dir.glob("*.f90")))
    if not procedure_files:
        raise SystemExit("No procedure files found in out/procedures/. Run Phase 1 first.")

    # NEW: Load module metadata for variable detection
    print("Loading module metadata...")
    module_info = load_module_metadata()
    
    if module_info:
        print(f"Found {len(module_info)} modules:")
        for mod_name, info in module_info.items():
            var_count = len(info['variables'])
            proc_count = len(info['procedures'])
            print(f"  - {mod_name}: {var_count} variables, {proc_count} procedures")
    else:
        print("No module metadata found (Phase 01 may not have extracted modules)")
    
    print()

    wrote = 0
    for uf in procedure_files:
        unit_text = uf.read_text(encoding="utf-8", errors="replace")
        proc_name = infer_proc_name_from_unit(unit_text, uf.stem)
        safe = sanitize_name(proc_name)

        # NEW: Detect parent module and module variable usage
        parent_module = infer_parent_module(proc_name, module_info)
        module_vars_used = []
        
        if parent_module and parent_module in module_info:
            # Exclude parameter constants; keep declaration order (the
            # canonical signature ordering — see framework_config
            # .packet_module_vars).
            param_names = module_info[parent_module].get('parameters', set())
            module_vars = [v for v in module_info[parent_module]['variables']
                           if v.lower() not in param_names]
            module_vars_used = detect_module_var_usage(unit_text, parent_module, module_vars)
            
            if module_vars_used:
                print(f"✅ {proc_name}: uses MODULE vars {', '.join(module_vars_used)}")
            else:
                print(f"⚪ {proc_name}: no MODULE variables used")
        else:
            print(f"⚪ {proc_name}: no parent module")

        packet: Dict[str, Any] = {
            "proc_name": proc_name,
            "unit_file": str(uf),
            "parent_module": parent_module,  # NEW
            "module_vars_used": module_vars_used,  # NEW
            "fortran_source": unit_text,
            "openmp_directives_in_unit": openmp_lines(unit_text),
            "skeleton": skeletonize_fortran(unit_text),
        }

        out_path = proc_json_path(packets_dir, safe)
        out_path.write_text(json.dumps(packet, indent=2), encoding="utf-8")

        wrote += 1

    print()
    print(f"Wrote {wrote} packets to out/packets/ (from {len(procedure_files)} Phase-1 units).")


if __name__ == "__main__":
    import argparse
    from framework_config import init as init_config, add_config_arg

    ap = argparse.ArgumentParser(description="Phase 02.1: procedure packet extraction")
    add_config_arg(ap)
    init_config(ap.parse_args().config)
    main()