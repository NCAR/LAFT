# -----------------------------------------------------------
# Phase 2.2: Fortran Dependency and Read/Write Analysis
#
# This script analyzes each Fortran procedure file (*.F90, *.f90) in out/procedures/ to extract:
#   - Procedure name and argument list.
#   - Calls to other procedures.
#   - Variables read and written (including which arguments are mutated).
#   - Syntax validity (tree-sitter parse errors).
#   - Writes a dependency packet for each procedure to out/packets/PROCNAME_deps.json.
#   - Writes a summary of all procedures to out/packets/_ALL_deps.json.
#
# This step is essential for dependency tracking and JAX translation, identifying
# argument mutation and inter-procedure calls.
# -----------------------------------------------------------

from __future__ import annotations

import json
import re
from pathlib import Path
from collections import deque
from typing import Iterable, Set, Dict, Any, List, Optional

from tree_sitter import Parser, Node
from tree_sitter_languages import get_language

# framework_config lives in LAFT/config/; this script now lives in
# workflow_frontend/ (symlinked from LAFT/ into each project) — resolve back
# to LAFT/config to import it. Same convention as workflow_bridge/phase03_make_bridge.py.
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "config"))

from framework_config import get_config

FORTRAN = get_language("fortran")


# ---------- generic utilities ----------

def walk(node: Node) -> Iterable[Node]:
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


def node_text(src: bytes, n: Node) -> str:
    return src[n.start_byte:n.end_byte].decode("utf-8", errors="replace")


def is_identifier_node(n: Node) -> bool:
    # This grammar uses identifier/name nodes in different places.
    return n.type in ("identifier", "name")


def identifiers_under(src: bytes, node: Node) -> List[str]:
    """Return identifier strings appearing under node (best-effort)."""
    out: List[str] = []
    for n in walk(node):
        if is_identifier_node(n):
            out.append(node_text(src, n).strip())
    return out


def normalize_ident(name: str) -> str:
    return name.strip().lower()

def sanitize_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name)
    return name.strip("_") or "unknown"


def unique_json_path(dirpath: Path, stem: str, suffix: str) -> Path:
    """
    Avoid collisions: stem+suffix, stem__2+suffix, ...
    """
    p = dirpath / f"{stem}{suffix}"
    if not p.exists():
        return p
    k = 2
    while True:
        p2 = dirpath / f"{stem}__{k}{suffix}"
        if not p2.exists():
            return p2
        k += 1

def parse_fortran(src: bytes):
    parser = Parser()
    parser.set_language(FORTRAN)
    return parser.parse(src)
# ---------- Phase 1 metadata (arg types) ----------

_PHASE1_CACHE: Optional[Dict[str, Any]] = None

def load_phase1_index(out_root: Optional[str] = None) -> Dict[str, Any]:
    global _PHASE1_CACHE
    if _PHASE1_CACHE is not None:
        return _PHASE1_CACHE

    p = Path(out_root) / "phase1_index.json" if out_root else get_config().phase1_index
    if not p.exists():
        _PHASE1_CACHE = {}
        return _PHASE1_CACHE

    _PHASE1_CACHE = json.loads(p.read_text(encoding="utf-8"))
    return _PHASE1_CACHE


def get_character_args_for_proc(proc_name: str, out_root: Optional[str] = None) -> Set[str]:
    """
    Return argument names that are CHARACTER for this proc, using phase1_index.json.
    """
    idx = load_phase1_index(out_root)
    procs = idx.get("procedures", []) if isinstance(idx, dict) else []
    target = proc_name.strip().lower()

    for ent in procs:
        if (ent.get("name") or "").strip().lower() == target:
            meta = ent.get("arg_metadata", {}) or {}
            # meta keys are already lowercased in your phase01
            return {
                k.strip().lower()
                for k, v in meta.items()
                if isinstance(v, dict) and (v.get("dtype") == "character")
            }
    return set()

# ---------- extraction: procedure signature ----------

def get_proc_name_and_args(src: bytes, root: Node) -> tuple[str, List[str]]:
    """
    Best-effort: find first subroutine/function statement and extract its name + parameter identifiers.
    """
    stmt = None
    stmt_kind = None
    for n in walk(root):
        if n.type == "subroutine_statement":
            stmt = n
            stmt_kind = "subroutine"
            break
        if n.type == "function_statement":
            stmt = n
            stmt_kind = "function"
            break

    if stmt is None:
        return "<unknown>", []

    name_node = stmt.child_by_field_name("name")
    params_node = stmt.child_by_field_name("parameters")

    name = node_text(src, name_node).strip() if name_node else "<unknown>"
    args: List[str] = []
    if params_node:
        # parameters usually contains identifiers
        args = [normalize_ident(x) for x in identifiers_under(src, params_node)]
        # de-dup while preserving order
        seen = set()
        args = [a for a in args if not (a in seen or seen.add(a))]
    return name, args


# ---------- extraction: calls ----------

def extract_calls(src: bytes, root: Node) -> Set[str]:
    """
    Capture callees from call statements.
    The tree-sitter Fortran grammar uses node type 'subroutine_call'
    (not 'call_statement') and field name 'subroutine' (not 'name').
    """
    calls: Set[str] = set()

    for n in walk(root):
        if n.type == "subroutine_call":
            name_node = n.child_by_field_name("subroutine")
            if name_node:
                calls.add(normalize_ident(node_text(src, name_node)))
            else:
                # fallback: first identifier under the subroutine_call
                ids = identifiers_under(src, n)
                if ids:
                    calls.add(normalize_ident(ids[0]))

        elif n.type == "call_expression":
            # Fortran function references (x = foo(y)) parse as call_expression.
            # Array indexing does too, but the later filter to known internal
            # procedure names drops array names and intrinsics.
            ids = [c for c in n.children if c.type == "identifier"]
            if ids:
                calls.add(normalize_ident(node_text(src, ids[0])))

    return calls


# ---------- extraction: writes & reads ----------

def first_identifier_under(src: bytes, node: Node) -> Optional[str]:
    for n in walk(node):
        if is_identifier_node(n):
            return normalize_ident(node_text(src, n))
    return None


def lhs_written_idents(src: bytes, assign_node: Node) -> Set[str]:
    """
    Only the base target is written.
    Example: a(n,i)=...  -> {"a"} (NOT {"a","n","i"}).
    """
    for field in ("left", "lhs", "variable"):
        lhs = assign_node.child_by_field_name(field)
        if lhs:
            base = first_identifier_under(src, lhs)
            return {base} if base else set()

    # Conservative fallback: first identifier token before '='
    txt = node_text(src, assign_node)
    if "=" in txt:
        left_txt = txt.split("=", 1)[0]
        toks = re.findall(r"[A-Za-z_]\w*", left_txt)
        if toks:
            return {normalize_ident(toks[0])}
    return set()



def rhs_read_idents(src: bytes, assign_node: Node) -> Set[str]:
    """
    Return identifiers read on RHS of an assignment.
    We try to find an RHS field; fallback: identifiers after '='.
    """
    for field in ("right", "rhs", "value", "expression"):
        rhs = assign_node.child_by_field_name(field)
        if rhs:
            return {normalize_ident(x) for x in identifiers_under(src, rhs)}

    txt = node_text(src, assign_node)
    if "=" in txt:
        right_txt = txt.split("=", 1)[1]
        import re
        toks = re.findall(r"[A-Za-z_]\w*", right_txt)
        return {normalize_ident(t) for t in toks}

    return set()


def extract_reads_writes(src: bytes, root: Node) -> tuple[Set[str], Set[str]]:
    """
    Best-effort read/write sets.
    Writes: LHS idents of assignments.
    Reads: RHS idents of assignments + idents in conditions/loop bounds (approx).
    """
    writes: Set[str] = set()
    reads: Set[str] = set()

    for n in walk(root):
        # Assignment node type can vary. We handle the common ones and add a fallback.
        if n.type in ("assignment_statement", "assignment"):
            w = lhs_written_idents(src, n)
            r = rhs_read_idents(src, n)
            writes |= w
            reads |= r

        # Pull reads from if conditions, do loop bounds, where, etc.
        if n.type in ("if_statement", "do_statement", "where_statement", "elseif_clause"):
            reads |= {normalize_ident(x) for x in identifiers_under(src, n)}

    # Reads should not include writes-by-definition only (we keep both sets; later you can subtract if you want)
    return reads, writes

# ---------- effects: I/O detection ----------

# ---------- effects: I/O detection (external vs internal errmsg writes) ----------

_EXTERNAL_IO_PATTERNS = [
    r'^\s*print\b',
    r'^\s*read\s*\(',
    r'^\s*open\s*\(',
    r'^\s*close\s*\(',
    r'^\s*inquire\s*\(',
    r'^\s*rewind\b',
    r'^\s*backspace\b',
    r'^\s*flush\b',
    r'^\s*stop\b',
    r'^\s*pause\b',
]

def _first_write_unit(code_lower: str) -> Optional[str]:
    """
    Best-effort parse of WRITE(<unit>, ...) first argument.
    Returns token string (lowered) or None.
    """
    # remove "write" prefix
    m = re.search(r"^\s*write\s*\(\s*([^,\)]+)", code_lower)
    if not m:
        return None
    token = m.group(1).strip()
    # strip keywords like "unit=" if present
    token = token.replace("unit=", "").strip()
    # strip quotes/spaces
    token = token.strip()
    return token or None

def detect_io_lines_split(
    fortran_text: str,
    character_args: Set[str],
    cap: int = 20
) -> Dict[str, List[str]]:
    """
    Split I/O hits into:
      - external: real file/stdout I/O (disqualifies pure-jit core)
      - internal: WRITE(errmsg,...) style formatting into a character buffer
    """
    external: List[str] = []
    internal: List[str] = []

    for i, raw in enumerate(fortran_text.splitlines(), start=1):
        if not raw.strip():
            continue

        stripped = raw.lstrip()
        if stripped.startswith("!"):
            continue

        code = raw.split("!")[0].strip()
        code_lower = code.lower()

        # External I/O keywords
        if any(re.search(pat, code_lower) for pat in _EXTERNAL_IO_PATTERNS):
            external.append(f"{i}: {raw.rstrip()}")
            if len(external) + len(internal) >= cap:
                break
            continue

        # WRITE(...) needs special handling
        if re.search(r'^\s*write\s*\(', code_lower):
            unit_tok = _first_write_unit(code_lower)

            # unknown parse -> conservative external
            if unit_tok is None:
                external.append(f"{i}: {raw.rstrip()}")
                if len(external) + len(internal) >= cap:
                    break
                continue

            # WRITE(*) => external
            if unit_tok == "*":
                external.append(f"{i}: {raw.rstrip()}")
                if len(external) + len(internal) >= cap:
                    break
                continue

            # numeric unit => external
            if re.fullmatch(r"\d+", unit_tok or ""):
                external.append(f"{i}: {raw.rstrip()}")
                if len(external) + len(internal) >= cap:
                    break
                continue

            # identifier unit: internal if it's a CHARACTER arg (errmsg) / common names
            unit_ident = re.sub(r"[^a-z0-9_]", "", unit_tok)
            if unit_ident in character_args or unit_ident in {"errmsg", "err_msg", "message", "msg"}:
                internal.append(f"{i}: {raw.rstrip()}")
            else:
                external.append(f"{i}: {raw.rstrip()}")

            if len(external) + len(internal) >= cap:
                break

    return {"external": external, "internal": internal}


def _is_dim_name(name: str) -> bool:
    # Single source: [heuristics].dim_names in config/project.toml
    return get_config().is_dim_name(name)

def _is_scalar_output_flag(name: str) -> bool:
    # Single source: [heuristics].scalar_flag_names in config/project.toml
    return get_config().is_scalar_output_flag(name)

def _is_scalar_only(fortran_text: str) -> bool:
    """
    True if the procedure has no array arguments.
    Arrays are detected by ':: varname(' or 'dimension(' in declarations.
    Scalar-only procedures translate to plain Python (no JAX/JIT).
    """
    array_patterns = [
        r"::\s*\w+\s*\(",
        r"\bdimension\s*\(",
    ]
    for pat in array_patterns:
        if re.search(pat, fortran_text, re.IGNORECASE):
            return False
    return True


def compute_scalar_out_args(args: List[str], writes_to_args: List[str]) -> List[str]:
    writes = {w.lower() for w in (writes_to_args or [])}
    out: List[str] = []
    for a in (args or []):
        al = a.lower()
        if al in writes and (not _is_dim_name(a)) and _is_scalar_output_flag(a):
            out.append(a)
    return out

# ---------- main driver ----------

def analyze_unit(path: Path) -> Dict[str, Any]:
    src = path.read_bytes()
    tree = parse_fortran(src)
    root = tree.root_node

    proc_name, args = get_proc_name_and_args(src, root)
    calls = sorted(extract_calls(src, root))
    reads, writes = extract_reads_writes(src, root)

    # Identify which args are mutated (super important for JAX translation)
    args_set = set(args)
    writes_to_args = sorted(args_set.intersection(writes))

    scalar_out_args = compute_scalar_out_args(args, writes_to_args)


    # --- NEW: effects metadata (I/O detection) ---
    unit_text = src.decode("utf-8", errors="replace")

    # NEW: use Phase 1 arg type metadata to distinguish internal errmsg writes
    character_args = get_character_args_for_proc(proc_name)

    io_split = detect_io_lines_split(unit_text, character_args=character_args, cap=20)
    io_hits_external = io_split["external"]
    io_hits_internal = io_split["internal"]

    has_external_io = len(io_hits_external) > 0
    has_internal_error_write = len(io_hits_internal) > 0

    out: Dict[str, Any] = {
        "proc_name": proc_name,
        "scalar_only": _is_scalar_only(unit_text),
        "scalar_out_args": scalar_out_args,
        "args": args,
        "calls": calls,
        "reads": sorted(reads),
        "writes": sorted(writes),
        "writes_to_args": writes_to_args,
        "syntax_ok": ("ERROR" not in {n.type for n in walk(root)}),
        "unit_file": str(path),

        "effects": {
            # IMPORTANT: redefine has_io to mean EXTERNAL I/O ONLY
            "has_io": has_external_io,

            # NEW: keep detail flags so prompts/translation can handle Option A
            "has_external_io": has_external_io,
            "has_internal_error_write": has_internal_error_write,

            # NEW: show categorized hits
            "io_hits_external": io_hits_external,
            "io_hits_internal": io_hits_internal,
        },
    }
    return out


def main():
    """
    Phase 2.2: Fortran Dependency and Read/Write Analysis

    Analyzes each Fortran procedure file (*.F90, *.f90) in out/procedures/ to extract:
      - Procedure name and argument list.
      - Calls to other procedures.
      - Variables read and written (including which arguments are mutated).
      - Syntax validity (tree-sitter parse errors).
      - Writes a dependency packet for each procedure to out/packets/PROCNAME_deps.json.
      - Writes a summary of all procedures to out/packets/_ALL_deps.json.

    This step is essential for dependency tracking and JAX translation, identifying
    argument mutation and inter-procedure calls.
    """

    cfg = get_config()
    units_dir = cfg.procedures_dir
    packets_dir = cfg.packets_dir
    packets_dir.mkdir(parents=True, exist_ok=True)

    unit_files = sorted(units_dir.glob("*.F90")) + sorted(units_dir.glob("*.f90"))
    if not unit_files:
        raise SystemExit("No procedure files found in out/procedures/. Run Phase 1 extraction first.")

    summary: List[Dict[str, Any]] = []
    for uf in unit_files:
        dep = analyze_unit(uf)
        summary.append(dep)

    # Filter each proc's calls to only names of known internal procedures.
    # This drops Fortran intrinsics (cpu_time, abs, etc.) and external library calls.
    known_procs = {dep["proc_name"].lower() for dep in summary}
    for dep in summary:
        dep["calls"] = sorted(c for c in dep["calls"] if c.lower() in known_procs)

    # INVARIANT (see workflow_translator/prompt_policies/orchestration_boundary_rules.md):
    # every non-scalar procedure gets the full JAX contract — including the
    # scheme's top-level main routine and model-facing wrappers. Position in
    # the call graph grants no exemption; there is NO host-side classification
    # for procedures with grid/state arrays. Host-side layers are limited to
    # scalar-only procs (config readers, LUT file loaders, init sequencing),
    # which are excluded from the seed automatically by scalar_only and only
    # become jax_required if a jitted proc transitively calls them.
    #
    # Compute jax_required: True for all non-scalar procedures that run inside
    # @jax.jit, and for any procedure transitively reachable as a callee from
    # one of those; anything a jitted core calls must be JAX-compatible even
    # if it has no arrays of its own.
    jax_req: Set[str] = {dep["proc_name"].lower() for dep in summary
                         if not dep["scalar_only"]}
    queue: "deque[str]" = deque(list(jax_req))
    while queue:
        node = queue.popleft()
        node_dep = next((d for d in summary if d["proc_name"].lower() == node), None)
        if node_dep is None:
            continue
        for callee in node_dep["calls"]:  # already lowercase, already filtered
            if callee not in jax_req:
                jax_req.add(callee)
                queue.append(callee)
    for dep in summary:
        dep["jax_required"] = dep["proc_name"].lower() in jax_req

    for dep in summary:
        # write per-proc packet (Phase 2.3 artifact) — overwrite deterministically
        # (unique_json_path created stale <proc>__2_deps.json variants on re-runs,
        # leaving downstream merges reading first-run data)
        safe = sanitize_name(dep["proc_name"])
        out_path = packets_dir / f"{safe}_deps.json"
        out_path.write_text(json.dumps(dep, indent=2), encoding="utf-8")

    (packets_dir / "_ALL_deps.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Analyzed {len(unit_files)} procedures.")
    print(f"Wrote packets to {packets_dir}/ (including _ALL_deps.json).")


if __name__ == "__main__":
    import argparse
    from framework_config import init as init_config, add_config_arg

    ap = argparse.ArgumentParser(description="Phase 02.2: dependency / read-write analysis")
    add_config_arg(ap)
    init_config(ap.parse_args().config)
    main()
