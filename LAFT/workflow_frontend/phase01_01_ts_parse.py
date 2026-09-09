# workflow_frontend/phase01_01_ts_parse.py
# -----------------------------------------------------------
# Phase 1: Enhanced Fortran Procedure Extraction with Tree-sitter
#
# NEW: Extracts type information (rank, dtype, dimensions) using tree-sitter AST
#
# Enhancements over original phase01_01_ts_parse.py:
#   - Adds ArgMetadata dataclass for type information
#   - Extracts variable declarations using tree-sitter
#   - Stores type metadata in out\phase1_index.json
#   - Bridge generator can read types directly from index
#   - MODULE metadata extraction with USE statements
# -----------------------------------------------------------

from __future__ import annotations

import json
import re
import shutil 
from dataclasses import dataclass, asdict
from pathlib import Path

# framework_config lives in LAFT/config/; this script now lives in
# workflow_frontend/ (symlinked from LAFT/ into each project) — resolve back
# to LAFT/config so the __main__ block can import it. Same convention as
# workflow_bridge/phase03_make_bridge.py.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from typing import Iterable, Optional, List, Tuple, Dict

from tree_sitter import Parser, Node
from tree_sitter_languages import get_language

FORTRAN = get_language("fortran")


# ---------------- utilities ----------------
def _strip_fortran_comment(line: str) -> str:
    # Remove anything after '!' (Fortran comment), but keep string literals simple-case
    # (good enough for declarations)
    return line.split("!", 1)[0]

def _join_fortran_continuations(lines: list[str]) -> str:
    """
    Join Fortran continuation lines using '&'.
    This is a simple join that works well for declaration statements.
    """
    out = []
    buf = ""
    for raw in lines:
        line = _strip_fortran_comment(raw).rstrip()
        if not line.strip():
            continue

        if buf:
            # continuation if previous ended with '&' or current starts with '&'
            if buf.rstrip().endswith("&"):
                buf = buf.rstrip()[:-1] + " " + line.lstrip()
            elif line.lstrip().startswith("&"):
                buf = buf + " " + line.lstrip()[1:].lstrip()
            else:
                out.append(buf)
                buf = line
        else:
            buf = line

    if buf:
        out.append(buf)

    return "\n".join(out)

def _infer_dims_from_proc_text(proc_text: str, param_names_lower: list[str]) -> dict[str, list[str]]:
    """
    Fallback: infer dims from Fortran declarations in the raw procedure text.
    Returns dict: name -> dims list (e.g. [":", ":"] or ["1:ncol", "1:nz"])
    """
    joined = _join_fortran_continuations(proc_text.splitlines()).lower()

    dims_map: dict[str, list[str]] = {}
    for name in param_names_lower:
        # Look for a declaration containing :: ... name( ... )
        # This avoids grabbing call sites.
        # Example: real(kind_phys), intent(in) :: cpair(:,:)
        pat = rf"::[^\n]*\b{re.escape(name)}\s*\(([^)]*)\)"
        m = re.search(pat, joined)
        if m:
            inside = m.group(1)
            parts = [p.strip() for p in inside.split(",") if p.strip()]
            dims_map[name] = parts
    return dims_map

def _parse_dims_from_decl_text(decl_text: str, var_name: str) -> list[str]:
    # Look for "var_name(...)" first
    m = re.search(rf"\b{re.escape(var_name)}\s*\(([^)]*)\)", decl_text)
    if m:
        inside = m.group(1)
        return [p.strip() for p in inside.split(",") if p.strip()]

    # Otherwise look for a DIMENSION(...) attribute that applies to all vars
    m = re.search(r"\bdimension\s*\(([^)]*)\)", decl_text)
    if m:
        inside = m.group(1)
        return [p.strip() for p in inside.split(",") if p.strip()]

    return []

def walk(node: Node) -> Iterable[Node]:
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


def node_text(src: bytes, node: Node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def has_error(root: Node) -> bool:
    return any(n.type == "ERROR" for n in walk(root))


def extract_openmp_directives(src: bytes) -> list[str]:
    lines = src.decode("utf-8", errors="replace").splitlines()
    hits = []
    for i, line in enumerate(lines, start=1):
        if line.lstrip().lower().startswith("!$omp"):
            hits.append(f"{i}: {line.rstrip()}")
    return hits


def sanitize_name(name: str) -> str:
    safe = []
    for ch in name:
        if ch.isalnum() or ch in ("_", "-"):
            safe.append(ch)
        else:
            safe.append("_")
    out = "".join(safe).strip("_")
    return out or "unknown"


def unique_path(dirpath: Path, stem: str, suffix: str = ".F90") -> Path:
    p = dirpath / f"{stem}{suffix}"
    if not p.exists():
        return p
    k = 2
    while True:
        p2 = dirpath / f"{stem}__{k}{suffix}"
        if not p2.exists():
            return p2
        k += 1


# ---------------- NEW: Type extraction ----------------

@dataclass
class ArgMetadata:
    """Type metadata for a procedure argument"""
    name: str
    dtype: str              # 'real', 'integer', 'logical', 'character'
    rank: int               # 0=scalar, 1=1D, 2=2D, etc.
    dimensions: list[str]   # e.g., ['n', 'n'] for a(n,n)
    intent: Optional[str]   # 'in', 'out', 'inout', or None


def extract_type_from_node(type_node: Node, src: bytes) -> str:
    """Extract type string from a type specification node"""
    text = node_text(src, type_node).strip().lower()
    
    # Handle type specifiers like "real(8)", "integer", etc.
    if text.startswith('real'):
        return 'real'
    elif text.startswith('integer'):
        return 'integer'
    elif text.startswith('logical'):
        return 'logical'
    elif text.startswith('character'):
        return 'character'
    else:
        return 'real'  # Default


def extract_dimensions_from_node(dim_node: Node, src: bytes) -> list[str]:
    text = node_text(src, dim_node).strip()
    if not text:
        return []

    if "(" in text and ")" in text:
        inside = text[text.find("(") + 1 : text.rfind(")")]
    else:
        inside = text

    parts = [p.strip() for p in inside.split(",") if p.strip()]
    return parts



def extract_variable_declarations(proc_node: Node, src: bytes, param_names: list[str]) -> Dict[str, ArgMetadata]:
    """
    Extract variable type declarations using tree-sitter.
    Only extracts info for variables that are procedure parameters.
    
    Args:
        proc_node: The subroutine/function node
        src: Source bytes
        param_names: List of parameter names from procedure signature
        
    Returns:
        Dict mapping parameter name to ArgMetadata
    """
    arg_metadata = {}
    param_names_lower = [p.lower() for p in param_names]
    
    # Find all variable declarations in the procedure
    for node in walk(proc_node):
        if node.type == 'variable_declaration':
            # Try to extract type, name, dimensions, intent
            dtype = None
            var_names = []
            dimensions = []
            intent = None
            
            # Walk through declaration components
            for child in node.children:
                if child.type in ['intrinsic_type', 'type_spec']:
                    dtype = extract_type_from_node(child, src)
                
                elif child.type == 'identifier':
                    var_name = node_text(src, child).strip().lower()
                    if var_name:
                        var_names.append(var_name)
                
                elif child.type == 'dimension_specification':
                    dimensions = extract_dimensions_from_node(child, src)
                
                elif 'intent' in node_text(src, child).lower():
                    intent_text = node_text(src, child).lower()
                    if 'inout' in intent_text:
                        intent = 'inout'
                    elif 'out' in intent_text:
                        intent = 'out'
                    elif 'in' in intent_text:
                        intent = 'in'
            
            # Store metadata for parameters only (dims may be per-variable)
            decl_text = node_text(src, node).strip().lower()

            for var_name in var_names:
                if var_name in param_names_lower:
                    dims = dimensions

                    # Fallback: handle assumed-shape like (:,:) and dims attached to declarator
                    if not dims:
                        dims = _parse_dims_from_decl_text(decl_text, var_name)

                    rank = len(dims) if dims else 0

                    arg_metadata[var_name] = ArgMetadata(
                        name=var_name,
                        dtype=dtype or "real",
                        rank=rank,
                        dimensions=dims or [],
                        intent=intent,
                    )
    
    # For parameters not found in declarations, create default entries
    for param_name in param_names_lower:
        if param_name not in arg_metadata:
            arg_metadata[param_name] = ArgMetadata(
                name=param_name,
                dtype='real',
                rank=0,
                dimensions=[],
                intent=None
            )
    # --- FINAL fallback: scan raw procedure text for dims like cpair(:,:) ---
    proc_text = node_text(src, proc_node)
    inferred = _infer_dims_from_proc_text(proc_text, param_names_lower)

    for pname in param_names_lower:
        meta = arg_metadata.get(pname)
        if meta is None:
            continue
        # If rank still looks scalar but we can infer dims, upgrade it
        if (meta.rank == 0 or not meta.dimensions) and (pname in inferred):
            meta.dimensions = inferred[pname]
            meta.rank = len(meta.dimensions)
            arg_metadata[pname] = meta

    return arg_metadata


# ---------------- signature extraction ----------------

def get_proc_name_and_args(src: bytes, proc_node: Node) -> tuple[Optional[str], list[str]]:
    """Return (name, args) for a procedure node"""
    if proc_node.type == "subroutine":
        stmt_type = "subroutine_statement"
    elif proc_node.type == "function":
        stmt_type = "function_statement"
    else:
        return None, []

    stmt = next((c for c in proc_node.children if c.type == stmt_type), None)
    if stmt is None:
        return None, []

    name_node = stmt.child_by_field_name("name")
    if name_node is None:
        return None, []

    name = node_text(src, name_node).strip()
    if not name or name.lower() in ("subroutine", "function"):
        return None, []

    params_node = stmt.child_by_field_name("parameters")
    args: list[str] = []
    if params_node:
        for n in walk(params_node):
            if n.type in ("identifier", "name"):
                args.append(node_text(src, n).strip())

        seen = set()
        args = [a for a in args if a and not (a.lower() in seen or seen.add(a.lower()))]

    return name, args


def _block_name_from_stmt(src: bytes, block_node: Node, stmt_type: str) -> Optional[str]:
    """Get name from a block statement node - ENHANCED"""
    
    # Try original method
    stmt = next((c for c in block_node.children if c.type == stmt_type), None)
    if stmt:
        name_node = stmt.child_by_field_name("name")
        if name_node:
            name = node_text(src, name_node).strip()
            if name:
                return name
    
    # Fallback 1: Look for identifier in statement
    if stmt:
        for child in stmt.children:
            if child.type in ("identifier", "name"):
                name = node_text(src, child).strip()
                if name and name.lower() not in ("module", "program", "subroutine", "function"):
                    return name
    
    # Fallback 2: Parse text directly
    if stmt:
        text = node_text(src, stmt).strip()
        # Pattern: "MODULE name" or "PROGRAM name"
        match = re.search(r'(?:module|program)\s+([a-z_][a-z0-9_]*)', text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Fallback 3: Look in entire block text
    text = node_text(src, block_node).strip()
    lines = text.splitlines()
    if lines:
        first_line = lines[0].strip().lower()
        # Pattern: "module name"
        match = re.search(r'module\s+([a-z_][a-z0-9_]*)', first_line)
        if match:
            return match.group(1)
    
    return None

def is_real_procedure_slice(text: str) -> bool:
    """Defensive check: avoid writing fragments"""
    t = text.strip()
    if not t:
        return False
    lines = [ln.strip().lower() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return False

    first = lines[0]
    is_procedure = (
        first.startswith("subroutine")
        or first.startswith("function")
        or re.match(r"(real|integer|logical|character)\s+function\b", first)
    )
    if not is_procedure:
        return False

    joined = "\n".join(lines)
    if ("end subroutine" not in joined) and ("end function" not in joined):
        if not any(ln == "end" for ln in lines):
            return False

    return True


def is_real_program_slice(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    lines = [ln.strip().lower() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return False
    if not lines[0].startswith("program"):
        return False
    joined = "\n".join(lines)
    if ("end program" not in joined) and (not any(ln == "end" for ln in lines)):
        return False
    return True


def is_real_module_slice(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    lines = [ln.strip().lower() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return False
    if not lines[0].startswith("module"):
        return False
    if lines[0].startswith("module procedure"):
        return False
    joined = "\n".join(lines)
    if ("end module" not in joined) and (not any(ln == "end" for ln in lines)):
        return False
    return True

# ---------------- NEW: Module metadata extraction ----------------

def extract_use_statements(module_node: Node, src: bytes) -> list[str]:
    """
    Extract USE statement module names
    
    Example Fortran:
        USE ccpp_kinds, ONLY: kind_phys
        USE netcdf
        
    Returns: ['ccpp_kinds', 'netcdf']
    """
    uses = []
    
    for node in walk(module_node):
        if node.type == "use_statement":
            # Try to get module name from USE statement
            name_node = node.child_by_field_name("name")
            if name_node:
                module_name = node_text(src, name_node).strip()
                if module_name and module_name.lower() not in uses:
                    uses.append(module_name.lower())
            else:
                # Fallback: look for identifier after USE keyword
                text = node_text(src, node).lower()
                # Pattern: "use module_name"
                match = re.search(r'use\s+([a-z_][a-z0-9_]*)', text)
                if match:
                    module_name = match.group(1)
                    if module_name not in uses:
                        uses.append(module_name)
    
    return uses


def extract_module_variables(module_node: Node, src: bytes) -> list[dict]:
    """
    Extract module-level variable declarations
    
    Example Fortran:
        REAL(kind_phys) :: lv, pref, rhoqr
        INTEGER :: some_flag
        
    Returns: [
        {'name': 'lv', 'type': 'real', 'kind': 'kind_phys'},
        {'name': 'pref', 'type': 'real', 'kind': 'kind_phys'},
        ...
    ]
    """
    variables = []
    
    # Look for variable declarations at module level
    # (not inside CONTAINS section)
    contains_found = False
    
    for child in module_node.children:
        # Stop at CONTAINS - anything after is procedures
        if child.type == "contains_statement":
            contains_found = True
            break
        
        # Look for variable declarations before CONTAINS
        if child.type == "variable_declaration":
            # Extract type
            dtype = None
            var_entries = []  # (name, dims_or_None, init_text_or_None)
            kind = None
            decl_dims = None  # from a dimension(...) attribute — applies to all entities

            for decl_child in child.children:
                if decl_child.type in ['intrinsic_type', 'type_spec']:
                    type_text = node_text(src, decl_child).lower()
                    if 'real' in type_text:
                        dtype = 'real'
                    elif 'integer' in type_text:
                        dtype = 'integer'
                    elif 'logical' in type_text:
                        dtype = 'logical'
                    elif 'character' in type_text:
                        dtype = 'character'

                    # Try to extract kind
                    kind_match = re.search(r'\(\s*([a-z_][a-z0-9_]*)\s*\)', type_text)
                    if kind_match:
                        kind = kind_match.group(1)

                elif decl_child.type == 'type_qualifier':
                    # dimension(d1,d2,...) attribute — record the dim tokens
                    # (parameter names or literals) for every entity declared here
                    q_text = node_text(src, decl_child).lower()
                    if q_text.startswith('dimension'):
                        for q in decl_child.children:
                            if q.type == 'argument_list':
                                decl_dims = [
                                    node_text(src, a).strip().lower()
                                    for a in q.children
                                    if a.type not in ('(', ')', ',')
                                ]

                elif decl_child.type == 'identifier':
                    var_name = node_text(src, decl_child).strip()
                    if var_name:
                        var_entries.append((var_name.lower(), None, None))

                elif decl_child.type == 'call_expression':
                    # entity-level dims, e.g. `real :: x(5,3)`
                    ids = [c for c in decl_child.children if c.type == 'identifier']
                    dims = None
                    for c in decl_child.children:
                        if c.type == 'argument_list':
                            dims = [
                                node_text(src, a).strip().lower()
                                for a in c.children
                                if a.type not in ('(', ')', ',')
                            ]
                    if ids:
                        var_entries.append((node_text(src, ids[0]).strip().lower(), dims, None))

                elif decl_child.type == 'assignment_statement':
                    # Initialized declaration, e.g. `integer, parameter :: densize = 5`.
                    # The declared name is the first direct identifier child (LHS);
                    # the initialization text (RHS) is kept verbatim so dummy
                    # generators can resolve real parameter values.
                    ids = [c for c in decl_child.children if c.type == 'identifier']
                    if ids:
                        var_name = node_text(src, ids[0]).strip()
                        if var_name:
                            full = node_text(src, decl_child)
                            init = full.split("=", 1)[1].strip() if "=" in full else None
                            var_entries.append((var_name.lower(), None, init))

            # Add each variable
            for var_name, entity_dims, init in var_entries:
                var_info = {
                    'name': var_name,
                    'type': dtype or 'real'
                }
                if kind:
                    var_info['kind'] = kind
                dims = entity_dims or decl_dims
                if dims:
                    var_info['dims'] = dims
                    var_info['rank'] = len(dims)
                if init is not None:
                    var_info['init'] = init
                variables.append(var_info)

    return variables


def extract_public_private_entities(module_node: Node, src: bytes) -> tuple[list[str], list[str]]:
    """
    Extract PUBLIC and PRIVATE declarations
    
    Example Fortran:
        PUBLIC :: kessler_init, kessler_run
        PRIVATE :: lv, pref, rhoqr
        
    Returns: (public_list, private_list)
    """
    public = []
    private = []
    
    for node in walk(module_node):
        if node.type == "access_statement":
            text = node_text(src, node).lower()
            
            # Extract identifiers from the statement
            identifiers = []
            for child in walk(node):
                if child.type in ('identifier', 'name'):
                    ident = node_text(src, child).strip().lower()
                    if ident and ident not in ('public', 'private'):
                        identifiers.append(ident)
            
            if 'public' in text:
                public.extend(identifiers)
            elif 'private' in text:
                private.extend(identifiers)
    
    return public, private


# ---------------- extraction ----------------

@dataclass
class UnitIndexEntry:
    kind: str               # "subroutine" | "function"
    name: str
    args: list[str]
    arg_metadata: dict      # NEW: Dict of ArgMetadata (will be serialized)
    source_file: str
    unit_file: str
    start_byte: int
    end_byte: int


@dataclass
class BlockIndexEntry:
    kind: str               # "program" | "module"
    name: str
    source_file: str
    block_file: str
    start_byte: int
    end_byte: int


def extract_procedures(src: bytes, root: Node) -> tuple[list[UnitIndexEntry], list[str]]:
    """
    Extract subroutines/functions WITH type metadata.
    """
    candidates: list[Node] = [n for n in walk(root) if n.type in ("subroutine", "function")]

    # Prefer outer-most procedures
    candidates_sorted = sorted(candidates, key=lambda n: (n.start_byte, -(n.end_byte - n.start_byte)))
    filtered: list[Node] = []
    occupied: list[tuple[int, int]] = []
    for n in candidates_sorted:
        s, e = n.start_byte, n.end_byte
        if any(s >= os and e <= oe for (os, oe) in occupied):
            continue
        filtered.append(n)
        occupied.append((s, e))

    warnings: list[str] = []
    entries: list[UnitIndexEntry] = []

    for n in filtered:
        kind = n.type
        name, args = get_proc_name_and_args(src, n)
        if not name:
            warnings.append(f"Skipped {kind} at bytes [{n.start_byte},{n.end_byte}] (missing name)")
            continue

        body = node_text(src, n)
        if not is_real_procedure_slice(body):
            first_line = body.splitlines()[0] if body.splitlines() else ""
            warnings.append(f"Skipped {kind} '{name}' (fragment? first line: {first_line})")
            continue

        # NEW: Extract type metadata using tree-sitter
        arg_metadata_dict = extract_variable_declarations(n, src, args)
        
        # Convert to serializable dict
        arg_metadata_serialized = {
            arg_name: asdict(metadata) 
            for arg_name, metadata in arg_metadata_dict.items()
        }

        entries.append(
            UnitIndexEntry(
                kind=kind,
                name=name,
                args=args,
                arg_metadata=arg_metadata_serialized,  # NEW
                source_file="",
                unit_file="",
                start_byte=n.start_byte,
                end_byte=n.end_byte,
            )
        )

    return entries, warnings


def extract_program_blocks(src: bytes, root: Node) -> tuple[list[BlockIndexEntry], list[str]]:
    """Extract PROGRAM blocks"""
    candidates: list[Node] = [n for n in walk(root) if n.type == "program"]

    candidates_sorted = sorted(candidates, key=lambda n: (n.start_byte, -(n.end_byte - n.start_byte)))
    filtered: list[Node] = []
    occupied: list[tuple[int, int]] = []
    for n in candidates_sorted:
        s, e = n.start_byte, n.end_byte
        if any(s >= os and e <= oe for (os, oe) in occupied):
            continue
        filtered.append(n)
        occupied.append((s, e))

    warnings: list[str] = []
    entries: list[BlockIndexEntry] = []

    for n in filtered:
        name = _block_name_from_stmt(src, n, "program_statement") or "main"
        body = node_text(src, n)
        if not is_real_program_slice(body):
            first_line = body.splitlines()[0] if body.splitlines() else ""
            warnings.append(f"Skipped program '{name}' (fragment? first line: {first_line})")
            continue

        entries.append(
            BlockIndexEntry(
                kind="program",
                name=name,
                source_file="",
                block_file="",
                start_byte=n.start_byte,
                end_byte=n.end_byte,
            )
        )

    return entries, warnings


def extract_module_blocks(src: bytes, root: Node) -> tuple[list[BlockIndexEntry], list[str]]:
    """Extract MODULE blocks"""
    candidates: list[Node] = [n for n in walk(root) if n.type == "module"]

    candidates_sorted = sorted(candidates, key=lambda n: (n.start_byte, -(n.end_byte - n.start_byte)))
    filtered: list[Node] = []
    occupied: list[tuple[int, int]] = []
    for n in candidates_sorted:
        s, e = n.start_byte, n.end_byte
        if any(s >= os and e <= oe for (os, oe) in occupied):
            continue
        filtered.append(n)
        occupied.append((s, e))

    warnings: list[str] = []
    entries: list[BlockIndexEntry] = []

    for n in filtered:
        name = _block_name_from_stmt(src, n, "module_statement")
        if not name:
            warnings.append(f"Skipped module at bytes [{n.start_byte},{n.end_byte}] (missing name)")
            continue

        body = node_text(src, n)
        if not is_real_module_slice(body):
            first_line = body.splitlines()[0] if body.splitlines() else ""
            warnings.append(f"Skipped module '{name}' (fragment? first line: {first_line})")
            continue

        entries.append(
            BlockIndexEntry(
                kind="module",
                name=name,
                source_file="",
                block_file="",
                start_byte=n.start_byte,
                end_byte=n.end_byte,
            )
        )

    return entries, warnings


# ---------------- main ----------------

def _extract_one_file(
    p: Path,
    out_procedures: Path,
    out_programs: Path,
    out_modules: Path,
    write_modules: bool,
) -> dict:
    """
    Extract every unit from ONE Fortran file into the (already-prepared) output
    directories, and return that file's contribution to the phase-1 index.

    Split out of main() so a project can list several files in
    [source].fortran_files. Everything that depends on tree-sitter BYTE OFFSETS
    — most importantly matching procedures to their parent module by byte range
    — stays scoped to a single file here. Merging those matches across files
    would be wrong: byte ranges restart at 0 in every file, so a procedure from
    one file can fall inside another file's module range.
    """
    src = p.read_bytes()

    parser = Parser()
    parser.set_language(FORTRAN)
    tree = parser.parse(src)
    root = tree.root_node

    syntax_ok = not has_error(root)
    print(f"File: {p}")
    print(f"Tree-sitter syntax OK? {syntax_ok}")

    omp = extract_openmp_directives(src)
    print(f"OpenMP directives found: {len(omp)}")

    # Extract procedures with type metadata
    proc_entries, proc_warnings = extract_procedures(src, root)
    proc_index: list[dict] = []
    
    for ent in proc_entries:
        safe = sanitize_name(ent.name)
        unit_file = unique_path(out_procedures, safe, ".F90")
        unit_text = src[ent.start_byte:ent.end_byte].decode("utf-8", errors="replace")
        unit_file.write_text(unit_text, encoding="utf-8")

        ent.source_file = str(p)
        ent.unit_file = str(unit_file)
        proc_index.append(asdict(ent))
        
        # Print type info
        print(f"\n  {ent.name}:")
        for arg_name in ent.args:
            if arg_name.lower() in ent.arg_metadata:
                meta = ent.arg_metadata[arg_name.lower()]
                rank_str = f"{meta['rank']}D" if meta['rank'] > 0 else "scalar"
                print(f"    - {arg_name}: {meta['dtype']} {rank_str}")

    # Programs
    prog_entries, prog_warnings = extract_program_blocks(src, root)
    prog_index: list[dict] = []
    for ent in prog_entries:
        safe = sanitize_name(ent.name)
        block_file = unique_path(out_programs, safe, ".F90")
        block_text = src[ent.start_byte:ent.end_byte].decode("utf-8", errors="replace")
        block_file.write_text(block_text, encoding="utf-8")

        ent.source_file = str(p)
        ent.block_file = str(block_file)
        prog_index.append(asdict(ent))

    # Modules - ENHANCED with metadata extraction
    mod_entries: list[BlockIndexEntry] = []
    mod_warnings: list[str] = []
    mod_index: list[dict] = []
    mod_metadata: list[dict] = []  # NEW: Store module metadata
    if write_modules:
        # Extract module blocks (existing)
        mod_entries, mod_warnings = extract_module_blocks(src, root)
        
        # Get module nodes for metadata extraction
        mod_nodes = [n for n in walk(root) if n.type == "module"]
        
        for ent in mod_entries:
            # Save module code (existing)
            safe = sanitize_name(ent.name)
            block_file = unique_path(out_modules, safe, ".F90")
            block_text = src[ent.start_byte:ent.end_byte].decode("utf-8", errors="replace")
            block_file.write_text(block_text, encoding="utf-8")

            ent.source_file = str(p)
            ent.block_file = str(block_file)
            mod_index.append(asdict(ent))
            
            # NEW: Extract and save metadata for this module
            # Find the corresponding module node
            module_node = None
            for n in mod_nodes:
                node_name = _block_name_from_stmt(src, n, "module_statement")
                if node_name and node_name.lower() == ent.name.lower():
                    module_node = n
                    break
            
            if module_node:
                # Extract metadata
                uses_modules = extract_use_statements(module_node, src)
                module_variables = extract_module_variables(module_node, src)
                public_entities, private_entities = extract_public_private_entities(module_node, src)
                
                # FIXED: Get procedures from already-extracted procedure list
                # Check which procedures belong to this module by byte range
                procedures = []
                for proc in proc_entries:
                    # Check if procedure is within this module's byte range
                    if proc.start_byte >= ent.start_byte and proc.end_byte <= ent.end_byte:
                        procedures.append(proc.name.lower())
                
                # Build metadata dict
                metadata = {
                    'module_name': ent.name.lower(),
                    'source_file': str(p),
                    'uses_modules': uses_modules,
                    'module_variables': module_variables,
                    'public_entities': public_entities,
                    'private_entities': private_entities,
                    'procedures': procedures  # Now includes kessler_init, kessler_run
                }
                
                mod_metadata.append(metadata)
                
                # Save individual JSON file
                json_file = out_modules / f"{ent.name.lower()}.json"
                json_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                
                print(f"  ✅ Saved module metadata: {json_file.name}")
                
    warnings = proc_warnings + prog_warnings + mod_warnings

    return {
        "source_file": str(p),
        "syntax_ok": syntax_ok,
        "openmp_directives": omp,
        "warnings": warnings,
        "procedures": proc_index,
        "programs": prog_index,
        "modules": mod_index if write_modules else [],
        "module_metadata": mod_metadata if write_modules else [],
    }


def main(
    input_path,
    out_root: str = "out",
    write_modules: bool = False,
):
    """
    Phase 1: Enhanced Fortran Extraction with Tree-sitter

    NEW: Extracts type information for all procedure arguments
    NEW: Extracts module metadata (USE statements, module variables)
    NEW: accepts EITHER one path or a list of paths ([source].fortran_files).

    Multi-file runs write ONE merged out/phase1_index.json. Every index entry
    already records its own `source_file`, so downstream stages (phase02,
    bridge, prompts, validators) need no change — they iterate the per-entry
    lists and never read the index's top-level `source_file`.
    """
    paths = [Path(input_path)] if isinstance(input_path, (str, Path)) else [Path(x) for x in input_path]
    if not paths:
        raise SystemExit("phase01: no Fortran source files given")

    print(f"NEW: Enhanced with type extraction! 🚀")
    if len(paths) > 1:
        print(f"Parsing {len(paths)} Fortran files\n")

    out_root_p = Path(out_root)
    out_procedures = out_root_p / "procedures"
    out_programs = out_root_p / "programs"
    out_modules = out_root_p / "modules"

    # Clean previous extraction (ensures fresh start). ONCE, before the loop —
    # doing this per file would delete the previous file's extracted units.
    print("Cleaning output directories...")
    for directory in [out_procedures, out_programs]:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    if write_modules:
        if out_modules.exists():
            shutil.rmtree(out_modules)
        out_modules.mkdir(parents=True, exist_ok=True)

    print("✓ Directories cleaned and ready\n")

    per_file = [
        _extract_one_file(p, out_procedures, out_programs, out_modules, write_modules)
        for p in paths
    ]

    # Merge. Concatenation is safe for every list here because each entry
    # carries its own source_file and its own on-disk unit_file/block_file
    # (unique_path() already de-duplicates names across files).
    proc_index = [e for f in per_file for e in f["procedures"]]
    prog_index = [e for f in per_file for e in f["programs"]]
    mod_index = [e for f in per_file for e in f["modules"]]
    mod_metadata = [e for f in per_file for e in f["module_metadata"]]
    omp = [d for f in per_file for d in f["openmp_directives"]]
    warnings = [w for f in per_file for w in f["warnings"]]

    # Write enhanced index
    out_index = out_root_p / "phase1_index.json"
    out_index.parent.mkdir(parents=True, exist_ok=True)
    out_index.write_text(
        json.dumps(
            {
                # Kept singular for backward compatibility (first/only source);
                # `source_files` is the authoritative list for multi-file runs.
                "source_file": str(paths[0]),
                "source_files": [str(p) for p in paths],
                "syntax_ok": all(f["syntax_ok"] for f in per_file),
                "syntax_ok_by_file": {f["source_file"]: f["syntax_ok"] for f in per_file},
                "openmp_directives": omp,
                "warnings": warnings,
                "procedures_written": len(proc_index),
                "programs_written": len(prog_index),
                "modules_written": len(mod_index) if write_modules else 0,
                "procedures": proc_index,
                "programs": prog_index,
                "modules": mod_index if write_modules else [],
                "module_metadata": mod_metadata if write_modules else [],  # NEW: Include module metadata
                "enhanced": True,  # NEW: Flag indicating type metadata present
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n✅ Wrote {len(proc_index)} procedure units to: {out_procedures}/")
    print(f"✅ Wrote {len(prog_index)} program blocks to: {out_programs}/")
    if write_modules:
        print(f"✅ Wrote {len(mod_index)} module blocks to: {out_modules}/")
        print(f"✅ Saved {len(mod_metadata)} module metadata files")
    if warnings:
        print(f"⚠️  Warnings: {len(warnings)} (see {out_index})")

    bad = [f["source_file"] for f in per_file if not f["syntax_ok"]]
    if bad:
        print(f"\n⚠️  Tree-sitter reported syntax errors in {len(bad)} of {len(paths)} file(s):")
        for b in bad:
            print(f"     - {b}")
        print("     Extraction still ran. Common causes: cpp directives (#include,")
        print("     #ifdef) and preprocessor function-macros. Spot-check the extracted")
        print("     units for these files before trusting their packets.")

    print(f"\n🎉 Enhanced Phase 1 complete with type metadata!")


if __name__ == "__main__":
    import argparse
    from framework_config import init as init_config, add_config_arg

    ap = argparse.ArgumentParser(description="Phase 01: Fortran extraction (tree-sitter)")
    add_config_arg(ap)
    cfg = init_config(ap.parse_args().config)

    if not cfg.fortran_files:
        raise SystemExit("[source].fortran_files is empty — nothing to parse")
    main([str(f) for f in cfg.fortran_files], out_root=str(cfg.out_root), write_modules=True)