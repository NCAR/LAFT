# -----------------------------------------------------------
# Phase 4.1: Generate Prompts (Bridge Mode + MODULE INOUT)
#
# ENHANCED: MODULE_VARIABLE_POLICY support (INOUT pattern)
# MODULE variables are treated as INOUT parameters
# -----------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

# framework_config lives in LAFT/config/; this script now lives in a
# workflow folder symlinked from LAFT/ into each project — resolve back to
# LAFT/config to import it. Same convention as workflow_bridge/phase03_make_bridge.py.
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "config"))

from framework_config import get_config, packet_module_vars, safe_py_name

def _as_list(x) -> List[str]:
    return x if isinstance(x, list) else []


def _is_scalar_only_proc(merged: Dict[str, Any]) -> bool:
    """
    Returns True if this procedure has NO array arguments.

    Detects arrays by scanning the Fortran source for:
      - ':: varname(' pattern  (e.g. 'real :: a(n)' or 'real :: a(:)')
      - 'dimension(' attribute (e.g. 'real, dimension(n) :: a')

    character(len=...) is scalar — '(' appears before '::', not after.

    Scalar-only procedures should use plain Python, not JAX/JIT.
    """
    import re
    fortran_source = merged.get("phase2_packet", {}).get("fortran_source", "")
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

def _is_dim_name(name: str) -> bool:
    # Single source: [heuristics].dim_names in config/project.toml
    return get_config().is_dim_name(name)

def _is_scalar_output_flag(name: str) -> bool:
    # Single source: [heuristics].scalar_flag_names in config/project.toml
    return get_config().is_scalar_output_flag(name)

def _strip_fortran_comments(source: str) -> str:
    """Remove full-line and inline Fortran comments, then collapse blank lines."""
    stripped = []
    for line in source.splitlines():
        # Remove inline comment (everything from first ! onward)
        code = line.split("!")[0].rstrip()
        # Skip lines that are now blank
        if code.strip():
            stripped.append(code)
    return "\n".join(stripped)


def _load_policy(filename: str) -> str:
    """Load a policy block from [prompts].policies_dir/<filename>. Raises FileNotFoundError if missing."""
    cfg = get_config()
    search_paths = [
        cfg.policies_dir / filename,
        cfg.out_root / filename,
        Path(filename),
    ]
    for path in search_paths:
        if path.exists():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Policy file '{filename}' not found. Searched: {[str(p) for p in search_paths]}"
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_module_info(merged: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Extract MODULE variable information from merged packet."""
    parent_module  = merged.get("parent_module", "")
    # Shared rule (declaration-ordered direct refs minus config excludes) —
    # bridges/wrappers/lint use the same one, so signatures always agree.
    module_vars_used = packet_module_vars(merged)
    # A module var that is also a dummy argument is already in the signature —
    # keeping it would mandate a duplicate parameter (e.g. dnu in
    # sedimentation_liquid). Mirrors the dedup in phase03_make_bridge.
    arg_names = {a.lower() for a in (merged.get("deps", {}) or {}).get("args", []) or []}
    module_vars_used = [v for v in module_vars_used if v.lower() not in arg_names]
    return parent_module, module_vars_used


def compute_signature_policy(
    deps: Dict[str, Any],
    module_vars_used: List[str] = None
) -> Dict[str, Any]:
    """Compute signature policy with MODULE variable support."""
    module_vars_used = module_vars_used or []

    args          = _as_list(deps.get("args", []))
    writes_to_args = set(a.lower() for a in _as_list(deps.get("writes_to_args", [])))

    scalar_out   = [
        a for a in args
        if (a.lower() in writes_to_args)
        and (not _is_dim_name(a))
        and _is_scalar_output_flag(a)
    ]
    scalar_out_l = {a.lower() for a in scalar_out}

    wrapper_args = [a for a in args if a.lower() not in scalar_out_l]
    core_args    = list(wrapper_args)

    if module_vars_used:
        wrapper_args.extend(module_vars_used)
        core_args.extend(module_vars_used)

    returns = [a for a in wrapper_args if a.lower() in writes_to_args] + scalar_out

    if module_vars_used:
        for var in module_vars_used:
            if var not in returns:
                returns.append(var)

    # Sanitize ONLY here, after every raw-Fortran-name comparison above is
    # done. These four lists are emitted into the prompt as Python identifiers
    # ("Use these signatures EXACTLY"), so a Fortran name that is a Python
    # keyword — SCALE's `is` — would otherwise instruct the translator to write
    # a file that cannot parse. Same convention as the bridge, so the generated
    # signatures still line up. See framework_config.safe_py_name.
    return {
        "wrapper_args": [safe_py_name(a) for a in wrapper_args],
        "core_args":    [safe_py_name(a) for a in core_args],
        "scalar_out":   [safe_py_name(a) for a in scalar_out],
        "returns":      [safe_py_name(a) for a in returns],
    }


def build_signature_section(
    proc_name: str,
    sig: Dict[str, Any],
    parent_module: str = "",
    module_vars_used: List[str] = None
) -> str:
    """Build signature section with MODULE variable documentation."""
    module_vars_used = module_vars_used or []

    s  = "\n## Signature policy (Bridge Mode)\n"
    s += f"- Wrapper signature MUST be: `{proc_name}({', '.join(sig['wrapper_args'])})`\n"
    s += f"- Core signature MUST be: `{proc_name}_core({', '.join(sig['core_args'])})`\n"
    s += "- Use these signatures EXACTLY — do not add, drop, or reorder parameters.\n"
    s += "- Do NOT add parameters for variables that appear only in Fortran comments\n"
    s += "  or commented-out code (dead code). The bridge layer is generated from the\n"
    s += "  signature above; an invented parameter breaks the bridge call.\n"

    if module_vars_used and parent_module:
        s += f"\n**MODULE VARIABLES (from {parent_module}):**\n"
        for var in module_vars_used:
            s += f"  - `{var}`: MODULE variable (INOUT)\n"
        s += "\nMODULE variables are INOUT parameters - include in signature and return them!\n"

    if sig["scalar_out"]:
        s += "\n- Scalar output-only flags are returned, not passed as parameters.\n"
        s += f"  - Dropped from signatures: {', '.join(sig['scalar_out'])}\n"

    if sig["returns"]:
        s += f"\n- Return updated values: `{', '.join(sig['returns'])}`\n"
    else:
        s += "\n- If nothing is written, return `None`.\n"

    s += "\n**Important:** Do NOT include dropped scalar output-only flags in parameters.\n"
    return s


def build_module_variable_section(parent_module: str, module_vars_used: List[str]) -> str:
    """Build MODULE variable policy section for prompt."""
    if not module_vars_used or not parent_module:
        return ""

    policy  = _load_policy("MODULE_VARIABLE_POLICY.md")
    section  = "\n" + "=" * 70 + "\n"
    section += "MODULE VARIABLE POLICY (CRITICAL - READ CAREFULLY)\n"
    section += "=" * 70 + "\n\n"
    section += f"This procedure uses MODULE variables from `{parent_module}`:\n"
    for var in module_vars_used:
        section += f"  - {var}\n"
    section += "\n"
    section += policy
    section += "\n" + "=" * 70 + "\n\n"
    return section


# ---------------------------------------------------------------------------
# OpenMP hints
# ---------------------------------------------------------------------------

def build_openmp_section(omp_directives: List[str]) -> str:
    if not omp_directives:
        return ""
    hints = "\n".join(f"  {line}" for line in omp_directives)
    return f"""
## OpenMP hints (for parallelizable loops)

The original Fortran code contains these OpenMP directives:

{hints}

Use these to identify:
- loops that can be vectorized with `jax.vmap`
- loops that need `jax.lax.fori_loop` or `jax.lax.scan`

JAX will automatically parallelize operations where safe.
"""

# ---------------------------------------------------------------------------
# Scalar-only policy block
# ---------------------------------------------------------------------------

SCALAR_ONLY_POLICY_BLOCK = """## SCALAR-ONLY PROCEDURE — Plain Python, No JAX

This procedure operates **only on scalar values** (no arrays).
Generating a JAX `_core` with `jax.jit` would cause unnecessary GPU dispatch
overhead for what are just a few scalar assignments.

### Rules

1. **Do NOT generate a `_core` function.**  Generate a SINGLE plain Python function only.
2. **Do NOT import JAX** unless absolutely needed (e.g. `jnp.asarray` for a scalar result
   that must interop with a JAX array downstream — even then, prefer plain Python floats).
3. **Do NOT use `jnp.asarray`, `jnp.float64`, or any `jnp.*` call** for scalar assignments.
   Just use plain Python arithmetic:
   ```python
   lv    = float(lv_in)
   pref  = float(pref_in) / 100.0
   rhoqr = float(rhoqr_in)
   errflg = 0
   errmsg = ""
   ```
4. **String outputs** (`errmsg`) are plain Python strings — assign directly.
5. **Module variables** are plain Python floats passed in and returned — no JAX wrappers.

### Output format

Return Python code only (no explanations). Include:
1) No JAX imports (or minimal if genuinely needed)
2) Module-level docstring marking this as SCALAR-ONLY
3) A SINGLE function `{proc_name}(...)` — NO `{proc_name}_core`
4) Return the updated values as plain Python scalars
"""


# ---------------------------------------------------------------------------
# Prompt headers
# ---------------------------------------------------------------------------

def get_scalar_only_prompt_header(proc_name: str) -> str:
    return f"""# Fortran → Python Translation Task (SCALAR-ONLY)

Translate a single Fortran procedure into a **plain Python function**.

This procedure contains NO array arguments — only scalars — and is NOT called
from within any JAX/JIT context, so plain Python is safe and optimal.
**Do NOT generate a `{proc_name}_core` function. Do NOT use JAX.**

## Hard constraints (must follow)
- Generate ONE function only: `{proc_name}(...)` — no `_core`.
- Use plain Python arithmetic and Python types (float, int, str).
- Do NOT import jax, jax.numpy, or numpy.
- Do not invent globals. Only use what's in the Fortran unit.

{SCALAR_ONLY_POLICY_BLOCK.format(proc_name=proc_name)}
"""


def get_scalar_jax_prompt_header(proc_name: str) -> str:
    return f"""# Fortran → Python/JAX Translation Task (SCALAR, JAX-REQUIRED)

Translate a single Fortran procedure into a **JAX-compatible Python function**.

This procedure has NO array arguments, but it IS called from within a
`@jax.jit`-compiled context.  That means its arguments may be JAX-traced
scalar values, and the function must be fully JAX-compatible.

**Do NOT generate a `{proc_name}_core` function or add `@jax.jit`.**
The JIT boundary lives in the caller; this function just needs to be safe
to call from inside one.

## Hard constraints (must follow)
- Generate ONE function only: `{proc_name}(...)` — no `_core`, no `@jax.jit`.
- Use `jnp` scalar operations (e.g. `jnp.where`, `jnp.sqrt`, `jnp.exp`).
- Do NOT use Python `if` / `else` on arguments — they may be JAX-traced.
  Use `jnp.where(condition, true_val, false_val)` instead.
- Do NOT call `int()`, `float()`, `bool()` on arguments inside the function.
- Import `jax.numpy as jnp` but do NOT import `jax` directly or set
  `os.environ["JAX_ENABLE_X64"]` (the caller's top-level file handles that).
- Do not invent globals. Only use what's in the Fortran unit.

{SCALAR_ONLY_POLICY_BLOCK.format(proc_name=proc_name)}
"""


def get_prompt_header(proc_name: str) -> str:
    return f"""# Fortran → Python/JAX Translation Task

Translate a single Fortran procedure into Python with a two-layer structure:

- `{proc_name}_core(...)`: JAX-safe pure compute (no I/O, no side effects)
- `{proc_name}(...)`: Python wrapper (may do I/O if allowed)

Use:
- `import jax`
- `import jax.numpy as jnp`
- `jax.lax` for loops when needed (`lax.fori_loop`, `lax.scan`, `lax.while_loop`)
- Prefer vectorized `jnp` array operations over explicit loops wherever iterations are independent

{_load_policy("jax_x64_header.md")}

## Hard constraints (must follow)
- NO NumPy: do not import or use `numpy` at all.
- NO in-place mutation of inputs. If Fortran writes to an argument, return the updated value(s).
- The translated core must be `jax.jit`-compatible:
  - avoid host conversions (`.tolist()`, `float(x)`, `.item()` inside computation)
  - avoid Python loops over array values; use `lax.*` primitives or vectorized ops instead.
- Do not invent globals. Only use what's in the Fortran unit.

{_load_policy("layout_policy.md")}

{_load_policy("fortran_intrinsics_map.md")}

{_load_policy("jax_dtype_rules.md")}

{_load_policy("return_type_rules.md")}

{_load_policy("jax_jit_boundary_rules.md")}

{_load_policy("jax_vectorization_rules.md")}

{_load_policy("jax_control_flow_rules.md")}

{_load_policy("code_annotation_rules.md")}

{_load_policy("smell_test.md")}

## Output format
Return Python code only (no explanations). Include:
1) imports (with `os.environ` line first, `import functools` included)
2) module-level docstring
3) `{proc_name}_core(...)` decorated with `@functools.partial(jax.jit, static_argnames=(...))`
   (unless wrapper-only policy applies)
4) `{proc_name}(...)` wrapper — calls `{proc_name}_core(...)` directly, no extra jit needed
5) Second line of file MUST be: `# JIT boundary check: PASSED` or
   `# JIT boundary check: FIXED — <description of what was fixed>`
6) Every loop, conditional, and non-trivial assignment MUST carry an inline `# [TAG]` annotation
"""

# ---------------------------------------------------------------------------
# Main prompt builder
# ---------------------------------------------------------------------------

def generate_prompt(proc_name: str, merged: Dict[str, Any], out_dir: Path):
    """Generate translation prompt for a procedure with MODULE support."""

    parent_module, module_vars_used = extract_module_info(merged)

    deps         = merged.get("deps", {})
    effects      = deps.get("effects", {})
    has_io       = effects.get("has_io", False)

    sig = compute_signature_policy(deps, module_vars_used)

    scalar_only = _is_scalar_only_proc(merged)

    if scalar_only:
        prompt = get_scalar_only_prompt_header(proc_name)
    else:
        prompt = get_prompt_header(proc_name)

    if module_vars_used and parent_module:
        prompt += build_module_variable_section(parent_module, module_vars_used)

    prompt += build_signature_section(proc_name, sig, parent_module, module_vars_used)

    if has_io:
        prompt += "\n" + _load_policy("io_policy.md")

    if any(a in sig["wrapper_args"] for a in get_config().error_arg_names):
        prompt += "\n" + _load_policy("error_policy.md")

    if not scalar_only:
        omp_directives = deps.get("openmp_directives", [])
        if omp_directives:
            prompt += build_openmp_section(omp_directives)

    phase2_packet  = merged.get("phase2_packet", {})
    fortran_source = _strip_fortran_comments(phase2_packet.get("fortran_source", ""))
    prompt += f"\n---\n\n# Fortran source to translate:\n\n```fortran\n{fortran_source}\n```\n"

    prompt_file = out_dir / f"{proc_name}.md"
    prompt_file.write_text(prompt, encoding="utf-8")

    tag = " [SCALAR-ONLY]" if scalar_only else ""
    if module_vars_used and parent_module:
        print(f"✅ {proc_name}{tag} [MODULE: {', '.join(module_vars_used)}] → {prompt_file}")
    else:
        print(f"✅ {proc_name}{tag} → {prompt_file}")

# ---------------------------------------------------------------------------
# Multi-pass prompt generator
# ---------------------------------------------------------------------------

def generate_pass_prompts(proc_name: str, merged: Dict[str, Any], out_dir: Path,
                          jax_required: bool = True):
    """Generate prompts for a procedure.

    Pass count is determined by two orthogonal flags:
      scalar_only=False               → 4 passes (full JAX array treatment)
      scalar_only=True, jax_required  → 2 passes (JAX-scalar: pass1 + pass2 JIT safety)
      scalar_only=True, not required  → 1 pass  (plain Python, no JAX needed)
    """

    parent_module, module_vars_used = extract_module_info(merged)

    deps         = merged.get("deps", {})
    effects      = deps.get("effects", {})
    has_io       = effects.get("has_io", False)

    sig = compute_signature_policy(deps, module_vars_used)

    scalar_only = _is_scalar_only_proc(merged)

    phase2_packet  = merged.get("phase2_packet", {})
    fortran_source = _strip_fortran_comments(phase2_packet.get("fortran_source", ""))
    fortran_block  = f"\n---\n\n# Fortran source to translate:\n\n```fortran\n{fortran_source}\n```\n"

    # ------------------------------------------------------------------ #
    # PASS 1 — First Draft
    # ------------------------------------------------------------------ #

    if scalar_only and not jax_required:
        pass1 = get_scalar_only_prompt_header(proc_name)
    elif scalar_only and jax_required:
        pass1 = get_scalar_jax_prompt_header(proc_name)
    else:
        pass1  = f"""# Fortran → Python/JAX Translation — Pass 1: First Draft

Translate the Fortran procedure below into Python with a two-layer structure:
- `{proc_name}_core(...)`: JAX-safe pure compute (no I/O, no side effects), decorated with `@functools.partial(jax.jit, static_argnames=(...))`
- `{proc_name}(...)`: Python wrapper

## Hard constraints
- NO NumPy: do not import or use `numpy` at all.
- NO in-place mutation. If Fortran writes to an argument, return the updated value(s).
- The core must be `jax.jit`-compatible: no Python loops over arrays, no host conversions inside core.
- Do not invent globals. Only use what's in the Fortran unit.
- This two-layer jitted contract applies to EVERY procedure with array
  arguments — including a scheme's top-level main routine and model-facing
  wrappers. "It mostly orchestrates other calls" is NOT an exemption (see
  Orchestration Boundary Rules below).

## Output format
Return Python code only (no explanations). Include:
1) Imports (os.environ line first, then functools, then jax)
2) Module-level docstring
3) `{proc_name}_core(...)` with `@functools.partial(jax.jit, static_argnames=(...))`
4) `{proc_name}(...)` wrapper

"""
        pass1 += _load_policy("jax_x64_header.md") + "\n"
        pass1 += _load_policy("orchestration_boundary_rules.md") + "\n"
        pass1 += _load_policy("layout_policy.md") + "\n"
        pass1 += _load_policy("fortran_intrinsics_map.md") + "\n"
        pass1 += _load_policy("return_type_rules.md") + "\n"
        # Project-specific rules (e.g. the scheme's hot loops / LUT idioms),
        # configured per project in [prompts].project_policies.
        for policy_file in get_config().project_policies:
            pass1 += _load_policy(policy_file) + "\n"

    if module_vars_used and parent_module:
        pass1 += build_module_variable_section(parent_module, module_vars_used)

    pass1 += build_signature_section(proc_name, sig, parent_module, module_vars_used)

    if has_io:
        pass1 += "\n" + _load_policy("io_policy.md")

    if any(a in sig["wrapper_args"] for a in get_config().error_arg_names):
        pass1 += "\n" + _load_policy("error_policy.md")

    if not scalar_only:
        omp_directives = deps.get("openmp_directives", [])
        if omp_directives:
            pass1 += build_openmp_section(omp_directives)

    pass1 += fortran_block

    pass1_file = out_dir / f"{proc_name}_pass1.md"
    pass1_file.write_text(pass1, encoding="utf-8")

    if scalar_only and not jax_required:
        tag = " [SCALAR-ONLY/PYTHON]"
    elif scalar_only and jax_required:
        tag = " [SCALAR-JAX]"
    else:
        tag = ""
    module_tag = f" [MODULE: {', '.join(module_vars_used)}]" if (module_vars_used and parent_module) else ""

    if scalar_only and not jax_required:
        # Plain Python: only pass 1 needed
        print(f"✅ {proc_name}{tag}{module_tag}")
        print(f"   → {pass1_file} ({sum(1 for _ in pass1.splitlines())} lines)")
        return

    # ------------------------------------------------------------------ #
    # PASS 2 — JIT Safety
    # ------------------------------------------------------------------ #

    general_rules = _load_policy("01_general_rules.md")
    jit_rules = _load_policy("jax_jit_boundary_rules.md")

    pass2 = f"""# JAX Translation Review — Pass 2: JIT Safety

The code below is a Python/JAX translation of a Fortran procedure.

**Your task:** fix every JIT/tracer boundary violation found in the code.

{general_rules}

{jit_rules}

## Instructions
1. Read the code and apply the rules above.
2. Fix every violation: forbidden Python `if`/`for`/`while` on JAX values, tracer infection, static-int misuse, host value extraction (`.item()`, `int()`, `float()`), strings inside `_core`, etc.
3. On the second line of the file add: `# JIT boundary check: FIXED — <what you changed>` if fixes were made, or `# JIT boundary check: PASSED` if none were needed.
4. Return the **complete corrected file** — code only, no explanations.

---
# Code to review:

<<<PREVIOUS_CODE>>>
"""

    pass2_file = out_dir / f"{proc_name}_pass2.md"
    pass2_file.write_text(pass2, encoding="utf-8")

    if scalar_only and jax_required:
        # Scalar-JAX: pass1 (JAX-scalar draft) + pass2 (JIT safety) only.
        # No arrays → no vectorization (pass3) or dtype annotation (pass4) needed.
        print(f"✅ {proc_name}{tag}{module_tag}")
        print(f"   → {pass1_file} ({sum(1 for _ in pass1.splitlines())} lines)")
        print(f"   → {pass2_file} (JIT safety)")
        return


    # ------------------------------------------------------------------ #
    # PASS 3 — Vectorization & Control Flow
    # ------------------------------------------------------------------ #

    orch_rules = _load_policy("orchestration_boundary_rules.md")
    vec_rules  = _load_policy("jax_vectorization_rules.md")
    ctrl_rules = _load_policy("jax_control_flow_rules.md")

    pass3 = f"""# JAX Translation Review — Pass 3: Vectorization & Control Flow

The code below is a Python/JAX translation of a Fortran procedure.

**Your task:** optimize loops and fix control flow patterns.

{general_rules}

{orch_rules}

{vec_rules}

{ctrl_rules}

## Instructions
1. Replace independent loops with vectorized `jnp` operations or `jax.vmap` (follow the VECTORIZATION PRIORITY LADDER).
2. Replace any remaining `lax.fori_loop` with vectorized ops where iterations are independent.
3. Fix any mixed Python/JAX loop nesting (Python `for` wrapping `lax.*` or `jax.vmap` is forbidden).
4. Ensure all `lax` loop state tuples are consistent in dtype and shape.
5. Return the **complete corrected file** — code only, no explanations.

---
# Code to review:

<<<PREVIOUS_CODE>>>
"""

    pass3_file = out_dir / f"{proc_name}_pass3.md"
    pass3_file.write_text(pass3, encoding="utf-8")

    # ------------------------------------------------------------------ #
    # PASS 4 — dtype, Annotations & Self-check
    # ------------------------------------------------------------------ #

    dtype_rules      = _load_policy("jax_dtype_rules.md")
    annotation_rules = _load_policy("code_annotation_rules.md")
    smell_test       = _load_policy("smell_test.md")

    pass4 = f"""# JAX Translation Review — Pass 4: dtype, Annotations & Self-check

The code below is a Python/JAX translation of a Fortran procedure.

**Your task:** finalize dtype consistency, add annotation tags, and verify against the self-check checklist.

{general_rules}

{dtype_rules}

{annotation_rules}

{smell_test}

## Instructions
1. Add `dtype=jnp.float64` or `dtype=jnp.int32` to every `jnp.zeros`, `jnp.ones`, `jnp.empty`, `jnp.asarray` call missing it.
2. Add inline `# [TAG]` annotations to every loop, conditional, and non-trivial assignment in `_core`.
3. Run through the self-check checklist mentally — fix anything that fails.
4. Return the **complete corrected file** — code only, no explanations.

---
# Code to review:

<<<PREVIOUS_CODE>>>
"""

    pass4_file = out_dir / f"{proc_name}_pass4.md"
    pass4_file.write_text(pass4, encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Print summary
    # ------------------------------------------------------------------ #

    print(f"✅ {proc_name}{module_tag}")
    print(f"   → {pass1_file} ({sum(1 for _ in pass1.splitlines())} lines)")
    print(f"   → {pass2_file} (template)")
    print(f"   → {pass3_file} (template)")
    print(f"   → {pass4_file} (template)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cfg = get_config()
    packets_dir = cfg.packets_dir
    prompts_dir = cfg.prompts_dir
    prompts_dir.mkdir(parents=True, exist_ok=True)

    merged_files = sorted(packets_dir.glob("*_merged.json"))

    if not merged_files:
        print("❌ No *_merged.json files found in out/packets/")
        print("   Run phase 3.1 merge first")
        return 1

    # Load jax_required flags from _ALL_deps.json (computed by phase02_02)
    all_deps_file = packets_dir / "_ALL_deps.json"
    jax_required_map: Dict[str, bool] = {}
    if all_deps_file.exists():
        all_deps = json.loads(all_deps_file.read_text(encoding="utf-8"))
        jax_required_map = {
            e["proc_name"].lower(): e.get("jax_required", True)
            for e in all_deps
        }
    else:
        print("⚠️  _ALL_deps.json not found — defaulting all procedures to jax_required=True")

    print(f"Generating prompts for {len(merged_files)} procedures...")
    print(f"Output: {prompts_dir}")
    print()

    for mf in merged_files:
        proc_name = mf.stem.replace("_merged", "")
        merged    = json.loads(mf.read_text(encoding="utf-8"))
        jax_req   = jax_required_map.get(proc_name.lower(), True)
        generate_pass_prompts(proc_name, merged, prompts_dir, jax_required=jax_req)

    print()
    print(f"✅ Generated pass prompts for {len(merged_files)} procedures in {prompts_dir}/")
    return 0


if __name__ == "__main__":
    import argparse
    from framework_config import init as init_config, add_config_arg

    ap = argparse.ArgumentParser(description="Phase 04.1: LLM prompt generation")
    add_config_arg(ap)
    init_config(ap.parse_args().config)
    exit(main())