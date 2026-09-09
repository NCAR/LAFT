#!/usr/bin/env python3
"""
Phase 03: Bridge Code Generation (PRODUCTION VERSION)
LAFT bridge generator — project-agnostic (paths and hardware from config/project.toml)

ENHANCED WITH MODULE_VARIABLE_POLICY SUPPORT (INOUT Pattern)
MODULE variables are treated as INOUT parameters (Option A)

STRATEGY (PATH C, device-side layout conversion — promoted 2026-08-11):
- Arrays ship to the device UNCHANGED (pure H2D, no host permute)
- Rank>=2 axis reversal happens INSIDE a per-procedure jitted wrapper
  around {proc}_core, both directions — XLA fuses the permute into the
  kernel (measured 2.38x end-to-end @1e6, bit-identical; PBS 6834200/6834351)
- Outputs return pure D2H, already standard C-order NumPy (ncol, nz)
- CHARACTER args stay host-side (strings cannot cross the jit boundary);
  scalar-only procedures call the translated wrapper directly
- Bridge public signature/returns UNCHANGED
- MODULE variables passed as INOUT parameters
- INIT procedures: Return updated MODULE vars to driver
- COMPUTE procedures: Return updated MODULE vars to driver
- Driver manages MODULE state explicitly (no hidden state)
- REPORT loop swap opportunities (educational only)
- WARN about GPU inefficiency for small arrays

Author: Auto-generated for the LAFT framework
Date: 2026-03-09 (generalized 2026-07-09)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# workflow_bridge/ is a per-project SYMLINK to LAFT/workflow_bridge/, so
# Path(__file__).resolve() follows it back to the shared LAFT/ tree — safe
# here only to locate the sibling LAFT/config/ (where framework_config.py
# lives) for the import. Per-project paths come from get_config(), which
# discovers the project root by walking up from the cwd (runs always cd to
# the project root first) — same convention as workflow_profiler/.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "config"))

from framework_config import get_config, packet_module_vars, safe_py_name  # noqa: E402


# ============================================================================
# Data Structures
# ============================================================================

class LoopSwapSafety(Enum):
    """Conservative loop swap safety classification"""
    DEFINITELY_UNSAFE = "unsafe"
    UNKNOWN = "unknown"
    PROBABLY_SAFE = "probably_safe"


@dataclass
class SafetyAnalysis:
    """Results from loop swap safety analysis"""
    safety: LoopSwapSafety
    reasons: List[str]
    warnings: List[str]
    
    def is_safe(self) -> bool:
        return self.safety == LoopSwapSafety.PROBABLY_SAFE
    
    def report(self) -> str:
        """Generate human-readable safety report"""
        lines = []
        lines.append(f"Safety: {self.safety.value.upper()}")
        
        if self.reasons:
            lines.append("\nReasons:")
            for r in self.reasons:
                lines.append(f"  • {r}")
        
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  ⚠️  {w}")
        
        return "\n".join(lines)


@dataclass
class ParamInfo:
    """Parameter metadata extracted from phase1_index.json"""
    name: str
    rank: int              # 0=scalar, 1=1D, 2=2D, ...
    dtype: str             # real, integer, logical, character
    is_output: bool        # True if in writes_to_args
    dimensions: List[str]  # e.g., [":", ":"] for 2D assumed-shape
    intent: Optional[str]  # in, out, inout, or None
    
    def estimated_size_mb(self, typical_n: int = 1000) -> float:
        """Estimate typical array size in MB for performance analysis"""
        if self.rank == 0:
            return 0.0
        
        # Rough estimate: assume each dimension is ~typical_n
        elements = typical_n ** self.rank
        
        # Estimate bytes per element
        if self.dtype in ('real', 'double'):
            bytes_per_elem = 8
        elif self.dtype == 'integer':
            bytes_per_elem = 4
        elif self.dtype == 'logical':
            bytes_per_elem = 1
        elif self.dtype == 'character':
            bytes_per_elem = 1
        else:
            bytes_per_elem = 8
        
        return (elements * bytes_per_elem) / (1024 * 1024)


# ============================================================================
# Conservative Safety Analyzer (UNCHANGED from original)
# ============================================================================

class ConservativeSafetyAnalyzer:
    """
    Conservative loop-swap safety analyzer for Fortran code.
    
    Philosophy: When in doubt, classify as UNSAFE.
    Only mark PROBABLY_SAFE when we have high confidence.
    """
    
    def __init__(self):
        self.loop_details: List[Dict] = []
    
    def analyze(self, fortran_source: str, proc_name: str) -> SafetyAnalysis:
        """
        Analyze Fortran procedure for loop-swap safety.
        
        Returns:
            SafetyAnalysis with conservative classification
        """
        self.loop_details = []
        
        reasons = []
        warnings = []
        
        # Default: UNKNOWN (we don't have enough information)
        safety = LoopSwapSafety.UNKNOWN
        reasons.append("Insufficient analysis to prove safety")
        
        # Check for obvious red flags
        if self._check_io(fortran_source):
            safety = LoopSwapSafety.DEFINITELY_UNSAFE
            reasons.clear()
            reasons.append("Contains I/O operations (WRITE/READ/PRINT)")
        
        if self._check_module_usage(fortran_source):
            warnings.append("Uses MODULE variables - verify no cross-column dependencies")
        
        # Extract and analyze loops
        loops = self._extract_loops(fortran_source)
        self.loop_details = loops
        
        if not loops:
            reasons.append("No DO loops detected")
            return SafetyAnalysis(safety, reasons, warnings)
        
        # Analyze each loop
        all_safe = True
        for loop in loops:
            loop_safety = self._analyze_loop(loop)
            
            if loop_safety == LoopSwapSafety.DEFINITELY_UNSAFE:
                all_safe = False
                safety = LoopSwapSafety.DEFINITELY_UNSAFE
                reasons.clear()
                reasons.append(f"Loop {loop['index_var']} has unsafe patterns")
                break
            
            if loop_safety == LoopSwapSafety.UNKNOWN:
                all_safe = False
        
        if all_safe and len(loops) > 0:
            safety = LoopSwapSafety.PROBABLY_SAFE
            reasons.clear()
            reasons.append("All loops appear safe for column-major → row-major swap")
        
        return SafetyAnalysis(safety, reasons, warnings)
    
    def get_loop_details(self) -> List[Dict]:
        """Return detailed information about analyzed loops"""
        return self.loop_details
    
    def _extract_loops(self, code: str) -> List[Dict]:
        """Extract DO loop information"""
        loops = []
        
        # Pattern: DO var = start, end [, step]
        pattern = r'DO\s+(\w+)\s*=\s*([^,]+),\s*([^,\n]+)(?:,\s*([^!\n]+))?'
        
        for match in re.finditer(pattern, code, re.IGNORECASE):
            index_var = match.group(1)
            start_expr = match.group(2).strip()
            end_expr = match.group(3).strip()
            step_expr = match.group(4).strip() if match.group(4) else "1"
            
            loops.append({
                'index_var': index_var,
                'start': start_expr,
                'end': end_expr,
                'step': step_expr,
                'line': match.group(0),
            })
        
        return loops
    
    def _analyze_loop(self, loop: Dict) -> LoopSwapSafety:
        """Analyze single loop for safety"""
        
        # Check for obvious problems
        index_var = loop['index_var'].upper()
        
        # Loop-index name hints from [heuristics] in config/project.toml:
        # column iteration (safe for swap) vs level iteration (careful review)
        heur = get_config().section("heuristics")
        column_indicators = heur.get("column_indicators", ['COL', 'ICOL', 'I', 'N'])
        level_indicators = heur.get("level_indicators", ['K', 'LEV', 'KLEV', 'KK', 'ILEV'])
        
        if any(ind in index_var for ind in column_indicators):
            return LoopSwapSafety.PROBABLY_SAFE
        
        if any(ind in index_var for ind in level_indicators):
            return LoopSwapSafety.UNKNOWN
        
        return LoopSwapSafety.UNKNOWN
    
    def _check_io(self, code: str) -> bool:
        """Check for I/O operations"""
        io_keywords = ['WRITE', 'READ', 'PRINT']
        code_upper = code.upper()
        return any(kw in code_upper for kw in io_keywords)
    
    def _check_module_usage(self, code: str) -> bool:
        """Check for USE statements"""
        return bool(re.search(r'USE\s+\w+', code, re.IGNORECASE))


# ============================================================================
# Performance Analyzer (UNCHANGED from original)
# ============================================================================

class PerformanceAnalyzer:
    """Analyze GPU/CPU performance characteristics.

    Hardware numbers come from [hardware] in config/project.toml so the
    advisor's warnings match the cluster the project actually runs on.
    """

    _HW = get_config().section("hardware")
    GPU_NAME = _HW.get("gpu_name", "GPU")
    GPU_LAUNCH_OVERHEAD_US = _HW.get("gpu_launch_overhead_us", 15)
    GPU_TRANSFER_SETUP_US = _HW.get("gpu_transfer_setup_us", 30)
    PCIe_BANDWIDTH_GBS = _HW.get("pcie_bandwidth_gbs", 50)
    GPU_MEMORY_BANDWIDTH_GBS = _HW.get("gpu_mem_bandwidth_gbs", 1200)
    GPU_COMPUTE_TFLOPS = _HW.get("gpu_compute_tflops", 9.7)

    @staticmethod
    def analyze_gpu_efficiency(params: List[ParamInfo], typical_n: int = 100) -> Tuple[bool, str]:
        """Determine if GPU usage would be inefficient on the configured GPU."""
        max_array_mb = 0.0
        total_array_mb = 0.0
        for p in params:
            if p.rank > 0:
                size_mb = p.estimated_size_mb(typical_n)
                max_array_mb = max(max_array_mb, size_mb)
                total_array_mb += size_mb
        
        if max_array_mb == 0.0:
            return False, ""
        
        kernel_launch_us = PerformanceAnalyzer.GPU_LAUNCH_OVERHEAD_US
        transfer_setup_us = PerformanceAnalyzer.GPU_TRANSFER_SETUP_US
        
        total_array_bytes = total_array_mb * 1024 * 1024
        transfer_time_us = (total_array_bytes * 2) / (PerformanceAnalyzer.PCIe_BANDWIDTH_GBS * 1e9) * 1e6
        
        total_overhead_us = kernel_launch_us + transfer_setup_us + transfer_time_us
        
        if max_array_mb < 0.1:
            warning = (
                f"Arrays are very small (~{max_array_mb*1024:.1f} KB). "
                f"{PerformanceAnalyzer.GPU_NAME} GPU overhead (~{total_overhead_us:.0f}μs: "
                f"{kernel_launch_us}μs launch + {transfer_setup_us}μs setup + "
                f"{transfer_time_us:.0f}μs transfer) likely exceeds compute time. "
                f"Recommendation: Use CPU execution for this workload."
            )
            return True, warning
        
        if max_array_mb < 1.0:
            if transfer_time_us > 100:
                warning = (
                    f"Arrays are relatively small (~{max_array_mb:.2f} MB). "
                    f"{PerformanceAnalyzer.GPU_NAME} PCIe transfer (~{transfer_time_us:.0f}μs @ {PerformanceAnalyzer.PCIe_BANDWIDTH_GBS} GB/s) "
                    f"plus launch overhead (~{kernel_launch_us + transfer_setup_us}μs) may dominate. "
                    f"GPU beneficial only if compute time > ~{total_overhead_us:.0f}μs."
                )
                return True, warning
        
        return False, ""


# ============================================================================
# Bridge Generator (ENHANCED WITH MODULE INOUT PATTERN)
# ============================================================================

class BridgeGenerator:
    """
    Production bridge code generator for LAFT.
    
    ENHANCED: MODULE_VARIABLE_POLICY support (INOUT pattern)
    
    Reads:
    - out/phase1_index.json (parameter metadata)
    - out/packets/*_merged.json (dependencies, Fortran source, MODULE info)
    
    Generates:
    - out/bridge/*_bridge.py (bridge code with MODULE INOUT parameters)
    - out/reports/*_analysis.txt (analysis reports)
    """
    
    def __init__(
        self,
        phase1_index: str = "out/phase1_index.json",
        packets_dir: str = "out/packets",
        jax_dir: str = "out/jax",
        bridge_dir: str = "out/bridge",
        reports_dir: str = "out/reports/bridge",
        enable_profiling: bool = False,
        optional_kwargs: Optional[Dict[str, List[str]]] = None,
    ):
        self.phase1_index_path = Path(phase1_index)
        self.packets_dir = Path(packets_dir)
        self.jax_dir = Path(jax_dir)
        self.bridge_dir = Path(bridge_dir)
        self.reports_dir = Path(reports_dir)
        self.enable_profiling = enable_profiling
        # [bridge.optional_kwargs] in project.toml: proc name -> vars emitted as
        # None-default pass-through kwargs (e.g. LUT tables; None = stub mode)
        self.optional_kwargs = {k.lower(): list(v)
                                for k, v in (optional_kwargs or {}).items()}
        
        self.bridge_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.phase1_data: Optional[dict] = None
        self.argmeta_by_proc: Dict[str, Dict[str, dict]] = {}
        
        self._load_phase1_index()
        
        self.safety_analyzer = ConservativeSafetyAnalyzer()
        self.perf_analyzer = PerformanceAnalyzer()
        
        print(f"📂 Phase 1 index:  {self.phase1_index_path}")
        print(f"📂 Packets:        {self.packets_dir}")
        print(f"📂 JAX code:       {self.jax_dir}")
        print(f"📂 Bridge output:  {self.bridge_dir}")
        print(f"📂 Reports output: {self.reports_dir}")
    
    def _load_phase1_index(self):
        """Load phase1_index.json and extract arg_metadata"""
        if not self.phase1_index_path.exists():
            print(f"⚠️  Phase 1 index not found: {self.phase1_index_path}")
            return
        
        self.phase1_data = json.loads(self.phase1_index_path.read_text(encoding="utf-8"))
        
        procedures = self.phase1_data.get("procedures", [])
        for proc in procedures:
            proc_name = proc.get("name", "").lower()
            arg_metadata = proc.get("arg_metadata", {})
            
            arg_metadata_lower = {k.lower(): v for k, v in arg_metadata.items()}
            
            if proc_name:
                self.argmeta_by_proc[proc_name] = arg_metadata_lower
        
        print(f"✅ Loaded {len(procedures)} procedures from phase1_index.json")
    
    def load_merged_packet(self, packet_file: Path) -> Dict:
        """Load *_merged.json packet"""
        with packet_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    
    def extract_parameters(self, merged_packet: Dict) -> List[ParamInfo]:
        """Extract parameter information from merged packet + phase1 metadata."""
        proc_name = merged_packet.get("proc_name", "")
        proc_name_lower = proc_name.lower()
        
        deps = merged_packet.get("deps", {})
        args = deps.get("args", [])
        writes_to_args = set(w.lower() for w in deps.get("writes_to_args", []))
        
        arg_metadata = self.argmeta_by_proc.get(proc_name_lower, {})
        
        if not arg_metadata:
            print(f"  ⚠️  No arg_metadata found for {proc_name}")
        
        params = []
        for arg_name in args:
            arg_lower = arg_name.lower()
            metadata = arg_metadata.get(arg_lower, {})
            
            params.append(ParamInfo(
                name=arg_lower,
                rank=metadata.get("rank", 0),
                dtype=metadata.get("dtype", "real"),
                is_output=(arg_lower in writes_to_args),
                dimensions=metadata.get("dimensions", []),
                intent=metadata.get("intent", None),
            ))
        
        return params
    
    # ========================================================================
    # MODULE VARIABLE EXTRACTION
    # ========================================================================
    
    def extract_module_info(self, merged_packet: Dict) -> Tuple[str, List[str]]:
        """
        Extract MODULE variable information from merged packet.
        
        Implements MODULE_VARIABLE_POLICY (INOUT pattern):
        - parent_module: Which MODULE this procedure belongs to
        - module vars: declaration-ordered direct references minus config
          excludes — the shared rule in framework_config.packet_module_vars

        Returns:
            (parent_module, module_vars_used)
        """
        parent_module = merged_packet.get("parent_module", "")
        module_vars_used = packet_module_vars(merged_packet)

        return parent_module, module_vars_used
    
    # ========================================================================
    # INIT vs COMPUTE DETECTION (Simplified for INOUT pattern)
    # ========================================================================
    
    def is_initialization_procedure(
        self,
        proc_name: str,
        module_vars_used: List[str],
        writes: List[str]
    ) -> bool:
        """
        Detect if procedure is an initialization routine.
        
        NOTE: In INOUT pattern, this is mainly for documentation purposes.
        Both INIT and COMPUTE procedures return MODULE vars.
        
        Detection Rules:
        1. Name contains 'init' (case-insensitive)
        2. Writes to MODULE variables (suggests setting them up)
        
        Args:
            proc_name: Procedure name
            module_vars_used: List of MODULE variables used
            writes: List of all variables written by procedure
            
        Returns:
            True if this is an initialization procedure
        """
        if not module_vars_used:
            return False
        
        # Rule 1: Name check
        if 'init' in proc_name.lower():
            return True
        
        # Rule 2: Writes to ALL module variables (suggests initialization)
        writes_lower = {w.lower() for w in writes}
        if all(v.lower() in writes_lower for v in module_vars_used):
            return True
        
        return False
    
    # ========================================================================
    # Conversion Functions
    # ========================================================================
    
    def generate_conversion_functions(self, params: List[ParamInfo]) -> str:
        """Generate device-transfer helpers (pure H2D/D2H, no host permute)"""

        if not any(p.rank > 0 for p in params):
            return "# No array conversions needed (all parameters are scalars)\n\n"

        code = '"""\n'
        code += "Layout Conversion (PATH C: device-side)\n\n"
        code += "STRATEGY:\n"
        code += "- Ship host arrays to the device AS-IS (pure H2D, no host permute)\n"
        code += "- Reverse axes INSIDE the jitted device wrapper — XLA fuses the\n"
        code += "  permute into the kernel (HBM bandwidth, not host memcpy)\n"
        code += "- Pure D2H on the way back: outputs already (ncol, nz) C-order,\n"
        code += "  fetched in ONE batched jax.device_get (single stream sync)\n\n"
        code += "Two bridge contracts per array-bearing procedure:\n"
        code += "  1. {proc}_bridge        — host-facing: standard C-order NumPy arrays\n"
        code += "                            in Fortran (ncol, nz) index order in/out\n"
        code += "                            (H2D + jitted device wrapper + ONE batched D2H)\n"
        code += "  2. {proc}_bridge_device — device-resident: the jitted wrapper itself;\n"
        code += "                            jax.Arrays already on the device in/out, no\n"
        code += "                            transfers. For callers that keep the model\n"
        code += "                            state on the device across the per-step call\n"
        code += "                            sequence (fetch once at the end).\n"
        code += '"""\n\n'
        code += "import functools\n\n"
        code += "import numpy as np\n"
        code += "import jax\n"
        code += "import jax.numpy as jnp\n\n"

        code += "def to_device(arr):\n"
        code += '    """Host → device unchanged (pure H2D; no host-side permute)"""\n'
        code += "    return jnp.asarray(arr, dtype=jnp.float64)\n\n"

        code += "def to_host(arr):\n"
        code += '    """Device → host NumPy for a SINGLE array (blocking D2H).\n'
        code += "\n"
        code += "    The bridge body does NOT use this per-array path: outputs are\n"
        code += "    fetched together in one batched jax.device_get (one sync total\n"
        code += '    instead of one per array). Kept for tests and interactive use."""\n'
        code += "    return np.array(arr, dtype=np.float64)\n\n"

        return code
    
    # ========================================================================
    # Bridge Function Generation (INOUT PATTERN)
    # ========================================================================
    
    def generate_bridge_function(
        self,
        proc_name: str,
        params: List[ParamInfo],
        parent_module: str = "",
        module_vars_used: List[str] = None,
        is_init: bool = False,
        orig_name: str = "",
        optional_vars: List[str] = None,
    ) -> str:
        """
        Generate bridge function with MODULE vars as INOUT parameters.

        PATH C (device-side layout conversion):
        - Array-bearing procedures get a per-procedure jitted device wrapper
          that calls {proc}_core DIRECTLY: arrays ship to the device
          unchanged (pure H2D), rank>=2 axis reversal happens inside the
          jitted region on both sides (XLA fuses the permute into the
          kernel). CHARACTER args stay host-side — strings cannot cross the
          jit boundary — and pass through the bridge unchanged; the
          translated wrapper's host-side string/validation logic does not
          run on this path (it is not jit-traceable: Python branching on
          traced values, int() concretization, string returns).
        - Scalar-only procedures keep the direct call to the translated
          wrapper (no layout work exists; string handling fully preserved).
        - Bridge public signature and returns are UNCHANGED either way.

        INOUT Pattern (unchanged):
        - MODULE vars appear in bridge signature
        - Driver passes MODULE vars explicitly
        - Bridge returns updated MODULE vars
        - No hidden state dictionary

        Args:
            proc_name: Procedure name
            params: List of parameter info
            parent_module: Parent MODULE name (for documentation)
            module_vars_used: MODULE variables used by procedure
            is_init: True if this is an initialization procedure

        Returns:
            Generated bridge function code
        """
        module_vars_used = module_vars_used or []
        optional_vars = optional_vars or []
        has_module_vars = bool(module_vars_used and parent_module)

        has_arrays = any(p.rank > 0 for p in params)
        output_params = [p for p in params if p.is_output]

        # Translated modules/functions keep the original Fortran casing
        # (out/jax/G_of_mu.py defines G_of_mu) — import by that name and
        # alias to the lowercase name the bridge code uses.
        src_name = orig_name or proc_name

        # Fortran names become Python identifiers from here down, so every one
        # is routed through safe_py_name (SCALE's `is` is a Python keyword —
        # see framework_config). Emission-time only: all matching logic above
        # stays on the raw Fortran names.
        _py = safe_py_name
        mvars_py = [_py(v) for v in module_vars_used]
        optvars_py = [_py(v) for v in optional_vars] if optional_vars else []

        # Main bridge function signature (WITH module vars as parameters!)
        func_args = [_py(p.name) for p in params]
        if has_module_vars:
            func_args.extend(mvars_py)  # Add MODULE vars to signature
        if optional_vars:
            # Pass-through kwargs (e.g. LUT tables); None = stub mode
            func_args.extend(f"{v}=None" for v in optvars_py)

        def module_docstring_block() -> str:
            if not has_module_vars:
                return ""
            proc_type = "INIT" if is_init else "COMPUTE"
            s = f"\n    MODULE VARIABLES (from {parent_module}):\n"
            for var in module_vars_used:
                s += f"      - {var} (INOUT)\n"
            s += "    \n"
            s += f"    Procedure type: {proc_type}\n"
            s += f"    Pattern: MODULE vars passed as INOUT parameters\n"
            return s

        # --------------------------------------------------------------------
        # Scalar-only path: no layout work exists — call the translated
        # wrapper directly (host-side string/validation logic fully preserved)
        # --------------------------------------------------------------------
        if not has_arrays:
            if src_name != proc_name:
                code = f"from out.jax.{src_name} import {src_name} as {proc_name}\n\n"
            else:
                code = f"from out.jax.{proc_name} import {proc_name}\n\n"

            code += f"def {proc_name}_bridge({', '.join(func_args)}):\n"
            code += f'    """\n'
            code += f"    Production bridge for {proc_name} (scalar-only).\n\n"
            code += f"    No array arguments — no layout conversion exists; calls the\n"
            code += f"    translated wrapper directly (strings and Python control flow\n"
            code += f"    stay host-side).\n"
            code += module_docstring_block()
            code += f'    """\n'

            call_parts = [_py(p.name) for p in params]
            if has_module_vars:
                call_parts.extend(mvars_py)
            if optional_vars:
                # Pass-through kwargs: forwarded as keywords, never returned
                call_parts.extend(f"{v}={v}" for v in optvars_py)
            call_args = ", ".join(call_parts)

            returns = [f"{_py(p.name)}_out" for p in output_params]
            if has_module_vars:
                returns.extend(mvars_py)

            if returns:
                code += f"    {', '.join(returns)} = {proc_name}({call_args})\n"
                code += f"\n    return {', '.join(returns)}\n"
            else:
                code += f"    {proc_name}({call_args})\n"
                code += "    return None\n"
            return code

        # --------------------------------------------------------------------
        # Array-bearing path: PATH C — pure H2D/D2H + in-jit axis reversal
        # around {proc}_core (exactly the measured device-transpose wrapper;
        # PBS 6834200/6834351, 2.38x @1e6 bit-identical)
        # --------------------------------------------------------------------
        if src_name != proc_name:
            code = (f"from out.jax.{src_name} import "
                    f"{src_name}_core as {proc_name}_core\n\n")
        else:
            code = f"from out.jax.{proc_name} import {proc_name}_core\n\n"

        # Core contract (jax_jit_boundary_rules.md + signature policy): core
        # args = bridge args minus CHARACTER params, order preserved; core
        # returns = written non-CHARACTER params in arg order, then MODULE
        # vars.
        nonchar = [p for p in params if p.dtype != "character"]
        has_char = len(nonchar) != len(params)

        # Statics: every integer/logical scalar — a SUPERSET of the core's
        # own static_argnames (integers only, per jax_jit_boundary_rules.md).
        # Over-marking is safe FOR CORRECTNESS: a concrete value is valid
        # anywhere inside the trace. Under-marking is fatal: a value traced
        # here that reaches one of the core's statics fails to trace.
        # EXCEPTION — per-step-varying scalars (timestep counters it/kount/
        # itimestep): a static that changes every call re-specializes and
        # recompiles the kernel every call (~18 s/step observed), so those
        # stay traced (framework_config.is_per_step_scalar; extend via
        # [heuristics].per_step_scalar_names). Numeric (real) scalars stay
        # traced. Drivers must pass host scalars (int/bool) for statics —
        # they are Fortran scalar arguments, so they always are.
        statics = [_py(p.name) for p in params
                   if p.rank == 0 and p.dtype in ("integer", "logical")
                   and not get_config().is_per_step_scalar(p.name)]

        # PUBLIC device-resident entry (second bridge contract, 2026-08-19):
        # {proc}_bridge_device is the jitted wrapper itself — device arrays
        # in, device arrays out, no H2D/D2H — so a caller that keeps the
        # model state on the device across the per-step call sequence pays
        # the transfer once (or never) instead of once per procedure. The
        # host-facing {proc}_bridge below is built ON it and is unchanged.
        # _{proc}_device stays as an alias for the pre-08-19 private name.
        dev_name = f"{proc_name}_bridge_device"
        dev_alias = f"_{proc_name}_device"
        dev_args = [_py(p.name) for p in nonchar] + mvars_py
        dev_sig = list(dev_args)
        if optional_vars:
            dev_sig.extend(f"{v}=None" for v in optvars_py)

        core_rets = [f"{_py(p.name)}" for p in nonchar if p.is_output] + mvars_py

        code += "# Device-side layout wrapper (PATH C): arrays arrive in Fortran\n"
        code += "# (ncol, nz) index order; axes are reversed INSIDE the jitted region\n"
        code += "# so the permute runs on the device and XLA can fuse it into the\n"
        code += "# kernel. CHARACTER args never cross this boundary (not JAX types).\n"
        code += "# PUBLIC device-resident entry — see its docstring.\n"
        if statics:
            statics_tuple = "(" + ", ".join(f"'{s}'" for s in statics) + ",)"
            code += f"@functools.partial(jax.jit, static_argnames={statics_tuple})\n"
        else:
            code += "@jax.jit\n"
        code += f"def {dev_name}({', '.join(dev_sig)}):\n"
        code += f'    """\n'
        code += f"    DEVICE-RESIDENT bridge entry for {proc_name} (contract 2, no transfers).\n\n"
        code += f"    Same arguments as {proc_name}_bridge minus CHARACTER params, in the\n"
        code += f"    same order; array arguments are jax.Arrays ALREADY ON THE DEVICE in\n"
        code += f"    Fortran (ncol, nz) index order (the layout to_device(host) produces);\n"
        code += f"    integer/logical scalars are static Python values; MODULE vars are\n"
        code += f"    passed and returned (INOUT pattern). Returns the written non-CHARACTER\n"
        code += f"    params in argument order, then MODULE vars — as device arrays, left\n"
        code += f"    on the device (no device_get). Chain these entries across the\n"
        code += f"    scheme's own per-step phase sequence and fetch once at the end;\n"
        code += f"    the outputs are bit-identical to {proc_name}_bridge's (same jitted code — the\n"
        code += f"    host bridge is this function wrapped in to_device / device_get).\n"
        code += f'    """\n'

        rev_in = [p for p in nonchar if p.rank >= 2]
        if rev_in:
            code += "    # In-jit input reversal: reverse ALL axes on rank>=2\n"
            code += "    # (covers rank 3+; rank 0/1 pass through)\n"
            for p in rev_in:
                code += f"    {_py(p.name)} = jnp.transpose({_py(p.name)})\n"

        core_call_parts = list(dev_args)
        if optional_vars:
            # Pass-through kwargs: forwarded as keywords, never returned
            core_call_parts.extend(f"{v}={v}" for v in optvars_py)
        core_call = ", ".join(core_call_parts)

        if core_rets:
            code += f"    {', '.join(core_rets)} = {proc_name}_core({core_call})\n"
        else:
            code += f"    {proc_name}_core({core_call})\n"

        rev_out = [p for p in nonchar if p.is_output and p.rank >= 2]
        if rev_out:
            code += "    # In-jit output reversal: back to Fortran (ncol, nz) order\n"
            for p in rev_out:
                code += f"    {_py(p.name)} = jnp.transpose({_py(p.name)})\n"

        if core_rets:
            code += f"    return {', '.join(core_rets)}\n\n"
        else:
            code += "    return None\n\n"

        # Back-compat alias for the pre-2026-08-19 private name
        code += f"{dev_alias} = {dev_name}  # alias (private name kept for compatibility)\n\n"

        # Main bridge function (public signature/returns UNCHANGED)
        code += f"def {proc_name}_bridge({', '.join(func_args)}):\n"
        code += f'    """\n'
        code += f"    Production HOST-FACING bridge for {proc_name} (contract 1).\n\n"
        code += f"    Device-resident callers use {proc_name}_bridge_device instead\n"
        code += f"    (contract 2: device arrays in/out, no transfers); this function is\n"
        code += f"    exactly to_device -> {proc_name}_bridge_device -> batched device_get.\n\n"
        code += f"    Strategy (PATH C): ship arrays to the device unchanged (pure H2D),\n"
        code += f"    reverse axes inside the jitted device wrapper (fused by XLA), call\n"
        code += f"    {proc_name}_core, reverse back in-jit, pure D2H. Inputs and outputs\n"
        code += f"    stay standard C-order NumPy in Fortran (ncol, nz) index order.\n"
        if has_char:
            code += f"\n    CHARACTER args stay host-side: this path calls the jitted core\n"
            code += f"    directly, so wrapper-level string handling does not run here;\n"
            code += f"    CHARACTER inputs/outputs pass through the bridge unchanged.\n"
        code += module_docstring_block()
        code += f'    """\n'

        # H2D — no host-side permute
        code += "    # Ship arrays to the device unchanged (pure H2D, no host permute)\n"
        for p in nonchar:
            if p.rank > 0:
                code += f"    {_py(p.name)}_dev = to_device({_py(p.name)})\n"
        code += "\n"

        # Call the device wrapper with keywords (statics need names)
        call_kwargs = []
        for p in nonchar:
            pn = _py(p.name)
            call_kwargs.append(f"{pn}={pn}_dev" if p.rank > 0 else f"{pn}={pn}")
        call_kwargs.extend(f"{v}={v}" for v in mvars_py)
        if optional_vars:
            call_kwargs.extend(f"{v}={v}" for v in optvars_py)

        unpack = [f"{_py(p.name)}_out" for p in nonchar if p.is_output] + mvars_py

        code += "    # Compute — in-jit layout conversion + jitted core\n"
        if unpack:
            code += (f"    {', '.join(unpack)} = "
                     f"{dev_name}({', '.join(call_kwargs)})\n")
        else:
            code += f"    {dev_name}({', '.join(call_kwargs)})\n"

        # Output conversion — pure D2H; CHARACTER outputs pass through the
        # bridge unchanged (input values) to keep return arity/positions.
        if not output_params and not has_module_vars:
            code += "\n    return None\n"
            return code

        # ONE batched D2H for every device-resident output (arrays and
        # numeric scalars): jax.device_get on the tuple starts all copies
        # and synchronizes once, instead of one blocking round-trip per
        # array (np.array(jax_array) syncs per call — measured 16 stream
        # syncs/call on Kessler). Host-side casts below are views, not
        # copies, when dtype/layout already match.
        dget = [_py(p.name) for p in output_params if p.dtype != "character"]
        code += "\n    # Convert outputs — ONE batched D2H (single sync); layout already\n"
        code += "    # Fortran (ncol, nz) C-order\n"
        if dget:
            lhs = ", ".join(f"{pn}_host" for pn in dget)
            rhs = ", ".join(f"{pn}_out" for pn in dget)
            if len(dget) == 1:
                code += f"    ({lhs},) = jax.device_get(({rhs},))\n"
            else:
                code += f"    {lhs} = jax.device_get(({rhs}))\n"
        fortran_returns = []
        for p in output_params:
            pn = _py(p.name)
            if p.dtype == "character":
                fortran_returns.append(pn)  # host passthrough, core never saw it
            elif p.rank == 0:
                code += f"    {pn}_fortran = np.asarray({pn}_host).item()\n"
                fortran_returns.append(f"{pn}_fortran")
            else:
                code += f"    {pn}_fortran = np.asarray({pn}_host, dtype=np.float64)\n"
                fortran_returns.append(f"{pn}_fortran")

        # Add MODULE vars (always returned in INOUT pattern)
        if has_module_vars:
            fortran_returns.extend(mvars_py)

        code += f"\n    return {', '.join(fortran_returns)}\n"

        return code
    
    # ========================================================================
    # Analysis Report (UNCHANGED - for brevity, keeping simple version)
    # ========================================================================
    
    def generate_analysis_report(
        self,
        proc_name: str,
        merged_packet: Dict,
        params: List[ParamInfo]
    ) -> str:
        """Generate comprehensive analysis report"""
        
        phase2_packet = merged_packet.get("phase2_packet", {})
        fortran_source = phase2_packet.get("fortran_source", "")
        
        safety = self.safety_analyzer.analyze(fortran_source, proc_name)
        
        report = []
        report.append("=" * 70)
        report.append(f"BRIDGE ANALYSIS: {proc_name}")
        report.append("=" * 70)
        report.append("")
        report.append("STRATEGY: PATH C device-side layout conversion (pure H2D/D2H, "
                      "in-jit axis reversal); JIT via emitted device wrapper + _core decorator")
        report.append("")
        
        # Safety analysis
        report.append("LOOP SWAP SAFETY ANALYSIS")
        report.append("-" * 70)
        report.append(safety.report())
        report.append("")
        
        # Performance analysis
        inefficient, warning = self.perf_analyzer.analyze_gpu_efficiency(params)
        if inefficient:
            report.append("GPU EFFICIENCY WARNING")
            report.append("-" * 70)
            report.append(warning)
            report.append("")
        
        return "\n".join(report)
    
    # ========================================================================
    # Module Generation (SIMPLIFIED - No state dictionary)
    # ========================================================================
    
    def generate_bridge_module(
        self,
        proc_name: str,
        merged_packet: Dict,
        params: List[ParamInfo],
        orig_name: str = "",
    ):
        """
        Generate complete bridge module (INOUT pattern - no state dictionary).

        Writes:
            - out/bridge/{proc_name}_bridge.py
        """
        parent_module, module_vars_used = self.extract_module_info(merged_packet)

        # A module var that is also a dummy argument is already in the
        # signature — appending it again would emit a duplicate parameter
        # (a Python SyntaxError).
        param_names = {p.name.lower() for p in params}
        module_vars_used = [v for v in module_vars_used
                            if v.lower() not in param_names]

        # Vars configured as pass-through kwargs are pulled out of the INOUT
        # list; the config list is authoritative (it may name vars the
        # dependency analysis missed).
        optional_vars = self.optional_kwargs.get(proc_name.lower(), [])
        optional_lower = {v.lower() for v in optional_vars}
        module_vars_used = [v for v in module_vars_used
                            if v.lower() not in optional_lower]

        deps = merged_packet.get("deps", {})
        writes = deps.get("writes", [])
        is_init = self.is_initialization_procedure(proc_name, module_vars_used, writes)
        proc_type = "INIT" if is_init else "COMPUTE"
        
        # Header docstring
        code = '"""\n'
        code += f"Bridge: {proc_name}\n"
        code += "Strategy: PATH C device-side layout conversion — pure H2D/D2H,\n"
        code += "in-jit axis reversal fused by XLA; public contract unchanged\n"
        
        if module_vars_used and parent_module:
            code += f"\nMODULE: {parent_module}\n"
            code += f"MODULE vars: {', '.join(module_vars_used)}\n"
            code += f"Procedure type: {proc_type}\n"
        if optional_vars:
            code += f"Optional pass-through kwargs (None = stub mode): {', '.join(optional_vars)}\n"
        
        code += "\nMODULE_VARIABLE_POLICY (INOUT Pattern):\n"
        code += "- MODULE vars passed as function parameters (INOUT)\n"
        code += "- Driver manages MODULE state explicitly\n"
        code += "- No hidden state dictionary\n"
        code += "- INIT: Returns updated MODULE vars to driver\n"
        code += "- COMPUTE: Returns updated MODULE vars to driver\n"
        code += '"""\n\n'
        
        # NO state dictionary generation!
        
        # Conversion functions
        code += self.generate_conversion_functions(params)
        
        # Bridge function
        code += self.generate_bridge_function(
            proc_name,
            params,
            parent_module,
            module_vars_used,
            is_init,
            orig_name=orig_name,
            optional_vars=optional_vars,
        )
        
        # Write bridge file
        bridge_file = self.bridge_dir / f"{proc_name}_bridge.py"
        bridge_file.write_text(code, encoding="utf-8")
        
        print(f"   ✅ Bridge: {bridge_file.name}")
        
        # Write analysis report
        report = self.generate_analysis_report(proc_name, merged_packet, params)
        report_file = self.reports_dir / f"{proc_name}_analysis.txt"
        report_file.write_text(report, encoding="utf-8")
        
        print(f"   📊 Report: {report_file.name}")
    
    # ========================================================================
    # Main Processing
    # ========================================================================
    
    def process_all(self):
        """Process all procedures and generate bridge code"""
        
        if not self.phase1_data:
            raise RuntimeError("Phase 1 index not loaded")
        
        procedures = self.phase1_data.get("procedures", [])
        
        if not procedures:
            raise RuntimeError("No procedures found in phase1_index.json")
        
        print(f"📂 Found {len(procedures)} procedures to process")
        
        for proc_data in procedures:
            orig_name = proc_data.get("name", "")
            proc_name = orig_name.lower()

            if not proc_name:
                continue
            
            print(f"📦 Generating bridge: {proc_name}")
            
            # Load merged packet (case-insensitive fallback for mixed-case Fortran names)
            packet_file = self.packets_dir / f"{proc_name}_merged.json"
            if not packet_file.exists():
                target = f"{proc_name}_merged.json".lower()
                matches = [f for f in self.packets_dir.glob("*_merged.json")
                           if f.name.lower() == target]
                if matches:
                    packet_file = matches[0]
                else:
                    print(f"  ⚠️  Merged packet not found: {packet_file}")
                    continue
            
            merged_packet = self.load_merged_packet(packet_file)
            
            # Extract module info
            parent_module, module_vars_used = self.extract_module_info(merged_packet)
            
            if module_vars_used and parent_module:
                proc_type = "INIT" if self.is_initialization_procedure(
                    proc_name, module_vars_used, merged_packet.get("deps", {}).get("writes", [])
                ) else "COMPUTE"
                print(f"   MODULE vars: {', '.join(module_vars_used)} (from {parent_module}) [{proc_type}]")
            
            # Extract parameters
            params = self.extract_parameters(merged_packet)
            print(f"   Parameters: {len(params)}")
            
            for p in params:
                ptype = "output" if p.is_output else "input"
                rank_str = "scalar" if p.rank == 0 else f"{p.rank}D"
                print(f"     - {p.name}: {p.dtype} {rank_str} ({ptype})")
            
            # Generate bridge
            self.generate_bridge_module(proc_name, merged_packet, params,
                                        orig_name=orig_name)


def main():
    """Main entry point"""
    
    print("=" * 70)
    print("Phase 03: Bridge Code Generation (ENHANCED)")
    print("LAFT Bridge Generation")
    print("MODULE_VARIABLE_POLICY: INOUT Pattern (Option A)")
    print("=" * 70)
    
    cfg = get_config()
    generator = BridgeGenerator(
        phase1_index=str(cfg.phase1_index),
        packets_dir=str(cfg.packets_dir),
        jax_dir=str(cfg.jax_dir),
        bridge_dir=str(cfg.bridge_dir),
        reports_dir=str(cfg.bridge_reports_dir),
        optional_kwargs=cfg.section("bridge").get("optional_kwargs", {}),
    )

    # Check that phase1_index exists
    if not generator.phase1_data:
        print("❌ Cannot proceed without phase1_index.json")
        print("   Run Phase 01 first: python workflow_frontend/phase01_01_ts_parse.py")
        return 1
    
    procedures = generator.phase1_data.get("procedures", [])
    print(f"✅ Loaded {len(procedures)} procedures from phase1_index.json")
    
    # Process all procedures
    generator.process_all()
    
    print("=" * 70)
    print("🎉 BRIDGE GENERATION COMPLETE")
    print("=" * 70)
    print(f"   Bridges: {generator.bridge_dir}")
    print(f"   Reports: {generator.reports_dir}")
    print(f"   Strategy: PATH C device-side layout conversion (pure H2D/D2H, in-jit axis reversal)")
    print(f"   MODULE_VARIABLE_POLICY: INOUT Pattern")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    import argparse
    from framework_config import init as init_config, add_config_arg

    ap = argparse.ArgumentParser(description="Phase 03: bridge code generation")
    add_config_arg(ap)
    init_config(ap.parse_args().config)
    exit(main())
