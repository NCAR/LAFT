#!/usr/bin/env python3
"""
Phase 02_04: Module Dependency Analysis

Analyzes Fortran modules to determine:
- Module dependency graph (which modules USE which)
- Translation order (topological sort)
- Dependency levels (for parallel translation)
- Circular dependency detection

Output: out/module_dependencies.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict, deque

# framework_config lives in LAFT/config/; this script now lives in
# workflow_frontend/ (symlinked from LAFT/ into each project) — resolve back
# to LAFT/config to import it. Same convention as workflow_bridge/phase03_make_bridge.py.
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "config"))

from framework_config import get_config


def load_phase01_modules(modules_dir: Path = Path("out/modules")) -> List[Dict[str, Any]]:
    """
    Load all module metadata from Phase 01
    
    Returns:
        List of module metadata dictionaries
    """
    modules = []
    
    if not modules_dir.exists():
        print(f"⚠️  Module directory not found: {modules_dir}")
        return modules
    
    for module_file in sorted(modules_dir.glob("*.json")):
        try:
            with open(module_file, 'r') as f:
                module_data = json.load(f)
                modules.append(module_data)
        except Exception as e:
            print(f"⚠️  Error loading {module_file}: {e}")
    
    return modules


def build_dependency_graph(modules: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """
    Build module dependency graph
    
    Args:
        modules: List of module metadata
        
    Returns:
        Dict mapping module_name -> set of dependencies
    """
    graph = {}
    
    for module in modules:
        module_name = module.get('module_name', '')
        uses_modules = module.get('uses_modules', [])
        
        if not module_name:
            continue
        
        # Only include dependencies that are in our translation set
        all_module_names = {m.get('module_name', '') for m in modules}
        internal_deps = {dep for dep in uses_modules if dep in all_module_names}
        
        graph[module_name] = internal_deps
    
    return graph


def detect_circular_dependencies(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """
    Detect circular dependencies in the graph
    
    Returns:
        List of cycles (each cycle is a list of module names)
    """
    cycles = []
    
    def dfs(node: str, path: List[str], visited: Set[str], rec_stack: Set[str]):
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                dfs(neighbor, path.copy(), visited, rec_stack)
            elif neighbor in rec_stack:
                # Found a cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                cycles.append(cycle)
        
        rec_stack.remove(node)
    
    visited = set()
    for node in graph:
        if node not in visited:
            dfs(node, [], visited, set())
    
    return cycles


def topological_sort(graph: Dict[str, Set[str]]) -> List[str]:
    """
    Perform topological sort on dependency graph
    
    Returns:
        List of module names in translation order
        
    Raises:
        ValueError: If circular dependencies detected
    """
    # Check for cycles first
    cycles = detect_circular_dependencies(graph)
    if cycles:
        cycle_strs = [' → '.join(cycle) for cycle in cycles]
        raise ValueError(f"Circular dependencies detected:\n" + "\n".join(cycle_strs))
    
    # Kahn's algorithm for topological sort.
    #
    # graph[node] is the set of modules `node` DEPENDS ON, so a node's in-degree
    # is the count of its own (in-graph) dependencies — not the count of modules
    # that depend on it. Incrementing the dependency's counter instead inverts
    # the edge direction and strands every module that anything else uses.
    # Single-module projects have in-degree 0 under either reading, which is why
    # this only surfaces once a project lists more than one module.
    in_degree = {
        node: sum(1 for dep in graph[node] if dep in graph)
        for node in graph
    }

    # Queue of nodes with no dependencies (sorted for reproducible output —
    # graph values are sets, so insertion order alone is not stable)
    queue = deque(sorted(node for node in graph if in_degree[node] == 0))
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        # For each node that depends on current node
        for other_node in graph:
            if node in graph[other_node]:
                in_degree[other_node] -= 1
                if in_degree[other_node] == 0:
                    queue.append(other_node)
    
    # Check if all nodes were processed
    if len(result) != len(graph):
        missing = set(graph.keys()) - set(result)
        raise ValueError(f"Could not sort all nodes. Missing: {missing}")
    
    return result


def compute_dependency_levels(graph: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """
    Compute dependency levels for parallel translation
    
    Level 0: No dependencies
    Level 1: Only depends on level 0
    Level 2: Depends on level 0 or 1
    etc.
    
    Returns:
        Dict mapping "level_N" -> list of module names
    """
    levels = defaultdict(list)
    module_levels = {}
    
    # Helper to compute level for a module
    def get_level(module: str, visited: Set[str]) -> int:
        if module in module_levels:
            return module_levels[module]
        
        if module in visited:
            # Shouldn't happen if no cycles, but be safe
            return 0
        
        visited.add(module)
        deps = graph.get(module, set())
        
        if not deps:
            level = 0
        else:
            # Level is 1 + max level of dependencies
            dep_levels = [get_level(dep, visited.copy()) for dep in deps if dep in graph]
            level = 1 + max(dep_levels) if dep_levels else 0
        
        module_levels[module] = level
        return level
    
    # Compute level for each module
    for module in graph:
        level = get_level(module, set())
        levels[f"level_{level}"].append(module)
    
    # Sort modules within each level
    for level_key in levels:
        levels[level_key] = sorted(levels[level_key])
    
    return dict(levels)


def analyze_module_dependencies(
    modules_dir: Path = Path("out/modules"),
    packets_dir: Path = Path("out/packets"),
    output_file: Path = Path("out/module_dependencies.json")
) -> Dict[str, Any]:
    """
    Main function: Analyze module dependencies
    
    Returns:
        Complete dependency analysis data
    """
    print("=" * 70)
    print("Phase 02_04: Module Dependency Analysis")
    print("=" * 70)
    
    # Load Phase 01 module metadata
    print("\n[1/5] Loading Phase 01 module metadata...")
    modules = load_phase01_modules(modules_dir)
    
    if not modules:
        print("⚠️  No modules found. Creating empty dependency file.")
        empty_result = {
            "modules": [],
            "dependency_graph": {},
            "translation_order": [],
            "dependency_levels": {},
            "circular_dependencies": [],
            "external_dependencies": []
        }
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(empty_result, f, indent=2)
        return empty_result
    
    print(f"   Found {len(modules)} modules")
    
    # Load procedure metadata from packets to enrich module info
    print("\n[2/5] Enriching with procedure information...")
    module_procedures = defaultdict(list)
    
    # Read from packets directory (Phase 02_01 output)
    if packets_dir.exists():
        for packet_file in sorted(packets_dir.glob("*.json")):
            # Skip deps, merged, and summary files
            if packet_file.name.endswith("_deps.json") or \
               packet_file.name.endswith("_merged.json") or \
               packet_file.name == "_ALL_deps.json":
                continue
            
            try:
                with open(packet_file, 'r') as f:
                    packet_data = json.load(f)
                    parent_module = packet_data.get('parent_module')
                    proc_name = packet_data.get('proc_name')
                    if parent_module and proc_name:
                        module_procedures[parent_module].append(proc_name)
            except Exception as e:
                print(f"⚠️  Error loading {packet_file}: {e}")
    
    # Build dependency graph
    print("\n[3/5] Building dependency graph...")
    graph = build_dependency_graph(modules)
    print(f"   Graph has {len(graph)} nodes")
    
    # Detect circular dependencies
    print("\n[4/5] Checking for circular dependencies...")
    cycles = detect_circular_dependencies(graph)
    
    if cycles:
        print("   ❌ Circular dependencies detected:")
        for cycle in cycles:
            print(f"      {' → '.join(cycle)}")
    else:
        print("   ✅ No circular dependencies")
    
    # Compute translation order (only if no cycles)
    translation_order = []
    dependency_levels = {}
    
    if not cycles:
        print("\n[5/5] Computing translation order...")
        try:
            translation_order = topological_sort(graph)
            print(f"   Translation order: {' → '.join(translation_order)}")
            
            dependency_levels = compute_dependency_levels(graph)
            print(f"   Dependency levels: {len(dependency_levels)}")
            for level_key in sorted(dependency_levels.keys()):
                modules_in_level = dependency_levels[level_key]
                print(f"      {level_key}: {', '.join(modules_in_level)}")
        except ValueError as e:
            print(f"   ❌ Error computing order: {e}")
    else:
        print("\n[5/5] Skipping translation order (circular dependencies present)")
    
    # Identify external dependencies
    all_module_names = set(graph.keys())
    external_deps = set()
    
    for module_data in modules:
        uses = module_data.get('uses_modules', [])
        for dep in uses:
            if dep not in all_module_names:
                external_deps.add(dep)
    
    # Build enriched module list
    enriched_modules = []
    for module_data in modules:
        module_name = module_data.get('module_name', '')
        
        # Compute level
        level = -1
        for level_key, mods in dependency_levels.items():
            if module_name in mods:
                level = int(level_key.split('_')[1])
                break
        
        enriched = {
            'name': module_name,
            'file': module_data.get('source_file', ''),
            'uses': sorted(list(graph.get(module_name, set()))),
            'level': level,
            'procedures': sorted(module_procedures.get(module_name, [])),
            'module_variables': module_data.get('module_variables', [])
        }
        enriched_modules.append(enriched)
    
    # Build result
    result = {
        "modules": enriched_modules,
        "dependency_graph": {k: sorted(list(v)) for k, v in graph.items()},
        "translation_order": translation_order,
        "dependency_levels": dependency_levels,
        "circular_dependencies": [' → '.join(cycle) for cycle in cycles],
        "external_dependencies": sorted(list(external_deps)),
        "can_parallelize": {
            level_key: len(modules) > 1
            for level_key, modules in dependency_levels.items()
        }
    }
    
    # Save to file
    print(f"\n💾 Saving dependency analysis to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✅ Module dependency analysis complete!")
    print("=" * 70)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Modules analyzed: {len(enriched_modules)}")
    print(f"  Translation order: {len(translation_order)} modules")
    print(f"  Dependency levels: {len(dependency_levels)}")
    print(f"  External dependencies: {len(external_deps)}")
    if external_deps:
        print(f"    {', '.join(sorted(external_deps))}")
    print(f"  Circular dependencies: {len(cycles)}")
    
    return result


def generate_call_graph_svg(
    deps_file: Path = Path("out/packets/_ALL_deps.json"),
    output_svg: Path = Path("out/procedure_call_graph.svg"),
) -> None:
    """
    Generate a simple SVG of the procedure call graph.
    Procedures are laid out in rows by call depth.
    Roots are computed dynamically (procedures not called by anyone).
    Pure Python — no external packages required.
    """
    if not deps_file.exists():
        print(f"⚠️  {deps_file} not found, skipping SVG generation.")
        return

    with open(deps_file) as f:
        all_deps = json.load(f)

    # Case-insensitive name map: lowercase -> original display name
    name_map: Dict[str, str] = {entry["proc_name"].lower(): entry["proc_name"] for entry in all_deps}
    known_lower: Set[str] = set(name_map.keys())

    # Build call graph with original display names; calls list is already lowercase
    calls: Dict[str, Set[str]] = {}
    for entry in all_deps:
        display = entry["proc_name"]
        callees = {name_map[c] for c in entry.get("calls", []) if c in known_lower}
        calls[display] = callees

    # Compute which procedures are called by at least one other procedure
    called_by_someone: Set[str] = set()
    for callees in calls.values():
        called_by_someone |= callees

    # Roots = procedures not called by anyone (true entry points)
    all_names = set(name_map.values())
    roots = all_names - called_by_someone

    # BFS from roots to assign depth levels (callers at top, callees below)
    level_of: Dict[str, int] = {}
    queue = deque()
    for r in sorted(roots):
        level_of[r] = 0
        queue.append(r)

    while queue:
        node = queue.popleft()
        # sorted: set iteration order is hash-seed dependent and decides which
        # parent assigns a shared callee's level — sort for deterministic output
        for callee in sorted(calls.get(node, set())):
            if callee not in level_of:
                level_of[callee] = level_of[node] + 1
                queue.append(callee)

    # Any node not reached (shouldn't happen with correct roots) goes at bottom
    max_level = max(level_of.values(), default=0)
    for name in all_names:
        if name not in level_of:
            level_of[name] = max_level + 1

    # Group by level
    levels: Dict[int, List[str]] = defaultdict(list)
    for name, lvl in level_of.items():
        levels[lvl].append(name)
    for lvl in levels:
        levels[lvl] = sorted(levels[lvl])

    # Layout constants
    NODE_W, NODE_H = 180, 28
    H_GAP, V_GAP = 20, 50
    MARGIN = 30

    max_cols = max(len(v) for v in levels.values())
    total_w = max_cols * (NODE_W + H_GAP) + 2 * MARGIN
    total_h = (max_level + 2) * (NODE_H + V_GAP) + 2 * MARGIN

    # Compute node positions: centre x, centre y
    pos: Dict[str, tuple] = {}
    for lvl, names in levels.items():
        row_w = len(names) * (NODE_W + H_GAP) - H_GAP
        x_start = (total_w - row_w) / 2
        y = MARGIN + lvl * (NODE_H + V_GAP) + NODE_H / 2
        for i, name in enumerate(names):
            x = x_start + i * (NODE_W + H_GAP) + NODE_W / 2
            pos[name] = (x, y)

    # Color by level
    def node_color(lvl: int) -> str:
        palette = ["#4C8BF5", "#34A853", "#FBBC04", "#EA4335",
                   "#8E24AA", "#00ACC1", "#F4511E", "#607D8B"]
        return palette[lvl % len(palette)]

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(total_w)}" height="{int(total_h)}">',
        '<rect width="100%" height="100%" fill="#1E1E2E"/>',
        '<g font-family="monospace" font-size="11">',
    ]

    # Edges first (drawn under nodes)
    svg_lines.append('<g stroke="#555577" stroke-width="1" fill="none" opacity="0.6">')
    for caller, callees in calls.items():
        if caller not in pos:
            continue
        x1, y1 = pos[caller]
        for callee in sorted(callees):  # sorted: set order is hash-seed dependent
            if callee not in pos:
                continue
            x2, y2 = pos[callee]
            svg_lines.append(
                f'<line x1="{x1:.1f}" y1="{y1+NODE_H/2:.1f}" '
                f'x2="{x2:.1f}" y2="{y2-NODE_H/2:.1f}"/>'
            )
    svg_lines.append('</g>')

    # Nodes
    for name, (cx, cy) in pos.items():
        lvl = level_of[name]
        fill = node_color(lvl)
        x, y = cx - NODE_W / 2, cy - NODE_H / 2
        svg_lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" '
            f'rx="5" fill="{fill}" opacity="0.85"/>'
        )
        svg_lines.append(
            f'<text x="{cx:.1f}" y="{cy+4:.1f}" text-anchor="middle" '
            f'fill="white" font-weight="bold">{name}</text>'
        )

    svg_lines += ["</g>", "</svg>"]

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_svg.write_text("\n".join(svg_lines), encoding="utf-8")
    print(f"\n📊 Procedure call graph saved to: {output_svg}")


def main():
    """Entry point for Phase 02_04"""
    try:
        cfg = get_config()
        result = analyze_module_dependencies(
            modules_dir=cfg.modules_dir,
            packets_dir=cfg.packets_dir,
            output_file=cfg.module_deps_file,
        )

        if result.get('circular_dependencies'):
            print("\n⚠️  Warning: Circular dependencies detected!")
            print("   Translation order cannot be determined.")
            print("   Fix circular dependencies before proceeding.")
            return 1

        generate_call_graph_svg(
            deps_file=cfg.all_deps_file,
            output_svg=cfg.call_graph_svg,
        )

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    import argparse
    from framework_config import init as init_config, add_config_arg

    ap = argparse.ArgumentParser(description="Phase 02.4: module dependency analysis")
    add_config_arg(ap)
    init_config(ap.parse_args().config)
    sys.exit(main())
