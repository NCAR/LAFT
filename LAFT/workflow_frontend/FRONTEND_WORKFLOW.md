# Frontend Workflow — build the framework's base representation

You are running the framework's **first stage**: parse the project's Fortran
source and extract the structured representation every downstream workflow
consumes (procedures, modules, dependencies, packets). Nothing else can run
until this stage is green. This document is **codebase-agnostic**: every
project value comes from `config/project.toml` (chiefly `[source]` and the
`[heuristics]` name-sets).

**Working directory: the project root** (the directory containing `config/`).
Run every command from there.

Hand-off gates:
- **Enter** when starting a new project, or after the Fortran source
  (`[source].fortran_files`) changes — the packets are stale and must be
  rebuilt.
- **Exit** to the bridge workflow (`workflow_bridge/BRIDGE_WORKFLOW.md`) once
  the packets and index exist and verify (below).

## Environment note

`phase01_01_ts_parse.py` and `phase02_02_deps_JSON.py` use **tree-sitter** —
run them in the tree-sitter environment (the JAX/validation env does not have
`tree_sitter`). `phase02_01`, `phase02_03`, and `phase02_04` are plain Python.

The tree-sitter env is `ts`, built from `LAFT/envs/ts.yml`. It pins the
compatible pair (tree_sitter 0.20.4 + tree_sitter_languages 1.10.2): with a
newer tree_sitter, `tree_sitter_languages.get_language("fortran")` raises
`TypeError: __init__() takes exactly 1 argument (2 given)`.

## What this stage produces (the "base")

| Artifact | Consumed by |
|---|---|
| `out/procedures/`, `out/modules/`, `out/programs/` | extraction record; phase02 |
| `out/phase1_index.json` | phase03 bridge (arg metadata), phase04, phase05 |
| `out/packets/<proc>.json` → `_deps.json` → `_merged.json` | every generator (bridge, prompts, wrappers) + validators |
| `out/packets/_ALL_deps.json` | translator plan (topological order, pass counts) |
| `out/module_dependencies.json`, `out/procedure_call_graph.svg` | module-graph reference |

## Steps — run in order (each depends on the previous)

Run each script; after each, confirm its output exists and is non-empty before
proceeding. There is **no fix loop** here — this is a deterministic pipeline.
If a script errors, the cause is almost always the environment (tree-sitter
missing) or malformed/unsupported Fortran, not a per-procedure judgment call.

### 1 — Parse (tree-sitter env)

```bash
python workflow_frontend/phase01_01_ts_parse.py
```
Parses `[source].fortran_files`, splits out each procedure/module/program, and
extracts type metadata (rank, dtype, dimensions). Writes `out/procedures/`,
`out/modules/`, `out/programs/`, and `out/phase1_index.json`.

**Multiple source files are supported.** List them all in
`[source].fortran_files`; they are parsed into one merged `phase1_index.json`
(each entry keeps its own `source_file`, and `source_files` /
`syntax_ok_by_file` record the set). Procedure→module matching is done
per-file, since tree-sitter byte offsets restart at 0 in each file.

**Watch the syntax report.** `syntax_ok` is the AND across files, and the
script prints which files failed. Extraction still runs on a failed file —
tree-sitter's Fortran grammar does not model cpp directives (`#include`,
`#ifdef`) or preprocessor function-macros, so a file using them reports errors
while parsing fine everywhere else. Localized errors are normal for such
sources; **spot-check the extracted units of any failing file** before trusting
its packets, since an error node inside a procedure body can truncate it.

### 2 — Base packets

```bash
python workflow_frontend/phase02_01_procs_JSON.py
```
One base packet per procedure (Fortran source, skeleton, OpenMP directives,
parent module, and the parameter-filtered direct module-var references).
Writes `out/packets/<proc>.json`.

### 3 — Dependencies (tree-sitter env)

```bash
python workflow_frontend/phase02_02_deps_JSON.py
```
Per-procedure dependency analysis (args, writes-to-args, effects, calls,
`jax_required`, `scalar_only`). Writes `out/packets/<proc>_deps.json` and the
aggregate `out/packets/_ALL_deps.json`.

### 4 — Merge packets

```bash
python workflow_frontend/phase02_03_merge_packets_JSON.py
```
Merges base + deps and computes the transitive module-var closure in module
**declaration order** (the canonical signature ordering — see
`framework_config.packet_module_vars`). Writes `out/packets/<proc>_merged.json`.

### 5 — Module dependency graph

```bash
python workflow_frontend/phase02_04_module_dependencies.py
```
Builds the module/procedure dependency graph. Writes
`out/module_dependencies.json` and `out/procedure_call_graph.svg`.

## Verify (exit gate)

- `out/phase1_index.json` exists and lists the expected procedures.
- `out/packets/` has a `<proc>_merged.json` for every procedure, and
  `_ALL_deps.json` is non-empty.
- No script reported an error.

When these hold, the base is built — proceed to
`workflow_bridge/BRIDGE_WORKFLOW.md`.

## Notes

- `framework_config.py` (the config loader every workflow imports) lives in the
  shared `config/` dir, not here — it is more fundamental than any one stage.
  These scripts reach it via the standard `SCRIPT_DIR.parent / "config"` shim.
