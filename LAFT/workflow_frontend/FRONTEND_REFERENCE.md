# Frontend Reference — Tree-Sitter Fortran Parsing

> **Where this fits:** stage 0 of the pipeline. `../ORCHESTRATOR.md` owns stage
> sequencing; `FRONTEND_WORKFLOW.md` is the authoritative procedure to follow.
> This document is the deep-dive reference for how the parser works and what it
> emits.

> **Note:** examples in this document come from the Kessler case study; the tool
> is project-agnostic (driven by `config/project.toml`) and works identically in
> every project.

**Tool:** `workflow_frontend/phase01_01_ts_parse.py`  
**Purpose:** Parse Fortran source code and extract metadata for translation pipeline  
**Technology:** Tree-sitter (incremental parsing library)  
**Input:** Fortran source files (`.f90`, `.F90`)  
**Output:** JSON metadata files organized by program unit type  

---

## Table of Contents

1. [Overview](#overview)
2. [What is Tree-Sitter?](#what-is-tree-sitter)
3. [Fortran Program Unit Types](#fortran-program-unit-types)
4. [Input and Output](#input-and-output)
5. [What Gets Extracted](#what-gets-extracted)
6. [Output Directory Structure](#output-directory-structure)
7. [Usage](#usage)
8. [Troubleshooting](#troubleshooting)

---

## Overview

**Phase 01 is the foundation of the entire translation pipeline.** It converts Fortran source code (text) into structured metadata (JSON) that later phases can process programmatically.

### What Phase 01 Does:

```
INPUT:                          PROCESS:                    OUTPUT:
┌─────────────────┐            ┌──────────────┐           ┌─────────────────┐
│ kessler_run.f90 │  ────────> │ Tree-Sitter  │  ───────> │ JSON Metadata   │
│ (Fortran text)  │            │   Parser     │           │ (structured)    │
└─────────────────┘            └──────────────┘           └─────────────────┘
                                                                    │
                                                                    ├─> units/
                                                                    ├─> modules/
                                                                    ├─> procedures/
                                                                    └─> programs/
```

### Why This is Needed:

**Problem:** Fortran is complex - can't use regex or simple text parsing  
**Solution:** Use a proper parser (tree-sitter) that understands Fortran grammar  
**Benefit:** Reliable extraction of procedures, parameters, types, dimensions, etc.

---

## What is Tree-Sitter?

### Quick Explanation:

**Tree-sitter** is a parsing library that builds a **syntax tree** from source code.

**Tree-sitter produces:**
```
subroutine_definition
├── name: "add"
├── parameters
│   ├── parameter: "x"
│   ├── parameter: "y"
│   └── parameter: "result"
├── body
│   ├── declaration
│   │   ├── type: "INTEGER"
│   │   └── variables: ["x", "y", "result"]
│   └── assignment
│       ├── left: "result"
│       └── right: binary_expression
│           ├── left: "x"
│           ├── operator: "+"
│           └── right: "y"
└── end
```

**This tree structure lets us programmatically extract:**
- Subroutine name: `add`
- Parameters: `x, y, result`
- Types: `INTEGER`
- Operations: `result = x + y`

---

## Fortran Program Unit Types

Fortran organizes code into different **program units**. Phase 01 recognizes and separates these:

### 1. Compilation Unit (File)

**What it is:** The entire `.f90` file

**Saved to:** `out/units/kessler_run.json`

**Contains:** File-level metadata (what modules/procedures are in this file)

---

### 2. Module

**What it is:** A container for related procedures and data

**Saved to:** `out/modules/module_name.json`

**Contains:**
- Module name
- Module variables (global state)
- Public/private entities
- List of procedures inside module

**Why modules matter:** Module variables = global state → Critical for JAX translation!

---

### 3. Procedure (Subroutine or Function)

**What it is:** A callable routine

**Two types:**

#### A. Subroutine (no return value)

#### B. Function (returns a value)

**Saved to:** `out/procedures/procedure_name.json`

**Contains:**
- Procedure name
- Procedure kind (subroutine vs function)
- Parameters (name, type, intent, rank)
- Return value (for functions)
- Skeleton (structure without body)

---

### 4. Program (Main Entry Point)

**What it is:** Executable entry point (like `main()` in C)

**Saved to:** `out/programs/my_program.json`

**Contains:**
- Program name
- Modules used
- Procedures called

**Note:** Physics libraries rarely have PROGRAM units (they're libraries, not executables)

---

## Input and Output

### Input: Fortran Source Files

**Location:** `data/src/*.f90`

**Content:** Standard Fortran 90/95/2003/2008 code

---

### Output: JSON Metadata Files

**Location:** `out/`

**Directory structure:**
```
out/
├── units/              # File-level metadata
│   └── kessler_run.json
│
├── modules/            # Module-level metadata
│   └── kessler.json
│
├── procedures/         # Procedure-level metadata
│   ├── kessler_init.json
│   └── kessler_run.json
│
└── programs/           # Program-level metadata (often empty)
```

---

## What Gets Extracted

- Parameter types and ranks (scalar, 1D, 2D)
- INTENT (in, out, inout) - determines data flow
- Dimensions - needed for array sizing
- OpenMP directives - for parallelization info

---

## Output Directory Structure

```
out/
│
├── units/                          # File-level metadata
│   ├── kessler_init.json          # Info about kessler_init.f90 file
│   └── kessler_run.json           # Info about kessler_run.f90 file
│
├── modules/                        # Module-level metadata
│   └── kessler.json               # Info about MODULE kessler
│                                  # - Module variables (lv, pref, rhoqr)
│                                  # - Public/private entities
│                                  # - Contained procedures
│
├── procedures/                     # Procedure-level metadata (MOST IMPORTANT)
│   ├── kessler_init.json          # SUBROUTINE kessler_init details
│   │                              # - Parameters with types/intents
│   │                              # - Used for bridge generation
│   │
│   └── kessler_run.json           # SUBROUTINE kessler_run details
│                                  # - All parameter metadata
│                                  # - Array dimensions
│                                  # - OpenMP directives
│
└── programs/                       # Main program metadata (rarely used)
    └── (usually empty for libraries)
```

---

## Usage

### Basic Usage:

```bash
cd kessler

# Run Phase 01
python3 workflow_frontend/phase01_01_ts_parse.py

# Output:
# Parsing: data/src/kessler_init.f90
# Parsing: data/src/kessler_run.f90
# 
# ✅ Parsed 2 files
# ✅ Found 1 module: kessler
# ✅ Found 2 procedures: kessler_init, kessler_run
# ✅ Metadata saved to out/
```

### Output Files Created:

```bash
# Check what was created
ls out/units/
# kessler_init.json  kessler_run.json

ls out/modules/
# kessler.json

ls out/procedures/
# kessler_init.json  kessler_run.json

ls out/programs/
# (usually empty for libraries)
```

### Inspect Generated Metadata:

```bash
# View procedure metadata
cat out/procedures/kessler_run.json | jq .

# View module metadata (shows module variables!)
cat out/modules/kessler.json | jq .

# Check parameter types
cat out/procedures/kessler_run.json | jq '.parameters[] | {name, type, intent, rank}'
```

---

## Advanced Features

### 1. OpenMP Directive Detection

**Fortran code:**
```fortran
!$OMP PARALLEL DO PRIVATE(i,j) SHARED(ncol,nz)
DO i = 1, ncol
  DO j = 1, nz
    result(i,j) = compute(i,j)
  END DO
END DO
!$OMP END PARALLEL DO
```

**Extracted:**
```json
{
  "openmp_directives": [
    {
      "type": "omp parallel do",
      "clauses": ["private(i,j)", "shared(ncol,nz)"],
      "location": "line 42"
    }
  ]
}
```

---

### 2. Array Dimension Extraction

**Fortran code:**
```fortran
REAL, INTENT(IN) :: temperature(ncol, nz)
```

**Extracted:**
```json
{
  "name": "temperature",
  "type": "real",
  "intent": "in",
  "rank": 2,
  "dimensions": ["ncol", "nz"]
}
```

**Used for:** Bridge generation (knows to create 2D conversion functions)

---

### 3. Character String Length Detection

**Fortran code:**
```fortran
CHARACTER(len=64), INTENT(OUT) :: scheme_name
CHARACTER(len=*), INTENT(OUT) :: errmsg
```

**Extracted:**
```json
[
  {
    "name": "scheme_name",
    "type": "character",
    "intent": "out",
    "length": 64
  },
  {
    "name": "errmsg",
    "type": "character",
    "intent": "out",
    "length": "*"
  }
]
```

**Used for:** Knowing these are strings (can't be returned from JIT in JAX!)

---

### 4. Module Variable Detection

**Fortran code:**
```fortran
MODULE kessler
  REAL :: lv, pref, rhoqr  ! Module variables
```

**Extracted in `out/modules/kessler.json`:**
```json
{
  "module_variables": [
    {"name": "lv", "type": "real"},
    {"name": "pref", "type": "real"},
    {"name": "rhoqr", "type": "real"}
  ]
}
```

**Used for:** 
- Phase 02: Detect module state
- Phase 04: Tell LLM to convert to parameters

---

## Troubleshooting

### Problem 1: No Output Files Generated

**Symptom:**
```bash
python3 workflow_frontend/phase01_01_ts_parse.py
# No files in out/procedures/
```

**Possible causes:**
- Fortran file not in `data/src/`
- Tree-sitter not installed
- Fortran grammar not compiled

---

### Problem 2: Incomplete Metadata

**Symptom:**
```json
{
  "proc_name": "kessler_run",
  "parameters": []  // Empty!
}
```

**Cause:** Tree-sitter couldn't parse the Fortran (syntax error or unsupported feature)

**Solution:**
- Check Fortran file syntax
- Check tree-sitter-fortran supports this Fortran version
- Look for parser errors in output

---

### Problem 3: Module Variables Not Detected

**Symptom:**
```json
{
  "module_name": "kessler",
  "module_variables": []  // Should have lv, pref, rhoqr!
}
```

**Cause:** Parser didn't recognize module variable declarations

**Solution:**
- Check module variables are declared correctly
- Check they're outside CONTAINS section
- Verify parser version supports this syntax

---

## How Phase 01 Fits in the Pipeline

```
Phase 01 (Tree-Sitter Parse)
    │
    ├─> units/*.json ─────┐
    ├─> modules/*.json ───┤
    ├─> procedures/*.json ┼─> Phase 02 (Dependency Analysis)
    └─> programs/*.json ──┘       │
                                  ├─> Analyzes reads/writes
                                  ├─> Detects module state
                                  └─> Merges into packets/*.json
                                          │
                                          ├─> Phase 03 (Bridge Generation)
                                          ├─> Phase 04 (LLM Prompts)
                                          └─> Phase 05 (Validation)
```

**Phase 01 provides the foundation** - all later phases depend on its metadata!

---

## Summary

### What Phase 01 Does:
✅ Parses Fortran source code into syntax trees  
✅ Extracts metadata about programs, modules, procedures  
✅ Identifies parameters, types, intents, dimensions  
✅ Detects module variables (global state)  
✅ Finds OpenMP directives  
✅ Saves everything as JSON for later phases  

### Key Outputs:
- `out/procedures/` - **Most important** - procedure signatures
- `out/modules/` - **Critical** - module variables (global state)
- `out/units/` - File organization info
- `out/programs/` - Main programs (rare)

### Why It Matters:
- Foundation of entire pipeline
- Enables automatic bridge generation
- Detects module state issues for JAX
- Provides metadata for LLM translation prompts

**Phase 01 converts unstructured Fortran text into structured metadata that machines can process!** 🎯

---

**End of Phase 01 Documentation**

**Next:** Phase 02 - Dependency Analysis (uses Phase 01 metadata)
