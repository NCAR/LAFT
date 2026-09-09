# validation/ — Shared translation validators

A reference on the `validation/` folder: what it holds, **who uses it**, how it
fits into the workflows that depend on it, and the full rule/output reference
for both validators. This document is codebase-agnostic — every project value
comes from `config/project.toml`. Placeholders used throughout: `{proc}` is a
procedure name, `<llm>` a translator label.

## Where this fits

- **Pipeline sequencing** — `ORCHESTRATOR.md` is the entry point and the sole
  authority on stage order, entry gates, and the mandatory user-approval pause
  between stages. This manual never re-states it; it describes the *validators*
  the translate and profile stages call.
- **Neighbouring stage docs:**
  - `workflow_translator/TRANSLATE_WORKFLOW.md` — the translator playbook
    (Steps 3–5 are the validation gate); `workflow_translator/TRANSLATE_REFERENCE.md`.
  - `workflow_profiler/PROFILE_WORKFLOW.md` — the profiler playbook (Step 5
    re-validation); `workflow_profiler/PROFILER_MANUAL.md`.
  - `workflow_bridge/BRIDGE_WORKFLOW.md` / `BRIDGE_REFERENCE.md` — the bridge
    that the lint signature check compares against.
  - `workflow_frontend/FRONTEND_WORKFLOW.md` — produces the packets both
    validators read.

## Why this folder exists

Two workflows need to answer the same question — *"is this JAX translation
correct?"* — at different moments:

- **`workflow_translator/`** validates a **fresh** translation before
  declaring it done (TRANSLATE_WORKFLOW.md, Steps 3–5).
- **`workflow_profiler/`** re-validates **after** applying hardware-efficiency
  fixes, to prove the fixes did not change the numerics (PROFILE_WORKFLOW.md,
  Step 5 re-check).

Rather than duplicate the validators inside either workflow, they live here —
a shared peer folder both workflows call. `validation/` is symlinked from
`LAFT/` into each project (`laft-<scheme>/validation → ../LAFT/validation`),
exactly like `workflow_bridge/`, `workflow_translator/`, and
`workflow_profiler/`. Run every command from the **project root** (the
directory containing `config/`).

## What's in it

| Tool | Checks | Runs where |
|---|---|---|
| `phase05_01_lint_translation.py` | **Static** structure/JIT-rules lint of `out/jax/*.py`: JAX-only (no NumPy), the `{proc}_core` + `{proc}` two-layer contract for array procs, scalar-JAX vs scalar-Python contracts, no host compute loops, and **bridge/JAX signature consistency** (the generated bridge's call must match the translation's `def`). Writes `out/lint/{proc}_lint.json` + `out/lint/_summary.json`. | Login node, local, fast |
| `phase05_02_runtime_validate.py` | **Runtime** import/execute smoke test: builds inputs from packet metadata, calls each `{proc}` wrapper (through the bridge when present) and `{proc}_core` under `jax.jit`, checks `imported/wrapper_ok/core_ok` and finiteness. Writes `out/validation/{proc}_runtime.json`. | GPU node via `pbsJobs/jax_gpu_runtimevalid.sh` |

Both import `framework_config` from the shared `config/` dir (via the standard
`SCRIPT_DIR.parent / "config"` path shim), and both use
`framework_config.packet_module_vars()` — the *same* module-var rule the
bridge and wrapper generators use — so a signature mismatch is a real defect,
never a tooling disagreement. All output directories (`lint_dir`,
`validation_dir`, `jax_dir`, `bridge_dir`, `packets_dir`, `issues_dir`) are
resolved from `config/framework_config.py`, not hard-coded.

## What is deliberately NOT here

- **Completeness check** (`workflow_translator/phase04_04_completeness_check.py`,
  TRANSLATE Step 2.5 — moved out of `validation/` 2026-08-25) and the
  **semantic audit** (`workflow_translator/phase05_01b_semantic_audit.py`,
  Step 3.5) — translation-**completeness** and **fidelity** checks (vs the
  Fortran source) used only by the translator, so they live with the
  translator, not here. Their gate (`workflow_translator/audit_gate.py`) is
  what `pbsJobs/jax_gpu_runtimevalid.sh` consults before the runtime
  validator in this folder is allowed to run.
- **Bridge test suite** — the project's `bridge_test/test_*_layout.py` suite,
  run through `workflow_bridge/run_bridge_tests.py`. It is the bridge stage's
  own gate (and is re-run against the real translation at TRANSLATE_WORKFLOW
  Step 4.5), so it lives with the bridge workflow.
- **Driver + comparison** — the end-to-end Fortran-vs-JAX numerical check. The
  driver and comparison scripts are **hand-authored per project**
  (`[driver].script`, `[comparison].script`, living in `out/driver/`), and are
  launched by per-project PBS jobs (`<[hpc].driver_job>`,
  `pbsJobs/jax_gpu_compvalues.sh`). There is no shared driver *tool* to house
  here — only the lint and runtime validators generalize across projects.
- **Repair** — there is no repair *script*. Repair is always performed
  **in-context by the driving agent**, editing `out/jax/{proc}.py` directly in
  response to a validator's output. The fix log lives in `out/issues/fix_log.md`.

## The shared validate → repair loop

Both workflows run the same gated chain; each step blocks the next, and any
failure drops into the agent-authored fix loop (max 5 attempts per (file,
error)) before re-validating from the top:

```
   lint (local)                        ← phase05_01_lint_translation.py
     │  pass
     ▼
   runtime (PBS)                        ← phase05_02_runtime_validate.py
     │  pass                              (pbsJobs/jax_gpu_runtimevalid.sh)
     ▼
   driver (PBS)                         ← <[hpc].driver_job>  (per-project)
     │  pass
     ▼
   comparison (PBS): MAE ≈ eps, ALL PASS ← pbsJobs/jax_gpu_compvalues.sh
     │  pass
     ▼
   GREEN
   ── any step FAIL ──► agent edits out/jax/{proc}.py, logs to
                        out/issues/fix_log.md, re-runs from lint
```

The two callers differ only in what precedes this chain and how they enter it:

- **Translator** adds a semantic-audit step (its own, before runtime) and runs
  the chain on the freshly authored translation.
- **Profiler** runs the chain as a *re-check* after a hardware fix; if it
  applied no fix (staged code byte-identical to the validated snapshot), the
  re-check is skipped as redundant.

## How each workflow invokes it

```bash
# translator (TRANSLATE_WORKFLOW.md) and profiler (PROFILE_WORKFLOW.md re-check)
python validation/phase05_01_lint_translation.py      # local; score must be 100
qsub   pbsJobs/jax_gpu_runtimevalid.sh                 # runs validation/phase05_02_runtime_validate.py
qsub   <[hpc].driver_job>                              # per-project driver
qsub   pbsJobs/jax_gpu_compvalues.sh                   # per-project comparison
```

---

# Part 1 — The linter (`phase05_01_lint_translation.py`)

## Lint vs runtime validation

| Aspect | Lint | Runtime validate |
|---|---|---|
| **What** | Static text/AST analysis | Actually executes the code |
| **Checks** | Structure, contracts, signatures | Import, callable, JIT-compilable |
| **Speed** | Very fast (no execution) | Medium (GPU node, JIT compile) |
| **Catches** | Contract violations, bad patterns, bridge/JAX signature drift | Runtime errors, JIT errors, NaN/Inf |
| **Where** | Login node | PBS GPU job |
| **Order** | First | Only after lint is green |

Both are needed. Lint catches whole classes of defect without paying for a GPU
job; runtime catches what only execution can reveal.

Neither one checks **physics correctness**, compares against Fortran, or
validates numerical accuracy — that is the driver + comparison step at the end
of the chain.

## The three contracts

The linter first classifies each procedure from its merged packet, then holds
the file to exactly one contract. Which contract applies is recorded in the
lint JSON `metadata` (including `jax_required`).

1. **Array procedure** (any proc touching grid/state arrays) — the full
   two-layer JAX contract: `import jax` + `import jax.numpy as jnp`, a
   `def {proc}(...)` wrapper, a `def {proc}_core(...)` JIT-able core, no NumPy,
   no `swapaxes`, no host compute loops. This applies to **every** non-scalar
   proc, including the scheme's top-level routine and model-facing wrappers,
   regardless of any `jax_required` value in a stale `_ALL_deps.json`.
2. **Scalar-JAX** (scalar-only proc that is `jax_required=True`, or that
   voluntarily imports `jnp`) — scalar JAX arithmetic, **no** `_core`, **no**
   `@jax.jit`, **no** bare `import jax`, and it must not set `JAX_ENABLE_X64`
   itself (the caller owns that).
3. **Scalar-only plain Python** (scalar proc with `jax_required=False` that does
   not use `jnp`) — plain Python, no JAX imports, no `_core`. NumPy is allowed
   here **only** when `has_io=True` (host-side I/O loaders materialize arrays).

`jax_required` per procedure is read from `out/packets/_ALL_deps.json`; if that
file is missing the linter warns and defaults every procedure to
`jax_required=True` (the strict contract).

## What the linter checks

### Common to every contract

**1. Bridge metadata exists** — `out/bridge/{proc}_bridge*` is present
(case-insensitive lookup). *Why:* the framework is bridge-mode only; a missing
bridge means the bridge stage was not run or the proc was renamed. *Fail means:*
run/re-run the bridge stage before linting.

**2. Core has no I/O** — the `_core` block contains no `print(`, `input(`, or
`open(`. *Why:* the core is JIT-compiled and cannot do host I/O. *Fail means:*
the core will raise at trace time; move the I/O to the wrapper.

**3. Wrapper I/O policy** — if the packet says `has_io=False`, the wrapper must
contain no I/O either; if `has_io=True`, wrapper I/O is allowed. *Why:* `has_io`
is a declared property of the procedure that the bridge and runtime validator
both act on; a wrapper that prints when `has_io=False` means the flag or the
translation is wrong.

**4. Wrapper args match expected** — the wrapper's parameter list is checked
against the packet's Fortran args plus the module vars from
`packet_module_vars()`. The rule is deliberately asymmetric:

- **FAIL** on a *missing required Fortran arg*, or on a parameter that is
  neither a Fortran arg nor a known module variable (an invented name).
- **Allowed, but noted in `details`:** omitted `OPTIONAL` Fortran args; module
  vars not threaded (baked as constants or returned instead); extra module vars
  threaded (vestigial from earlier packet data).

Scalar **output-only** flags (`isok`, `ierr`, `info`, `status`, …) are excluded
from the expected parameter set — they are *returned*, not passed:

```fortran
SUBROUTINE compute(n, a, b, isok, ierr)
  INTEGER, INTENT(IN)    :: n
  REAL,    INTENT(INOUT) :: a(n), b(n)
  LOGICAL, INTENT(OUT)   :: isok
  INTEGER, INTENT(OUT)   :: ierr
END SUBROUTINE
```

```python
def compute(n, a, b):          # isok, ierr NOT parameters
    ...
    return a, b, isok, ierr    # isok, ierr in the RETURNS
```

*Why:* modified arrays and output flags must come back through the return in
pure-functional style; the bridge unpacks them positionally.

**5. Bridge call matches JAX signature** — the linter parses the *actual call*
to `{proc}(...)` inside `out/bridge/{proc}_bridge.py` and compares it to the
`def {proc}` in `out/jax/{proc}.py`:

- positional args must bind to the same-named params, in order;
- keyword args must name params the JAX `def` actually has;
- every non-defaulted JAX param must be bound by the call.
  (Defaulted JAX params the bridge doesn't pass are allowed, noted only.)

*Why:* the bridge is generated from packets alone and never reads
`out/jax/*.py`. When a translation adds a parameter for a module var only
mentioned in a Fortran comment, drops an `intent(out)` arg, or reorders
params, the bridge call either raises `TypeError` or — far worse — silently
misbinds values. This check catches silent misbinding before any GPU job runs.
It compares the *call site*, not the bridge's own signature, because the bridge
may legitimately declare params in a different order and reorder at the call.

### Array-procedure contract only

**6. No host compute loops (element stores)** — no array-element assignment
inside a Python `for`/`while`. *Why:* grid mutation must live inside the jitted
`_core` as vectorized ops; per-element host stores defeat the entire point of
the translation. The failure detail names the count and first line number.

**7. No Python loop wrapping JAX calls** — no `jnp`/`lax`/`jax`/`_core`
reference inside a Python `for`/`while`. *Why:* this is Level 5 of the
VECTORIZATION PRIORITY LADDER and is never acceptable — it re-enters the JAX
dispatch path once per iteration.

**8. Uses jax / uses jnp / no numpy** — `import jax` and
`import jax.numpy as jnp` present; no `import numpy` and no `np.` usage.
*Why:* JAX compiles to GPU, NumPy does not, and a stray `np.` silently pins
that computation to the host — typically a 10–100x slowdown, and it breaks
JIT tracing outright inside a core.

**9. Has wrapper** — `def {proc}(...)` exists. *Why:* the bridge calls the
wrapper by name; it is the layer that handles strings and error flags, which
cannot live in a jitted core.

**10. No swapaxes in module** — `swapaxes(` does not appear. *Why:* in bridge
mode, layout conversion (Fortran column-major ↔ row-major) happens in the
**bridge layer**. JAX code is pure row-major with no transpose logic. A
`swapaxes()` in the translation means the model either double-converts or
converts in the wrong place.

**11. Has core** — `def {proc}_core(...)` exists. *Exception:* wrapper-only is
allowed for **pure I/O routines**, defined as `has_io=True` **and** no writes
to arguments **and** no calls to other procedures. In that case the check
records "core optional (wrapper-only I/O)" as a pass. *Why:* the core is the
JIT-compilable compute layer; a compute routine without one has nothing to
compile.

### Rules grouped by what they protect

- **GPU performance** — no NumPy, no `swapaxes`, no host compute loops, no
  Python loop around JAX calls. Violations still *run*, which is why static
  linting matters: they are silently 10–100x slower, not broken.
- **JIT compilation** — core exists, core has no I/O, scalar-JAX has no
  `@jax.jit`. Violations crash at trace time.
- **Interface integrity** — wrapper signature, bridge-call match, wrapper I/O
  policy. Violations either raise `TypeError` at the bridge boundary or
  misbind arguments silently.

## Lint output

### Console

```
=== Linting {proc} ===
[OK] has bridge metadata
[OK] uses jax
[OK] uses jnp
[OK] no numpy
[OK] has wrapper
[OK] no swapaxes in module
[OK] has core
[OK] core has no I/O
[OK] wrapper has no I/O
[FAIL] wrapper args match expected
[OK] bridge call matches JAX signature

======================================================================
LINT SUMMARY
======================================================================
Procedures: 1/2 passed (1 failed)
Checks:     20/21 passed (1 failed)
Score:      95.2%
======================================================================
```

### Per-procedure JSON — `out/lint/{proc}_lint.json`

```json
{
  "proc": "{proc}",
  "timestamp": "2026-07-17T10:30:00",
  "metadata": {
    "jax_file": "out/jax/{proc}.py",
    "jax_required": true
  },
  "checks": [
    {"name": "has bridge metadata", "passed": true,  "details": null},
    {"name": "uses jax",            "passed": true,  "details": null},
    {"name": "no numpy",            "passed": false, "details": "Found NumPy usage"},
    {"name": "wrapper args match expected", "passed": false,
     "details": "missing required Fortran args: ['dt']"}
  ],
  "summary": {
    "total_checks": 11,
    "passed": 9,
    "failed": 2,
    "score": 81.8,
    "overall": "FAIL"
  }
}
```

### Roll-up — `out/lint/_summary.json`

```json
{
  "timestamp": "2026-07-17T10:30:05",
  "procedures": {"total": 2, "passed": 1, "failed": 1},
  "checks":     {"total": 21, "passed": 20, "failed": 1, "score": 95.2},
  "details": [
    {"proc": "{proc}", "passed": 11, "failed": 0, "score": 100.0, "overall": "PASS"}
  ]
}
```

**The gate is `checks.score == 100` and every procedure `overall: "PASS"`.** A
partial score is a failed gate, not a "mostly good" translation.

Because the JSON is per-procedure and per-check, archived lint dirs
(`translations/<llm>/lint/`, written by `tools/copy_AI_results.sh`) give an
objective, reproducible framework-compliance metric that can be compared across
translators without re-running anything.

## Common lint failures and fixes

### NumPy instead of JAX

```
[FAIL] no numpy   →  "Found NumPy usage"
```

```python
import numpy as np            # wrong
arr = np.zeros((10, 10))
```

```python
import jax.numpy as jnp       # correct
arr = jnp.zeros((10, 10))
```

### Layout conversion in JAX code

```
[FAIL] no swapaxes in module
```

```python
def {proc}(x, ...):
    x = jnp.swapaxes(x, 0, 1)   # wrong — F→C conversion belongs in the bridge
```

Delete it and work directly with row-major data; the bridge already converted.

### I/O in the core

```
[FAIL] core has no I/O
```

```python
def {proc}_core(...):
    print("debug:", x)          # wrong — not JIT-able
```

```python
def {proc}_core(...):
    return x                    # pure

def {proc}(...):
    x = {proc}_core(...)
    print("debug:", x)          # correct — I/O in the wrapper
```

### Wrong wrapper signature

```
[FAIL] wrapper args match expected
       missing required Fortran args: ['b']; unknown extra params: ['isok']
```

```python
def compute(n, a, isok):        # wrong: dropped b, took isok as a parameter
    return a, isok
```

```python
def compute(n, a, b):           # correct
    isok = True
    return a, b, isok
```

### Bridge/JAX signature drift

```
[FAIL] bridge call matches JAX signature
       positional misbinding: arg 'qv' binds to param 'qc' (position 3)
```

The bridge is generated from packets and is *not* the thing to edit by hand —
fix the translation's `def` to match the packet-derived parameter order. If the
packet itself is wrong, that is a frontend/bridge-stage defect: re-run those
stages (`ORCHESTRATOR.md`) rather than patching the JAX file.

### Fix cycle

```bash
python validation/phase05_01_lint_translation.py
# inspect only the failures
jq '.checks[] | select(.passed == false)' out/lint/{proc}_lint.json
# edit out/jax/{proc}.py, log the attempt in out/issues/fix_log.md, then re-lint
python validation/phase05_01_lint_translation.py
```

Repeat until the score is 100. Each cycle counts against the fix-attempt
budget (5 attempts per (file, error)) defined in TRANSLATE_WORKFLOW.md §6.

---

# Part 2 — Runtime validation (`phase05_02_runtime_validate.py`)

## What it is

A **smoke test**: does the translated code run at all? It is not a correctness
check. The distinction matters — a smoke test asks "does the engine start?",
the driver/comparison step asks "does the vehicle go where it should?".

It exists because LLM translations carry runtime defects that are invisible on
the page: a module that will not import, a wrapper that crashes on real
argument types, a core that traces fine by eye but fails JIT compilation. This
step catches all three in one GPU job, before any expensive end-to-end run.

It runs over **every** `out/packets/*_merged.json` procedure, and it is
**bridge-mode only** — a procedure whose bridge is missing is reported as
failed rather than silently validated under an older layout policy.

## Three-level validation

### Level 1 — Import

Can `out/jax/{proc}.py` be imported?

- **Pass:** no syntax errors, no import errors, all dependencies resolvable.
- **Fail:** syntax error, missing `import jax` / `import jax.numpy as jnp`, or
  the module cannot be found on the path.

### Level 2 — Wrapper callable

Can the wrapper be called with generated inputs? When a bridge file exists the
call goes **through** `{proc}_bridge` with **Fortran-layout** inputs, so the
full stack (bridge conversion + wrapper) is exercised.

- **Pass:** the signature accepts the inputs, execution completes, values are
  returned, outputs are finite.
- **Fail:** wrong parameter count, dtype mismatch, crash mid-computation, or
  NaN/Inf in the outputs.

### Level 3 — Core under JIT

Can `{proc}_core` be `jax.jit`-compiled and run? Inputs are transposed to JAX
layout for this call. Skipped when `effects.has_io == True` or no `_core`
exists (the wrapper-only I/O case).

- **Pass:** compilation succeeds, the compiled function runs, no forbidden
  operations, outputs finite.
- **Fail:** `ConcretizationTypeError`, a non-JAX return type (e.g. a string),
  or a crash inside the compiled function.

**Level 3 is the most valuable of the three** — it is where JAX-specific
defects surface, and they are exactly the ones that reading the code does not
reveal.

## Input generation

Inputs are built from the merged packet's argument metadata, not hand-written:
`build_inputs()` walks the argument list and module vars, deriving dtype, rank,
and extents from the packet, with typed fallbacks per Fortran kind
(`integer`/`logical`/`character`/real). Grids are small and values uniform —
enough to execute every branch shape, not enough to be physically meaningful.
2-D arrays are constructed in Fortran layout and transposed at the boundary
that needs the other order. Character arguments that name a file are pointed at
a scratch path under `out/validation/`.

Project overrides (`[runtime_validation.arg_values]` in `config/project.toml`,
added 2026-08-26): a table keyed by procedure name (or `"*"` for all) mapping an
argument name to the value to use instead of the dummy: a string containing
`/` is a project-relative path; `"ramp:<step>"` is an array decreasing along
the last Fortran axis (`step*(n..1)`, e.g. heights with `kbot = nk` so layer
thicknesses are non-zero); a number/bool is a scalar, broadcast to the
argument's declared shape. Use it only where the uniform dummies make the
*Fortran* itself undefined — an init routine that opens lookup tables
(`lookup_file_dir = "…/lookup_tables"`, `ncat = 1` so it only reads the tables
actually shipped), or a host-model wrapper that builds its atmosphere from
`psfc`/`temp`/heights (1 K, 1 Pa, dz = 0 → saturation underflows to 0 → ÷0).
Without it, every CHARACTER argument is `""`, every integer scalar `DUMMY_N`,
every array uniform 1.0.

Consequence: finite outputs here mean *the code executes*, never *the physics is
right*. Numerical truth comes only from the driver + comparison step.

## Output — `out/validation/{proc}_runtime.json`

```json
{
  "proc": "{proc}",
  "imported": true,
  "wrapper_called": true,
  "core_jit_called": true,
  "wrapper_ok": true,
  "core_ok": true,
  "wrapper_exception": null,
  "core_exception": null,
  "wrapper_finite": true,
  "core_finite": true,
  "notes": []
}
```

| Field | Meaning |
|---|---|
| `imported` | Module imported successfully |
| `wrapper_called` | The wrapper call was attempted |
| `core_jit_called` | Core JIT compilation was attempted (false when skipped) |
| `wrapper_ok` | Wrapper executed without raising |
| `core_ok` | Jitted core executed without raising |
| `wrapper_exception` | Traceback string if the wrapper failed, else `null` |
| `core_exception` | Traceback string if the core failed, else `null` |
| `wrapper_finite` | Wrapper outputs contain no NaN/Inf (`null` if not evaluated) |
| `core_finite` | Core outputs contain no NaN/Inf (`null` if not evaluated) |
| `notes` | Informational messages (e.g. why a core test was skipped) |

**The gate is `imported && wrapper_ok && core_ok` for every procedure.** The
process exits `0` when all pass and `2` when any fail — the PBS log is the
authoritative record; never trust a JSON file without confirming the job's
`.o` log is from the current run.

## Reading the results

```bash
jq . out/validation/{proc}_runtime.json
grep -l '"wrapper_ok": false' out/validation/*.json
grep -l '"core_ok": false'    out/validation/*.json
```

| Console line | Meaning | Where to look |
|---|---|---|
| `imported=False wrapper_ok=False core_ok=False` | Cannot even import | syntax, imports, module path |
| `imported=True wrapper_ok=False core_ok=False` | Imports, wrapper crashes | wrapper signature, argument dtypes |
| `imported=True wrapper_ok=True core_ok=False` | Wrapper fine, JIT fails | traced-value misuse in the core |
| all `True`, `*_finite` false | Runs but produces NaN/Inf | division by zero, `log` of a non-positive, instability |

## Common runtime failures and fixes

### Missing JAX import

```
NameError: name 'jnp' is not defined
```

Add `import jax` and `import jax.numpy as jnp` at the top of the file. (Lint
catches this first for array procs; it shows up at runtime when a dependency
module is the one missing the import.)

### Concretization error

```
ConcretizationTypeError: Abstract tracer value encountered where a concrete value is expected
```

Cause: `float()`, `int()`, `.item()`, or a Python `if` on a traced array.

```python
if float(dt) < 0:                      # wrong
    errflg = 1
```

```python
errflg = jnp.where(dt < 0, 1, 0)       # correct
```

Use `jnp.where` for conditionals, `lax.cond`/`lax.select` for branch bodies,
and `lax.while_loop`/`lax.fori_loop` for trip counts that depend on data.

### String returned from a jitted core

```
TypeError: function {proc}_core returned a value of type <class 'str'>
which is not a valid JAX type
```

Cause: message/status strings inside the core. Return a numeric code from the
core and map it to a string in the wrapper.

### Unbound variable in a loop

```
UnboundLocalError: local variable 'x' referenced before assignment
```

Cause: a variable mutated inside `lax.while_loop`/`lax.fori_loop` that is not
part of the carried state tuple. Every value that changes across iterations
must be threaded through the carry.

### Wrong parameter count at the bridge boundary

```
TypeError: {proc}() takes 15 positional arguments but 17 were given
```

Cause: the bridge and the JAX signature have diverged. Lint's
"bridge call matches JAX signature" check should have caught this — if it did
not, the bridge is stale: re-run the bridge stage, or fix the JAX signature to
match the packet-derived one.

---

# Validation checklist

**Before running the validators:**

- [ ] `out/jax/{proc}.py` exists for every procedure in `out/packets/`
- [ ] `out/bridge/{proc}_bridge.py` exists (bridge stage gate is `PASS`)
- [ ] `out/packets/_ALL_deps.json` present (otherwise every proc defaults to
      the strict `jax_required=True` contract)
- [ ] JAX environment active with `jax`, `jaxlib` installed (the runtime job
      sets `JAX_ENABLE_X64=1` itself)
- [ ] Running from the **project root** (the directory containing `config/`)

**After lint:**

- [ ] `out/lint/_summary.json` → `checks.score == 100`
- [ ] Every `out/lint/{proc}_lint.json` → `summary.overall == "PASS"`
- [ ] Any `details` note on an allowed-but-noted signature item was read and
      understood, not skimmed

**After runtime validation:**

- [ ] The PBS `.o` log was read and is from *this* run (no traceback, fresh
      timestamps) — never judge from JSON alone
- [ ] Every `{proc}_runtime.json` → `imported`, `wrapper_ok`, `core_ok` all true
- [ ] No `wrapper_finite` / `core_finite` false
- [ ] `notes` reviewed for skipped core tests, and each skip is legitimately a
      wrapper-only I/O routine

**If anything fails:**

- [ ] Diagnose from the exception text, not the symptom
- [ ] Apply a minimal surgical fix to `out/jax/{proc}.py`
- [ ] Record file, error, diagnosis, attempt, and verification evidence in
      `out/issues/fix_log.md`
- [ ] Re-run the chain **from lint**, not from the failed step
- [ ] Stop and report after 5 attempts on the same (file, error)

**If everything passes:** continue down the chain — driver, then comparison
(`MAE ≈ machine epsilon`, `ALL PASS`) — per the calling workflow, and return to
`ORCHESTRATOR.md` for the stage hand-off and its mandatory approval pause.
