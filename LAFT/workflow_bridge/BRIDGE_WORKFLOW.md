# Bridge Workflow — generate the bridge layer and gate it with tests

You are generating the Fortran↔JAX bridge layer for the project's target
procedures and verifying it BEFORE any LLM translation exists. This document
is **codebase-agnostic**: every project value comes from `config/project.toml`
(paths, `[hardware]` advisor numbers, `[heuristics]` loop-index hints,
`[bridge.optional_kwargs]`).

**Working directory: the project root** (the directory containing `config/`).

Hand-off gates:
- **Enter** this workflow after phases 01–02 are complete: `out/phase1_index.json`
  and `out/packets/*_merged.json` exist.
- **Exit** to the translator workflow (`workflow_translator/TRANSLATE_WORKFLOW.md`)
  only when the bridge test gate below is green.

---

## Why the bridge is testable before translation

The generated bridge is pure deterministic plumbing (PATH C, device-side
layout conversion):

```
C-order (ncol, nz) → to_device (pure H2D) → in-jit axis reversal →
  {proc}_core call → in-jit reversal back →
  ONE batched jax.device_get (pure D2H, single sync) → C-order (ncol, nz)
```

Its correctness — argument wiring order, in-jit reversal on exactly the
rank≥2 variables, output layout, module-var INOUT threading — is independent
of what the kernel computes. Substituting a **pass-through core** (returns
its inputs in output order; it must be a pure jit-traceable function, since
the bridge calls it inside its jitted device wrapper) reduces the bridge to
`reverse → identity → reverse`, so every array must come back
**bit-identical, original shape, standard C-order** `(ncol, nz)`. Any wiring
or reversal defect breaks that identity immediately.

Note the array-bearing bridge imports and calls `out.jax.<proc>.<proc>_core`
directly — CHARACTER args stay host-side (strings cannot cross the jit
boundary) and pass through the bridge unchanged. Scalar-only bridges still
import and call the translated wrapper `<proc>`.

## Two bridge contracts per array-bearing procedure (since 2026-08-19)

Every generated `out/bridge/<proc>_bridge.py` exposes the same jitted code
behind two entries:

| entry | arrays in / out | transfers | for |
|---|---|---|---|
| `<proc>_bridge(...)` — **contract 1, host-facing** | C-order NumPy in Fortran `(ncol, nz)` index order | H2D on every array input, ONE batched `jax.device_get` on the outputs, per call | a host caller (a Fortran model via f2py, the project driver's default path, the `_compare_results` e2e basis) |
| `<proc>_bridge_device(...)` — **contract 2, device-resident** | `jax.Array`s **already on the device**, same `(ncol, nz)` index order (what `to_device(host)` produces); outputs left on the device | none — it *is* the jitted device wrapper (in-jit axis reversal around `<proc>_core`) | a caller that keeps the model state on the device across the **scheme's own per-step phase sequence** (and across timesteps in a GPU-resident host model) and fetches once at the end — not a pairing of different translated schemes |

Contract 1 is literally `to_device → <proc>_bridge_device → device_get`, so
the two entries are bit-identical by construction. Argument order, static
integer/logical scalars, MODULE-var INOUT threading and the return tuple are
the same in both, minus CHARACTER params (strings never enter the jitted
region; contract 2 has none in its signature). `_<proc>_device` remains as
an alias of `<proc>_bridge_device` for anything that used the pre-08-19
private name.

Why contract 2 exists: for a scheme whose kernel is a few flops per element
(a 3-flop update over ~2 GB of state per call at ncol=10⁶, in one LAFT
project outside this repository) the per-call PCIe round trip of contract 1
costs more than the whole Fortran serial loop — measured 0.32× e2e while the
kernel is 47× — and no bridge design can fix that; only not transferring per
call can (run-only 250 ms host-bridged vs 1.73 ms device-resident at 10⁶).
Contract 2 is **additive**: projects that never call it see
no change in contract 1's behaviour or performance.

Scalar-only procedures have no device entry (nothing to keep resident); they
keep the single host-facing bridge that calls the translated wrapper.

### Which contract does the application use? `[driver].contracts` + `LAFT_DRIVER_MODE`

Both contracts are always *generated and equivalence-tested* (that is this
workflow, and it is translation-independent). Which one(s) the application
actually *uses* is the project's declared decision, recorded in
`[driver].contracts` in `config/project.toml` — `[1, 2]` = undecided,
exercise both; one value = pinned (see the config template for the full
lifecycle). The translator workflow (TRANSLATE_WORKFLOW.md §5.0) runs the
driver + comparison once per declared contract, selecting the mode through
the frozen framework convention **`LAFT_DRIVER_MODE`** — environment
variable, values `host` (contract 1; the default when unset) and
`device_resident` (contract 2), read by the hand-authored per-project driver
(reference implementation:
`kessler/out/driver/kessler_jax_driver.py`). The
contract-2 equivalence test in the dependent suite (below) is the **entry
condition** for certifying any device-resident driver mode: mode
certification only tells you something if the two entries are already proven
bit-identical at the bridge level.

## Step 1 — Generate

```bash
python workflow_bridge/phase03_make_bridge.py
```

Reads `out/phase1_index.json` + `out/packets/*_merged.json`; writes
`out/bridge/<proc>_bridge.py` and per-procedure analysis reports (loop-swap
safety, GPU-efficiency warnings) under `out/reports/bridge/`. MODULE
variables follow the INOUT pattern (passed as parameters, returned to the
driver — no hidden state).

## Step 2 — Test gate (translation-independent)

```bash
qsub -v TEST_SCRIPT=workflow_bridge/run_bridge_tests.py pbsJobs/jax_cpu_test.sh
```

The suite executes JAX, so it is submitted as a CPU PBS job
(`pbsJobs/jax_cpu_test.sh` runs `python workflow_bridge/run_bridge_tests.py`
on a compute node; read the log in `out/jobs/`). The runner executes the
project's whole `bridge_test/` suite and writes the durable gate
artifacts under `out/reports/bridge/`:

- `bridge_test_results.json` — machine-readable, **top-level `status:
  PASS/FAIL`** (same convention as the driver/comparison JSONs). Later
  workflows check this instead of re-running the suite.
- `bridge_report.md` — human-readable: generated bridges, analysis-report
  summary, per-test outcomes, verdict.

The runner exits 0 on PASS, 1 on FAIL. Gate rule: every
`bridge_test/test_*_layout.py` test passes with zero skips among them, and
nothing anywhere fails. The translation-dependent suite runs too but is
informational only — `skipped` is its normal state before a translation
exists.

To iterate on a single file without regenerating the report (from an
interactive compute-node session, since it executes JAX):

```bash
python -m pytest bridge_test/test_<proc>_bridge_layout.py -v
```

Bridge tests are **per-project code** (they hardcode each procedure's
signature and slot order), so they live in the project's own `bridge_test/`
folder, not here — the same shared/per-project split as
`[driver].script` and `[profiler].inputs_script`. Reference implementation:
`kessler/bridge_test/test_kessler_run_bridge_layout.py`.

A layout test file must:

1. **Inject a fake pass-through core** into `sys.modules` under the
   translated module's name (e.g. `out.jax.<proc>`) *before* importing the
   bridge — the array-bearing bridge imports `<proc>_core` at module level
   (scalar-only bridges import the wrapper `<proc>`), and this makes the
   test runnable with `out/jax/` empty. The fake core must be a **pure
   jit-traceable function** (no recording into Python state of concrete
   values — it runs under `jax.jit`, so its arguments are tracers; assert on
   tracer *shapes/dtypes* inside, or validate wiring via the roundtrip
   identity). Pop any cached real modules first, and restore `sys.modules`
   in `teardown_module` so this file and the translation-dependent file are
   order-independent.
2. Cover, with **every field carrying unique values** (so a swapped argument
   cannot cancel out):
   - **Transfer helpers**: `to_device`/`to_host` roundtrip preserves values,
     shape, and dtype (float64); no permutation happens host-side.
   - **Roundtrip identity**: `bridge(**inputs)` returns each array
     bit-identical to its input (`assert_array_equal`, not almost-equal —
     the axis reversal is exact in float64).
   - **Output layout**: outputs are original-shape, standard C-order
     `(ncol, nz)` NumPy arrays.
   - **Kernel-side layout**: inside the fake core, each rank≥2 argument's
     *shape* must be the reverse of the input shape (the in-jit reversal
     happened); rank 0/1 shapes unchanged.
   - **Scalars**: reach the core unconverted (shape/index integers arrive
     as concrete Python values — they are jit statics).
   - **Strings**: CHARACTER args must NOT reach the core; CHARACTER outputs
     pass through the bridge unchanged in their return slots.
   - **Module vars**: reach the core and come back in their return slots
     (INOUT pattern).

**Done criterion:** `out/reports/bridge/bridge_test_results.json` says
`"status": "PASS"` → the bridge layer is verified and the translator
workflow can start.

## If the test gate fails

The defect is in the **generator or its inputs, never in the generated
file** — do not hand-edit `out/bridge/*.py`:

1. Wrong argument order / missing variable → check the procedure's
   `out/packets/<proc>_merged.json` (`deps.args`, `writes_to_args`,
   `module_vars_used`) and `arg_metadata` in `out/phase1_index.json`
   (phase-01/02 output).
2. Wrong-rank converter → `arg_metadata` rank for that variable.
3. Genuine codegen bug → fix `workflow_bridge/phase03_make_bridge.py`
   (shared across all projects), regenerate, re-run the gate.

## Out of scope here

`bridge_test/test_<proc>_bridge.py` — the translation-**dependent** suite
(bridge vs direct-JAX equivalence, physics sanity, error propagation through
the real kernel, **and contract-2 equivalence: `<proc>_bridge_device` on
`to_device(...)` inputs must return, after `jax.device_get`, arrays
bit-identical to `<proc>_bridge` on the same host inputs** — one such test per
array-bearing procedure is part of the dependent gate since 2026-08-19) —
needs `out/jax/<proc>.py` filled in and runs in the
**translator workflow's Step 4.5**
(`python workflow_bridge/run_bridge_tests.py --require-dependent`, after
runtime validation, before the driver job). It skips itself with a clear
reason when the translation (or bridge) is absent. Numerical agreement with
the original Fortran is validated later still, by the driver comparison
stage (`[comparison].script`).

## New project checklist

Every project's `bridge_test/` is per-project code, but the required
check-set is the same everywhere — suites differ by SCHEME SHAPE (how many
procedures; whether procs chain through device-resident intermediates;
whether a kernel has a closed form), never by drift. Items 2–5 create the
test files, item 6 documents them, item 7 gates.

1. Phases 01–02 green, then Step 1 (generation).
2. **Layout suite** (translation-independent — the Step-2 gate). Pick the
   shape by procedure count:
   - **few procedures** → copy
     `kessler/bridge_test/test_kessler_run_bridge_layout.py` per
     procedure; adapt procedure name, field lists, ranks, slot indices,
     module vars to the scheme's signature (read the generated
     `out/bridge/<proc>_bridge.py` — the signature and return tuple are
     the contract);
   - **many procedures** (tens of them) → parameterise ONE file over all
     generated bridges instead of hand-writing dozens.
3. If a wiring test asserts which scalars are jit-static, keep per-step
   counters (`it`, `kount`, `itimestep` — anything the driver loop changes
   every call) in the TRACED-scalars check, never in the statics list: a
   per-step static re-specializes and recompiles the kernel every step
   (tens of seconds per step on a large orchestrator). phase03 excludes these automatically
   (framework_config.is_per_step_scalar; extend via
   `[heuristics].per_step_scalar_names`).
4. **Translation-dependent functionality file** `test_<proc>_bridge.py` —
   bridge-vs-direct-kernel equivalence, physics sanity, and error-path
   propagation against the REAL kernel. It skips itself with a clear
   reason while `out/jax/` is empty and becomes part of the gate at the
   translator workflow's Step 4.5 (`--require-dependent`). Template:
   `kessler/bridge_test/test_kessler_run_bridge.py`.
5. **Contract-2 equivalence test** in the dependent suite (template:
   `kessler/bridge_test/test_kessler_device_entry.py`,
   parameterised over the array-bearing procedures — the right shape for
   many-proc projects too). Add a **chain test** ONLY where procedures
   share device-resident intermediates;
   add a closed-form `test_formula` check ONLY where a kernel has one.
   Also copy the **vertical-symmetry test** template
   `LAFT/workflow_bridge/test_direction_symmetry.py` into `bridge_test/`
   unchanged (2026-08-25) when the suite is the parameterised
   many-procedure kind — the template imports the `CONTRACTS`,
   `_make_inputs` and `_import_line` helpers of a
   `test_all_bridges_layout.py`. It derives its own cases from the Fortran
   (procedures with an identifier-stride level loop whose bounds and
   direction are arguments), flips the column + reverses the direction and
   requires identical physics; with no such procedure it records `n/a` in
   `out/reports/bridge/direction_symmetry.json` and passes (never skips).
   The kessler suite (per-procedure files, written before the template)
   does not carry it; kessler would record `n/a` anyway — its level stride
   `lyr_step` is a local, not an argument.
6. **`bridge_test/TESTING_GUIDE.md` — REQUIRED for every project**
   (decided 2026-08-20): documents the suite's files, what each test
   asserts, and how to run them. Each project's file set differs (scheme
   shape drives it), so the guide is what makes the suite navigable
   without reading every test. Model:
   `kessler/bridge_test/TESTING_GUIDE.md`.
7. Run the gate; iterate per "If the test gate fails".
