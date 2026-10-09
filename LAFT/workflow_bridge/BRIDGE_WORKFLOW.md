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

This step creates no test: the suite is framework code in
`workflow_bridge/` (below) and this is the first of its two runs — against the
generated bridges alone, with a fake kernel in place of the translation. The
second run is the translator workflow's Step 4.5, against the real translation.

```bash
qsub -v TEST_SCRIPT=workflow_bridge/run_bridge_tests.py pbsJobs/jax_cpu_test.sh
```

The suite executes JAX, so it is submitted as a CPU PBS job
(`pbsJobs/jax_cpu_test.sh` runs `python workflow_bridge/run_bridge_tests.py`
on a compute node; read the log in `out/jobs/`). The runner executes the
**shared suite** plus whatever the project adds in its own `bridge_test/`
(optional, see below), and writes the durable gate artifacts under
`out/reports/bridge/`:

- `bridge_test_results.json` — machine-readable, **top-level `status:
  PASS/FAIL`** (same convention as the driver/comparison JSONs), with the
  list of files that ran under `suite`. Later workflows check this instead
  of re-running the suite.
- `bridge_report.md` — human-readable: generated bridges, suite, analysis-
  report summary, per-test outcomes, verdict.

The runner exits 0 on PASS, 1 on FAIL. Gate rule: every layout test (any
file whose name contains `_layout`) passes with zero skips among them, and
nothing anywhere fails. The translation-dependent tests run too but are
informational only — `skipped` is their normal state before a translation
exists.

### The shared suite (since 2026-10-09) — nothing written per project

Three files in `LAFT/workflow_bridge/`, reached through the project's
`workflow_bridge/` symlink, run from the project root. None is copied or
edited per project: each derives the procedure's contract from the
project's packets through the generator's own extraction
(`BridgeGenerator.extract_parameters()` / `extract_module_info()` over
`out/phase1_index.json` + `out/packets/`, plus `[bridge.optional_kwargs]`
and the per-step-scalar rule from `config/project.toml`), so a new
procedure, a renamed argument or a changed module-var list needs no test
edit.

| File | Needs translation? | Checks, per generated bridge |
|---|---|---|
| `test_generated_bridge_layout.py` | **No** — plants a pass-through fake kernel | array-bearing: transfer helpers (`to_device` no host permute, bit-exact roundtrip, C-order); both entries + alias, contract-2 signature = contract 1 minus CHARACTER; `static_argnames` = integer/logical scalars minus per-step counters; every return slot bit-identical under the identity core; outputs NumPy / original shape / float64 / C-contiguous, scalar outputs plain Python values; rank ≥ 2 inputs reach the core with all axes reversed; statics concrete, real scalars / per-step counters / module vars traced; CHARACTER args never enter the jitted region and pass through; optional kwargs forwarded (`None` when omitted). Scalar-only: no device entry, no transfers; wrapper receives every argument unchanged, returns passed straight back. Plus coverage: every `out/bridge/*_bridge.py` ↔ a packet. |
| `test_generated_bridge_device_entry.py` | **Yes** — executes the real kernel | translation present; public device entry + alias; contract 2 on `to_device(host inputs)` returns, after `device_get`, values bit-identical (NaN-equal) to contract 1, same arity/shape/dtype, array outputs left on the device. Skips with reason while `out/jax/` holds no array-bearing translation. |
| `test_direction_symmetry.py` | **Yes** — executes the host bridge twice | self-selecting from the Fortran (identifier-stride level loop whose bounds and direction are arguments): flipped column + reversed direction must give flipped-identical outputs; records n/a as a PASS in `out/reports/bridge/direction_symmetry.json` when no procedure qualifies (never a skip). |

Inputs for the translation-dependent files are synthetic and packet-derived
(integer size arguments bound to the array extents, index arguments inside
every array, small positive real values). A kernel that is not defined on
synthetic data — lookup-table paths, iteration counts that depend on
physical magnitudes, index arguments that must be consistent with each
other — gets physically plausible inputs from the **optional per-project
hook** `bridge_test/bridge_inputs.py :: inputs(proc) -> dict | None`
(Kessler needs none).

Why the shared files are at least as strong as the hand-written per-project
files they replaced, and where their limit is (they prove the generator
wrote what its own inputs say; a wrong packet is the frontend stage's and
the semantic audit's job): `docs/LAYOUT_TEST_MUTATION_CHECK_2026-10-09.md`
(14/14 injected defects caught vs 11/14 by the hand-written Kessler file).

### Per-project additions (`bridge_test/`, optional)

Only for what the packets cannot know. Each file self-skips with a reason
while the translation it needs is absent:

- **physics / functionality** — bridge vs direct-kernel call on physically
  plausible inputs, sanity ranges, error-path propagation (template:
  `kessler/bridge_test/test_kessler_run_bridge.py`);
- **a chain test** where procedures hand device-resident intermediates to
  each other across the scheme's per-step phase sequence;
- **a closed-form `test_formula`** where a kernel has one;
- **a hand-transcribed pin** of a key procedure's contract, which fails when
  the packets drift even though the shared suite — which trusts the
  packets — still passes (a per-procedure file that transcribes the
  signature, as the retired `kessler/bridge_test/test_kessler_run_bridge_layout.py`
  did — kept in the repository history);
- **the `bridge_inputs.py` hook** described above.

A project with none of these has no `bridge_test/` folder at all.

To iterate on one file without regenerating the report (from an
interactive compute-node session, since it executes JAX):

```bash
python -m pytest workflow_bridge/test_generated_bridge_layout.py -v
```

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

The translation-**dependent** tests — the shared
`test_generated_bridge_device_entry.py` (contract-2 equivalence: `<proc>_bridge_device`
on `to_device(...)` inputs must return, after `jax.device_get`, arrays
bit-identical to `<proc>_bridge` on the same host inputs, one check per
array-bearing procedure), the shared `test_direction_symmetry.py`, and the
project's own physics / chain files — need `out/jax/<proc>.py` filled in and
run in the **translator workflow's Step 4.5**
(`python workflow_bridge/run_bridge_tests.py --require-dependent`, after
runtime validation, before the driver job). They skip themselves with a clear
reason when the translation (or bridge) is absent. Numerical agreement with
the original Fortran is validated later still, by the driver comparison
stage (`[comparison].script`).

## New project checklist

The required check-set is the same everywhere and is **shared code** —
no project writes a layout, contract-2 or symmetry test. What a project
may add depends on its SCHEME SHAPE (whether procs chain through
device-resident intermediates; whether a kernel has a closed form; whether
synthetic inputs can drive it).

1. Phases 01–02 green, then Step 1 (generation).
2. Run the gate (Step 2). It needs no per-project file: the shared layout
   test parameterises itself over every generated bridge.
3. If a wiring test asserts which scalars are jit-static, keep per-step
   counters (`it`, `kount`, `itimestep` — anything the driver loop changes
   every call) traced, never static: a per-step static re-specializes and
   recompiles the kernel every step (tens of seconds per step on a large
   orchestrator). phase03 excludes
   these automatically (`framework_config.is_per_step_scalar`; extend via
   `[heuristics].per_step_scalar_names`), and the shared layout test checks
   the rule.
4. **Only if a kernel is not defined on synthetic data**, write
   `bridge_test/bridge_inputs.py` with `inputs(proc) -> dict | None`
   (convention documented in `test_generated_bridge_device_entry.py`).
5. **Only where it applies**, add per-project files to `bridge_test/`: a
   physics / functionality test (template:
   `kessler/bridge_test/test_kessler_run_bridge.py`), a chain test
   where procedures share device-resident intermediates, a closed-form
   `test_formula` where a kernel has one.
6. If the project has per-project files, a short `bridge_test/TESTING_GUIDE.md`
   naming them and what each asserts (the shared suite is documented here,
   not per project).
7. Run the gate; iterate per "If the test gate fails".
