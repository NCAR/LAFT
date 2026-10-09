# Shared bridge tests vs hand-written tests — evidence (2026-10-09)

**Question.** Can the per-project, hand-written bridge layout test be replaced
by one shared file that derives the contract from the packets? The shared file
is `workflow_bridge/test_generated_bridge_layout.py`; the hand-written
reference is `kessler/bridge_test/test_kessler_run_bridge_layout.py`.

**Method.** `tools/mutation_check_layout_tests.py` copies the Kessler project
(config, packets, phase-1 index, generated bridges, the hand-written test) to a
scratch directory, applies one deliberate defect at a time to the generated
`out/bridge/kessler_run_bridge.py`, and runs both test files on each mutant.
"Caught" = at least one test fails. Both files pass the unmutated control.

## Result

| # | Defect injected into the generated bridge | hand-written | shared |
|---|---|---|---|
| M1 | swap two array args at the core call (`qv`↔`qc`) | caught | caught |
| M2 | drop a module var from the host return (`rhoqr`) | caught | caught |
| M3 | missing in-jit **input** reversal (`theta`) | caught | caught |
| M4 | missing in-jit **output** reversal (`qv`) | caught | caught |
| M5 | real scalar made jit-static (`dt`) | **missed** | caught |
| M6 | swap two return slots of the device entry (`precl`↔`relhum`) | caught | caught |
| M7 | integer static dropped (`errflg` becomes traced) | caught | caught |
| M8 | CHARACTER passthrough replaced by a constant (`errmsg` → `''`) | **missed** | caught |
| M9 | swap two static scalars at the core call (`lyr_surf`↔`lyr_toa`) | caught | caught |
| M10 | scalar output not converted to a host value (`.item()` dropped) | caught | caught¹ |
| M11 | `to_device` uses float32 | caught | caught |
| M12 | host-side permute in `to_device` (double conversion) | caught | caught |
| M13 | private device alias broken (`_kessler_run_device = None`) | **missed** | caught |
| M14 | module var not threaded to the core (`rhoqr` replaced by `pref`) | caught | caught |

¹ Missed on the first run: the shared file only rejected a `jax.Array`, and the
dropped `.item()` leaked a NumPy `int64`. The check now requires a plain Python
`int`/`float`/`bool` for scalar outputs (what phase03 emits). Fixed and re-run
before this table was written.

**14/14 mutants caught by the shared file; 11/14 by the hand-written file.**
The three the hand-written file misses are contract rules it never encoded
(per-step/real scalars stay traced; CHARACTER passthrough — its inputs used
`errmsg=''`, so a constant `''` looked right; the contract-2 alias).

## The shared suite in a full run

On 2026-10-09 the Kessler project was run end to end (frontend, bridge,
translation, profile) with the shared suite as the only bridge gate:

| step | shared suite | hand-written suite it replaces |
|---|---|---|
| Stage-1 bridge gate (translation-independent) | 14 passed, 0 skipped | 9 passed |
| Step 4.5, against the fresh translation (`--require-dependent`) | 25 passed, 0 skipped — 14 layout + 11 dependent (contract-2 ≡ contract-1 bit-identity, symmetry n/a-as-pass, the project's physics test) | — |

No layout or device-entry test file was written for the project. The shared
device-entry test was also run against each of the nine archived Kessler
translations the project holds in its development tree (4 tests each, 1–4 s).

## What this does and does not show

- The shared files check that the **generator wrote what its own inputs say**
  (wiring order, in-jit reversal on exactly the rank≥2 arguments, return
  slots, static vs traced scalars, CHARACTER passthrough, module-var INOUT,
  transfer helpers, the two entries and the alias, contract 2 ≡ contract 1 on
  the real kernel). They derive the expected contract from the same packets
  phase03 read, so they cannot catch a wrong packet; that is the frontend
  stage's and the semantic audit's job.
- The hand-written file remains useful as a **pin**: a transcribed signature
  that fails when the packets drift. It is no longer needed for coverage.

## Two limits of synthetic inputs, learned by running real kernels

1. **Integer size arguments must equal the array extents.** With arbitrary
   integers two Kessler translations failed inside the kernel. The builder
   binds size arguments from the phase-1 `dimensions` when the Fortran
   declares them (`its:ite` → 1, extent), else by name (`[heuristics].dim_names`,
   `[profiler].ncol_args`, a built-in list of common grid-size names) in
   Fortran index order; every other integer cycles through 1..min(extent).
2. **Magnitudes must keep data-dependent loops short.** On random inputs in
   [10, 100) with `dt ≈ 100`, one Kessler translation's sub-stepping
   `lax.while_loop` ran for more than ten minutes, and a jitted loop cannot be
   interrupted from inside pytest. The device-entry builder uses small
   positive values; all nine archived translations then pass in seconds. What
   small values are not enough for is what the per-project inputs hook is for.

## Reproduce

```bash
conda activate jax-validate
python LAFT/tools/mutation_check_layout_tests.py            # exit 0 = every mutant caught
cd kessler && python -m pytest workflow_bridge/test_generated_bridge_layout.py -v
```
