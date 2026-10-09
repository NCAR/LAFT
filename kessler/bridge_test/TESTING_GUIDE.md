# Kessler — per-project bridge test additions

The bridge suite itself is **shared** (`workflow_bridge/`, documented in
`workflow_bridge/BRIDGE_WORKFLOW.md` §Step 2): layout/wiring gate, contract-2
equivalence and vertical symmetry derive their contract from this project's
packets and need no file here. The hand-written layout and device-entry files
this folder used to hold were removed on 2026-10-09 (last in commit
`2455961`; the shared files were shown at least as strong — see
`LAFT/docs/LAYOUT_TEST_MUTATION_CHECK_2026-10-09.md`).

What stays here is what the packets cannot know:

| File | Needs translation? | Checks |
|---|---|---|
| `test_kessler_run_bridge.py` | **Yes** — executes `out/jax/kessler_run.py` | synthetic, spatially uniform atmosphere (ncol=10, nz=5, θ=300 K, qv=10 g/kg, qc=1 g/kg, qr=0.1 g/kg): bridge returns the 12 expected outputs with `errflg == 0`; bridge matches a direct JAX call with hand-applied conversions (arrays to 1.5e-10, flags/module vars exact, strings passthrough); 2-D outputs `(ncol, nz)` C-order; physics sanity (θ in 200–400 K, qv/qc/qr/precl ≥ 0, relhum in 0–100 %); `dt < 0` → `errflg == 1` raised numerically inside the core; `lv, pref, rhoqr` threaded and returned |

No `bridge_inputs.py` hook: the shared tests' synthetic inputs drive both
Kessler kernels (verified against the archived translations). No chain test: a single array-bearing procedure has
nothing to chain.

```bash
cd kessler                           # jax-validate env
python workflow_bridge/run_bridge_tests.py                      # Stage-1 gate (shared layout suite)
python workflow_bridge/run_bridge_tests.py --require-dependent  # Stage 2, Step 4.5 (needs out/jax/)
python -m pytest bridge_test/test_kessler_run_bridge.py -v      # this file alone
```

None of these compares against the original **Fortran** — that is the
end-to-end driver comparison stage (`out/driver/kessler_compare_fortran_jax.py`,
via `pbsJobs/jax_gpu_compvalues.sh`).
