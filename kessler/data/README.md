# `kessler/data` — the translated scheme and the validation reference

Two Kessler sources live here, and they play different roles.

| | `exp1_baseline/src/kessler.F90` | `exp2_jd/` |
|---|---|---|
| role | **the scheme that was translated** (`[source].fortran_files`): the Fortran embedded in every archive's prompts | **the validation reference**: inputs, reference driver, outputs, envelope |
| scheme | per-column work arrays `r(nz)`, two procedures (`kessler_init`, `kessler_run`) | all-column `r(ncol,nz)`, three procedures (adds the `all_equal` reduction helper) |
| grid | — (the scheme is grid-agnostic) | 128 × 56, dt = 60 s |
| driver | — | `JD_kessler_driver.F90`, inputs from `generate_fortran_inputs_jd.py` |
| reference I/O data | — | `exp2_jd/fortran_io/` |

The two schemes compute the same physics with a different array layout; the
four translations, made from the per-column text, validate against the JD
reference outputs inside the measured envelope. The JD all-column source is
kept with the reference driver that compiles it; translating it is a separate
run (point `[source].fortran_files` at it and re-run Stage 0 — the frontend then
extracts three procedures).

## Notes

- **Provenance of `exp2_jd/fortran_io`:** the data records `seed=0`, but
  `generate_fortran_inputs_jd.py` declares `SEED=42`. The on-disk data is the
  authoritative reference the translations were validated against;
  regenerating from the script may not reproduce it bit for bit.
- The JD `kessler.F90` needs `gfortran -ffree-line-length-none` (lines > 132
  columns); the Makefile sets it.
