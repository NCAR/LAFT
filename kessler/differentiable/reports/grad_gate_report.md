# Differentiable gate — kessler_run

**Status: PASS** · 2026-10-06T19:22:04Z · JAX 0.6.2 (cpu) · loop=scan, n_max=16

Module `differentiable/kessler_run_diff.py`; reference `translations/gemini31pro-jd/jax/kessler_run.validated.py`.

| Check | Result |
|---|---|
| G1_equivalence | PASS |
| G2_finite[hard] | PASS |
| G3_ad[hard] | PASS |
| G4_fd[hard] | PASS |
| G2_finite[soft] | PASS |
| G3_ad[soft] | PASS |
| G4_fd[soft] | PASS |
| G5_soft_convergence | PASS |

## G1 — hard clamps vs reference (max |diff| / max |ref|)

| Field | vs validated translation | vs Fortran | scan vs while |
|---|---|---|---|
| theta | 1.6e-16 | 3.2e-16 | 1.6e-16 |
| qv | 1.3e-15 | 2.0e-15 | 1.3e-15 |
| qc | 1.9e-15 | 2.1e-15 | 1.9e-15 |
| qr | 8.3e-16 | 8.3e-16 | 8.3e-16 |
| precl | 3.0e-16 | 4.6e-16 | 3.0e-16 |
| relhum | 2.6e-15 | 2.8e-15 | 2.6e-15 |

Sub-steps used: max 9 of n_max = 16; errflg = 0.

## Gradients — clamp = hard

Reverse vs forward mode: rel err 3.6e-15 (scan), 3.6e-15 (while). Input-direction FD rel err 2.8e-08.

| Parameter | dL/dp (reverse) | central FD | rel err |
|---|---|---|---|
| autoconv_rate | -5.733584e+01 | -5.733584e+01 | 1.5e-08 |
| autoconv_threshold | 1.373754e+01 | 1.373754e+01 | 1.9e-08 |
| accretion_coeff | -8.263681e-01 | -8.263681e-01 | 6.5e-11 |
| accretion_exp | 7.773700e+00 | 7.773700e+00 | 3.7e-11 |
| fallspeed_coeff | 2.201759e-02 | 2.201759e-02 | 7.8e-10 |
| fallspeed_exp | -8.896663e+00 | -8.896663e+00 | 4.4e-10 |
| vent_a | 8.920515e-04 | 8.920514e-04 | 8.8e-08 |
| vent_b | 6.357143e-05 | 6.357143e-05 | 1.8e-08 |
| vent_exp | -1.050330e-01 | -1.050330e-01 | 3.5e-08 |
| evap_exp | -1.231071e-01 | -1.231071e-01 | 7.3e-09 |
| evap_denom_c | -6.897686e-10 | -6.897686e-10 | 1.6e-08 |
| evap_denom_d | -1.408971e-08 | -1.408971e-08 | 3.1e-08 |
| tetens_e0 | -8.768298e-02 | -8.768298e-02 | 1.8e-09 |
| tetens_a | -3.189012e-02 | -3.189012e-02 | 3.8e-10 |
| tetens_t0 | 2.195587e-02 | 2.195587e-02 | 5.8e-11 |
| tetens_t1 | -2.271131e-03 | -2.271130e-03 | 1.2e-08 |
| cond_coeff | -9.142078e-06 | -9.142078e-06 | 1.9e-09 |

Exactly-zero gradients: none.

## Gradients — clamp = soft

Reverse vs forward mode: rel err 3.7e-15 (scan), 3.5e-15 (while). Input-direction FD rel err 9.0e-07.

| Parameter | dL/dp (reverse) | central FD | rel err |
|---|---|---|---|
| autoconv_rate | -5.723022e+01 | -5.723022e+01 | 1.5e-08 |
| autoconv_threshold | 1.372095e+01 | 1.372095e+01 | 3.3e-08 |
| accretion_coeff | -8.258854e-01 | -8.258854e-01 | 3.6e-10 |
| accretion_exp | 7.770330e+00 | 7.770330e+00 | 5.8e-11 |
| fallspeed_coeff | 2.211431e-02 | 2.211431e-02 | 1.2e-09 |
| fallspeed_exp | -8.937532e+00 | -8.937532e+00 | 1.0e-09 |
| vent_a | 9.089111e-04 | 9.089104e-04 | 7.2e-07 |
| vent_b | 6.514331e-05 | 6.514330e-05 | 7.9e-08 |
| vent_exp | -1.072666e-01 | -1.072666e-01 | 3.7e-08 |
| evap_exp | -1.256619e-01 | -1.256619e-01 | 1.6e-08 |
| evap_denom_c | -7.438993e-10 | -7.438991e-10 | 2.8e-07 |
| evap_denom_d | -1.424762e-08 | -1.424762e-08 | 9.3e-08 |
| tetens_e0 | -8.734004e-02 | -8.734004e-02 | 1.4e-09 |
| tetens_a | -3.225576e-02 | -3.225576e-02 | 3.6e-10 |
| tetens_t0 | 2.183426e-02 | 2.183426e-02 | 2.6e-08 |
| tetens_t1 | -2.299699e-03 | -2.299699e-03 | 9.4e-09 |
| cond_coeff | -8.960830e-06 | -8.960830e-06 | 1.8e-08 |

Exactly-zero gradients: none.

## G5 — soft → hard convergence (max |soft - hard| / max |hard|)

| soft_width | theta | qv | qc | qr | precl | relhum |
|---|---|---|---|---|---|---|
| 1e-05 | 5.3e-04 | 1.8e-03 | 9.3e-03 | 2.2e-03 | 1.1e-03 | 2.3e-01 |
| 1e-06 | 3.3e-05 | 2.2e-04 | 1.0e-03 | 1.8e-04 | 1.0e-04 | 8.3e-02 |
| 1e-07 | 4.9e-06 | 3.3e-05 | 1.1e-04 | 1.9e-05 | 9.2e-06 | 8.3e-03 |
| 1e-08 | 5.8e-07 | 3.7e-06 | 1.2e-05 | 2.1e-06 | 8.7e-07 | 8.3e-04 |

## Timing (informational)

- grad_ms[hard]: 16.8 ms
- grad_ms[soft]: 29.5 ms
- forward_ms[hard,scan]: 3.8 ms
- forward_ms[hard,while]: 2.5 ms
