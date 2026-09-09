# Bridge workflow report

- **Date**: 2026-09-04 10:32:06
- **Gate status**: **PASS**
- Gate rule: all bridge_test/test_*_layout.py tests pass, zero skips among them, no failures/errors anywhere; AND every translation-dependent test runs and passes (--require-dependent)

## Generated bridges

- `out/bridge/kessler_init_bridge.py`
- `out/bridge/kessler_run_bridge.py`

## Phase 03 analysis reports

- `kessler_init_analysis.txt`: Safety: UNKNOWN
- `kessler_run_analysis.txt`: Safety: UNSAFE — ⚠️ GPU efficiency warning

## Test gate (translation-independent)

| Test | Outcome |
|------|---------|
| `TestTransferHelpers::test_to_device_no_host_permute` | passed |
| `TestTransferHelpers::test_roundtrip_identity` | passed |
| `TestTransferHelpers::test_to_host_c_order` | passed |
| `TestBridgeWiring::test_passthrough_roundtrip_identity` | passed |
| `TestBridgeWiring::test_outputs_c_order_layout` | passed |
| `TestBridgeWiring::test_core_receives_reversed_layout` | passed |
| `TestBridgeWiring::test_static_scalars_concrete_at_trace` | passed |
| `TestBridgeWiring::test_strings_stay_host_side` | passed |
| `TestBridgeWiring::test_module_vars_threaded` | passed |

## Translation-dependent suite (informational)

State: **passed**

| Test | Outcome |
|------|---------|
| `TestDeviceEntryContract::test_public_entry_and_alias` | passed |
| `TestDeviceEntryContract::test_device_entry_bit_identical_to_host_bridge` | passed |
| `TestBridgeFunctionality::test_runs_successfully` | passed |
| `TestBridgeFunctionality::test_bridge_vs_direct_jax` | passed |
| `TestBridgeFunctionality::test_driver_layout_preserved` | passed |
| `TestBridgeFunctionality::test_physics_sanity` | passed |
| `TestBridgeFunctionality::test_error_handling_negative_dt` | passed |
| `TestBridgeFunctionality::test_module_vars_inout_pattern` | passed |

These tests belong to the translator workflow's validation stage; `skipped` is the normal state before a translation exists.

Machine-readable verdict: `out/reports/bridge/bridge_test_results.json`
