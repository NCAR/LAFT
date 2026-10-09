# Bridge workflow report

- **Date**: 2026-10-09 12:20:11
- **Gate status**: **PASS**
- Gate rule: all bridge_test/test_*_layout.py tests pass, zero skips among them, no failures/errors anywhere; AND every translation-dependent test runs and passes (--require-dependent)

## Generated bridges

- `out/bridge/kessler_init_bridge.py`
- `out/bridge/kessler_run_bridge.py`

## Suite

Shared (LAFT/workflow_bridge/, contract derived from the packets):

- `workflow_bridge/test_generated_bridge_layout.py`
- `workflow_bridge/test_generated_bridge_device_entry.py`
- `workflow_bridge/test_direction_symmetry.py`

Per-project additions (bridge_test/):

- `bridge_test/test_kessler_run_bridge.py`

## Phase 03 analysis reports

- `kessler_init_analysis.txt`: Safety: UNKNOWN
- `kessler_run_analysis.txt`: Safety: UNSAFE — ⚠️ GPU efficiency warning

## Test gate (translation-independent)

| Test | Outcome |
|------|---------|
| `test_generated_bridge_layout::test_bridges_found` | passed |
| `test_generated_bridge_layout::test_every_generated_bridge_has_a_contract` | passed |
| `test_generated_bridge_layout::test_partition_array_vs_scalar` | passed |
| `TestArrayBridge::test_transfer_helpers[kessler_run]` | passed |
| `TestArrayBridge::test_two_entries_and_alias[kessler_run]` | passed |
| `TestArrayBridge::test_static_argnames_match_rule[kessler_run]` | passed |
| `TestArrayBridge::test_passthrough_roundtrip_identity[kessler_run]` | passed |
| `TestArrayBridge::test_outputs_c_order_and_dtype[kessler_run]` | passed |
| `TestArrayBridge::test_core_sees_reversed_layout[kessler_run]` | passed |
| `TestArrayBridge::test_static_scalars_concrete_and_others_traced[kessler_run]` | passed |
| `TestArrayBridge::test_character_args_stay_host_side[kessler_run]` | passed |
| `TestArrayBridge::test_optional_kwargs_forwarded[kessler_run]` | passed |
| `TestScalarBridge::test_no_device_entry_and_no_transfers[kessler_init]` | passed |
| `TestScalarBridge::test_wrapper_called_and_returns_passed_through[kessler_init]` | passed |

## Translation-dependent suite (informational)

State: **passed**

| Test | Outcome |
|------|---------|
| `test_generated_bridge_device_entry::test_array_bearing_procedures_known` | passed |
| `TestDeviceEntryContract::test_translation_present[kessler_run]` | passed |
| `TestDeviceEntryContract::test_public_entry_and_alias[kessler_run]` | passed |
| `TestDeviceEntryContract::test_device_entry_bit_identical_to_host_bridge[kessler_run]` | passed |
| `test_direction_symmetry::test_direction_symmetry_applicability` | passed |
| `TestBridgeFunctionality::test_runs_successfully` | passed |
| `TestBridgeFunctionality::test_bridge_vs_direct_jax` | passed |
| `TestBridgeFunctionality::test_driver_layout_preserved` | passed |
| `TestBridgeFunctionality::test_physics_sanity` | passed |
| `TestBridgeFunctionality::test_error_handling_negative_dt` | passed |
| `TestBridgeFunctionality::test_module_vars_inout_pattern` | passed |

These tests belong to the translator workflow's validation stage; `skipped` is the normal state before a translation exists.

Machine-readable verdict: `out/reports/bridge/bridge_test_results.json`
