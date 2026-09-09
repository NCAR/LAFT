#!/usr/bin/env python3
"""
compare_fortran_jax.py
----------------------
Compares Fortran kessler_run outputs (ground truth) against JAX driver outputs.

Both drivers read the SAME inputs from data/exp2_jd/fortran_io/inputs/ and write
their outputs to separate folders. This script loads both and diffs them.

Output locations
----------------
    Fortran : data/exp2_jd/fortran_io/outputs/   (written by kessler_fortran_driver)
    JAX     : out/driver/                (written by kessler_JAX_driver.py)

Workflow
--------
    # 1. Run JAX driver
    python tools/phase05_03_kessler_JAX_driver.py

    # 2. Compare
    python tools/compare_fortran_jax.py

    # 3. Check the report in terminal and also saved to out/driver/compare_results_fortran_jax.txt
"""

import sys
import json
import hashlib
import traceback
from datetime import datetime
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
FORTRAN_OUT = PROJECT_ROOT / "data" / "exp2_jd" / "fortran_io" / "outputs"
JAX_OUT     = PROJECT_ROOT / "out" / "driver"
INP_DIR     = PROJECT_ROOT / "data" / "exp2_jd" / "fortran_io" / "inputs"

# Output report file
REPORT_FILE = PROJECT_ROOT / "out" / "driver" / "compare_results_fortran_jax.txt"
# Machine-readable result (convenience; the text report + PBS log stay authoritative)
RESULT_JSON = PROJECT_ROOT / "out" / "driver" / "compare_fortran_jax.json"

# Variables to compare (name, is_2d)
VARIABLES = [
    ("theta",  True),
    ("qv",     True),
    ("qc",     True),
    ("qr",     True),
    ("precl",  False),
    ("relhum", True),
]

# Per-variable max-abs-error tolerance gates (editable). Derived from
# TRANSLATE_WORKFLOW.md: theta/relhum near ~1e-13, mixing ratios near ~1e-17 on a
# unit-scale grid; "MAE > ~1e-10 suspects a bug". These are conservative ceilings
# above which the diff is treated as a real disagreement, not round-off.
MAX_ABS_ERR_TOL = {
    "theta":  1e-9,
    "relhum": 1e-9,
    "qv":     1e-12,
    "qc":     1e-12,
    "qr":     1e-12,
    "precl":  1e-12,
}

VARIABLE_UNITS = {
    "theta": ("K", 1.0),
    "qv": ("raw", 1.0),
    "qc": ("raw", 1.0),
    "qr": ("raw", 1.0),
    "precl": ("raw", 1.0),
    "relhum": ("%", 1.0),
}

SEP  = "=" * 188
SEP2 = "-" * 188


# =============================================================================
# I/O helpers
# =============================================================================

def load_meta() -> dict:
    meta = {}
    with open(INP_DIR / "meta.txt") as f:
        for line in f:
            k, v = line.strip().split("=")
            meta[k] = float(v) if "." in v else int(v)
    return meta


def load_outputs(folder: Path, ncol: int, nz: int) -> dict:
    """Load all output variables from a folder."""
    out = {}
    for name, is_2d in VARIABLES:
        fpath = folder / f"{name}.txt"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing: {fpath}")
        raw = np.loadtxt(fpath)
        out[name] = raw.reshape(ncol, nz) if is_2d else raw.flatten()
    return out


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(jax_out: dict, fort_out: dict) -> dict:
    metrics = {}
    for name, _ in VARIABLES:
        a    = jax_out[name].astype(np.float64)
        b    = fort_out[name].astype(np.float64)
        diff = np.abs(a - b)
        metrics[name] = {
            "mae":         float(np.mean(diff)),
            "rmse":        float(np.sqrt(np.mean(diff**2))),
            "max_abs_err": float(np.max(diff)),
            "rel_mae":     float(np.mean(diff) / (np.mean(np.abs(b)) + 1e-30)),
        }
    return metrics


# =============================================================================
# Bitwise comparison
# =============================================================================
#
# Two levels, both exact — no tolerance anywhere in this section:
#
#   1. VALUE BITS. Every output is float64 on both sides, so "bitwise
#      identical" means the two IEEE-754 doubles have the same 64-bit pattern.
#      Where they differ, the ULP distance says HOW far apart they are in
#      representable-number steps: 1 ULP is the smallest possible disagreement,
#      the last bit of the mantissa. This is a strictly sharper statement than
#      max|err| — it is scale-free and does not need a tolerance to interpret.
#
#      This is trustworthy only because both drivers write 20 decimal digits;
#      float64 round-trips exactly through >=17 significant digits, so parsing
#      the text files recovers the original bits with no loss. If either writer
#      is ever narrowed below 17 digits, the text round-trip — not the physics
#      — becomes the thing this measures.
#
#   2. FILE BYTES. sha256 of the raw .txt files. Equal digests mean the two
#      drivers produced character-for-character identical output. This is
#      stricter than (1) (formatting counts) and is reported for provenance.
#
# Reported, NOT gated: an independent Fortran and JAX implementation agreeing
# to a few ULP is the expected, correct outcome — different summation order in
# a reduction is enough to move the last bit. PASS/FAIL stays on the tolerance
# gates in MAX_ABS_ERR_TOL. Bit-identity is evidence, not the criterion.


def _total_order_key(x: np.ndarray) -> np.ndarray:
    """Map float64 to a uint64 key whose integer order matches float order.

    Lets ULP distance be a plain subtraction. Non-negatives map above the
    midpoint, negatives are order-reversed below it, so the mapping is
    monotonic across zero and never overflows.
    """
    u = x.astype(np.float64).view(np.uint64)
    neg = (u >> np.uint64(63)).astype(bool)
    return np.where(neg, ~u, u + np.uint64(1 << 63))


def bitwise_metrics(jax_out: dict, fort_out: dict) -> dict:
    """Per-variable exact bit comparison of the float64 values."""
    out = {}
    for name, _ in VARIABLES:
        a = jax_out[name].astype(np.float64).ravel()
        b = fort_out[name].astype(np.float64).ravel()

        same_bits = a.view(np.uint64) == b.view(np.uint64)
        n_total = int(same_bits.size)
        n_same = int(np.count_nonzero(same_bits))

        finite = np.isfinite(a) & np.isfinite(b)
        if np.all(finite):
            ka, kb = _total_order_key(a), _total_order_key(b)
            ulp = np.where(ka > kb, ka - kb, kb - ka)
            max_ulp = int(ulp.max()) if ulp.size else 0
            mean_ulp = float(ulp.astype(np.float64).mean()) if ulp.size else 0.0
        else:
            # ULP is undefined against a NaN/Inf; the finite gate elsewhere
            # already fails this case, so just flag it rather than guess.
            max_ulp, mean_ulp = None, None

        out[name] = {
            "bitwise_identical": n_same == n_total,
            "n_total": n_total,
            "n_bit_identical": n_same,
            "n_differing": n_total - n_same,
            "frac_bit_identical": (n_same / n_total) if n_total else 1.0,
            "max_ulp_diff": max_ulp,
            "mean_ulp_diff": mean_ulp,
            "all_finite_both": bool(np.all(finite)),
        }
    return out


def file_digests() -> dict:
    """sha256 of the raw output text files — byte-level identity."""
    out = {}
    for name, _ in VARIABLES:
        fp, jp = FORTRAN_OUT / f"{name}.txt", JAX_OUT / f"{name}.txt"
        fd = hashlib.sha256(fp.read_bytes()).hexdigest() if fp.exists() else None
        jd = hashlib.sha256(jp.read_bytes()).hexdigest() if jp.exists() else None
        out[name] = {
            "fortran_sha256": fd,
            "jax_sha256": jd,
            "identical": bool(fd is not None and fd == jd),
        }
    return out


def physics_check(out: dict) -> dict:
    return {
        "theta_200_400K": bool(np.all(out["theta"] > 200) and np.all(out["theta"] < 400)),
        "qv_nonneg":      bool(np.all(out["qv"] >= 0)),
        "qc_nonneg":      bool(np.all(out["qc"] >= 0)),
        "qr_nonneg":      bool(np.all(out["qr"] >= 0)),
        "precl_nonneg":   bool(np.all(out["precl"] >= 0)),
        "relhum_nonneg":  bool(np.all(out["relhum"] >= 0)),
    }


def format_file_value(value: float) -> str:
    """Format values like the Fortran output files: 20 decimals, 3-digit exponent."""
    mantissa, exponent = f"{float(value):.20E}".split("E")
    exponent_i = int(exponent)
    sign = "+" if exponent_i >= 0 else "-"
    return f"{mantissa}E{sign}{abs(exponent_i):03d}"


# =============================================================================
# Comparison-stage validation (is the comparison itself trustworthy?)
# =============================================================================

def newest_mtime(folder: Path) -> float:
    """Newest mtime among the compared output files in a folder (0.0 if none)."""
    mtimes = [(folder / f"{n}.txt").stat().st_mtime
              for n, _ in VARIABLES if (folder / f"{n}.txt").exists()]
    return max(mtimes) if mtimes else 0.0


def shape_consistency(jax_out: dict, fort_out: dict) -> dict:
    """Confirm Fortran and JAX arrays have identical shapes per variable.

    A reshape mismatch (e.g. column- vs row-major write) would otherwise inflate
    the diff into a large-but-bogus disagreement, so this guards invariant #3.
    """
    out = {}
    for name, _ in VARIABLES:
        js, fs = list(jax_out[name].shape), list(fort_out[name].shape)
        out[name] = {"jax": js, "fortran": fs, "match": js == fs}
    return out


def identity_self_test(fort_out: dict) -> dict:
    """Diff the Fortran outputs against themselves: every metric MUST be 0.

    Validates the loader + metric code independently of the translation
    (invariant #4). Any non-zero value means the comparison harness is broken.
    """
    m = compute_metrics(fort_out, fort_out)
    worst = max(v["max_abs_err"] for v in m.values()) if m else 0.0
    return {"max_abs_err": float(worst), "passed": worst == 0.0}


def build_result_json(metrics: dict, fort_out: dict, jax_out: dict,
                      ncol: int, nz: int, meta: dict) -> dict:
    """Assemble the machine-readable comparison result with a computed PASS/FAIL.

    The text report + PBS log remain authoritative; this encodes the documented
    invariants and tolerance gates so the outcome is machine-checkable too.
    """
    shapes = shape_consistency(jax_out, fort_out)
    shapes_ok = all(s["match"] for s in shapes.values())

    fort_phys = physics_check(fort_out)
    jax_phys  = physics_check(jax_out)
    fort_phys_ok = all(fort_phys.values())
    jax_phys_ok  = all(jax_phys.values())

    self_test = identity_self_test(fort_out)

    fresh = {
        "inputs_mtime":  newest_mtime(INP_DIR),
        "fortran_mtime": newest_mtime(FORTRAN_OUT),
        "jax_mtime":     newest_mtime(JAX_OUT),
    }
    # Outputs should be at least as new as the inputs they were produced from.
    freshness_ok = (fresh["fortran_mtime"] >= fresh["inputs_mtime"]
                    and fresh["jax_mtime"] >= fresh["inputs_mtime"])

    per_var = {}
    metrics_ok = True
    for name, _ in VARIABLES:
        tol = MAX_ABS_ERR_TOL.get(name, 1e-9)
        ok = metrics[name]["max_abs_err"] <= tol
        metrics_ok = metrics_ok and ok
        per_var[name] = {**metrics[name], "tol_max_abs_err": tol, "within_tol": ok}

    overall = (shapes_ok and self_test["passed"] and fort_phys_ok
               and jax_phys_ok and freshness_ok and metrics_ok)

    # Exact bit comparison — recorded, deliberately not part of `overall`
    # (see the Bitwise comparison section header for why).
    bitwise = bitwise_metrics(jax_out, fort_out)
    digests = file_digests()
    bitwise_all = all(v["bitwise_identical"] for v in bitwise.values())
    ulps = [v["max_ulp_diff"] for v in bitwise.values() if v["max_ulp_diff"] is not None]

    return {
        "stage": "comparison",
        "status": "PASS" if overall else "FAIL",
        "timestamp": datetime.now().isoformat(),
        "grid": {"ncol": int(ncol), "nz": int(nz),
                 "dt": float(meta.get("dt", 0)), "seed": int(meta.get("seed", 0))},
        "invariants": {
            "shapes_match": shapes_ok,
            "identity_self_test_passed": self_test["passed"],
            "fortran_physics_all_pass": fort_phys_ok,
            "jax_physics_all_pass": jax_phys_ok,
            "outputs_fresh_vs_inputs": freshness_ok,
            "all_within_tolerance": metrics_ok,
        },
        "bitwise": {
            "gated": False,
            "note": ("exact IEEE-754 bit comparison of the float64 outputs, "
                     "reported as evidence; PASS/FAIL is set by the "
                     "MAX_ABS_ERR_TOL gates"),
            "all_variables_bit_identical": bitwise_all,
            "worst_max_ulp_diff": (max(ulps) if ulps else None),
            "per_variable": bitwise,
            "file_sha256": digests,
            "all_files_byte_identical": all(v["identical"] for v in digests.values()),
        },
        "self_test": self_test,
        "freshness": fresh,
        "shapes": shapes,
        "physics": {"fortran": fort_phys, "jax": jax_phys},
        "metrics": per_var,
        "exception": None,
    }


def write_result_json(result: dict) -> None:
    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


# =============================================================================
# Report builder
# =============================================================================

def build_report(metrics: dict, fort_out: dict, jax_out: dict,
                 ncol: int, nz: int, meta: dict) -> str:
    lines = []

    def add(line=""):
        lines.append(line)

    add(SEP)
    add("  Kessler Fortran vs JAX Comparison")
    add(SEP)
    add("")
    add(f"  Grid: ncol={ncol}, nz={nz}, dt={meta['dt']}, seed={int(meta['seed'])}")
    add("")
    add(f"[1] Loading Fortran outputs from {FORTRAN_OUT}...")
    add("   OK")
    add(f"[2] Loading JAX outputs from {JAX_OUT}...")
    add("   OK")
    add("[3] Computing metrics...")
    add("   OK")
    add("")
    add(SEP)
    add("  FORTRAN vs JAX — ACCURACY COMPARISON REPORT")
    add(SEP)
    add(f"  Grid        : {ncol} cols x {nz} levels")
    add(f"  Fortran dir : {FORTRAN_OUT}")
    add(f"  JAX dir     : {JAX_OUT}")

    # --- Diagnostics side by side ---
    add("")
    add("  OUTPUT DIAGNOSTICS")
    add("")
    add(
        f"  {'Variable':<10}  {'Unit':<8}  "
        f"{'F min':>28}  {'F max':>28}  "
        f"{'JAX min':>28}  {'JAX max':>28}"
    )
    add(SEP2)
    for name, _ in VARIABLES:
        unit, scale = VARIABLE_UNITS[name]
        fvals = fort_out[name] * scale
        jvals = jax_out[name] * scale
        add(
            f"  {name:<10}  {unit:<8}  "
            f"{format_file_value(fvals.min()):>28}  "
            f"{format_file_value(fvals.max()):>28}  "
            f"{format_file_value(jvals.min()):>28}  "
            f"{format_file_value(jvals.max()):>28}"
        )

    # --- Numerical accuracy ---
    add("")
    add("")
    add("  NUMERICAL ACCURACY  (JAX vs Fortran)")
    add("")
    add(f"  {'Variable':<10}  {'MAE':>16}  {'RMSE':>16}  {'Max|Err|':>16}  {'Rel MAE':>10}")
    add(f"  {'-'*10}  {'-'*16}  {'-'*16}  {'-'*16}  {'-'*10}")
    for name, _ in VARIABLES:
        m = metrics[name]
        add(
            f"  {name:<10}  {m['mae']:>16.6e}  {m['rmse']:>16.6e}  "
            f"{m['max_abs_err']:>16.6e}  {m['rel_mae']:>10.2e}"
        )

    # --- Bitwise ---
    bitwise = bitwise_metrics(jax_out, fort_out)
    digests = file_digests()
    add("")
    add("")
    add("  BITWISE COMPARISON  (exact IEEE-754 bits — no tolerance; reported, not a pass gate)")
    add("")
    add(f"  {'Variable':<10}  {'Bit-identical':>16}  {'Identical/Total':>20}  "
        f"{'Max ULP':>12}  {'Mean ULP':>12}  {'File bytes':>12}")
    add(f"  {'-'*10}  {'-'*16}  {'-'*20}  {'-'*12}  {'-'*12}  {'-'*12}")
    for name, _ in VARIABLES:
        b = bitwise[name]
        mu = "n/a" if b["max_ulp_diff"] is None else f"{b['max_ulp_diff']:d}"
        av = "n/a" if b["mean_ulp_diff"] is None else f"{b['mean_ulp_diff']:.3f}"
        add(
            f"  {name:<10}  {('YES' if b['bitwise_identical'] else 'no'):>16}  "
            f"{b['n_bit_identical']:>9d}/{b['n_total']:<10d}  "
            f"{mu:>12}  {av:>12}  "
            f"{('same' if digests[name]['identical'] else 'differ'):>12}"
        )
    ulps = [b["max_ulp_diff"] for b in bitwise.values() if b["max_ulp_diff"] is not None]
    add("")
    if all(b["bitwise_identical"] for b in bitwise.values()):
        add("  All variables are bit-for-bit identical to the Fortran reference.")
    else:
        worst = max(ulps) if ulps else None
        add(f"  Not bit-identical. Worst disagreement: {worst} ULP "
            f"(1 ULP = the last mantissa bit, the smallest representable difference).")
        add("  Expected for an independent implementation: reduction/order differences")
        add("  move the final bit. Correctness is judged by the tolerance gates above.")

    # --- Physics sanity ---
    add("")
    add("")
    add("  PHYSICS SANITY CHECKS")
    add("")
    for label, out in [("Fortran", fort_out), ("JAX", jax_out)]:
        checks = physics_check(out)
        all_ok = all(checks.values())
        status = "ALL PASS" if all_ok else "SOME FAIL"
        add(f"  {label}: {status}")
        for check, ok in checks.items():
            icon = "OK  " if ok else "FAIL"
            add(f"    [{icon}]  {check}")

    add("")
    add(SEP)
    add("")

    return "\n".join(lines)


def save_report(report_text: str, output_path: Path) -> None:
    output_path.write_text(report_text, encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def _fail_json(reason: str) -> None:
    """Write a minimal FAIL result for an early abort (missing files, etc.)."""
    write_result_json({
        "stage": "comparison",
        "status": "FAIL",
        "timestamp": datetime.now().isoformat(),
        "exception": reason,
    })


def main():
    print(SEP)
    print("  Kessler Fortran vs JAX Comparison")
    print(SEP)

    if not (INP_DIR / "meta.txt").exists():
        print("ERROR: meta.txt not found.")
        print("  Run first: python data/exp2_jd/src/generate_fortran_inputs_jd.py")
        _fail_json(f"meta.txt not found in {INP_DIR}")
        return 1

    meta = load_meta()
    ncol = meta["ncol"]
    nz   = meta["nz"]
    print(f"\n  Grid: ncol={ncol}, nz={nz}, dt={meta['dt']}, seed={int(meta['seed'])}")

    print(f"\n[1] Loading Fortran outputs from {FORTRAN_OUT}...")
    try:
        fort_out = load_outputs(FORTRAN_OUT, ncol, nz)
        print("   OK")
    except FileNotFoundError as e:
        print(f"   ERROR: {e}")
        print("   Run: ./data/exp2_jd/src/JD_kessler_driver")
        _fail_json(f"missing Fortran outputs: {e}")
        return 1

    print(f"[2] Loading JAX outputs from {JAX_OUT}...")
    try:
        jax_out = load_outputs(JAX_OUT, ncol, nz)
        print("   OK")
    except FileNotFoundError as e:
        print(f"   ERROR: {e}")
        print("   Run: python tools/phase05_03_kessler_JAX_driver.py")
        _fail_json(f"missing JAX outputs: {e}")
        return 1

    try:
        print("[3] Computing metrics...")
        metrics = compute_metrics(jax_out, fort_out)
        print("   OK")

        report_text = build_report(metrics, fort_out, jax_out, ncol, nz, meta)

        # Keep printing to terminal
        print(report_text)

        # Also save to file
        save_report(report_text, REPORT_FILE)
        print(f"Saved comparison report to: {REPORT_FILE}")

        # Machine-readable result with computed PASS/FAIL (convenience artifact).
        result = build_result_json(metrics, fort_out, jax_out, ncol, nz, meta)
        write_result_json(result)
        print(f"Saved comparison result JSON to: {RESULT_JSON}  (status={result['status']})")
    except Exception:
        # Record the traceback for inspection, then re-raise so the full
        # traceback lands in the PBS log (the authoritative signal).
        _fail_json(traceback.format_exc())
        raise

    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
