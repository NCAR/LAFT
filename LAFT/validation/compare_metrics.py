#!/usr/bin/env python3
"""
Shared Fortran-vs-JAX comparison METRICS (no pass/fail) — phase 05.3.

Computes, for every output variable, the same numbers each project's
comparison script used to compute by hand, as soon as the two data sets
exist: the reference Fortran output and the JAX driver output. It produces
metrics only. The acceptance criterion stays per project
([comparison].script applies it; `docs/README.md` §Generic vs
project-specific) — this engine never prints PASS or FAIL.

Per variable:
    shapes (reference, jax, match), element count, finiteness,
    min/max of each side, non-zero counts,
    max_abs_err  E_v = max_i |jax_i - ref_i|   (+ where: flat index)
    mae, rmse, corr,
    rel_max_abs_err = E_v / max_i |ref_i|,   rrmse = rmse / rms(ref),
    bitwise: n_bit_identical, frac, max/mean ULP distance (float64 both sides),
    match at the reference's own significant digits (when declared),
    sha256 of the files.
Plus an identity self-test (reference vs itself: every metric must be 0 —
proves the loader + metric code, not the translation).

Data layout — `[comparison.data]` in config/project.toml:

    [comparison.data]
    format     = "per_variable_files"   # one text file per variable, same
                                        #   name on both sides  (Kessler)
    reference  = "data/exp2_jd/fortran_io/outputs"   # directory
    jax        = "out/driver"                        # directory
    variables  = ["theta", "qv", "qc", "qr", "precl", "relhum"]

    [comparison.data]
    format     = "block_file"           # one whitespace table per side,
                                        #   one column per variable
    reference  = "_p3-reference/out_p3.dat"          # file
    jax        = "out/driver/out_p3_jax.dat"         # file
    variables  = [...]                  # column names (falls back to
                                        #   [comparison].columns)
    reference_sig_digits = 4            # optional: the reference writer's
                                        #   precision -> match_at_ref_digits
    # optional: [comparison.floor_columns] (or [comparison.data.floor]) —
    # {name = floor}: elements where BOTH sides sit at/below the floor are
    # "no signal" and excluded from the difference metrics (e.g. dBZ = -99)

Outputs, next to the JAX data:  compare_metrics.json, compare_metrics.txt.
Exit 0 when computed (or when the section is absent — reported, not an
error, so projects without a comparison are unaffected); 1 when the
declared data is missing or unreadable.

Run from the project root (pbsJobs/jax_gpu_compvalues.sh runs it before the
project's comparison script):
    python validation/compare_metrics.py [--config PATH]
Import:
    from compare_metrics import compute_metrics   # -> dict (same as the JSON)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from framework_config import get_config, init as init_config, add_config_arg  # noqa: E402


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _data_section(cfg) -> Optional[dict]:
    cmp_ = cfg.section("comparison")
    data = cmp_.get("data")
    if not isinstance(data, dict) or not data:
        return None
    d = dict(data)
    d.setdefault("variables", cmp_.get("columns", []))
    d.setdefault("block_rows", cmp_.get("block_rows"))
    floors = d.get("floor") or cmp_.get("floor_columns") or {}
    d["floor"] = {str(k): float(v) for k, v in dict(floors).items()}
    return d


def _sha(p: Path) -> Optional[str]:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def load_sides(cfg, data: dict) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], dict]:
    """-> (reference {var: array}, jax {var: array}, files {var or '_': {reference, jax}})"""
    root = Path(cfg.root)
    fmt = data["format"]
    variables: List[str] = list(data["variables"])
    if not variables:
        raise ValueError("[comparison.data].variables is empty (and [comparison].columns is absent)")
    ref_p, jax_p = root / data["reference"], root / data["jax"]
    files: dict = {}

    if fmt == "per_variable_files":
        ref, jax = {}, {}
        for v in variables:
            fr, fj = ref_p / f"{v}.txt", jax_p / f"{v}.txt"
            for p in (fr, fj):
                if not p.exists():
                    raise FileNotFoundError(f"missing data file: {p}")
            ref[v] = np.loadtxt(fr, dtype=np.float64)
            jax[v] = np.loadtxt(fj, dtype=np.float64)
            files[v] = {"reference": str(fr), "jax": str(fj),
                        "reference_sha256": _sha(fr), "jax_sha256": _sha(fj)}
    elif fmt == "block_file":
        for p in (ref_p, jax_p):
            if not p.exists():
                raise FileNotFoundError(f"missing data file: {p}")
        r, x = np.loadtxt(ref_p, dtype=np.float64), np.loadtxt(jax_p, dtype=np.float64)
        r, x = np.atleast_2d(r), np.atleast_2d(x)
        if r.shape[1] != len(variables) or x.shape[1] != len(variables):
            raise ValueError(f"column count: reference {r.shape[1]}, jax {x.shape[1]}, "
                             f"declared variables {len(variables)}")
        ref = {v: r[:, i] for i, v in enumerate(variables)}
        jax = {v: x[:, i] for i, v in enumerate(variables)}
        files["_"] = {"reference": str(ref_p), "jax": str(jax_p),
                      "reference_sha256": _sha(ref_p), "jax_sha256": _sha(jax_p),
                      "reference_rows": int(r.shape[0]), "jax_rows": int(x.shape[0])}
    else:
        raise ValueError(f"unknown [comparison.data].format {fmt!r} "
                         "(per_variable_files | block_file)")
    return ref, jax, files


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _total_order_key(x: np.ndarray) -> np.ndarray:
    """float64 -> uint64 key whose integer order matches float order, so ULP
    distance is a plain subtraction (monotonic across zero, no overflow)."""
    u = np.ascontiguousarray(x, dtype=np.float64).view(np.uint64)
    neg = (u >> np.uint64(63)).astype(bool)
    return np.where(neg, ~u, u + np.uint64(1 << 63))


def _round_sig(x: np.ndarray, digits: int) -> np.ndarray:
    """Round to `digits` significant digits (0 stays 0)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    nz = x != 0
    if np.any(nz):
        v = x[nz]
        factor = np.power(10.0, (digits - 1) - np.floor(np.log10(np.abs(v))))
        out[nz] = np.round(v * factor) / factor
    return out


def _f(x) -> Optional[float]:
    """JSON-safe float (NaN/Inf -> None)."""
    x = float(x)
    return x if np.isfinite(x) else None


def variable_metrics(ref: np.ndarray, jax: np.ndarray, floor: Optional[float] = None,
                     ref_sig_digits: Optional[int] = None) -> Dict[str, Any]:
    r_full, x_full = np.asarray(ref, dtype=np.float64), np.asarray(jax, dtype=np.float64)
    m: Dict[str, Any] = {
        "shape_reference": list(r_full.shape), "shape_jax": list(x_full.shape),
        "shapes_match": r_full.shape == x_full.shape,
        "n": int(r_full.size),
        "reference_all_finite": bool(np.all(np.isfinite(r_full))),
        "jax_all_finite": bool(np.all(np.isfinite(x_full))),
        "reference_min": _f(np.min(r_full)) if r_full.size else None,
        "reference_max": _f(np.max(r_full)) if r_full.size else None,
        "jax_min": _f(np.min(x_full)) if x_full.size else None,
        "jax_max": _f(np.max(x_full)) if x_full.size else None,
        "reference_nonzero": int(np.count_nonzero(r_full)),
        "jax_nonzero": int(np.count_nonzero(x_full)),
    }
    if not m["shapes_match"]:
        n = min(r_full.size, x_full.size)
        r, x = r_full.ravel()[:n], x_full.ravel()[:n]
        m["note"] = f"shapes differ; difference metrics over the first {n} elements"
    else:
        r, x = r_full.ravel(), x_full.ravel()

    # --- difference metrics (optionally above a "no signal" floor) ---
    if floor is not None:
        mask = (r > floor) | (x > floor)
        m["floor"] = floor
        m["n_compared"] = int(np.count_nonzero(mask))
        rc, xc = r[mask], x[mask]
    else:
        m["n_compared"] = int(r.size)
        rc, xc = r, x
    if rc.size:
        diff = xc - rc
        adiff = np.abs(diff)
        imax = int(np.argmax(adiff))
        ref_absmax = float(np.max(np.abs(rc)))
        ref_rms = float(np.sqrt(np.mean(rc ** 2)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        m.update({
            "max_abs_err": _f(adiff[imax]),
            "max_abs_err_at": (int(np.flatnonzero(mask)[imax]) if floor is not None else imax),
            "reference_at_max": _f(rc[imax]), "jax_at_max": _f(xc[imax]),
            "mae": _f(np.mean(adiff)),
            "rmse": _f(rmse),
            "rel_max_abs_err": _f(adiff[imax] / ref_absmax) if ref_absmax > 0 else None,
            "rrmse": _f(rmse / ref_rms) if ref_rms > 0 else None,
            # correlation over the FULL column (floor included), as the original
            # per-project scripts computed it — the floor only masks the differences
            "corr": (_f(np.corrcoef(r, x)[0, 1]) if np.std(r) > 0 and np.std(x) > 0 else None),
        })
    else:
        m.update({"max_abs_err": None, "max_abs_err_at": None, "reference_at_max": None,
                  "jax_at_max": None, "mae": None, "rmse": None, "rel_max_abs_err": None,
                  "rrmse": None, "corr": None})

    # --- exact comparisons (full arrays, no floor) ---
    same_bits = r.view(np.uint64) == x.view(np.uint64)
    n_total, n_same = int(same_bits.size), int(np.count_nonzero(same_bits))
    finite = np.isfinite(r) & np.isfinite(x)
    if n_total and np.all(finite):
        kr, kx = _total_order_key(r), _total_order_key(x)
        ulp = np.where(kr > kx, kr - kx, kx - kr)
        max_ulp, mean_ulp = int(ulp.max()), float(ulp.astype(np.float64).mean())
    else:
        max_ulp, mean_ulp = None, None
    m["bitwise"] = {
        "identical": n_same == n_total,
        "n_bit_identical": n_same, "n_total": n_total,
        "frac_bit_identical": (n_same / n_total) if n_total else 1.0,
        "max_ulp_diff": max_ulp, "mean_ulp_diff": mean_ulp,
    }
    if ref_sig_digits:
        rr, xr = _round_sig(r, ref_sig_digits), _round_sig(x, ref_sig_digits)
        agree = int(np.count_nonzero(rr == xr))
        m["match_at_ref_digits"] = {
            "digits": int(ref_sig_digits), "n_match": agree, "n_total": n_total,
            "frac": (agree / n_total) if n_total else 1.0, "all": agree == n_total,
        }
    return m


def compute_metrics(cfg=None) -> Dict[str, Any]:
    cfg = cfg or get_config()
    data = _data_section(cfg)
    stamp = datetime.now().isoformat(timespec="seconds")
    if data is None:
        return {"stage": "comparison_metrics", "status": "not_configured", "timestamp": stamp,
                "note": "no [comparison.data] in config/project.toml — nothing computed"}
    ref, jax, files = load_sides(cfg, data)
    floors = data.get("floor", {})
    digits = data.get("reference_sig_digits")
    per_var = {v: variable_metrics(ref[v], jax[v], floors.get(v), digits) for v in data["variables"]}
    self_test = {v: variable_metrics(ref[v], ref[v], floors.get(v), digits) for v in data["variables"]}
    worst = max((s["max_abs_err"] or 0.0) for s in self_test.values())
    worst_ulp = [m["bitwise"]["max_ulp_diff"] for m in per_var.values()
                 if m["bitwise"]["max_ulp_diff"] is not None]
    return {
        "stage": "comparison_metrics",
        "status": "computed",
        "timestamp": stamp,
        "project": cfg.project_name,
        "data": {"format": data["format"], "reference": data["reference"], "jax": data["jax"],
                 "variables": list(data["variables"]), "block_rows": data.get("block_rows"),
                 "reference_sig_digits": digits, "floor": floors},
        "files": files,
        "summary": {
            "all_shapes_match": all(m["shapes_match"] for m in per_var.values()),
            "jax_all_finite": all(m["jax_all_finite"] for m in per_var.values()),
            "worst_max_abs_err": _f(max((m["max_abs_err"] or 0.0) for m in per_var.values())),
            "worst_rel_max_abs_err": _f(max((m["rel_max_abs_err"] or 0.0) for m in per_var.values())),
            "worst_rrmse": _f(max((m["rrmse"] or 0.0) for m in per_var.values())),
            "all_bitwise_identical": all(m["bitwise"]["identical"] for m in per_var.values()),
            "worst_max_ulp_diff": (max(worst_ulp) if worst_ulp else None),
            "all_match_at_ref_digits": (all(m["match_at_ref_digits"]["all"] for m in per_var.values())
                                        if digits else None),
            "files_byte_identical": all(f["reference_sha256"] == f["jax_sha256"] for f in files.values()),
            "identity_self_test_passed": worst == 0.0,
        },
        "metrics": per_var,
        "note": ("metrics only — the acceptance criterion is the project's "
                 "[comparison].script; see docs/README.md §Generic vs project-specific"),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _g(x, fmt="{:.4g}"):
    return "n/a" if x is None else fmt.format(x)


def render_text(result: Dict[str, Any]) -> str:
    L = []
    L.append("=" * 100)
    L.append(f"{result.get('project', '')} — Fortran vs JAX comparison METRICS (no pass/fail; "
             f"criterion = the project's comparison script)")
    L.append(f"  run: {result['timestamp']}   status: {result['status']}")
    if result["status"] != "computed":
        L.append(f"  {result.get('note', '')}")
        return "\n".join(L) + "\n"
    d = result["data"]
    L.append(f"  format: {d['format']}   reference: {d['reference']}   jax: {d['jax']}")
    if d.get("reference_sig_digits"):
        L.append(f"  reference precision declared: {d['reference_sig_digits']} significant digits")
    if d.get("floor"):
        L.append(f"  floors (no-signal, excluded from difference metrics): {d['floor']}")
    L.append("=" * 100)
    L.append(f"{'variable':>12s} {'n':>7s} {'ref_max':>11s} {'jax_max':>11s} {'max|err|':>11s} "
             f"{'rel':>9s} {'MAE':>11s} {'RMSE':>11s} {'rRMSE':>9s} {'corr':>8s} {'bit-id':>11s} {'maxULP':>12s}")
    L.append("-" * 100)
    for v, m in result["metrics"].items():
        b = m["bitwise"]
        L.append(f"{v:>12s} {m['n']:>7d} {_g(m['reference_max']):>11s} {_g(m['jax_max']):>11s} "
                 f"{_g(m['max_abs_err'], '{:.3e}'):>11s} {_g(m['rel_max_abs_err'], '{:.2e}'):>9s} "
                 f"{_g(m['mae'], '{:.3e}'):>11s} {_g(m['rmse'], '{:.3e}'):>11s} "
                 f"{_g(m['rrmse'], '{:.2e}'):>9s} {_g(m['corr'], '{:.5f}'):>8s} "
                 f"{b['n_bit_identical']:>5d}/{b['n_total']:<5d} {_g(b['max_ulp_diff'], '{:d}'):>12s}")
    s = result["summary"]
    L.append("-" * 100)
    L.append(f"  shapes match: {s['all_shapes_match']}   jax finite: {s['jax_all_finite']}   "
             f"identity self-test: {'ok' if s['identity_self_test_passed'] else 'FAILED'}")
    L.append(f"  worst max|err|: {_g(s['worst_max_abs_err'], '{:.3e}')}   worst rel: "
             f"{_g(s['worst_rel_max_abs_err'], '{:.2e}')}   worst rRMSE: {_g(s['worst_rrmse'], '{:.2e}')}")
    L.append(f"  all bitwise identical: {s['all_bitwise_identical']}   worst max ULP: "
             f"{_g(s['worst_max_ulp_diff'], '{:d}')}   files byte-identical: {s['files_byte_identical']}")
    if s.get("all_match_at_ref_digits") is not None:
        L.append(f"  all match at {d['reference_sig_digits']} significant digits: {s['all_match_at_ref_digits']}")
    L.append("=" * 100)
    L.append(result["note"])
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Fortran-vs-JAX comparison metrics (no pass/fail)")
    add_config_arg(ap)
    ap.add_argument("--out-dir", default=None,
                    help="where to write compare_metrics.{json,txt} (default: next to the JAX data)")
    args = ap.parse_args()
    init_config(args.config)
    cfg = get_config()
    try:
        result = compute_metrics(cfg)
    except (FileNotFoundError, ValueError, OSError) as e:
        print(f"❌ compare_metrics: {e}")
        return 1
    data = _data_section(cfg)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif data:
        jp = Path(cfg.root) / data["jax"]
        out_dir = jp if jp.is_dir() else jp.parent
    else:
        out_dir = Path(cfg.root) / "out" / "driver"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    text = render_text(result)
    (out_dir / "compare_metrics.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"Written: {out_dir / 'compare_metrics.json'}\nWritten: {out_dir / 'compare_metrics.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
