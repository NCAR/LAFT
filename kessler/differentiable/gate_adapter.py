"""
gate_adapter — Kessler hooks for LAFT/workflow_differentiable/phase06_01_grad_gate.py.

The gate is project-agnostic; this file supplies everything Kessler-shaped:
the reference case (the 128 x 56 Fortran inputs the translations were validated
on), how to call the differentiable module and the validated reference
translation, the Fortran ground truth, and the scalar loss used for the
gradient checks. Interface (all required):

    load_case() -> dict
    default_params() -> pytree
    input_fields() -> list[str]        # inputs whose gradients are checked
    output_fields() -> list[str]
    run(case, params, *, inputs=None, clamp, soft_width, loop, n_max) -> dict
    run_reference(case, reference_path) -> dict   # validated translation
    fortran_reference() -> dict | None
    loss(outputs, case) -> scalar
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from kessler_run_diff import DEFAULT_PARAMS, kessler_run_diff  # noqa: E402

FORTRAN_IO = ROOT / "data" / "exp2_jd" / "fortran_io"
ARRAYS = ["cpair", "rair", "rho", "z", "pk", "theta", "qv", "qc", "qr"]
OUTPUTS = ["theta", "qv", "qc", "qr", "precl", "relhum"]
# kessler_init(lv_in, pref_in, rhoqr_in) as called by JD_kessler_driver.F90;
# pref is stored in hPa (pref_in / 100), the value the translations' core takes.
LV, PREF_HPA, RHOQR = 2.5e6, 1000.0, 1000.0


def _meta():
    out = {}
    for line in (FORTRAN_IO / "inputs" / "meta.txt").read_text().split():
        k, v = line.split("=")
        out[k] = float(v) if "." in v else int(v)
    return out


def load_case() -> dict:
    meta = _meta()
    ncol, nz = meta["ncol"], meta["nz"]
    arrays = {n: jnp.asarray(np.loadtxt(FORTRAN_IO / "inputs" / f"{n}.txt").reshape(ncol, nz).T)
              for n in ARRAYS}  # Fortran (ncol, nz) -> bridge layout (nz, ncol)
    case = {"ncol": ncol, "nz": nz, "dt": meta["dt"], "lyr_surf": meta["lyr_surf"],
            "lyr_toa": meta["lyr_toa"], "arrays": arrays}
    ref = fortran_reference()
    # Loss normalisation: each output's mean magnitude in the Fortran answer, so
    # every field contributes O(1) to the scalar loss.
    case["loss_scales"] = {k: float(np.abs(v).mean()) for k, v in ref.items()}
    return case


def default_params():
    return DEFAULT_PARAMS


def input_fields():
    return ["theta", "qv", "qc", "qr"]


def output_fields():
    return list(OUTPUTS)


def run(case, params, *, inputs=None, clamp="hard", soft_width=1e-6, loop="scan", n_max=32):
    a = {**case["arrays"], **(inputs or {})}
    o = kessler_run_diff(case["dt"], *(a[n] for n in ARRAYS), LV, PREF_HPA, RHOQR,
                         params=params, clamp=clamp, soft_width=soft_width, loop=loop,
                         n_max=n_max, lyr_surf=case["lyr_surf"], lyr_toa=case["lyr_toa"])
    return o._asdict()


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"_ref_{path.stem.replace('.', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_reference(case, reference_path: str) -> dict:
    mod = _load_module(ROOT / reference_path)
    a = case["arrays"]
    ncol, nz = case["ncol"], case["nz"]
    out = mod.kessler_run_core(ncol, nz, case["dt"], case["lyr_surf"], case["lyr_toa"],
                               *(a[n] for n in ARRAYS), jnp.zeros(ncol), jnp.zeros((nz, ncol)),
                               0, LV, PREF_HPA, RHOQR)
    return dict(zip(OUTPUTS, out[:6]))


def fortran_reference() -> dict:
    meta = _meta()
    ncol, nz = meta["ncol"], meta["nz"]
    d = FORTRAN_IO / "outputs"
    out = {n: np.loadtxt(d / f"{n}.txt").reshape(ncol, nz).T for n in OUTPUTS if n != "precl"}
    out["precl"] = np.loadtxt(d / "precl.txt").ravel()
    return out


def loss(outputs, case):
    s = case["loss_scales"]
    return sum(jnp.mean(outputs[k]) / s[k] for k in OUTPUTS)
