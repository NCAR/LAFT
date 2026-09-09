#!/usr/bin/env python3
"""
generate_fortran_inputs_jd.py
-----------------------------
Generates atmospheric inputs mirroring the initialization in
JD_kessler_driver.F90 and saves them to data/fortran_io/inputs/ so the
JAX driver (translations/gemini/driver/kessler_JAX_driver.py) can load them.

Per-column scaling factor arr[i] ~ N(mean=1, std=0.1) via Box-Muller,
matching the Fortran driver's random-number approach with a fixed seed.

Usage
-----
    cd kessler
    python data/src_jd/generate_fortran_inputs_jd.py

Output
------
    data/fortran_io/inputs/
        cpair.txt   rair.txt    rho.txt
        z.txt       pk.txt      theta.txt
        qv.txt      qc.txt      qr.txt
        precl.txt   meta.txt
"""

import sys
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Grid / time — matches JD_kessler_driver.F90
# ---------------------------------------------------------------------------
NCOL     = 128
NZ       = 56
DT       = 60.0
LYR_SURF = 1
LYR_TOA  = NZ
SEED     = 42   # fixed seed for reproducibility

# ---------------------------------------------------------------------------


def create_atmospheric_profile(ncol: int, nz: int, seed: int) -> dict:
    """
    Mirrors the initialization in JD_kessler_driver.F90.

    arr(i) = 1.0 + 0.1 * ztmp   [Box-Muller, mean=1, std=0.1]

    z(i,k)     = arr(i) * 100.0 * (k-1)          [m]
    rho(i,k)   = arr(i) * 1.2  * exp(-z/8000)    [kg/m^3]
    pk(i,k)    = arr(i) * 1.0                     [Exner-like]
    theta(i,k) = arr(i) * (300 - 0.006*z)         [K]
    qv(i,k)    = arr(i) * 0.010                   [kg/kg]
    qc(i,k)    = arr(i) * 0.01                    [kg/kg]
    qr(i,k)    = arr(i) * 0.01                    [kg/kg]
    cpair(i,k) = 1004.0                           [J/(kg K)]
    rair(i,k)  = 287.0                            [J/(kg K)]
    precl(i)   = 0.0                              [m/s, set by kessler_run]
    """
    rng = np.random.default_rng(seed)

    # Per-column scaling: mean=1, stddev=0.1  (mirrors Fortran arr(i))
    arr = 1.0 + 0.1 * rng.standard_normal(ncol)   # shape (ncol,)

    # Heights (m): Fortran k=1..nz → Python index k=0..nz-1 → (k-1) offset = k index
    k_idx = np.arange(nz, dtype=np.float64)        # 0, 1, ..., nz-1
    z = arr[:, np.newaxis] * (100.0 * k_idx)       # (ncol, nz)

    # Dry-air constants (uniform)
    cpair = np.full((ncol, nz), 1004.0)
    rair  = np.full((ncol, nz), 287.0)

    # Density
    rho = arr[:, np.newaxis] * (1.2 * np.exp(-z / 8000.0))

    # Exner-like pressure function
    pk = arr[:, np.newaxis] * np.ones((ncol, nz))

    # Potential temperature
    theta = arr[:, np.newaxis] * (300.0 - 0.006 * z)

    # Water vapour (uniform per column)
    qv = arr[:, np.newaxis] * np.full((ncol, nz), 0.010)

    # Cloud water
    qc = arr[:, np.newaxis] * np.full((ncol, nz), 0.01)

    # Rain water
    qr = arr[:, np.newaxis] * np.full((ncol, nz), 0.01)

    # Surface precipitation: zero initial (set by kessler_run)
    precl = np.zeros(ncol)

    return dict(
        ncol=ncol, nz=nz, dt=DT,
        lyr_surf=LYR_SURF, lyr_toa=LYR_TOA,
        cpair=cpair, rair=rair, rho=rho, z=z, pk=pk,
        theta=theta, qv=qv, qc=qc, qr=qr, precl=precl,
    )


def save_2d(path: Path, arr: np.ndarray) -> None:
    """Save (ncol, nz) array: one row per column, columns = levels."""
    np.savetxt(path, arr, fmt='%.20E')


def save_1d(path: Path, arr: np.ndarray) -> None:
    np.savetxt(path, arr.reshape(1, -1), fmt='%.20E')


def main():
    print("=" * 60)
    print("  Kessler — Fortran input generator (JD driver)")
    print(f"  Grid  : {NCOL} cols x {NZ} levels,  dt={DT} s")
    print(f"  Seed  : {SEED}")
    print("=" * 60)

    inp_dir = PROJECT_ROOT / "data" / "fortran_io" / "inputs"
    out_dir = PROJECT_ROOT / "data" / "fortran_io" / "outputs"

    inp_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Input  dir : {inp_dir}")
    print(f"  Output dir : {out_dir}")

    print("\n[1] Generating atmospheric profile...")
    atm = create_atmospheric_profile(NCOL, NZ, seed=SEED)

    print("[2] Saving input files...")
    save_2d(inp_dir / "cpair.txt",  atm["cpair"])
    save_2d(inp_dir / "rair.txt",   atm["rair"])
    save_2d(inp_dir / "rho.txt",    atm["rho"])
    save_2d(inp_dir / "z.txt",      atm["z"])
    save_2d(inp_dir / "pk.txt",     atm["pk"])
    save_2d(inp_dir / "theta.txt",  atm["theta"])
    save_2d(inp_dir / "qv.txt",     atm["qv"])
    save_2d(inp_dir / "qc.txt",     atm["qc"])
    save_2d(inp_dir / "qr.txt",     atm["qr"])
    save_1d(inp_dir / "precl.txt",  atm["precl"])

    with open(inp_dir / "meta.txt", "w") as f:
        f.write(f"ncol={NCOL}\n")
        f.write(f"nz={NZ}\n")
        f.write(f"dt={DT}\n")
        f.write(f"lyr_surf={LYR_SURF}\n")
        f.write(f"lyr_toa={LYR_TOA}\n")
        f.write(f"seed={SEED}\n")

    print("   Saved: cpair  rair  rho  z  pk  theta  qv  qc  qr  precl  meta")

    print("\n[3] Input diagnostics:")
    print(f"   theta  : {atm['theta'].min():.2f} - {atm['theta'].max():.2f} K")
    print(f"   rho    : {atm['rho'].min():.4f} - {atm['rho'].max():.4f} kg/m^3")
    print(f"   z      : {atm['z'].min():.1f}  - {atm['z'].max():.1f}  m")
    print(f"   pk     : {atm['pk'].min():.4f} - {atm['pk'].max():.4f}")
    print(f"   qv     : {atm['qv'].min()*1000:.4f} - {atm['qv'].max()*1000:.4f} g/kg")
    print(f"   qc     : {atm['qc'].mean()*1000:.4f} g/kg (mean)")
    print(f"   qr     : {atm['qr'].mean()*1000:.4f} g/kg (mean)")

    print("\nDone.")


if __name__ == "__main__":
    main()
