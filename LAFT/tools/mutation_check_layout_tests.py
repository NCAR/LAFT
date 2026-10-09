#!/usr/bin/env python3
"""
Mutation check: is the SHARED generic layout test
(LAFT/workflow_bridge/test_generated_bridge_layout.py) at least as strong as
the HAND-WRITTEN Kessler layout test
(kessler/bridge_test/test_kessler_run_bridge_layout.py)?

Builds a throw-away copy of kessler (config + out/{bridge,packets,modules,
phase1_index.json} + the hand-written layout test + a workflow_bridge symlink
to the real LAFT), applies one deliberate defect at a time to the generated
out/bridge/kessler_run_bridge.py, and runs BOTH test files on each mutant.
A mutant is "caught" by a file when that file has >= 1 failing/erroring test.
The control (no mutation) must be passed by both.

Results of the 2026-10-09 run: docs/LAYOUT_TEST_MUTATION_CHECK_2026-10-09.md.

Run (jax-validate env; CPU only, a few seconds per mutant):
    python LAFT/tools/mutation_check_layout_tests.py [--scratch DIR]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LAFT = Path(__file__).resolve().parent.parent
REPO = LAFT.parent
SRC = REPO / "kessler"

HAND = "bridge_test/test_kessler_run_bridge_layout.py"
GENERIC = "workflow_bridge/test_generated_bridge_layout.py"
BRIDGE = "out/bridge/kessler_run_bridge.py"
# The hand-written file was retired from kessler/bridge_test on 2026-10-09 (superseded
# by the shared file). The check reads it from this repository's history so it stays
# reproducible: this is the last commit that carried it.
HAND_COMMIT = "2455961"


def build_scratch(scratch: Path):
    if scratch.exists():
        shutil.rmtree(scratch)
    (scratch / "config").mkdir(parents=True)
    shutil.copy(SRC / "config/project.toml", scratch / "config/project.toml")
    (scratch / "out").mkdir()
    shutil.copy(SRC / "out/__init__.py", scratch / "out/__init__.py")
    for sub in ("bridge", "packets", "modules"):
        shutil.copytree(SRC / "out" / sub, scratch / "out" / sub,
                        ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(SRC / "out/phase1_index.json", scratch / "out/phase1_index.json")
    (scratch / "out/reports/bridge").mkdir(parents=True)
    (scratch / "bridge_test").mkdir()
    if (SRC / HAND).exists():
        shutil.copy(SRC / HAND, scratch / HAND)
    else:
        text = subprocess.run(["git", "-C", str(REPO), "show", f"{HAND_COMMIT}:kessler/{HAND}"],
                              capture_output=True, text=True, check=True).stdout
        (scratch / HAND).write_text(text, encoding="utf-8")
    os.symlink(LAFT / "workflow_bridge", scratch / "workflow_bridge")


def replace_nth(text, old, new, n):
    """Replace the n-th (1-based) occurrence of old; assert it exists."""
    parts = text.split(old)
    assert len(parts) > n, f"anchor occurs {len(parts)-1}x, need >= {n}: {old!r}"
    return old.join(parts[:n]) + new + old.join(parts[n:])


def replace_once(text, old, new):
    assert text.count(old) == 1, f"anchor occurs {text.count(old)}x, need exactly 1: {old!r}"
    return text.replace(old, new)


# Anchors are exact substrings of the phase03 output for kessler_run (PATH C,
# contract 2). If the generator's emitted text changes, the assertion in the
# replace helpers names the anchor that no longer matches.
MUTANTS = {
    "M0 control (no mutation)":
        lambda t: t,
    "M1 swap two array args at the core call (qv<->qc)":
        lambda t: replace_once(t,
            "= kessler_run_core(ncol, nz, dt, lyr_surf, lyr_toa, cpair, rair, rho, z, pk, theta, qv, qc,",
            "= kessler_run_core(ncol, nz, dt, lyr_surf, lyr_toa, cpair, rair, rho, z, pk, theta, qc, qv,"),
    "M2 drop a module var from the host return (rhoqr)":
        lambda t: replace_once(t, "errflg_fortran, lv, pref, rhoqr", "errflg_fortran, lv, pref"),
    "M3 missing in-jit INPUT reversal (theta)":
        lambda t: replace_nth(t, "    theta = jnp.transpose(theta)\n", "", 1),
    "M4 missing in-jit OUTPUT reversal (qv)":
        lambda t: replace_nth(t, "    qv = jnp.transpose(qv)\n", "", 2),
    "M5 real scalar made jit-static (dt)":
        lambda t: replace_once(t, "static_argnames=('ncol', 'nz', 'lyr_surf', 'lyr_toa', 'errflg',)",
                               "static_argnames=('ncol', 'nz', 'dt', 'lyr_surf', 'lyr_toa', 'errflg',)"),
    "M6 swap two return slots of the device entry (precl<->relhum)":
        lambda t: replace_once(t, "    return theta, qv, qc, qr, precl, relhum, errflg, lv, pref, rhoqr",
                               "    return theta, qv, qc, qr, relhum, precl, errflg, lv, pref, rhoqr"),
    "M7 integer static dropped (errflg becomes traced)":
        lambda t: replace_once(t, "static_argnames=('ncol', 'nz', 'lyr_surf', 'lyr_toa', 'errflg',)",
                               "static_argnames=('ncol', 'nz', 'lyr_surf', 'lyr_toa',)"),
    "M8 CHARACTER passthrough replaced by a constant (errmsg -> '')":
        lambda t: replace_once(t, "scheme_name, errmsg, errflg_fortran", "scheme_name, '', errflg_fortran"),
    "M9 swap two static scalars at the core call (lyr_surf<->lyr_toa)":
        lambda t: replace_once(t, "= kessler_run_core(ncol, nz, dt, lyr_surf, lyr_toa,",
                               "= kessler_run_core(ncol, nz, dt, lyr_toa, lyr_surf,"),
    "M10 scalar output not converted to a host value (errflg .item() dropped)":
        lambda t: replace_once(t, "errflg_fortran = np.asarray(errflg_host).item()",
                               "errflg_fortran = errflg_host"),
    "M11 to_device uses float32":
        lambda t: replace_once(t, "return jnp.asarray(arr, dtype=jnp.float64)",
                               "return jnp.asarray(arr, dtype=jnp.float32)"),
    "M12 host-side permute in to_device (double conversion)":
        lambda t: replace_once(t, "return jnp.asarray(arr, dtype=jnp.float64)",
                               "return jnp.asarray(np.asarray(arr).T, dtype=jnp.float64)"),
    "M13 private device alias broken":
        lambda t: replace_once(t, "_kessler_run_device = kessler_run_bridge_device",
                               "_kessler_run_device = None"),
    "M14 module var not threaded to the core (rhoqr replaced by pref)":
        lambda t: replace_once(t, "errflg=errflg, lv=lv, pref=pref, rhoqr=rhoqr)",
                               "errflg=errflg, lv=lv, pref=pref, rhoqr=pref)"),
}


def run(python: str, scratch: Path, testfile: str) -> dict:
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([python, "-m", "pytest", testfile, "-q", "-p", "no:cacheprovider",
                        "--no-header", "-rN"],
                       cwd=scratch, env=env, capture_output=True, text=True, timeout=600)
    tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr.strip()[-200:]
    m_pass = re.search(r"(\d+) passed", tail)
    failed = sum(int(x) for x in re.findall(r"(\d+) (?:failed|errors?)", tail))
    return {"exit": p.returncode, "summary": tail, "failed": failed,
            "passed": int(m_pass.group(1)) if m_pass else 0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scratch", type=Path, default=None,
                    help="throw-away project dir (default: a fresh temp dir)")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter with jax + pytest (default: this one)")
    ap.add_argument("--json", type=Path, default=None, help="write per-mutant results here")
    args = ap.parse_args()

    scratch = args.scratch or Path(tempfile.mkdtemp(prefix="laft_mut_kessler_"))
    build_scratch(scratch)
    pristine = (SRC / BRIDGE).read_text(encoding="utf-8")

    rows, ok = [], True
    for name, mut in MUTANTS.items():
        (scratch / BRIDGE).write_text(mut(pristine), encoding="utf-8")
        hand = run(args.python, scratch, HAND)
        gen = run(args.python, scratch, GENERIC)
        rows.append({"mutant": name, "hand": hand, "generic": gen})
        is_control = name.startswith("M0")
        if is_control:
            ok &= hand["exit"] == 0 and gen["exit"] == 0
        else:
            ok &= gen["exit"] != 0     # the shared file must catch every mutant
        print(f"{name:72s} hand: {'CAUGHT' if hand['exit'] else 'missed'} ({hand['failed']}) | "
              f"generic: {'CAUGHT' if gen['exit'] else 'missed'} ({gen['failed']})", flush=True)
    (scratch / BRIDGE).write_text(pristine, encoding="utf-8")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print("\nscratch:", scratch)
    print("RESULT:", "PASS — control green on both files, every mutant caught by the shared file"
          if ok else "FAIL — see the table above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
