"""
phase06_01_grad_gate — Stage 4 gate for a differentiable variant.

Run from the project root:

    python ../LAFT/workflow_differentiable/phase06_01_grad_gate.py

Reads [differentiable] from config/project.toml, loads the project's adapter
(see kessler/differentiable/gate_adapter.py for the interface) and checks:

  G1 equivalence  hard clamps reproduce the validated translation (and the
                  Fortran outputs, if the adapter provides them) to
                  tol_reference; the scan and while loops agree; n_max suffices.
  G2 finite       jax.grad of the adapter's loss w.r.t. every parameter and
                  every listed input is finite, in every clamp mode.
  G3 ad           reverse mode agrees with forward mode (jax.jvp) along a
                  random direction — through the scan AND through the
                  independent while-loop path — to tol_ad.
  G4 fd           reverse mode agrees with central finite differences, per
                  parameter and along a random input direction, to tol_fd.
  G5 soft         (if "soft" in clamp_modes) the soft answer converges
                  monotonically to the hard one as soft_width shrinks.

Informational (not gated): parameters whose gradient is exactly zero in each
mode, and forward / gradient wall-clock.

Writes <report_dir>/grad_gate_results.json (top-level "status": PASS|FAIL)
and grad_gate_report.md. Exit 0 = PASS.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from framework_config import add_config_arg, init  # noqa: E402

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scaled_maxdiff(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _rel(x, y):
    return float(abs(x - y) / max(abs(x), abs(y), 1e-300))


def _tree_dot(a, b):
    return float(sum(jnp.vdot(x, y) for x, y in zip(jax.tree_util.tree_leaves(a),
                                                       jax.tree_util.tree_leaves(b))))


def _timed(fn, reps=3):
    jax.block_until_ready(fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append(time.perf_counter() - t)
    return 1e3 * float(np.median(ts))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_config_arg(ap)
    args = ap.parse_args(argv)
    cfg = init(args.config)
    if not cfg.differentiable_enabled:
        print("[differentiable] not enabled in config/project.toml — Stage 4 skipped.")
        return 0
    D = cfg.differentiable
    root = cfg.root
    A = _load(root / D["adapter"], "laft_diff_adapter")
    rng = np.random.default_rng(D["seed"])
    opts = dict(loop=D["loop"], n_max=int(D["n_max"]))

    case = A.load_case()
    p0 = A.default_params()
    fields = A.output_fields()
    in_fields = A.input_fields()
    checks, info = {}, {}

    # ---------------- G1 equivalence ----------------
    hard = A.run(case, p0, clamp="hard", **opts)
    other_loop = "while" if D["loop"] == "scan" else "scan"
    hard_other = A.run(case, p0, clamp="hard", loop=other_loop, n_max=int(D["n_max"]))
    ref = A.run_reference(case, D["reference"])
    fort = A.fortran_reference() if hasattr(A, "fortran_reference") else None
    g1 = {"vs_reference": {f: _scaled_maxdiff(hard[f], ref[f]) for f in fields},
          "loop_scan_vs_while": {f: _scaled_maxdiff(hard[f], hard_other[f]) for f in fields},
          "errflg": int(hard["errflg"]),
          "n_substeps_max": int(np.max(hard["n_substeps"])), "n_max": int(D["n_max"])}
    if fort is not None:
        g1["vs_fortran"] = {f: _scaled_maxdiff(hard[f], fort[f]) for f in fields}
    tol = float(D["tol_reference"])
    g1["pass"] = (g1["errflg"] == 0
                  and all(v <= tol for v in g1["vs_reference"].values())
                  and all(v <= tol for v in g1["loop_scan_vs_while"].values())
                  and all(v <= tol for v in g1.get("vs_fortran", {}).values()))
    checks["G1_equivalence"] = g1

    # ---------------- G2-G4 per clamp mode ----------------
    pnames = list(getattr(p0, "_fields", [str(i) for i in range(len(jax.tree_util.tree_leaves(p0)))]))
    x0 = {k: case["arrays"][k] for k in in_fields}
    dir_p = jax.tree_util.tree_map(lambda v: jnp.asarray(v * rng.standard_normal(), jnp.float64), p0)
    dir_x = {k: jnp.asarray(np.asarray(v) * rng.standard_normal(np.shape(v)), jnp.float64)
             for k, v in x0.items()}
    for mode in D["clamp_modes"]:
        kw = dict(clamp=mode, soft_width=float(D["soft_width"]))

        def L(p, x, loop=D["loop"]):
            return A.loss(A.run(case, p, inputs=x, loop=loop, n_max=int(D["n_max"]), **kw), case)

        L0, (gp, gx) = jax.value_and_grad(L, argnums=(0, 1))(p0, x0)
        L0 = float(L0)
        gp_vals = dict(zip(pnames, (float(v) for v in jax.tree_util.tree_leaves(gp))))
        finite = (all(np.isfinite(v) for v in gp_vals.values())
                  and all(bool(jnp.isfinite(v).all()) for v in gx.values()))
        checks[f"G2_finite[{mode}]"] = {"pass": finite,
                                         "nonfinite_params": [k for k, v in gp_vals.items()
                                                              if not np.isfinite(v)]}

        # G3: reverse vs forward, through both loop implementations
        rev = _tree_dot(gp, dir_p) + _tree_dot(gx, dir_x)
        g3 = {"reverse": rev}
        for lp in ("scan", "while"):
            _, fwd = jax.jvp(lambda p, x: L(p, x, loop=lp), (p0, x0), (dir_p, dir_x))
            g3[f"forward_{lp}"] = float(fwd)
            g3[f"rel_err_{lp}"] = _rel(rev, float(fwd))
        g3["pass"] = all(g3[f"rel_err_{lp}"] <= float(D["tol_ad"]) for lp in ("scan", "while"))
        checks[f"G3_ad[{mode}]"] = g3

        # G4: finite differences, one parameter at a time + one input direction
        leaves, treedef = jax.tree_util.tree_flatten(p0)
        fd = {}
        for i, name in enumerate(pnames):
            h = float(D["fd_rel_step"]) * max(abs(float(leaves[i])), 1e-12)
            up = list(leaves); up[i] = float(leaves[i]) + h
            dn = list(leaves); dn[i] = float(leaves[i]) - h
            est = (float(L(jax.tree_util.tree_unflatten(treedef, up), x0))
                   - float(L(jax.tree_util.tree_unflatten(treedef, dn), x0))) / (2 * h)
            # Error in the scale-free sensitivity p * dL/dp, floored at 1e-8 |L|
            # so parameters with negligible influence do not report noise.
            pv = abs(float(leaves[i]))
            denom = max(abs(gp_vals[name]) * pv, abs(est) * pv, 1e-8 * abs(L0))
            fd[name] = {"grad": gp_vals[name], "fd": est,
                        "rel_err": abs(gp_vals[name] - est) * pv / denom}
        h = float(D["fd_rel_step"])
        xp = {k: x0[k] + h * dir_x[k] for k in x0}
        xm = {k: x0[k] - h * dir_x[k] for k in x0}
        est_x = (float(L(p0, xp)) - float(L(p0, xm))) / (2 * h)
        rev_x = _tree_dot(gx, dir_x)
        tol_fd = float(D["tol_fd"])
        g4 = {"params": fd, "inputs_direction": {"grad": rev_x, "fd": est_x,
                                                 "rel_err": _rel(rev_x, est_x)}}
        g4["failing"] = [k for k, v in fd.items() if v["rel_err"] > tol_fd]
        g4["pass"] = not g4["failing"] and g4["inputs_direction"]["rel_err"] <= tol_fd
        checks[f"G4_fd[{mode}]"] = g4
        info[f"zero_gradient_params[{mode}]"] = [k for k, v in gp_vals.items() if v == 0.0]
        info[f"grad_ms[{mode}]"] = _timed(lambda: jax.grad(lambda p: L(p, x0))(p0))

    # ---------------- G5 soft -> hard convergence ----------------
    if "soft" in D["clamp_modes"]:
        sweep = sorted((float(w) for w in D["soft_width_sweep"]), reverse=True)
        per_field = {f: [] for f in fields}
        for w in sweep:
            s = A.run(case, p0, clamp="soft", soft_width=w, **opts)
            for f in fields:
                per_field[f].append(_scaled_maxdiff(s[f], hard[f]))
        diffs = [max(v[i] for v in per_field.values()) for i in range(len(sweep))]
        mono = all(b < a for v in per_field.values() for a, b in zip(v, v[1:]))
        checks["G5_soft_convergence"] = {"widths": sweep, "max_scaled_diff": diffs,
                                         "per_field": per_field, "pass": bool(mono)}

    for lp in ("scan", "while"):
        info[f"forward_ms[hard,{lp}]"] = _timed(
            lambda: A.run(case, p0, clamp="hard", loop=lp, n_max=int(D["n_max"]))["theta"])

    status = "PASS" if all(c["pass"] for c in checks.values()) else "FAIL"
    result = {"status": status, "target_proc": D["target_proc"], "module": D["module"],
              "reference": D["reference"], "generated": datetime.now(timezone.utc).isoformat(),
              "jax_version": jax.__version__, "backend": jax.default_backend(),
              "settings": {k: D[k] for k in ("loop", "n_max", "clamp_modes", "soft_width",
                                             "soft_width_sweep", "tol_reference", "tol_ad",
                                             "tol_fd", "fd_rel_step", "seed")},
              "checks": checks, "info": info}
    out = root / D["report_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "grad_gate_results.json").write_text(json.dumps(result, indent=2))
    (out / "grad_gate_report.md").write_text(_markdown(result, pnames))
    for k, c in checks.items():
        print(f"  {k:28s} {'PASS' if c['pass'] else 'FAIL'}")
    print(f"Stage 4 gate: {status}  ->  {out / 'grad_gate_results.json'}")
    return 0 if status == "PASS" else 1


def _markdown(r, pnames) -> str:
    c, s = r["checks"], r["settings"]
    L = [f"# Differentiable gate — {r['target_proc']}", "",
         f"**Status: {r['status']}** · {r['generated'][:19]}Z · JAX {r['jax_version']} "
         f"({r['backend']}) · loop={s['loop']}, n_max={s['n_max']}", "",
         f"Module `{r['module']}`; reference `{r['reference']}`.", "",
         "| Check | Result |", "|---|---|"]
    for k, v in c.items():
        L.append(f"| {k} | {'PASS' if v['pass'] else '**FAIL**'} |")
    g1 = c["G1_equivalence"]
    L += ["", "## G1 — hard clamps vs reference (max |diff| / max |ref|)", "",
          "| Field | vs validated translation | vs Fortran | scan vs while |", "|---|---|---|---|"]
    for f, v in g1["vs_reference"].items():
        L.append(f"| {f} | {v:.1e} | {g1.get('vs_fortran', {}).get(f, float('nan')):.1e} "
                 f"| {g1['loop_scan_vs_while'][f]:.1e} |")
    L += ["", f"Sub-steps used: max {g1['n_substeps_max']} of n_max = {g1['n_max']}; "
          f"errflg = {g1['errflg']}.", ""]
    for mode in s["clamp_modes"]:
        g3, g4 = c[f"G3_ad[{mode}]"], c[f"G4_fd[{mode}]"]
        L += [f"## Gradients — clamp = {mode}", "",
              f"Reverse vs forward mode: rel err {g3['rel_err_scan']:.1e} (scan), "
              f"{g3['rel_err_while']:.1e} (while). Input-direction FD rel err "
              f"{g4['inputs_direction']['rel_err']:.1e}.", "",
              "| Parameter | dL/dp (reverse) | central FD | rel err |", "|---|---|---|---|"]
        for p in pnames:
            v = g4["params"][p]
            L.append(f"| {p} | {v['grad']:.6e} | {v['fd']:.6e} | {v['rel_err']:.1e} |")
        z = r["info"][f"zero_gradient_params[{mode}]"]
        L += ["", f"Exactly-zero gradients: {', '.join(z) if z else 'none'}.", ""]
    if "G5_soft_convergence" in c:
        g5 = c["G5_soft_convergence"]
        fs = list(g5["per_field"])
        L += ["## G5 — soft → hard convergence (max |soft - hard| / max |hard|)", "",
              "| soft_width | " + " | ".join(fs) + " |", "|---" * (len(fs) + 1) + "|"]
        for i, w in enumerate(g5["widths"]):
            L.append(f"| {w:.0e} | " + " | ".join(f"{g5['per_field'][f][i]:.1e}" for f in fs) + " |")
        L.append("")
    L += ["## Timing (informational)", ""]
    L += [f"- {k}: {v:.1f} ms" for k, v in r["info"].items() if k.split("[")[0].endswith("_ms")]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
