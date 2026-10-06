#!/usr/bin/env python3
"""
Tests for the differentiable Kessler variant (differentiable/kessler_run_diff.py).

Separate from bridge_test/ (the bridge and validated-translation tests) by
design: these exercise the Stage 4 variant only and never touch out/ or the
frozen translations/ archives beyond reading one of them as the reference.
The full gate (all parameters, FD per parameter, soft-width sweep) is
LAFT/workflow_differentiable/phase06_01_grad_gate.py; these are the fast pins.

Run from the project root:
    python -m pytest differentiable/tests -v
"""

import os
os.environ["JAX_ENABLE_X64"] = "1"

import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import gate_adapter as A  # noqa: E402
from kessler_run_diff import DEFAULT_PARAMS, KesslerParams, kessler_run_diff  # noqa: E402

REFERENCE = "translations/gemini31pro-jd/jax/kessler_run.validated.py"
N_MAX = 16
FIELDS = A.output_fields()


@pytest.fixture(scope="module")
def case():
    return A.load_case()


@pytest.fixture(scope="module")
def hard(case):
    return A.run(case, DEFAULT_PARAMS, n_max=N_MAX)


def _scaled(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / np.abs(b).max())


# ---------------- equivalence (hard clamps = the validated scheme) ----------------

def test_hard_matches_fortran(case, hard):
    ref = A.fortran_reference()
    for f in FIELDS:
        assert _scaled(hard[f], ref[f]) < 1e-13, f
    assert int(hard["errflg"]) == 0


def test_hard_matches_validated_translation(case, hard):
    ref = A.run_reference(case, REFERENCE)
    for f in FIELDS:
        assert _scaled(hard[f], ref[f]) < 1e-13, f


def test_scan_and_while_agree(case, hard):
    w = A.run(case, DEFAULT_PARAMS, loop="while", n_max=N_MAX)
    for f in FIELDS:
        assert _scaled(hard[f], w[f]) < 1e-13, f
    np.testing.assert_array_equal(hard["n_substeps"], w["n_substeps"])


def test_reversed_column_orientation(case, hard):
    a = {k: v[::-1] for k, v in case["arrays"].items()}
    nz = case["nz"]
    o = kessler_run_diff(case["dt"], *(a[n] for n in A.ARRAYS), A.LV, A.PREF_HPA, A.RHOQR,
                         lyr_surf=nz, lyr_toa=1, n_max=N_MAX)
    for f in ["theta", "qv", "qc", "qr"]:
        np.testing.assert_allclose(np.asarray(getattr(o, f))[::-1], hard[f], rtol=0, atol=1e-15)
    np.testing.assert_allclose(o.precl, hard["precl"], rtol=1e-14)


# ---------------- error flags ----------------

def test_budget_exhaustion_is_flagged(case):
    o = A.run(case, DEFAULT_PARAMS, n_max=2)
    assert int(o["errflg"]) == 3


def test_nonpositive_dt_is_flagged(case):
    o = A.run({**case, "dt": 0.0}, DEFAULT_PARAMS, n_max=N_MAX)
    assert int(o["errflg"]) == 1
    assert bool(jnp.all(o["precl"] == 0.0))


# ---------------- learnable constants ----------------

def test_keyword_overrides_equal_params_pytree(case):
    a = case["arrays"]
    args = (case["dt"], *(a[n] for n in A.ARRAYS), A.LV, A.PREF_HPA, A.RHOQR)
    o1 = kessler_run_diff(*args, accretion_coeff=3.0, n_max=N_MAX)
    o2 = kessler_run_diff(*args, params=KesslerParams(accretion_coeff=3.0), n_max=N_MAX)
    np.testing.assert_array_equal(o1.qr, o2.qr)
    o0 = kessler_run_diff(*args, n_max=N_MAX)
    assert not np.array_equal(np.asarray(o0.qr), np.asarray(o1.qr))


def test_unknown_parameter_is_rejected(case):
    a = case["arrays"]
    with pytest.raises(TypeError, match="unknown Kessler parameter"):
        kessler_run_diff(case["dt"], *(a[n] for n in A.ARRAYS), A.LV, A.PREF_HPA, A.RHOQR,
                         accretion_coef=3.0)


@pytest.mark.parametrize("clamp", ["hard", "soft"])
def test_param_gradients_finite_nonzero_and_match_forward_mode(case, clamp):
    def L(p):
        return A.loss(A.run(case, p, clamp=clamp, n_max=N_MAX), case)

    g = jax.grad(L)(DEFAULT_PARAMS)
    leaves = jax.tree_util.tree_leaves(g)
    assert all(np.isfinite(float(v)) for v in leaves)
    assert all(float(v) != 0.0 for v in leaves)          # every constant matters here
    d = jax.tree_util.tree_map(lambda v: v * 0.01, DEFAULT_PARAMS)
    _, t = jax.jvp(L, (DEFAULT_PARAMS,), (d,))
    rev = sum(float(a) * float(b) for a, b in zip(leaves, jax.tree_util.tree_leaves(d)))
    assert abs(rev - float(t)) <= 1e-8 * abs(float(t))


def test_param_gradient_matches_finite_difference(case):
    def L(k2):
        return A.loss(A.run(case, DEFAULT_PARAMS._replace(accretion_coeff=k2), n_max=N_MAX), case)

    k2, h = DEFAULT_PARAMS.accretion_coeff, 1e-6 * DEFAULT_PARAMS.accretion_coeff
    fd = (float(L(k2 + h)) - float(L(k2 - h))) / (2 * h)
    assert np.isclose(float(jax.grad(L)(k2)), fd, rtol=1e-6)


def test_input_gradients_are_finite(case):
    x0 = {k: case["arrays"][k] for k in A.input_fields()}
    g = jax.grad(lambda x: A.loss(A.run(case, DEFAULT_PARAMS, inputs=x, n_max=N_MAX), case))(x0)
    assert all(bool(jnp.isfinite(v).all()) for v in g.values())


# ---------------- soft clamps ----------------

def test_soft_converges_to_hard(case, hard):
    errs = []
    for w in (1e-6, 1e-7, 1e-8):
        s = A.run(case, DEFAULT_PARAMS, clamp="soft", soft_width=w, n_max=N_MAX)
        errs.append(max(_scaled(s[f], hard[f]) for f in ["theta", "qv", "qc", "qr", "precl"]))
    assert errs[0] > errs[1] > errs[2]
    assert errs[0] < 2e-3


def test_soft_clamps_keep_mixing_ratios_nonnegative(case):
    # softplus is > 0 mathematically; in floating point it underflows to 0 where
    # a species is fully removed, so the guarantee is >= 0.
    s = A.run(case, DEFAULT_PARAMS, clamp="soft", soft_width=1e-6, n_max=N_MAX)
    for f in ["qv", "qc", "qr"]:
        assert bool((s[f] >= 0).all()), f


def test_soft_clamps_pass_gradient_through_inactive_autoconversion(case):
    # Raise the threshold above every qc in the case: autoconversion is OFF, so the
    # hard-clamp gradient w.r.t. the threshold is exactly zero; soft is not.
    p = DEFAULT_PARAMS._replace(autoconv_threshold=0.05)

    def g(clamp):
        return float(jax.grad(lambda a: A.loss(A.run(case, p._replace(autoconv_threshold=a),
                                                     clamp=clamp, soft_width=1e-3,
                                                     n_max=N_MAX), case))(0.05))

    assert g("hard") == 0.0
    assert g("soft") != 0.0
