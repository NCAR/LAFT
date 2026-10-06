#!/usr/bin/env python3
"""Unit tests for diff_primitives — project-agnostic, no Fortran or translation needed.

Each primitive is pinned on the property it exists for: the hard form equals the
original expression exactly, and reverse-mode AD through it is finite.

Run:  pytest workflow_differentiable/test_diff_primitives.py
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from diff_primitives import bounded_while, make_clamps, safe_div, safe_pow, soft_relu


X = jnp.array([0.0, 1e-12, 1e-3, 0.5, 2.0])


@pytest.mark.parametrize("p", [0.1364, 0.525, 0.875, 2.0])
def test_safe_pow_matches_pow_and_has_finite_grad(p):
    np.testing.assert_array_equal(safe_pow(X, p), X ** p)
    g = jax.grad(lambda x: safe_pow(x, p).sum())(X)
    assert bool(jnp.isfinite(g).all())
    assert float(g[0]) == 0.0
    # the naive form is what breaks: NaN/inf at x = 0 for p < 1
    if p < 1:
        assert not bool(jnp.isfinite(jax.grad(lambda x: (x ** p).sum())(X)).all())


def test_safe_pow_exponent_gradient_is_finite_at_zero():
    g = jax.grad(lambda p: safe_pow(X, p).sum())(0.5)
    assert np.isfinite(float(g))


def test_safe_div_inside_where():
    v = jnp.array([0.0, 2.0, 4.0])
    m = jnp.abs(v) > 1e-12

    def naive(v):
        return jnp.where(m, 1.0 / v, 7.0).sum()

    def safe(v):
        return jnp.where(m, safe_div(1.0, v, m), 7.0).sum()

    assert float(naive(v)) == float(safe(v))
    assert not bool(jnp.isfinite(jax.grad(naive)(v)).all())
    assert bool(jnp.isfinite(jax.grad(safe)(v)).all())


def test_hard_clamps_are_exact():
    a, b = jnp.array([-1.0, 0.0, 3.0]), jnp.array([0.5, 0.0, 1.0])
    C = make_clamps("hard")
    np.testing.assert_array_equal(C.relu(a), jnp.maximum(a, 0.0))
    np.testing.assert_array_equal(C.max(a, b), jnp.maximum(a, b))
    np.testing.assert_array_equal(C.min(a, b), jnp.minimum(a, b))


def test_soft_clamps_converge_and_pass_gradient():
    x = jnp.linspace(-1e-3, 1e-3, 101)
    errs = [float(jnp.abs(soft_relu(x, w) - jnp.maximum(x, 0)).max()) for w in (1e-4, 1e-5, 1e-6)]
    assert errs[0] > errs[1] > errs[2]
    assert errs[-1] <= 1e-6 * np.log(2) * (1 + 1e-9)
    assert bool((soft_relu(x, 1e-5) > 0).all())                    # positivity kept
    g = jax.grad(lambda x: soft_relu(x, 1e-4).sum())(jnp.array([-1e-4]))
    assert float(g[0]) > 0.0                                        # hard relu: exactly 0
    C = make_clamps("soft", 1e-6)
    a, b = jnp.array([1.0, -2.0]), jnp.array([0.0, 0.0])
    assert bool((C.max(a, b) >= jnp.maximum(a, b)).all())
    assert bool((C.min(a, b) <= jnp.minimum(a, b)).all())


def test_make_clamps_rejects_unknown_mode():
    with pytest.raises(ValueError):
        make_clamps("smooth")


def _halving(x0):
    # data-dependent trip count: halve until below 1
    return (lambda s: s[0] >= 1.0), (lambda s: (s[0] * 0.5, s[1] + s[0]))


@pytest.mark.parametrize("loop", ["scan", "while"])
def test_bounded_while_matches_while_loop(loop):
    cond, body = _halving(None)
    init = (jnp.asarray(37.0), jnp.asarray(0.0))
    ref = lax.while_loop(cond, body, init)
    out, n, exhausted = bounded_while(cond, body, init, n_max=20, loop=loop)
    assert float(out[0]) == float(ref[0]) and float(out[1]) == float(ref[1])
    assert int(n) == 6 and not bool(exhausted)


def test_bounded_while_reports_exhausted_budget():
    cond, body = _halving(None)
    _, n, exhausted = bounded_while(cond, body, (jnp.asarray(37.0), jnp.asarray(0.0)),
                                    n_max=3, loop="scan")
    assert int(n) == 3 and bool(exhausted)


def test_bounded_while_freezes_instead_of_stepping():
    # A body that is NOT a no-op once the condition fails: the scan must freeze.
    cond = lambda s: s[1] < 2
    body = lambda s: (s[0] + 1.0, s[1] + 1)
    out, n, _ = bounded_while(cond, body, (jnp.asarray(0.0), jnp.asarray(0)), n_max=10)
    assert float(out[0]) == 2.0 and int(n) == 2


def test_bounded_while_scan_is_reverse_differentiable():
    def f(x, loop):
        cond = lambda s: s[1] < 1.0
        body = lambda s: (s[0] * x, s[1] + 0.3)
        return bounded_while(cond, body, (jnp.asarray(1.0), jnp.asarray(0.0)),
                             n_max=8, loop=loop)[0][0]

    g = jax.grad(lambda x: f(x, "scan"))(1.1)
    _, t = jax.jvp(lambda x: f(x, "while"), (1.1,), (1.0,))
    assert np.isclose(float(g), 4 * 1.1 ** 3) and np.isclose(float(g), float(t))
    with pytest.raises(ValueError):
        jax.grad(lambda x: f(x, "while"))(1.1)
