"""
diff_primitives — reverse-mode-safe building blocks for differentiable variants.

A validated LAFT translation is a faithful port of Fortran: it runs forward and
supports forward-mode AD (jax.jvp / jax.jacfwd), but three constructs that are
harmless in Fortran break reverse-mode AD (jax.grad) in JAX:

  1. lax.while_loop with a data-dependent trip count (CFL sub-cycling,
     convergence iterations) — JAX refuses to transpose it.
  2. Fractional powers x**p (0 < p < 1) evaluated at x == 0 — the derivative
     p*x**(p-1) is infinite, and 0 * inf = NaN in the backward pass.
  3. Divisions inside jnp.where(mask, a / b, c) — the unselected branch still
     divides by zero; its infinite partial times a zero cotangent is NaN.

This module provides the replacements (bounded_while, safe_pow, safe_div) plus
an optional soft-clamp family so gradients also flow through max/min/positivity
limiters. Every primitive reproduces the original expression exactly in its
default ("hard") form, so a differentiable variant can be gated against the
validated translation at round-off.

Project-agnostic: imported by each project's differentiable module, e.g.
kessler/differentiable/kessler_run_diff.py. See DIFFERENTIABLE_WORKFLOW.md.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import lax

CLAMP_MODES = ("hard", "soft")
LOOP_MODES = ("scan", "while")


# ---------------------------------------------------------------------------
# NaN-free replacements for operations that are singular at the boundary
# ---------------------------------------------------------------------------

def safe_pow(x, p):
    """``x ** p`` for ``x > 0``, else 0, with a finite (zero) gradient at ``x <= 0``.

    The double ``where`` keeps ``0 ** (p - 1)`` out of the AD graph. Identical
    to ``x ** p`` for ``x > 0`` and to Fortran's ``0 ** p = 0`` at ``x == 0``.
    ``p`` may be a traced (learnable) exponent: d/dp = x**p * log(x) is also
    guarded, since log is only ever evaluated on the positive branch.
    """
    pos = x > 0.0
    return jnp.where(pos, jnp.where(pos, x, 1.0) ** p, 0.0)


def safe_div(num, den, mask):
    """``num / den`` where ``mask``; finite garbage elsewhere (caller selects it away).

    Use inside ``jnp.where(mask, safe_div(num, den, mask), other)`` in place of
    ``jnp.where(mask, num / den, other)``.
    """
    return num / jnp.where(mask, den, 1.0)


# ---------------------------------------------------------------------------
# Clamps: hard (exact max/min) or soft (softplus-smoothed, width-controlled)
# ---------------------------------------------------------------------------

def soft_relu(x, width):
    """``width * softplus(x / width)`` — a smooth ``max(x, 0)``.

    Never negative (> 0 in exact arithmetic; underflows to 0 far on the clamped
    side), so positivity limiters keep their guarantee. Converges to ``max(x, 0)`` as
    ``width -> 0`` with error at most ``width * log(2)`` (attained at x = 0),
    and has gradient ``sigmoid(x / width)`` — nonzero on the clamped side, so a
    limiter that is active still passes (exponentially attenuated) sensitivity.
    ``width`` carries the units of ``x``.
    """
    return width * jax.nn.softplus(x / width)


class Clamps(NamedTuple):
    """max/min/relu with one switch. ``mode`` is static; ``width`` may be traced."""

    mode: str
    width: Any

    def relu(self, x):
        if self.mode == "hard":
            return jnp.maximum(x, 0.0)
        return soft_relu(x, self.width)

    def max(self, a, b):
        if self.mode == "hard":
            return jnp.maximum(a, b)
        return b + soft_relu(a - b, self.width)

    def min(self, a, b):
        if self.mode == "hard":
            return jnp.minimum(a, b)
        return a - soft_relu(a - b, self.width)


def make_clamps(mode: str = "hard", width=1e-6) -> Clamps:
    if mode not in CLAMP_MODES:
        raise ValueError(f"clamp mode must be one of {CLAMP_MODES}, got {mode!r}")
    return Clamps(mode, width)


# ---------------------------------------------------------------------------
# Bounded while: reverse-mode-differentiable replacement for lax.while_loop
# ---------------------------------------------------------------------------

def bounded_while(cond_fn: Callable, body_fn: Callable, init, *, n_max: int,
                  loop: str = "scan", checkpoint: bool = True) -> Tuple[Any, Any, Any]:
    """Run ``body_fn`` while ``cond_fn`` holds, for at most ``n_max`` iterations.

    loop="scan"  : ``lax.scan`` of fixed length ``n_max``. Once ``cond_fn`` is
                   False the carry is FROZEN (not stepped with a zero step), so
                   the result equals ``lax.while_loop`` whenever the true trip
                   count is <= n_max. Reverse-mode differentiable; cost is
                   always n_max iterations, so size n_max from the problem.
    loop="while" : plain ``lax.while_loop`` (fast forward path, forward-mode AD
                   only); ``n_max`` is still enforced so both paths agree.

    Freezing matters: a "zero-length" step is often not a no-op (e.g. a
    saturation adjustment that does not scale with the step).

    Returns ``(final_carry, n_iterations, exhausted)`` where ``exhausted`` is
    True if ``cond_fn`` still holds after n_max iterations (budget too small).
    """
    if loop not in LOOP_MODES:
        raise ValueError(f"loop must be one of {LOOP_MODES}, got {loop!r}")
    n_max = int(n_max)
    if n_max < 1:
        raise ValueError("n_max must be >= 1")

    if loop == "while":
        def c(s):
            carry, n = s
            return cond_fn(carry) & (n < n_max)

        def b(s):
            carry, n = s
            return body_fn(carry), n + 1

        final, n = lax.while_loop(c, b, (init, jnp.asarray(0, jnp.int32)))
        return final, n, cond_fn(final)

    step = jax.checkpoint(body_fn) if checkpoint else body_fn

    def masked(s, _):
        carry, n = s
        active = cond_fn(carry)
        stepped = step(carry)
        carry = jax.tree_util.tree_map(lambda new, old: jnp.where(active, new, old),
                                       stepped, carry)
        return (carry, n + active.astype(jnp.int32)), None

    (final, n), _ = lax.scan(masked, (init, jnp.asarray(0, jnp.int32)), None, length=n_max)
    return final, n, cond_fn(final)
