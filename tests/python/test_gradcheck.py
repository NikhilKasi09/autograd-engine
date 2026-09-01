"""Gradients against finite differences, rather than against a second derivation.

Everything in test_autograd_backward.py compares a gradient to a hand-written
NumPy expression. That catches a typo. It cannot catch a mistake in the calculus,
because the expected value came from the same derivation - dA = dout @ B.T
appears in ops.py and again in the test, and if the transpose belongs elsewhere
both say so together.

These tests never write a derivative down. They perturb an input, watch the
output move, and check the engine agrees. What that buys, and what it does not:

  - It sees any wrong gradient above the tolerance, and the tolerance is
    computed from |f| rather than chosen - see gradcheck.tolerance.
  - It is blind at the relu kink. A symmetric difference cannot reach the
    subgradient at exactly 0, so away_from_zero never samples there and the
    strict comparison stays test_ops.py's job.
"""

from __future__ import annotations

import numpy as np
import pytest

import autograd
import gradcheck
from autograd import _core


def u(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, shape).astype(np.float32)


def kinkless(rng: np.random.Generator, *shape: int) -> np.ndarray:
    """Uniform, but held off the relu kink. Both signs still present."""
    return gradcheck.away_from_zero(u(rng, *shape))


# --------------------------------------------------------------------------
# The harness
# --------------------------------------------------------------------------


def test_the_harness_module_is_importable_beside_the_tests() -> None:
    """gradcheck.py is a sibling with no __init__.py, so this is not free - it
    works because pytest prepends the test file's directory to sys.path."""
    assert gradcheck.GRADCHECK_H == 3e-3


def test_the_step_stays_below_the_relu_kink_margin() -> None:
    """At margin 1e-2 and h=1e-1 a correct relu gradient reads 45% wrong."""
    assert gradcheck.GRADCHECK_H < gradcheck.KINK_MARGIN


def test_the_tolerance_grows_with_the_function_magnitude() -> None:
    """The floor is cancellation in |f|, not a property of the op."""
    assert gradcheck.tolerance(10.0) > gradcheck.tolerance(1.0)


def test_the_tolerance_bottoms_out_rather_than_going_to_zero() -> None:
    """A single float32 rounding is there however small |f| gets."""
    assert gradcheck.tolerance(0.0) == gradcheck.FLOOR_MIN


def test_a_case_too_large_to_check_raises_instead_of_passing() -> None:
    """The guard against an adaptive bound widening until it accepts anything.

    Loosening past the ceiling is never the right answer - the case has to
    shrink, because float32 genuinely cannot resolve the gradient any better.
    """
    with pytest.raises(ValueError, match="over the 1e-03 ceiling"):
        gradcheck.tolerance(1e6)


def test_the_error_metric_is_absolute_below_a_gradient_of_one() -> None:
    """Documented rather than fixed: the 1.0 floor makes small gradients an
    absolute comparison. Inert at these input scales, and the assertion is here
    so it stays a known property rather than a surprise."""
    tiny = np.array([[1e-6]])
    assert gradcheck.max_relative_error(tiny, np.zeros((1, 1))) == pytest.approx(1e-6)


# --------------------------------------------------------------------------
# One entry per op, every operand, awkward shapes
# --------------------------------------------------------------------------

CASES = [
    pytest.param(lambda L: autograd.add(L[0], L[1]),
                 lambda r: [u(r, 2, 3), u(r, 2, 3)], id="add"),
    pytest.param(lambda L: autograd.add(L[0], L[1]),
                 lambda r: [u(r, 7), u(r, 7)], id="add-rank1"),
    pytest.param(lambda L: autograd.add(L[0], L[1]),
                 lambda r: [u(r, 2, 3, 4), u(r, 2, 3, 4)], id="add-rank3"),
    pytest.param(lambda L: autograd.mul(L[0], L[1]),
                 lambda r: [u(r, 2, 3), u(r, 2, 3)], id="mul"),
    pytest.param(lambda L: autograd.mul(L[0], L[1]),
                 lambda r: [u(r, 2, 3, 4), u(r, 2, 3, 4)], id="mul-rank3"),
    # Never square: a square case cannot tell a swapped pair, a missing
    # transpose and a transposed result apart.
    pytest.param(lambda L: autograd.matmul(L[0], L[1]),
                 lambda r: [u(r, 2, 3), u(r, 3, 4)], id="matmul-2x3x4"),
    pytest.param(lambda L: autograd.matmul(L[0], L[1]),
                 lambda r: [u(r, 7, 9), u(r, 9, 8)], id="matmul-7x9x8"),
    # M == 1 is batch size one, and it used to raise in backward - the
    # regression lives in test_autograd_backward.py.
    pytest.param(lambda L: autograd.matmul(L[0], L[1]),
                 lambda r: [u(r, 1, 5), u(r, 5, 3)], id="matmul-1x5x3"),
    pytest.param(lambda L: autograd.matmul(L[0], L[1]),
                 lambda r: [u(r, 4, 1), u(r, 1, 3)], id="matmul-4x1x3"),
    pytest.param(lambda L: autograd.relu(L[0]),
                 lambda r: [kinkless(r, 2, 3)], id="relu"),
    pytest.param(lambda L: autograd.relu(L[0]),
                 lambda r: [kinkless(r, 2, 3, 4)], id="relu-rank3"),
    pytest.param(lambda L: autograd.sum(L[0]),
                 lambda r: [u(r, 2, 3)], id="sum-full"),
    pytest.param(lambda L: autograd.sum(L[0], (1, 3)),
                 lambda r: [u(r, 2, 3)], id="sum-axis0"),
    pytest.param(lambda L: autograd.sum(L[0], (2, 1)),
                 lambda r: [u(r, 2, 3)], id="sum-axis1"),
    pytest.param(lambda L: autograd.expand(L[0], (3, 4)),
                 lambda r: [u(r, 1, 4)], id="expand-bias"),
    pytest.param(lambda L: autograd.expand(L[0], (2, 5)),
                 lambda r: [u(r, 2, 1)], id="expand-col"),
    pytest.param(lambda L: autograd.expand(L[0], (3, 4)),
                 lambda r: [u(r, 1, 1)], id="expand-both"),
]


@pytest.mark.parametrize("build, make", CASES)
def test_the_jacobian_matches_finite_differences(build, make, rng) -> None:
    gradcheck.check_jacobian(build, make(rng))


# --------------------------------------------------------------------------
# Strided leaves
# --------------------------------------------------------------------------
#
# A leaf may legally be a view. Mul, Matmul and Relu all read their operand's
# strides, and a harness that always rebuilt with from_numpy would only ever
# feed contiguous ones. The base array keeps its own shape; `views` supplies the
# leaf the engine actually sees.

TRANSPOSE = lambda t: t.transpose(0, 1)                     # noqa: E731


@pytest.mark.parametrize("build, make, views", [
    pytest.param(lambda L: autograd.mul(L[0], L[1]),
                 lambda r: [u(r, 3, 2), u(r, 2, 3)],
                 [TRANSPOSE, None], id="mul-transposed-operand"),
    pytest.param(lambda L: autograd.matmul(L[0], L[1]),
                 lambda r: [u(r, 3, 2), u(r, 3, 4)],
                 [TRANSPOSE, None], id="matmul-transposed-a"),
    pytest.param(lambda L: autograd.matmul(L[0], L[1]),
                 lambda r: [u(r, 2, 3), u(r, 4, 3)],
                 [None, TRANSPOSE], id="matmul-transposed-b"),
    pytest.param(lambda L: autograd.relu(L[0]),
                 lambda r: [kinkless(r, 3, 2)],
                 [TRANSPOSE], id="relu-transposed"),
    pytest.param(lambda L: autograd.sum(L[0]),
                 lambda r: [u(r, 3, 2)],
                 [TRANSPOSE], id="sum-transposed"),
    # Sliced off a wider base, so the row stride exceeds the row width - the
    # seam a view onto a larger buffer arrives through.
    pytest.param(lambda L: autograd.matmul(L[0], L[1]),
                 lambda r: [u(r, 2, 6), u(r, 3, 4)],
                 [lambda t: t.slice(1, 1, 3), None], id="matmul-sliced-a"),
], )
def test_a_strided_leaf_gets_the_same_gradient(build, make, views, rng) -> None:
    gradcheck.check_jacobian(build, make(rng), views=views)


# --------------------------------------------------------------------------
# The negative controls
# --------------------------------------------------------------------------


def test_a_relu_input_on_the_kink_makes_the_check_fail(rng) -> None:
    """A value nearer zero than h makes the step cross the kink, so it measures
    a slope that exists nowhere.

    This test, not away_from_zero, is what proves the mechanism. Disabling
    away_from_zero fails nothing: a uniform draw lands inside the step on only
    1 seed in 200, so the margin removes a rare flake rather than catching a
    bug. Worth keeping for that, worth not overstating.
    """
    a = u(rng, 2, 3)
    a[0, 0] = 1e-5      # inside the step, so the difference straddles zero

    with pytest.raises(AssertionError, match="exceeds tol"):
        gradcheck.check_jacobian(lambda L: autograd.relu(L[0]), [a])


def test_a_deliberately_scaled_gradient_is_caught(rng) -> None:
    """The tolerance is the detection threshold, so prove the threshold bites.

    A gradient wrong by (1+e) produces exactly e relative error. Scaling by 1%
    is far above any tolerance this harness computes, so a check that passed
    here would be vacuous.
    """
    a, b = u(rng, 2, 3), u(rng, 3, 4)
    build = lambda L: autograd.matmul(L[0], L[1])           # noqa: E731

    num = gradcheck.numerical_jacobian(build, [a, b], 0)
    ana = gradcheck.analytic_jacobian(build, [a, b], 0)
    tol = gradcheck.tolerance(gradcheck.forward_magnitude(build, [a, b]))

    assert gradcheck.max_relative_error(num, ana) <= tol
    assert gradcheck.max_relative_error(num, ana * 1.01) > tol


# --------------------------------------------------------------------------
# What backward promises the engine
# --------------------------------------------------------------------------


@pytest.mark.parametrize("build, make", CASES)
def test_backward_returns_one_correctly_shaped_gradient_per_parent(build, make, rng) -> None:
    """Total, and shaped like the operand rather than like grad_out. Matmul is
    the case where those differ."""
    leaves = gradcheck.make_leaves(make(rng))
    out = build(leaves)

    grads = out.grad_fn.backward(_core.zeros_like(out.data))

    assert len(grads) == len(out.grad_fn.parents)
    for grad, parent in zip(grads, out.grad_fn.parents):
        assert grad.shape == parent.shape


# --------------------------------------------------------------------------
# Composites
# --------------------------------------------------------------------------


def test_a_diamond_accumulates_correctly(rng) -> None:
    """One leaf reaching the root by two paths. The walk-order case from the
    graph tests, checked numerically instead of against a hand-derived 5."""
    gradcheck.check_jacobian(
        lambda L: autograd.add(autograd.mul(L[0], L[0]), L[0]),
        [u(rng, 2, 3)],
    )


def test_a_deep_chain_matches(rng) -> None:
    """Repeated ops, so an error that only compounds has somewhere to show."""
    gradcheck.check_jacobian(
        lambda L: autograd.relu(autograd.add(
            autograd.mul(autograd.relu(autograd.mul(L[0], L[1])), L[0]), L[1])),
        [kinkless(rng, 2, 3), kinkless(rng, 2, 3)],
    )


def test_a_cubic_matches_where_truncation_is_real(rng) -> None:
    """Every op here is linear or bilinear, so central differences are exact up
    to rounding. x*x*x is the one case with a genuine h^2 truncation term, and
    it is what puts h at the bottom of a bowl rather than as low as it will go.
    """
    gradcheck.check_jacobian(
        lambda L: autograd.mul(autograd.mul(L[0], L[0]), L[0]),
        [rng.uniform(0.5, 1.5, (2, 3)).astype(np.float32)],
    )


def mlp(L: list[autograd.Tensor]) -> autograd.Tensor:
    """x @ w1 + b1 -> relu -> @ w2, summed. The shape of a real training step."""
    hidden = autograd.relu(autograd.add(autograd.matmul(L[0], L[1]),
                                        autograd.expand(L[2], (2, 4))))
    return autograd.sum(autograd.matmul(hidden, L[3]))


def test_the_two_layer_mlp_matches_over_several_directions(rng) -> None:
    """The flagship case, contracted rather than fully expanded.

    Three seeds because one random v is blind to any error orthogonal to it, and
    this is the graph most likely to hide one.
    """
    gradcheck.check_directional(
        mlp,
        [u(rng, 2, 3), u(rng, 3, 4), kinkless(rng, 1, 4), u(rng, 4, 2)],
    )


def test_the_mlp_jacobian_matches_element_by_element(rng) -> None:
    """The output is a scalar, so the full Jacobian is affordable here too and
    nothing has to be taken on a random direction's word."""
    gradcheck.check_jacobian(
        mlp,
        [u(rng, 2, 3), u(rng, 3, 4), kinkless(rng, 1, 4), u(rng, 4, 2)],
    )
