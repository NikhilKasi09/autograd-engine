"""The backward pass: walk order, accumulation, and what the graph does not keep.

Written before the implementation, so these fail on a fresh scaffold.

Everything here is add-only. Mul.backward lands in step 4, and the one test that
mentions mul never calls backward - it uses the forward pass to build a graph
with something saved in it, because Add saves nothing and a lifetime test over
an add-only graph would be asserting about an empty tuple.

The two tests that carry the step are the ordering diamond and the lifetime
check. Every other test in this file passes on a breadth-first walk and on a
graph riddled with reference cycles.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

import autograd
from autograd import _core


def leaf(a: np.ndarray, requires_grad: bool = True) -> autograd.Tensor:
    raw = _core.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    return autograd.Tensor(raw, requires_grad=requires_grad)


def scalar(v: float) -> autograd.Tensor:
    return leaf(np.array([v]))


# --------------------------------------------------------------------------
# Accumulation
# --------------------------------------------------------------------------


def test_a_leaf_used_once_gets_the_incoming_gradient() -> None:
    x = leaf(np.array([1.0, 2.0, 3.0]))
    y = leaf(np.array([10.0, 20.0, 30.0]))

    autograd.add(x, y).backward(_core.from_numpy(np.ones(3, dtype=np.float32)))

    assert np.array_equal(np.asarray(x.grad), np.ones(3))
    assert np.array_equal(np.asarray(y.grad), np.ones(3))


def test_a_leaf_used_twice_accumulates_both_contributions() -> None:
    """The simplest multi-use case. An engine that assigns rather than
    accumulates reads 1 here instead of 2."""
    x = leaf(np.array([1.0, 2.0, 3.0]))

    autograd.add(x, x).backward(_core.from_numpy(np.ones(3, dtype=np.float32)))

    assert np.array_equal(np.asarray(x.grad), np.full(3, 2.0))


def test_the_walk_order_survives_a_diamond() -> None:
    """The test that earns this step.

    a = 2x, b = 3x, c = 5x, so dx = 5. `a` feeds both `b` and `c`. A walk that
    finishes `a` as soon as `c` hands it a gradient - before `b`'s contribution
    has arrived - drops a path and reads 3.

    Verified by simulation before this was written: a LIFO stack pop reads 3, a
    FIFO queue pop reads 5. So a queue is NOT a useful mutation of this test;
    breaking the walk with a stack pop is.
    """
    x = scalar(2.0)

    a = autograd.add(x, x)
    b = autograd.add(a, x)
    c = autograd.add(b, a)

    assert c.to_numpy()[0] == pytest.approx(10.0)  # c = 5x

    c.backward()

    assert np.asarray(x.grad)[0] == pytest.approx(5.0)


def test_backward_returns_none() -> None:
    x = scalar(1.0)

    assert autograd.add(x, x).backward() is None


# --------------------------------------------------------------------------
# Who receives a gradient
# --------------------------------------------------------------------------


def test_a_parent_that_does_not_require_grad_is_left_alone() -> None:
    """The mixed graph.

    Add.backward computes a gradient for both parents regardless; the engine is
    what decides who keeps one. Writing .grad onto a tensor that never asked is
    silent here and shows up at phase 8 as an optimizer stepping a frozen
    tensor.
    """
    x = leaf(np.array([1.0, 2.0]), requires_grad=True)
    y = leaf(np.array([3.0, 4.0]), requires_grad=False)

    autograd.add(x, y).backward(_core.from_numpy(np.ones(2, dtype=np.float32)))

    assert np.array_equal(np.asarray(x.grad), np.ones(2))
    assert y.grad is None


def test_an_intermediate_keeps_no_gradient_by_default() -> None:
    x = scalar(3.0)

    mid = autograd.add(x, x)
    autograd.add(mid, x).backward()

    assert mid.grad is None
    # The leaf is still right; dropping the intermediate is a storage decision,
    # not an arithmetic one.
    assert np.asarray(x.grad)[0] == pytest.approx(3.0)


def test_retain_grad_makes_an_intermediate_keep_its_gradient() -> None:
    x = scalar(3.0)

    mid = autograd.add(x, x)
    mid.retain_grad()
    autograd.add(mid, x).backward()

    assert mid.grad is not None
    assert np.asarray(mid.grad)[0] == pytest.approx(1.0)


def test_a_stored_gradient_matches_the_data_layout() -> None:
    x = leaf(np.arange(6, dtype=np.float32).reshape(2, 3))

    autograd.add(x, x).backward(_core.from_numpy(np.ones((2, 3), dtype=np.float32)))

    assert x.grad.shape == x.data.shape
    assert x.grad.is_contiguous()


# --------------------------------------------------------------------------
# Repeat passes
# --------------------------------------------------------------------------


def test_two_backward_calls_accumulate() -> None:
    """No retain_graph flag: nothing is freed, so a second pass simply sums.
    That is what makes zero_grad() in a training loop meaningful."""
    x = scalar(1.0)
    out = autograd.add(x, x)

    out.backward()
    out.backward()

    assert np.asarray(x.grad)[0] == pytest.approx(4.0)


def test_zero_grad_between_passes_restores_the_single_pass_answer() -> None:
    x = scalar(1.0)
    out = autograd.add(x, x)

    out.backward()
    x.zero_grad()
    out.backward()

    assert np.asarray(x.grad)[0] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# The seed
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param([1], id="rank-1"),
        pytest.param([1, 1], id="rank-2"),
        pytest.param([1, 1, 1], id="rank-3"),
    ],
)
def test_the_implicit_seed_works_at_any_rank(shape: list[int]) -> None:
    """numel() == 1 is equally true of (1, 1, 1), and _core requires the index
    arity to equal the rank - so the seed index is (0,) * rank, not [0]."""
    x = autograd.Tensor(_core.zeros(shape), requires_grad=True)

    autograd.add(x, x).backward()

    assert np.asarray(x.grad).ravel()[0] == pytest.approx(2.0)


def test_an_explicit_gradient_may_be_strided() -> None:
    """add_into reads any strides on its source, so a transposed seed is legal.

    Values in order is the assertion. The shape is 2x3 and the values distinct
    precisely so a wrong-order result cannot pass; a sum would wave it through.
    """
    seed = np.arange(6, dtype=np.float32).reshape(3, 2)
    x = leaf(np.zeros((2, 3)))

    strided = _core.from_numpy(seed).transpose(0, 1)
    assert not strided.is_contiguous()

    autograd.add(x, x).backward(strided)

    assert np.array_equal(np.asarray(x.grad), 2.0 * seed.T)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


# NotImplementedError is a SUBCLASS of RuntimeError, so a bare
# pytest.raises(RuntimeError) is satisfied by the unwritten stub and these two
# tests go green against an empty engine. Both therefore match on the message,
# which also forces the failure to say which condition was violated.


def test_backward_on_a_tensor_that_does_not_require_grad_raises() -> None:
    a = leaf(np.zeros([1]), requires_grad=False)
    out = autograd.add(a, leaf(np.zeros([1]), requires_grad=False))

    with pytest.raises(RuntimeError, match="require"):
        out.backward()


def test_an_implicit_seed_on_a_non_scalar_raises() -> None:
    x = leaf(np.zeros((2, 3)))
    out = autograd.add(x, x)

    with pytest.raises(RuntimeError, match="scalar"):
        out.backward()


def test_a_mismatched_explicit_gradient_raises() -> None:
    x = leaf(np.zeros((2, 3)))
    out = autograd.add(x, x)

    with pytest.raises(ValueError):
        out.backward(_core.zeros([3, 2]))


# --------------------------------------------------------------------------
# Lifetime
# --------------------------------------------------------------------------


def _build_graph_and_watch_it() -> tuple[autograd.Tensor, list[weakref.ref]]:
    """Build a graph from leaves created HERE, and weakref its interior.

    The leaves must be local. Mul saves the raw operands it was handed, and for
    a leaf that is the very _core.Tensor a caller would still be holding through
    x.data - so a caller-supplied leaf keeps its weakref alive for reasons that
    have nothing to do with cycles, and the test would pass on a broken graph.
    """
    x = leaf(np.arange(4, dtype=np.float32).reshape(2, 2))
    y = leaf(np.full((2, 2), 3.0))

    mid = autograd.mul(x, y)
    root = autograd.mul(mid, x)

    watched = [
        weakref.ref(x),
        weakref.ref(y),
        weakref.ref(mid),
        weakref.ref(x.data),
        weakref.ref(y.data),
        weakref.ref(mid.data),
    ]
    return root, watched


def test_dropping_the_root_frees_the_whole_graph_without_the_collector() -> None:
    """The exact assertion: plain refcounting reclaims it, so there is no cycle.

    Calling gc.collect() before the assertion would make this pass on a graph
    made entirely of cycles, which is the failure it exists to detect. The saved
    raw buffers are the half that matters - wrapper objects dying while a
    megabyte of saved activations survives is the phase 8 leak.
    """
    gc.disable()
    try:
        root, watched = _build_graph_and_watch_it()

        assert all(ref() is not None for ref in watched), "graph died early"

        del root

        alive = [i for i, ref in enumerate(watched) if ref() is not None]
        assert alive == [], f"still reachable after dropping the root: {alive}"
    finally:
        gc.enable()
