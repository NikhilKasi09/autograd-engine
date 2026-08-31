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
from autograd.ops import _gemm_into


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
# Multiplication: the saved-tensor path
# --------------------------------------------------------------------------


def test_each_operand_receives_the_others_value() -> None:
    """d(a*b)/da is b, and d(a*b)/db is a.

    The values are distinct and the shape is non-square on purpose: with a == b,
    or on a symmetric shape, a backward that returned the pair the wrong way
    round passes.
    """
    a = np.arange(6, dtype=np.float32).reshape(2, 3) + 1.0
    b = np.arange(6, dtype=np.float32).reshape(2, 3) * 10.0 + 3.0

    x, y = leaf(a), leaf(b)
    autograd.mul(x, y).backward(_core.from_numpy(np.ones((2, 3), dtype=np.float32)))

    assert np.array_equal(np.asarray(x.grad), b)
    assert np.array_equal(np.asarray(y.grad), a)


def test_squaring_a_tensor_accumulates_both_uses() -> None:
    """d(x*x)/dx = 2x - saving and accumulation exercised together."""
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = leaf(a)

    autograd.mul(x, x).backward(_core.from_numpy(np.ones(3, dtype=np.float32)))

    assert np.array_equal(np.asarray(x.grad), 2.0 * a)


def test_a_chain_matches_a_hand_derived_gradient() -> None:
    """f = (x*y)*x = x^2 y, so df/dx = 2xy and df/dy = x^2."""
    a = np.arange(6, dtype=np.float32).reshape(2, 3) + 1.0
    b = np.arange(6, dtype=np.float32).reshape(2, 3) + 2.0

    x, y = leaf(a), leaf(b)
    autograd.mul(autograd.mul(x, y), x).backward(
        _core.from_numpy(np.ones((2, 3), dtype=np.float32))
    )

    assert np.allclose(np.asarray(x.grad), 2.0 * a * b, rtol=1e-6, atol=1e-6)
    assert np.allclose(np.asarray(y.grad), a * a, rtol=1e-6, atol=1e-6)


def test_the_walk_order_survives_a_diamond_with_multiplication() -> None:
    """The version step 3 could not run, now that Mul.backward exists.

    a = x^2, b = a + x, c = b*a = x^4 + x^3, so dc/dx = 4x^3 + 3x^2 = 44 at x=2.
    The add-only diamond proves the order; this one proves it holds when the two
    contributions arriving at `a` have different magnitudes, which is where a
    dropped path stops being a whole missing term and starts being a plausible
    wrong number.
    """
    x = scalar(2.0)

    a = autograd.mul(x, x)
    b = autograd.add(a, x)
    c = autograd.mul(b, a)

    assert c.to_numpy()[0] == pytest.approx(24.0)  # x^4 + x^3

    c.backward()

    assert np.asarray(x.grad)[0] == pytest.approx(44.0)


def test_a_frozen_operand_of_mul_keeps_no_gradient() -> None:
    a = np.array([2.0, 3.0], dtype=np.float32)
    b = np.array([5.0, 7.0], dtype=np.float32)

    x = leaf(a, requires_grad=True)
    y = leaf(b, requires_grad=False)
    autograd.mul(x, y).backward(_core.from_numpy(np.ones(2, dtype=np.float32)))

    assert np.array_equal(np.asarray(x.grad), b)
    assert y.grad is None


def test_mul_backward_reads_a_strided_incoming_gradient() -> None:
    """The root's gradient is the caller's buffer, so it can be strided, and
    Mul.backward feeds it straight to a kernel. Values in order is the assertion.
    """
    seed = np.arange(6, dtype=np.float32).reshape(3, 2) + 1.0
    a = np.arange(6, dtype=np.float32).reshape(2, 3) + 1.0
    b = np.arange(6, dtype=np.float32).reshape(2, 3) * 2.0 + 5.0

    x, y = leaf(a), leaf(b)
    strided = _core.from_numpy(seed).transpose(0, 1)
    assert not strided.is_contiguous()

    autograd.mul(x, y).backward(strided)

    assert np.array_equal(np.asarray(x.grad), seed.T * b)
    assert np.array_equal(np.asarray(y.grad), seed.T * a)


def test_a_stored_mul_gradient_matches_the_data_layout() -> None:
    x = leaf(np.arange(6, dtype=np.float32).reshape(2, 3) + 1.0)
    y = leaf(np.arange(6, dtype=np.float32).reshape(2, 3) + 4.0)

    autograd.mul(x, y).backward(_core.from_numpy(np.ones((2, 3), dtype=np.float32)))

    for t in (x, y):
        assert t.grad.shape == t.data.shape
        assert t.grad.is_contiguous()


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


# --------------------------------------------------------------------------
# matmul - the shapes are the mirror of the forward
# --------------------------------------------------------------------------


def test_matmul_backward_matches_a_hand_derived_gradient() -> None:
    """Non-square at every position, which is what makes this test worth having.

    Forward is {2,3} @ {3,4} -> {2,4}, so dA is {2,3} and dB is {3,4} and the
    incoming gradient is {2,4}. All three differ. On a square case a swapped
    pair, a missing transpose and a transposed result all run and all pass.
    """
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.arange(12, dtype=np.float32).reshape(3, 4)

    x, y = leaf(a), leaf(b)
    out = autograd.matmul(x, y)

    dout = np.arange(8, dtype=np.float32).reshape(2, 4) + 1.0
    out.backward(_core.from_numpy(np.ascontiguousarray(dout)))

    assert np.allclose(np.asarray(x.grad), dout @ b.T, rtol=1e-6, atol=1e-6)
    assert np.allclose(np.asarray(y.grad), a.T @ dout, rtol=1e-6, atol=1e-6)


def test_matmul_gradients_take_their_shape_from_the_operands() -> None:
    """zeros_like(grad_out) is {2,4} - the wrong shape for BOTH gradients."""
    x, y = leaf(np.ones((2, 3))), leaf(np.ones((3, 4)))
    out = autograd.matmul(x, y)

    out.backward(_core.from_numpy(np.ones((2, 4), dtype=np.float32)))

    assert x.grad.shape == (2, 3)
    assert y.grad.shape == (3, 4)
    assert x.grad.is_contiguous()
    assert y.grad.is_contiguous()


def test_matmul_backward_reads_a_strided_incoming_gradient() -> None:
    """gemm rejects a non-unit inner stride on ALL THREE operands.

    An incoming gradient is not guaranteed contiguous - Add.backward passes one
    straight through. A _gemm_into that only contiguifies the transposed operand
    raises ValueError here the moment a matmul sits downstream of an add.
    """
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.arange(12, dtype=np.float32).reshape(3, 4)

    x, y = leaf(a), leaf(b)
    out = autograd.matmul(x, y)

    # A {2,4} view whose inner stride is 4, not 1.
    dout = np.arange(8, dtype=np.float32).reshape(4, 2) + 1.0
    strided = _core.from_numpy(np.ascontiguousarray(dout)).transpose(0, 1)
    assert not strided.is_contiguous()

    out.backward(strided)

    assert np.allclose(np.asarray(x.grad), dout.T @ b.T, rtol=1e-6, atol=1e-6)
    assert np.allclose(np.asarray(y.grad), a.T @ dout.T, rtol=1e-6, atol=1e-6)


def test_matmul_leaves_a_frozen_operand_without_a_gradient() -> None:
    x = leaf(np.ones((2, 3)), requires_grad=True)
    y = leaf(np.ones((3, 4)), requires_grad=False)

    out = autograd.matmul(x, y)
    out.backward(_core.from_numpy(np.ones((2, 4), dtype=np.float32)))

    assert y.grad is None
    assert x.grad is not None


def test_a_leaf_matmulled_with_itself_accumulates_both_contributions() -> None:
    a = np.arange(4, dtype=np.float32).reshape(2, 2) + 1.0
    x = leaf(a)

    out = autograd.matmul(x, x)
    dout = np.ones((2, 2), dtype=np.float32)
    out.backward(_core.from_numpy(dout))

    # d/dX of X@X is dout @ X.T + X.T @ dout - one contribution per use.
    assert np.allclose(np.asarray(x.grad), dout @ a.T + a.T @ dout, rtol=1e-6, atol=1e-6)


def test_gemm_into_refuses_a_non_contiguous_destination() -> None:
    """The chokepoint's asymmetry, asserted directly.

    Contiguifying an input is a slow no-op. Contiguifying the OUTPUT fills a
    temporary and drops it, leaving the caller's buffer zero with nothing
    raised - so the helper must reject rather than repair.
    """
    a = _core.zeros([2, 3])
    b = _core.zeros([3, 4])
    tr_out = _core.zeros([4, 2]).transpose(0, 1)
    assert not tr_out.is_contiguous()

    with pytest.raises(ValueError):
        _gemm_into(a, b, tr_out)


# --------------------------------------------------------------------------
# relu - the mask, and the op that could close the graph into a cycle
# --------------------------------------------------------------------------


def test_relu_backward_masks_on_the_sign_of_its_input() -> None:
    a = np.array([[-2.0, 3.0], [0.5, -4.0]], dtype=np.float32)
    x = leaf(a)

    out = autograd.relu(x)
    dout = np.array([[5.0, 7.0], [-1.0, 9.0]], dtype=np.float32)
    out.backward(_core.from_numpy(dout))

    # Exact - representable, and a mask either passes a value or zeroes it.
    assert np.array_equal(np.asarray(x.grad), np.where(a > 0, dout, 0.0))


def test_relu_gives_no_gradient_at_exactly_zero() -> None:
    """The subgradient choice, pinned where a user can see it.

    relu has no derivative at 0 and every framework picks one. Picking 0 keeps
    the graph agreeing with the forward kernel, which outputs 0 there.
    """
    x = leaf(np.array([[0.0]]))

    out = autograd.relu(x)
    out.backward(_core.from_numpy(np.array([[5.0]], dtype=np.float32)))

    assert np.asarray(x.grad)[0, 0] == 0.0


def test_relu_returns_exactly_one_gradient() -> None:
    """A single-input op returns a ONE-tuple; the trailing comma is load-bearing.

    engine.backward zips parents against the returned tuple without strict=True,
    so a bare tensor is silently truncated rather than raising. This is what
    stands in for that missing strict=.
    """
    x = leaf(np.array([[1.0, -1.0]]))
    out = autograd.relu(x)

    grads = out.grad_fn.backward(_core.from_numpy(np.ones((1, 2), dtype=np.float32)))

    assert isinstance(grads, tuple)
    assert len(grads) == len(out.grad_fn.parents) == 1


def test_relu_gradient_takes_its_shape_from_the_input() -> None:
    x = leaf(np.ones((2, 3)))
    out = autograd.relu(x)
    out.backward(_core.from_numpy(np.ones((2, 3), dtype=np.float32)))

    assert x.grad.shape == (2, 3)
    assert x.grad.is_contiguous()


def test_relu_backward_reads_a_strided_incoming_gradient() -> None:
    a = np.array([[1.0, -1.0, 2.0], [3.0, 4.0, -5.0]], dtype=np.float32)
    x = leaf(a)
    out = autograd.relu(x)

    dout = np.arange(6, dtype=np.float32).reshape(3, 2) + 1.0
    strided = _core.from_numpy(np.ascontiguousarray(dout)).transpose(0, 1)
    assert not strided.is_contiguous()

    out.backward(strided)

    assert np.array_equal(np.asarray(x.grad), np.where(a > 0, dout.T, 0.0))


def _build_relu_graph_and_watch_it() -> tuple[autograd.Tensor, list[weakref.ref]]:
    """Same contract as the mul factory: leaves created HERE, never passed in.

    Relu saves the buffer it produced, so the interesting weakref is the one on
    an intermediate's .data - that is the buffer the node holds, and the object
    a cycle would strand.
    """
    x = leaf(np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32))
    y = leaf(np.full((2, 2), 3.0))

    hidden = autograd.relu(autograd.mul(x, y))
    root = autograd.mul(hidden, x)

    watched = [
        weakref.ref(x),
        weakref.ref(y),
        weakref.ref(hidden),
        weakref.ref(x.data),
        weakref.ref(hidden.data),
    ]
    return root, watched


def test_dropping_a_relu_graph_frees_it_without_the_collector() -> None:
    """Where saving the output WRAPPER instead of the buffer becomes visible.

    Every value test above passes either way. Here a saved graph Tensor closes
    output -> grad_fn -> saved -> output, and refcounting alone stops being
    enough. No gc.collect() before the assert: it would make this pass on a
    graph made entirely of cycles.
    """
    gc.disable()
    try:
        root, watched = _build_relu_graph_and_watch_it()

        assert all(ref() is not None for ref in watched), "graph died early"

        del root

        alive = [i for i, ref in enumerate(watched) if ref() is not None]
        assert alive == [], f"still reachable after dropping the root: {alive}"
    finally:
        gc.enable()
