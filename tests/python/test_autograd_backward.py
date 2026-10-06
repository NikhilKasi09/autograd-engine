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
import autograd.engine
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


class _ScaleByData(autograd.engine.Function):
    """a * b where b is treated as data: backward returns None for it."""

    def forward(self, a: _core.Tensor, b: _core.Tensor) -> _core.Tensor:
        self.saved = (b,)
        out = _core.zeros_like(a)
        _core.mul(a, b, out)
        return out

    def backward(self, grad_out: _core.Tensor):
        (b,) = self.saved
        grad_a = _core.zeros_like(grad_out)
        _core.mul(grad_out, b, grad_a)
        return (grad_a, None)


def test_an_input_an_op_declares_non_differentiable_gets_no_gradient() -> None:
    """None from backward means "not differentiable", even for a parent that
    asked. This used to be a KeyError: the walk reached the parent and found
    nothing waiting for it."""
    x = leaf(np.array([1.0, 2.0]))
    data = leaf(np.array([3.0, 4.0]), requires_grad=True)

    _ScaleByData.apply(x, data).backward(_core.from_numpy(np.ones(2, dtype=np.float32)))

    assert np.array_equal(np.asarray(x.grad), [3.0, 4.0])
    assert data.grad is None


def test_nothing_upstream_of_a_non_differentiable_input_gets_a_gradient() -> None:
    """The skip carries on up: `mid` got nothing, so it has nothing to hand
    its own parents and must not call backward with a missing gradient."""
    x = leaf(np.array([1.0, 2.0]))
    y = leaf(np.array([5.0, 6.0]))
    mid = autograd.add(y, y)
    mid.retain_grad()

    _ScaleByData.apply(x, mid).backward(_core.from_numpy(np.ones(2, dtype=np.float32)))

    assert np.array_equal(np.asarray(x.grad), [10.0, 12.0])
    assert mid.grad is None
    assert y.grad is None


def test_a_tensor_still_gets_the_gradient_from_its_differentiable_uses() -> None:
    """Used once as data and once as a real operand: only the second counts."""
    x = leaf(np.array([1.0, 2.0]))
    y = leaf(np.array([5.0, 6.0]))

    out = autograd.add(_ScaleByData.apply(x, y), y)
    out.backward(_core.from_numpy(np.ones(2, dtype=np.float32)))

    assert np.array_equal(np.asarray(y.grad), [1.0, 1.0])


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


def test_matmul_with_a_single_row_can_be_differentiated() -> None:
    """Regression: {1,K} @ {K,N} raised in backward for every K > 1.

    dB = a.T @ dout, and a.T of a {1,K} is {K,1}. That IS contiguous - the
    stride of an extent-1 dimension is never stepped - so contiguous() no-opped,
    but gemm wants stride(1) == 1 literally and rejected it. Two definitions of
    contiguous that agree everywhere except on a dimension of extent one.

    M == 1 is batch size one, which the nn layer hits on its first step.
    """
    a = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    b = np.arange(15.0).reshape(5, 3)
    dout = np.array([[1.0, 2.0, 3.0]])

    x, w = leaf(a), leaf(b)
    autograd.matmul(x, w).backward(_core.from_numpy(np.ascontiguousarray(dout, np.float32)))

    assert np.allclose(np.asarray(x.grad), dout @ b.T, rtol=1e-6, atol=1e-6)
    assert np.allclose(np.asarray(w.grad), a.T @ dout, rtol=1e-6, atol=1e-6)


def test_matmul_with_a_single_inner_dimension_can_be_differentiated() -> None:
    """The mirror of the above: K == 1 breaks dA = dout @ b.T the same way."""
    a = np.array([[1.0], [2.0], [3.0]])
    b = np.array([[4.0, 5.0, 6.0]])
    dout = np.ones((3, 3))

    x, w = leaf(a), leaf(b)
    autograd.matmul(x, w).backward(_core.from_numpy(np.ascontiguousarray(dout, np.float32)))

    assert np.allclose(np.asarray(x.grad), dout @ b.T, rtol=1e-6, atol=1e-6)
    assert np.allclose(np.asarray(w.grad), a.T @ dout, rtol=1e-6, atol=1e-6)


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


# --------------------------------------------------------------------------
# sum - a gradient that is a view rather than a buffer
# --------------------------------------------------------------------------


def test_sum_backward_spreads_the_gradient_over_every_element() -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    x = leaf(a)

    autograd.sum(x).backward()

    assert np.array_equal(np.asarray(x.grad), np.ones((2, 3), dtype=np.float32))


def test_sum_backward_over_one_axis_matches_numpy() -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    x = leaf(a)

    out = autograd.sum(x, shape=(1, 3))
    dout = np.array([[2.0, 3.0, 4.0]], dtype=np.float32)
    out.backward(_core.from_numpy(dout))

    # Each column's gradient is broadcast back down its rows.
    assert np.array_equal(np.asarray(x.grad), np.repeat(dout, 2, axis=0))


def test_summing_to_a_scalar_needs_no_explicit_seed() -> None:
    """The path a loss takes: numel() == 1, so backward() seeds itself."""
    x = leaf(np.arange(4, dtype=np.float32).reshape(2, 2))

    autograd.sum(x).backward()

    assert x.grad is not None


def test_sum_backward_returns_a_stride_zero_view_not_a_buffer() -> None:
    """The assertion that fails if the body quietly materialises the gradient.

    zeros_like plus a kernel would give identical values everywhere. The point
    of keeping the output's rank is that expand alone is legal here, so the
    gradient costs no allocation - and add_into reads a stride-0 source fine.
    """
    x = leaf(np.ones((2, 3)))
    out = autograd.sum(x, shape=(1, 3))

    dout = _core.from_numpy(np.ones((1, 3), dtype=np.float32))
    (grad,) = out.grad_fn.backward(dout)

    assert grad.shape == (2, 3)
    assert not grad.is_contiguous()
    assert grad.shares_storage_with(dout)
    assert 0 in grad.strides


def test_sum_backward_accepts_a_strided_incoming_gradient() -> None:
    """No .contiguous().reshape(...) anywhere, so a strided seed is not special.

    This is the case that would raise if Sum's output were rank 1: reshape
    throws on a non-contiguous tensor.
    """
    x = leaf(np.ones((2, 3)))
    out = autograd.sum(x, shape=(1, 3))

    # Transposing a {3,1} is NOT enough: it gives strides (1, 1), and
    # is_contiguous skips extent-1 dimensions, so it comes back contiguous.
    # A {1,3} that is genuinely strided needs a real gap on the last axis -
    # transpose a {3,2} to {2,3} with strides (1, 2), then take one row.
    base = np.array([[2.0, 0.0], [3.0, 0.0], [4.0, 0.0]], dtype=np.float32)
    strided = _core.from_numpy(base).transpose(0, 1).slice(0, 0, 1)

    assert strided.shape == (1, 3)
    assert strided.strides == (1, 2)
    assert not strided.is_contiguous()
    assert np.array_equal(np.asarray(strided), np.array([[2.0, 3.0, 4.0]]))

    out.backward(strided)

    assert np.array_equal(
        np.asarray(x.grad), np.repeat(np.array([[2.0, 3.0, 4.0]], np.float32), 2, axis=0)
    )


def test_a_leaf_feeding_a_sum_and_something_else_accumulates_both() -> None:
    """A stride-0 view and a plain buffer accumulating into one .grad.

    Phase 5 had one pass-through; Sum returns a view OVER the incoming gradient,
    so add_into now sees sources of two different kinds in a single pass. No
    true overlap is reachable today - accumulate always allocates dst fresh -
    and the point is to pin that while the reason it holds is still one line.
    """
    a = np.arange(4, dtype=np.float32).reshape(2, 2) + 1.0
    x = leaf(a)

    total = autograd.sum(x)                       # {1,1}, gradient is a view
    doubled = autograd.mul(x, x)                  # {2,2}, gradient is a buffer
    root = autograd.add(total, autograd.sum(doubled))

    root.backward()

    # d/dx of (sum(x) + sum(x*x)) is 1 + 2x elementwise.
    assert np.allclose(np.asarray(x.grad), 1.0 + 2.0 * a, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------
# expand - Sum's dual, and the phase's deliverable
# --------------------------------------------------------------------------


def test_expand_backward_collapses_to_the_input_shape() -> None:
    """The bias gradient. Its shape is the INPUT's, not the gradient's."""
    x = leaf(np.array([[1.0, 2.0, 4.0]], dtype=np.float32))

    out = autograd.expand(x, (2, 3))
    dout = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32)
    out.backward(_core.from_numpy(dout))

    assert x.grad.shape == (1, 3)
    assert np.array_equal(np.asarray(x.grad), dout.sum(axis=0, keepdims=True))


def test_expand_then_sum_round_trips() -> None:
    """Both ops and the one kernel underneath them, in a single assertion."""
    row = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
    x = leaf(row)

    autograd.sum(autograd.expand(x, (4, 3)), shape=(1, 3)).backward(
        _core.from_numpy(np.ones((1, 3), dtype=np.float32))
    )

    # Each element is read four times going up and summed four times coming
    # back, so the gradient is the expansion factor.
    assert np.array_equal(np.asarray(x.grad), np.full((1, 3), 4.0, dtype=np.float32))


def test_a_bias_shaped_chain_matches_numpy() -> None:
    """Phase 8's Linear layer, written by hand. Why Expand is in this phase."""
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    w = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = np.array([[0.5, 1.5, 2.5, 3.5]], dtype=np.float32)

    x, wt, bias = leaf(a), leaf(w), leaf(b)

    out = autograd.add(autograd.matmul(x, wt), autograd.expand(bias, (2, 4)))
    dout = np.arange(8, dtype=np.float32).reshape(2, 4) + 1.0
    out.backward(_core.from_numpy(np.ascontiguousarray(dout)))

    assert np.allclose(out.to_numpy(), a @ w + b, rtol=1e-6, atol=1e-6)
    assert np.allclose(np.asarray(bias.grad), dout.sum(axis=0, keepdims=True), rtol=1e-6, atol=1e-6)
    assert np.allclose(np.asarray(x.grad), dout @ w.T, rtol=1e-6, atol=1e-6)
    assert np.allclose(np.asarray(wt.grad), a.T @ dout, rtol=1e-6, atol=1e-6)


def test_a_two_layer_mlp_trains_end_to_end() -> None:
    """THE deliverable of phase 6. Every op in it, arranged as phase 8 will.

    matmul -> expand bias -> add -> relu -> matmul -> expand bias -> add -> sum,
    then backward() with no seed, checked against a hand-written NumPy pass.
    """
    rng = np.random.default_rng(0xC0FFEE)
    x_ = rng.standard_normal((4, 5)).astype(np.float32)
    w1_ = rng.standard_normal((5, 3)).astype(np.float32)
    b1_ = rng.standard_normal((1, 3)).astype(np.float32)
    w2_ = rng.standard_normal((3, 2)).astype(np.float32)
    b2_ = rng.standard_normal((1, 2)).astype(np.float32)

    x = leaf(x_, requires_grad=False)
    w1, b1, w2, b2 = leaf(w1_), leaf(b1_), leaf(w2_), leaf(b2_)

    h = autograd.relu(autograd.add(autograd.matmul(x, w1), autograd.expand(b1, (4, 3))))
    out = autograd.add(autograd.matmul(h, w2), autograd.expand(b2, (4, 2)))
    loss = autograd.sum(out)

    loss.backward()

    # The same pass by hand.
    z1 = x_ @ w1_ + b1_
    a1 = np.maximum(z1, 0.0)
    z2 = a1 @ w2_ + b2_

    dz2 = np.ones_like(z2)
    db2 = dz2.sum(axis=0, keepdims=True)
    dw2 = a1.T @ dz2
    da1 = dz2 @ w2_.T
    dz1 = np.where(z1 > 0, da1, 0.0)
    db1 = dz1.sum(axis=0, keepdims=True)
    dw1 = x_.T @ dz1

    assert np.allclose(loss.to_numpy(), z2.sum(), rtol=1e-5, atol=1e-5)
    for got, want, name in [
        (w1.grad, dw1, "w1"), (b1.grad, db1, "b1"),
        (w2.grad, dw2, "w2"), (b2.grad, db2, "b2"),
    ]:
        assert got is not None, f"{name} received no gradient"
        assert got.shape == want.shape, f"{name}: {got.shape} != {want.shape}"
        assert got.is_contiguous(), f"{name} gradient is not contiguous"
        assert np.allclose(np.asarray(got), want, rtol=1e-5, atol=1e-5), name

    assert x.grad is None


def _build_expand_graph_and_watch_it() -> tuple[autograd.Tensor, list[weakref.ref]]:
    """Expand's output SHARES its input's buffer, so a naive lifetime argument
    might expect that to keep something alive. It does not: the view holds a
    refcount on the storage, not on the graph tensor."""
    x = leaf(np.array([[1.0, 2.0]], dtype=np.float32))
    y = leaf(np.full((2, 2), 3.0))

    wide = autograd.expand(x, (2, 2))
    root = autograd.mul(wide, y)

    watched = [weakref.ref(x), weakref.ref(y), weakref.ref(wide), weakref.ref(wide.data)]
    return root, watched


def test_dropping_an_expand_graph_frees_it_without_the_collector() -> None:
    gc.disable()
    try:
        root, watched = _build_expand_graph_and_watch_it()
        assert all(ref() is not None for ref in watched), "graph died early"

        del root

        alive = [i for i, ref in enumerate(watched) if ref() is not None]
        assert alive == [], f"still reachable after dropping the root: {alive}"
    finally:
        gc.enable()
