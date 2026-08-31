"""The graph: what a forward pass records, and what it declines to record.

Written before the implementation, so these fail on a fresh scaffold.

No gradients here - backward does not exist yet. What is pinned is the shape of
the recording: who the parents are, by identity; that requires_grad propagates;
that no node is built when nobody asked for one; and that `saved` holds raw
buffers rather than graph tensors, which is the rule the whole cycle argument
rests on.
"""

from __future__ import annotations

import numpy as np
import pytest

import autograd
from autograd import _core
from autograd.ops import Add, Matmul, Mul, Relu


def leaf(a: np.ndarray, requires_grad: bool = False) -> autograd.Tensor:
    raw = _core.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    return autograd.Tensor(raw, requires_grad=requires_grad)


# --------------------------------------------------------------------------
# Forward values
# --------------------------------------------------------------------------


def test_add_matches_numpy() -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.arange(6, dtype=np.float32).reshape(2, 3) * 10

    out = autograd.add(leaf(a), leaf(b))

    # Exact, not allclose. These are representable.
    assert np.array_equal(out.to_numpy(), a + b)


def test_mul_is_elementwise_not_a_matrix_product() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    out = autograd.mul(leaf(a), leaf(a))

    assert np.array_equal(out.to_numpy(), a * a)
    # The assertion that makes the previous line mean something: on a 2x2 the
    # two answers have the same shape, so shape alone would not catch it.
    assert not np.array_equal(out.to_numpy(), a @ a)


def test_the_forward_output_is_a_fresh_buffer() -> None:
    """The op allocates its own output rather than writing into an operand."""
    x = leaf(np.ones((2, 3)))

    out = autograd.add(x, x)

    assert not out.data.shares_storage_with(x.data)
    assert out.data.is_contiguous()


# --------------------------------------------------------------------------
# What gets recorded
# --------------------------------------------------------------------------


def test_a_node_records_its_parents_by_identity() -> None:
    x = leaf(np.ones((2, 2)), requires_grad=True)
    y = leaf(np.ones((2, 2)))

    out = autograd.add(x, y)

    assert isinstance(out.grad_fn, Add)
    # `is`, not ==. The graph has to lead back to these exact objects, because
    # that is where the gradients are going to be written.
    assert out.grad_fn.parents[0] is x
    assert out.grad_fn.parents[1] is y
    assert out.is_leaf is False


def test_parents_keep_argument_order() -> None:
    """Backward returns gradients positionally, so the order is load-bearing."""
    x = leaf(np.ones((2, 2)), requires_grad=True)
    y = leaf(np.zeros((2, 2)), requires_grad=True)

    assert autograd.mul(x, y).grad_fn.parents == (x, y)
    assert autograd.mul(y, x).grad_fn.parents == (y, x)


@pytest.mark.parametrize(
    ("a_needs", "b_needs", "expected"),
    [
        pytest.param(True, True, True, id="both"),
        pytest.param(True, False, True, id="first-only"),
        pytest.param(False, True, True, id="second-only"),
        pytest.param(False, False, False, id="neither"),
    ],
)
def test_requires_grad_propagates_from_either_input(
    a_needs: bool, b_needs: bool, expected: bool
) -> None:
    out = autograd.add(
        leaf(np.ones((2, 2)), requires_grad=a_needs),
        leaf(np.ones((2, 2)), requires_grad=b_needs),
    )

    assert out.requires_grad is expected


def test_no_node_is_built_when_nothing_requires_grad() -> None:
    """The test that earns its place.

    A requires_grad-only assertion passes on an engine that builds the node
    anyway and merely flags it False. That version leaks a graph per forward
    pass for the whole of phase 8, and shows up as memory growth rather than as
    a wrong number - which is the kind of bug that survives to the demo.
    """
    out = autograd.mul(leaf(np.ones((2, 2))), leaf(np.ones((2, 2))))

    assert out.grad_fn is None
    assert out.is_leaf is True
    assert out.requires_grad is False


# --------------------------------------------------------------------------
# The cycle rule
# --------------------------------------------------------------------------


def test_saved_holds_raw_tensors_and_never_graph_tensors() -> None:
    """Asserted directly rather than inferred from the lifetime test passing.

    Saving a graph Tensor would close the loop output -> grad_fn -> saved ->
    output, and hand the whole graph to the cycle collector. Phase 6's relu is
    where the temptation actually arrives.
    """
    x = leaf(np.arange(4).reshape(2, 2), requires_grad=True)
    y = leaf(np.ones((2, 2)))

    saved = autograd.mul(x, y).grad_fn.saved

    assert len(saved) > 0
    for item in saved:
        assert isinstance(item, _core.Tensor)
        assert not isinstance(item, autograd.Tensor)


def test_mul_saves_the_operand_values() -> None:
    """Values, so a version saving the wrong pair is visible now, not in step 4."""
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.arange(6, dtype=np.float32).reshape(2, 3) * 10 + 1.0

    saved = autograd.mul(leaf(a, requires_grad=True), leaf(b)).grad_fn.saved

    assert np.array_equal(np.asarray(saved[0]), a)
    assert np.array_equal(np.asarray(saved[1]), b)


# --------------------------------------------------------------------------
# The graph layer adds nothing
# --------------------------------------------------------------------------


def test_a_shape_mismatch_raises_from_core_unchanged() -> None:
    """No broadcasting is invented at the graph layer.

    _core requires shapes to match exactly and says so; the graph must not
    quietly grow an expand() on the way in.
    """
    with pytest.raises(ValueError):
        autograd.add(leaf(np.ones((2, 3))), leaf(np.ones((3, 2))))


def test_a_transposed_operand_is_read_in_place() -> None:
    """The same guard the op bindings carry: no helpful .contiguous() on the way in.

    Values in the right order is the assertion. A sum would pass on a wrong-order
    result, and so would a symmetric input, which is why the shape is 2x3.
    """
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.arange(6, dtype=np.float32).reshape(3, 2) * 10

    x = autograd.Tensor(
        _core.from_numpy(np.ascontiguousarray(a)).transpose(0, 1), requires_grad=True
    )
    assert not x.data.is_contiguous()

    out = autograd.add(x, leaf(b))

    assert np.array_equal(out.to_numpy(), a.T + b)


# --------------------------------------------------------------------------
# matmul - what a matrix product records
# --------------------------------------------------------------------------


def test_matmul_records_a_node_with_its_parents_in_order() -> None:
    x = leaf(np.ones((2, 3)), requires_grad=True)
    y = leaf(np.ones((3, 4)), requires_grad=True)

    out = autograd.matmul(x, y)

    assert isinstance(out.grad_fn, Matmul)
    assert out.grad_fn.parents == (x, y)
    assert out.requires_grad


def test_matmul_output_has_the_product_shape_and_its_own_storage() -> None:
    x = leaf(np.ones((2, 3)), requires_grad=True)
    y = leaf(np.ones((3, 4)), requires_grad=True)

    out = autograd.matmul(x, y)

    assert out.shape == (2, 4)
    assert out.data.is_contiguous()
    assert not out.data.shares_storage_with(x.data)
    assert not out.data.shares_storage_with(y.data)


def test_matmul_forward_matches_numpy() -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.arange(12, dtype=np.float32).reshape(3, 4)

    out = autograd.matmul(leaf(a), leaf(b))

    assert np.allclose(out.to_numpy(), a @ b, rtol=1e-6, atol=1e-6)


def test_matmul_saves_raw_buffers_and_never_graph_tensors() -> None:
    """The cycle rule, restated for the first op that saves two big operands."""
    x = leaf(np.ones((2, 3)), requires_grad=True)
    y = leaf(np.ones((3, 4)), requires_grad=True)

    out = autograd.matmul(x, y)

    assert len(out.grad_fn.saved) == 2
    for item in out.grad_fn.saved:
        assert isinstance(item, _core.Tensor)
        assert not isinstance(item, autograd.Tensor)


def test_matmul_builds_no_node_when_nothing_requires_grad() -> None:
    out = autograd.matmul(leaf(np.ones((2, 3))), leaf(np.ones((3, 4))))
    assert out.grad_fn is None
    assert not out.requires_grad


# --------------------------------------------------------------------------
# relu - the op that saves its own output
# --------------------------------------------------------------------------


def test_relu_records_a_single_parent() -> None:
    x = leaf(np.array([[-1.0, 2.0]]), requires_grad=True)

    out = autograd.relu(x)

    assert isinstance(out.grad_fn, Relu)
    assert out.grad_fn.parents == (x,)


def test_relu_forward_clamps_at_zero() -> None:
    a = np.array([[-2.0, -0.0, 0.0, 3.0]], dtype=np.float32)

    out = autograd.relu(leaf(a))

    assert np.array_equal(out.to_numpy(), np.maximum(a, 0.0))


def test_relu_saves_its_own_output_buffer_and_not_the_wrapper() -> None:
    """The one-word change that closes the graph into a cycle.

    Saving `result` rather than the raw buffer gives identical numbers on every
    other test in the suite. Here the saved item must BE the output's buffer -
    same object - and must not be a graph Tensor.
    """
    x = leaf(np.array([[-1.0, 2.0]]), requires_grad=True)

    out = autograd.relu(x)

    assert len(out.grad_fn.saved) == 1
    saved = out.grad_fn.saved[0]

    assert isinstance(saved, _core.Tensor)
    assert not isinstance(saved, autograd.Tensor)
    assert saved is out.data
