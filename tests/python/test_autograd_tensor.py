"""The graph tensor's data half: wrapping, the gradient slot, detach.

Written before the implementation, so these fail on a fresh scaffold.

Nothing here builds a graph - there are no ops yet. What is being pinned is the
relationship between `autograd.Tensor` and the `_core.Tensor` it holds, and the
one assertion that carries real weight is identity rather than value. A wrapper
that cloned on construction, or a detach() that cloned, would give correct
values everywhere in this file and would then make phase 8's optimizer update a
buffer nobody reads.
"""

from __future__ import annotations

import numpy as np
import pytest

import autograd
from autograd import _core


def raw(a: np.ndarray) -> _core.Tensor:
    return _core.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_a_fresh_tensor_is_a_leaf_with_no_gradient() -> None:
    t = autograd.Tensor(raw(np.arange(6).reshape(2, 3)))

    assert t.requires_grad is False
    assert t.grad is None
    assert t.grad_fn is None
    assert t.is_leaf is True
    assert t.retains_grad is False


def test_requires_grad_is_taken_from_the_constructor() -> None:
    t = autograd.Tensor(raw(np.zeros((2, 2))), requires_grad=True)

    assert t.requires_grad is True
    # Asking for gradients does not create one; that is what a backward pass is
    # for, and it is why the slot is Optional rather than an eager zero buffer.
    assert t.grad is None
    assert t.is_leaf is True


def test_the_graph_wraps_the_core_tensor_rather_than_copying_it() -> None:
    """Identity, not values. A wrapper that cloned would pass a value test.

    Phase 8's optimizer mutates `param.data` in place. If construction copied,
    every update would land on a buffer nothing else can see, and the training
    loop would run to completion learning nothing.
    """
    data = raw(np.arange(6).reshape(2, 3))
    t = autograd.Tensor(data)

    assert t.data is data


def test_shape_forwards_to_the_wrapped_tensor() -> None:
    t = autograd.Tensor(raw(np.zeros((2, 3, 4))))

    assert t.shape == (2, 3, 4)
    assert t.shape == t.data.shape


def test_to_numpy_returns_the_values() -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = autograd.Tensor(raw(a))

    assert np.array_equal(t.to_numpy(), a)


# --------------------------------------------------------------------------
# The gradient slot
# --------------------------------------------------------------------------


def test_zero_grad_clears_the_slot_to_none() -> None:
    """None, not a zeroed buffer.

    The distinction is between "no gradient yet" and "a gradient that came out
    zero" - the second is a real answer, the first is a disconnected parameter,
    and a training loop that cannot tell them apart hides the bug.
    """
    t = autograd.Tensor(raw(np.zeros((2, 2))), requires_grad=True)
    t.grad = _core.zeros([2, 2])

    t.zero_grad()

    assert t.grad is None


def test_retain_grad_sets_the_flag() -> None:
    # Nothing reads this until the engine exists. It is declared now so the
    # engine fills a body rather than growing the type its tests were written
    # against.
    t = autograd.Tensor(raw(np.zeros((2, 2))), requires_grad=True)
    assert t.retains_grad is False

    t.retain_grad()

    assert t.retains_grad is True


# --------------------------------------------------------------------------
# detach
# --------------------------------------------------------------------------


def test_detach_shares_the_buffer_and_leaves_the_graph() -> None:
    """`out.data is t.data` - the exact assertion, not shares_storage_with.

    A clone() inside detach passes any value comparison. It also silently makes
    the detached handle useless for its actual job, which is writing to a
    parameter without recording the write.
    """
    t = autograd.Tensor(raw(np.arange(6).reshape(2, 3)), requires_grad=True)

    out = t.detach()

    assert out is not t
    assert out.data is t.data
    assert out.requires_grad is False
    assert out.grad_fn is None
    assert out.is_leaf is True


def test_a_write_through_a_detached_tensor_reaches_the_original() -> None:
    """The consequence of sharing, asserted rather than implied."""
    t = autograd.Tensor(raw(np.zeros((2, 2))), requires_grad=True)

    t.detach().data[1, 1] = 7.0

    assert t.data[1, 1] == 7.0


# --------------------------------------------------------------------------
# Not yet implemented
# --------------------------------------------------------------------------


def test_backward_is_declared_but_not_yet_implemented() -> None:
    """Pins the surface. The engine lands in a later step; the method exists now
    so that step writes a body instead of adding a method."""
    t = autograd.Tensor(raw(np.zeros([1])), requires_grad=True)

    with pytest.raises(NotImplementedError):
        t.backward()


# --------------------------------------------------------------------------
# repr
# --------------------------------------------------------------------------


def test_repr_names_the_shape_and_whether_it_requires_grad() -> None:
    r = repr(autograd.Tensor(raw(np.zeros((2, 3))), requires_grad=True))

    assert "Tensor" in r
    assert "2" in r and "3" in r
    assert "requires_grad" in r
