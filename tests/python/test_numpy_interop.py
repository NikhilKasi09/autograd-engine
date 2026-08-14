"""The Tensor <-> NumPy boundary: buffer protocol, to_numpy, from_numpy.

Written before the binding, so these fail on a fresh scaffold.

The recurring theme is that strides cross the buffer protocol in BYTES while
Tensor::stride is in ELEMENTS. Passing elements does not fail - it hands numpy
a wrong-but-plausible layout, so it reads the right buffer with the wrong step
and returns the right values in the wrong order. That is invisible to a
checksum, invisible to sum(), invisible on a square symmetric matrix, and
invisible at rank 1. Every test below that could have been written on a square
matrix of ones is deliberately written on an asymmetric shape with distinct
values instead.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

from autograd import _core


def base_2x3() -> tuple[_core.Tensor, np.ndarray]:
    """A 2x3 of 0..5. Asymmetric, and every element distinguishable."""
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    return _core.from_numpy(a), a


# --------------------------------------------------------------------------
# from_numpy
# --------------------------------------------------------------------------


def test_from_numpy_round_trips_values_and_shape() -> None:
    t, a = base_2x3()

    assert t.shape == (2, 3)
    assert t.strides == (3, 1)
    assert t.is_contiguous()
    assert t[1, 2] == 5.0


def test_from_numpy_accepts_float64_and_fortran_order() -> None:
    # forcecast converts dtype, c_style converts layout. Both, together, so the
    # conversion path is exercised rather than the fast path.
    a = np.asfortranarray(np.arange(6, dtype=np.float64).reshape(2, 3))
    t = _core.from_numpy(a)

    assert t.shape == (2, 3)
    assert t.is_contiguous()
    assert t[0, 1] == 1.0
    assert t[1, 0] == 3.0


def test_from_numpy_accepts_a_strided_view() -> None:
    # arr[::2] is non-contiguous, so numpy has to materialise a copy before the
    # binding sees it. A test that only ever passes a fresh C-order float32
    # array never touches that path.
    a = np.arange(12, dtype=np.float32).reshape(6, 2)[::2]
    assert not a.flags.c_contiguous

    t = _core.from_numpy(a)

    assert t.shape == (3, 2)
    assert t[0, 0] == 0.0
    assert t[1, 0] == 4.0
    assert t[2, 1] == 9.0


def test_from_numpy_copies_rather_than_aliasing() -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = _core.from_numpy(a)

    a[0, 0] = 99.0
    assert t[0, 0] == 0.0


def test_from_numpy_handles_every_rank_up_to_max() -> None:
    # The old row-loop conversion was hardcoded to rank 2. One flat memcpy is
    # rank-general, and this is what says so.
    for rank in range(1, _core.MAX_RANK + 1):
        shape = tuple(range(2, 2 + rank))
        a = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
        t = _core.from_numpy(a)

        assert t.shape == shape
        assert t.numel() == a.size
        # Last element, which is the one a short copy would miss.
        assert t[tuple(s - 1 for s in shape)] == a.flat[-1]


@pytest.mark.parametrize(
    "array",
    [
        pytest.param(np.float32(1.0), id="rank-0"),
        pytest.param(np.ones((2, 2, 2, 2, 2), dtype=np.float32), id="rank-above-MAX_RANK"),
        pytest.param(np.ones((0, 3), dtype=np.float32), id="zero-extent"),
    ],
)
def test_from_numpy_rejects_bad_shapes(array: np.ndarray) -> None:
    with pytest.raises(ValueError):
        _core.from_numpy(array)


# --------------------------------------------------------------------------
# The buffer protocol - np.asarray(t) as a view
# --------------------------------------------------------------------------


def test_asarray_of_a_contiguous_tensor_shares_memory() -> None:
    t, a = base_2x3()
    view = np.asarray(t)

    assert view.dtype == np.float32
    assert view.shape == (2, 3)
    assert view.flags.c_contiguous
    assert np.array_equal(view, a)


def test_asarray_of_a_transposed_tensor_has_the_right_order() -> None:
    """The byte-vs-element stride test, and the reason for arange values.

    An implementation passing element strides, or ignoring strides and reading
    the buffer flat, produces the right multiset of values in the wrong order.
    Comparing sums, or testing on a square symmetric matrix, waves all of that
    through. 2x3 with distinct values makes it visible in the first element.
    """
    t, a = base_2x3()
    view = np.asarray(t.transpose(0, 1))

    assert view.shape == (3, 2)
    assert not view.flags.c_contiguous
    assert np.array_equal(view, a.T)


def test_asarray_strides_are_bytes() -> None:
    # Asserted on a TRANSPOSED tensor so the two strides differ from each other
    # per dimension: a contiguous rank-2 tensor would still pass this if one
    # dimension happened to be right by luck.
    t, _ = base_2x3()
    tr = t.transpose(0, 1)
    view = np.asarray(tr)

    assert tr.strides == (1, 3)
    assert view.strides == (4, 12)
    assert tr.strides == tuple(s // np.dtype(np.float32).itemsize for s in view.strides)


def test_asarray_of_a_sliced_tensor_starts_at_the_slice() -> None:
    """Proves data()'s folded-in offset is used, not the storage base."""
    a = np.arange(12, dtype=np.float32).reshape(4, 3)
    t = _core.from_numpy(a)
    s = t.slice(0, 1, 2)

    view = np.asarray(s)

    assert view.shape == (2, 3)
    assert view[0, 0] == a[1, 0]
    assert np.array_equal(view, a[1:3])
    # A slice keeps its parent's row stride, so the array is still C-contiguous
    # here; what matters is that it points at the right place.
    assert np.shares_memory(view, np.asarray(t))


def test_asarray_of_an_expanded_tensor_is_read_only() -> None:
    """The flag that carries a C++ invariant across the boundary.

    tensor.hpp says an expanded view must never be written through, and C++
    enforces that by requiring contiguity on every mutating path. NumPy has no
    such rule, so without readonly a single store corrupts four logical
    elements - and step 2's __setitem__ guard would disagree with numpy about
    whether the same buffer is writable.
    """
    base = _core.from_numpy(np.arange(3, dtype=np.float32).reshape(1, 3))
    view = np.asarray(base.expand([4, 3]))

    assert view.shape == (4, 3)
    assert view.strides[0] == 0
    assert not view.flags.writeable
    assert np.array_equal(view, np.broadcast_to(np.arange(3, dtype=np.float32), (4, 3)))

    with pytest.raises(ValueError):
        view[0, 0] = 1.0


def test_asarray_of_a_materialised_expand_is_writable_again() -> None:
    # readonly is derived from the tensor in hand, not sticky. Also proves
    # contiguous() materialised the repeats rather than copying the strides.
    base = _core.from_numpy(np.arange(3, dtype=np.float32).reshape(1, 3))
    dense = base.expand([4, 3]).contiguous()

    view = np.asarray(dense)

    assert view.strides == (12, 4)
    assert view.flags.writeable
    assert np.array_equal(view, np.broadcast_to(np.arange(3, dtype=np.float32), (4, 3)))


def test_writing_through_the_view_reaches_the_tensor() -> None:
    # Proves it is genuinely a view rather than a copy hiding behind a protocol.
    t, _ = base_2x3()
    np.asarray(t)[0, 0] = 5.0

    assert t[0, 0] == 5.0


def test_the_array_keeps_the_tensor_alive() -> None:
    """Py_buffer.obj holds a strong reference, so this needs no code to work.

    If it is ever broken - by hand-building a py::array_t over t.data() instead
    of exporting a buffer - this is a use-after-free that will often read the
    right bytes anyway. Run it under the debug tree.
    """
    t, a = base_2x3()
    view = np.asarray(t)

    # numpy 2.x sets .base to a memoryview, NOT to the Tensor. Assert that
    # something is holding the reference, never that it is `t`.
    assert view.base is not None

    del t
    gc.collect()

    assert np.array_equal(view, a)


def test_asarray_works_at_every_rank() -> None:
    for rank in range(1, _core.MAX_RANK + 1):
        shape = tuple(range(2, 2 + rank))
        a = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)

        assert np.array_equal(np.asarray(_core.from_numpy(a)), a)


# --------------------------------------------------------------------------
# to_numpy - the explicit copy
# --------------------------------------------------------------------------


def test_to_numpy_returns_an_owning_array() -> None:
    t, a = base_2x3()
    out = t.to_numpy()

    assert out.dtype == np.float32
    assert out.flags.c_contiguous
    assert out.base is None
    assert not np.shares_memory(out, np.asarray(t))
    assert np.array_equal(out, a)


def test_to_numpy_of_a_transposed_tensor_materialises_the_transpose() -> None:
    """The test that catches the shortcut.

    A to_numpy that memcpys numel() floats straight out of t.data() gives the
    wrong values here - it would return the buffer in its original order and
    label it (3, 2).
    """
    t, a = base_2x3()
    out = t.transpose(0, 1).to_numpy()

    assert out.shape == (3, 2)
    assert out.flags.c_contiguous
    assert np.array_equal(out, a.T)


def test_to_numpy_of_a_sliced_tensor_does_not_read_past_the_slice() -> None:
    # The other half of the same shortcut: a slice has fewer elements than the
    # buffer behind it, so copying from data() without going through
    # contiguous() reads whatever follows.
    a = np.arange(12, dtype=np.float32).reshape(4, 3)
    t = _core.from_numpy(a)

    out = t.slice(0, 2, 2).to_numpy()

    assert out.shape == (2, 3)
    assert np.array_equal(out, a[2:4])


def test_to_numpy_of_an_expanded_tensor_materialises_the_repeats() -> None:
    base = _core.from_numpy(np.arange(3, dtype=np.float32).reshape(1, 3))
    out = base.expand([4, 3]).to_numpy()

    assert out.shape == (4, 3)
    assert out.flags.writeable
    assert np.array_equal(out, np.broadcast_to(np.arange(3, dtype=np.float32), (4, 3)))


def test_to_numpy_does_not_alias_the_source() -> None:
    t, _ = base_2x3()
    out = t.to_numpy()

    out[0, 0] = 99.0
    assert t[0, 0] == 0.0


def test_round_trip_through_numpy_preserves_values() -> None:
    _, a = base_2x3()
    assert np.array_equal(_core.from_numpy(a).to_numpy(), a)
