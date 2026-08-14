"""_core.Tensor: construction, introspection, the six views, and indexing.

Written before the binding, so these fail on a fresh scaffold. Each one pins a
property that a plausible-looking wrong binding would break; where that is not
obvious, the test says which wrong binding it is aimed at.

The recurring theme is `shares_storage_with`. Almost every view test could be
written against shape alone, and almost every such test would pass against a
binding that quietly returned a clone. The sharing assertion is what makes it a
test of the view rather than a test of the arithmetic.
"""

from __future__ import annotations

import copy

import pytest

from autograd import _core


def test_max_rank_is_exposed() -> None:
    # Everything below assumes 4. Assert it rather than hardcoding it in a
    # comment that will not be updated.
    assert _core.MAX_RANK == 4


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_construction_sets_shape_and_row_major_strides() -> None:
    t = _core.Tensor([2, 3])

    assert t.shape == (2, 3)
    assert t.strides == (3, 1)
    assert t.rank() == 2
    assert t.numel() == 6
    assert t.is_contiguous()


def test_construction_accepts_any_sequence() -> None:
    # pybind11/stl.h converts any Python sequence to std::vector, so a tuple
    # has to work as well as a list. Worth pinning: the binding takes a vector
    # precisely because there is no std::span caster, and that conversion is
    # the load-bearing part.
    assert _core.Tensor((2, 3)).shape == _core.Tensor([2, 3]).shape


def test_rank_1_is_allowed() -> None:
    t = _core.Tensor([5])
    assert t.rank() == 1
    assert t.shape == (5,)
    assert t.strides == (1,)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param([], id="rank-0"),
        pytest.param([2, 2, 2, 2, 2], id="rank-above-MAX_RANK"),
        pytest.param([0, 3], id="zero-extent-first"),
        pytest.param([3, 0], id="zero-extent-last"),
    ],
)
def test_bad_shapes_raise_value_error(shape: list[int]) -> None:
    # ValueError specifically, not Exception. The loose form passes on a
    # TypeError from a signature mismatch and so proves nothing about whether
    # the C++ std::invalid_argument was translated at all.
    with pytest.raises(ValueError):
        _core.Tensor(shape)


def test_zeros_and_tensor_agree() -> None:
    assert _core.zeros([2, 3]).shape == _core.Tensor([2, 3]).shape


def test_zeros_like_takes_the_shape_not_the_strides() -> None:
    """The reason zeros_like exists.

    An out-parameter has to be contiguous. Handed a transposed tensor,
    zeros_like must produce a legal output buffer with that tensor's SHAPE and
    fresh row-major strides - not a copy of its strides, which would be
    non-contiguous and rejected by every op in step 4.
    """
    transposed = _core.Tensor([2, 3]).transpose(0, 1)
    assert transposed.strides == (1, 3)
    assert not transposed.is_contiguous()

    out = _core.zeros_like(transposed)

    assert out.shape == (3, 2)
    assert out.strides == (2, 1)
    assert out.is_contiguous()
    assert not out.shares_storage_with(transposed)


# --------------------------------------------------------------------------
# Views
# --------------------------------------------------------------------------


def test_transpose_swaps_shape_and_strides_and_shares_storage() -> None:
    t = _core.Tensor([2, 3])
    tr = t.transpose(0, 1)

    assert tr.shape == (3, 2)
    assert tr.strides == (1, 3)
    assert not tr.is_contiguous()
    # The assertion a shape-only test misses: proves a handle came back, not a
    # clone that happens to have the right extents.
    assert tr.shares_storage_with(t)


def test_transpose_rejects_a_bad_dimension() -> None:
    with pytest.raises(ValueError):
        _core.Tensor([2, 3]).transpose(0, 2)


def test_permute_reorders_extents_and_strides() -> None:
    t = _core.Tensor([2, 3, 4])
    p = t.permute([2, 0, 1])

    assert p.shape == (4, 2, 3)
    assert p.strides == (t.strides[2], t.strides[0], t.strides[1])
    assert p.shares_storage_with(t)


def test_permute_rejects_a_repeated_dim() -> None:
    # Right length, right rank, still not a permutation. A binding that only
    # checked the length would build a tensor with two dimensions aliasing one
    # axis and report nothing.
    with pytest.raises(ValueError):
        _core.Tensor([2, 3]).permute([0, 0])


def test_slice_moves_the_offset_and_leaves_strides_alone() -> None:
    """Slice is where a row stride wider than the row comes from.

    Shape alone would pass against a binding that returned a fresh contiguous
    copy of the rows. The strides assertion is what pins that only the offset
    moved - which is the layout the GEMM kernels' leading-dimension handling
    exists for, and what step 5's strongest test relies on.
    """
    t = _core.Tensor([4, 3])
    s = t.slice(0, 1, 2)

    assert s.shape == (2, 3)
    assert s.strides == (3, 1)
    assert s.shares_storage_with(t)


def test_slice_rejects_a_range_off_the_end() -> None:
    t = _core.Tensor([4, 3])
    with pytest.raises(ValueError):
        t.slice(0, 3, 2)


def test_slice_rejects_an_empty_range() -> None:
    with pytest.raises(ValueError):
        _core.Tensor([4, 3]).slice(0, 0, 0)


def test_expand_gives_stride_zero() -> None:
    t = _core.Tensor([1, 3])
    e = t.expand([4, 3])

    assert e.shape == (4, 3)
    # Stride 0 is the whole mechanism: every index on that axis addresses one
    # element. Step 3 turns this into numpy's read-only flag.
    assert e.strides == (0, 1)
    assert e.shares_storage_with(t)


def test_expand_rejects_a_dimension_wider_than_one() -> None:
    with pytest.raises(ValueError):
        _core.Tensor([2, 3]).expand([4, 3])


def test_reshape_gives_fresh_row_major_strides() -> None:
    t = _core.Tensor([2, 6])
    r = t.reshape([3, 4])

    assert r.shape == (3, 4)
    assert r.strides == (4, 1)
    assert r.shares_storage_with(t)


def test_reshape_rejects_a_non_contiguous_tensor() -> None:
    # Deliberately not a copy fallback. One function with two performance
    # profiles behind identical syntax is how a training loop gets mysteriously
    # slow.
    transposed = _core.Tensor([2, 6]).transpose(0, 1)
    with pytest.raises(ValueError):
        transposed.reshape([3, 4])


def test_reshape_rejects_a_different_element_count() -> None:
    with pytest.raises(ValueError):
        _core.Tensor([2, 6]).reshape([3, 5])


def test_contiguous_on_a_contiguous_tensor_is_a_handle_not_a_copy() -> None:
    """The fast path, which nothing else here reaches.

    Every kernel boundary calls contiguous(). If it copied unconditionally,
    every op would silently allocate, and the only symptom would be a
    performance one - the values would all be right.
    """
    t = _core.Tensor([2, 3])
    assert t.contiguous().shares_storage_with(t)


def test_contiguous_on_a_view_materialises_it() -> None:
    t = _core.Tensor([2, 3])
    tr = t.transpose(0, 1)
    dense = tr.contiguous()

    assert dense.is_contiguous()
    assert dense.shape == (3, 2)
    assert not dense.shares_storage_with(t)


def test_a_view_of_a_view_composes() -> None:
    t = _core.Tensor([4, 6])
    v = t.slice(0, 1, 2).slice(1, 2, 3).transpose(0, 1)

    assert v.shape == (3, 2)
    assert v.strides == (1, 6)
    assert v.shares_storage_with(t)


def test_a_view_outlives_its_base() -> None:
    """The ownership claim, tested rather than asserted in a comment.

    The returned Tensor carries its own shared_ptr<Storage>, so dropping every
    Python reference to the base must not free the buffer. If this is wrong it
    is a use-after-free, which may well still pass - run it under the debug
    tree, where assert_within_storage is live.
    """
    import gc

    t = _core.Tensor([4, 6])
    t[0, 0] = 7.0
    v = t.slice(0, 0, 2)
    del t
    gc.collect()

    assert v[0, 0] == 7.0


# --------------------------------------------------------------------------
# Indexing
#
# C++ operator() checks arity only, and only in debug. Everything below has to
# hold in the Release build, which is the one that ships.
# --------------------------------------------------------------------------


def test_setitem_then_getitem_round_trips() -> None:
    t = _core.Tensor([2, 3])
    t[1, 2] = 4.5
    assert t[1, 2] == 4.5
    assert t[0, 0] == 0.0  # construction zeroes


def test_indexing_reads_through_a_views_strides() -> None:
    t = _core.Tensor([2, 3])
    t[0, 1] = 9.0

    assert t.transpose(0, 1)[1, 0] == 9.0


def test_negative_indices_count_from_the_end() -> None:
    # Normalised, then bounds-checked. Rejecting them would be merely annoying;
    # letting -1 become SIZE_MAX is a wild read.
    t = _core.Tensor([2, 3])
    t[1, 2] = 3.0

    assert t[-1, -1] == 3.0


@pytest.mark.parametrize(
    "index",
    [
        pytest.param((2, 0), id="row-out-of-range"),
        pytest.param((0, 3), id="column-out-of-range"),
        pytest.param((-3, 0), id="negative-out-of-range"),
    ],
)
def test_out_of_range_indices_raise_index_error(index: tuple[int, ...]) -> None:
    with pytest.raises(IndexError):
        _core.Tensor([2, 3])[index]


@pytest.mark.parametrize(
    "index",
    [
        pytest.param(0, id="too-few"),
        pytest.param((0, 0, 0), id="too-many"),
    ],
)
def test_wrong_arity_raises_index_error(index: object) -> None:
    # Partial indexing would have to be a rank-reducing view, and the minimum
    # rank here is 1. slice() is the view API.
    with pytest.raises(IndexError):
        _core.Tensor([2, 3])[index]


def test_setitem_is_bounds_checked_too() -> None:
    # The one that actually corrupts memory if it is missing.
    with pytest.raises(IndexError):
        _core.Tensor([2, 3])[5, 5] = 1.0


def test_setitem_refuses_to_write_through_an_expanded_view() -> None:
    """One store must not land on several logical elements.

    Reading through stride 0 is fine and stays allowed - that is what expand is
    for. Writing is the operation that has no sensible meaning, and this is the
    only door C++ does not already close by requiring contiguity.

    Step 3 reports the same condition to numpy as readonly, so both routes to
    the buffer agree about whether it can be written.
    """
    e = _core.Tensor([1, 3]).expand([4, 3])
    assert e.strides[0] == 0

    assert e[0, 0] == 0.0  # reads still work

    with pytest.raises(ValueError):
        e[0, 0] = 5.0


def test_setitem_still_works_through_a_transposed_view() -> None:
    # The check is deliberately narrower than "must be contiguous". A transpose
    # has no aliasing, so writing through one is well defined and stays legal.
    t = _core.Tensor([2, 3])
    tr = t.transpose(0, 1)
    assert not tr.is_contiguous()

    tr[2, 1] = 8.0
    assert t[1, 2] == 8.0


def test_rank_1_indexing_takes_a_bare_int() -> None:
    t = _core.Tensor([3])
    t[1] = 2.0
    assert t[1] == 2.0


# --------------------------------------------------------------------------
# Copy semantics
# --------------------------------------------------------------------------


def test_copy_is_shallow_and_deepcopy_is_not() -> None:
    """The most surprising fact about this type, made discoverable.

    In C++ the compiler stopped catching accidental copies the moment Tensor
    became copyable; in Python there was never a compiler. Both halves are
    bound so the answer is reachable from the REPL.
    """
    t = _core.Tensor([2, 3])
    t[0, 0] = 1.0

    shallow = copy.copy(t)
    deep = copy.deepcopy(t)

    assert shallow.shares_storage_with(t)
    assert not deep.shares_storage_with(t)

    # And the aliasing is real, not just reported.
    shallow[0, 0] = 2.0
    assert t[0, 0] == 2.0
    assert deep[0, 0] == 1.0


def test_repr_names_the_layout() -> None:
    r = repr(_core.Tensor([2, 3]).transpose(0, 1))

    assert "Tensor" in r
    # Shape and strides both, because a repr showing only the shape cannot
    # distinguish a view from a fresh tensor - which is the thing you are
    # squinting at a repr to find out.
    assert "3" in r and "2" in r
    assert "stride" in r.lower()
