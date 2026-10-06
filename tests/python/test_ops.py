"""The elementwise op bindings.

Written before the binding, so these fail on a fresh scaffold.

The C++ [ops] tests already prove these kernels handle strides. Repeating them
here proves something different: that the BINDING did not add anything. The
failure mode worth guarding against is a well-meaning .contiguous() on the way
in - it would pass every contiguous test, give correct values everywhere, and
silently delete the stride-general behaviour phase 3 exists for.

So the inputs here are transposed and expanded, and the comparisons are
elementwise. A sum or a checksum passes on a wrong-order result.
"""

from __future__ import annotations

import numpy as np
import pytest

from autograd import _core


def tensor(a: np.ndarray) -> _core.Tensor:
    return _core.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


# --------------------------------------------------------------------------
# add
# --------------------------------------------------------------------------


def test_add_on_contiguous_operands() -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.arange(6, dtype=np.float32).reshape(2, 3) * 10

    ta, tb = tensor(a), tensor(b)
    out = _core.zeros_like(ta)
    _core.add(ta, tb, out)

    assert np.array_equal(np.asarray(out), a + b)


def test_add_reads_a_transposed_input_in_place() -> None:
    """The test that catches a binding calling .contiguous() on its inputs.

    Such a binding would pass every other test in this file. Here it would also
    pass, values-wise - which is the point: this test is about the transposed
    operand being READ through its strides, and the only way to see the
    difference is that the result must equal a.T elementwise, in order.
    """
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.arange(6, dtype=np.float32).reshape(3, 2) * 10

    ta_t = tensor(a).transpose(0, 1)
    assert not ta_t.is_contiguous()

    out = _core.zeros_like(ta_t)
    _core.add(ta_t, tensor(b), out)

    assert np.array_equal(np.asarray(out), a.T + b)


def test_add_reads_an_expanded_input_in_place() -> None:
    # Broadcasting a bias row, which is what phase 8's Linear layer does. A
    # sum-based check would pass on a wrong-order result, so compare elementwise.
    a = np.arange(12, dtype=np.float32).reshape(4, 3)
    bias = np.array([[100.0, 200.0, 300.0]], dtype=np.float32)

    expanded = tensor(bias).expand([4, 3])
    assert expanded.strides[0] == 0

    out = _core.zeros_like(tensor(a))
    _core.add(tensor(a), expanded, out)

    assert np.array_equal(np.asarray(out), a + bias)


def test_add_overwrites_rather_than_accumulates() -> None:
    # add is not add_into. A prefilled out must be replaced, not added to.
    a = np.ones((2, 3), dtype=np.float32)
    out = tensor(np.full((2, 3), 7.0, dtype=np.float32))

    _core.add(tensor(a), tensor(a), out)

    assert np.array_equal(np.asarray(out), np.full((2, 3), 2.0, dtype=np.float32))


def test_add_rejects_a_non_contiguous_output() -> None:
    a = tensor(np.ones((2, 3), dtype=np.float32))
    bad_out = _core.zeros([3, 2]).transpose(0, 1)
    assert not bad_out.is_contiguous()

    with pytest.raises(ValueError):
        _core.add(a, a, bad_out)


def test_add_rejects_mismatched_shapes() -> None:
    # Pins "no implicit broadcasting" as a Python-visible contract, not just a
    # C++ one. A caller who wants it writes .expand() where it can be seen.
    a = tensor(np.ones((2, 3), dtype=np.float32))
    b = tensor(np.ones((1, 3), dtype=np.float32))
    out = _core.zeros([2, 3])

    with pytest.raises(ValueError):
        _core.add(a, b, out)


def test_add_rejects_a_mismatched_output_shape() -> None:
    a = tensor(np.ones((2, 3), dtype=np.float32))
    out = _core.zeros([3, 2])

    with pytest.raises(ValueError):
        _core.add(a, a, out)


# --------------------------------------------------------------------------
# mul, scale, relu
# --------------------------------------------------------------------------


def test_mul_is_elementwise_not_a_matrix_product() -> None:
    # A 2x2 is the one shape where the two could be confused, so use it.
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    out = _core.zeros([2, 2])
    _core.mul(tensor(a), tensor(a), out)

    assert np.array_equal(np.asarray(out), a * a)
    assert not np.array_equal(np.asarray(out), a @ a)


def test_mul_reads_a_transposed_input() -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.arange(6, dtype=np.float32).reshape(3, 2) + 1

    ta_t = tensor(a).transpose(0, 1)
    out = _core.zeros_like(ta_t)
    _core.mul(ta_t, tensor(b), out)

    assert np.array_equal(np.asarray(out), a.T * b)


def test_scale_multiplies_by_a_python_float() -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)

    out = _core.zeros([2, 3])
    _core.scale(tensor(a), 2.5, out)

    assert np.array_equal(np.asarray(out), a * 2.5)


def test_scale_reads_a_transposed_input() -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)

    ta_t = tensor(a).transpose(0, 1)
    out = _core.zeros_like(ta_t)
    _core.scale(ta_t, -1.0, out)

    assert np.array_equal(np.asarray(out), -a.T)


def test_relu_clamps_at_zero() -> None:
    # Straddle zero, and include an exact zero: relu(0) must be 0, not -0.0 or
    # a NaN from a stray comparison.
    a = np.array([[-2.0, -0.5, 0.0], [0.5, 2.0, -1e30]], dtype=np.float32)

    out = _core.zeros([2, 3])
    _core.relu(tensor(a), out)

    assert np.array_equal(np.asarray(out), np.maximum(a, 0.0))


def test_relu_reads_an_expanded_input() -> None:
    row = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    expanded = tensor(row).expand([4, 3])

    out = _core.zeros([4, 3])
    _core.relu(expanded, out)

    assert np.array_equal(np.asarray(out), np.maximum(np.broadcast_to(row, (4, 3)), 0.0))


# --------------------------------------------------------------------------
# relu_backward - the mask
# --------------------------------------------------------------------------


def test_relu_backward_masks_on_the_reference_sign() -> None:
    grad = np.array([5.0, 7.0, -4.0, 0.0], dtype=np.float32)
    ref = np.array([-2.0, 0.0, 3.0, 1.0], dtype=np.float32)

    out = _core.zeros([4])
    _core.relu_backward(tensor(grad), tensor(ref), out)

    # Exact, not allclose - these are representable. np.where gives the same
    # subgradient choice at ref == 0 as the kernel's strict >.
    assert np.array_equal(np.asarray(out), np.where(ref > 0, grad, 0.0))


def test_relu_backward_reads_a_transposed_gradient_in_place() -> None:
    """The stride path, exercised rather than merely assumed.

    The result must equal grad.T elementwise, in order - a flat walk of the
    transposed buffer holds the same six numbers rearranged, so a sum or a
    checksum would wave that through.

    What this canNOT catch is a binding that calls .contiguous() on its inputs:
    that is clone(), which reads through the strides and preserves the logical
    values, so it is a slow no-op rather than a wrong answer. Measured - the
    mutation leaves the whole suite green. What the case buys is that the
    strided read really runs, so the debug tree's assert_within_storage and
    ASan have something to fire on. The contiguity trap that IS detectable is
    on the destination; see test_sum_into_rejects_a_non_contiguous_destination.
    """
    grad = np.arange(6, dtype=np.float32).reshape(2, 3)
    ref = np.ones((3, 2), dtype=np.float32)

    grad_t = tensor(grad).transpose(0, 1)
    assert not grad_t.is_contiguous()

    out = _core.zeros([3, 2])
    _core.relu_backward(grad_t, tensor(ref), out)

    assert np.array_equal(np.asarray(out), grad.T)


def test_relu_backward_reads_a_transposed_reference_in_place() -> None:
    grad = np.full((3, 2), 9.0, dtype=np.float32)
    ref = np.arange(-3, 3, dtype=np.float32).reshape(2, 3)

    out = _core.zeros([3, 2])
    _core.relu_backward(tensor(grad), tensor(ref).transpose(0, 1), out)

    assert np.array_equal(np.asarray(out), np.where(ref.T > 0, 9.0, 0.0))


def test_relu_backward_rejects_mismatched_shapes() -> None:
    a = _core.zeros([2, 3])
    with pytest.raises(ValueError, match="shape mismatch"):
        _core.relu_backward(a, _core.zeros([3, 2]), a)


def test_relu_backward_rejects_a_non_contiguous_output() -> None:
    a = _core.zeros([2, 3])
    tr_out = _core.zeros([3, 2]).transpose(0, 1)
    assert not tr_out.is_contiguous()

    with pytest.raises(ValueError, match="output must be contiguous"):
        _core.relu_backward(a, a, tr_out)


# --------------------------------------------------------------------------
# sum_into - the broadcast collapse
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dst_shape, axis",
    [
        pytest.param([1, 3], 0, id="down-the-rows"),
        pytest.param([2, 1], 1, id="across-the-columns"),
    ],
)
def test_sum_into_collapses_one_axis(dst_shape: list[int], axis: int) -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)

    dst = _core.zeros(dst_shape)
    _core.sum_into(dst, tensor(a))

    assert np.array_equal(np.asarray(dst), a.sum(axis=axis, keepdims=True))


def test_sum_into_collapses_every_axis_to_a_tensor() -> None:
    """The difference from _core.sum, which returns a Python float.

    A graph node's forward has to produce a _core.Tensor, so a full reduction
    needs this rather than the scalar form.
    """
    a = np.arange(6, dtype=np.float32).reshape(2, 3)

    dst = _core.zeros([1, 1])
    _core.sum_into(dst, tensor(a))

    assert np.asarray(dst).shape == (1, 1)
    assert np.asarray(dst)[0, 0] == pytest.approx(a.sum())


def test_sum_into_accumulates_into_a_prefilled_destination() -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    dst = tensor(np.full((1, 3), 100.0, dtype=np.float32))

    _core.sum_into(dst, tensor(a))

    # Overwriting instead of accumulating gives [3, 5, 7] - the right shape and
    # a plausible magnitude, and wrong.
    assert np.array_equal(np.asarray(dst), 100.0 + a.sum(axis=0, keepdims=True))


def test_sum_into_reads_a_transposed_source() -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)

    dst = _core.zeros([1, 2])
    _core.sum_into(dst, tensor(a).transpose(0, 1))

    # Column sums of a.T. A flat walk of the transposed buffer pairs the
    # elements wrongly and returns [6, 9] - the same total, split wrong.
    assert np.array_equal(np.asarray(dst), a.T.sum(axis=0, keepdims=True))


def test_sum_into_reads_an_expanded_source() -> None:
    """The round trip: expand out with stride 0, collapse straight back.

    Nothing materialises a 4x3 buffer in either direction, which is the whole
    reason the bias gradient does not allocate per step.
    """
    row = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)

    dst = _core.zeros([1, 3])
    _core.sum_into(dst, tensor(row).expand([4, 3]))

    assert np.array_equal(np.asarray(dst), 4.0 * row)


def test_sum_into_rejects_a_rank_change() -> None:
    with pytest.raises(ValueError, match="rank mismatch"):
        _core.sum_into(_core.zeros([6]), _core.zeros([2, 3]))


def test_sum_into_rejects_an_extent_that_is_neither_matching_nor_one() -> None:
    with pytest.raises(ValueError, match="incompatible shape"):
        _core.sum_into(_core.zeros([2, 2]), _core.zeros([2, 3]))


def test_sum_into_rejects_a_destination_larger_than_the_source() -> None:
    with pytest.raises(ValueError, match="incompatible shape"):
        _core.sum_into(_core.zeros([4, 3]), _core.zeros([2, 3]))


def test_sum_into_rejects_a_non_contiguous_destination() -> None:
    """The one contiguity trap in this layer that a test can actually catch.

    A binding contiguifying its destination would accumulate into a temporary
    copy, drop it, and return None - leaving the caller's buffer untouched with
    no error anywhere. Contiguifying an INPUT is undetectable by value, since
    it preserves what the strides mean; contiguifying an OUTPUT throws the
    write away. Measured: the mutation fails exactly this test and nothing else.
    """
    tr_dst = _core.zeros([3, 2]).transpose(0, 1)
    assert not tr_dst.is_contiguous()

    with pytest.raises(ValueError, match="output must be contiguous"):
        _core.sum_into(tr_dst, _core.zeros([2, 3]))


# --------------------------------------------------------------------------
# add_into - the accumulating primitive
# --------------------------------------------------------------------------


def test_add_into_accumulates_into_a_prefilled_destination() -> None:
    """Mirrors the C++ harness's 'run every case twice' property.

    A test with a zeroed dst cannot distinguish `dst += src` from `dst = src`,
    and phase 6's whole gradient accumulation is built on it being `+=`.
    """
    before = np.arange(6, dtype=np.float32).reshape(2, 3) + 100.0
    src = np.arange(6, dtype=np.float32).reshape(2, 3)

    dst = tensor(before)
    _core.add_into(dst, tensor(src))

    assert np.array_equal(np.asarray(dst), before + src)
    # Explicitly not the overwrite result.
    assert not np.array_equal(np.asarray(dst), src)


def test_add_into_twice_accumulates_twice() -> None:
    # The actual phase 6 shape: a tensor used twice, two contributions.
    src = np.ones((2, 3), dtype=np.float32)
    dst = _core.zeros([2, 3])

    _core.add_into(dst, tensor(src))
    _core.add_into(dst, tensor(src))

    assert np.array_equal(np.asarray(dst), src * 2)


def test_add_into_reads_an_expanded_source() -> None:
    row = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    dst = tensor(np.zeros((4, 3), dtype=np.float32))

    _core.add_into(dst, tensor(row).expand([4, 3]))

    assert np.array_equal(np.asarray(dst), np.broadcast_to(row, (4, 3)))


def test_add_into_rejects_a_non_contiguous_destination() -> None:
    dst = _core.zeros([3, 2]).transpose(0, 1)
    src = tensor(np.ones((2, 3), dtype=np.float32))

    with pytest.raises(ValueError):
        _core.add_into(dst, src)


# --------------------------------------------------------------------------
# sum
# --------------------------------------------------------------------------


def test_sum_adds_every_element() -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert _core.sum(tensor(a)) == pytest.approx(float(a.sum()))


def test_sum_of_an_expanded_view_counts_the_repeats() -> None:
    """Stride 0 is read, not skipped.

    An expanded view of a 1x3 to 4x3 logically holds twelve elements, and the
    arithmetically correct sum counts each repeat four times. A contiguous-only
    sum test never touches this.
    """
    row = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    base = tensor(row)

    assert _core.sum(base.expand([4, 3])) == pytest.approx(4.0 * _core.sum(base))
    assert _core.sum(base.expand([4, 3])) == pytest.approx(24.0)


def test_sum_of_a_transposed_view_matches() -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert _core.sum(tensor(a).transpose(0, 1)) == pytest.approx(float(a.sum()))


def test_sum_accumulates_in_double() -> None:
    """The measured reason sum does not accumulate in float32.

    Naive left-to-right float32 accumulation over 100000 elements drifts about
    1.4e-4 relative from the true value - fourteen times the 1e-5 tolerance used
    everywhere else in this project. This is the test that fails if the double
    accumulator is ever dropped for a float one.
    """
    n = 100_000
    a = np.full(n, 0.1, dtype=np.float32)

    got = _core.sum(tensor(a))
    expected = float(np.asarray(a, dtype=np.float64).sum())

    assert got == pytest.approx(expected, rel=1e-6)

    # And confirm the naive float32 walk really would miss by more than that,
    # so this test is not passing for want of precision to lose.
    naive = np.float32(0.0)
    for value in a:
        naive = np.float32(naive + value)
    assert abs(float(naive) - expected) / expected > 1e-5


def test_sum_of_a_single_element() -> None:
    assert _core.sum(_core.zeros([1])) == 0.0


# --------------------------------------------------------------------------
# exp, log
# --------------------------------------------------------------------------


def test_exp_and_log_match_numpy() -> None:
    a = np.array([[0.0, 1.0, -1.0], [2.0, -3.0, 0.5]], dtype=np.float32)

    e = _core.zeros([2, 3])
    _core.exp(tensor(a), e)
    assert np.allclose(np.asarray(e), np.exp(a), rtol=1e-6, atol=0.0)

    back = _core.zeros([2, 3])
    _core.log(e, back)
    assert np.allclose(np.asarray(back), a, rtol=1e-6, atol=1e-6)


def test_exp_reads_a_transposed_input() -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)

    out = _core.zeros([3, 2])
    _core.exp(tensor(a).transpose(0, 1), out)

    # Elementwise, so a flat walk shows up as the right values in the wrong order.
    assert np.allclose(np.asarray(out), np.exp(a.T), rtol=1e-6, atol=0.0)


def test_log_reads_an_expanded_input() -> None:
    row = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
    expanded = tensor(row).expand([2, 3])
    assert expanded.strides[0] == 0

    out = _core.zeros([2, 3])
    _core.log(expanded, out)

    assert np.allclose(np.asarray(out), np.log(np.broadcast_to(row, (2, 3))), rtol=1e-6)


@pytest.mark.parametrize("op", ["exp", "log"])
def test_exp_and_log_reject_a_non_contiguous_output(op: str) -> None:
    tr_out = _core.zeros([3, 2]).transpose(0, 1)

    with pytest.raises(ValueError, match="output must be contiguous"):
        getattr(_core, op)(_core.zeros([2, 3]), tr_out)


@pytest.mark.parametrize("op", ["exp", "log"])
def test_exp_and_log_reject_a_mismatched_output_shape(op: str) -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        getattr(_core, op)(_core.zeros([2, 3]), _core.zeros([3, 2]))


# --------------------------------------------------------------------------
# reduce_max - sum_into's shape rules, the opposite write contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "out_shape, axis",
    [([1, 3], 0), ([2, 1], 1)],
    ids=["down-the-rows", "across-the-columns"],
)
def test_reduce_max_collapses_one_axis(out_shape: list[int], axis: int) -> None:
    a = np.array([[1.0, 7.0, 3.0], [9.0, 2.0, 4.0]], dtype=np.float32)

    out = _core.zeros(out_shape)
    _core.reduce_max(tensor(a), out)

    assert np.array_equal(np.asarray(out), a.max(axis=axis, keepdims=True))


def test_reduce_max_of_an_all_negative_row_is_negative() -> None:
    """Seeding the output with zero instead of -inf returns 0 here, and passes
    every other test in this section."""
    a = np.array([[-5.0, -2.0, -9.0], [-1.0, -4.0, -3.0]], dtype=np.float32)

    out = _core.zeros([2, 1])
    _core.reduce_max(tensor(a), out)

    assert np.array_equal(np.asarray(out), [[-2.0], [-1.0]])


def test_reduce_max_overwrites_a_prefilled_output() -> None:
    """The opposite of sum_into, which accumulates."""
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = tensor(np.full((2, 1), 100.0, dtype=np.float32))

    _core.reduce_max(tensor(a), out)

    assert np.array_equal(np.asarray(out), [[2.0], [5.0]])


def test_reduce_max_reads_a_transposed_input() -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)

    out = _core.zeros([3, 1])
    _core.reduce_max(tensor(a).transpose(0, 1), out)

    # A flat walk pairs the buffer as {0 1} {2 3} {4 5} and returns [1, 3, 5].
    assert np.array_equal(np.asarray(out), a.T.max(axis=1, keepdims=True))


def test_reduce_max_reads_an_expanded_input() -> None:
    row = np.array([[1.0, 8.0, 4.0]], dtype=np.float32)

    out = _core.zeros([4, 1])
    _core.reduce_max(tensor(row).expand([4, 3]), out)

    assert np.array_equal(np.asarray(out), np.full((4, 1), 8.0))


def test_reduce_max_rejects_a_rank_change() -> None:
    with pytest.raises(ValueError, match="rank mismatch"):
        _core.reduce_max(_core.zeros([2, 3]), _core.zeros([2]))


def test_reduce_max_rejects_an_extent_that_is_neither_matching_nor_one() -> None:
    with pytest.raises(ValueError, match="incompatible shape"):
        _core.reduce_max(_core.zeros([2, 3]), _core.zeros([2, 2]))


def test_reduce_max_rejects_a_non_contiguous_output() -> None:
    tr_out = _core.zeros([3, 2]).transpose(0, 1)

    with pytest.raises(ValueError, match="output must be contiguous"):
        _core.reduce_max(_core.zeros([2, 3]), tr_out)
