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
