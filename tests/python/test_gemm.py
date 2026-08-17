"""The GEMM binding: C += A @ B over Tensor operands.

Written before the binding, so these fail on a fresh scaffold. Replaces
test_gemm_numpy.py, which drives the phase 2 throwaway; that file goes at
step 6 once this one is green.

Two properties are carried over from the C++ harness deliberately, because
losing either would make the suite look thorough while proving less:

  - every shape runs against NumPy, not against another kernel of ours. Two of
    our own implementations agreeing is weaker evidence than agreeing with a
    third party nobody here wrote.
  - every case runs TWICE, with C zeroed and with C prefilled. The contract is
    `C += A @ B`; zeroing C before every call makes `=` and `+=`
    indistinguishable, and phase 6 is built entirely on `+=`.
"""

from __future__ import annotations

import numpy as np
import pytest

from autograd import _core

KERNELS = ["naive", "ikj", "tiled", "avx2", "tiled_simd", "multithreaded"]

# The same table the C++ harness uses: each dimension independently crosses the
# vector width (8), the register block (4, 16) and the tile (64). Degenerate
# and prime shapes first - tile-boundary bugs hide perfectly in square aligned
# cases.
SHAPES = [
    (1, 1, 1),
    (1, 512, 1),
    (64, 96, 1),
    (3, 5, 7),
    (5, 17, 9),
    (8, 8, 8),
    (17, 31, 13),
    (63, 65, 64),
    (65, 65, 65),
    (129, 130, 131),
    (128, 256, 512),
    (256, 256, 256),
]


def operands(M: int, N: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    # [0.5, 1.5) rather than [0, 1), matching the C++ harness: no expected value
    # drifts near zero, so a region no code path wrote stays visibly wrong.
    rng = np.random.default_rng(0xC0FFEE ^ (M * 73856093) ^ (N * 19349663) ^ (K * 83492791))
    a = (rng.random((M, K), dtype=np.float32) + 0.5).astype(np.float32)
    b = (rng.random((K, N), dtype=np.float32) + 0.5).astype(np.float32)
    return a, b


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("prefill", [0.0, 3.25], ids=["zeroed", "prefilled"])
def test_matches_numpy(kernel: str, shape: tuple[int, int, int], prefill: float) -> None:
    M, N, K = shape
    a, b = operands(M, N, K)

    A = _core.from_numpy(a)
    B = _core.from_numpy(b)
    C = _core.from_numpy(np.full((M, N), prefill, dtype=np.float32))

    _core.gemm(A, B, C, kernel)

    # prefill + a @ b, not a @ b. This is what makes the parametrisation worth
    # having rather than doubling the runtime for nothing.
    expected = prefill + a @ b
    assert np.allclose(np.asarray(C), expected, rtol=1e-5, atol=1e-5)


def test_gemm_returns_none_and_writes_in_place() -> None:
    a, b = operands(8, 8, 8)
    A, B = _core.from_numpy(a), _core.from_numpy(b)
    C = _core.zeros([8, 8])

    assert _core.gemm(A, B, C) is None
    assert np.allclose(np.asarray(C), a @ b, rtol=1e-5, atol=1e-5)


def test_gemm_accumulates_across_two_calls() -> None:
    # The phase 6 shape, stated directly rather than inferred from the
    # prefilled parametrisation.
    a, b = operands(8, 8, 8)
    A, B = _core.from_numpy(a), _core.from_numpy(b)
    C = _core.zeros([8, 8])

    _core.gemm(A, B, C)
    _core.gemm(A, B, C)

    assert np.allclose(np.asarray(C), 2.0 * (a @ b), rtol=1e-5, atol=1e-5)


def test_gemm_accepts_a_sliced_operand() -> None:
    """The strongest single test in this file.

    slice(1, 0, 8) on an 8x16 gives an 8x8 whose row stride is 16 and whose
    inner stride is 1 - a leading dimension wider than the row, which is
    exactly what gemm_check_shapes accepts and what the kernels' lda/ldb/ldc
    handling exists for. Phase 3 made this case live rather than theoretical.

    A binding that "helpfully" called .contiguous() on its inputs would pass
    every other test in this file while silently deleting what this one
    measures - the values would all still be correct.
    """
    wide = (np.random.default_rng(7).random((8, 16), dtype=np.float32) + 0.5).astype(np.float32)
    b = (np.random.default_rng(8).random((8, 8), dtype=np.float32) + 0.5).astype(np.float32)

    A = _core.from_numpy(wide).slice(1, 0, 8)
    assert A.shape == (8, 8)
    assert A.strides == (16, 1)  # row stride exceeds the row width
    assert not A.is_contiguous()

    C = _core.zeros([8, 8])
    _core.gemm(A, _core.from_numpy(b), C)

    assert np.allclose(np.asarray(C), wide[:, :8] @ b, rtol=1e-5, atol=1e-5)


def test_gemm_accepts_a_sliced_output() -> None:
    # The same leading-dimension case on C, which is the one an ldc/lda mix-up
    # breaks. Rows outside the slice must be untouched.
    a, b = operands(4, 4, 4)
    wide = _core.from_numpy(np.zeros((4, 8), dtype=np.float32))
    C = wide.slice(1, 0, 4)
    assert C.strides == (8, 1)

    _core.gemm(_core.from_numpy(a), _core.from_numpy(b), C)

    full = np.asarray(wide)
    assert np.allclose(full[:, :4], a @ b, rtol=1e-5, atol=1e-5)
    # Nothing spilled past the slice.
    assert np.array_equal(full[:, 4:], np.zeros((4, 4), dtype=np.float32))


# --------------------------------------------------------------------------
# The conversions that are the point of this step
# --------------------------------------------------------------------------


def test_a_transposed_operand_raises_rather_than_silently_doing_nothing() -> None:
    """The silent no-op, converted.

    A transposed tensor has stride(1) != 1, which gemm_check_shapes rejects.
    Before this step that call RETURNED NORMALLY and left C at its prefill, so
    a caller got a wrong answer with no signal whatsoever.

    match= is what makes this a test of the conversion rather than a test that
    something, somewhere, went wrong: pytest.raises(Exception) here would also
    pass on a TypeError from a signature mismatch.
    """
    a, b = operands(8, 8, 8)
    A_t = _core.from_numpy(a).transpose(0, 1)
    C = _core.zeros([8, 8])

    with pytest.raises(ValueError, match="unit inner stride"):
        _core.gemm(A_t, _core.from_numpy(b), C)


def test_a_transposed_output_raises() -> None:
    a, b = operands(8, 8, 8)
    C_t = _core.zeros([8, 8]).transpose(0, 1)

    with pytest.raises(ValueError, match="unit inner stride"):
        _core.gemm(_core.from_numpy(a), _core.from_numpy(b), C_t)


def test_rejecting_gemm_leaves_the_output_untouched() -> None:
    # The property the C++ harness's magnitude guard depends on, asserted from
    # Python: a rejected call must not have half-written C before noticing.
    C = _core.from_numpy(np.full((8, 8), 3.25, dtype=np.float32))
    a, b = operands(8, 8, 8)

    with pytest.raises(ValueError):
        _core.gemm(_core.from_numpy(a).transpose(0, 1), _core.from_numpy(b), C)

    assert np.array_equal(np.asarray(C), np.full((8, 8), 3.25, dtype=np.float32))


def test_a_rank_3_operand_raises() -> None:
    A = _core.zeros([2, 2, 2])
    B = _core.zeros([2, 2])
    C = _core.zeros([2, 2])

    with pytest.raises(ValueError, match="rank 2"):
        _core.gemm(A, B, C)


def test_mismatched_inner_dimensions_raise() -> None:
    A = _core.zeros([4, 8])
    B = _core.zeros([7, 4])  # 8 != 7
    C = _core.zeros([4, 4])

    with pytest.raises(ValueError, match="shape mismatch"):
        _core.gemm(A, B, C)


def test_a_wrong_output_shape_raises() -> None:
    A = _core.zeros([4, 8])
    B = _core.zeros([8, 6])
    C = _core.zeros([4, 4])  # should be 4x6

    with pytest.raises(ValueError, match="shape mismatch"):
        _core.gemm(A, B, C)


@pytest.mark.parametrize("alias", ["A", "B"])
def test_an_output_aliasing_an_input_raises(alias: str) -> None:
    """restrict makes overlap undefined, not slow.

    From C++ this is a documented precondition. From Python it is a segfault
    produced by mistyping a variable name, so the binding refuses it.
    """
    square = _core.from_numpy(operands(8, 8, 8)[0])
    other = _core.zeros([8, 8])

    A, B = (square, other) if alias == "A" else (other, square)

    with pytest.raises(ValueError, match="share storage"):
        _core.gemm(A, B, square)


def test_an_output_sharing_a_buffer_at_a_disjoint_offset_also_raises() -> None:
    # Documented over-strictness: this case is genuinely safe, and is rejected
    # anyway because the check compares storage identity rather than address
    # ranges. Pinned so the behaviour is a decision, not a surprise.
    buf = _core.zeros([4, 16])
    A = buf.slice(1, 0, 4)
    C = buf.slice(1, 8, 4)  # no overlap with A at all

    with pytest.raises(ValueError, match="share storage"):
        _core.gemm(A, _core.zeros([4, 4]), C)


def test_an_unknown_kernel_raises() -> None:
    A, B, C = _core.zeros([4, 4]), _core.zeros([4, 4]), _core.zeros([4, 4])

    with pytest.raises(ValueError, match="unknown kernel"):
        _core.gemm(A, B, C, "definitely_not_a_kernel")


def test_the_kernel_name_is_validated_before_the_operands_are_used() -> None:
    # Ordering: an unknown name must be reported even when the operands are
    # fine, and a bad operand must be reported even when the name is fine.
    # Neither check may swallow the other.
    good = _core.zeros([4, 4])
    with pytest.raises(ValueError, match="unknown kernel"):
        _core.gemm(good, good.clone(), good.clone(), "nope")


def test_num_threads_is_accepted_and_changes_nothing_numerically() -> None:
    # The row partitioning is a load-balance decision, not a numerical one, so
    # every thread count must give the same answer to tolerance.
    a, b = operands(600, 600, 600)  # above ~512, where threading actually pays
    A, B = _core.from_numpy(a), _core.from_numpy(b)
    expected = a @ b

    for n in (1, 2, 8):
        C = _core.zeros([600, 600])
        _core.gemm(A, B, C, "multithreaded", n)
        assert np.allclose(np.asarray(C), expected, rtol=1e-5, atol=1e-5)


def test_a_non_positive_thread_count_raises() -> None:
    A, B, C = _core.zeros([4, 4]), _core.zeros([4, 4]), _core.zeros([4, 4])

    with pytest.raises(ValueError):
        _core.gemm(A, B, C, "multithreaded", 0)
