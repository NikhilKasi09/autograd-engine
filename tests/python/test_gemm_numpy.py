"""Cross-check the kernels against NumPy.

The C++ harness compares against its own reference implementation, which is
strong but self-contained: both were written by the same person on the same
assumptions. NumPy is the independent third opinion.

These fail until the bindings in bindings/bindings.cpp are implemented.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import _core  # noqa: E402

KERNELS = ["naive", "ikj", "tiled", "avx2", "tiled_simd", "multithreaded"]

# The same awkward shapes the C++ harness uses: each dimension independently
# crosses the vector width (8), the register block (4, 16) and the tile (64).
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


def _operands(M: int, N: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    # [0.5, 1.5) rather than [0, 1), matching the C++ harness: no expected value
    # drifts near zero, so a region no code path wrote stays visibly wrong.
    rng = np.random.default_rng(0xC0FFEE ^ (M * 73856093) ^ (N * 19349663) ^ (K * 83492791))
    a = (rng.random((M, K), dtype=np.float32) + 0.5).astype(np.float32)
    b = (rng.random((K, N), dtype=np.float32) + 0.5).astype(np.float32)
    return a, b


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_matches_numpy(kernel: str, shape: tuple[int, int, int]) -> None:
    M, N, K = shape
    a, b = _operands(M, N, K)

    got = _core.gemm(a, b, kernel)

    assert got.shape == (M, N)
    assert got.dtype == np.float32
    # float32 accumulation reorders differently in every kernel, so this is a
    # tolerance check, not equality. Same 1e-5 the C++ harness uses.
    assert np.allclose(got, a @ b, rtol=1e-5, atol=1e-5)


def test_result_is_a_fresh_array() -> None:
    """Operands must not be modified, and the result must not alias them."""
    a, b = _operands(8, 8, 8)
    a_before, b_before = a.copy(), b.copy()

    out = _core.gemm(a, b)

    assert np.array_equal(a, a_before)
    assert np.array_equal(b, b_before)
    assert not np.shares_memory(out, a)
    assert not np.shares_memory(out, b)


def test_accepts_float64_and_fortran_order() -> None:
    """forcecast should convert dtype and layout rather than rejecting them."""
    a = np.asfortranarray(np.random.default_rng(1).random((16, 8)))  # float64, F-order
    b = np.random.default_rng(2).random((8, 16))                     # float64, C-order

    out = _core.gemm(a, b)

    assert out.dtype == np.float32
    assert np.allclose(out, (a @ b).astype(np.float32), rtol=1e-5, atol=1e-5)


def test_rejects_mismatched_shapes() -> None:
    a = np.ones((4, 8), dtype=np.float32)
    b = np.ones((7, 4), dtype=np.float32)  # 8 != 7

    with pytest.raises(Exception):
        _core.gemm(a, b)


def test_rejects_non_2d() -> None:
    a = np.ones((4, 4, 4), dtype=np.float32)
    b = np.ones((4, 4), dtype=np.float32)

    with pytest.raises(Exception):
        _core.gemm(a, b)
