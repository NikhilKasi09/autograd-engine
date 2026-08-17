"""Module-level smoke test for the pybind11 layer.

Everything of substance is tested in test_tensor.py, test_numpy_interop.py,
test_ops.py and test_gemm.py. What is left here is the surface itself: that the
module imports, that it exposes what the other files assume, and that the phase
2 throwaway is genuinely gone rather than shadowed by an overload.
"""

from __future__ import annotations

import numpy as np
import pytest

from autograd import _core

# The whole public surface. Adding to this list is a deliberate act, which is
# the point - it is the one place the shape of _core is written down.
EXPECTED = [
    "Tensor",
    "MAX_RANK",
    "zeros",
    "zeros_like",
    "from_numpy",
    "add",
    "mul",
    "scale",
    "relu",
    "add_into",
    "sum",
    "gemm",
]


def test_module_imports_and_is_documented() -> None:
    assert _core.__doc__
    assert "autograd engine" in _core.__doc__


def test_the_expected_surface_is_present() -> None:
    missing = [name for name in EXPECTED if not hasattr(_core, name)]
    assert not missing, f"missing from _core: {missing}"


def test_tensor_methods_are_present() -> None:
    methods = [
        "shape",
        "strides",
        "rank",
        "numel",
        "is_contiguous",
        "shares_storage_with",
        "clone",
        "zero_",
        "transpose",
        "permute",
        "slice",
        "expand",
        "reshape",
        "contiguous",
        "to_numpy",
    ]
    missing = [name for name in methods if not hasattr(_core.Tensor, name)]
    assert not missing, f"missing from _core.Tensor: {missing}"


def test_the_phase_2_throwaway_is_gone() -> None:
    """add and gemm were both scalar/numpy functions before phase 4.

    Deleting them matters beyond tidiness: while both existed, pybind11 stacked
    them as overloads of the same names, so help() showed only "Overloaded
    function" and the real contract was buried. This asserts the tensor form is
    the only form left.
    """
    # The old add(float, float) -> float is gone; add is the elementwise op.
    with pytest.raises(TypeError):
        _core.add(2.0, 3.0)

    # The old gemm(ndarray, ndarray) -> ndarray is gone; gemm takes tensors and
    # writes into its third argument.
    with pytest.raises(TypeError):
        _core.gemm(np.ones((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32))


def test_docstrings_survived_the_overload_collapse() -> None:
    # With one signature each, help() shows the contract rather than
    # "Overloaded function." This is what deleting the throwaway bought.
    assert "Overloaded function" not in (_core.add.__doc__ or "")
    assert "elementwise" in _core.add.__doc__
    assert "Overloaded function" not in (_core.gemm.__doc__ or "")
    assert "Accumulates" in _core.gemm.__doc__
