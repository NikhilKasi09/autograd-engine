"""Toolchain smoke test for the pybind11 layer.

Proves the compiled module imports and round-trips a value. Replaced by real
tensor tests in roadmap phase 4.
"""

from autograd import _core


def test_module_imports() -> None:
    assert hasattr(_core, "add")


def test_add() -> None:
    assert _core.add(2, 3) == 5.0
    assert _core.add(-1.5, 0.5) == -1.0
