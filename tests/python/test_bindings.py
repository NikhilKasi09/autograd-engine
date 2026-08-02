"""Toolchain smoke test for the pybind11 layer.

Proves the compiled module imports and round-trips a value. Replaced by real
tensor tests in roadmap phase 4.
"""

import sys
from pathlib import Path

# The module is built into python/ by CMake rather than installed.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "python"))

import _core  # noqa: E402


def test_module_imports() -> None:
    assert hasattr(_core, "add")


def test_add() -> None:
    assert _core.add(2, 3) == 5.0
    assert _core.add(-1.5, 0.5) == -1.0
