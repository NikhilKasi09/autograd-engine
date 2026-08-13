"""Shared fixtures and the import guard for the Python test suite.

The compiled module is not installed. CMake drops it into the tree that built
it, and every tree holds a complete `autograd` package, so the only question is
which one is on sys.path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# The Release tree writes its module next to the sources, and that is the
# fallback so a bare `pytest tests/python` from the repo root works.
#
# APPEND, never insert. PYTHONPATH entries are already in sys.path ahead of
# this one, so `PYTHONPATH=build-debug/python pytest ...` still wins. Inserting
# at the front here would silently test the Release module while you believed
# you were testing the debug one.
_RELEASE_ROOT = Path(__file__).resolve().parents[2] / "python"
if str(_RELEASE_ROOT) not in sys.path:
    sys.path.append(str(_RELEASE_ROOT))

try:
    from autograd import _core
except ImportError as exc:
    # Fail, never skip. A suite that skips on a broken build goes green with
    # zero tests, and this repo has produced that failure once already.
    raise RuntimeError(
        "could not import autograd._core. Build it first:\n"
        "    cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\n"
        "For any other tree, put that tree's package root on PYTHONPATH:\n"
        "    PYTHONPATH=build-debug/python .venv/bin/pytest tests/python"
    ) from exc

# Set by the ctest entry so a wrong-tree import is a hard failure there. Unset
# for a manual run, where the header line below is the diagnostic instead.
_EXPECTED = os.environ.get("GEMM_EXPECTED_MODULE_DIR")
if _EXPECTED and Path(_core.__file__).parent != Path(_EXPECTED).resolve():
    raise RuntimeError(
        f"imported the wrong _core: {_core.__file__}\nexpected it under: {_EXPECTED}"
    )


def pytest_report_header(config: pytest.Config) -> str:
    """Name the tree the module actually came from, on every run.

    A wrong-tree import is otherwise invisible until two numbers disagree.
    """
    return f"autograd._core: {_core.__file__}"


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded generator. `Tensor::randomize` is unseeded and is not bound."""
    return np.random.default_rng(0xC0FFEE)
