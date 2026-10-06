"""Generator through the binding.

The Catch2 [random] cases already cover the C++ behaviour. What is left for
pytest is what the binding itself could get wrong: whether the seed actually
reaches the engine, whether Python holds one generator or silently gets a fresh
one per call, and whether the C++ rejections arrive as ValueError rather than
passing through as something unhandled.

The one this file is really for is the third: a Generator that pybind11 copied
on the way in or out would hand back an independent stream, and every value
test above it would still pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from autograd import _core


def drain(g: _core.Generator, n: int, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """One fill, as a numpy copy in flat buffer order."""
    t = _core.zeros([n])
    g.uniform(t, lo, hi)
    return t.to_numpy()


def test_the_same_seed_reproduces() -> None:
    assert np.array_equal(drain(_core.Generator(0xC0FFEE), 32),
                          drain(_core.Generator(0xC0FFEE), 32))


def test_different_seeds_diverge() -> None:
    assert not np.array_equal(drain(_core.Generator(1), 32),
                              drain(_core.Generator(2), 32))


def test_the_seed_argument_is_keyword_addressable() -> None:
    """py::arg("seed"), not a positional-only binding."""
    assert np.array_equal(drain(_core.Generator(seed=5), 8),
                          drain(_core.Generator(5), 8))


def test_one_generator_holds_its_stream_across_calls() -> None:
    """The binding-level version of the C++ stream-advance case.

    Distinct from it: this fails if pybind11 hands Python a COPY of the
    generator rather than a reference to the one it owns, which the C++ suite
    cannot see. Deleting copy and move in the header is what makes that a
    compile error rather than a silent one, so this test is the proof that the
    deletion is doing its job through the binding too.
    """
    g = _core.Generator(7)
    assert not np.array_equal(drain(g, 32), drain(g, 32))


def test_values_land_in_the_requested_range() -> None:
    g = _core.Generator(11)
    t = _core.zeros([16, 4])
    g.uniform(t, -2.0, 3.0)

    values = t.to_numpy()
    assert values.min() >= -2.0
    assert values.max() < 3.0


def test_uniform_fills_the_tensor_it_is_given() -> None:
    """Out-parameter style, matching every op in _core. Returns None."""
    g = _core.Generator(13)
    t = _core.zeros([8, 8])

    assert g.uniform(t, 1.0, 2.0) is None
    assert (t.to_numpy() >= 1.0).all()


def test_a_non_contiguous_destination_raises() -> None:
    g = _core.Generator(17)

    transposed = _core.zeros([4, 6]).transpose(0, 1)
    with pytest.raises(ValueError, match="contiguous"):
        g.uniform(transposed, 0.0, 1.0)

    expanded = _core.zeros([1, 6]).expand([4, 6])
    with pytest.raises(ValueError, match="contiguous"):
        g.uniform(expanded, 0.0, 1.0)


def test_an_inverted_or_empty_range_raises() -> None:
    g = _core.Generator(19)
    t = _core.zeros([4])

    with pytest.raises(ValueError):
        g.uniform(t, 1.0, 0.0)
    with pytest.raises(ValueError):
        g.uniform(t, 1.0, 1.0)


def test_the_generator_is_documented() -> None:
    assert _core.Generator.__doc__
    assert _core.Generator.uniform.__doc__
