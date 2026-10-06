"""SGD: the step, and the three ways it could quietly do nothing."""

from __future__ import annotations

import numpy as np
import pytest

import autograd
from autograd import _core, nn, optim


def param(a: np.ndarray) -> nn.Parameter:
    return nn.Parameter(_core.from_numpy(np.ascontiguousarray(a, dtype=np.float32)))


def set_grad(p: nn.Parameter, a: np.ndarray) -> None:
    p.grad = _core.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


# --------------------------------------------------------------------------
# step
# --------------------------------------------------------------------------


def test_step_moves_a_parameter_against_its_gradient() -> None:
    p = param(np.array([[1.0, 2.0, 3.0]]))
    set_grad(p, np.array([[10.0, -20.0, 0.0]]))

    optim.SGD([p], lr=0.1).step()

    assert np.allclose(p.to_numpy(), [[0.0, 4.0, 3.0]], rtol=1e-6, atol=1e-6)


def test_step_writes_into_the_parameters_own_buffer() -> None:
    """In place. A step that rebound p.data to a new tensor would leave every
    other holder of the buffer looking at the old weights."""
    p = param(np.ones((2, 2)))
    buffer = p.data
    set_grad(p, np.ones((2, 2)))

    optim.SGD([p], lr=0.5).step()

    assert p.data is buffer
    assert np.array_equal(np.asarray(buffer), np.full((2, 2), 0.5))


def test_step_leaves_the_gradient_as_it_found_it() -> None:
    p = param(np.ones(3))
    set_grad(p, np.array([1.0, 2.0, 3.0]))

    optim.SGD([p], lr=0.1).step()

    assert np.array_equal(np.asarray(p.grad), [1.0, 2.0, 3.0])


def test_step_skips_a_parameter_with_no_gradient() -> None:
    """A parameter the loss never touched has grad None. Not an error."""
    used = param(np.ones(2))
    unused = param(np.ones(2))
    set_grad(used, np.ones(2))

    optim.SGD([used, unused], lr=0.25).step()

    assert np.array_equal(used.to_numpy(), [0.75, 0.75])
    assert np.array_equal(unused.to_numpy(), [1.0, 1.0])


def test_two_steps_on_the_same_gradient_move_twice() -> None:
    p = param(np.array([1.0]))
    set_grad(p, np.array([1.0]))
    opt = optim.SGD([p], lr=0.25)

    opt.step()
    opt.step()

    assert p.to_numpy()[0] == pytest.approx(0.5)


def test_a_step_reads_a_strided_gradient() -> None:
    """Sum.backward hands the engine an expanded view, and .grad is built with
    add_into from whatever arrives - so a gradient is contiguous in practice,
    but the step should not depend on it."""
    p = param(np.zeros((2, 3)))
    p.grad = _core.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32)).expand([2, 3])

    optim.SGD([p], lr=1.0).step()

    assert np.array_equal(p.to_numpy(), [[-1.0, -2.0, -3.0], [-1.0, -2.0, -3.0]])


# --------------------------------------------------------------------------
# What the optimiser holds
# --------------------------------------------------------------------------


def test_the_optimiser_accepts_the_generator_parameters_returns() -> None:
    """parameters() is a generator and is exhausted after one pass. An
    optimiser that kept it would step once and then do nothing, forever, with
    no error - so the SECOND step is the one asserted on."""
    layer = nn.Linear(2, 3, generator=_core.Generator(0))
    opt = optim.SGD(layer.parameters(), lr=1.0)
    before = layer.bias.to_numpy()

    for _ in range(2):
        set_grad(layer.bias, np.ones((1, 3)))
        opt.step()

    assert np.array_equal(layer.bias.to_numpy(), before - 2.0)


def test_the_optimiser_steps_the_very_parameters_the_module_holds() -> None:
    layer = nn.Linear(2, 3, generator=_core.Generator(0))
    opt = optim.SGD(layer.parameters(), lr=0.1)

    assert len(opt.params) == 2
    assert opt.params[0] is layer.weight
    assert opt.params[1] is layer.bias


def test_a_parameter_shared_by_two_layers_is_stepped_once() -> None:
    """Tied weights. If parameters() listed the shared one twice, the optimiser
    would apply its gradient twice and the step would be double the size."""
    class Tied(nn.Module):
        def __init__(self) -> None:
            self.a = nn.Linear(2, 2, generator=_core.Generator(0), bias=False)
            self.b = nn.Linear(2, 2, generator=_core.Generator(0), bias=False)
            self.b.weight = self.a.weight

    model = Tied()
    before = model.a.weight.to_numpy()
    opt = optim.SGD(model.parameters(), lr=0.5)
    set_grad(model.a.weight, np.ones((2, 2)))

    opt.step()

    assert len(opt.params) == 1
    assert np.allclose(model.a.weight.to_numpy(), before - 0.5, atol=1e-6)


# --------------------------------------------------------------------------
# zero_grad
# --------------------------------------------------------------------------


def test_zero_grad_drops_every_gradient() -> None:
    a, b = param(np.ones(2)), param(np.ones(3))
    set_grad(a, np.ones(2))
    set_grad(b, np.ones(3))

    optim.SGD([a, b], lr=0.1).zero_grad()

    assert a.grad is None
    assert b.grad is None


def test_without_zero_grad_the_second_step_uses_both_gradients() -> None:
    """backward() accumulates, so this is what forgetting zero_grad costs."""
    w = param(np.array([[1.0, 1.0]]))
    x = autograd.Tensor(_core.from_numpy(np.array([[1.0], [2.0]], dtype=np.float32)))
    opt = optim.SGD([w], lr=0.1)

    def backward_once() -> None:
        autograd.sum(autograd.matmul(w, x)).backward()   # d/dw = [1, 2]

    backward_once()
    opt.step()                      # w = [0.9, 0.8]
    backward_once()                 # grad is now [2, 4]
    opt.step()                      # w = [0.7, 0.4]
    assert np.allclose(w.to_numpy(), [[0.7, 0.4]], atol=1e-6)

    opt.zero_grad()
    backward_once()                 # grad back to [1, 2]
    opt.step()                      # w = [0.6, 0.2]
    assert np.allclose(w.to_numpy(), [[0.6, 0.2]], atol=1e-6)
