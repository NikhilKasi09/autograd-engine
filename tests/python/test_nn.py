"""The nn layer: who owns a parameter, and how an optimiser finds it.

Gradients are not this file's job - linear and the losses are in
test_gradcheck.py and test_torch_crosscheck.py with everything else. What is
here is the structure: what parameters() returns, what a Linear is built with,
and that a Module is only a holder for the free function underneath it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import autograd
from autograd import _core, nn


def leaf(a: np.ndarray, requires_grad: bool = False) -> autograd.Tensor:
    raw = _core.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    return autograd.Tensor(raw, requires_grad=requires_grad)


def param(*shape: int) -> nn.Parameter:
    return nn.Parameter(_core.zeros(list(shape)))


# --------------------------------------------------------------------------
# Parameter
# --------------------------------------------------------------------------


def test_a_parameter_requires_grad_by_default() -> None:
    """The one difference from Tensor, which defaults to False."""
    assert param(2, 3).requires_grad
    assert not autograd.Tensor(_core.zeros([2, 3])).requires_grad


def test_a_parameter_is_a_leaf_tensor_over_the_buffer_it_was_given() -> None:
    raw = _core.zeros([2, 3])
    p = nn.Parameter(raw)

    assert isinstance(p, autograd.Tensor)
    assert p.is_leaf
    assert p.data is raw


def test_an_op_on_a_parameter_returns_a_plain_tensor() -> None:
    """Otherwise every activation would look trainable to parameters()."""
    out = autograd.add(param(2, 3), param(2, 3))

    assert not isinstance(out, nn.Parameter)
    assert out.requires_grad


# --------------------------------------------------------------------------
# Module.parameters
# --------------------------------------------------------------------------


class Two(nn.Module):
    def __init__(self) -> None:
        self.first = param(2, 3)
        self.second = param(1, 3)


class Nested(nn.Module):
    def __init__(self) -> None:
        self.head = param(4)
        self.inner = Two()
        self.tail = param(5)


def test_parameters_come_back_by_identity_in_assignment_order() -> None:
    m = Two()

    found = list(m.parameters())

    assert len(found) == 2
    assert found[0] is m.first
    assert found[1] is m.second


def test_parameters_recurses_into_child_modules() -> None:
    """Without the recursion a model's layers never reach the optimiser, and
    nothing raises - they just do not train."""
    m = Nested()

    found = list(m.parameters())

    assert [id(p) for p in found] == [
        id(m.head), id(m.inner.first), id(m.inner.second), id(m.tail)
    ]


def test_parameters_ignores_everything_that_is_not_a_parameter_or_module() -> None:
    class Mixed(nn.Module):
        def __init__(self) -> None:
            self.w = param(2, 2)
            self.frozen = autograd.Tensor(_core.zeros([2, 2]), requires_grad=True)
            self.width = 2
            self.name = "mixed"
            self.none = None
            self.in_a_list = [param(3)]

    m = Mixed()

    assert [id(p) for p in m.parameters()] == [id(m.w)]


def test_a_parameter_assigned_twice_on_one_module_is_listed_once() -> None:
    class Tied(nn.Module):
        def __init__(self) -> None:
            self.a = param(2, 2)
            self.b = self.a

    assert len(list(Tied().parameters())) == 1


def test_a_parameter_shared_between_two_children_is_listed_once() -> None:
    """The case the dedupe exists for, and the one a per-module seen set
    misses: each child would report the shared weight as new."""
    shared = param(2, 2)

    class Child(nn.Module):
        def __init__(self) -> None:
            self.w = shared
            self.own = param(3)

    class Parent(nn.Module):
        def __init__(self) -> None:
            self.left = Child()
            self.right = Child()

    m = Parent()
    found = list(m.parameters())

    assert [id(p) for p in found] == [id(shared), id(m.left.own), id(m.right.own)]


def test_a_child_module_used_twice_contributes_its_parameters_once() -> None:
    class Parent(nn.Module):
        def __init__(self) -> None:
            self.a = Two()
            self.b = self.a

    assert len(list(Parent().parameters())) == 2


def test_parameters_can_be_called_again() -> None:
    """Each call starts a fresh walk; the seen set is not kept on the module."""
    m = Nested()

    assert len(list(m.parameters())) == 4
    assert len(list(m.parameters())) == 4


def test_zero_grad_clears_every_parameter_including_a_childs() -> None:
    m = Nested()
    for p in m.parameters():
        p.grad = _core.zeros_like(p.data)

    m.zero_grad()

    assert all(p.grad is None for p in m.parameters())


def test_calling_a_module_runs_forward() -> None:
    class Doubler(nn.Module):
        def forward(self, x: autograd.Tensor) -> autograd.Tensor:
            return autograd.scale(x, 2.0)

    out = Doubler()(leaf(np.array([1.0, 2.0])))

    assert np.array_equal(out.to_numpy(), [2.0, 4.0])


def test_a_module_without_forward_says_so() -> None:
    with pytest.raises(NotImplementedError):
        nn.Module()(leaf(np.ones(2)))


# --------------------------------------------------------------------------
# linear, the free function
# --------------------------------------------------------------------------


@pytest.mark.parametrize("M", [2, 1, 5], ids=["batch2", "batch1", "batch5"])
def test_linear_matches_numpy_at_any_batch_size(M: int, rng) -> None:
    """The bias expand reads M off x, so nothing is fixed at construction."""
    x = rng.uniform(-1, 1, (M, 3)).astype(np.float32)
    w = rng.uniform(-1, 1, (3, 4)).astype(np.float32)
    b = rng.uniform(-1, 1, (1, 4)).astype(np.float32)

    out = nn.linear(leaf(x), leaf(w), leaf(b))

    assert out.shape == (M, 4)
    assert np.allclose(out.to_numpy(), x @ w + b, rtol=1e-5, atol=1e-6)


def test_linear_without_a_bias_is_the_matmul() -> None:
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    w = np.arange(12, dtype=np.float32).reshape(3, 4)

    out = nn.linear(leaf(x), leaf(w))

    assert np.array_equal(out.to_numpy(), x @ w)


def test_linear_broadcasts_the_bias_without_copying_it() -> None:
    """The bias reaches the add as a stride-0 view over its own buffer."""
    b = leaf(np.ones((1, 4)), requires_grad=True)

    out = nn.linear(leaf(np.ones((3, 2))), leaf(np.ones((2, 4))), b)

    expanded = out.grad_fn.parents[1]
    assert expanded.shape == (3, 4)
    assert expanded.data.shares_storage_with(b.data)
    assert expanded.data.strides[0] == 0


# --------------------------------------------------------------------------
# Linear, the module
# --------------------------------------------------------------------------


def test_linear_stores_the_weight_in_by_out_and_the_bias_as_a_row() -> None:
    """{in, out}, the transpose of torch's layout. Not square on purpose."""
    layer = nn.Linear(3, 5, generator=_core.Generator(0))

    assert layer.weight.shape == (3, 5)
    assert layer.bias.shape == (1, 5)
    assert isinstance(layer.weight, nn.Parameter)
    assert isinstance(layer.bias, nn.Parameter)


def test_linear_lists_weight_then_bias() -> None:
    layer = nn.Linear(3, 5, generator=_core.Generator(0))

    found = list(layer.parameters())

    assert len(found) == 2
    assert found[0] is layer.weight
    assert found[1] is layer.bias


def test_linear_without_a_bias_has_one_parameter_and_still_runs() -> None:
    layer = nn.Linear(3, 5, generator=_core.Generator(0), bias=False)

    assert layer.bias is None
    assert len(list(layer.parameters())) == 1
    assert layer(leaf(np.ones((2, 3)))).shape == (2, 5)


def test_linear_weights_fill_the_fan_in_range_and_the_bias_is_zero() -> None:
    fan_in = 16
    bound = 1.0 / math.sqrt(fan_in)
    layer = nn.Linear(fan_in, 64, generator=_core.Generator(1))

    w = layer.weight.to_numpy()

    assert w.min() >= -bound
    assert w.max() < bound
    # 1024 draws: both halves of the range are reached, so the bound is the
    # real one and not something much smaller.
    assert w.min() < -0.9 * bound
    assert w.max() > 0.9 * bound
    assert np.array_equal(layer.bias.to_numpy(), np.zeros((1, 64)))


def test_linear_is_reproducible_from_the_seed() -> None:
    a = nn.Linear(3, 5, generator=_core.Generator(42))
    b = nn.Linear(3, 5, generator=_core.Generator(42))

    assert np.array_equal(a.weight.to_numpy(), b.weight.to_numpy())


def test_two_layers_from_one_generator_get_different_weights() -> None:
    """One stream, drawn from twice. Identical layers would mean the generator
    restarted, and a network of identical layers has nothing to learn from."""
    g = _core.Generator(42)
    a = nn.Linear(3, 5, generator=g)
    b = nn.Linear(3, 5, generator=g)

    assert not np.array_equal(a.weight.to_numpy(), b.weight.to_numpy())


def test_linear_needs_a_generator() -> None:
    with pytest.raises(TypeError):
        nn.Linear(3, 5)


def test_the_module_is_only_a_holder_for_the_free_function(rng) -> None:
    layer = nn.Linear(3, 4, generator=_core.Generator(3))
    x = leaf(rng.uniform(-1, 1, (2, 3)))

    via_module = layer(x).to_numpy()
    via_function = nn.linear(x, layer.weight, layer.bias).to_numpy()

    assert np.array_equal(via_module, via_function)


def test_backward_through_a_linear_reaches_both_parameters_and_not_the_input(rng) -> None:
    layer = nn.Linear(3, 4, generator=_core.Generator(3))
    x = leaf(rng.uniform(-1, 1, (2, 3)))

    autograd.sum(layer(x)).backward()

    assert layer.weight.grad.shape == (3, 4)
    assert layer.bias.grad.shape == (1, 4)
    # d(sum)/db is the batch size in every slot.
    assert np.array_equal(np.asarray(layer.bias.grad), np.full((1, 4), 2.0))
    assert x.grad is None


# --------------------------------------------------------------------------
# mse_loss
# --------------------------------------------------------------------------


def test_mse_loss_is_the_mean_squared_difference(rng) -> None:
    p = rng.uniform(-1, 1, (2, 3)).astype(np.float32)
    t = rng.uniform(-1, 1, (2, 3)).astype(np.float32)

    out = nn.mse_loss(leaf(p), leaf(t))

    assert out.to_numpy()[0, 0] == pytest.approx(np.mean((p - t) ** 2), rel=1e-6)


def test_mse_loss_keeps_the_rank_of_its_input() -> None:
    """{1,1} rather than {1}, the same choice sum makes and for the same reason."""
    assert nn.mse_loss(leaf(np.ones((2, 3))), leaf(np.zeros((2, 3)))).shape == (1, 1)
    assert nn.mse_loss(leaf(np.ones(4)), leaf(np.zeros(4))).shape == (1,)


def test_mse_loss_with_sum_reduction_skips_the_average() -> None:
    p = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    t = np.zeros((1, 3), dtype=np.float32)

    assert nn.mse_loss(leaf(p), leaf(t), reduction="sum").to_numpy()[0, 0] == 14.0
    assert nn.mse_loss(leaf(p), leaf(t)).to_numpy()[0, 0] == pytest.approx(14.0 / 3.0)


def test_mse_loss_of_identical_inputs_is_exactly_zero() -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)

    assert nn.mse_loss(leaf(a), leaf(a)).to_numpy()[0, 0] == 0.0


def test_mse_loss_rejects_an_unknown_reduction() -> None:
    with pytest.raises(ValueError, match="reduction must be"):
        nn.mse_loss(leaf(np.ones(2)), leaf(np.ones(2)), reduction="none")


def test_mse_loss_gradient_is_two_over_n_times_the_difference() -> None:
    """Hand-derived, with values that make a halved gradient obvious.

    The difference is squared with mul(diff, diff), so diff gets a gradient
    from each side of the multiply. An engine that kept only one of them
    returns exactly half of this.
    """
    p = leaf(np.array([[1.0, 2.0, 4.0]]), requires_grad=True)
    t = leaf(np.array([[0.0, 0.0, 1.0]]))

    nn.mse_loss(p, t).backward()

    assert np.allclose(np.asarray(p.grad), [[2 / 3, 4 / 3, 2.0]], rtol=1e-6)
    assert t.grad is None


def test_mse_loss_adds_no_node_type_of_its_own() -> None:
    """A composite: the root is the Scale that takes the mean."""
    out = nn.mse_loss(leaf(np.ones((2, 3)), requires_grad=True), leaf(np.zeros((2, 3))))

    assert type(out.grad_fn).__name__ == "Scale"
