"""The phase 8 deliverable: a model, a loss and an optimiser, training.

Everything below is built only from what the package exports - Module, Linear,
relu, the two losses and SGD - arranged the way phase 9 will arrange them.
The thresholds were measured over five seeds before being written down, and
each sits well clear of the worst run.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

import autograd
from autograd import _core, nn, optim


class MLP(nn.Module):
    """Two layers held as attributes, which is what makes parameters() recurse."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int, generator: _core.Generator) -> None:
        self.fc1 = nn.Linear(n_in, n_hidden, generator=generator)
        self.fc2 = nn.Linear(n_hidden, n_out, generator=generator)

    def forward(self, x: autograd.Tensor) -> autograd.Tensor:
        return self.fc2(autograd.relu(self.fc1(x)))


def tensor(a: np.ndarray) -> autograd.Tensor:
    return autograd.Tensor(_core.from_numpy(np.ascontiguousarray(a, dtype=np.float32)))


def blobs(per_class: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three well-separated clusters in the plane: inputs, labels, one-hot."""
    rng = np.random.default_rng(0)
    centres = np.array([[2.0, 0.0], [-1.0, 1.7], [-1.0, -1.7]])
    x = np.concatenate([c + 0.5 * rng.standard_normal((per_class, 2)) for c in centres])
    labels = np.repeat(np.arange(3), per_class)
    onehot = np.zeros((len(labels), 3), dtype=np.float32)
    onehot[np.arange(len(labels)), labels] = 1.0
    return x.astype(np.float32), labels, onehot


def train(model: nn.Module, x: np.ndarray, target: np.ndarray, *, steps: int,
          lr: float, loss_fn=nn.cross_entropy) -> list[float]:
    """The loop: zero, forward, loss, backward, step. Returns every loss."""
    opt = optim.SGD(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(tensor(x)), tensor(target))
        loss.backward()
        opt.step()
        losses.append(float(loss.to_numpy().ravel()[0]))
    return losses


def accuracy(model: nn.Module, x: np.ndarray, labels: np.ndarray) -> float:
    return float((model(tensor(x)).to_numpy().argmax(axis=1) == labels).mean())


# --------------------------------------------------------------------------
# It trains
# --------------------------------------------------------------------------


def test_an_mlp_learns_to_classify_three_clusters() -> None:
    x, labels, onehot = blobs()
    model = MLP(2, 8, 3, _core.Generator(0))

    assert accuracy(model, x, labels) < 0.9      # not solved by the init

    losses = train(model, x, onehot, steps=60, lr=0.5)

    # Starts near log(3), the loss of knowing nothing about three classes.
    assert losses[0] == pytest.approx(np.log(3.0), abs=0.3)
    assert losses[-1] < 0.05
    assert accuracy(model, x, labels) == 1.0


def test_the_loss_falls_on_every_single_step() -> None:
    """Full-batch gradient descent on a smooth loss at a sane step size has no
    reason to go up. A sign error, a stale gradient or a parameter stepped
    twice all show here before they show in the final accuracy."""
    x, _, onehot = blobs()

    losses = train(MLP(2, 8, 3, _core.Generator(0)), x, onehot, steps=60, lr=0.5)

    assert all(after < before for before, after in zip(losses, losses[1:]))


def test_an_mlp_learns_xor() -> None:
    """Not linearly separable, so this only works if the hidden layer and the
    relu are both carrying gradient. A model that trained just its last layer
    could not do it."""
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    labels = np.array([0, 1, 1, 0])
    onehot = np.eye(2, dtype=np.float32)[labels]
    model = MLP(2, 8, 2, _core.Generator(0))

    losses = train(model, x, onehot, steps=400, lr=0.5)

    assert losses[-1] < 0.05
    assert accuracy(model, x, labels) == 1.0


def test_a_linear_layer_recovers_a_known_line_under_mse() -> None:
    """Regression against y = x @ w + b with the answer known in advance."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (64, 3)).astype(np.float32)
    true_w = np.array([[1.5], [-2.0], [0.5]], dtype=np.float32)
    y = x @ true_w + 0.3

    layer = nn.Linear(3, 1, generator=_core.Generator(0))
    losses = train(layer, x, y, steps=200, lr=0.5, loss_fn=nn.mse_loss)

    assert losses[-1] < 1e-8
    assert np.allclose(layer.weight.to_numpy(), true_w, atol=1e-3)
    assert np.allclose(layer.bias.to_numpy(), [[0.3]], atol=1e-3)


def test_training_is_reproducible_from_the_seed() -> None:
    """The reason Generator exists. Same seed, same data, same losses - to the
    bit, not approximately."""
    x, _, onehot = blobs()

    first = train(MLP(2, 8, 3, _core.Generator(7)), x, onehot, steps=10, lr=0.5)
    second = train(MLP(2, 8, 3, _core.Generator(7)), x, onehot, steps=10, lr=0.5)

    assert first == second


def test_every_parameter_of_the_model_is_trained() -> None:
    """Four parameters, two per layer, and one step moves all of them. A
    parameters() that missed the child modules would move none and raise
    nothing."""
    x, _, onehot = blobs()
    model = MLP(2, 8, 3, _core.Generator(0))
    before = [p.to_numpy() for p in model.parameters()]

    train(model, x, onehot, steps=1, lr=0.5)

    after = [p.to_numpy() for p in model.parameters()]
    assert len(before) == 4
    for b, a in zip(before, after):
        assert not np.array_equal(b, a)


# --------------------------------------------------------------------------
# It lets go
# --------------------------------------------------------------------------


def _train_one_step_and_watch_it() -> tuple[MLP, list[weakref.ref]]:
    """Run one full step and weakref everything the step made.

    Built here, not passed in, for the reason the other lifetime helpers give:
    a tensor the caller still holds stays alive for innocent reasons.
    """
    x, _, onehot = blobs(per_class=4)
    model = MLP(2, 8, 3, _core.Generator(0))
    opt = optim.SGD(model.parameters(), lr=0.5)

    inputs, target = tensor(x), tensor(onehot)
    hidden = autograd.relu(model.fc1(inputs))
    logits = model.fc2(hidden)
    loss = nn.cross_entropy(logits, target)
    loss.backward()
    opt.step()

    watched = [
        weakref.ref(hidden), weakref.ref(logits), weakref.ref(loss),
        weakref.ref(hidden.data), weakref.ref(logits.data),
        # The softmax buffer the loss saved: the biggest thing a step makes.
        weakref.ref(loss.grad_fn.saved[0]),
    ]
    return model, watched


def test_a_finished_step_leaves_nothing_behind_but_the_parameters() -> None:
    """The phase 8 leak, if there were one.

    A model that remembered its last output, or an optimiser that held the
    loss, would keep a whole step's activations alive - and every value test
    above would still pass. No gc.collect(): refcounting alone has to do it.
    """
    gc.disable()
    try:
        model, watched = _train_one_step_and_watch_it()

        alive = [i for i, ref in enumerate(watched) if ref() is not None]
        assert alive == [], f"still reachable after the step: {alive}"

        # And the parameters are untouched by all that dying.
        assert len(list(model.parameters())) == 4
        assert all(p.grad is not None for p in model.parameters())
    finally:
        gc.enable()
