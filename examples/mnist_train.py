"""Train a 784-128-10 MLP on MNIST with this engine.

    python examples/mnist_data.py           # once, to fetch the data
    python examples/mnist_train.py --seed 0

NumPy is used to load, shuffle and score. Every number the model computes -
forward, loss, gradients, the update - comes from the engine.
"""

from __future__ import annotations

import argparse
import time
from typing import NamedTuple

import numpy as np

import autograd
from autograd import _core, nn, optim

import mnist_data

# Fixed before the first run and not tuned afterwards.
HIDDEN = 128
BATCH_SIZE = 64
LEARNING_RATE = 0.1
EPOCHS = 10


class MLP(nn.Module):
    """Two linear layers with a relu between them. Outputs logits, not probabilities."""

    def __init__(self, generator: _core.Generator, hidden: int = HIDDEN) -> None:
        self.fc1 = nn.Linear(784, hidden, generator=generator)
        self.fc2 = nn.Linear(hidden, 10, generator=generator)

    def forward(self, x: autograd.Tensor) -> autograd.Tensor:
        return self.fc2(autograd.relu(self.fc1(x)))


class Result(NamedTuple):
    losses: list[float]            # one per step, every epoch end to end
    accuracy: list[float]          # test accuracy after each epoch
    seconds: list[float]           # training time per epoch, scoring excluded


def tensor(a: np.ndarray) -> autograd.Tensor:
    """Copy a NumPy batch into a graph leaf that needs no gradient."""
    return autograd.Tensor(_core.from_numpy(a))


def train_epoch(
    model: nn.Module,
    opt: optim.SGD,
    x: np.ndarray,
    onehot: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> list[float]:
    """One pass over the training set. Returns the loss at every step."""
    losses = []
    for xb, yb in mnist_data.batches(x, onehot, batch_size, rng):
        opt.zero_grad()                     # backward accumulates, so clear first
        loss = nn.cross_entropy(model(tensor(xb)), tensor(yb))
        loss.backward()
        opt.step()
        losses.append(loss.data[0, 0])      # the loss is a {1,1} tensor
    return losses


def accuracy(model: nn.Module, x: np.ndarray, labels: np.ndarray, chunk: int = 1000) -> float:
    """Fraction of rows whose largest logit is the right class."""
    correct = 0
    for start in range(0, len(x), chunk):
        logits = model(tensor(x[start:start + chunk])).to_numpy()
        correct += int((logits.argmax(axis=1) == labels[start:start + chunk]).sum())
    return correct / len(x)


def run(
    data: mnist_data.Dataset,
    *,
    seed: int = 0,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    hidden: int = HIDDEN,
    log=None,
) -> Result:
    """Train from `seed` and score on the test set after every epoch.

    The seed drives both the initial weights and the shuffle, so a run is
    reproducible from this one number.
    """
    model = MLP(_core.Generator(seed), hidden)
    opt = optim.SGD(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    onehot = mnist_data.one_hot(data.train_labels)

    result = Result([], [], [])
    for epoch in range(epochs):
        start = time.perf_counter()
        losses = train_epoch(model, opt, data.train_x, onehot, batch_size, rng)
        result.seconds.append(time.perf_counter() - start)

        result.losses.extend(losses)
        result.accuracy.append(accuracy(model, data.test_x, data.test_labels))

        if log is not None:
            log(f"epoch {epoch + 1:2d}  loss {np.mean(losses):.4f}  "
                f"test accuracy {result.accuracy[-1]:.4f}  {result.seconds[-1]:.2f}s")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    args = parser.parse_args()

    run(mnist_data.load(), seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, hidden=args.hidden, log=print)


if __name__ == "__main__":
    main()
