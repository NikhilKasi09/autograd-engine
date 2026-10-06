"""The same MNIST run in PyTorch, as the baseline.

    python examples/mnist_torch.py --seed 0

Built to match mnist_train.py rather than assumed to: the starting weights are
copied out of this engine's own initialisation, and the batches come from the
same generator in the same order. What is left to differ is the arithmetic.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from autograd import _core

import mnist_data
import mnist_train
from mnist_train import BATCH_SIZE, EPOCHS, HIDDEN, LEARNING_RATE, Result


def build(seed: int, hidden: int = HIDDEN) -> torch.nn.Sequential:
    """A torch MLP holding exactly the weights mnist_train.MLP starts with."""
    ours = mnist_train.MLP(_core.Generator(seed), hidden)
    model = torch.nn.Sequential(
        torch.nn.Linear(784, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 10)
    )

    with torch.no_grad():
        for theirs, mine in ((model[0], ours.fc1), (model[2], ours.fc2)):
            # This engine stores a weight {in, out}; torch stores {out, in}.
            theirs.weight.copy_(torch.from_numpy(mine.weight.to_numpy().T.copy()))
            # And a bias {1, out}, where torch's is {out}.
            theirs.bias.copy_(torch.from_numpy(mine.bias.to_numpy()[0]))
    return model


def accuracy(model: torch.nn.Module, x: np.ndarray, labels: np.ndarray, chunk: int = 1000) -> float:
    correct = 0
    with torch.no_grad():
        for start in range(0, len(x), chunk):
            logits = model(torch.from_numpy(x[start:start + chunk])).numpy()
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
    threads: int | None = None,
    log=None,
) -> Result:
    """mnist_train.run, in torch. `threads` pins torch's thread count if given."""
    if threads is not None:
        torch.set_num_threads(threads)

    model = build(seed, hidden)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    # torch's cross_entropy takes class indices, and wants them as int64.
    labels = data.train_labels.astype(np.int64)

    result = Result([], [], [])
    for epoch in range(epochs):
        start = time.perf_counter()
        losses = []
        # Same generator, same length, so the same permutation as mnist_train.
        for xb, yb in mnist_data.batches(data.train_x, labels, batch_size, rng):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(
                model(torch.from_numpy(xb)), torch.from_numpy(yb)
            )
            loss.backward()
            opt.step()
            losses.append(loss.item())
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
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    run(mnist_data.load(), seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, hidden=args.hidden, threads=args.threads, log=print)


if __name__ == "__main__":
    main()
