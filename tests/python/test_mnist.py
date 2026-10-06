"""The phase 9 deliverable: the engine trains on real data.

These need the MNIST files, and skip without them - the pytest header says
which. Fetch them with `python examples/mnist_data.py`.

The thresholds were measured over seeds 0-4 before being written down: after
one epoch the worst seed scored 91.97%, and no first loss was further than
0.022 from ln 10.
"""

from __future__ import annotations

import math

import pytest

import mnist_data
import mnist_train

pytestmark = pytest.mark.skipif(
    not mnist_data.is_present(), reason="MNIST not downloaded: python examples/mnist_data.py"
)


@pytest.fixture(scope="module")
def data() -> mnist_data.Dataset:
    return mnist_data.load()


@pytest.fixture(scope="module")
def one_epoch(data: mnist_data.Dataset) -> mnist_train.Result:
    """One epoch from seed 0, shared: it is the slow part of this file."""
    return mnist_train.run(data, seed=0, epochs=1)


def first_steps(data: mnist_data.Dataset, steps: int) -> mnist_data.Dataset:
    """The same dataset with a training set just long enough for `steps` batches."""
    n = steps * mnist_train.BATCH_SIZE
    return data._replace(train_x=data.train_x[:n], train_labels=data.train_labels[:n])


def test_the_first_loss_is_that_of_knowing_nothing(one_epoch: mnist_train.Result) -> None:
    """ln 10: ten classes, and a fresh model has no reason to prefer any."""
    assert one_epoch.losses[0] == pytest.approx(math.log(10.0), abs=0.1)


def test_one_epoch_takes_the_model_past_ninety_percent(one_epoch: mnist_train.Result) -> None:
    """On the TEST set, which the model never trained on."""
    assert len(one_epoch.losses) == 938          # 60000 / 64, the short batch kept
    assert one_epoch.accuracy[0] > 0.90


def test_a_run_is_reproducible_from_its_seed(data: mnist_data.Dataset) -> None:
    """To the bit. The seed drives the weights and the shuffle, and the graph's
    GEMM is single threaded, so nothing is left to vary."""
    small = first_steps(data, 100)

    first = mnist_train.run(small, seed=7, epochs=1)
    second = mnist_train.run(small, seed=7, epochs=1)

    assert first.losses == second.losses
    assert first.accuracy == second.accuracy


def test_a_different_seed_is_a_different_run(data: mnist_data.Dataset) -> None:
    """Otherwise the test above would pass with the seed ignored."""
    small = first_steps(data, 20)

    assert mnist_train.run(small, seed=7, epochs=1).losses != \
           mnist_train.run(small, seed=8, epochs=1).losses
