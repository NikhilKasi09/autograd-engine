"""The MNIST loader, tested without MNIST.

Every file here is synthetic and written to tmp_path, so none of this needs
the network or the real data. The images are 2x3 with distinct values: a
non-square shape is what lets a swapped height and width show.
"""

from __future__ import annotations

import gzip
import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

import mnist_data


def write_idx(path: Path, array: np.ndarray, *, magic: bytes = b"\x00\x00\x08") -> str:
    """Write `array` as a gzipped IDX file and return its SHA-256."""
    header = magic + bytes([array.ndim]) + struct.pack(f">{array.ndim}I", *array.shape)
    path.write_bytes(gzip.compress(header + array.astype(np.uint8).tobytes()))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def images() -> np.ndarray:
    """Three 2x3 images, every pixel different."""
    return np.arange(18, dtype=np.uint8).reshape(3, 2, 3) * 10


def fake_dataset(root: Path, *, test_labels: int = 2) -> dict[str, str]:
    """Four files shaped like MNIST's, and the checksums that go with them."""
    root.mkdir(parents=True, exist_ok=True)
    contents = {
        "train_images": images(),
        "train_labels": np.array([7, 0, 3], dtype=np.uint8),
        "test_images": images()[:2],
        "test_labels": np.arange(test_labels, dtype=np.uint8),
    }
    return {
        mnist_data.FILES[key]: write_idx(root / mnist_data.FILES[key], array)
        for key, array in contents.items()
    }


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_read_idx_returns_the_values_in_file_order(tmp_path: Path) -> None:
    write_idx(tmp_path / "x.gz", images())

    got = mnist_data.read_idx(tmp_path / "x.gz")

    assert got.shape == (3, 2, 3)
    assert got.dtype == np.uint8
    # Elementwise, not a sum: a transposed read holds the same values.
    assert np.array_equal(got, images())


def test_read_idx_reads_a_rank_one_label_file(tmp_path: Path) -> None:
    labels = np.array([5, 0, 4, 1, 9], dtype=np.uint8)
    write_idx(tmp_path / "y.gz", labels)

    assert np.array_equal(mnist_data.read_idx(tmp_path / "y.gz"), labels)


def test_read_idx_returns_an_array_the_caller_can_write_to(tmp_path: Path) -> None:
    write_idx(tmp_path / "x.gz", images())

    mnist_data.read_idx(tmp_path / "x.gz")[0, 0, 0] = 1


def test_read_idx_rejects_a_bad_magic(tmp_path: Path) -> None:
    write_idx(tmp_path / "x.gz", images(), magic=b"\x01\x00\x08")

    with pytest.raises(ValueError, match="bad magic"):
        mnist_data.read_idx(tmp_path / "x.gz")


def test_read_idx_rejects_a_type_other_than_unsigned_byte(tmp_path: Path) -> None:
    # 0x0D is float32 in the IDX spec.
    write_idx(tmp_path / "x.gz", images(), magic=b"\x00\x00\x0d")

    with pytest.raises(ValueError, match="type code"):
        mnist_data.read_idx(tmp_path / "x.gz")


def test_read_idx_rejects_a_payload_that_disagrees_with_its_header(tmp_path: Path) -> None:
    # The header promises 3x2x3 = 18 bytes and 17 follow.
    header = b"\x00\x00\x08\x03" + struct.pack(">3I", 3, 2, 3)
    (tmp_path / "x.gz").write_bytes(gzip.compress(header + bytes(17)))

    with pytest.raises(ValueError, match="payload"):
        mnist_data.read_idx(tmp_path / "x.gz")


# --------------------------------------------------------------------------
# Checksums and the files on disk
# --------------------------------------------------------------------------


def test_verify_rejects_a_wrong_checksum_and_leaves_the_file(tmp_path: Path) -> None:
    write_idx(tmp_path / "x.gz", images())

    with pytest.raises(ValueError, match="checksum mismatch"):
        mnist_data.verify(tmp_path / "x.gz", "0" * 64)

    assert (tmp_path / "x.gz").is_file()


def test_load_reads_four_files_into_scaled_flat_images(tmp_path: Path) -> None:
    sums = fake_dataset(tmp_path)

    data = mnist_data.load(tmp_path, checksums=sums)

    assert data.train_x.shape == (3, 6) and data.train_x.dtype == np.float32
    assert data.test_x.shape == (2, 6)
    # Row-major flatten of image 1, and 255 maps to 1.
    assert np.allclose(data.train_x[1], np.arange(6, 12) * 10 / 255.0)
    assert np.array_equal(data.train_labels, [7, 0, 3])
    assert np.array_equal(data.test_labels, [0, 1])


def test_load_says_how_to_fetch_the_data_when_it_is_missing(tmp_path: Path) -> None:
    assert not mnist_data.is_present(tmp_path)

    with pytest.raises(FileNotFoundError, match="mnist_data.py"):
        mnist_data.load(tmp_path)


def test_load_rejects_a_file_that_fails_its_checksum(tmp_path: Path) -> None:
    sums = fake_dataset(tmp_path)
    sums[mnist_data.FILES["test_labels"]] = "0" * 64

    with pytest.raises(ValueError, match="checksum mismatch"):
        mnist_data.load(tmp_path, checksums=sums)


def test_load_rejects_images_and_labels_of_different_lengths(tmp_path: Path) -> None:
    sums = fake_dataset(tmp_path, test_labels=5)

    with pytest.raises(ValueError, match="2 images but 5 labels"):
        mnist_data.load(tmp_path, checksums=sums)


# --------------------------------------------------------------------------
# Downloading, against file:// mirrors
# --------------------------------------------------------------------------


def test_download_fetches_every_file_from_a_mirror(tmp_path: Path) -> None:
    sums = fake_dataset(tmp_path / "mirror")

    mnist_data.download(
        tmp_path / "data", mirrors=[(tmp_path / "mirror").as_uri() + "/"], checksums=sums
    )

    assert mnist_data.is_present(tmp_path / "data")
    assert len(mnist_data.load(tmp_path / "data", checksums=sums).train_x) == 3


def test_download_falls_back_to_the_next_mirror(tmp_path: Path) -> None:
    sums = fake_dataset(tmp_path / "good")
    (tmp_path / "empty").mkdir()

    mnist_data.download(
        tmp_path / "data",
        mirrors=[(tmp_path / "empty").as_uri() + "/", (tmp_path / "good").as_uri() + "/"],
        checksums=sums,
    )

    assert mnist_data.is_present(tmp_path / "data")


def test_download_never_lands_a_file_that_fails_its_checksum(tmp_path: Path) -> None:
    sums = fake_dataset(tmp_path / "mirror")
    sums = {name: "0" * 64 for name in sums}

    with pytest.raises(RuntimeError, match="could not download"):
        mnist_data.download(
            tmp_path / "data", mirrors=[(tmp_path / "mirror").as_uri() + "/"], checksums=sums
        )

    # Neither the file nor the temporary it was fetched into.
    assert list((tmp_path / "data").iterdir()) == []


# --------------------------------------------------------------------------
# One-hot and batching
# --------------------------------------------------------------------------


def test_one_hot_puts_a_single_one_in_each_row() -> None:
    got = mnist_data.one_hot(np.array([2, 0, 9], dtype=np.uint8))

    assert got.shape == (3, 10) and got.dtype == np.float32
    assert np.array_equal(got.argmax(axis=1), [2, 0, 9])
    assert np.array_equal(got.sum(axis=1), [1, 1, 1])


def epoch(n: int, batch_size: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Batches of a dataset where row i of x is [i, i] and y[i] is -i."""
    x = np.repeat(np.arange(n, dtype=np.float32)[:, None], 2, axis=1)
    y = -np.arange(n, dtype=np.float32)[:, None]
    return list(mnist_data.batches(x, y, batch_size, np.random.default_rng(seed)))


def test_batches_cover_every_row_exactly_once() -> None:
    seen = np.concatenate([x[:, 0] for x, _ in epoch(50, 8, seed=0)])

    assert sorted(seen) == list(range(50))


def test_batches_keep_each_image_with_its_own_label() -> None:
    """The one that matters. Shuffling x and y separately passes everything
    else in this file and trains a model on noise."""
    for x, y in epoch(50, 8, seed=0):
        assert np.array_equal(x[:, 0], -y[:, 0])


def test_batches_are_shuffled() -> None:
    first = np.concatenate([x[:, 0] for x, _ in epoch(50, 8, seed=0)])

    assert not np.array_equal(first, np.arange(50))


def test_the_short_last_batch_is_kept() -> None:
    sizes = [len(x) for x, _ in epoch(50, 8, seed=0)]

    assert sizes == [8, 8, 8, 8, 8, 8, 2]


def test_the_same_seed_gives_the_same_order() -> None:
    a = np.concatenate([x[:, 0] for x, _ in epoch(50, 8, seed=3)])
    b = np.concatenate([x[:, 0] for x, _ in epoch(50, 8, seed=3)])

    assert np.array_equal(a, b)


def test_two_epochs_from_one_generator_differ() -> None:
    """One generator carried across epochs, so each epoch sees a new order."""
    x = np.arange(50, dtype=np.float32)[:, None]
    rng = np.random.default_rng(0)

    first = np.concatenate([b for b, _ in mnist_data.batches(x, x, 8, rng)])
    second = np.concatenate([b for b, _ in mnist_data.batches(x, x, 8, rng)])

    assert not np.array_equal(first, second)
