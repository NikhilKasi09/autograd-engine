"""MNIST: fetch it, check it, parse it, cut it into batches.

Everything here is NumPy and the standard library. It sits outside the autograd
package on purpose - the engine computes nothing with NumPy, and loading a
dataset is not the engine's job.

Run this file to download the data:  python examples/mnist_data.py
"""

from __future__ import annotations

import gzip
import hashlib
import os
import struct
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator, Mapping, NamedTuple, Sequence

import numpy as np

# Not committed: data/ is in .gitignore.
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "mnist"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

# Tried in order. Both were checked to serve byte-identical files.
MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)

# SHA-256 of each .gz as downloaded.
SHA256 = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}


class Dataset(NamedTuple):
    """Images as float32 {n, 784} in [0, 1]; labels as uint8 {n}."""

    train_x: np.ndarray
    train_labels: np.ndarray
    test_x: np.ndarray
    test_labels: np.ndarray


def verify(path: Path, expected: str) -> None:
    """Raise ValueError unless the file's SHA-256 is `expected`. Never deletes."""
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"{path.name}: checksum mismatch (got {actual}, expected {expected})")


def read_idx(path: Path) -> np.ndarray:
    """Parse one gzipped IDX file into a uint8 array of the shape its header gives.

    The header is two zero bytes, a type code, a rank, then one BIG-endian
    uint32 per dimension. The data follows, row-major.
    """
    raw = gzip.open(path, "rb").read()

    if len(raw) < 4 or raw[0] != 0 or raw[1] != 0:
        raise ValueError(f"{path.name}: not an IDX file (bad magic)")

    # 0x08 is unsigned byte, the only type MNIST uses.
    if raw[2] != 0x08:
        raise ValueError(f"{path.name}: unsupported IDX type code {raw[2]:#04x}")

    rank = raw[3]
    header = 4 + 4 * rank
    if rank == 0 or len(raw) < header:
        raise ValueError(f"{path.name}: truncated IDX header")

    # '>' is big-endian. Reading these little-endian turns 60000 into 1625948160.
    shape = struct.unpack(f">{rank}I", raw[4:header])

    if len(raw) - header != int(np.prod(shape, dtype=np.int64)):
        raise ValueError(
            f"{path.name}: header says {shape} but the payload is {len(raw) - header} bytes"
        )

    # frombuffer gives a read-only view of `raw`; copy so the caller owns it.
    return np.frombuffer(raw, dtype=np.uint8, offset=header).reshape(shape).copy()


def flatten(images: np.ndarray) -> np.ndarray:
    """uint8 {n, h, w} -> float32 {n, h*w}, scaled from [0, 255] to [0, 1]."""
    return images.reshape(len(images), -1).astype(np.float32) / np.float32(255.0)


def one_hot(labels: np.ndarray, classes: int = 10) -> np.ndarray:
    """Labels {n} -> float32 {n, classes}, a single 1 per row.

    The loss takes one-hot rows because the engine has no integer tensor.
    """
    out = np.zeros((len(labels), classes), dtype=np.float32)
    out[np.arange(len(labels)), labels] = 1.0
    return out


def is_present(data_dir: Path = DATA_DIR) -> bool:
    """True when all four files are there. Does not check their contents."""
    return all((data_dir / name).is_file() for name in FILES.values())


def download(
    data_dir: Path = DATA_DIR,
    *,
    mirrors: Sequence[str] = MIRRORS,
    checksums: Mapping[str, str] = SHA256,
) -> None:
    """Fetch any missing file, trying each mirror in turn.

    A file only appears under its real name after its checksum passes, so a
    half-finished or corrupt download is never mistaken for the data.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    for name in FILES.values():
        target = data_dir / name
        if target.is_file():
            continue

        part = data_dir / (name + ".part")
        errors = []
        for mirror in mirrors:
            try:
                with urllib.request.urlopen(mirror + name, timeout=60) as response:
                    part.write_bytes(response.read())
                verify(part, checksums[name])
            except (urllib.error.URLError, OSError, ValueError) as exc:
                errors.append(f"{mirror}: {exc}")
                part.unlink(missing_ok=True)   # our own temporary, nobody else's file
                continue

            os.replace(part, target)
            break
        else:
            raise RuntimeError(f"could not download {name}:\n  " + "\n  ".join(errors))


def load(data_dir: Path = DATA_DIR, *, checksums: Mapping[str, str] = SHA256) -> Dataset:
    """Verify and parse the four files. Does not download."""
    if not is_present(data_dir):
        raise FileNotFoundError(
            f"no MNIST files in {data_dir}. Fetch them with:\n"
            "    python examples/mnist_data.py"
        )

    arrays = {}
    for key, name in FILES.items():
        verify(data_dir / name, checksums[name])
        arrays[key] = read_idx(data_dir / name)

    # An image file and its label file are separate downloads; nothing but
    # this ties them together.
    for split in ("train", "test"):
        n_images, n_labels = len(arrays[f"{split}_images"]), len(arrays[f"{split}_labels"])
        if n_images != n_labels:
            raise ValueError(f"{split}: {n_images} images but {n_labels} labels")

    return Dataset(
        flatten(arrays["train_images"]),
        arrays["train_labels"],
        flatten(arrays["test_images"]),
        arrays["test_labels"],
    )


def batches(
    x: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """One shuffled pass over (x, y) in batches. The last one may be short.

    ONE permutation indexes both arrays. Shuffling each with its own would
    still give well-formed batches, of images paired with the wrong labels.
    """
    order = rng.permutation(len(x))
    for start in range(0, len(x), batch_size):
        idx = order[start:start + batch_size]
        yield x[idx], y[idx]


if __name__ == "__main__":
    download()
    data = load()
    print(f"train: {data.train_x.shape}, test: {data.test_x.shape}")
    print("train labels per digit:", *np.bincount(data.train_labels, minlength=10))
