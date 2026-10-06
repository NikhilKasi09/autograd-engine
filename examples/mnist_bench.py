"""Where a training step goes, and how long an epoch takes against PyTorch.

    python examples/mnist_bench.py --profile
    python examples/mnist_bench.py --compare
    python examples/mnist_bench.py --compare --other-root ../autograd-base/python

--profile wraps every _core function in a wall-clock timer and reports what a
step spends where. --compare times whole epochs, each variant in its own
process, alternating, and reports medians and ranges.

Noise on this machine reaches 20% between sessions, so a number here means
nothing next to one from another day. The only comparison worth making is
between variants that alternated in a single --compare run.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

EXAMPLES = Path(__file__).resolve().parent
RELEASE_ROOT = EXAMPLES.parent / "python"


# --------------------------------------------------------------------------
# --profile
# --------------------------------------------------------------------------

# Every _core function the graph, the loss and the optimiser call.
FUNCTIONS = (
    "from_numpy", "zeros", "zeros_like", "add", "mul", "scale", "relu", "exp", "log",
    "reduce_max", "add_into", "relu_backward", "sum_into", "gemm",
)

# The Tensor methods that can copy. transpose and expand only build views.
METHODS = ("contiguous", "clone")


def dims(t) -> str:
    return "x".join(str(n) for n in t.shape)


class Recorder:
    """Seconds and call counts, keyed by (function, what it was called on)."""

    def __init__(self) -> None:
        self.seconds: dict[tuple[str, str], float] = defaultdict(float)
        self.calls: dict[tuple[str, str], int] = defaultdict(int)

    def wrap(self, name: str, fn):
        def timed(*args, **kwargs):
            start = time.perf_counter()
            out = fn(*args, **kwargs)
            elapsed = time.perf_counter() - start

            key = (name, self.describe(name, args, out))
            self.seconds[key] += elapsed
            self.calls[key] += 1
            return out
        return timed

    @staticmethod
    def describe(name: str, args: tuple, out) -> str:
        """The shapes involved, so one function at two sizes is two rows."""
        if name == "gemm":
            return f"{dims(args[0])} @ {dims(args[1])}"
        if name == "contiguous":
            # contiguous() on an already contiguous tensor hands back a view.
            copied = not out.shares_storage_with(args[0])
            return f"{dims(args[0])}, {'copied' if copied else 'no copy'}"
        if name in ("zeros", "from_numpy"):
            return dims(out)
        return dims(args[0])


def profile(steps: int, warmup: int) -> None:
    import autograd  # noqa: F401 - imported for the package, patched through _core
    from autograd import _core, optim

    import mnist_data
    import mnist_train

    data = mnist_data.load()
    onehot = mnist_data.one_hot(data.train_labels)

    def run(count: int, on_warm=None) -> float:
        """Train from scratch, timing `count` steps after the warm-up ones.

        Returns seconds per step, shuffle excluded. on_warm runs once, at the
        boundary between warming up and timing.
        """
        model = mnist_train.MLP(_core.Generator(0))
        opt = optim.SGD(model.parameters(), lr=mnist_train.LEARNING_RATE)
        batches = mnist_data.batches(data.train_x, onehot, mnist_train.BATCH_SIZE,
                                     np.random.default_rng(0))
        total = 0.0
        for step, (xb, yb) in enumerate(batches):
            if step == warmup + count:
                break
            if step == warmup and on_warm is not None:
                on_warm()
            start = time.perf_counter()
            opt.zero_grad()
            loss = mnist_train.nn.cross_entropy(model(mnist_train.tensor(xb)), mnist_train.tensor(yb))
            loss.backward()
            opt.step()
            if step >= warmup:
                total += time.perf_counter() - start
        return total / count

    # Unwrapped first: this is the real step time.
    bare = run(steps)

    recorder = Recorder()
    originals = {name: getattr(_core, name) for name in FUNCTIONS}
    methods = {name: getattr(_core.Tensor, name) for name in METHODS}
    try:
        for name, fn in originals.items():
            setattr(_core, name, recorder.wrap(name, fn))
        for name, fn in methods.items():
            setattr(_core.Tensor, name, recorder.wrap(name, fn))

        def forget() -> None:
            # Drop what the warm-up steps recorded, so both runs time the same steps.
            recorder.seconds.clear()
            recorder.calls.clear()

        wrapped = run(steps, on_warm=forget)
    finally:
        for name, fn in originals.items():
            setattr(_core, name, fn)
        for name, fn in methods.items():
            setattr(_core.Tensor, name, fn)

    us = {key: 1e6 * s / steps for key, s in recorder.seconds.items()}
    per_step = {key: n / steps for key, n in recorder.calls.items()}
    step_us = 1e6 * wrapped
    kernels = sum(us.values())

    print(f"module: {_core.__file__}")
    print(f"step, unwrapped: {1e6 * bare:8.1f} us")
    print(f"step, wrapped:   {step_us:8.1f} us   (the timers themselves cost the difference)")
    print()

    print(f"{'by function':<34}{'calls/step':>11}{'us/step':>10}{'share':>8}")
    by_name: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for (name, _), value in us.items():
        by_name[name][0] += value
    for (name, _), value in per_step.items():
        by_name[name][1] += value
    for name, (value, count) in sorted(by_name.items(), key=lambda kv: -kv[1][0]):
        print(f"{name:<34}{count:>11.1f}{value:>10.1f}{value / step_us:>8.1%}")
    rest = step_us - kernels
    print(f"{'Python, between the kernel calls':<34}{'':>11}{rest:>10.1f}{rest / step_us:>8.1%}")
    print()

    print(f"{'largest single items':<34}{'calls/step':>11}{'us/step':>10}{'share':>8}")
    for key, value in sorted(us.items(), key=lambda kv: -kv[1])[:14]:
        label = f"{key[0]} {key[1]}"
        print(f"{label:<34}{per_step[key]:>11.1f}{value:>10.1f}{value / step_us:>8.1%}")


# --------------------------------------------------------------------------
# --compare
# --------------------------------------------------------------------------


def worker(kind: str, epochs: int, threads: int | None) -> None:
    """Runs in a child process: train, print the epoch times as JSON."""
    import mnist_data

    data = mnist_data.load()
    if kind == "torch":
        import torch

        import mnist_torch
        result = mnist_torch.run(data, seed=0, epochs=epochs, threads=threads)
        where, used = torch.__version__, torch.get_num_threads()
    else:
        from autograd import _core

        import mnist_train
        result = mnist_train.run(data, seed=0, epochs=epochs)
        where, used = _core.__file__, 1

    print(json.dumps({"seconds": result.seconds, "accuracy": result.accuracy,
                      "where": where, "threads": used}))


def launch(kind: str, root: Path, epochs: int, threads: int | None) -> dict:
    """One child process, with `root` deciding which build of the engine it imports."""
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", kind,
               "--epochs", str(epochs)]
    if threads is not None:
        command += ["--threads", str(threads)]

    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root), str(EXAMPLES)]))
    out = subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    report = json.loads(out.stdout.strip().splitlines()[-1])

    # Check the module that was imported, not the path that was asked for.
    if kind == "ours" and Path(report["where"]).resolve().parent.parent != root.resolve():
        raise RuntimeError(f"asked for {root} but the worker imported {report['where']}")
    return report


def compare(runs: int, epochs: int, other_root: Path | None) -> None:
    variants: list[tuple[str, str, Path, int | None]] = [("this engine", "ours", RELEASE_ROOT, None)]
    if other_root is not None:
        variants.append((f"engine at {other_root}", "ours", other_root, None))
    variants += [("torch, 1 thread", "torch", RELEASE_ROOT, 1),
                 ("torch, all threads", "torch", RELEASE_ROOT, None)]

    times: dict[str, list[float]] = defaultdict(list)
    reports: dict[str, dict] = {}
    for _ in range(runs):
        # One of each per round, so slow drift in the machine hits all alike.
        for label, kind, root, threads in variants:
            report = launch(kind, root, epochs, threads)
            # A run's figure is its median epoch; the first includes warm-up.
            times[label].append(statistics.median(report["seconds"]))
            reports[label] = report

    base = times["this engine"]
    print(f"{runs} runs of {epochs} epochs each, alternating. Seconds per epoch.")
    print(f"{'':<34}{'median':>8}{'min':>8}{'max':>8}{'vs this':>9}  overlaps this engine")
    for label, _, _, _ in variants:
        t = times[label]
        overlap = min(t) <= max(base) and min(base) <= max(t)
        ratio = statistics.median(t) / statistics.median(base)
        detail = f"{reports[label]['threads']} threads" if label.startswith("torch") else ""
        print(f"{label:<34}{statistics.median(t):>8.3f}{min(t):>8.3f}{max(t):>8.3f}"
              f"{ratio:>8.2f}x  {'yes' if overlap else 'no':<4}{detail}")
    print(f"test accuracy after {epochs} epochs: " + ", ".join(
        f"{label} {reports[label]['accuracy'][-1]:.4f}" for label, _, _, _ in variants))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--steps", type=int, default=400, help="--profile: steps to time")
    parser.add_argument("--warmup", type=int, default=50, help="--profile: steps before timing")
    parser.add_argument("--runs", type=int, default=7, help="--compare: runs per variant")
    parser.add_argument("--epochs", type=int, default=3, help="--compare: epochs per run")
    parser.add_argument("--other-root", type=Path, default=None,
                        help="--compare: a second build's package root to time alongside")
    parser.add_argument("--worker", choices=("ours", "torch"), help=argparse.SUPPRESS)
    parser.add_argument("--threads", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker(args.worker, args.epochs, args.threads)
    elif args.profile:
        profile(args.steps, args.warmup)
    elif args.compare:
        compare(args.runs, args.epochs, args.other_root)
    else:
        parser.error("pick --profile or --compare")


if __name__ == "__main__":
    main()
