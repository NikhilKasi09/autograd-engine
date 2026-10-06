"""Layers and losses: what turns a graph of ops into a model that can train.

A Module holds Parameters and nothing else. The arithmetic lives in free
functions - linear, mse_loss, cross_entropy - so it can be checked against
finite differences with fresh leaves, which a module holding fixed parameters
cannot be.
"""

from __future__ import annotations

import math
from typing import Iterator

from . import _core
from .engine import Function
from .ops import add, expand, matmul, mul, reduce_sum, scale
from .tensor import Tensor


class Parameter(Tensor):
    """A trainable leaf. The type is the marker Module.parameters() looks for."""

    def __init__(self, data: _core.Tensor, requires_grad: bool = True) -> None:
        """Same as Tensor, except requires_grad defaults to True."""
        super().__init__(data, requires_grad)


class Module:
    """Base class for anything that owns Parameters.

    There is no registry and no __setattr__ hook. Assign a Parameter or another
    Module to an attribute and parameters() finds it by looking.
    """

    def parameters(self) -> Iterator[Parameter]:
        """Every Parameter under this module, in assignment order, each once."""
        yield from self._parameters(set())

    def _parameters(self, seen: set[int]) -> Iterator[Parameter]:
        """The walk behind parameters().

        `seen` is shared down the whole recursion. A fresh set per module would
        not catch a parameter shared between two children, which is the only
        case a dedupe is for.
        """
        for value in vars(self).values():
            if isinstance(value, Parameter):
                if id(value) not in seen:
                    seen.add(id(value))
                    yield value
            elif isinstance(value, Module):
                yield from value._parameters(seen)

    def zero_grad(self) -> None:
        """Drop the gradient on every parameter."""
        for p in self.parameters():
            p.zero_grad()

    def forward(self, *args: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args: Tensor) -> Tensor:
        return self.forward(*args)


def linear(x: Tensor, w: Tensor, b: Tensor | None = None) -> Tensor:
    """x @ w + b, for x {M,K}, w {K,N} and b {1,N}.

    The bias is expanded to {M,N} here, with M read off x. That is a stride-0
    view, so no {M,N} copy of the bias is ever made.
    """
    out = matmul(x, w)
    if b is None:
        return out
    return add(out, expand(b, out.shape))


class Linear(Module):
    """A fully connected layer: weight {in, out}, bias {1, out}.

    The weight is stored {in, out}, the transpose of torch's {out, in}, so
    forward is a plain x @ w. Torch's layout would need w.T on every call, and
    gemm here has no transpose flag - that would be a copy per forward pass.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        generator: _core.Generator,
        bias: bool = True,
    ) -> None:
        """Weights uniform on +-1/sqrt(in_features), bias zero.

        Each output sums in_features terms, so its variance grows with
        in_features unless the weights shrink to match. This bound keeps it
        roughly constant across layer widths, and is torch's nn.Linear default.

        The bias starts at zero: random weights already break the symmetry
        between units, so a random bias adds nothing.
        """
        bound = 1.0 / math.sqrt(in_features)

        w = _core.zeros([in_features, out_features])
        generator.uniform(w, -bound, bound)
        self.weight = Parameter(w)

        # None rather than a zero tensor that never trains.
        self.bias = Parameter(_core.zeros([1, out_features])) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return linear(x, self.weight, self.bias)


def _check_reduction(reduction: str) -> None:
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")


def mse_loss(pred: Tensor, target: Tensor, *, reduction: str = "mean") -> Tensor:
    """Squared error between pred and target, averaged over every element.

    Built from ops that already exist, so there is no MSE node - backward runs
    through add, mul, sum and scale. The result keeps pred's rank with 1s, like
    sum: {1,1} for a rank-2 input.

    reduction="sum" skips the average.
    """
    _check_reduction(reduction)

    # pred - target. There is no subtract op; scale by -1 and add.
    diff = add(pred, scale(target, -1.0))

    # diff is both operands here, so it receives two gradient contributions
    # and the engine has to sum them.
    total = reduce_sum(mul(diff, diff))

    if reduction == "sum":
        return total
    return scale(total, 1.0 / math.prod(pred.shape))


class SoftmaxCrossEntropy(Function):
    """Softmax and cross-entropy as one node: logits {M,C}, targets {M,C}.

    One node rather than exp, log and divide ops chained together, because the
    chained version overflows on large logits and this one cannot. It also
    makes the gradient (softmax - target), with no exp or log to differentiate.

    Each target row must sum to 1 - one-hot, or soft labels. Saves the softmax
    probabilities and the target.
    """

    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = reduction

    def forward(self, logits: _core.Tensor, target: _core.Tensor) -> _core.Tensor:
        if len(logits.shape) != 2:
            raise ValueError("cross_entropy: logits must be rank 2, {batch, classes}")
        M, C = logits.shape

        # Mean over the batch, or not. Backward needs the same factor.
        self.factor = 1.0 / M if self.reduction == "mean" else 1.0

        # Shift each row so its largest logit is 0. Softmax does not change,
        # and now every exp below is of something <= 0 and cannot overflow.
        shift = _core.zeros([M, 1])
        _core.reduce_max(logits, shift)
        _core.scale(shift, -1.0, shift)
        z = _core.zeros([M, C])
        _core.add(logits, shift.expand([M, C]), z)

        # log of each row's sum of exps. The row's max contributes exp(0) = 1,
        # so the sum is at least 1 and the log is safe.
        e = _core.zeros([M, C])
        _core.exp(z, e)
        log_sum = _core.zeros([M, 1])
        _core.sum_into(log_sum, e)
        _core.log(log_sum, log_sum)

        # log-softmax = z - log_sum
        _core.scale(log_sum, -1.0, log_sum)
        logp = _core.zeros([M, C])
        _core.add(z, log_sum.expand([M, C]), logp)

        # Softmax itself, for backward. Reuses e, which is finished with.
        _core.exp(logp, e)
        self.saved = (e, target)

        # loss = -sum(target * logp), over the batch. Reuses z the same way.
        _core.mul(logp, target, z)
        out = _core.zeros([1, 1])
        _core.sum_into(out, z)
        _core.scale(out, -self.factor, out)
        return out

    def backward(self, grad_out: _core.Tensor) -> tuple[_core.Tensor | None, ...]:
        p, target = self.saved

        # softmax - target
        d = _core.zeros_like(p)
        _core.scale(target, -1.0, d)
        _core.add(p, d, d)

        # Times grad_out. It is 1 when the loss is the root, which is the only
        # case where leaving this out would go unnoticed.
        _core.mul(d, grad_out.expand(list(p.shape)), d)
        _core.scale(d, self.factor, d)

        # The target is data, not something to differentiate.
        return (d, None)


def cross_entropy(logits: Tensor, target: Tensor, *, reduction: str = "mean") -> Tensor:
    """Softmax cross-entropy of logits {M,C} against targets {M,C}, as {1,1}.

    Takes raw logits, not probabilities - the softmax is inside. Targets are
    one row per sample summing to 1, usually one-hot; there are no class
    indices because there is no integer tensor.

    Averaged over the batch; reduction="sum" skips the average. The target
    receives no gradient.
    """
    _check_reduction(reduction)
    return SoftmaxCrossEntropy.apply(logits, target, reduction=reduction)
