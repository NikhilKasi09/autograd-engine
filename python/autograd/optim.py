"""Optimisers: what turns a gradient into a changed parameter."""

from __future__ import annotations

from typing import Iterable

from . import _core
from .tensor import Tensor


class SGD:
    """Plain gradient descent: p -= lr * p.grad. No momentum, no weight decay."""

    def __init__(self, params: Iterable[Tensor], lr: float) -> None:
        # A list, because Module.parameters() is a generator. Holding the
        # generator itself would work for one step and silently do nothing
        # on every step after it.
        self.params = list(params)
        self.lr = lr

    def step(self) -> None:
        """Move every parameter one step down its gradient.

        Writes into the parameter's own buffer, so anything else holding that
        buffer sees the update. A parameter with no gradient is left alone -
        zero_grad() sets None, and an unused parameter is not an error.
        """
        for p in self.params:
            if p.grad is None:
                continue

            # -lr * grad into a scratch buffer, then added on. Scaling p.grad
            # in place would save the allocation and corrupt the gradient.
            update = _core.zeros_like(p.grad)
            _core.scale(p.grad, -self.lr, update)
            _core.add_into(p.data, update)

    def zero_grad(self) -> None:
        """Drop every gradient. Call between steps: backward() accumulates."""
        for p in self.params:
            p.zero_grad()
