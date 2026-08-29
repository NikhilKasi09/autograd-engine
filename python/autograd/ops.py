"""The differentiable operations. One Function subclass, one free function each.

Free functions rather than operators: every call site names the op that builds
the node, which is what you want while reading a graph back.

Two ops in phase 5, chosen so that nothing here needs a new C++ kernel. Add
needs no saved state; Mul needs both operands, so between them they cover the
node shapes the engine has to handle. matmul, relu and sum land in phase 6.
"""

from __future__ import annotations

from . import _core
from .engine import Function
from .tensor import Tensor


class Add(Function):
    """out = a + b. Saves nothing - the gradient does not depend on the inputs."""

    def forward(self, a: _core.Tensor, b: _core.Tensor) -> _core.Tensor:
        out = _core.zeros_like(a)   # allocate a same-shape buffer, zeroed
        _core.add(a, b, out)        # fill it — the kernel writes into `out`
        return out

    def backward(self, grad_out: _core.Tensor) -> tuple[_core.Tensor | None, ...]:
        """d(a+b)/da and d(a+b)/db are both 1, so grad_out passes straight
        through to each parent."""
        return (grad_out, grad_out)


class Mul(Function):
    """out = a * b, elementwise. Saves both operands: each one's gradient is the
    other times grad_out."""

    def forward(self, a: _core.Tensor, b: _core.Tensor) -> _core.Tensor:
        self.saved = (a, b)
        out = _core.zeros_like(a)   # allocate a same-shape buffer, zeroed
        _core.mul(a, b, out)        # fill it — the kernel writes into `out`
        return out

    def backward(self, grad_out: _core.Tensor) -> tuple[_core.Tensor | None, ...]:
        # Written in step 4.
        raise NotImplementedError


def add(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise sum. Shapes must match exactly; no implicit broadcasting."""
    return Add.apply(a, b)


def mul(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise product. Not a matrix product - that arrives in phase 6."""
    return Mul.apply(a, b)
