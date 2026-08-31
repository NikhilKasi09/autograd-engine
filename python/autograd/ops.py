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
        a, b = self.saved
        grad_a = _core.zeros_like(grad_out)
        _core.mul(grad_out, b, grad_a)
        grad_b = _core.zeros_like(grad_out)
        _core.mul(grad_out, a, grad_b)
        return (grad_a, grad_b)


def _gemm_into(
    a: _core.Tensor,
    b: _core.Tensor,
    out: _core.Tensor,
    *,
    ta: bool = False,
    tb: bool = False,
) -> None:
    """out += (a or a.T) @ (b or b.T). The graph's ONLY _core.gemm call site.

    Both inputs are materialised contiguous when they are not already - gemm
    rejects a non-unit inner stride on all three operands, and an incoming
    gradient is not guaranteed contiguous.

    `out` is NOT contiguified and raises instead. Contiguifying it would hand
    gemm a temporary copy, fill that, drop it, and leave the caller's buffer
    untouched with nothing raised anywhere.

    Everything transpose-related lives here so phase 9 can add gemm's
    transa/transb by editing one function rather than every call site in the
    graph. Until then a transposed operand costs an allocation and a copy.
    """

    if not out.is_contiguous():
        raise ValueError("Error: out must be contiguous")

    if ta:
        a = a.transpose(0, 1)

    if tb:
        b = b.transpose(0, 1)

    a = a.contiguous()
    b = b.contiguous()

    _core.gemm(a, b, out)


class Matmul(Function):
    """out = a @ b, a matrix product. Saves both operands.

    Shapes are the mirror of the forward and are easy to transpose by mistake:
    forward is {M,K} @ {K,N} -> {M,N}, so dA = dC @ b.T is {M,N} @ {N,K} and
    dB = a.T @ dC is {K,M} @ {M,N}. On a square case all four have the same
    shape and every wrong version still runs.
    """

    def forward(self, a: _core.Tensor, b: _core.Tensor) -> _core.Tensor:
        self.saved = (a, b)
        M, _ = a.shape
        _, N = b.shape
        out = _core.zeros([M, N])
        _gemm_into(a, b, out)
        return out

    def backward(self, grad_out: _core.Tensor) -> tuple[_core.Tensor | None, ...]:
        a, b = self.saved
        M, K = a.shape
        _, N = b.shape

        da = _core.zeros([M, K])
        _gemm_into(grad_out, b, da, tb=True)   # dA = grad_out @ b.T

        db = _core.zeros([K, N])
        _gemm_into(a, grad_out, db, ta=True)   # dB = a.T @ grad_out

        return (da, db)


class Relu(Function):
    """out = max(a, 0), elementwise. Saves its OWN OUTPUT to mask with.

    relu(x) is positive exactly where x is, so the output serves as the mask and
    the node keeps one buffer alive rather than two - which is also what torch
    does. What must be saved is the raw buffer, never the graph Tensor wrapping
    it: the wrapper holds this node, so saving the wrapper closes the loop and
    the graph stops being freeable by refcount alone. Every value test in the
    suite passes either way; only the lifetime test sees the difference.

    apply() builds that wrapper after forward returns, so inside forward the
    buffer is all there is - which is the structural reason this is hard to get
    wrong here, and the reason it is worth a test anyway.
    """

    def forward(self, a: _core.Tensor) -> _core.Tensor:
        out = _core.zeros_like(a)
        _core.relu(a, out)
        self.saved = (out,)
        return out

    def backward(self, grad_out: _core.Tensor) -> tuple[_core.Tensor | None, ...]:
        (saved_output,) = self.saved
        into = _core.zeros_like(grad_out)
        _core.relu_backward(grad_out, saved_output, into)
        return (into,)


def add(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise sum. Shapes must match exactly; no implicit broadcasting."""
    return Add.apply(a, b)


def mul(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise product. Not a matrix product - that is matmul."""
    return Mul.apply(a, b)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix product, {M,K} @ {K,N} -> {M,N}. Rank 2 only."""
    return Matmul.apply(a, b)


def relu(a: Tensor) -> Tensor:
    """Elementwise max(a, 0)."""
    return Relu.apply(a)
