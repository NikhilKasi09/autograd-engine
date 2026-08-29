"""The graph tensor: a `_core.Tensor` plus the slots the backward pass needs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _core

# Only for the annotation on grad_fn.
if TYPE_CHECKING:
    from .engine import Function


class Tensor:
    """A node in the autograd graph. Holds a `_core.Tensor`; is not one."""

    # data          - the buffer, shape and strides. Never None.
    # grad          - accumulated gradient, a raw _core.Tensor; None until a pass writes one.
    # requires_grad - whether a backward pass should accumulate into this.
    # grad_fn       - the Function that produced this tensor, or None for a leaf.
    # retains_grad  - set by retain_grad(); makes a non-leaf keep its gradient.

    def __init__(self, data: _core.Tensor, requires_grad: bool = False) -> None:
        """Wrap `data` as a leaf. Does not copy: `self.data is data`."""
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self.grad_fn = None
        self.retains_grad = False

    @property
    def shape(self) -> tuple[int, ...]:
        """Extents, forwarded from the wrapped tensor."""
        return self.data.shape

    @property
    def is_leaf(self) -> bool:
        """True when no Function produced this tensor. Derived, never stored."""
        return self.grad_fn is None

    def zero_grad(self) -> None:
        """Drop the accumulated gradient. None, not a zeroed buffer."""
        self.grad = None

    def detach(self) -> Tensor:
        """A new node over the same buffer, outside the graph."""
        return Tensor(self.data, requires_grad=False)

    def retain_grad(self) -> None:
        """Ask a backward pass to populate `grad` on this non-leaf."""
        self.retains_grad = True

    def backward(self, gradient: _core.Tensor | None = None) -> None:
        """Accumulate gradients back to every requiring leaf. None seeds 1.0 on a scalar."""
        from .engine import backward
        backward(self, gradient)

    def to_numpy(self) -> np.ndarray:
        """An owning NumPy copy of the values."""
        return self.data.to_numpy()

    def __repr__(self) -> str:
        """Name the shape, requires_grad, and the producing op."""
        op = type(self.grad_fn).__name__ if self.grad_fn is not None else None
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}, grad_fn={op})"
