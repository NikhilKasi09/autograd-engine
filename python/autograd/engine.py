"""The node type. One Function instance per operation performed."""

from __future__ import annotations

from . import _core
from .tensor import Tensor


class Function:
    """A node in the graph: what produced a tensor, and what it was made from.

    The INSTANCE is the node. PyTorch splits a stateless Function from a
    per-call ctx; here one object holds both, so there is exactly one thing to
    inspect when a graph is wrong.

    Every edge points BACKWARD - a tensor holds its grad_fn, a grad_fn holds its
    parents, and nothing holds a result. That is what lets a dropped graph go by
    refcount alone, with no cycle collector involved.
    """

    # parents - the graph Tensors this node consumed, in argument order.
    # saved   - RAW _core.Tensors kept for backward. Never graph Tensors: saving
    #           a graph Tensor is the one move that would create a forward edge,
    #           and phase 6's relu will want to save its own output.

    @classmethod
    def apply(cls, *inputs: Tensor) -> Tensor:
        """Run the op, and record a node if any input requires grad."""
        raw_inputs = tuple(t.data for t in inputs)

        if any(t.requires_grad for t in inputs):
            node = cls()
            raw_result = node.forward(*raw_inputs)
            result = Tensor(raw_result, requires_grad=True)
            result.grad_fn = node
            node.parents = inputs
            return result
        else:
            raw_result = cls().forward(*raw_inputs)
            return Tensor(raw_result)

    def forward(self, *raw: _core.Tensor) -> _core.Tensor:
        """Compute the result from raw tensors, allocating its own output.

        Allocate with _core.zeros_like and call the out-parameter kernel. Save
        whatever backward will need onto self.saved here.
        """
        raise NotImplementedError

    def backward(self, grad_out: _core.Tensor) -> tuple[_core.Tensor | None, ...]:
        """Gradient of the output w.r.t. each parent, in parents order.

        Total: return a gradient for every parent regardless of its
        requires_grad. The engine decides who receives one. None is reserved for
        genuinely non-differentiable inputs, which phase 6 has and phase 5 does
        not.
        """
        raise NotImplementedError
