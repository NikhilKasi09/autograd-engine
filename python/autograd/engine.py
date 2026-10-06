"""The node type, and the backward pass that walks the nodes."""

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

    # parents    - the graph Tensors this node consumed, in argument order.
    # needs_grad - one bool per parent: did it require grad when the op ran?
    # saved   - RAW _core.Tensors kept for backward. Never graph Tensors: saving
    #           a graph Tensor is the one move that would create a forward edge,
    #           and phase 6's relu will want to save its own output.

    @classmethod
    def apply(cls, *inputs: Tensor, **params) -> Tensor:
        """Run the op, and record a node if any input requires grad.

        Positional means differentiable: every one is a graph Tensor, lands in
        parents, and receives a gradient. Keyword means configuration - a target
        shape, an axis - which goes to the op's constructor and is never seen by
        the walk. The alternative, smuggling a shape in as a tensor argument and
        returning None for it, uses the non-differentiable-input path for
        something that is not an input.

        Function declares no __init__, so an op that takes no configuration
        rejects a stray keyword with a TypeError at the call.
        """
        raw_inputs = tuple(t.data for t in inputs)

        if any(t.requires_grad for t in inputs):
            node = cls(**params)
            # Read once, here. backward uses it to skip work for a parent that
            # wants no gradient, and the walk uses it to skip the parent.
            node.needs_grad = tuple(t.requires_grad for t in inputs)
            raw_result = node.forward(*raw_inputs)
            result = Tensor(raw_result, requires_grad=True)
            result.grad_fn = node
            node.parents = inputs
            return result
        else:
            raw_result = cls(**params).forward(*raw_inputs)
            return Tensor(raw_result)

    def forward(self, *raw: _core.Tensor) -> _core.Tensor:
        """Compute the result from raw tensors.

        Usually that means allocating an output with _core.zeros_like or
        _core.zeros and calling the out-parameter kernel. A pure reshaping op
        may instead return a VIEW that shares its input's storage - Expand does,
        because materialising a broadcast is the allocation per forward pass
        that stride 0 exists to avoid. Nothing in the engine writes through an
        input, so sharing is safe; it is only worth knowing that an op output is
        no longer guaranteed to own its buffer.

        Save whatever backward will need onto self.saved here.
        """
        raise NotImplementedError

    def backward(self, grad_out: _core.Tensor) -> tuple[_core.Tensor | None, ...]:
        """Gradient of the output w.r.t. each parent, in parents order.

        Return None for a parent that gets no gradient: either it cannot have
        one, like the target of a loss, or self.needs_grad says it does not
        want one. The second is optional - returning a gradient nobody needs is
        wasted work, never a wrong answer, so skip it where computing it costs
        something. Matmul's is a whole GEMM; Add's is free.

        The walk drops that parent and everything only reachable through it.
        """
        raise NotImplementedError


def topological_sort(root: Tensor) -> list[Tensor]:
    visited = set()
    res = []
    stack = [(root, False)]
    while stack:
        node, processed = stack.pop()
        if processed:
            res.append(node)
            continue
        if node in visited:
            continue
        visited.add(node)
        stack.append((node, True))
        if node.grad_fn is not None:
            for parent in node.grad_fn.parents:
                stack.append((parent, False))
    return list(reversed(res))


def backward(root: Tensor, gradient: _core.Tensor | None = None) -> None:
    """Accumulate gradients from root back to every leaf that asked for one."""
    # Check gradients
    if not root.requires_grad:
        raise RuntimeError("Error: root must require gradient")

    if gradient is None and root.data.numel() != 1:
        raise RuntimeError("Error: root must be a scalar if no gradient exists")

    # Root gradient must be one - creates tensor of all 0s except a single 1 in the correct position (seed)
    if gradient is None:
        rank = len(root.shape)
        i = (0,) * rank
        seed = _core.zeros_like(root.data)
        seed[i] = 1.0
        gradient = seed

    sorted_nodes = topological_sort(root)
    grads = {root: gradient}

    # accumulation helper

    def accumulate(grads: dict, tensor: Tensor, contribution: _core.Tensor) -> None:
        if tensor not in grads:
            grads[tensor] = _core.zeros_like(tensor.data)
        _core.add_into(grads[tensor], contribution)

    for node in sorted_nodes:
        # Nothing reached this node: every path to it went through an input
        # some op declared non-differentiable. Nothing to pass on either.
        grad = grads.get(node)
        if grad is None:
            continue

        if not node.is_leaf: # The node is not a leaf, it was created by add or mul so needs to propogate back
            fn = node.grad_fn
            for parent, wanted, contribution in zip(fn.parents, fn.needs_grad, fn.backward(grad)):
                # An op may hand back a gradient it got for free. Declining it
                # here is what saves the buffer it would be collected into.
                if wanted and contribution is not None:
                    accumulate(grads, parent, contribution)

        # Set the gradient on leafs that need it
        if (node.is_leaf and node.requires_grad) or node.retains_grad:
            if node.grad is None:
                node.grad = _core.zeros_like(node.data)
            _core.add_into(node.grad, grad)
