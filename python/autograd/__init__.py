"""Micro-PyTorch autograd engine.

Two layers. `_core` is the compiled half: the C++ `Tensor`, its views, the
elementwise kernels and the six GEMM kernels. Around it sits the graph - an
`autograd.Tensor` that *holds* a `_core.Tensor`, a `Function` node per
operation, and a `backward()` that walks the graph in reverse topological order.

`_core.Tensor` is deliberately NOT re-exported. `autograd.Tensor` is the graph
type; reach the raw one as `autograd._core.Tensor` when you want a buffer
without a node attached.

This file is copied into whichever build tree produced `_core`, so every tree
holds a complete, importable package. Which tree ends up on sys.path is decided
by the environment, never by pytest config - see the note in pyproject.toml.
"""

from . import _core
from .ops import add, mul
from .tensor import Tensor

__all__ = ["_core", "Tensor", "add", "mul"]
