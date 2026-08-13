"""Micro-PyTorch autograd engine.

Phase 4: this package is the compiled core and nothing else. `_core` holds the
C++ Tensor and the forward kernels; the autograd graph arrives in phase 5.

`_core.Tensor` is deliberately NOT re-exported as `autograd.Tensor`. That name
is reserved for the phase 5 graph type, which will *hold* a `_core.Tensor`
rather than be one. Reach the raw type as `autograd._core.Tensor`. Deciding
this now costs a comment; deciding it in phase 5 costs a rename across every
test.

This file is copied into whichever build tree produced `_core`, so every tree
holds a complete, importable package. Which tree ends up on sys.path is decided
by the environment, never by pytest config - see the note in pyproject.toml.
"""

from . import _core

__all__ = ["_core"]
