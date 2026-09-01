"""Finite-difference gradient checking against the engine's own forward pass.

Every other gradient test compares against a hand-derived NumPy expression - the
same calculus that produced the implementation, written out twice. Finite
differences don't know the calculus, so they agree only if the backward pass is
genuinely right.

CLAUDE.md asks for float64 at h=1e-5. There is no float64 here, and at h=1e-5 a
correct gradient measures 4e-2 wrong: subtracting two nearly-equal float32s
destroys the digits the answer lives in. h is squeezed from both sides - by
cancellation below (the floor goes as eps*|f|/h) and by the relu kink above (a
step across zero measures a slope that isn't there; 45% at margin 1e-2, h=1e-1).
h = 3e-3 sits between them.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

import autograd
from autograd import _core

# Takes freshly-built leaves, returns the output. Called once per perturbation,
# since a forward pass consumes its inputs.
Builder = Callable[[list[autograd.Tensor]], autograd.Tensor]

# Applied to the raw tensor before wrapping, so a leaf can be strided.
ViewFn = Callable[[_core.Tensor], _core.Tensor]

EPS_F32 = 6e-8          # half of float32 machine epsilon
GRADCHECK_H = 3e-3      # the only sweep column where every case lands under 1e-4
KINK_MARGIN = 1e-2      # must exceed h, or the step straddles the relu kink
SAFETY = 3.0            # measured floor was 0.26-1.04x the model, over 21 cases
FLOOR_MIN = 2e-5        # 3x the tightest floor seen (7.0e-6)
TOL_CEILING = 1e-3      # past this the check catches nothing; trips at |f| ~ 17


def tolerance(f_magnitude: float, h: float = GRADCHECK_H) -> float:
    """Honest tolerance for a function whose output reaches `f_magnitude`.

    The floor is set by |f| - the size of the numbers being subtracted - not by
    the op or the gradient. Summing N floats with an identically-1.0 gradient,
    the floor still grows 1.3e-5 -> 2.0e-3 as N goes 6 -> 6144. So this is
    computed, not looked up: a per-op constant would only hold for the shapes it
    was measured on.

    A gradient wrong by (1+e) produces exactly e relative error, so the number
    returned here is the smallest error the check can see.
    """
    tol = max(SAFETY * EPS_F32 * f_magnitude / h, FLOOR_MIN)
    if tol > TOL_CEILING:
        raise ValueError(
            f"gradcheck: |f|={f_magnitude:.3g} at h={h:.3g} needs tol={tol:.2e}, "
            f"over the {TOL_CEILING:.0e} ceiling. Shrink the case or the inputs; "
            f"do not loosen the bound."
        )
    return tol


def away_from_zero(a: np.ndarray, margin: float = KINK_MARGIN) -> np.ndarray:
    """Push values out of (-margin, margin), keeping sign. Every relu input.

    A uniform draw lands inside the step on about 1 seed in 200, so this removes
    a rare flake rather than catching anything. The subgradient at exactly 0 is
    unreachable by a symmetric difference either way, so the boundary is never
    sampled here.
    """
    sign = np.where(a >= 0.0, 1.0, -1.0)
    return (sign * np.maximum(np.abs(a), margin)).astype(np.float32)


def _as_list(x, n: int) -> list:
    """One value for all n, or one per leaf."""
    if x is None or isinstance(x, (bool, np.bool_)):
        return [x] * n
    out = list(x)
    if len(out) != n:
        raise ValueError(f"gradcheck: expected {n} entries, got {len(out)}")
    return out


def make_leaves(
    arrays: Sequence[np.ndarray],
    requires_grad: bool | Sequence[bool] = True,
    views: Sequence[ViewFn | None] | None = None,
) -> list[autograd.Tensor]:
    """Fresh leaves from float32 arrays, optionally as strided views.

    `arrays` are the source of truth and are never mutated - every evaluation
    rebuilds through from_numpy, so there is no in-place perturbation to undo.

    `views[k]` is applied to the raw tensor before wrapping, so the engine gets a
    transposed or sliced leaf. Mul, Matmul and Relu all read operand strides, and
    a harness that only ever built with from_numpy would never exercise that.
    """
    rg = _as_list(requires_grad, len(arrays))
    vs = _as_list(views, len(arrays))
    leaves = []
    for a, r, v in zip(arrays, rg, vs):
        raw = _core.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        if v is not None:
            raw = v(raw)
        leaves.append(autograd.Tensor(raw, requires_grad=bool(r)))
    return leaves


def _view_permutation(base_shape: tuple[int, ...], view: ViewFn | None) -> np.ndarray:
    """Map each view position to the base flat index it reads.

    Recovered by pushing arange through the same view rather than special-casing
    per view type, so transpose, slice and permute all work unchanged. Exact in
    float32 well past any size used here.
    """
    size = int(np.prod(base_shape))
    if view is None:
        return np.arange(size)
    probe = _core.from_numpy(np.arange(size, dtype=np.float32).reshape(base_shape))
    return np.asarray(view(probe)).ravel().astype(np.int64)


def _to_base_order(grad: np.ndarray, perm: np.ndarray, size: int) -> np.ndarray:
    """Scatter a view-ordered gradient back into base order.

    The engine returns zeros_like(leaf.data), which is shaped like the VIEW. The
    numerical side perturbs the base array. add.at rather than fancy assignment
    because a repeated index has to sum - that is the chain rule through a view
    that reads one element twice.
    """
    out = np.zeros(size, dtype=np.float64)
    np.add.at(out, perm, grad.ravel().astype(np.float64))
    return out


def forward_magnitude(
    f: Builder,
    arrays: Sequence[np.ndarray],
    views: Sequence[ViewFn | None] | None = None,
) -> float:
    """max |output| of one unperturbed pass. Feeds `tolerance`."""
    out = f(make_leaves(arrays, True, views))
    return float(np.max(np.abs(np.asarray(out.data))))


def numerical_jacobian(
    f: Builder,
    arrays: Sequence[np.ndarray],
    wrt: int,
    h: float = GRADCHECK_H,
    views: Sequence[ViewFn | None] | None = None,
) -> np.ndarray:
    """Central differences, shape (out.numel(), arrays[wrt].size).

    Perturb in float64 and cast to float32 so the stored value is the intended
    one. Reading the realised step back and dividing by that instead of 2h buys
    nothing - measured 5.10e-5 against 4.49e-5, which is noise.
    """
    cols = []
    for i in range(arrays[wrt].size):
        side = []
        for sign in (+1.0, -1.0):
            pert = [np.array(x, dtype=np.float64) for x in arrays]
            pert[wrt].reshape(-1)[i] += sign * h
            out = f(make_leaves([p.astype(np.float32) for p in pert], True, views))
            side.append(np.asarray(out.data).astype(np.float64).ravel().copy())
        cols.append((side[0] - side[1]) / (2.0 * h))
    return np.stack(cols, axis=1)


def analytic_jacobian(
    f: Builder,
    arrays: Sequence[np.ndarray],
    wrt: int,
    views: Sequence[ViewFn | None] | None = None,
) -> np.ndarray:
    """The engine's gradients, shape (out.numel(), arrays[wrt].size).

    Row j comes from seeding backward() with a basis vector. A fresh graph per
    row, not zero_grad() between rows - gradients accumulate here by design, and
    rebuilding makes row-to-row leakage impossible rather than merely avoided.
    """
    vs = _as_list(views, len(arrays))
    perm = _view_permutation(arrays[wrt].shape, vs[wrt])
    size = arrays[wrt].size

    probe = f(make_leaves(arrays, True, views))
    rows = []
    for j in range(int(np.prod(probe.shape))):
        leaves = make_leaves(arrays, True, views)
        out = f(leaves)
        seed = _core.zeros_like(out.data)
        seed[np.unravel_index(j, out.shape)] = 1.0
        out.backward(seed)
        rows.append(_to_base_order(np.asarray(leaves[wrt].grad), perm, size))
    return np.stack(rows, axis=0)


def max_relative_error(num: np.ndarray, ana: np.ndarray) -> float:
    """max |num - ana| / max(|ana|, |num|, 1.0).

    Mixed, not relative: below |grad| = 1 the floor turns it absolute. Kept
    because it removes the divide-by-zero (filterwarnings = ["error"] makes a
    NumPy warning a failure), and because on every case in the sweep it agreed
    with a scale-normalised metric to within 1.5x - these inputs give gradients
    of order 1, so the floor is currently inert.
    """
    denom = np.maximum(np.maximum(np.abs(ana), np.abs(num)), 1.0)
    return float(np.max(np.abs(num - ana) / denom))


def _report(num: np.ndarray, ana: np.ndarray, tol: float, label: str) -> None:
    """Assert, naming the worst element - a bare 'gradients differ' is useless."""
    err = max_relative_error(num, ana)
    if err > tol:
        flat = np.argmax(np.abs(num - ana) / np.maximum(np.maximum(np.abs(ana), np.abs(num)), 1.0))
        j, i = np.unravel_index(flat, num.shape)
        raise AssertionError(
            f"{label}: relative error {err:.3e} exceeds tol {tol:.3e}\n"
            f"  worst at output {j}, input {i}: "
            f"numerical {num[j, i]:.8g} vs analytic {ana[j, i]:.8g}"
        )


def check_jacobian(
    f: Builder,
    arrays: Sequence[np.ndarray],
    *,
    wrt: int | None = None,
    h: float = GRADCHECK_H,
    views: Sequence[ViewFn | None] | None = None,
    tol: float | None = None,
) -> None:
    """Assert the full Jacobian matches finite differences.

    The complete matrix, not a contraction - nothing can cancel. Affordable only
    because these shapes are tiny. `wrt=None` checks every operand: matmul's two
    gradients differ in shape, and a check that silently covered only the first
    would be the easiest hole in the suite.
    """
    if tol is None:
        tol = tolerance(forward_magnitude(f, arrays, views), h)
    targets = range(len(arrays)) if wrt is None else [wrt]
    for k in targets:
        num = numerical_jacobian(f, arrays, k, h, views)
        ana = analytic_jacobian(f, arrays, k, views)
        _report(num, ana, tol, f"jacobian wrt operand {k}")


def check_directional(
    f: Builder,
    arrays: Sequence[np.ndarray],
    *,
    seeds: Sequence[int] = (0, 1, 2),
    wrt: int | None = None,
    h: float = GRADCHECK_H,
    views: Sequence[ViewFn | None] | None = None,
    tol: float | None = None,
) -> None:
    """Contract the output with a random v, then compare. For outputs too big
    for a full Jacobian.

    The sum(out * v) contraction is done in NumPy, not with autograd.mul and
    autograd.sum - building it in the engine would put Mul and Sum inside every
    composite check and let a bug in one hide a bug in another.

    Several seeds because one v is blind to any error orthogonal to it.
    """
    if tol is None:
        tol = tolerance(forward_magnitude(f, arrays, views), h)
    vs = _as_list(views, len(arrays))
    targets = range(len(arrays)) if wrt is None else [wrt]

    probe = f(make_leaves(arrays, True, views))
    out_shape = probe.shape

    for seed in seeds:
        v = np.random.default_rng(seed).uniform(-1.0, 1.0, out_shape).astype(np.float32)
        v64 = v.astype(np.float64)

        for k in targets:
            perm = _view_permutation(arrays[k].shape, vs[k])

            num = np.zeros(arrays[k].size, dtype=np.float64)
            for i in range(arrays[k].size):
                side = []
                for sign in (+1.0, -1.0):
                    pert = [np.array(x, dtype=np.float64) for x in arrays]
                    pert[k].reshape(-1)[i] += sign * h
                    out = f(make_leaves([p.astype(np.float32) for p in pert], True, views))
                    side.append(float(np.sum(out.to_numpy().astype(np.float64) * v64)))
                num[i] = (side[0] - side[1]) / (2.0 * h)

            leaves = make_leaves(arrays, True, views)
            f(leaves).backward(_core.from_numpy(v))
            ana = _to_base_order(np.asarray(leaves[k].grad), perm, arrays[k].size)

            _report(num[None, :], ana[None, :], tol,
                    f"directional wrt operand {k}, seed {seed}")
