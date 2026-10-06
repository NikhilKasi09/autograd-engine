"""The engine's gradients against PyTorch's, on identical inputs.

Finite differences in float32 top out around 1e-4 - that is what cancellation
affords, not a matter of effort. Torch in float64 is an independent oracle two
orders of magnitude tighter, so this is where a small systematic error would
show up that gradcheck.py would wave through.

Three places the engine deliberately diverges from torch, each constructed here
rather than assumed:

  - torch broadcasts implicitly; this engine has an explicit expand op, so the
    torch side calls .expand() to mirror it.
  - Sum keeps rank with 1s ({2,3} -> {1,1}), so torch needs keepdim=True.
  - .grad here is a raw buffer, not a graph tensor, so there is no
    gradient-of-gradient to compare.

Skipped when torch is absent, since it is the oracle rather than the artefact.
GEMM_REQUIRE_TORCH makes that a hard error - see conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest

import autograd
import gradcheck
from autograd import _core, nn

torch = pytest.importorskip("torch")

# float32 forward against float64 torch. The gap is the engine's own rounding,
# roughly sqrt(K) * 1e-7 for a K-term accumulation, not the comparison's.
RTOL = 1e-6
ATOL = 1e-6


def u(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, shape).astype(np.float32)


def as_torch(arrays: list[np.ndarray]) -> list[torch.Tensor]:
    """float64 leaves holding exactly the float32 values the engine has."""
    return [torch.tensor(a.astype(np.float64), requires_grad=True) for a in arrays]


def compare(engine_leaves, torch_leaves, label: str = "") -> None:
    for i, (mine, theirs) in enumerate(zip(engine_leaves, torch_leaves)):
        got = np.asarray(mine.grad).astype(np.float64)
        want = theirs.grad.detach().numpy()
        assert np.allclose(got, want, rtol=RTOL, atol=ATOL), (
            f"{label} operand {i}: max diff {np.max(np.abs(got - want)):.3e}\n"
            f"  engine {got.ravel()[:6]}\n  torch  {want.ravel()[:6]}"
        )


def run_both(arrays, engine_fn, torch_fn, seed_shape) -> None:
    """Same seed into both graphs, then compare every leaf gradient."""
    v = np.random.default_rng(7).uniform(-1.0, 1.0, seed_shape).astype(np.float32)

    leaves = gradcheck.make_leaves(arrays)
    engine_fn(leaves).backward(_core.from_numpy(v))

    tleaves = as_torch(arrays)
    torch_fn(tleaves).backward(torch.tensor(v.astype(np.float64)))

    compare(leaves, tleaves)


# --------------------------------------------------------------------------
# One op at a time
# --------------------------------------------------------------------------


def test_add_matches_torch(rng) -> None:
    run_both([u(rng, 2, 3), u(rng, 2, 3)],
             lambda L: autograd.add(L[0], L[1]),
             lambda T: T[0] + T[1], (2, 3))


def test_mul_matches_torch(rng) -> None:
    run_both([u(rng, 2, 3), u(rng, 2, 3)],
             lambda L: autograd.mul(L[0], L[1]),
             lambda T: T[0] * T[1], (2, 3))


@pytest.mark.parametrize("M, K, N", [(2, 3, 4), (7, 9, 8), (1, 5, 3), (4, 1, 3)],
                         ids=lambda v: str(v))
def test_matmul_matches_torch(M: int, K: int, N: int, rng) -> None:
    """Never square, and M == 1 included - the shape that used to raise."""
    run_both([u(rng, M, K), u(rng, K, N)],
             lambda L: autograd.matmul(L[0], L[1]),
             lambda T: T[0] @ T[1], (M, N))


def test_relu_matches_torch(rng) -> None:
    run_both([gradcheck.away_from_zero(u(rng, 2, 3))],
             lambda L: autograd.relu(L[0]),
             lambda T: torch.relu(T[0]), (2, 3))


def test_relu_agrees_with_torch_at_exactly_zero() -> None:
    """Both take the subgradient at 0 to be 0 - relu_backward compares strictly.

    Finite differences cannot reach this point at all, so torch is the only
    thing that can confirm it.
    """
    a = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    run_both([a], lambda L: autograd.relu(L[0]),
             lambda T: torch.relu(T[0]), (1, 3))

    leaves = gradcheck.make_leaves([a])
    autograd.relu(leaves[0]).backward(_core.from_numpy(np.ones((1, 3), np.float32)))
    assert np.asarray(leaves[0].grad)[0, 1] == 0.0


@pytest.mark.parametrize("shape, dims", [
    pytest.param((1, 1), (0, 1), id="full"),
    pytest.param((1, 3), (0,), id="axis0"),
    pytest.param((2, 1), (1,), id="axis1"),
])
def test_sum_matches_torch(shape, dims, rng) -> None:
    """keepdim=True on the torch side: Sum keeps rank with 1s, torch does not."""
    run_both([u(rng, 2, 3)],
             lambda L: autograd.sum(L[0], shape),
             lambda T: T[0].sum(dim=dims, keepdim=True), shape)


@pytest.mark.parametrize("base, target", [
    pytest.param((1, 4), (3, 4), id="bias-row"),
    pytest.param((2, 1), (2, 5), id="column"),
])
def test_expand_matches_torch(base, target, rng) -> None:
    """.expand() on the torch side, not broadcasting - the engine has no
    implicit broadcast and the point is to compare the same operation."""
    run_both([u(rng, *base)],
             lambda L: autograd.expand(L[0], target),
             lambda T: T[0].expand(target), target)


def test_scale_matches_torch(rng) -> None:
    run_both([u(rng, 2, 3)],
             lambda L: autograd.scale(L[0], -2.5),
             lambda T: T[0] * -2.5, (2, 3))


# --------------------------------------------------------------------------
# The nn layer
# --------------------------------------------------------------------------


@pytest.mark.parametrize("M", [2, 1], ids=["batch2", "batch1"])
def test_linear_matches_torch(M: int, rng) -> None:
    """Written out as x @ w + b rather than F.linear, which wants the weight
    {out, in} and would hide the layout difference instead of checking it."""
    run_both([u(rng, M, 3), u(rng, 3, 4), u(rng, 1, 4)],
             lambda L: nn.linear(L[0], L[1], L[2]),
             lambda T: T[0] @ T[1] + T[2].expand(M, 4), (M, 4))


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_mse_loss_matches_torch(reduction: str, rng) -> None:
    """reshape(1, 1) because the loss here keeps its rank, like sum."""
    run_both([u(rng, 2, 3), u(rng, 2, 3)],
             lambda L: nn.mse_loss(L[0], L[1], reduction=reduction),
             lambda T: torch.nn.functional.mse_loss(
                 T[0], T[1], reduction=reduction).reshape(1, 1),
             (1, 1))


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("M, C", [(3, 4), (1, 5)], ids=["3x4", "batch1"])
def test_cross_entropy_matches_torch(M: int, C: int, reduction: str, rng) -> None:
    """Against log_softmax written out, with the same one-hot target. Only the
    logits are compared: the target gets no gradient here by design, and torch
    would happily give it one."""
    logits = u(rng, M, C)
    target = np.zeros((M, C), dtype=np.float32)
    target[np.arange(M), rng.integers(0, C, M)] = 1.0
    v = np.array([[0.7]], dtype=np.float32)     # not 1, so grad_out matters

    leaves = gradcheck.make_leaves([logits, target])
    loss = nn.cross_entropy(leaves[0], leaves[1], reduction=reduction)
    loss.backward(_core.from_numpy(v))

    tl = torch.tensor(logits.astype(np.float64), requires_grad=True)
    tt = torch.tensor(target.astype(np.float64))
    ref = -(torch.log_softmax(tl, dim=1) * tt).sum()
    if reduction == "mean":
        ref = ref / M
    ref.reshape(1, 1).backward(torch.tensor(v.astype(np.float64)))

    assert loss.to_numpy()[0, 0] == pytest.approx(ref.item(), rel=1e-6)
    compare(leaves[:1], [tl])
    assert leaves[1].grad is None


# --------------------------------------------------------------------------
# The whole thing
# --------------------------------------------------------------------------


def test_the_two_layer_mlp_matches_torch(rng) -> None:
    """Every op, every shape change, one backward pass - the case CLAUDE.md
    actually names, and the one phase 8 will be built on."""
    arrays = [u(rng, 2, 3), u(rng, 3, 4),
              gradcheck.away_from_zero(u(rng, 1, 4)), u(rng, 4, 2)]

    def engine(L):
        hidden = autograd.relu(autograd.add(autograd.matmul(L[0], L[1]),
                                            autograd.expand(L[2], (2, 4))))
        return autograd.sum(autograd.matmul(hidden, L[3]))

    def ref(T):
        hidden = torch.relu(T[0] @ T[1] + T[2].expand(2, 4))
        return (hidden @ T[3]).sum(dim=(0, 1), keepdim=True)

    run_both(arrays, engine, ref, (1, 1))


def test_a_leaf_used_twice_accumulates_like_torch(rng) -> None:
    """Both paths reach the same leaf. Accumulation, not assignment."""
    run_both([u(rng, 2, 3)],
             lambda L: autograd.add(autograd.mul(L[0], L[0]), L[0]),
             lambda T: T[0] * T[0] + T[0], (2, 3))


def test_a_strided_leaf_matches_torch(rng) -> None:
    """The engine reads operand strides in place; torch transposes to match."""
    a, b = u(rng, 3, 2), u(rng, 3, 4)
    v = np.random.default_rng(7).uniform(-1, 1, (2, 4)).astype(np.float32)

    leaves = gradcheck.make_leaves([a, b], views=[lambda t: t.transpose(0, 1), None])
    autograd.matmul(leaves[0], leaves[1]).backward(_core.from_numpy(v))

    tleaves = as_torch([a, b])
    (tleaves[0].T @ tleaves[1]).backward(torch.tensor(v.astype(np.float64)))

    # The engine's gradient is shaped like the view it was handed, {2,3}; torch's
    # is shaped like the base, {3,2}.
    got = np.asarray(leaves[0].grad).astype(np.float64)
    assert np.allclose(got.T, tleaves[0].grad.numpy(), rtol=RTOL, atol=ATOL)
    compare(leaves[1:], tleaves[1:])


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def test_five_sgd_steps_on_an_mlp_match_torch(rng) -> None:
    """The only oracle for the optimiser, and for everything at once: the same
    weights, the same batch, five steps each, parameters compared at the end.

    Weights are {in, out} on both sides, so torch runs x @ w rather than
    nn.Linear, which would want them transposed.
    """
    from autograd import optim

    M, n_in, n_hidden, n_out, lr = 6, 3, 5, 4, 0.3
    x = u(rng, M, n_in)
    target = np.zeros((M, n_out), dtype=np.float32)
    target[np.arange(M), rng.integers(0, n_out, M)] = 1.0
    init = [u(rng, n_in, n_hidden), u(rng, 1, n_hidden),
            u(rng, n_hidden, n_out), u(rng, 1, n_out)]

    params = [nn.Parameter(_core.from_numpy(a)) for a in init]
    opt = optim.SGD(params, lr=lr)
    xt, tt = gradcheck.make_leaves([x, target], requires_grad=False)
    for _ in range(5):
        opt.zero_grad()
        hidden = autograd.relu(nn.linear(xt, params[0], params[1]))
        nn.cross_entropy(nn.linear(hidden, params[2], params[3]), tt).backward()
        opt.step()

    tparams = as_torch(init)
    topt = torch.optim.SGD(tparams, lr=lr)
    tx = torch.tensor(x.astype(np.float64))
    ttarget = torch.tensor(target.astype(np.float64))
    for _ in range(5):
        topt.zero_grad()
        hidden = torch.relu(tx @ tparams[0] + tparams[1])
        logits = hidden @ tparams[2] + tparams[3]
        (-(torch.log_softmax(logits, dim=1) * ttarget).sum() / M).backward()
        topt.step()

    for i, (mine, theirs) in enumerate(zip(params, tparams)):
        assert np.allclose(mine.to_numpy(), theirs.detach().numpy(), rtol=1e-5, atol=1e-6), (
            f"parameter {i} drifted from torch after five steps")
