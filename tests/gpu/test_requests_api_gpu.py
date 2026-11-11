"""FoldForge API requests on the GPU: R1 (a weight replaced after ``clear_weight_caches`` is picked up by a cached H100 kernel path),
R18 (the multi-fold call of the DiT blocks on ``cuda`` equals one call per fold), R12 (``parallel_residual`` on ``triton``)."""

from __future__ import annotations

import typing

import pytest
import torch

from miniworld_engine import capture
from miniworld_engine.measure import config as C
from miniworld_engine.modules import AtomDiT, TokenDiT, parallel_residual
from miniworld_engine.modules.primitives import BackendUnsupported

pytestmark = pytest.mark.gpu


def _block(make, dtype: str = "master"):
    torch.manual_seed(0)
    m = make().cuda().float()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    return typing.cast("typing.Any", C.cast(m, dtype).eval())


def _gen(seed: int = 1) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


# ------------------------------------------------------------------------------------------------------------------------------ R1
@pytest.mark.h100
def test_clear_weight_caches_picks_up_a_replaced_weight():
    """The H100 token DiT inference block caches its weight pack (``token_dit.runners``, keyed on pointer / version). A weight rewritten
    without a version bump is served the old pack; after ``clear_weight_caches`` the new one."""
    block = _block(lambda: TokenDiT("protenix", 768, 384, 128, 16, 2), "bfloat16").set_backend("cuda")
    inp = block.make_inputs(length=128, samples=2, depth=None, dtype=torch.bfloat16, device="cuda", generator=_gen())
    with torch.no_grad():
        first = block(**inp)
        w = block.attention.to_query.weight
        v = w._version
        w.data.copy_(torch.randn(w.shape, device="cuda", generator=_gen(7)).to(w.dtype) * 0.05)  # no version bump
        assert w._version == v
        stale = block(**inp)
        capture.clear_weight_caches(block)
        fresh = block(**inp)
        ref = block.set_backend("pytorch")(**inp)
    assert torch.equal(stale, first), "the pack was not cached (the test does not exercise a cache)"
    assert not torch.equal(fresh, first)
    err = (fresh.float() - ref.float()).abs().max() / ref.float().abs().max()
    assert err < 2e-2, f"after the clear the block does not compute with the new weight (rel err {err:.3g})"


# ----------------------------------------------------------------------------------------------------------------------------- R18
FOLD_CASES = [
    ("token", lambda: TokenDiT("protenix", 768, 384, 128, 16, 2), "token_folds", 128),
    ("atom-pad", lambda: AtomDiT("protenix"), "atom_folds", 1000),
    ("atom-slide", lambda: AtomDiT("af3", key_window="slide"), "atom_folds", 1000),
]


@pytest.mark.h100
@pytest.mark.parametrize(("name", "make", "stream", "length"), FOLD_CASES, ids=[c[0] for c in FOLD_CASES])
def test_multi_fold_cuda_equals_one_call_per_fold(name, make, stream, length):
    block = _block(make).set_backend("cuda")
    folds, samples = 5, 5
    inp = block.make_inputs(length=length, samples=samples, depth=folds, dtype=torch.bfloat16, device="cuda", generator=_gen())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert block.unsupported(**inp) is None, block.unsupported(**inp)
        out = block(**inp)
        for b in range(folds):
            one = block(inp["single"][:, b : b + 1], inp["cond"][:, b : b + 1], inp["pair"][b : b + 1], inp["mask"][b : b + 1])
            err = (out[:, b : b + 1].float() - one.float()).abs().max() / one.float().abs().max()
            assert err < 1e-2, f"{name}: fold {b} differs from its own call (rel err {err:.3g})"
        ref = block.set_backend("pytorch")(**inp)
    err = (out.float() - ref.float()).abs().max() / ref.float().abs().max()
    assert err < 3e-2, f"{name}: the multi-fold cuda call is off the reference (rel err {err:.3g})"


@pytest.mark.h100
def test_multi_fold_cuda_refuses_training_and_fp32():
    block = _block(lambda: AtomDiT("protenix"), "float32").set_backend("cuda")
    inp = block.make_inputs(length=256, samples=2, depth=3, dtype=torch.float32, device="cuda", generator=_gen())
    assert "bfloat16" in (block.unsupported(**inp) or "")
    with pytest.raises(BackendUnsupported):
        block(**inp)


# ----------------------------------------------------------------------------------------------------------------------------- R12
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("branches", [1, 2, 3, 4])
def test_parallel_residual_triton(dtype, branches):
    g = _gen()
    x = torch.randn(1, 129, 130, 64, device="cuda", generator=g).to(dtype)
    ys = [(x.float() + 0.1 * torch.randn(x.shape, device="cuda", generator=g)).to(dtype) for _ in range(branches)]
    want = (x.double() + sum(y.double() - x.double() for y in ys)).to(dtype)
    got = parallel_residual(x, *ys, backend="triton")
    ref = parallel_residual(x, *ys, backend="pytorch")
    assert torch.equal(got, ref) or (got.float() - want.float()).abs().max() <= (ref.float() - want.float()).abs().max() + 1e-6
    # backward: d/dy_i = g, d/dx = (1 - n) g
    xs = x.detach().requires_grad_(True)
    yg = [y.detach().requires_grad_(True) for y in ys]
    up = torch.randn(x.shape, device="cuda", generator=g).to(dtype)
    parallel_residual(xs, *yg, backend="triton").backward(up)
    assert all(torch.equal(y.grad, up) for y in yg)  # pyright: ignore[reportArgumentType]
    torch.testing.assert_close(xs.grad, (up.float() * (1 - branches)).to(dtype))
