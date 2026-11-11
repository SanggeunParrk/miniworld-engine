"""FoldForge API requests on the GPU: R1 (a weight replaced after ``clear_weight_caches`` is picked up by a cached H100 kernel path),
R18 (the multi-fold call of the DiT blocks on ``cuda`` (H100) and ``triton`` equals one call per fold; on A100 one sm_80 call per fold), R12 (``parallel_residual`` on ``triton`` and ``cuda``)."""

from __future__ import annotations

import typing

import pytest
import torch

from miniworld_engine import capture
from miniworld_engine.measure import config as C
from miniworld_engine.modules import AtomDiT, ParallelResidual, TokenDiT, parallel_residual
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
#: (id, block, stream, length, cuda serves on A100): N = 1000 (not a multiple of 32) only on H100; FoldForge's atom layout is whole
#: 32-atom windows, the sm_80 atom kernels take N a multiple of 64
FOLD_CASES = [
    ("token", lambda: TokenDiT("protenix", 768, 384, 128, 16, 2), "token_folds", 128, True),
    ("atom-pad", lambda: AtomDiT("protenix"), "atom_folds", 1000, False),
    ("atom-slide", lambda: AtomDiT("af3", key_window="slide"), "atom_folds", 1000, False),
    ("atom-pad-1024", lambda: AtomDiT("protenix"), "atom_folds", 1024, True),
    ("atom-chai1-uid", lambda: AtomDiT("chai1"), "atom_folds", 1024, True),
]


def _h100() -> bool:
    return torch.cuda.get_device_capability() == (9, 0)


def _one_fold(inp: dict, b: int) -> dict:
    one = {"single": inp["single"][:, b : b + 1], "cond": inp["cond"][:, b : b + 1], "pair": inp["pair"][b : b + 1]}
    one["mask"] = inp["mask"][b : b + 1]
    if "uid" in inp:
        one["uid"] = inp["uid"][b : b + 1]
    return one


@pytest.mark.parametrize("backend", ["cuda", "triton"])
@pytest.mark.parametrize(("name", "make", "stream", "length", "a100"), FOLD_CASES, ids=[c[0] for c in FOLD_CASES])
def test_multi_fold_equals_one_call_per_fold(name, make, stream, length, a100, backend):
    """B folds in one call == B calls of one fold, and the reference. ``cuda``: H100 (the grouped-bias kernels) and A100 (one sm_80 call
    per fold)."""
    if backend == "cuda" and not _h100() and not a100:
        pytest.skip("N outside the sm_80 atom kernels' range")
    block = _block(make).set_backend(backend)
    folds, samples = 5, 5
    inp = block.make_inputs(
        length=length, samples=samples, depth=folds, dtype=torch.bfloat16, device="cuda", generator=_gen(), stream=stream
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert block.unsupported(**inp) is None, block.unsupported(**inp)
        out = block(**inp)
        for b in range(folds):
            one = block(**_one_fold(inp, b))
            err = (out[:, b : b + 1].float() - one.float()).abs().max() / one.float().abs().max()
            assert err < 1e-2, f"{name}: fold {b} differs from its own call (rel err {err:.3g})"
        ref = block.set_backend("pytorch")(**inp)
    # the real query rows only: a padding query's row is not an output (N = 1000: the cuda block keeps the clipped window for the
    # all-padding last window where the reference slides it)
    real = inp["mask"][0]
    out, ref = out[:, :, real].float(), ref[:, :, real].float()
    err = (out - ref).abs().max() / ref.abs().max()
    assert err < 3e-2, f"{name}: the multi-fold {backend} call is off the reference (rel err {err:.3g})"


@pytest.mark.parametrize("model", ["protenix", "chai1"])
def test_multi_fold_cuda_refuses_training_and_fp32(model):
    """B > 1 folds on ``cuda``: bf16 inference only (H100 and A100), refused with a reason."""
    block = _block(lambda: AtomDiT(model), "float32").set_backend("cuda")
    inp = block.make_inputs(length=256, samples=2, depth=3, dtype=torch.float32, device="cuda", generator=_gen(), stream="atom_folds")
    assert "bfloat16" in (block.unsupported(**inp) or "")
    with pytest.raises(BackendUnsupported):
        block(**inp)
    block = _block(lambda: AtomDiT(model)).set_backend("cuda").train()
    inp = block.make_inputs(length=256, samples=2, depth=3, dtype=torch.bfloat16, device="cuda", generator=_gen(), stream="atom_folds")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert "inference only" in (block.unsupported(**inp) or "")


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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("branches", [1, 2, 3, 4])
@pytest.mark.parametrize("shape", [(1, 129, 130, 64), (3, 5, 7)])  # whole 16-byte vectors / an odd size (the tail elements)
def test_parallel_residual_cuda(dtype, branches, shape):
    """The hand-written kernel (H100 sm_90a / A100 sm_80) is the reference's arithmetic in the same order: bitwise equal to ``pytorch``
    (and the Triton kernel); the gradients: ``d/dy_i = g`` (the same tensor), ``d/dx = (1 - n) g`` bitwise equal to the reference's."""
    g = _gen()
    x = torch.randn(shape, device="cuda", generator=g).to(dtype)
    ys = [(x.float() + 0.1 * torch.randn(x.shape, device="cuda", generator=g)).to(dtype) for _ in range(branches)]
    ref = parallel_residual(x, *ys, backend="pytorch")
    assert torch.equal(parallel_residual(x, *ys, backend="cuda"), ref)
    assert torch.equal(parallel_residual(x, *ys, backend="triton"), ref)
    # not 16-byte aligned (an offset view): the scalar kernel, same values
    xo = torch.empty(x.numel() + 1, device="cuda", dtype=dtype)[1:].view(shape).copy_(x)
    assert torch.equal(parallel_residual(xo, *ys, backend="cuda"), ref)
    # train: the module path and the reference's gradients
    m = ParallelResidual(d=shape[-1], branches=branches, backend="cuda")
    up = torch.randn(x.shape, device="cuda", generator=g).to(dtype)
    grads = []
    for backend in ("cuda", "pytorch"):
        xs = x.detach().requires_grad_(True)
        yg = [y.detach().requires_grad_(True) for y in ys]
        out = m(xs, *yg) if backend == "cuda" else parallel_residual(xs, *yg, backend="pytorch")
        out.backward(up)
        grads.append([xs.grad, *(y.grad for y in yg)])
    for a, b in zip(*grads, strict=True):
        assert torch.equal(a, b)


def test_parallel_residual_cuda_refuses_outside_its_range():
    x = torch.randn(4, 8, device="cuda")
    with pytest.raises(BackendUnsupported, match="branches"):
        parallel_residual(x, *[x] * 5, backend="cuda")
    with pytest.raises(BackendUnsupported, match="dtype"):
        parallel_residual(x.double(), x.double(), backend="cuda")
    with pytest.raises(BackendUnsupported, match="CUDA tensors"):
        parallel_residual(x.cpu(), x.cpu(), backend="cuda")


@pytest.mark.h100
def test_multi_fold_atom_pair_bias_is_hoisted_once():
    """The sampler calls a block many times with the same pair / mask objects: the multi-fold call's pair bias is made once (the folds'
    expanded key mask is read as its one row, not copied per call)."""
    from miniworld_engine.kernels.integrations import local_dit_h100

    block = _block(lambda: AtomDiT("protenix")).set_backend("cuda")
    inp = block.make_inputs(length=512, samples=2, depth=3, dtype=torch.bfloat16, device="cuda", generator=_gen(), stream="atom_folds")
    capture.clear_weight_caches(block)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        first = block(**inp)
        size = len(local_dit_h100._HOIST_BIAS)
        for _ in range(3):
            again = block(**inp)
    assert size >= 1 and len(local_dit_h100._HOIST_BIAS) == size
    assert torch.equal(first, again)
