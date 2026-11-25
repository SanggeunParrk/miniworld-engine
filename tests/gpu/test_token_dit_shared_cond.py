"""The H100 token DiT training block with the conditioning shared by the samples (``kernels/integrations/token_dit_train._shared_cond``).

FoldForge's ``TokenDiTStack`` hands every block ONE [L, d] conditioning expanded (stride 0) over the A samples. The block runs the
conditioning GEMMs (G, Gg forward; dchat, dcg and the dWn, dWg weight gradients backward) on L rows instead of A L: the row kernels read row
``r % L`` of G / GG, and the backward sums dG / dGg over the samples (``sample_sum2``, fp32, sample order) before its GEMMs. The result is not
bitwise the per-sample route's (the sum order differs) but stays inside bf16 rounding of it; it is deterministic.
"""

from __future__ import annotations

import typing

import pytest
import torch

from foldjet_engine.kernels.integrations import token_dit_train as TT
from foldjet_engine.measure import config as C
from foldjet_engine.modules.token_dit import TokenDiTProtenix

pytestmark = [pytest.mark.gpu, pytest.mark.h100]


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _block() -> typing.Any:
    module = TokenDiTProtenix(d_single=768, d_cond=384, d_pair=128, n_head=16, n=2).cuda().float()
    C.randomize(module, torch.Generator(device="cuda").manual_seed(3))
    m: typing.Any = C.cast(module, "master")
    return m.train().set_backend("cuda")


def _step(m, base, shared: bool, A: int, L: int, *, expanded: bool = True):
    """One forward + backward; every gradient (inputs, the conditioning base, parameters) and the output, as fp32 tensors."""
    TT.SHARED_COND = shared
    try:
        g = torch.Generator(device="cuda").manual_seed(1)
        inp = m.make_inputs(length=L, samples=A, depth=None, dtype=torch.bfloat16, device="cuda", generator=g, stream="token_single_shared_cond")
        s = inp["single"].detach().clone().requires_grad_(True)
        pr = inp["pair"].detach().clone().requires_grad_(True)
        cl = inp["cond"].detach()[:1].clone().requires_grad_(True)
        cond = cl.expand(A, -1, -1, -1) if expanded else cl.expand(A, -1, -1, -1).contiguous()
        dout = torch.randn(A, 1, L, 768, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7)).bfloat16()
        m.zero_grad(set_to_none=True)
        kw = {k: v for k, v in inp.items() if k not in ("single", "cond", "pair")}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(single=s, cond=cond, pair=pr, **kw)
        out.backward(dout)
        grads = {"single": s.grad, "pair": pr.grad, **{n: p.grad for n, p in m.named_parameters()}}
        if expanded:
            grads["cond"] = cl.grad
        return out.detach().float(), {k: typing.cast(torch.Tensor, v).float().clone() for k, v in grads.items()}
    finally:
        TT.SHARED_COND = True


def _rel_max(x: torch.Tensor, ref: torch.Tensor) -> float:
    return float((x - ref).abs().max() / ref.abs().max().clamp_min(1e-30))


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
@pytest.mark.parametrize("A,L", [(8, 256), (5, 384), (3, 200), (48, 128)])
def test_shared_route_matches_the_per_sample_route(A, L) -> None:
    m = _block()
    out0, g0 = _step(m, None, False, A, L)
    out1, g1 = _step(m, None, True, A, L)
    # the sum over the samples moves before the GEMM (and the dG partial sums are rounded once): well inside bf16 rounding of the gradients
    assert _rel_max(out1, out0) < 2e-3, "output"
    for name, ref in g0.items():
        assert bool(torch.isfinite(g1[name]).all()), name
        assert _rel_max(g1[name], ref) < 2e-2, name


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
def test_shared_route_is_deterministic() -> None:
    m = _block()
    out_a, ga = _step(m, None, True, 8, 256)
    out_b, gb = _step(m, None, True, 8, 256)
    assert torch.equal(out_a, out_b)
    for name, ref in ga.items():
        assert torch.equal(gb[name], ref), name


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
def test_dense_conditioning_keeps_the_per_sample_route() -> None:
    """A conditioning that is not a stride-0 expand (a dense [A, 1, L, d]) never takes the shared route: the switch changes nothing."""
    m = _block()
    out_a, ga = _step(m, None, True, 8, 256, expanded=False)
    out_b, gb = _step(m, None, False, 8, 256, expanded=False)
    assert torch.equal(out_a, out_b)
    for name, ref in ga.items():
        assert torch.equal(gb[name], ref), name


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("A,n1,n2", [(1, 64, 32), (5, 1536, 768), (48, 384 * 3072 // 8, 384 * 1536 // 8)])
def test_sample_sum2_is_the_fixed_order_sum(dtype, A, n1, n2) -> None:
    from foldjet_engine.kernels.conditioned_transition.cuda.train import ext

    T = ext(768, 48, rpb=TT._native_rpb(), empty_uniform=False, dc=384, n=2, dp=128)
    g = torch.Generator(device="cuda").manual_seed(0)
    x1 = torch.randn(A, n1, device="cuda", generator=g).to(dtype)
    x2 = torch.randn(A, n2, device="cuda", generator=g).to(dtype)
    o1, o2 = torch.empty(2, n1, device="cuda", dtype=dtype), torch.empty(2, n2, device="cuda", dtype=dtype)
    T.sample_sum2(x1, o1, x2, o2)
    for x, o in ((x1, o1), (x2, o2)):
        acc = torch.zeros(x.shape[1], device="cuda")
        for a in range(A):  # the kernel's order: fp32, sample 0, 1, ..
            acc += x[a].float()
        hi, lo = o[0].float(), o[1].float()
        if dtype is torch.bfloat16:
            assert torch.equal(o[0], acc.to(dtype))  # hi = the sum rounded to bf16
        bound = (2.0**-15 if dtype is torch.bfloat16 else 2.0**-20) * acc.abs()  # hi + lo = the sum, to the second operand's rounding
        assert bool(((hi + lo - acc).abs() <= bound).all())
