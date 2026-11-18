"""The H100 bf16 token DiT's pair bias on the AttentionPairBias tensor-core kernels (``token_dit_train._pair_apb``).

Training: the forward's bias is ``Wf . LN(pair)`` on bf16 mma (row statistics from the tensor cores), the backward gives dpair and the
per-block dWf partials in one pass (a pair wider than 128 channels saves its row statistics and its fp32 unmasked bias). The dbias it reads
is fp32 and in the log2 bias's scale (``_native_attn_bwd90(.., dbf32)``). Inference: the hoisted bias of a 256 / 512-wide pair is the same
kernel (``conditioned_transition.cuda``'s ``make_bias``), no LN rows + cuBLAS. Checked against the fp32 PyTorch block (the engine's accuracy
standard: no worse than the TF32 row kernels it replaces, up to bf16 rounding) and for determinism.
"""

from __future__ import annotations

import typing

import pytest
import torch

from miniworld_engine.kernels.integrations import token_dit_train as TT
from miniworld_engine.measure import config as C
from miniworld_engine.modules.token_dit import TokenDiTProtenix

pytestmark = [pytest.mark.gpu, pytest.mark.h100]


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _block(dp: int) -> typing.Any:
    module = TokenDiTProtenix(d_single=768, d_cond=384, d_pair=dp, n_head=16, n=2).cuda().float()
    C.randomize(module, torch.Generator(device="cuda").manual_seed(3))
    return module


def _inputs(m, A: int, L: int):
    g = torch.Generator(device="cuda").manual_seed(1)
    inp = m.make_inputs(length=L, samples=A, depth=None, dtype=torch.bfloat16, device="cuda", generator=g, stream="token_single_shared_cond")
    ex = torch.randn(L, L, device="cuda", generator=torch.Generator(device="cuda").manual_seed(11)) * 0.5
    return inp, ex


def _step(m, inp, ex, A: int, L: int, *, apb: bool, ref: bool = False):
    """One forward + backward; the output and every gradient (single, pair, the extra term, parameters) as fp32."""
    TT.PAIR_APB = apb
    try:
        dt = torch.float32 if ref else torch.bfloat16
        s = inp["single"].detach().clone().to(dt).requires_grad_(True)
        pr = inp["pair"].detach().clone().to(dt).requires_grad_(True)
        cl = inp["cond"].detach()[:1].clone().to(dt).requires_grad_(True)
        e = ex.clone().requires_grad_(True)
        dout = torch.randn(A, 1, L, 768, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7)).to(dt)
        m.zero_grad(set_to_none=True)
        kw = {k: v for k, v in inp.items() if k not in ("single", "cond", "pair")}
        if ref:
            out = m(single=s, cond=cl.expand(A, -1, -1, -1), pair=pr, extra_pair_bias=e, **kw)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = m(single=s, cond=cl.expand(A, -1, -1, -1), pair=pr, extra_pair_bias=e, **kw)
        out.backward(dout)
        grads = {"single": s.grad, "pair": pr.grad, "extra": e.grad, **{n: p.grad for n, p in m.named_parameters()}}
        return out.detach().float(), {k: typing.cast(torch.Tensor, v).float().clone() for k, v in grads.items()}
    finally:
        TT.PAIR_APB = True


def _rel_max(x: torch.Tensor, ref: torch.Tensor) -> float:
    return float((x - ref).abs().max() / ref.abs().max().clamp_min(1e-30))


def _rel_rms(x: torch.Tensor, ref: torch.Tensor) -> float:
    return float((x - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-30))


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
@pytest.mark.parametrize("dp,A,L", [(128, 2, 256), (128, 5, 384), (256, 2, 256), (512, 2, 256)])
def test_apb_pair_bias_matches_reference(dp, A, L) -> None:
    module = _block(dp)
    m: typing.Any = C.cast(module, "master").train().set_backend("cuda")
    inp, ex = _inputs(m, A, L)
    ref = TokenDiTProtenix(d_single=768, d_cond=384, d_pair=dp, n_head=16, n=2).cuda().float()
    ref.load_state_dict(module.state_dict())
    ref.train().set_backend("pytorch")
    out_r, g_r = _step(ref, inp, ex, A, L, apb=True, ref=True)
    out_o, g_o = _step(m, inp, ex, A, L, apb=False)
    out_n, g_n = _step(m, inp, ex, A, L, apb=True)
    # the pair-bias route changes bf16 roundings only: never worse than the TF32 row kernels (+ 25 %), up to a bf16 floor
    assert _rel_rms(out_n, out_r) <= max(1.25 * _rel_rms(out_o, out_r), 8e-3), "output"
    for name in ("pair", "extra", "attention.ln_pair.weight", "attention.to_bias.weight", "single"):
        assert bool(torch.isfinite(g_n[name]).all()), name
        e_n, e_o = _rel_rms(g_n[name], g_r[name]), _rel_rms(g_o[name], g_r[name])
        assert e_n <= max(1.25 * e_o, 2e-2), (name, e_n, e_o)
    for name, gn in g_n.items():
        assert bool(torch.isfinite(gn).all()), name


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
@pytest.mark.parametrize("dp", [128, 256])
def test_apb_pair_bias_is_deterministic(dp) -> None:
    module = _block(dp)
    m: typing.Any = C.cast(module, "master").train().set_backend("cuda")
    inp, ex = _inputs(m, 2, 256)
    out_a, ga = _step(m, inp, ex, 2, 256, apb=True)
    out_b, gb = _step(m, inp, ex, 2, 256, apb=True)
    assert torch.equal(out_a, out_b)
    for name in ga:
        assert torch.equal(ga[name], gb[name]), name


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
@pytest.mark.parametrize("dp", [256, 512])
def test_wide_inference_bias_matches_the_ln_rows_route(dp) -> None:
    """The hoisted bias of a 256 / 512-wide pair: the one-pass kernel against LN rows + cuBLAS (``_pair_bias_wide``), masked keys at -1e30."""
    from miniworld_engine.kernels.augmented_attention.cuda import apb
    from miniworld_engine.kernels.conditioned_transition.cuda import _pair_bias_wide

    L = 256
    g = torch.Generator(device="cuda").manual_seed(5)
    z = (torch.randn(L * L, dp, device="cuda", generator=g) * 1.3 + 0.4).bfloat16()
    wf = (torch.randn(16, dp, device="cuda", generator=g) * 0.1).bfloat16()
    mask = torch.ones(L, dtype=torch.bool, device="cuda")
    mask[L - 5 :] = False
    new = apb.pair_bias(z, wf, mask, L, 1e-5, -1e30)
    old = _pair_bias_wide(z, wf, mask, L)
    assert bool((new[:, :, L - 5 :] == -1e30).all()) and bool((old[:, :, L - 5 :] == -1e30).all())
    keep = mask.view(1, 1, L).expand(16, L, L)
    assert _rel_rms(new.float()[keep], old.float()[keep]) < 1.5e-2
