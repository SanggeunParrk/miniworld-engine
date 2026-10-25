"""The portable kernels of the backbone pair embedding (``kernels/pair_embed_geom/triton``) on any card: forward values, the table gradient, the fused attention-bias
LayerNorm (``ln_eps``) and the decoder's CA-distance pair, against the dense PyTorch builders and against the kernels' own bins; masked / ragged / multi-chain
inputs; bitwise determinism of the backward (also across every candidate launch configuration). The A/B switch puts every call on the portable kernels
(H100 included); on H100 the CUDA kernels are compared in ``tests/gpu/sm90/test_pair_ops_portable_vs_cuda_gpu.py``."""
import pytest
import torch
import torch.nn.functional as F
from settings_helpers import forced_triton as portable_switch   # the internal A/B switch: the one place the tests name it

from miniworld_engine.kernels.pair_embed_geom import (
    pair_embed_ca,
    pair_embed_ca_reference,
    pair_embed_geom,
    pair_embed_geom_reference,
    pair_embed_geom_train,
    refusal,
    refusal_ca,
)

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def _backbone(B, n, g):
    steps = torch.randn(B, n, 3, device="cuda", generator=g)
    ca = (steps / steps.norm(dim=-1, keepdim=True) * 0.38).cumsum(1)
    off = torch.randn(B, n, 3, 3, device="cuda", generator=g) * 0.12
    return torch.stack([ca + off[:, :, 0], ca, ca + off[:, :, 1], ca + off[:, :, 2]], 2)


def _inputs(B, n, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    bb = _backbone(B, n, g)
    lengths = torch.randint(max(1, n // 2), n + 1, (B,), device="cuda", generator=g)
    mask = torch.arange(n, device="cuda")[None] < lengths[:, None]
    ca = mask & (torch.rand(B, n, device="cuda", generator=g) > 0.03)
    cb = ca & (torch.rand(B, n, device="cuda", generator=g) > 0.1)
    bb[..., 3, :] = bb[..., 3, :] * cb[..., None]    # a residue without CB (glycine) has its CB at the origin
    idx = torch.randint(1, 4, (B, n), device="cuda", generator=g).cumsum(1)
    chains = torch.where(mask, torch.randint(0, 3, (B, n), device="cuda", generator=g), torch.full((B, n), -1, device="cuda"))
    return g, bb, ca, cb, mask, idx, chains


def _rows_from_codes(codes, kind):
    """dense [B, n, n, NT] fp32 one-hot (x chain product) rebuilt from the kernel's own codes"""
    B, n = codes.shape[:2]
    w0, w1 = codes[..., 0].long() & 0xFFFFFFFF, codes[..., 1].long() & 0xFFFFFFFF
    cvf = codes[..., 2].contiguous().view(torch.float32)
    f = torch.zeros(B, n, n, 317 if kind == "geom" else 157, device=codes.device)
    live = (w0 & (1 << 31)) != 0
    f.scatter_add_(-1, (w0 & 127)[..., None], live[..., None].float())
    if kind == "geom":
        for base, sh in [(127, 8), (148, 13), (169, 18), (190, 23)]:
            sel = live & ((w0 & (1 << 28)) != 0)
            f.scatter_add_(-1, (base + ((w0 >> sh) & 31))[..., None], sel[..., None].float())
        for k in range(5):
            sel = live & ((w0 & (1 << 29)) != 0)
            f.scatter_add_(-1, (211 + 21 * k + ((w1 >> (5 * k)) & 31))[..., None], sel[..., None].float())
        f[..., 316] = cvf * live.float()
    else:
        f.scatter_add_(-1, (127 + ((w0 >> 8) & 31))[..., None], live[..., None].float())
    return f


def _padded_pairs_are_zero(out, mask):
    pm = (mask[:, :, None] & mask[:, None, :])[..., None]
    assert not out[~pm.expand_as(out)].any(), "padded pairs must be exactly zero"


SHAPES = [(2, 64), (3, 100), (2, 200), (1, 257), (1, 1), (2, 2), (48, 24)]


@pytest.mark.parametrize(("B", "n"), SHAPES)
def test_forward_matches_the_dense_builder(B, n):
    g, bb, ca, cb, mask, idx, chains = _inputs(B, n, B * n)
    w = torch.randn(256, 317, device="cuda", generator=g) * 0.3
    with portable_switch():
        assert refusal(w, bb, ca, cb, mask, idx, chains) is None
        out = pair_embed_geom(w, bb, ca, cb, mask, idx, chains)
    assert out.dtype == torch.bfloat16 and out.shape == (B, n, n, 256)
    ref = pair_embed_geom_reference(w, bb, ca, cb, mask, idx, chains)
    bad = (out.float() - ref).abs() > 0.02 + 0.02 * ref.abs()
    frac = float(bad.any(-1).float().mean())      # a bin flipped by a last-bit difference of a distance / angle is allowed for a tiny fraction of the pairs
    # the fp32 reference bins a handful of pairs into the neighbouring bin of the autocast kernels (fewer than 1 % of the pairs of this random backbone at the worst shape;
    # on H100 the portable kernels' bins are asserted bit-identical to the CUDA kernels' in tests/gpu/sm90/test_pair_ops_portable_vs_cuda_gpu.py)
    assert frac < 1e-2, f"{frac:.2e} of the pairs differ"
    _padded_pairs_are_zero(out, mask)


@pytest.mark.parametrize(("B", "n"), [(2, 64), (3, 100), (1, 257), (2, 2)])
def test_geom_train_forward_and_weight_gradient(B, n):
    from miniworld_engine.kernels.pair_embed_geom import triton as pt

    g, bb, ca, cb, mask, idx, chains = _inputs(B, n, B * n + 1)
    w = (torch.randn(256, 317, device="cuda", generator=g) * 0.3).requires_grad_()
    dout = torch.randn(B, n, n, 256, device="cuda", generator=g).bfloat16()
    with portable_switch():
        out = pair_embed_geom_train(w, bb, ca, cb, mask, idx, chains)
        assert out.dtype == torch.bfloat16 and out.requires_grad
        out.backward(dout)
        dw = w.grad.clone()
        w3 = w.detach().clone().requires_grad_()
        pair_embed_geom_train(w3, bb, ca, cb, mask, idx, chains).backward(dout)
    assert torch.equal(w3.grad, dw), "the backward must be bitwise deterministic"
    w2 = w.detach().clone().requires_grad_()
    ref = pair_embed_geom_reference(w2, bb, ca, cb, mask, idx, chains)
    ref.backward(dout.float())
    rel = float((dw - w2.grad).norm() / w2.grad.norm())
    frac = float((((out.float() - ref.detach()).abs() > 0.02 + 0.02 * ref.detach().abs()).any(-1)).float().mean())
    assert frac < 2e-3, f"{frac:.2e} of the pairs differ in the forward"
    assert rel < 1e-2 + 3 * frac**0.5, f"weight gradient differs from the dense builder's by {rel:.2e} ({frac:.1e} of the pairs differ)"
    # the kernels' own bins: the gradient is the dense product up to fp32 summation order and one bf16 rounding
    tbl = w.detach()[:, :317].t().to(torch.bfloat16).contiguous()
    _, codes = pt.portable_forward(tbl, bb.float().contiguous(), ca.to(torch.uint8), cb.to(torch.uint8), mask.to(torch.uint8), idx.to(torch.int32),
                          chains.to(torch.int32), torch.linspace(0.1, 2, 20, device="cuda"), torch.linspace(-torch.pi, torch.pi, 20, device="cuda"),
                          with_codes=True)
    exact = torch.einsum("bijr,bijc->rc", _rows_from_codes(codes, "geom"), dout.float())
    got = pt.portable_backward(dout.contiguous(), codes, 0).float()
    assert float((got - exact).abs().max() / exact.abs().max()) < 8e-3
    assert torch.equal(pt.portable_backward(dout.contiguous(), codes, 0).float(), got)


@pytest.mark.parametrize(("B", "n"), [(2, 64), (3, 100), (1, 257), (2, 1), (48, 24)])
def test_ca_pair_forward_and_weight_gradient(B, n):
    from miniworld_engine.kernels.pair_embed_geom import triton as pt

    g, bb, ca, cb, mask, idx, chains = _inputs(B, n, B * n + 2)
    cax = bb[:, :, 1] * mask[..., None]
    w = (torch.randn(256, 157, device="cuda", generator=g) * 0.3).requires_grad_()
    dout = torch.randn(B, n, n, 256, device="cuda", generator=g).bfloat16()
    with portable_switch():
        assert refusal_ca(w, cax, mask, None) is None
        out = pair_embed_ca(w, cax, mask)
        out.backward(dout)
        dw = w.grad.clone()
        w3 = w.detach().clone().requires_grad_()
        pair_embed_ca(w3, cax, mask).backward(dout)
    assert torch.equal(w3.grad, dw), "the backward must be bitwise deterministic"
    w2 = w.detach().clone().requires_grad_()
    ref = pair_embed_ca_reference(w2, cax, mask)
    ref.backward(dout.float())
    rel = float((dw - w2.grad).norm() / w2.grad.norm())
    frac = float((((out.float() - ref.detach()).abs() > 0.02 + 0.02 * ref.detach().abs()).any(-1)).float().mean())
    assert frac < 2e-3 and rel < 1e-2 + 3 * frac**0.5, (frac, rel)
    _padded_pairs_are_zero(out, mask)
    tbl = w.detach().t().to(torch.bfloat16).contiguous()
    _, codes = pt.portable_forward_ca(tbl, cax.float().contiguous(), mask.to(torch.uint8), torch.arange(1, n + 1, device="cuda").expand(B, n).to(torch.int32).contiguous(),
                             torch.linspace(0.1, 3.0, 29, device="cuda"), with_codes=True)
    exact = torch.einsum("bijr,bijc->rc", _rows_from_codes(codes, "ca"), dout.float())
    assert float((pt.portable_backward(dout.contiguous(), codes, 1).float() - exact).abs().max() / exact.abs().max()) < 8e-3


@pytest.mark.parametrize(("kind", "B", "n"), [("geom", 2, 64), ("geom", 3, 100), ("geom", 1, 257), ("ca", 2, 200), ("ca", 3, 100), ("geom", 1, 1)])
def test_layernorm_fused_in_the_pair_builder(kind, B, n):
    """`ln_eps`: LayerNorm(pair) (no affine) comes out of the kernel and the pair is never stored; values and the weight gradient agree with torch's layer_norm of the
    same builder's pair, padded pairs are exactly zero, the backward is bitwise reproducible."""
    g, bb, ca, cb, mask, idx, chains = _inputs(B, n, B * n + 5)
    w = (torch.randn(256, 317 if kind == "geom" else 157, device="cuda", generator=g) * 0.3).requires_grad_()
    dout = torch.randn(B, n, n, 256, device="cuda", generator=g).bfloat16()
    cax = bb[:, :, 1] * mask[..., None]
    build = (lambda w_, **kw: pair_embed_geom_train(w_, bb, ca, cb, mask, idx, chains, **kw)) if kind == "geom" else (lambda w_, **kw: pair_embed_ca(w_, cax, mask, **kw))
    with portable_switch():
        hat = build(w, ln_eps=1e-5)
        hat.backward(dout)
        dw = w.grad.clone()
        w2 = w.detach().clone().requires_grad_()
        ref = F.layer_norm(build(w2), (256,), None, None, 1e-5)
        ref.backward(dout)
        w3 = w.detach().clone().requires_grad_()
        build(w3, ln_eps=1e-5).backward(dout)
    err = (hat.float() - ref.float()).abs()
    assert float((err > 0.02 + 0.02 * ref.float().abs()).float().mean()) < 1e-4, f"LayerNorm output differs: max {float(err.max()):.3e}"
    _padded_pairs_are_zero(hat, mask)
    assert float((dw - w2.grad).norm() / w2.grad.norm()) < 2e-2
    assert torch.equal(w3.grad, dw), "the backward must be bitwise deterministic"


def test_a_weight_that_needs_a_gradient_is_refused_by_the_forward_only_entry():
    g, bb, ca, cb, mask, idx, chains = _inputs(1, 16, 3)
    w = (torch.randn(256, 317, device="cuda", generator=g) * 0.3).requires_grad_()
    with portable_switch():
        assert "forward only" in (refusal(w, bb, ca, cb, mask, idx, chains) or "")
        assert refusal(w, bb, ca, cb, mask, idx, chains, train=True) is None
        assert "[256" in (refusal(w[:, :300], bb, ca, cb, mask, idx, chains, train=True) or "")


def _tuners():
    from miniworld_engine.kernels.pair_embed_geom.triton import main

    return {"bins": main._geom_bins, "ca_bins": main._ca_bins, "gather": main._geom_fwd_gather, "dw": main._geom_dw, "reduce": main._geom_dw_reduce,
            "ln_bwd": main._geom_ln_bwd}


@pytest.mark.parametrize("which", ["bins", "ca_bins", "gather", "dw", "reduce", "ln_bwd"])
def test_every_candidate_config_is_correct_and_gives_the_same_bits(which, monkeypatch):
    """The autotuner may pick any config of the default set on a card nobody has tuned: each one, alone, must give the right numbers, and -- the summation order of
    the table gradient does not depend on the tile -- the same bits as every other config."""
    from miniworld_engine.kernels.pair_embed_geom import triton as pt

    B, n = 2, 100
    g, bb, ca, cb, mask, idx, chains = _inputs(B, n, 11)
    w = torch.randn(256, 317, device="cuda", generator=g) * 0.3
    tbl = w.t().to(torch.bfloat16).contiguous()
    args = (tbl, bb.float().contiguous(), ca.to(torch.uint8), cb.to(torch.uint8), mask.to(torch.uint8), idx.to(torch.int32), chains.to(torch.int32),
            torch.linspace(0.1, 2, 20, device="cuda"), torch.linspace(-torch.pi, torch.pi, 20, device="cuda"))
    cax = (bb[:, :, 1] * mask[..., None]).float().contiguous()
    tc = (torch.randn(256, 157, device="cuda", generator=g) * 0.3).t().to(torch.bfloat16).contiguous()
    dout = torch.randn(B, n, n, 256, device="cuda", generator=g).bfloat16()
    tuner = _tuners()[which]
    configs = list(tuner.configs)
    assert len(configs) >= 3

    def run():
        out, codes = pt.portable_forward(*args, with_codes=True)
        outl = pt.portable_forward(*args, ln_eps=1e-5)
        oc, kc = pt.portable_forward_ca(tc, cax, mask.to(torch.uint8), idx.to(torch.int32), torch.linspace(0.1, 3.0, 29, device="cuda"), with_codes=True)
        return [out, codes, outl, oc, pt.portable_backward(dout, codes, 0), pt.portable_backward(dout, kc, 1), pt.portable_ln_backward(tbl, codes, dout, 0, 1e-5)]

    first = None
    for config in configs:
        monkeypatch.setattr(tuner, "configs", [config])
        monkeypatch.setattr(tuner, "cache", {})
        got = run()
        if first is None:
            first = got
            continue
        for a, e in zip(got, first, strict=True):
            if which in ("dw", "reduce", "bins", "ca_bins"):
                assert torch.equal(a, e), config
            else:
                assert float((a.float() - e.float()).norm() / e.float().norm().clamp_min(1e-12)) < 1e-5, config


def test_the_portable_ops_compile_and_capture():
    """The portable launchers are registered as custom ops with fakes, like the CUDA ones: a compiled training step (forward + backward, the LayerNorm folded in) gives the eager numbers."""
    B, n = 2, 64
    g, bb, ca, cb, mask, idx, chains = _inputs(B, n, 17)
    w = (torch.randn(256, 317, device="cuda", generator=g) * 0.3).requires_grad_()
    dout = torch.randn(B, n, n, 256, device="cuda", generator=g).bfloat16()

    def step(w_):
        out = pair_embed_geom_train(w_, bb, ca, cb, mask, idx, chains, ln_eps=1e-5)
        return out, torch.autograd.grad(out, (w_,), dout)[0]

    with portable_switch():
        eager = step(w)
        compiled = torch.compile(step)(w)
    assert torch.equal(compiled[0], eager[0]) and torch.equal(compiled[1], eager[1])
