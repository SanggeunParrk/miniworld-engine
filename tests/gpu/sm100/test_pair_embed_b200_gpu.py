"""La-Proteina's pair builders on B200 (kernels/pair_embed_geom, kernels/pair_embed_adaln, ``cuda/sm100``): the sm_90a forward gathers built for
sm_100a and the tcgen05 table gradient (pe_tbl_sm100.cuh). Checked as the H100 tests check the sm_90a build: against the dense PyTorch builders
(forward values, every gradient), against a dense gradient built from the kernel's own bins (exact up to fp32 summation order and one bf16 rounding),
the A/B kernels (the sm_90 mma.sync and shared-memory table gradients built for sm_100a) agree, padded pairs are zero, the backward is bitwise
deterministic, and the public calls route to the CUDA kernels on B200 (no PyTorch reference, no portable kernels)."""
import pytest
import torch
import torch.nn.functional as F

from miniworld_engine.kernels.pair_embed_adaln import pair_embed_adaln, pair_embed_adaln_reference
from miniworld_engine.kernels.pair_embed_adaln import refusal as refusal_adaln
from miniworld_engine.kernels.pair_embed_geom import (
    pair_embed_ca,
    pair_embed_ca_reference,
    pair_embed_geom,
    pair_embed_geom_reference,
    pair_embed_geom_train,
    refusal,
    refusal_ca,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100) required"),
]


def _geom():
    from miniworld_engine.kernels.pair_embed_geom.cuda import sm100

    return sm100


def _backbone(B, n, g):
    steps = torch.randn(B, n, 3, device="cuda", generator=g)
    ca = (steps / steps.norm(dim=-1, keepdim=True) * 0.38).cumsum(1)
    off = torch.randn(B, n, 3, 3, device="cuda", generator=g) * 0.12
    return torch.stack([ca + off[:, :, 0], ca, ca + off[:, :, 1], ca + off[:, :, 2]], 2)


def _inputs(B, n, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    bb = _backbone(B, n, g)
    lengths = torch.randint(n // 2, n + 1, (B,), device="cuda", generator=g)
    mask = torch.arange(n, device="cuda")[None] < lengths[:, None]
    ca = mask & (torch.rand(B, n, device="cuda", generator=g) > 0.03)
    cb = ca & (torch.rand(B, n, device="cuda", generator=g) > 0.1)
    bb[..., 3, :] = bb[..., 3, :] * cb[..., None]
    idx = torch.randint(1, 4, (B, n), device="cuda", generator=g).cumsum(1)
    chains = torch.where(mask, torch.randint(0, 3, (B, n), device="cuda", generator=g), torch.full((B, n), -1, device="cuda"))
    return g, bb, ca, cb, mask, idx, chains


def _rows_from_codes(codes, kind):
    """dense [B, n, n, NT] fp32 one-hot (x chain product) rebuilt from the kernel's own codes"""
    B, n = codes.shape[:2]
    w0, w1 = codes[..., 0].long() & 0xFFFFFFFF, codes[..., 1].long() & 0xFFFFFFFF
    w0 = w0 ^ (1 << 31)
    cvf = codes[..., 2].contiguous().view(torch.float32)
    nt = 317 if kind == "geom" else 157
    f = torch.zeros(B, n, n, nt, device=codes.device)
    live = (w0 & (1 << 31)) == 0
    f.scatter_add_(-1, (w0 & 127)[..., None], live[..., None].float())
    if kind == "geom":
        for base, sh in [(127, 8), (148, 13), (169, 18), (190, 23)]:
            sel = live & ((w0 & (1 << 28)) != 0)
            f.scatter_add_(-1, (base + ((w0 >> sh) & 31))[..., None], sel[..., None].float())
        for k in range(5):
            sel = live & ((w0 & (1 << 29)) != 0)
            f.scatter_add_(-1, (211 + 21 * k + ((w1 >> (5 * k)) & 31))[..., None], sel[..., None].float())
        f[..., 316] = cvf.bfloat16().float() * live.float()      # the table gradient reads the chain product as bf16
    else:
        f.scatter_add_(-1, (127 + ((w0 >> 8) & 31))[..., None], live[..., None].float())
    return f


def test_routes_to_the_cuda_kernels():
    g, bb, ca, cb, mask, idx, chains = _inputs(2, 64, 0)
    w = torch.randn(256, 317, device="cuda")
    assert refusal(w, bb, ca, cb, mask, idx, chains) is None
    assert refusal_ca(torch.randn(256, 157, device="cuda"), bb[:, :, 1], mask, idx) is None
    from miniworld_engine.kernels._pair_route import CUDA, route
    from miniworld_engine.kernels.pair_embed_adaln.interface import _CUDA_CAPS as caps_a
    from miniworld_engine.kernels.pair_embed_geom.interface import _CUDA_CAPS as caps_g

    assert route("pair_embed_geom", bb.device, cuda_caps=caps_g) == CUDA
    assert route("pair_embed_adaln", bb.device, cuda_caps=caps_a) == CUDA


@pytest.mark.parametrize(("B", "n"), [(2, 64), (3, 100), (1, 257)])
def test_geom_forward_matches_the_dense_builder(B, n):
    g, bb, ca, cb, mask, idx, chains = _inputs(B, n, B * n)
    w = torch.randn(256, 317, device="cuda", generator=g) * 0.3
    with torch.no_grad():
        out = pair_embed_geom(w, bb, ca, cb, mask, idx, chains)
        ref = pair_embed_geom_reference(w.bfloat16().float(), bb, ca, cb, mask, idx, chains)
    frac = float((((out.float() - ref).abs() > 0.02 + 0.02 * ref.abs()).any(-1)).float().mean())
    assert frac < 2e-3, f"{frac:.2e} of the pairs differ"
    pm = (mask[:, :, None] & mask[:, None, :])[..., None]
    assert not out[~pm.expand_as(out)].any(), "padded pairs must be exactly zero"


@pytest.mark.parametrize(("B", "n"), [(2, 64), (3, 100), (2, 200), (1, 257)])
def test_geom_train_forward_and_weight_gradient(B, n):
    sm = _geom()
    g, bb, ca, cb, mask, idx, chains = _inputs(B, n, B * n + 1)
    w = (torch.randn(256, 317, device="cuda", generator=g) * 0.3).requires_grad_()
    dout = torch.randn(B, n, n, 256, device="cuda", generator=g).bfloat16()
    out = pair_embed_geom_train(w, bb, ca, cb, mask, idx, chains)
    assert out.dtype == torch.bfloat16 and out.requires_grad
    out.backward(dout)
    dw = w.grad.clone()
    w2 = w.detach().clone().requires_grad_()
    ref = pair_embed_geom_reference(w2, bb, ca, cb, mask, idx, chains)
    ref.backward(dout.float())
    rel = float((dw - w2.grad).norm() / w2.grad.norm())
    frac = float((((out.float() - ref.detach()).abs() > 0.02 + 0.02 * ref.detach().abs()).any(-1)).float().mean())
    assert frac < 2e-3, f"{frac:.2e} of the pairs differ in the forward"
    assert rel < 1e-2 + 3 * frac**0.5, f"weight gradient differs from the dense builder's by {rel:.2e} ({frac:.1e} of the pairs differ)"

    tbl = w.detach()[:, :317].t().to(torch.bfloat16).contiguous()
    _, codes = sm.forward(tbl, bb.float().contiguous(), ca.to(torch.uint8), cb.to(torch.uint8), mask.to(torch.uint8), idx.to(torch.int32),
                          chains.to(torch.int32), torch.linspace(0.1, 2, 20, device="cuda"), torch.linspace(-torch.pi, torch.pi, 20, device="cuda"),
                          with_codes=True)
    exact = torch.einsum("bijr,bijc->rc", _rows_from_codes(codes, "geom"), dout.float())
    got = sm.backward(dout.contiguous(), codes, 0).float()
    err = float((got - exact).abs().max() / exact.abs().max())
    assert err < 8e-3, f"table gradient vs the dense product of the kernel's own bins: {err:.2e}"   # one bf16 rounding is 2^-9
    assert torch.equal(sm.backward(dout.contiguous(), codes, 0).float(), got), "the backward must be bitwise deterministic"
    for impl in (0, 2):                                               # the A/B kernels agree with the tcgen05 one
        other = sm.backward(dout.contiguous(), codes, 0, impl=impl).float()
        assert float((other - got).abs().max() / exact.abs().max()) < 8e-3, impl


@pytest.mark.parametrize(("B", "n"), [(2, 64), (3, 100), (2, 200), (1, 257)])
def test_ca_train_forward_and_weight_gradient(B, n):
    sm = _geom()
    g, bb, ca, cb, mask, idx, chains = _inputs(B, n, B * n + 2)
    cax = bb[:, :, 1] * mask[..., None]
    w = (torch.randn(256, 157, device="cuda", generator=g) * 0.3).requires_grad_()
    dout = torch.randn(B, n, n, 256, device="cuda", generator=g).bfloat16()
    out = pair_embed_ca(w, cax, mask)
    out.backward(dout)
    dw = w.grad.clone()
    w2 = w.detach().clone().requires_grad_()
    ref = pair_embed_ca_reference(w2, cax, mask)
    ref.backward(dout.float())
    rel = float((dw - w2.grad).norm() / w2.grad.norm())
    frac = float((((out.float() - ref.detach()).abs() > 0.02 + 0.02 * ref.detach().abs()).any(-1)).float().mean())
    assert frac < 2e-3, f"{frac:.2e} of the pairs differ in the forward"
    assert rel < 1e-2 + 3 * frac**0.5, f"weight gradient differs from the dense builder's by {rel:.2e} ({frac:.1e} of the pairs differ)"
    pm = (mask[:, :, None] & mask[:, None, :])[..., None]
    assert not out[~pm.expand_as(out)].any(), "padded pairs must be exactly zero"
    tbl = w.detach().t().to(torch.bfloat16).contiguous()
    _, codes = sm.forward_ca(tbl, cax.float().contiguous(), mask.to(torch.uint8), torch.arange(1, n + 1, device="cuda").expand(B, n).to(torch.int32).contiguous(),
                             torch.linspace(0.1, 3.0, 29, device="cuda"), with_codes=True)
    exact = torch.einsum("bijr,bijc->rc", _rows_from_codes(codes, "ca"), dout.float())
    got = sm.backward(dout.contiguous(), codes, 1).float()
    assert float((got - exact).abs().max() / exact.abs().max()) < 8e-3
    assert torch.equal(sm.backward(dout.contiguous(), codes, 1).float(), got), "the backward must be bitwise deterministic"
    for impl in (0, 2):
        other = sm.backward(dout.contiguous(), codes, 1, impl=impl).float()
        assert float((other - got).abs().max() / exact.abs().max()) < 8e-3, impl


def test_table_gradient_at_a_training_shape():
    """B 16 x L 256 (4 Mi pairs: about 900 tiles per CTA, every ring slot reused many times), both tables, against the shared-memory kernel."""
    sm = _geom()
    gen = torch.Generator(device="cuda").manual_seed(7)
    B, n = 16, 256
    for mode, nt in ((0, 317), (1, 157)):
        w0 = torch.randint(0, 127, (B, n, n), device="cuda", generator=gen)
        if mode == 0:
            for k in range(4):
                w0 |= torch.randint(0, 21, (B, n, n), device="cuda", generator=gen) << (8 + 5 * k)
            w0 |= (torch.rand(B, n, n, device="cuda", generator=gen) > 0.2).long() << 28
            w0 |= (torch.rand(B, n, n, device="cuda", generator=gen) > 0.3).long() << 29
        else:
            w0 |= torch.randint(0, 30, (B, n, n), device="cuda", generator=gen) << 8
        w0 |= (torch.rand(B, n, n, device="cuda", generator=gen) > 0.1).long() << 31
        codes = torch.zeros(B, n, n, 4, dtype=torch.int32, device="cuda")
        codes[..., 0] = torch.where(w0 >= 2**31, w0 - 2**32, w0).int()
        if mode == 0:
            w1 = torch.zeros_like(w0)
            for k in range(5):
                w1 |= torch.randint(0, 21, (B, n, n), device="cuda", generator=gen) << (5 * k)
            codes[..., 1] = w1.int()
            codes[..., 2] = (torch.randint(0, 3, (B, n, n), device="cuda", generator=gen).float()).view(torch.int32)
        g = torch.randn(B, n, n, 256, device="cuda", generator=gen).bfloat16()
        got = sm.backward(g, codes, mode).float()
        ref = sm.backward(g, codes, mode, impl=0).float()
        assert got.shape == (nt, 256)
        assert float((got - ref).abs().max() / ref.abs().max()) < 4e-3, mode


@pytest.mark.parametrize(("kind", "B", "n"), [("geom", 2, 64), ("geom", 1, 257), ("ca", 2, 200), ("ca", 3, 100)])
def test_layernorm_fused_in_the_pair_builder(kind, B, n):
    g, bb, ca, cb, mask, idx, chains = _inputs(B, n, B * n + 5)
    nrows = 317 if kind == "geom" else 157
    w = (torch.randn(256, nrows, device="cuda", generator=g) * 0.3).requires_grad_()
    dout = torch.randn(B, n, n, 256, device="cuda", generator=g).bfloat16()
    cax = bb[:, :, 1] * mask[..., None]
    build = (lambda w_, **kw: pair_embed_geom_train(w_, bb, ca, cb, mask, idx, chains, **kw)) if kind == "geom" else (lambda w_, **kw: pair_embed_ca(w_, cax, mask, **kw))
    hat = build(w, ln_eps=1e-5)
    hat.backward(dout)
    dw = w.grad.clone()
    w2 = w.detach().clone().requires_grad_()
    ref = F.layer_norm(build(w2), (256,), None, None, 1e-5)
    ref.backward(dout)
    err = (hat.float() - ref.float()).abs()
    assert float((err > 0.02 + 0.02 * ref.float().abs()).float().mean()) < 1e-4, f"LayerNorm output differs: max {float(err.max()):.3e}"
    pm = (mask[:, :, None] & mask[:, None, :])[..., None]
    assert not hat[~pm.expand_as(hat)].any(), "padded pairs must be exactly zero"
    rel = float((dw - w2.grad).norm() / w2.grad.norm())
    assert rel < 2e-2, f"weight gradient differs from torch's LayerNorm backward by {rel:.2e}"
    w3 = w.detach().clone().requires_grad_()
    build(w3, ln_eps=1e-5).backward(dout)
    assert torch.equal(w3.grad, dw), "the backward must be bitwise deterministic"


@pytest.mark.parametrize("ln_eps", [None, 1e-5])
def test_ca_forward_only_call_equals_the_training_forward(ln_eps):
    g, bb, ca, cb, mask, idx, chains = _inputs(2, 64, 9)
    cax = bb[:, :, 1] * mask[..., None]
    w = (torch.randn(256, 157, device="cuda", generator=g) * 0.3).requires_grad_()
    train = pair_embed_ca(w, cax, mask, ln_eps=ln_eps)
    with torch.no_grad():
        fwd_only = pair_embed_ca(w, cax, mask, ln_eps=ln_eps)
    assert torch.equal(fwd_only, train)


# ---- pair_embed_adaln ---------------------------------------------------------------------------------------------------------------
def _rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _adaln_inputs(B, n, with_sc, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", generator=g)
    w = (r(256, 217) * 0.3).requires_grad_()
    lnw, lnb = (1 + 0.1 * r(256)).requires_grad_(), (0.1 * r(256)).requires_grad_()
    gamma = torch.sigmoid(r(B, 256)).requires_grad_()
    beta = (0.2 * r(B, 256)).requires_grad_()
    xt = r(B, n, 3) * 0.8
    sc = r(B, n, 3) * 0.8 if with_sc else None
    idx = torch.randint(1, 4, (B, n), device="cuda", generator=g).cumsum(1)
    lengths = torch.randint(max(2, n // 2), n + 1, (B,), device="cuda", generator=g)
    mask = torch.arange(n, device="cuda")[None] < lengths[:, None]
    wout = r(B, n, n, 256)
    return (w, lnw, lnb, gamma, beta), xt, sc, idx, mask, wout


def _grads(fn, params, wout):
    out = fn()
    (out.float() * wout).sum().backward()
    gs = [p.grad.clone() for p in params]
    for p in params:
        p.grad = None
    return out.detach().float(), gs


@pytest.mark.parametrize(("B", "n", "with_sc"), [(2, 64, True), (3, 100, True), (2, 200, False), (4, 257, True), (1, 33, False)])
def test_adaln_matches_the_dense_builder(B, n, with_sc):
    params, xt, sc, idx, mask, wout = _adaln_inputs(B, n, with_sc)
    w, lnw, lnb, gamma, beta = params
    assert refusal_adaln(w, lnw, lnb, gamma, beta, xt, sc, idx, mask) is None
    pm = (mask[:, :, None] & mask[:, None, :])[..., None]
    wout = wout * pm
    ref_out, ref_g = _grads(lambda: pair_embed_adaln_reference(w, lnw, lnb, gamma, beta, xt, sc, idx, mask), params, wout)
    out, g = _grads(lambda: pair_embed_adaln(w, lnw, lnb, gamma, beta, xt, sc, idx, mask), params, wout)
    assert _rel(out, ref_out) < 1e-2, _rel(out, ref_out)
    assert not out[~pm.expand_as(out)].any(), "padded pairs must be exactly zero"
    for name, a, b in zip(("dw", "dln_w", "dln_b", "dgamma", "dbeta"), g, ref_g, strict=True):
        assert torch.isfinite(a).all(), name
        assert _rel(a, b) < 3e-2, f"{name}: {_rel(a, b):.2e}"


def test_adaln_backward_is_bitwise_deterministic_and_matches_the_ab_kernel():
    from miniworld_engine.kernels.pair_embed_adaln.cuda import sm100

    params, xt, sc, idx, mask, wout = _adaln_inputs(3, 150, True, seed=3)
    w, lnw, lnb, gamma, beta = params
    runs = [_grads(lambda: pair_embed_adaln(w, lnw, lnb, gamma, beta, xt, sc, idx, mask), params, wout) for _ in range(3)]
    for o, g in runs[1:]:
        assert torch.equal(o, runs[0][0])
        for a, b in zip(g, runs[0][1], strict=True):
            assert torch.equal(a, b)
    old = sm100.IMPL
    try:
        sm100.IMPL = 0                                   # the shared-memory table gradient (A/B)
        _, g0 = _grads(lambda: pair_embed_adaln(w, lnw, lnb, gamma, beta, xt, sc, idx, mask), params, wout)
    finally:
        sm100.IMPL = old
    assert _rel(g0[0], runs[0][1][0]) < 1e-5
    for a, b in zip(g0[1:], runs[0][1][1:], strict=True):
        assert torch.equal(a, b)                          # only the table gradient differs between the two
