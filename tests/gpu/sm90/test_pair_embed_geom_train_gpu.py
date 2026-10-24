"""Training path of the fused pair embeddings (kernels/pair_embed_geom): the encoder's backbone pair (``pair_embed_geom_train``) and the decoder's
CA-distance pair (``pair_embed_ca``) are differentiable in the Linear weight.  Checked against the dense PyTorch builders (forward values, the
weight gradient) and against a dense gradient built from the kernel's own bins (exact up to fp32 summation order), for padded / masked / multi-chain
inputs, and for bitwise determinism of the backward."""
import pytest
import torch

from miniworld_engine.kernels.pair_embed_geom import (
    pair_embed_ca,
    pair_embed_ca_reference,
    pair_embed_geom_reference,
    pair_embed_geom_train,
)
from miniworld_engine.kernels.pair_embed_geom.cuda import sm90 as _sm90

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="Hopper (sm_90) required"
    ),
]


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


def _rows_from_codes(codes, chains, kind):
    """dense [B, n, n, NT] fp32 one-hot (x chain product) rebuilt from the kernel's own codes"""
    B, n = codes.shape[:2]
    w0, w1 = codes[..., 0].long() & 0xFFFFFFFF, codes[..., 1].long() & 0xFFFFFFFF
    w0 = w0 ^ (1 << 31)   # bit 31 of the stored word is the live bit; flip it back to the kernel's "masked" convention
    cvf = codes[..., 2].contiguous().view(torch.float32)
    nt = 317 if kind == "geom" else 157
    f = torch.zeros(B, n, n, nt, device=codes.device)
    live = (w0 & (1 << 31)) == 0
    f.scatter_add_(-1, (w0 & 127)[..., None], live[..., None].float())
    if kind == "geom":
        for base, sh, flag in [(127, 8, 1 << 28), (148, 13, 1 << 28), (169, 18, 1 << 28), (190, 23, 1 << 28)]:
            sel = live & ((w0 & flag) != 0)
            f.scatter_add_(-1, (base + ((w0 >> sh) & 31))[..., None], sel[..., None].float())
        for k in range(5):
            sel = live & ((w0 & (1 << 29)) != 0)
            f.scatter_add_(-1, (211 + 21 * k + ((w1 >> (5 * k)) & 31))[..., None], sel[..., None].float())
        f[..., 316] = cvf * live.float()
    else:
        f.scatter_add_(-1, (127 + ((w0 >> 8) & 31))[..., None], live[..., None].float())
    return f


@pytest.mark.parametrize(("B", "n"), [(2, 64), (3, 100), (2, 200), (1, 257)])
def test_geom_train_forward_and_weight_gradient(B, n):
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
    # a pair whose angle / distance falls into the neighbouring bin moves its whole 256-channel gradient row to another table row: with random
    # gradients the Frobenius error grows like sqrt(flipped pairs); the exact check against the kernel's own bins is below
    assert rel < 1e-2 + 3 * frac**0.5, f"weight gradient differs from the dense builder's by {rel:.2e} (relative, Frobenius; {frac:.1e} of the pairs differ)"

    # the kernel's own bins: the gradient is the dense product up to fp32 summation order and one bf16 rounding
    w3 = w.detach().clone().requires_grad_()
    tbl = w3[:, :317].t().to(torch.bfloat16).contiguous()
    o2, codes = _sm90.forward(tbl, bb.float().contiguous(), ca.to(torch.uint8), cb.to(torch.uint8), mask.to(torch.uint8), idx.to(torch.int32),
                              chains.to(torch.int32), torch.linspace(0.1, 2, 20, device="cuda"), torch.linspace(-torch.pi, torch.pi, 20, device="cuda"),
                              with_codes=True)
    f = _rows_from_codes(codes, chains, "geom")
    exact = torch.einsum("bijr,bijc->rc", f, dout.float())
    got = _sm90.backward(dout.contiguous(), codes, 0).float()
    err = float((got - exact).abs().max() / exact.abs().max())
    assert err < 8e-3, f"table gradient vs the dense product of the kernel's own bins: {err:.2e}"   # one bf16 rounding is 2^-9 = 2e-3
    again = _sm90.backward(dout.contiguous(), codes, 0)
    assert torch.equal(again.float(), got), "the backward must be bitwise deterministic"
    smem = _sm90.backward(dout.contiguous(), codes, 0, impl=0).float()   # the shared-memory kernel (A/B) agrees with the tensor-core one
    assert float((smem - got).abs().max() / exact.abs().max()) < 8e-3


@pytest.mark.parametrize(("B", "n"), [(2, 64), (3, 100), (2, 200), (1, 257)])
def test_ca_train_forward_and_weight_gradient(B, n):
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
    _, codes = _sm90.forward_ca(tbl, cax.float().contiguous(), mask.to(torch.uint8), torch.arange(1, n + 1, device="cuda").expand(B, n).to(torch.int32).contiguous(),
                                torch.linspace(0.1, 3.0, 29, device="cuda"), with_codes=True)
    f = _rows_from_codes(codes, chains, "ca")
    exact = torch.einsum("bijr,bijc->rc", f, dout.float())
    got = _sm90.backward(dout.contiguous(), codes, 1).float()
    assert float((got - exact).abs().max() / exact.abs().max()) < 8e-3
    again = _sm90.backward(dout.contiguous(), codes, 1)
    assert torch.equal(again.float(), got), "the backward must be bitwise deterministic"
    smem = _sm90.backward(dout.contiguous(), codes, 1, impl=0).float()
    assert float((smem - got).abs().max() / exact.abs().max()) < 8e-3


@pytest.mark.parametrize(("kind", "B", "n"), [("geom", 2, 64), ("geom", 3, 100), ("geom", 1, 257), ("ca", 2, 200), ("ca", 3, 100)])
def test_layernorm_fused_in_the_pair_builder(kind, B, n):
    """`ln_eps`: the kernel returns LayerNorm(pair) (no affine) and never stores the pair; values and the weight gradient agree with torch's layer_norm
    applied to the same kernel's pair (same bins, so only the LayerNorm and its backward differ), padded pairs are exactly zero."""
    import torch.nn.functional as F

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
