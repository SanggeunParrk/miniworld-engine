"""The fused pair builder (kernels/pair_embed_adaln) agrees with the dense PyTorch builder: the pair, and the gradient of every
differentiable input (the Linear weight, the LayerNorm affine, the per-sample gamma / beta), with and without self-conditioning,
masked proteins and lengths that are not multiples of the tile; the backward is bitwise deterministic."""
import pytest
import torch

from miniworld_engine.kernels.pair_embed_adaln import pair_embed_adaln, pair_embed_adaln_reference, refusal

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="Hopper (sm_90) required"
    ),
]


def _rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _inputs(B, n, with_sc, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", generator=g)
    w = (r(256, 217) * 0.3).requires_grad_()
    lnw, lnb = (1 + 0.1 * r(256)).requires_grad_(), (0.1 * r(256)).requires_grad_()
    gamma = torch.sigmoid(r(B, 256)).requires_grad_()
    beta = (0.2 * r(B, 256)).requires_grad_()
    xt = r(B, n, 3) * 0.8
    sc = r(B, n, 3) * 0.8 if with_sc else None
    gaps = torch.randint(1, 4, (B, n), device="cuda", generator=g)
    idx = gaps.cumsum(1)
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
def test_matches_the_dense_builder(B, n, with_sc):
    params, xt, sc, idx, mask, wout = _inputs(B, n, with_sc)
    w, lnw, lnb, gamma, beta = params
    assert refusal(w, lnw, lnb, gamma, beta, xt, sc, idx, mask) is None
    pm = (mask[:, :, None] & mask[:, None, :])[..., None]
    wout = wout * pm
    ref_out, ref_g = _grads(lambda: pair_embed_adaln_reference(w, lnw, lnb, gamma, beta, xt, sc, idx, mask), params, wout)
    out, g = _grads(lambda: pair_embed_adaln(w, lnw, lnb, gamma, beta, xt, sc, idx, mask), params, wout)
    assert out.shape == ref_out.shape
    assert _rel(out, ref_out) < 1e-2, _rel(out, ref_out)
    assert not out[~pm.expand_as(out)].any(), "padded pairs must be exactly zero"
    for name, a, b in zip(("dw", "dln_w", "dln_b", "dgamma", "dbeta"), g, ref_g, strict=True):
        assert torch.isfinite(a).all(), name
        assert _rel(a, b) < 3e-2, f"{name}: {_rel(a, b):.2e}"


def test_backward_is_bitwise_deterministic():
    params, xt, sc, idx, mask, wout = _inputs(3, 150, True, seed=3)
    w, lnw, lnb, gamma, beta = params
    runs = [_grads(lambda: pair_embed_adaln(w, lnw, lnb, gamma, beta, xt, sc, idx, mask), params, wout) for _ in range(3)]
    for o, g in runs[1:]:
        assert torch.equal(o, runs[0][0])
        for a, b in zip(g, runs[0][1], strict=True):
            assert torch.equal(a, b)
