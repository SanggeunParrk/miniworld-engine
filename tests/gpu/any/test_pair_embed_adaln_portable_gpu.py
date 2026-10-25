"""The portable kernels of La-Proteina's initial pair builder (``kernels/pair_embed_adaln/triton``) on any card: the pair and the gradient of every differentiable input
(the Linear weight, the LayerNorm affine, the per-sample gamma / beta) against the dense PyTorch builder, with and without self-conditioning, masked proteins, lengths
that are not multiples of a tile, batches up to 48; bitwise determinism of the backward, also across every candidate launch configuration. The A/B switch puts every
call on the portable kernels (H100 included); the comparison with the CUDA kernels is in ``tests/gpu/sm90/test_pair_ops_portable_vs_cuda_gpu.py``."""
import pytest
import torch
from settings_helpers import forced_triton as portable_switch   # the internal A/B switch: the one place the tests name it

from miniworld_engine.kernels.pair_embed_adaln import pair_embed_adaln, pair_embed_adaln_reference, refusal

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


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
    idx = torch.randint(1, 4, (B, n), device="cuda", generator=g).cumsum(1)
    lengths = torch.randint(max(1, n // 2), n + 1, (B,), device="cuda", generator=g)
    mask = torch.arange(n, device="cuda")[None] < lengths[:, None]
    return (w, lnw, lnb, gamma, beta), xt, sc, idx, mask, r(B, n, n, 256)


def _grads(fn, params, wout):
    out = fn()
    (out.float() * wout).sum().backward()
    gs = [p.grad.clone() for p in params]
    for p in params:
        p.grad = None
    return out.detach().float(), gs


@pytest.mark.parametrize(("B", "n", "with_sc"), [(2, 64, True), (3, 100, True), (2, 200, False), (4, 257, True), (1, 33, False), (1, 1, True), (2, 2, False), (48, 24, True)])
def test_matches_the_dense_builder(B, n, with_sc):
    params, xt, sc, idx, mask, wout = _inputs(B, n, with_sc)
    w, lnw, lnb, gamma, beta = params
    pm = (mask[:, :, None] & mask[:, None, :])[..., None]
    wout = wout * pm
    ref_out, ref_g = _grads(lambda: pair_embed_adaln_reference(w, lnw, lnb, gamma, beta, xt, sc, idx, mask), params, wout)
    with portable_switch():
        assert refusal(w, lnw, lnb, gamma, beta, xt, sc, idx, mask) is None
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
    with portable_switch():
        runs = [_grads(lambda: pair_embed_adaln(w, lnw, lnb, gamma, beta, xt, sc, idx, mask), params, wout) for _ in range(3)]
    for o, g in runs[1:]:
        assert torch.equal(o, runs[0][0])
        for a, b in zip(g, runs[0][1], strict=True):
            assert torch.equal(a, b)


def test_other_widths_and_ranges():
    """A table of other widths (the bin counts are arguments, not constants), the distance ranges of the model's config, bf16 ``w``."""
    g = torch.Generator(device="cuda").manual_seed(5)
    r = lambda *s: torch.randn(*s, device="cuda", generator=g)
    B, n, nS, nX, nC = 2, 70, 63, 17, 40
    w = (r(256, nS + nX + nC) * 0.3).requires_grad_()
    lnw, lnb, gamma, beta = (1 + 0.1 * r(256)), 0.1 * r(256), torch.sigmoid(r(B, 256)), 0.2 * r(B, 256)
    xt, sc, idx = r(B, n, 3), r(B, n, 3), torch.arange(1, n + 1, device="cuda").expand(B, n)
    mask = torch.ones(B, n, dtype=torch.bool, device="cuda")
    kw = dict(n_sep=nS, n_xt=nX, n_sc=nC, xt_range=(0.2, 2.5), sc_range=(0.1, 3.5))
    ref = pair_embed_adaln_reference(w, lnw, lnb, gamma, beta, xt, sc, idx, mask, **kw)
    with portable_switch():
        out = pair_embed_adaln(w, lnw, lnb, gamma, beta, xt, sc, idx, mask, **kw)
    assert _rel(out.float(), ref) < 1e-2


def _tuners():
    from miniworld_engine.kernels.pair_embed_adaln.triton import main

    return {"fwd": main._adaln_fwd, "bwd": main._adaln_bwd, "dw": main._adaln_dw, "reduce": main._adaln_bwd_reduce}


@pytest.mark.parametrize("which", ["fwd", "bwd", "dw", "reduce"])
def test_every_candidate_config_is_correct_and_reproducible(which, monkeypatch):
    """The autotuner may pick any config of the default set on a card nobody has tuned: each one, alone, must give the right numbers and the same bits twice; the
    table-gradient kernels (their summation order does not depend on the tile) also the same bits as every other config."""
    tuner = _tuners()[which]
    configs = list(tuner.configs)
    assert len(configs) >= 3
    params, xt, sc, idx, mask, wout = _inputs(2, 100, True, seed=7)
    w, lnw, lnb, gamma, beta = params
    pm = (mask[:, :, None] & mask[:, None, :])[..., None]
    wout = wout * pm
    ref_out, ref_g = _grads(lambda: pair_embed_adaln_reference(w, lnw, lnb, gamma, beta, xt, sc, idx, mask), params, wout)
    first = None
    for config in configs:
        monkeypatch.setattr(tuner, "configs", [config])
        monkeypatch.setattr(tuner, "cache", {})
        with portable_switch():
            runs = [_grads(lambda: pair_embed_adaln(w, lnw, lnb, gamma, beta, xt, sc, idx, mask), params, wout) for _ in range(2)]
        for a, b in zip((runs[0][0], *runs[0][1]), (runs[1][0], *runs[1][1]), strict=True):
            assert torch.equal(a, b), config
        assert _rel(runs[0][0], ref_out) < 1e-2, config
        for name, a, b in zip(("dw", "dln_w", "dln_b", "dgamma", "dbeta"), runs[0][1], ref_g, strict=True):
            assert _rel(a, b) < 3e-2, (config, name)
        if which in ("dw", "reduce"):
            first = first or runs[0]
            if which == "dw":
                assert torch.equal(runs[0][1][0], first[1][0]), config
