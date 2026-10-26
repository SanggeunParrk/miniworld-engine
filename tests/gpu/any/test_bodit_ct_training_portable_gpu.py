"""The three Triton / cuBLAS pieces the portable bias-only DiT and ConditionedTransition TRAINING steps gained (2026-10-26):

* ``kernels._cast.cast_pack``: the fp32 master weights as bf16 in ONE launch (row blocks of the first ``n_cat`` land concatenated).
* ``kernels.bias_only_attention.triton.core_t``: the attention core in feature-major layout (value / gate GEMM, softmax(bias) v, gate, to_out)
  with no permute copy; forward AND backward against the fp64 einsum statements, the bf16 error held to the same-dtype PyTorch
  statements' own (x 1.5, floor 3e-3).
* ``kernels.bias_only_attention.triton.pair_bias``: LayerNorm (no offset) + the ``C -> H`` bias projection in one pass over the pair, forward and
  backward (the pair, the LayerNorm weight and the projection weight gradients), against the same fp64 statements.

Everything runs on any card with Triton (no H100 needed): the master-weight casts are driven directly.
"""
from __future__ import annotations

import pytest
import torch
from portable_helpers import rel

from miniworld_engine.kernels._cast import cast_pack
from miniworld_engine.kernels.bias_only_attention.triton.core_t import bias_only_core_t
from miniworld_engine.kernels.bias_only_attention.triton.pair_bias import pair_bias_logits, supported

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]
DEV, BF, F32 = "cuda", torch.bfloat16, torch.float32


@pytest.mark.parametrize("sizes", [(768, 768, 768, 384, 7), (1536, 1536, 768, 3, 1), (5, 5, 4095, 4097, 8192)],
                         ids=["adaln-like", "ct-like", "ragged"])
@pytest.mark.parametrize("n_cat", [1, 2])
def test_cast_pack_is_the_cast_of_each_weight(sizes, n_cat):
    torch.manual_seed(0)
    srcs = [torch.randn(s, 64, device=DEV) for s in sizes[:2]] + [torch.randn(s, device=DEV) for s in sizes[2:]]
    out = cast_pack(srcs, n_cat)
    assert len(out) == 1 + len(srcs) - n_cat
    assert torch.equal(out[0], torch.cat([s.to(BF) for s in srcs[:n_cat]], 0))
    for o, s in zip(out[1:], srcs[n_cat:], strict=True):
        assert o.dtype == BF and o.shape == s.shape and torch.equal(o, s.to(BF))


def _core_reference(x, wv, wg, wo, p, A, L):
    """The token-major statements: value / gate GEMM, the einsum over the keys, the sigmoid gate, to_out."""
    c = wv.shape[0]
    h = p.shape[0]
    vg = x @ torch.cat([wv, wg], 0).t()
    v, g = vg[:, :c], vg[:, c:]
    v = v.reshape(A, L, h, c // h)
    out = torch.einsum("hij,ajhd->aihd", p, v).reshape(A * L, c)
    return (torch.sigmoid(g) * out) @ wo.t()


@pytest.mark.parametrize("dtype", [BF, F32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("shape", [(3, 64, 16, 48), (2, 100, 12, 64), (4, 48, 24, 32), (1, 128, 8, 96)],
                         ids=["16x48", "12x64-ragged", "24x32", "8x96-one-sample"])
def test_core_t_matches_the_einsum_statements(dtype, shape):
    A, L, H, Dh = shape
    d, c = 256, H * Dh
    g = torch.Generator(device=DEV).manual_seed(1)
    rn = lambda *s, sc=1.0: torch.randn(*s, device=DEV, generator=g, dtype=F32) * sc   # noqa: E731
    x, wv, wg, wo = rn(A * L, d), rn(c, d, sc=d**-0.5), rn(c, d, sc=d**-0.5), rn(d, c, sc=c**-0.5)
    p = torch.softmax(rn(H, L, L), -1)
    dy = rn(A * L, d)

    def run(dt, master):
        xs, ps = x.to(dt).requires_grad_(True), p.to(dt).requires_grad_(True)
        ws = [w.to(F32 if master else dt).requires_grad_(True) for w in (wv, wg, wo)]
        return xs, ps, ws

    # fp64 gold and the same-dtype PyTorch statements
    xg = x.double().requires_grad_(True)
    pg = p.double().requires_grad_(True)
    wsg = [w.double().requires_grad_(True) for w in (wv, wg, wo)]
    og = _core_reference(xg, *wsg, pg, A, L)
    gg = torch.autograd.grad(og, [xg, *wsg, pg], dy.double())
    for master in ([False, True] if dtype is BF else [False]):
        xr, pr, wsr = run(dtype, False)
        orf = _core_reference(xr, *wsr, pr, A, L)
        gr = torch.autograd.grad(orf, [xr, *wsr, pr], dy.to(dtype))
        xe, pe, wse = run(dtype, master)
        oe = bias_only_core_t(xe, *wse, pe)
        ge = torch.autograd.grad(oe, [xe, *wse, pe], dy.to(dtype))
        floor = 3e-3 if dtype is BF else 6e-3
        factor = 1.5 if dtype is BF else 1.0
        assert rel(oe, og) <= max(factor * rel(orf, og), floor), (rel(oe, og), rel(orf, og))
        for i, (a, r, ref) in enumerate(zip(ge, gg, gr, strict=True)):
            assert rel(a, r) <= max(factor * rel(ref, r), 2 * floor), (i, master, rel(a, r), rel(ref, r))
        # a gradient comes back in its leaf's dtype (fp32 master weights keep fp32 gradients)
        for leaf, gr_ in zip([xe, *wse, pe], ge, strict=True):
            assert gr_.dtype == leaf.dtype


def _bias_reference(pair, gamma, weight, eps):
    c = pair.shape[-1]
    xh = torch.nn.functional.layer_norm(pair, (c,), None, None, eps)
    return (xh * gamma) @ weight.t()                                     # [B, L, L, H]


@pytest.mark.parametrize("dtype", [BF, F32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("case", [(1, 96, 128, 16), (2, 40, 128, 12), (1, 64, 64, 24), (1, 33, 256, 32)],
                         ids=["B1-H16-C128", "B2-H12-ragged", "H24-C64", "H32-C256-odd-L"])
def test_pair_bias_matches_the_layernorm_and_projection(dtype, case):
    B, L, C, H = case
    g = torch.Generator(device=DEV).manual_seed(2)
    rn = lambda *s, sc=1.0: torch.randn(*s, device=DEV, generator=g, dtype=F32) * sc   # noqa: E731
    pair, gamma, w = rn(B, L, L, C) * 2 + 0.5, 1 + 0.1 * rn(C), rn(H, C, sc=C**-0.5)
    dl = rn(B, H, L, L)
    assert supported(pair.to(dtype), H)

    def grads(fn, dt, gdt):
        pr, gm, wt = pair.to(dt).requires_grad_(True), gamma.to(gdt).requires_grad_(True), w.to(gdt).requires_grad_(True)
        out = fn(pr, gm, wt)
        return out, torch.autograd.grad(out, [pr, gm, wt], dl.to(out.dtype))

    gold_out, gold_g = grads(lambda p_, g_, w_: _bias_reference(p_, g_, w_, 1e-5).permute(0, 3, 1, 2), torch.float64, torch.float64)
    ref_out, ref_g = grads(lambda p_, g_, w_: _bias_reference(p_.float(), g_.float(), w_.float(), 1e-5).permute(0, 3, 1, 2).to(dtype),
                           dtype, F32)
    out, g_ = grads(lambda p_, g_, w_: pair_bias_logits(p_, g_, w_, 1e-5), dtype, F32)
    assert out.shape == (B, H, L, L) and out.dtype == dtype
    floor = 3e-3 if dtype is BF else 6e-3
    factor = 1.5 if dtype is BF else 1.0
    assert rel(out, gold_out) <= max(factor * rel(ref_out, gold_out), floor), (rel(out, gold_out), rel(ref_out, gold_out))
    for i, (a, r, ref) in enumerate(zip(g_, gold_g, ref_g, strict=True)):
        assert rel(a, r) <= max(factor * rel(ref, r), 2 * floor), (i, rel(a, r), rel(ref, r))
