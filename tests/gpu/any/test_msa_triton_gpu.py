"""The portable Triton OuterProductMean / MSAPairWeightedAveraging against fp32 autograd of the module's own statements.

Run with `force_triton_kernels=True` so the native H100 paths step aside and the Triton kernels are what is measured on any card.
Each error is bounded by the bf16 module's own error on the same inputs (x 1.5, floor 2e-3): the kernels may not be worse than
running the module's statements in bf16. Covered: masks (none / ragged / every key masked), the pair residual, both
normalization orders, non-power-of-two d_pair, batch > 1, row dropout, and one compiled graph for forward + backward.
"""
from __future__ import annotations

import copy

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.modules.exceptions import ImplementationType as I
from miniworld_engine.modules.msa_pair_weighted_averaging import (
    MSAPairWeightedAveraging,
)
from miniworld_engine.modules.outer_product import OuterProductMean

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]
DEV, BF = "cuda", torch.bfloat16


@pytest.fixture(autouse=True)
def triton_policy():
    previous = settings.current()
    settings.configure(force_triton_kernels=True)
    yield
    settings.configure(**vars(previous))


def rel(a, b):
    return (torch.linalg.vector_norm(a.float() - b.float()) / torch.linalg.vector_norm(b.float()).clamp_min(1e-30)).item()


def _randomise(m):
    torch.manual_seed(3)
    with torch.no_grad():
        for name, p in m.named_parameters():
            p.copy_(1.0 + 0.2 * torch.randn_like(p) if name.endswith(("ln_msa.weight", "ln_pair.weight", "product_norm.weight"))
                    else torch.randn_like(p) * (0.2 if p.ndim == 1 else p.shape[-1] ** -0.5))
    return m


def _grads(module, dtype, inputs, dy):
    m = copy.deepcopy(module).to(dtype)
    floating = [t is not None and t.is_floating_point() for t in inputs]
    xs = [t.detach().to(dtype).requires_grad_() if f else t for t, f in zip(inputs, floating, strict=True)]
    y = m(*xs)
    leaves = [t for t, f in zip(xs, floating, strict=True) if f] + list(m.parameters())
    return y, torch.autograd.grad(y, leaves, dy.to(dtype))


def _compare(module, inputs, pytorch_twin):
    """Kernel (bf16) and bf16 statements, each against fp32 statements."""
    y, g = _grads(module, BF, inputs, dy := torch.randn(*_out_shape(module, inputs), device=DEV))
    y32, g32 = _grads(pytorch_twin, torch.float32, inputs, dy)
    y16, g16 = _grads(pytorch_twin, BF, inputs, dy)
    assert rel(y, y32) <= max(2e-3, 1.5 * rel(y16, y32)), (rel(y, y32), rel(y16, y32))
    for i, (a, b, c) in enumerate(zip(g, g32, g16, strict=True)):
        assert rel(a, b) <= max(2e-3, 1.5 * rel(c, b)), (i, rel(a, b), rel(c, b))


def _out_shape(module, inputs):
    msa = inputs[0]
    if isinstance(module, OuterProductMean):
        return (msa.shape[0], msa.shape[2], msa.shape[2], module.to_out.weight.shape[0])
    return tuple(msa.shape)


def _twin(module):
    twin = copy.deepcopy(module)
    twin.implementation = I.PYTORCH
    return twin


@pytest.mark.parametrize("mask_kind", ["none", "ragged", "empty_rows"])
@pytest.mark.parametrize("normalize_before_proj", [True, False])
@pytest.mark.parametrize("d_pair", [128, 384])
def test_outer_product_mean(mask_kind, normalize_before_proj, d_pair):
    torch.manual_seed(0)
    b, s, n = 2, 200, 136
    m = _randomise(OuterProductMean(64, d_pair, 32, normalize_before_proj=normalize_before_proj, implementation=I.MINIWORLD).to(DEV))
    msa = torch.randn(b, s, n, 64, device=DEV)
    mask = {"none": None, "ragged": torch.rand(b, s, n, device=DEV) > 0.3,
            "empty_rows": (torch.rand(b, s, n, device=DEV) > 0.3) & (torch.arange(n, device=DEV) % 7 != 0)}[mask_kind]
    residual = torch.randn(b, n, n, d_pair, device=DEV)
    _compare(m, [msa, mask, None, residual], _twin(m))


@pytest.mark.parametrize("mask_kind", ["none", "ragged", "empty_rows"])
@pytest.mark.parametrize("variant", ["rf3_before", "rf3_after", "chai1", "chai1_wide_pair", "chai1_k4"])
def test_outer_product_mean_variants(monkeypatch, mask_kind, variant):
    """RoseTTAFold3's projection biases and Chai-1's 8 groups (+ product LayerNorm) through the Triton kernels, never the PyTorch
    statements (``_variant_reference`` is made to raise): output and every parameter / input gradient, held to the bf16 statements."""
    options = {"rf3_before": dict(d_pair=128, d_hidden=32, projection_bias=True),
               "rf3_after": dict(d_pair=128, d_hidden=32, projection_bias=True, normalize_before_proj=False),
               "chai1": dict(d_pair=128, d_hidden=64, groups=8),
               "chai1_wide_pair": dict(d_pair=256, d_hidden=64, groups=8),
               "chai1_k4": dict(d_pair=128, d_hidden=32, groups=8)}[variant]
    torch.manual_seed(0)
    b, s, n = 2, 200, 136
    m = _randomise(OuterProductMean(64, implementation=I.MINIWORLD, **options).to(DEV))
    msa = torch.randn(b, s, n, 64, device=DEV)
    mask = {"none": None, "ragged": torch.rand(b, s, n, device=DEV) > 0.3,
            "empty_rows": (torch.rand(b, s, n, device=DEV) > 0.3) & (torch.arange(n, device=DEV) % 7 != 0)}[mask_kind]
    residual = torch.randn(b, n, n, options["d_pair"], device=DEV)
    twin = _twin(m)
    monkeypatch.setattr(OuterProductMean, "_variant_reference", lambda *a, **k: pytest.fail("the PyTorch statements ran"))
    y, g = _grads(m, BF, [msa, mask, None, residual], dy := torch.randn(b, n, n, options["d_pair"], device=DEV))
    monkeypatch.undo()
    y32, g32 = _grads(twin, torch.float32, [msa, mask, None, residual], dy)
    y16, g16 = _grads(twin, BF, [msa, mask, None, residual], dy)
    assert rel(y, y32) <= max(2e-3, 1.5 * rel(y16, y32)), (rel(y, y32), rel(y16, y32))
    for i, (a, b_, c) in enumerate(zip(g, g32, g16, strict=True)):
        assert rel(a, b_) <= max(2e-3, 1.5 * rel(c, b_)), (i, rel(a, b_), rel(c, b_))


@pytest.mark.parametrize("mask_kind", ["none", "ragged", "all_masked"])
@pytest.mark.parametrize("d_pair", [128, 125])
@pytest.mark.parametrize("n_head,d_hidden", [(8, 32), (8, 8), (4, 8), (4, 12)])
def test_pair_weighted_averaging(mask_kind, d_pair, n_head, d_hidden):
    """d_hidden 8 is AF3's msa_value_dim (OpenFold3, Protenix, OpenDDE): the head is a 16-column tile with 8 columns masked."""
    if (n_head, d_hidden) != (8, 32) and d_pair != 128:
        pytest.skip("the narrow heads are crossed with the masks, not with the ragged pair width")
    torch.manual_seed(0)
    b, s, n = 2, 100, 136
    m = _randomise(MSAPairWeightedAveraging(64, d_pair, n_head, d_hidden, implementation=I.MINIWORLD).to(DEV)).eval()
    msa, pair = torch.randn(b, s, n, 64, device=DEV), torch.randn(b, n, n, d_pair, device=DEV)
    mask = {"none": None, "ragged": torch.rand(b, n, device=DEV) > 0.3,
            "all_masked": torch.zeros(b, n, dtype=torch.bool, device=DEV)}[mask_kind]
    _compare(m, [msa, pair, mask], _twin(m))


@pytest.mark.parametrize("n_head,d_hidden", [(8, 8), (4, 8)])
def test_narrow_head_is_the_zero_padded_head(n_head, d_hidden):
    """A d_hidden-8 head equals the same head padded with zero projection rows/columns to 16 (the padded columns contribute nothing),
    forward and every gradient, to bf16 round-off of the two cuBLAS contractions."""
    from miniworld_engine.kernels.pair_weighted_averaging.interface import (
        triton_pair_weighted_averaging,
    )

    torch.manual_seed(0)
    m = _randomise(MSAPairWeightedAveraging(64, 128, n_head, d_hidden).to(DEV)).to(BF)
    msa, pair = torch.randn(1, 40, 136, 64, device=DEV, dtype=BF), torch.randn(1, 136, 136, 128, device=DEV, dtype=BF)
    mask = torch.rand(1, 136, device=DEV) > 0.2
    keep = torch.rand(1, 136, 64, device=DEV) > 0.15
    wv, wg, wo = m.to_value.weight, m.to_gate.weight, m.to_out.weight

    def pad_rows(w):                                         # [H d, D] -> [H 16, D], zero rows
        return torch.cat([w.view(n_head, d_hidden, -1), w.new_zeros(n_head, 16 - d_hidden, w.shape[-1])], 1).flatten(0, 1)

    def pad_cols(w):                                         # [D, H d] -> [D, H 16]
        return torch.cat([w.view(w.shape[0], n_head, d_hidden), w.new_zeros(w.shape[0], n_head, 16 - d_hidden)], 2).flatten(1)

    def run(wv_, wg_, wo_):
        leaves = [t.detach().clone().requires_grad_() for t in (msa, pair, m.ln_msa.weight, m.ln_msa.bias, wv_, wg_,
                                                                m.ln_pair.weight, m.to_bias.weight, wo_)]
        out = triton_pair_weighted_averaging(leaves[0], leaves[1], mask, *leaves[2:], keep=keep, p_drop=0.15)
        g = torch.autograd.grad(out, leaves, torch.ones_like(out) * 0.1 + torch.sin(torch.arange(out.numel(), device=DEV).view_as(out)))
        return out, g

    out8, g8 = run(wv, wg, wo)
    out16, g16 = run(pad_rows(wv), pad_rows(wg), pad_cols(wo))
    assert rel(out8, out16) < 2e-3, rel(out8, out16)
    unpad = {4: lambda g: g.view(n_head, 16, -1)[:, :d_hidden].flatten(0, 1), 5: lambda g: g.view(n_head, 16, -1)[:, :d_hidden].flatten(0, 1),
             8: lambda g: g.view(g.shape[0], n_head, 16)[:, :, :d_hidden].flatten(1)}
    for i, (a, b) in enumerate(zip(g8, g16, strict=True)):
        b = unpad[i](b) if i in unpad else b
        assert rel(a, b) < 5e-3, (i, rel(a, b))


def test_pair_weighted_averaging_dropout_matches_its_keep_mask():
    """The fused dropout against the reference given the same keep-mask (the module draws it as Dropout(broadcast_dim=1))."""
    from miniworld_engine.kernels.pair_weighted_averaging.interface import (
        triton_pair_weighted_averaging,
    )
    from miniworld_engine.kernels.pair_weighted_averaging.reference import (
        pair_weighted_averaging_reference,
    )

    torch.manual_seed(0)
    m = _randomise(MSAPairWeightedAveraging(64, 128, 8, 32).to(DEV))
    msa, pair = torch.randn(1, 64, 128, 64, device=DEV), torch.randn(1, 128, 128, 128, device=DEV)
    keep = torch.rand(1, 128, 64, device=DEV) > 0.25
    w = [m.ln_msa.weight, m.ln_msa.bias, m.to_value.weight, m.to_gate.weight, m.ln_pair.weight,
         m.to_bias.weight, m.to_out.weight]
    got = triton_pair_weighted_averaging(msa.to(BF), pair.to(BF), None, *[t.to(BF) for t in w], keep=keep, p_drop=0.25)
    ref = pair_weighted_averaging_reference(msa, pair, None, *w, keep=keep, p_drop=0.25)
    # the update is small against msa, so the bf16 OUTPUT's own rounding is the floor the kernel is held to
    rounding = rel(ref.to(BF).float() - msa.to(BF).float(), ref - msa)
    assert rel(got.float() - msa.to(BF).float(), ref - msa) <= max(2e-3, 2 * rounding)
    m.implementation = I.MINIWORLD                        # the module's own draw, through the Triton path
    m.train()
    m.drop_msa.p_drop = 0.25
    out = m.to(BF)(msa.to(BF), pair.to(BF)) - msa.to(BF)
    dropped = (out == 0).all(dim=1)                      # one keep decision per (token, channel), shared over the MSA rows
    assert 0.15 < dropped.float().mean().item() < 0.35


def test_one_compiled_graph_forward_and_backward():
    from torch._dynamo.testing import CompileCounterWithBackend

    torch.manual_seed(0)
    opm = _randomise(OuterProductMean(64, 128, 32, implementation=I.MINIWORLD).to(DEV).to(BF))
    pwa = _randomise(MSAPairWeightedAveraging(64, 128, 8, 32, implementation=I.MINIWORLD).to(DEV).to(BF)).eval()
    msa = torch.randn(1, 64, 128, 64, device=DEV, dtype=BF, requires_grad=True)
    pair = torch.randn(1, 128, 128, 128, device=DEV, dtype=BF, requires_grad=True)
    mask = torch.rand(1, 128, device=DEV) > 0.2

    def block(msa, pair):
        pair = opm(msa, residual=pair)
        return pwa(msa, pair, mask), pair

    eager = block(msa, pair)
    eager_g = torch.autograd.grad(eager[0].float().sum() + eager[1].float().sum(), [msa, pair])
    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(block, backend=counter, fullgraph=True)
    for _ in range(3):
        out = compiled(msa, pair)
    got_g = torch.autograd.grad(out[0].float().sum() + out[1].float().sum(), [msa, pair])
    assert counter.frame_count == 1
    for a, b in zip((*out, *got_g), (*eager, *eager_g), strict=True):
        assert rel(a, b) < 1e-2
