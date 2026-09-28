"""The portable Triton OuterProductMean / MSAPairWeightedAveraging against fp32 autograd of the module's own statements.

Run with `engine_backend="triton"` so the native H100 paths step aside and the Triton kernels are what is measured on any card.
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
    settings.configure(engine_backend="triton")
    yield
    settings.configure(**vars(previous))


def rel(a, b):
    return (torch.linalg.vector_norm(a.float() - b.float()) / torch.linalg.vector_norm(b.float()).clamp_min(1e-30)).item()


def _randomise(m):
    torch.manual_seed(3)
    with torch.no_grad():
        for name, p in m.named_parameters():
            p.copy_(1.0 + 0.2 * torch.randn_like(p) if name.endswith(("ln_msa.weight", "ln_pair.weight"))
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


@pytest.mark.parametrize("mask_kind", ["none", "ragged", "all_masked"])
@pytest.mark.parametrize("d_pair", [128, 125])
def test_pair_weighted_averaging(mask_kind, d_pair):
    torch.manual_seed(0)
    b, s, n = 2, 100, 136
    m = _randomise(MSAPairWeightedAveraging(64, d_pair, 8, 32, implementation=I.MINIWORLD).to(DEV)).eval()
    msa, pair = torch.randn(b, s, n, 64, device=DEV), torch.randn(b, n, n, d_pair, device=DEV)
    mask = {"none": None, "ragged": torch.rand(b, n, device=DEV) > 0.3,
            "all_masked": torch.zeros(b, n, dtype=torch.bool, device=DEV)}[mask_kind]
    _compare(m, [msa, pair, mask], _twin(m))


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
    w = [m.ln_msa.weight, m.ln_msa.bias, m.to_value.weight, m.to_gate.weight, m.ln_pair.weight, m.ln_pair.bias,
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
