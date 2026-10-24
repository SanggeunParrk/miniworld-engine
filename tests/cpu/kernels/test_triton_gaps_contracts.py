"""What the portable Triton paths of the PWA value-dim-8 / OPM variants / TriangleAttention variants will and will not take, and that the
kernels' torch reference is the modules' own statements (CPU only: no kernel launches).

The kernels themselves are checked on a GPU (``tests/gpu/any/test_msa_triton_gpu.py``, ``test_triattn_variants_triton_gpu.py``,
``test_numerical.py``); here the decisions made before a launch.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from miniworld_engine.kernels.outer_product_mean.reference import (
    outer_product_mean_reference,
)
from miniworld_engine.kernels.outer_product_mean.triton.main import (
    refusal as opm_refusal,
)
from miniworld_engine.kernels.pair_weighted_averaging.triton.main import (
    refusal as pwa_refusal,
)
from miniworld_engine.kernels.triangle_attention.triton.variants import (
    refusal as tri_refusal,
)
from miniworld_engine.modules.exceptions import ImplementationType as I
from miniworld_engine.modules.outer_product import OuterProductMean


def _cuda_like(shape, dtype=torch.bfloat16):
    """A stand-in for a CUDA tensor: the refusals read only is_cuda / dtype / shape / ndim."""
    return SimpleNamespace(is_cuda=True, dtype=dtype, shape=tuple(shape), ndim=len(shape))


@pytest.mark.parametrize(("n_head", "d_hidden", "ok"), [(8, 32, True), (8, 8, True), (4, 8, True), (4, 12, True), (16, 8, True),
                                                      (1, 8, False), (3, 8, False), (8, 200, False), (8, 0, False)])
def test_pwa_takes_value_dim_8(n_head, d_hidden, ok):
    why = pwa_refusal(_cuda_like((1, 8, 16, 64)), _cuda_like((1, 16, 16, 128)), n_head, d_hidden)
    assert (why is None) == ok, why


def test_pwa_still_needs_a_power_of_two_msa_width_and_bf16():
    assert "d_msa" in str(pwa_refusal(_cuda_like((1, 8, 16, 48)), _cuda_like((1, 16, 16, 128)), 8, 8))
    assert "bf16" in str(pwa_refusal(_cuda_like((1, 8, 16, 64), torch.float32), _cuda_like((1, 16, 16, 128)), 8, 8))


@pytest.mark.parametrize(("d_hidden", "groups", "ok"), [(64, 8, True), (32, 8, True), (32, 1, True), (36, 8, False), (128, 2, False)])
def test_opm_groups(d_hidden, groups, ok):
    why = opm_refusal(_cuda_like((1, 8, 16, 64)), d_hidden, 128, interchain=False, groups=groups)
    assert (why is None) == ok, why


def test_tri_variant_contract():
    w = (_cuda_like((128, 128)),) * 6
    pair = _cuda_like((1, 16, 16, 128))
    assert tri_refusal(pair, w, 4, 128) is None
    assert "bf16" in str(tri_refusal(_cuda_like((1, 16, 16, 128), torch.float32), w, 4, 128))
    assert "head width" in str(tri_refusal(pair, w, 16, 128))          # 8-channel heads: not the Triton core's
    assert "QK-norm" in str(tri_refusal(pair, w, 4, 128, use_qk_norm=True))
    assert "CUDA" in str(tri_refusal(SimpleNamespace(is_cuda=False), w, 4, 128))


@pytest.mark.parametrize("options", [dict(projection_bias=True), dict(groups=8, d_hidden=64), dict(groups=4, d_hidden=32, projection_bias=True)])
@pytest.mark.parametrize("normalize_before_proj", [True, False])
def test_the_kernel_reference_is_the_modules_statements(options, normalize_before_proj):
    """``outer_product_mean_reference`` (explicit weights) equals ``OuterProductMean._variant_reference`` (fp64, CPU)."""
    torch.manual_seed(0)
    d_hidden = options.pop("d_hidden", 32)
    m = OuterProductMean(64, 96, d_hidden, implementation=I.PYTORCH, normalize_before_proj=normalize_before_proj, **options).double()
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.copy_(1 + 0.2 * torch.randn_like(p) if n.endswith("weight") and p.ndim == 1 else torch.randn_like(p) * 0.3)
    msa = torch.randn(2, 12, 20, 64, dtype=torch.float64)
    mask = torch.rand(2, 12, 20) > 0.3
    residual = torch.randn(2, 20, 20, 96, dtype=torch.float64)
    want = m._variant_reference(msa, mask, None, residual)
    extra = {}
    if m.groups > 1:
        extra = {"groups": m.groups, "product_norm_weight": m.product_norm.weight, "product_norm_bias": m.product_norm.bias,
                 "product_norm_eps": m.product_norm.eps}
    got = outer_product_mean_reference(msa, mask, m.ln_msa.weight.double(), m.ln_msa.bias.double(), m.to_left.weight, m.to_right.weight, m.to_out.weight,
                                       m.to_out.bias, residual, eps=m.ln_msa.eps, normalize_before_proj=normalize_before_proj,
                                       b_left=m.to_left.bias, b_right=m.to_right.bias, **extra)
    # the module's LayerNorm keeps its affine in fp32 (``_Fp32ParamsMixin``), so the two agree to fp32 round-off, not fp64's
    assert ((got - want).norm() / want.norm()).item() < 1e-5
