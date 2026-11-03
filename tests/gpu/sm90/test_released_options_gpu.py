# ruff: noqa: PT018, PT011, C408
"""H100: the released-checkpoint options on the MINIWORLD implementation. An option no kernel implements is REFUSED (a ``KernelRefusedWarning``,
then the PyTorch statements), never ignored; an option a kernel serves agrees with the PyTorch implementation.

``divide_by_length`` (TriMul): refused, bit-equal to the PYTORCH implementation. ``count_offset`` (OPM): refused outside the kernels' contract (the served calls: test_opm_divisor_gpu.py). ``transposed_bias=False`` / the gate and out biases
(TriangleAttention) and the single-track / ``parallel`` PairformerBlock: MINIWORLD vs PYTORCH within bf16 tolerance.
"""
from __future__ import annotations

import warnings

import pytest
import torch

from miniworld_engine.modules.dispatch import KernelRefusedWarning
from miniworld_engine.modules.exceptions import ImplementationType as IT
from miniworld_engine.modules.outer_product import OuterProductMean
from miniworld_engine.modules.pairformer import PairformerBlock, PairformerConfig
from miniworld_engine.modules.triangle_attention import TriangleAttention
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="H100 only"),
]

L, C = 128, 128


@pytest.fixture(autouse=True)
def _fresh_warnings():
    """``warn_refused`` warns once per (op, card, reason): each test sees its own."""
    from miniworld_engine.modules import dispatch

    dispatch._refused_warned.clear()


def _pair(b=1, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    pair = torch.randn(b, L, L, C, device="cuda", generator=g, dtype=torch.bfloat16)
    mask = torch.ones(b, L, dtype=torch.bool, device="cuda")
    mask[:, -9:] = False
    return pair, mask


def _fill(module, seed=1, std=0.05):
    g = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, device=p.device, generator=g, dtype=torch.float32).to(p.dtype) * std)
    return module


def _pair_of(cls, **kw):
    """(MINIWORLD module, PYTORCH module) with identical weights."""
    a = _fill(cls(implementation=IT.MINIWORLD, **kw).cuda().bfloat16())
    b = cls(implementation=IT.PYTORCH, **kw).cuda().bfloat16()
    b.load_state_dict(a.state_dict())
    return a.eval(), b.eval()


@pytest.mark.parametrize("outgoing", [True, False])
def test_trimul_divide_by_length_is_refused_not_ignored(outgoing):
    pair, mask = _pair()
    fast, ref = _pair_of(TriangleMultiplication, d_pair=C, outgoing=outgoing, p_drop=0.0, divide_by_length=True)
    plain = _fill(TriangleMultiplication(C, outgoing=outgoing, p_drop=0.0, implementation=IT.MINIWORLD).cuda().bfloat16()).eval()
    plain.load_state_dict(fast.state_dict())
    with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = fast(pair, mask)
    assert any(issubclass(w.category, KernelRefusedWarning) for w in caught)
    with torch.no_grad():
        torch.testing.assert_close(got, ref(pair, mask), atol=0, rtol=0)
        assert not torch.allclose(got.float(), plain(pair, mask).float(), atol=1e-3)  # the option was applied


def test_trimul_divide_by_length_trains():
    pair, mask = _pair()
    fast, _ = _pair_of(TriangleMultiplication, d_pair=C, p_drop=0.0, divide_by_length=True)
    x = pair.clone().requires_grad_()
    fast(x, mask).float().square().sum().backward()
    assert torch.isfinite(x.grad.float()).all() and fast.to_out.weight.grad is not None


@pytest.mark.parametrize("before", [True, False])
def test_opm_count_offset_outside_the_kernel_contract_is_refused_not_ignored(before):
    """Batch 2 is outside opm_train / composed (one MSA stack per call): refused (warning), bit-equal to PYTORCH. The served calls: test_opm_divisor_gpu.py."""
    g = torch.Generator(device="cuda").manual_seed(2)
    msa = torch.randn(2, 16, L, 64, device="cuda", dtype=torch.bfloat16, generator=g)
    mask = torch.rand(2, 16, L, device="cuda", generator=g) > 0.2
    kw = {"d_msa": 64, "d_pair": C, "d_hidden": 32, "count_offset": 1e-3, "normalize_before_proj": before}
    fast, ref = _pair_of(OuterProductMean, **kw)
    with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = fast(msa, mask)
    assert any(issubclass(w.category, KernelRefusedWarning) for w in caught)
    with torch.no_grad():
        torch.testing.assert_close(got, ref(msa, mask), atol=0, rtol=0)


@pytest.mark.parametrize("starting", [True, False])
@pytest.mark.parametrize("variant", [{"transposed_bias": False}, {"gate_bias": True, "out_bias": True, "transposed_bias": False}])
def test_triangle_attention_options_match_pytorch(starting, variant):
    pair, mask = _pair()
    fast, ref = _pair_of(TriangleAttention, d_pair=C, n_head=4, d_hidden=C, starting=starting, p_drop=0.0, **variant)
    with torch.no_grad():
        a, b = fast(pair, mask).float(), ref(pair, mask).float()
    assert (a - b).abs().max() <= 0.05 * b.abs().max() + 0.05, ((a - b).abs().max().item(), b.abs().max().item())


@pytest.mark.parametrize("parallel", [False, True])
def test_pairformer_single_track_matches_pytorch(parallel):
    pair, mask = _pair()
    single = torch.randn(1, L, 256, device="cuda", dtype=torch.bfloat16)
    cfg = PairformerConfig(d_pair=C, d_hidden_tri_multi=C, d_hidden_tri_attention=32, n_head_tri_attention=4, p_drop=0.0, d_single=256,
                           n_head_single=8, parallel=parallel, transposed_bias=False, divide_by_length=True)
    fast = _fill(PairformerBlock(cfg, implementation=IT.MINIWORLD).cuda().bfloat16()).eval()
    ref = PairformerBlock(cfg, implementation=IT.PYTORCH).cuda().bfloat16().eval()
    ref.load_state_dict(fast.state_dict())
    with torch.no_grad():
        (pa, sa), (pb, sb) = fast(pair, mask, single), ref(pair, mask, single)
    for a, b in ((pa, pb), (sa, sb)):
        assert (a.float() - b.float()).abs().max() <= 0.08 * b.float().abs().max() + 0.1
