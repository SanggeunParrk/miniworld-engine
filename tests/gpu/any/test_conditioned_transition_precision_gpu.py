"""Regression for premature BF16 SwiGLU rounding in the fused training forward."""
import json

import pytest
import torch
import triton

pytestmark = pytest.mark.gpu


def _error(actual, expected):
    delta = actual.float() - expected.float()
    return {"max_relative": (delta.abs().max() / expected.abs().max().clamp_min(1e-12)).item(),
            "relative_frobenius": (delta.norm() / expected.norm().clamp_min(1e-12)).item()}


@pytest.mark.parametrize("tile", [0, 1, 2])
@pytest.mark.parametrize(("dtype", "seed"), [(torch.bfloat16, seed) for seed in range(5)]
                         + [(torch.float32, 0)])
def test_fused_forward_original_band_and_bf16_representation_floor(dtype, seed, tile, monkeypatch):
    from miniworld_engine.autotune.shape_key import atom_key
    from miniworld_engine.kernels.checks.conditioned_transition import (
        _expand,
        _squeeze_gate_ref,
    )
    from miniworld_engine.kernels.conditioned_transition.triton import training

    configs = [
        triton.Config({"BLOCK_K_D": 32, "BLOCK_K_ND": 32, "BLOCK_M1": 64, "BLOCK_N": 128, "GROUP_M": 4}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_K_D": 32, "BLOCK_K_ND": 32, "BLOCK_M1": 32, "BLOCK_N": 64, "GROUP_M": 4}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_K_D": 32, "BLOCK_K_ND": 64, "BLOCK_M1": 32, "BLOCK_N": 64, "GROUP_M": 4}, num_warps=4, num_stages=3),
    ]
    monkeypatch.setattr(training._b2b_fwd_train_kernel, "configs", [configs[tile]])
    monkeypatch.setattr(training._b2b_fwd_train_kernel, "cache", {})
    torch.manual_seed(seed)
    # The original failure's exact unscaled input distribution and shapes.
    shapes = [(8192, 128), (8192, 128), (256, 128), (256, 128),
              (128, 256), (128, 128), (128,)]
    args = [torch.randn(shape, device="cuda", dtype=dtype) for shape in shapes]
    a, b, h = _expand(args[0], args[2], args[3])
    out, scale, y = _squeeze_gate_ref(h, args[1], *args[4:])
    actual = training._b2b_fwd_train(*args, shape_key=atom_key(8192))
    expected = (y, torch.cat([a, b], dim=1), h, out, scale)
    metrics = {}
    # Keep the official seed-0 checker band unchanged. Other seeds also test the
    # nearest-representable BF16 floor: some ideal BF16 outputs themselves exceed
    # that seed-specific band, so requiring it for every input is impossible.
    band = 0.0032 if dtype == torch.bfloat16 else 0.009
    floors = {}
    for label, value, reference in zip(("Y", "AB", "H", "Out", "Scale"), actual, expected, strict=True):
        assert torch.isfinite(value).all()
        metrics[label] = _error(value, reference)
        if dtype == torch.float32 or seed == 0:
            assert metrics[label]["max_relative"] <= band, (label, metrics[label], band)
        if dtype == torch.bfloat16:
            floors[label] = _error(reference.to(dtype), reference)
            # Test computational error above the representability floor without
            # relabelling a raw-band exceedance as a raw-band pass.
            # FP32 GEMM accumulation can move a value across a BF16 midpoint.
            # A5000 seed 1: AB exceeds the rounded-reference maximum by 1.26e-7;
            # the independent FP64 audit measures 2.20e-7 error in that FP32
            # reference itself. Two FP32 eps cover that arithmetic, while the
            # official raw checker band above remains unchanged.
            roundoff = 2 * torch.finfo(torch.float32).eps
            assert metrics[label]["max_relative"] <= floors[label]["max_relative"] + roundoff, (
                label, metrics[label], floors[label], roundoff)
            assert metrics[label]["relative_frobenius"] <= floors[label]["relative_frobenius"] + 1e-6
    print("CT_PRECISION_METRICS " + json.dumps({"dtype": str(dtype), "seed": seed, "tile": tile,
          "metrics": metrics, "band": band, "rounding_floors": floors,
          "raw_band_exceeded": [label for label, value in metrics.items() if value["max_relative"] > band],
          "final_output_rounding_floor": _error(y.to(dtype), y)}))
