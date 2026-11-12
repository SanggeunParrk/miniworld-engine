"""H100 TriangleAttention inference core of 64-channel heads (``kernels/triangle_attention/cuda/h100_dh``): the 128-query tile (4 warps x 32
queries, :func:`h100_dh.query_tile`) computes every query exactly as the 64-query tile does, so the two give bitwise equal outputs, masked
keys and a fully masked query row included."""

import pytest
import torch

from miniworld_engine.kernels.triangle_attention.cuda import h100_dh

pytestmark = [pytest.mark.gpu, pytest.mark.h100]


def test_query_tile_rule() -> None:
    assert h100_dh.query_tile(256, 64) == 64
    assert h100_dh.query_tile(384, 64) == 128
    assert h100_dh.query_tile(448, 64) == 64  # not a multiple of 128
    assert h100_dh.query_tile(2048, 64) == 128
    assert h100_dh.query_tile(2048, 32) == 64


@pytest.mark.parametrize(("d", "length"), [(64, 384), (64, 512), (32, 256)])
def test_query_tiles_bitwise_equal(d: int, length: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")
    H = 4
    g = torch.Generator(device="cuda").manual_seed(0)
    proj = (torch.randn(length * length, 4 * H * d, device="cuda", generator=g) * 2).bfloat16()
    bias = (torch.randn(1, H, length, length, device="cuda", generator=g) * 3).bfloat16()
    bias[..., -3:] = float("-inf")  # masked keys
    bias[..., 5, :] = float("-inf")  # a query without an unmasked key
    gb = torch.randn(H * d, device="cuda", generator=g)
    ext = h100_dh.ext()
    a = ext.triattn_dh_forward(proj, bias, gb, H, d, d**-0.5, 64)
    b = ext.triattn_dh_forward(proj, bias, gb, H, d, d**-0.5, 128)
    assert torch.isfinite(a).all()
    assert torch.equal(a, b)
