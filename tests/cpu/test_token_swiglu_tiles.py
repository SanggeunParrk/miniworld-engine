"""The H100 bf16 token DiT step's SwiGLU GEMM operands (``kernels/conditioned_transition/cuda``): the tile choice and the per-64 re-tiling of
the weight rows (CPU: layout only)."""

from __future__ import annotations

import torch

from miniworld_engine.kernels.conditioned_transition import cuda as C


def test_swiglu_tile_choice():
    # bf16: 128 x 128 tiles where they fill the last wave better (H100, 132 SMs, H 1536); fp32 keeps 128 x 256
    assert C._swiglu_ht(640, 1536, 132, bf16=True) == 64  # L 128, S 5
    assert C._swiglu_ht(1920, 1536, 132, bf16=True) == 64  # L 384
    for M in (1280, 2560, 3840):  # ties keep the larger tile
        assert C._swiglu_ht(M, 1536, 132, bf16=True) == 128
    assert C._swiglu_ht(640, 1536, 132) == 128


def test_wab64_layouts():
    K, H = 8, 256
    wa, wb, wu = (torch.arange(H, dtype=torch.float32)[:, None] + 1000 * i + torch.zeros(H, K) for i in range(3))
    # the token pack's per-128 rows [Wa 128 | Wb 128] per tile -> [Wa 64 | Wb 64]
    w128 = torch.stack([wa.view(H // 128, 128, K), wb.view(H // 128, 128, K)], 1).reshape(2 * H, K)
    w64 = C._bf16_wab64(w128.contiguous())
    want = torch.stack([wa.view(H // 64, 64, K), wb.view(H // 64, 64, K)], 1).reshape(2 * H, K)
    assert torch.equal(w64, want)
    # the up-gate [Wa; Wb; Wu] -> [Wa 64 | Wb 64 | Wu 64]
    w3 = C._bf16_wab64(torch.cat([wa, wb, wu]).contiguous(), up=True)
    want3 = torch.stack([wa.view(H // 64, 64, K), wb.view(H // 64, 64, K), wu.view(H // 64, 64, K)], 1).reshape(3 * H, K)
    assert torch.equal(w3, want3)
