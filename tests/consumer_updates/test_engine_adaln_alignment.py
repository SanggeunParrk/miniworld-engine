"""Hopper must execute the wide AdaLN epilogue, including tails, without misaligned SMEM."""

import pytest
import torch
import triton

from miniworld_engine.kernels.adaln.triton.inference import _adaln_gemm_gate_kernel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def run_case(m, n, k, dtype, config, offset=0):
    torch.manual_seed(95)

    def rand(rows, cols):
        backing = torch.randn(rows * cols + offset, device="cuda", dtype=dtype)
        return backing[offset:].view(rows, cols)

    x, c = rand(m, n), rand(m, k)
    sw, bw = rand(k, n) / k**0.5, rand(k, n) / k**0.5
    sb = rand(1, n).flatten()
    r = torch.rsqrt(x.float().var(-1, unbiased=False) + 0.03)
    c1 = x.float().mean(-1) * r
    y = torch.empty_like(x)
    bm, bn, bk, warps, stages, group = config
    compiled = _adaln_gemm_gate_kernel.fn[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        c,
        sw,
        sb,
        bw,
        x,
        r,
        c1,
        y,
        m,
        n,
        k,
        k,
        n,
        n,
        n,
        n,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=group,
        shape_key=0,
        num_warps=warps,
        num_stages=stages,
    )
    torch.cuda.synchronize()
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        ref = (c.float() @ sw.float() + sb.float()).sigmoid() * (
            x.float() * r[:, None] - c1[:, None]
        ) + c.float() @ bw.float()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
    error = (y.float() - ref).norm() / ref.norm()
    assert torch.isfinite(y).all()
    assert error < (0.014 if dtype == torch.bfloat16 else 0.002), float(error)
    return compiled


@pytest.mark.parametrize(("n", "k"), [(128, 128), (384, 384), (768, 384)])
@pytest.mark.parametrize(("m", "offset"), [(8192, 0), (133, 1)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_previously_fatal_tile(n, k, m, offset, dtype):
    run_case(m, n, k, dtype, (128, 256, 32, 4, 1, 1), offset)


@pytest.mark.parametrize(
    "config",
    [
        (bm, bn, bk, warps, stages, group)
        for bm, bn, bk, warps, stages, group in [
            (64, 64, 32, 4, 2, 1),
            (128, 256, 16, 4, 1, 2),
            (128, 256, 64, 4, 2, 4),
            (128, 256, 128, 4, 3, 8),
            (128, 256, 32, 8, 1, 16),
            (256, 128, 32, 4, 1, 1),
            (256, 256, 32, 8, 2, 16),
            (64, 256, 32, 4, 1, 1),
            (128, 128, 32, 4, 1, 1),
        ]
    ],
)
def test_nearby_tiles_with_all_axis_tails(config):
    run_case(133, 381, 123, torch.bfloat16, config, 1)


# Engine CI selects GPU checks explicitly.
pytestmark = [pytest.mark.gpu, *([pytestmark] if "pytestmark" in globals() else [])]
