"""B9+B10 SM90 rounding, pitched/transposed layouts, and tile/warp independence."""

import pytest
import torch


CONFIGS = [
    dict(BLOCK_M1=64, BLOCK_N=128, BLOCK_K=64, GROUP_M=1, num_warps=4, num_stages=2),
    dict(BLOCK_M1=128, BLOCK_N=128, BLOCK_K=64, GROUP_M=4, num_warps=8, num_stages=2),
    dict(BLOCK_M1=64, BLOCK_N=32, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=3),
    dict(BLOCK_M1=64, BLOCK_N=64, BLOCK_K=128, GROUP_M=2, num_warps=4, num_stages=3),
    dict(BLOCK_M1=64, BLOCK_N=128, BLOCK_K=32, GROUP_M=1, num_warps=4, num_stages=4),
    dict(BLOCK_M1=64, BLOCK_N=128, BLOCK_K=64, GROUP_M=1, num_warps=8, num_stages=2),
    dict(BLOCK_M1=128, BLOCK_N=128, BLOCK_K=64, GROUP_M=1, num_warps=4, num_stages=2),
]

CONFIGS.append(
    dict(BLOCK_M1=64, BLOCK_N=256, BLOCK_K=32, GROUP_M=1, num_warps=4, num_stages=2)
)

# Production winner: two gate K tiles leave a third stage available for
# front-GEMM prefetch, then both retired gate slots join the front ring.
CONFIGS.append(
    dict(BLOCK_M1=64, BLOCK_N=128, BLOCK_K=64, GROUP_M=1, num_warps=4, num_stages=3)
)


@pytest.mark.parametrize(
    "shape",
    [
        (512, 128, 1024, 128),
        (512, 256, 1024, 128),
        (523, 40, 72, 48),
        (523, 40, 72, 13),
    ],
)
@pytest.mark.parametrize("config", CONFIGS)
def test_dual_bwd_sm90_matches_triton(shape, config):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90 CUDA GPU")
    import triton
    from miniworld_engine.kernels.trimul_inproj.cute.parity_dual_bwd import (
        input_dual_bwd_sm90_impl,
    )
    from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import (
        _input_dual_bwd_kernel,
        dual_shape_key,
    )

    torch.manual_seed(73)
    m, kg, kp, n = shape
    g = torch.randn(m, kg + 8, device="cuda", dtype=torch.bfloat16)[:, :kg]
    f = torch.randn(kp, ((m + 7) // 8) * 8 + 8, device="cuda", dtype=torch.bfloat16)[
        :, :m
    ].t()
    w = torch.randn(n, kg + 8, device="cuda", dtype=torch.bfloat16)[:, :kg].t()
    v = torch.randn(kp, ((n + 7) // 8) * 8 + 8, device="cuda", dtype=torch.bfloat16)[
        :, :n
    ].requires_grad_()
    result = input_dual_bwd_sm90_impl(g, f, w, v, 128, config)
    reference = torch.empty_like(result)
    tiles = {
        k: val for k, val in config.items() if k not in ("num_warps", "num_stages")
    }
    _input_dual_bwd_kernel.fn[
        lambda _: (
            triton.cdiv(m, config["BLOCK_M1"]) * triton.cdiv(n, config["BLOCK_N"]),
        )
    ](
        g,
        f,
        w,
        v,
        reference,
        m,
        kg,
        kp,
        n,
        *g.stride(),
        *f.stride(),
        *w.stride(),
        *v.stride(),
        **tiles,
        shape_key=dual_shape_key(128, kg, kp, n),
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    expected = (
        f.float() @ v.float() + (g.float() @ w.float()).bfloat16().float()
    ).bfloat16()
    for target in (reference, expected):
        relative = (result.float() - target.float()).norm() / target.float().norm()
        assert relative.item() <= 1e-4
