"""Wider warp schedules must preserve every LN gradient and residual rounding."""
import pytest
import torch
import triton

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def inputs(width, dtype, transposed=False):
    torch.manual_seed(409)
    m = 263  # row tail and multiple persistent iterations
    x = torch.randn((width, m) if transposed else (m, width), device="cuda", dtype=dtype)
    if transposed:
        x = x.t()
    dy = torch.randn_like(x)
    w = torch.randn(width, device="cuda", dtype=torch.float32)
    xf = x.float()
    mean = xf.mean(-1)
    rstd = torch.rsqrt(xf.var(-1, unbiased=False) + 1e-5)
    return x, dy, w, mean, rstd


def reference(x, dy, w, rowscale=None):
    xr = x.float().detach().requires_grad_()
    wr = w.detach().requires_grad_()
    br = torch.zeros_like(w, requires_grad=True)
    y = torch.nn.functional.layer_norm(xr, (x.shape[1],), wr, br, 1e-5)
    if rowscale is not None:
        y = y * rowscale[:, None]
    return torch.autograd.grad(y, (xr, wr, br), dy.float())


def check(actual, expected, dtype):
    for got, want in zip(actual, expected, strict=True):
        assert torch.isfinite(got).all()
        error = (got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-10)
        assert error < (0.004 if dtype == torch.bfloat16 else 3e-6), error.item()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("width", [128, 137, 256, 384])
@pytest.mark.parametrize(("covering", "warps"), [(False, 8), (True, 16)])
def test_persistent_backward_wide_warps(dtype, transposed, width, covering, warps):
    from miniworld_engine.kernels.layernorm.triton.persistent import _ln_bwd_persistent
    x, dy, w, mean, rstd = inputs(width, dtype, transposed)
    dx = torch.empty_like(x)
    dw = torch.empty((3, width), device="cuda")
    db = torch.empty_like(dw)
    bk = triton.next_power_of_2(width) if covering else 64
    _ln_bwd_persistent.fn[(3, triton.cdiv(width, bk))](
        dx, dw, db, dy, x, w, mean, rstd, width, *x.stride(), x.shape[0], width,
        BLOCK_M1=32, BLOCK_K=bk, shape_key=0, num_warps=warps, num_stages=1)
    check((dx, dw.sum(0), db.sum(0)), reference(x, dy, w), dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rowscale", [False, True])
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("block_k", [64, 256])
def test_atomic_backward_mask_and_residual(dtype, rowscale, residual, block_k):
    from miniworld_engine.kernels.layernorm.triton.main import layer_norm_bwd_dx_fused
    from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import (
        _ln_bwd_residual_kernel,
    )
    x, dy, w, mean, rstd = inputs(137, dtype)
    scale = (torch.arange(x.shape[0], device="cuda") % 3 != 0).float() * 1.25
    dr = torch.randn_like(x)
    dx = torch.empty_like(x)
    dw = torch.zeros(137, device="cuda")
    db = torch.zeros_like(dw)
    args = [dx, dy, dw, db]
    if residual:
        args.append(dr)
    args += [x, w, mean, rstd, scale, 1, 1, *x.stride(), x.shape[0], x.shape[1]]
    kernel = _ln_bwd_residual_kernel if residual else layer_norm_bwd_dx_fused
    kernel.fn[(triton.cdiv(x.shape[0], 64),)](
        *args, BLOCK_M1=64, BLOCK_K=block_k, shape_key=0, HAS_ROWSCALE=rowscale,
        num_warps=4, num_stages=1)
    rdx, rdw, rdb = reference(x, dy, w, scale if rowscale else None)
    if residual:
        rdx = (rdx.to(dtype).float() + dr.float()).to(dtype)
    check((dx, dw, db), (rdx, rdw, rdb), dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("width", [137, 256])
def test_strided_atomic_backward(dtype, transposed, width):
    from miniworld_engine.kernels.layernorm_linear.triton.te_style import _ln_bwd_kernel
    x, dy, w, mean, rstd = inputs(width, dtype, transposed)
    dx = torch.empty_like(x)
    dw = torch.zeros(width, device="cuda")
    db = torch.zeros_like(dw)
    _ln_bwd_kernel.fn[(triton.cdiv(x.shape[0], 32),)](
        dy, x, w, mean, rstd, dx, dw, db, x.shape[0], width,
        *dy.stride(), *x.stride(), *dx.stride(), N_PAD=triton.next_power_of_2(width),
        BLOCK_M1=32, shape_key=0, num_warps=8, num_stages=1)
    check((dx, dw, db), reference(x, dy, w), dtype)
