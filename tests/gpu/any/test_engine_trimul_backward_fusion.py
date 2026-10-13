"""Backward fusion accuracy, tail/stride handling, and autograd branch placement."""

from typing import Any

import pytest
import torch
import triton

from miniworld_engine.autotune.shape_key import both_key, pack, token_key
from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import (
    _input_dual_bwd_kernel,
    _ln_bwd_residual_kernel,
    input_ln_residual,
)


def error(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8))


DUAL = [
    {"BLOCK_M1": m, "BLOCK_N": n, "BLOCK_K": k, "GROUP_M": g, "num_warps": w, "num_stages": s}
    for m, n, k, g, w, s in [
        (16, 32, 32, 1, 4, 2),
        (32, 64, 64, 2, 4, 3),
        (64, 128, 128, 4, 8, 4),
        (128, 256, 32, 8, 8, 2),
        (64, 64, 64, 8, 4, 4),
        (128, 128, 128, 2, 8, 3),
    ]
]


@pytest.mark.parametrize("cfg", DUAL)
@pytest.mark.parametrize("strided", [False, True])
def test_dual_tails_and_rounding(cfg, strided):
    torch.manual_seed(637)
    m = 5 * cfg["BLOCK_M1"] + 7
    n = 2 * cfg["BLOCK_N"] + 3
    kg = 69
    kp = 259
    kw: dict[str, Any] = {"device": "cuda", "dtype": torch.bfloat16}
    g = torch.randn(m, kg, **kw)
    f = torch.randn(kp, m, **kw).t()
    w = torch.randn(n, kg, **kw).t() / kg**0.5
    v = torch.randn(kp, n, **kw) / kp**0.5
    if strided:
        g = g.t().contiguous().t()
        v = v.t().contiguous().t()
    y = torch.full((m, n), float("nan"), **kw)
    _input_dual_bwd_kernel.fn[
        (triton.cdiv(m, cfg["BLOCK_M1"]) * triton.cdiv(n, cfg["BLOCK_N"]),)
    ](
        g,
        f,
        w,
        v,
        y,
        m,
        kg,
        kp,
        n,
        *g.stride(),
        *f.stride(),
        *w.stride(),
        *v.stride(),
        shape_key=token_key(17, KG=kg, KP=kp, N=n),
        **cfg,
    )
    ref = (g.float() @ w.float()).bfloat16().float() + f.float() @ v.float()
    assert torch.isfinite(y).all()
    assert error(y, ref) < 0.004


LN = [
    {"BLOCK_M1": m, "BLOCK_K": k, "num_warps": w, "num_stages": s}
    for m, k, w, s in [
        (1, 64, 1, 2),
        (4, 128, 4, 1),
        (16, 256, 8, 1),
        (32, 64, 4, 3),
        (64, 128, 16, 1),
        (128, 512, 32, 1),
        (2, 1024, 2, 1),
    ]
]


@pytest.mark.parametrize("cfg", LN)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_ln_residual_tails(cfg, dtype):
    torch.manual_seed(974)
    m = 137
    n = 193
    kw: dict[str, Any] = {"device": "cuda", "dtype": dtype}
    x = torch.randn(m, n, **kw)
    dy = torch.randn_like(x)
    dr = torch.randn_like(x)
    w = torch.randn(n, **kw) * 0.2 + 1
    mean = x.float().mean(1)
    rs = torch.rsqrt(x.float().var(1, unbiased=False) + 1e-5)
    dx = torch.full_like(x, float("nan"))
    dw = torch.zeros(n, device="cuda")
    db = torch.zeros_like(dw)
    _ln_bwd_residual_kernel.fn[(triton.cdiv(m, cfg["BLOCK_M1"]),)](
        dx,
        dy,
        dw,
        db,
        dr,
        x,
        w,
        mean,
        rs,
        rs,
        1,
        1,
        n,
        1,
        m,
        n,
        shape_key=pack(both_key(17), N=n),
        HAS_ROWSCALE=False,
        **cfg,
    )
    xh = (x.float() - mean[:, None]) * rs[:, None]
    u = dy.float() * w.float()
    gx = rs[:, None] * (
        u - u.mean(1, keepdim=True) - xh * (u * xh).mean(1, keepdim=True)
    )
    refs = (
        gx.to(dtype).float() + dr.float(),
        (dy.float() * xh).sum(0),
        dy.float().sum(0),
    )
    for actual, ref in zip((dx, dw, db), refs, strict=False):
        assert torch.isfinite(actual).all()
        assert error(actual, ref) < 0.004


@pytest.mark.parametrize("compiled", [False, True])
def test_autograd_residual_does_not_enter_parameter_grads(compiled):
    torch.manual_seed(377)
    kw: dict[str, Any] = {"device": "cuda", "dtype": torch.bfloat16}
    x = torch.randn(1, 17, 17, 128, **kw).requires_grad_()
    w = torch.randn(128, **kw).requires_grad_()
    b = torch.randn(128, **kw).requires_grad_()

    def fn(x, w, b):
        return input_ln_residual(x, w, b, 1e-5)

    call = torch.compile(fn, fullgraph=True, dynamic=False) if compiled else fn
    norm, residual = call(x, w, b)
    dy = torch.zeros_like(norm)
    dr = torch.randn_like(residual)
    gx, gw, gb = torch.autograd.grad((norm, residual), (x, w, b), (dy, dr))
    assert (
        torch.equal(gx, dr)
    )
    assert (
        torch.count_nonzero(gw) == 0
    )
    assert (
        torch.count_nonzero(gb) == 0
    )


# Engine CI selects GPU checks explicitly.
pytestmark = pytest.mark.gpu
