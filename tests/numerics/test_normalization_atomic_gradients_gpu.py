"""Check both pair-bias accumulation paths and Transition replica sums against autograd."""
import pytest
import torch
import triton

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def check(got, want, band):
    for actual, expected in zip(got, want, strict=True):
        assert torch.isfinite(actual).all()
        rel = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-10)
        assert rel < band, rel.item()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("heads", [4, 17])
@pytest.mark.parametrize("tile", [32, 128])
def test_pair_bias_parameter_gradients(dtype, heads, tile):
    from miniworld_engine.kernels.layernorm_linear.triton.pair_bias import _layer_norm_linear_bwd
    torch.manual_seed(131)
    m, n = 259, 65
    x = torch.randn(m, n, device="cuda", dtype=dtype)
    w = torch.randn(n, device="cuda", dtype=dtype)
    p = torch.randn(heads, n, device="cuda", dtype=dtype)
    # The real launcher promotes the upstream projection gradient to FP32.
    dy = torch.randn(m, heads, device="cuda", dtype=torch.float32)
    mean = x.float().mean(1)
    rs = torch.rsqrt(x.float().var(1, correction=0) + 1e-5)
    dx = torch.empty_like(x)
    dw = torch.zeros(n, device="cuda")
    dp = torch.zeros(heads, n, device="cuda")
    _layer_norm_linear_bwd.fn[(triton.cdiv(m, 32),)](
        dy, x, w, p, mean, rs, dx, dw, dp, n, m, n, heads,
        USE_DOT=heads >= 16, BLOCK_M1=32, BLOCK_K_D=tile, BLOCK_K_NH=16,
        shape_key=0, num_warps=4, num_stages=1)
    xr, wr, pr = [t.float().detach().requires_grad_() for t in (x, w, p)]
    from miniworld_engine.kernels.checks import _no_tf32
    with _no_tf32():
        y = torch.nn.functional.layer_norm(xr, (n,), wr, None, 1e-5) @ pr.T
        expected = torch.autograd.grad(y, (xr, wr, pr), dy)
    check((dx, dw, dp), expected, .004 if dtype == torch.bfloat16 else 4e-6)


@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("tile", [32, 128])
def test_transition_replica_parameter_gradients(private, tile):
    from miniworld_engine.kernels.transition.triton.fused import _transition_ln_bwd_kernel
    torch.manual_seed(619)
    m, n = 1031, 65
    x = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    w = torch.randn(n, device="cuda")
    xf = x.float()
    rs = torch.rsqrt(xf.var(1, correction=0) + 1e-5)
    folded = xf.mean(1) * rs
    dx = torch.empty_like(x)
    replicas = 64 if private else 1
    dw = torch.zeros(replicas, n, device="cuda")
    db = torch.zeros_like(dw)
    _transition_ln_bwd_kernel.fn[(triton.cdiv(m, 16),)](
        dy, x, rs, folded, w, dx, dw, db, m, n, 0, n, 1,
        n, 1, n, 1, BLOCK_M1=16, BLOCK_K=tile,
        NUM_REPLICAS=replicas, PRIVATIZE_DGDB=private, num_warps=4, num_stages=1)
    xr = xf.detach().requires_grad_()
    wr = w.detach().requires_grad_()
    br = torch.zeros_like(w, requires_grad=True)
    y = torch.nn.functional.layer_norm(xr, (n,), wr, br, 1e-5)
    expected = torch.autograd.grad(y, (xr, wr, br), dy.float())
    check((dx, dw.sum(0), db.sum(0)), expected, .004)
