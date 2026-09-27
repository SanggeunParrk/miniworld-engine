"""CUDA norm contracts; launch pytest through Slurm, never the login node."""
import pytest
import torch
from miniworld_engine.kernels.norm_cuda import cuda_layernorm, cuda_rmsnorm, cuda_layernorm_linear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def reference(x, w, b, rms, eps):
    acc = torch.float64 if x.dtype == torch.float64 else torch.float32
    xf = x.to(acc)
    if rms:
        y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
        if w is not None:
            y = y * w.to(acc)
    else:
        y = torch.nn.functional.layer_norm(xf, (x.shape[-1],), w.to(acc), None if b is None else b.to(acc), eps)
    return y.to(x.dtype)


@pytest.mark.parametrize('rms', [False, True])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize('width', [1, 33, 128, 384, 1024, 8193])
def test_forward_and_all_first_gradients(rms, dtype, width):
    torch.manual_seed(174)
    x = torch.randn(2, 5, width * 2, device='cuda', dtype=dtype)[..., ::2].detach().requires_grad_()
    acc = torch.float64 if dtype == torch.float64 else torch.float32
    w = torch.randn(width, device='cuda', dtype=acc, requires_grad=True)
    b = None if rms else torch.randn_like(w, requires_grad=True)
    y = cuda_rmsnorm(x, w) if rms else cuda_layernorm(x, w, b)
    expected = reference(x, w, b, rms, 1e-5)
    tol = {torch.float16: .003, torch.bfloat16: .02, torch.float32: 5e-5, torch.float64: 1e-10}[dtype]
    torch.testing.assert_close(y, expected, atol=tol, rtol=tol)
    dy = torch.randn_like(y)
    params = [v for v in (x, w, b) if v is not None]
    for actual, ref in zip(torch.autograd.grad(y, params, dy), torch.autograd.grad(expected, params, dy)):
        torch.testing.assert_close(actual, ref, atol=tol * 4, rtol=tol * 2)


@pytest.mark.parametrize('threads', [128, 256])
@pytest.mark.parametrize('rows', [1, 4, 16, 64, 256])
def test_launch_config_gradients(threads, rows):
    x = torch.randn(129, 384, device='cuda', requires_grad=True)
    w = torch.randn(384, device='cuda', requires_grad=True)
    b = torch.randn_like(w, requires_grad=True)
    y = cuda_layernorm(x, w, b, threads=threads, rows=rows)
    expected = reference(x, w, b, False, 1e-5)
    dy = torch.randn_like(y)
    for g, r in zip(torch.autograd.grad(y, (x, w, b), dy), torch.autograd.grad(expected, (x, w, b), dy)):
        torch.testing.assert_close(g, r, atol=1e-4, rtol=1e-4)


def test_projection_rounding_and_gradient_contract():
    x = torch.randn(3, 7, 33, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(33, device='cuda', requires_grad=True)
    b = torch.randn_like(g, requires_grad=True)
    w = torch.randn(17, 33, device='cuda', dtype=x.dtype, requires_grad=True)
    bias = torch.randn(17, device='cuda', dtype=x.dtype, requires_grad=True)
    y = cuda_layernorm_linear(x, g, b, w, bias)
    expected = torch.nn.functional.linear(reference(x, g, b, False, 1e-5), w, bias)
    torch.testing.assert_close(y, expected, atol=.1, rtol=.02)
    dy = torch.randn_like(y)
    for actual, ref in zip(torch.autograd.grad(y, (x, g, b, w, bias), dy), torch.autograd.grad(expected, (x, g, b, w, bias), dy)):
        torch.testing.assert_close(actual, ref, atol=.2, rtol=.03)


def test_invalid_config_rejected_before_launch():
    x = torch.randn(3, 1024, device='cuda')
    with pytest.raises(ValueError, match='configuration'):
        cuda_layernorm(x, threads=512)
