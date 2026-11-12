"""A100 (sm_80) hand-CUDA 3D RoPE (``kernels/rope/cuda/sm80``): the standalone rotation and the fused Q/K RMSNorm + rotation, forward and
input backward, against a float32 PyTorch statement of the same math -- on strided Q / K views of an interleaved QKV projection (the
kernel's row / position index is 64-bit; the views' strides exercise the per-row offsets)."""

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.a100]


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = cos.shape[-1]
    c, s = cos[:, :, None, :], sin[:, :, None, :]
    lo, hi, tail = x[..., :half], x[..., half : 2 * half], x[..., 2 * half :]
    return torch.cat([lo * c - hi * s, hi * c + lo * s, tail], dim=-1)


def _norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _inputs(n: int, s: int, h: int, d: int, half: int, dtype: torch.dtype, shared_angles: bool):
    gen = torch.Generator(device="cuda").manual_seed(0)
    qkv = torch.randn(n, s, 3, h, d, device="cuda", generator=gen).to(dtype)
    q, k = qkv[:, :, 0], qkv[:, :, 1]  # strided [N, S, H, D] views
    ang = torch.rand(1 if shared_angles else n, s, half, device="cuda", generator=gen) * 6.283
    return q, k, ang.cos(), ang.sin()


def _loads() -> bool:
    from miniworld_engine.kernels.rope.cuda import sm80

    return torch.cuda.get_device_capability() == (8, 0) and sm80.loads()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("n", "s", "h", "d", "half"), [(2, 1000, 4, 32, 16), (3, 257, 2, 64, 24), (1, 4099, 8, 128, 64)])
def test_rope_and_qk_norm_rope(dtype, n, s, h, d, half) -> None:
    if not _loads():
        pytest.skip("needs an A100 and the sm_80 RoPE build")
    from miniworld_engine.kernels.rope.cuda import sm80

    tol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    eps = torch.finfo(torch.float32).eps
    for shared in (True, False):
        q, k, cos, sin = _inputs(n, s, h, d, half, dtype, shared)
        assert sm80.supports_rope(q, cos, sin) and sm80.supports_qk(q, k, cos, sin)

        x = q.detach().requires_grad_()  # the strided view itself
        y = sm80.rope_3d(x, cos, sin)
        g = torch.randn_like(y)
        y.backward(g)
        xr = q.detach().float().requires_grad_()
        yr = _rotate(xr, cos, sin)
        yr.backward(g.float())
        torch.testing.assert_close(y.float(), yr, atol=tol, rtol=tol)
        assert x.grad is not None and xr.grad is not None
        torch.testing.assert_close(x.grad.float(), xr.grad, atol=tol, rtol=tol)

        qa, ka = q.detach().requires_grad_(), k.detach().requires_grad_()
        oq, ok = sm80.qk_norm_rope_3d(qa, ka, cos, sin, eps)
        gq, gk = torch.randn_like(oq), torch.randn_like(ok)
        torch.autograd.backward([oq, ok], [gq, gk])
        qr, kr = q.detach().float().requires_grad_(), k.detach().float().requires_grad_()
        oqr, okr = _rotate(_norm(qr, eps), cos, sin), _rotate(_norm(kr, eps), cos, sin)
        torch.autograd.backward([oqr, okr], [gq.float(), gk.float()])
        assert qa.grad is not None and ka.grad is not None and qr.grad is not None and kr.grad is not None
        for got, ref in ((oq, oqr), (ok, okr), (qa.grad, qr.grad), (ka.grad, kr.grad)):
            torch.testing.assert_close(got.float(), ref, atol=4 * tol, rtol=4 * tol)
