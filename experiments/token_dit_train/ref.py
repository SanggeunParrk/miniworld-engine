"""Shared test data and the fp64 reference for the training core."""
import math
import torch

LOG2E = 1.0 / math.log(2.0)
D = 48


def make(A, L, H=16, seed=0, bias_scale=1.0, mask_frac=0.0, dev="cuda"):
    """fp32 q, k, v [A, L, H*D] (RMS-normed rows like the model's qk_norm), bias [H, L, L], key mask [A, L] bool."""
    g = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(A, L, H * D, device=dev, generator=g)
    k = torch.randn(A, L, H * D, device=dev, generator=g)
    v = torch.randn(A, L, H * D, device=dev, generator=g)
    bias = torch.randn(H, L, L, device=dev, generator=g) * bias_scale
    mask = None
    if mask_frac > 0:
        mask = torch.rand(A, L, device=dev, generator=g) >= mask_frac
    return q, k, v, bias, mask


def reference(q, k, v, bias, mask=None, dtype=torch.float64):
    """O [A, L, H*D] and natural-log LSE [A, H, L]."""
    A, L, HD = q.shape
    H = HD // D
    qh, kh, vh = (t.to(dtype).view(A, L, H, D).transpose(1, 2) for t in (q, k, v))
    s = qh @ kh.transpose(-1, -2) / math.sqrt(D) + bias.to(dtype)[None]
    if mask is not None:
        s = s.masked_fill(~mask[:, None, None, :], torch.finfo(s.dtype).min)
    lse = torch.logsumexp(s, -1)
    o = torch.softmax(s, -1) @ vh
    return o.transpose(1, 2).reshape(A, L, HD), lse


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


def reference_grads(q, k, v, bias, do, mask=None, dtype=torch.float64):
    """fp64 autograd truth: (O, dq, dk, dv, dbias) for upstream gradient ``do`` [A, L, H*D]."""
    leaves = [t.detach().to(dtype).requires_grad_() for t in (q, k, v, bias)]
    o, _ = reference(*leaves, mask, dtype)
    o.backward(do.to(dtype))
    return (o.detach(), *(t.grad for t in leaves))
