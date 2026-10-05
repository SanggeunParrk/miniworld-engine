"""A100 general-shape CUDA/cuBLAS path, with opaque launches for compilation.

The tiled family kernels remain the fast paths. These operations cover their tails
without importing Triton or changing FP32 inputs to BF16. Attention recomputes
probabilities in backward and bounds temporary score storage by chunking pair rows.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from miniworld_engine import settings
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._nvcc import ensure_cuda_home, host_flags, load_extension


@torch.compiler.assume_constant_result
def _ampere(device):
    return torch.cuda.get_device_capability(device) == (8, 0)


def enabled(x):
    return (
        x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float32)
        and settings.current().engine_backend != "triton"
        and _ampere(x.device)
    )


@functools.lru_cache(None)
def _ext():
    ensure_cuda_home()
    return load_extension(
        name="a100_native_tails",
        sources=[str(Path(__file__).with_name("cuda_native_rows.cu"))],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-gencode=arch=compute_80,code=sm_80",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def _binary_fake(a, b, mode):
    """Fake of the A100 native binary op: the shape and dtype of its outputs, no work."""
    return a.new_empty(torch.broadcast_shapes(a.shape, b.shape))


@opaque(fake=_binary_fake, name="a100_native_binary")
def _binary(a: torch.Tensor, b: torch.Tensor, mode: int) -> torch.Tensor:
    """A100 native binary (CUDA / cuBLAS kernels), opaque to torch.compile."""
    aa, bb = torch.broadcast_tensors(a, b)
    return _ext().binary(
        aa.contiguous(), bb.to(a.dtype).contiguous(), a.new_empty(0), mode
    )[0]


def _binary_bwd_fake(a, b, dy, mode):
    """Fake of the A100 native binary bwd op: the shape and dtype of its outputs, no work."""
    return a.new_empty(a.shape), b.new_empty(b.shape)


@opaque(fake=_binary_bwd_fake, name="a100_native_binary_backward")
def _binary_bwd(
    a: torch.Tensor, b: torch.Tensor, dy: torch.Tensor, mode: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """A100 native binary backward (CUDA / cuBLAS kernels), opaque to torch.compile."""
    aa, bb = torch.broadcast_tensors(a, b)
    da, db = _ext().binary(
        aa.contiguous(), bb.to(a.dtype).contiguous(), dy.to(a.dtype).contiguous(), mode
    )
    return da.sum_to_size(a.shape), db.sum_to_size(b.shape).to(b.dtype)


class _Binary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, mode):
        ctx.save_for_backward(a, b)
        ctx.mode = mode
        return _binary(a, b, mode)

    @staticmethod
    def backward(ctx, dy):
        return *_binary_bwd(*ctx.saved_tensors, dy, ctx.mode), None


def add(a, b):
    return _Binary.apply(a, b, 0)


def mul(a, b):
    return _Binary.apply(a, b, 1)


def gate(a, b):
    return _Binary.apply(a, b, 2)


def swiglu(a, b):
    return _Binary.apply(a, b, 3)


def _linear_fake(x, w, b):
    """Fake of the A100 native linear op: the shape and dtype of its outputs, no work."""
    return x.new_empty((*x.shape[:-1], w.shape[0]))


@opaque(fake=_linear_fake, name="a100_native_linear")
def _linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
    """A100 native linear (CUDA / cuBLAS kernels), opaque to torch.compile."""
    with torch.autocast(device_type="cuda", enabled=False):
        return F.linear(x, w.to(x.dtype), None if b is None else b.to(x.dtype))


def _linear_bwd_fake(x, w, b, dy):
    """Fake of the A100 native linear bwd op: the shape and dtype of its outputs, no work."""
    return (
        x.new_empty(x.shape),
        w.new_empty(w.shape),
        x.new_empty(0) if b is None else torch.empty_like(b),
    )


@opaque(fake=_linear_bwd_fake, name="a100_native_linear_backward")
def _linear_bwd(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, dy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A100 native linear backward (CUDA / cuBLAS kernels), opaque to torch.compile."""
    xx, g = x.reshape(-1, x.shape[-1]), dy.reshape(-1, dy.shape[-1])
    dx = (g @ w.to(g.dtype)).reshape(x.shape)
    # Preserve FP32 accumulation/output without promoting BF16 operands to
    # SGEMM inputs; cuBLAS can use tensor cores for this exact operand pair.
    dw = (
        torch.mm(g.t(), xx, out_dtype=torch.float32)
        if g.dtype is torch.bfloat16 and xx.dtype is torch.bfloat16
        else g.float().t() @ xx.float()
    )
    return (
        dx,
        dw.to(w.dtype),
        x.new_empty(0) if b is None else g.float().sum(0).to(b.dtype),
    )


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        ctx.save_for_backward(x, w, b)
        return _linear(x, w, b)

    @staticmethod
    def backward(ctx, dy):
        with torch.autocast(device_type="cuda", enabled=False):
            dx, dw, db = _linear_bwd(*ctx.saved_tensors, dy)
        return dx, dw, None if ctx.saved_tensors[2] is None else db


def linear(x, w, b=None):
    if torch.is_autocast_enabled("cuda"):
        x = x.to(torch.get_autocast_dtype("cuda"))
    return _Linear.apply(x, w, b)


def _norm_fake(x, w, b, eps, rms):
    """Fake of the A100 native norm op: the shape and dtype of its outputs, no work."""
    return x.new_empty(x.shape)


@opaque(fake=_norm_fake, name="a100_native_norm")
def _norm(
    x: torch.Tensor,
    w: torch.Tensor | None,
    b: torch.Tensor | None,
    eps: float,
    rms: bool,
) -> torch.Tensor:
    """A100 native norm (CUDA / cuBLAS kernels), opaque to torch.compile."""
    xf = x.float()
    if rms:
        y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
        if w is not None:
            y = y * w.float()
    else:
        y = F.layer_norm(
            xf,
            (x.shape[-1],),
            None if w is None else w.float(),
            None if b is None else b.float(),
            eps,
        )
    return y.to(x.dtype).contiguous()


def _norm_bwd_fake(x, w, b, dy, eps, rms):
    """Fake of the A100 native norm bwd op: the shape and dtype of its outputs, no work."""
    return (
        torch.empty_like(x),
        x.new_empty(0) if w is None else torch.empty_like(w),
        x.new_empty(0) if b is None else torch.empty_like(b),
    )


@opaque(fake=_norm_bwd_fake, name="a100_native_norm_backward")
def _norm_bwd(
    x: torch.Tensor,
    w: torch.Tensor | None,
    b: torch.Tensor | None,
    dy: torch.Tensor,
    eps: float,
    rms: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A100 native norm backward (CUDA / cuBLAS kernels), opaque to torch.compile."""
    xf, g = x.float(), dy.float()
    xc = xf if rms else xf - xf.mean(-1, keepdim=True)
    inv = torch.rsqrt(xc.square().mean(-1, keepdim=True) + eps)
    z = xc * inv
    gz = g if w is None else g * w.float()
    dx = gz - z * (gz * z).mean(-1, keepdim=True)
    if not rms:
        dx = dx - gz.mean(-1, keepdim=True)
    dims = tuple(range(x.ndim - 1))
    dw = x.new_empty(0) if w is None else (g * z).sum(dims).to(w.dtype)
    db = x.new_empty(0) if b is None else g.sum(dims).to(b.dtype)
    return (dx * inv).to(x.dtype), dw, db


class _Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, eps, rms):
        ctx.save_for_backward(x, w, b)
        ctx.eps, ctx.rms = eps, rms
        return _norm(x, w, b, eps, rms)

    @staticmethod
    def backward(ctx, dy):
        dx, dw, db = _norm_bwd(*ctx.saved_tensors, dy, ctx.eps, ctx.rms)
        return (
            dx,
            None if ctx.saved_tensors[1] is None else dw,
            None if ctx.saved_tensors[2] is None else db,
            None,
            None,
        )


def norm(x, w=None, b=None, eps=1e-5, rms=False):
    eps = torch.finfo(x.dtype).eps if eps is None else eps
    if 1 <= x.shape[-1] <= 4096 and (w is None or b is None or w.dtype == b.dtype):
        # Family calls explicitly require CUDA; do not apply the standalone
        # norm dispatch's small-training preference for Triton.
        if rms:
            from miniworld_engine.kernels.rmsnorm.cuda.sm80 import rmsnorm

            return rmsnorm(x, w, eps)
        from miniworld_engine.kernels.layernorm.cuda.sm80 import layernorm

        return layernorm(x, w, b, eps)
    return _Norm.apply(x, w, b, eps, rms)


def _attention_fake(q, k, v, bias, mask, scale, bias_scale):
    """Fake of the A100 native attention op: the shape and dtype of its outputs, no work."""
    return torch.empty_like(q)


def _prob(q, k, bias, mask, scale, bias_scale):
    # [A,B,H,L,D]; FP32 cuBLAS avoids an accidental BF16 conversion of FP32 input.
    score = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    score.add_(bias.float().unsqueeze(0), alpha=bias_scale)
    score.masked_fill_(bias.unsqueeze(0) <= torch.finfo(bias.dtype).min / 2, -math.inf)
    if mask is not None:
        score.masked_fill_(~mask[:, :, None, None, :], -math.inf)
    return _ext().softmax(score.contiguous())


def _chunk(q):
    return max(1, 4_194_304 // math.prod(q.shape[1:4]) // q.shape[3])


@opaque(fake=_attention_fake, name="a100_native_attention")
def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float,
    bias_scale: float,
) -> torch.Tensor:
    """A100 native attention (CUDA / cuBLAS kernels), opaque to torch.compile."""
    out = torch.empty_like(q)
    chunk = _chunk(q)
    for i in range(0, q.shape[0], chunk):
        sl = slice(i, i + chunk)
        p = _prob(
            q[sl], k[sl], bias, None if mask is None else mask[sl], scale, bias_scale
        )
        out[sl] = torch.matmul(p, v[sl].float()).to(q.dtype)
    return out


def _attention_bwd_fake(q, k, v, bias, mask, dy, scale, bias_scale):
    """Fake of the A100 native attention bwd op: the shape and dtype of its outputs, no work."""
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(bias),
    )


@opaque(fake=_attention_bwd_fake, name="a100_native_attention_backward")
def _attention_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor | None,
    dy: torch.Tensor,
    scale: float,
    bias_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A100 native attention backward (CUDA / cuBLAS kernels), opaque to torch.compile."""
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    db = torch.zeros_like(bias, dtype=torch.float32)
    chunk = _chunk(q)
    for i in range(0, q.shape[0], chunk):
        sl = slice(i, i + chunk)
        p = _prob(
            q[sl], k[sl], bias, None if mask is None else mask[sl], scale, bias_scale
        )
        g = dy[sl].float()
        dv[sl] = (p.transpose(-1, -2) @ g).to(v.dtype)
        dp = g @ v[sl].float().transpose(-1, -2)
        ds = p * (dp - (dp * p).sum(-1, keepdim=True))
        db.add_(ds.sum(0), alpha=bias_scale)
        dq[sl] = ((ds @ k[sl].float()) * scale).to(q.dtype)
        dk[sl] = ((ds.transpose(-1, -2) @ q[sl].float()) * scale).to(k.dtype)
    return dq, dk, dv, db.to(bias.dtype)


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias, mask, scale, bias_scale):
        ctx.save_for_backward(q, k, v, bias, mask)
        ctx.scale, ctx.bias_scale = scale, bias_scale
        with torch.autocast(device_type="cuda", enabled=False):
            return _attention(q, k, v, bias, mask, scale, bias_scale)

    @staticmethod
    def backward(ctx, dy):
        with torch.autocast(device_type="cuda", enabled=False):
            grads = _attention_bwd(*ctx.saved_tensors, dy, ctx.scale, ctx.bias_scale)
        return *grads, None, None, None


def attention(q, k, v, bias, mask=None, scale=None, bias_scale=1.0):
    return _Attention.apply(
        q, k, v, bias, mask, q.shape[-1] ** -0.5 if scale is None else scale, bias_scale
    )


def _matmul_fake(a, b):
    """Fake of the A100 native matmul op: the shape and dtype of its outputs, no work."""
    return a.new_empty(
        (*torch.broadcast_shapes(a.shape[:-2], b.shape[:-2]), a.shape[-2], b.shape[-1])
    )


@opaque(fake=_matmul_fake, name="a100_native_matmul")
def _matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """A100 native matmul (CUDA / cuBLAS kernels), opaque to torch.compile."""
    return torch.matmul(a, b)


def _matmul_bwd_fake(a, b, dy):
    """Fake of the A100 native matmul bwd op: the shape and dtype of its outputs, no work."""
    return a.new_empty(a.shape), b.new_empty(b.shape)


@opaque(fake=_matmul_bwd_fake, name="a100_native_matmul_backward")
def _matmul_bwd(
    a: torch.Tensor, b: torch.Tensor, dy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A100 native matmul backward (CUDA / cuBLAS kernels), opaque to torch.compile."""
    return (dy @ b.transpose(-1, -2)).sum_to_size(a.shape).contiguous(), (
        a.transpose(-1, -2) @ dy
    ).sum_to_size(b.shape).contiguous()


class _Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        with torch.autocast(device_type="cuda", enabled=False):
            return _matmul(a, b)

    @staticmethod
    def backward(ctx, dy):
        with torch.autocast(device_type="cuda", enabled=False):
            return _matmul_bwd(*ctx.saved_tensors, dy)


def matmul(a, b):
    return _Matmul.apply(a, b)


def _softmax_fake(x, mask):
    """Fake of the A100 native softmax op: the shape and dtype of its outputs, no work."""
    return x.new_empty(x.shape)


@opaque(fake=_softmax_fake, name="a100_native_softmax")
def _softmax(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """A100 native softmax (CUDA / cuBLAS kernels), opaque to torch.compile."""
    y = x.float().clone()
    if mask is not None:
        y.masked_fill_(~mask, -math.inf)
    return _ext().softmax(y.contiguous()).to(x.dtype)


def _softmax_bwd_fake(p, dy):
    """Fake of the A100 native softmax bwd op: the shape and dtype of its outputs, no work."""
    return torch.empty_like(p)


@opaque(fake=_softmax_bwd_fake, name="a100_native_softmax_backward")
def _softmax_bwd(p: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """A100 native softmax backward (CUDA / cuBLAS kernels), opaque to torch.compile."""
    pp, g = p.float(), dy.float()
    return (pp * (g - (pp * g).sum(-1, keepdim=True))).to(p.dtype)


class _Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        p = _softmax(x, mask)
        ctx.save_for_backward(p)
        return p

    @staticmethod
    def backward(ctx, dy):
        return _softmax_bwd(ctx.saved_tensors[0], dy), None


def softmax(x, mask=None):
    return _Softmax.apply(x, mask)


def _cat_fake(xs, dim):
    """Fake of the A100 native cat op: the shape and dtype of its outputs, no work."""
    shape = list(xs[0].shape)
    shape[dim] = sum(x.shape[dim] for x in xs)
    return xs[0].new_empty(shape)


@opaque(fake=_cat_fake, name="a100_native_concat")
def _cat(xs: list[torch.Tensor], dim: int) -> torch.Tensor:
    """A100 native concat (CUDA / cuBLAS kernels), opaque to torch.compile."""
    return torch.cat(xs, dim)


class _Cat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dim, *xs):
        ctx.dim, ctx.sizes = dim, [x.shape[dim] for x in xs]
        return _cat(list(xs), dim)

    @staticmethod
    def backward(ctx, dy):
        return None, *dy.split(ctx.sizes, ctx.dim)


def cat(xs, dim=0):
    return _Cat.apply(dim, *xs)


def _pair_mask_fake(mask):
    """Fake of the A100 native pair mask op: the shape and dtype of its outputs, no work."""
    return mask.new_empty((mask.shape[0], mask.shape[1], mask.shape[1]))


@opaque(fake=_pair_mask_fake, name="a100_native_pair_mask")
def pair_mask(mask: torch.Tensor) -> torch.Tensor:
    """A100 native pair mask (CUDA / cuBLAS kernels), opaque to torch.compile."""
    return mask[:, :, None] & mask[:, None, :]
