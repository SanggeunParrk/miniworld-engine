"""Stride-aware Q/K RMSNorm + 3D RoPE, with a fused input backward.

Q/K remain views of interleaved QKV storage. Each program handles a tile of
(batch, position, head) rows, reducing over the head dimension. Angle tables
are constants, shared across heads; a partial rotary prefix is supported.
"""

from __future__ import annotations
import torch
import triton
import triton.language as tl
from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import both_key
from miniworld_engine.kernels._compile import opaque


@triton.jit
def _body(
    Q,
    K,
    C,
    S,
    GQ,
    GK,
    OQ,
    OK,
    N,
    SEQ: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    HALF: tl.constexpr,
    Q0,
    Q1,
    Q2,
    Q3,
    K0,
    K1,
    K2,
    K3,
    C0,
    C1,
    C2,
    S0,
    S1,
    S2,
    GQ0,
    GQ1,
    GQ2,
    GQ3,
    GK0,
    GK1,
    GK2,
    GK3,
    EPS: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BACKWARD: tl.constexpr,
):
    BD: tl.constexpr = triton.next_power_of_2(D)
    row = tl.program_id(0) * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    col = tl.arange(0, BD)
    n = row // (SEQ * H)
    pos = (row // H) % SEQ
    head = row % H
    mask = (row[:, None] < N * SEQ * H) & (col[None, :] < D)
    qb = n * Q0 + pos * Q1 + head * Q2
    kb = n * K0 + pos * K1 + head * K2
    q = tl.load(Q + qb[:, None] + col[None, :] * Q3, mask, other=0).to(tl.float32)
    k = tl.load(K + kb[:, None] + col[None, :] * K3, mask, other=0).to(tl.float32)
    rq = tl.rsqrt(tl.sum(q * q, 1) / D + EPS)
    rk = tl.rsqrt(tl.sum(k * k, 1) / D + EPS)
    nq = q * rq[:, None]
    nk = k * rk[:, None]
    rotary = col < 2 * HALF
    partner = tl.where(col < HALF, col + HALF, tl.where(rotary, col - HALF, col))
    index = tl.broadcast_to(partner[None, :], (BLOCK_M1, BD))
    cmask = (row[:, None] < N * SEQ * H) & rotary[None, :]
    c = tl.load(
        C + (n * C0 + pos * C1)[:, None] + (col % HALF)[None, :] * C2, cmask, other=1
    ).to(tl.float32)
    s = tl.load(
        S + (n * S0 + pos * S1)[:, None] + (col % HALF)[None, :] * S2, cmask, other=0
    ).to(tl.float32)
    if BACKWARD:
        gqb = n * GQ0 + pos * GQ1 + head * GQ2
        gkb = n * GK0 + pos * GK1 + head * GK2
        gq = tl.load(GQ + gqb[:, None] + col[None, :] * GQ3, mask, other=0).to(
            tl.float32
        )
        gk = tl.load(GK + gkb[:, None] + col[None, :] * GK3, mask, other=0).to(
            tl.float32
        )
        sign = tl.where(col < HALF, 1.0, -1.0)
        # Match the cast boundary between the rotation and fp32 RMSNorm autograd.
        gnq = (
            (gq * c + tl.gather(gq, index, 1) * s * sign[None, :])
            .to(Q.dtype.element_ty)
            .to(tl.float32)
        )
        gnk = (
            (gk * c + tl.gather(gk, index, 1) * s * sign[None, :])
            .to(K.dtype.element_ty)
            .to(tl.float32)
        )
        oq = rq[:, None] * (gnq - nq * (tl.sum(gnq * nq, 1) / D)[:, None])
        ok = rk[:, None] * (gnk - nk * (tl.sum(gnk * nk, 1) / D)[:, None])
    else:
        nq = nq.to(Q.dtype.element_ty).to(tl.float32)
        nk = nk.to(K.dtype.element_ty).to(tl.float32)
        sign = tl.where(col < HALF, -1.0, 1.0)
        oq = nq * c + tl.gather(nq, index, 1) * s * sign[None, :]
        ok = nk * c + tl.gather(nk, index, 1) * s * sign[None, :]
    tl.store(OQ + row[:, None] * D + col[None, :], oq, mask)
    tl.store(OK + row[:, None] * D + col[None, :], ok, mask)


@triton.autotune(configs=configs_for("qk_norm_rope_fwd_triton"), key=["shape_key"])
@triton.jit
def qk_norm_rope_fwd_kernel(
    Q,
    K,
    C,
    S,
    GQ,
    GK,
    OQ,
    OK,
    N,
    SEQ: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    HALF: tl.constexpr,
    Q0,
    Q1,
    Q2,
    Q3,
    K0,
    K1,
    K2,
    K3,
    C0,
    C1,
    C2,
    S0,
    S1,
    S2,
    GQ0,
    GQ1,
    GQ2,
    GQ3,
    GK0,
    GK1,
    GK2,
    GK3,
    EPS: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    shape_key,
):
    _body(
        Q,
        K,
        C,
        S,
        GQ,
        GK,
        OQ,
        OK,
        N,
        SEQ,
        H,
        D,
        HALF,
        Q0,
        Q1,
        Q2,
        Q3,
        K0,
        K1,
        K2,
        K3,
        C0,
        C1,
        C2,
        S0,
        S1,
        S2,
        GQ0,
        GQ1,
        GQ2,
        GQ3,
        GK0,
        GK1,
        GK2,
        GK3,
        EPS,
        BLOCK_M1,
        False,
    )


@triton.autotune(configs=configs_for("qk_norm_rope_bwd_triton"), key=["shape_key"])
@triton.jit
def qk_norm_rope_bwd_kernel(
    Q,
    K,
    C,
    S,
    GQ,
    GK,
    OQ,
    OK,
    N,
    SEQ: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    HALF: tl.constexpr,
    Q0,
    Q1,
    Q2,
    Q3,
    K0,
    K1,
    K2,
    K3,
    C0,
    C1,
    C2,
    S0,
    S1,
    S2,
    GQ0,
    GQ1,
    GQ2,
    GQ3,
    GK0,
    GK1,
    GK2,
    GK3,
    EPS: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    shape_key,
):
    _body(
        Q,
        K,
        C,
        S,
        GQ,
        GK,
        OQ,
        OK,
        N,
        SEQ,
        H,
        D,
        HALF,
        Q0,
        Q1,
        Q2,
        Q3,
        K0,
        K1,
        K2,
        K3,
        C0,
        C1,
        C2,
        S0,
        S1,
        S2,
        GQ0,
        GQ1,
        GQ2,
        GQ3,
        GK0,
        GK1,
        GK2,
        GK3,
        EPS,
        BLOCK_M1,
        True,
    )


def _forward_fake(q, k, cos, sin, eps, shape_key):
    """Allocate contiguous Q/K outputs for tracing without launching a kernel."""
    return (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty_like(k, memory_format=torch.contiguous_format),
    )


def _launch(launch_kernel, q, k, cos, sin, gq, gk, eps, key):
    n, seq, h, d = q.shape
    oq, ok = _forward_fake(q, k, cos, sin, eps, key)
    if not q.numel():
        return oq, ok
    launch_kernel[lambda m: (triton.cdiv(n * seq * h, m["BLOCK_M1"]),)](
        q,
        k,
        cos,
        sin,
        gq,
        gk,
        oq,
        ok,
        n,
        seq,
        h,
        d,
        cos.shape[-1],
        *q.stride(),
        *k.stride(),
        0 if cos.shape[0] == 1 else cos.stride(0),
        *cos.stride()[1:],
        0 if sin.shape[0] == 1 else sin.stride(0),
        *sin.stride()[1:],
        *gq.stride(),
        *gk.stride(),
        eps,
        shape_key=key,
    )
    return oq, ok


@opaque(fake=_forward_fake, name="qk_norm_rope_fwd")
def _forward(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    shape_key: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run paired Q/K RMSNorm and 3D RoPE preprocessing."""
    return _launch(qk_norm_rope_fwd_kernel, q, k, cos, sin, q, k, eps, shape_key)


def _backward_fake(q, k, cos, sin, gq, gk, eps, shape_key):
    """Allocate Q/K input gradients with the forward input shapes."""
    return _forward_fake(q, k, cos, sin, eps, shape_key)


@opaque(fake=_backward_fake, name="qk_norm_rope_bwd")
def _backward(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    gq: torch.Tensor,
    gk: torch.Tensor,
    eps: float,
    shape_key: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run paired input gradients for RMSNorm and 3D RoPE."""
    return _launch(qk_norm_rope_bwd_kernel, q, k, cos, sin, gq, gk, eps, shape_key)


class _QKNormRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin, eps):
        n, s, h, d = q.shape
        key = both_key(n * s * h, D=d, HALF=cos.shape[-1])
        ctx.save_for_backward(q, k, cos, sin)
        ctx.eps, ctx.key = eps, key
        return _forward(q, k, cos, sin, eps, key)

    @staticmethod
    def backward(ctx, gq, gk):
        q, k, cos, sin = ctx.saved_tensors
        dq, dk = _backward(q, k, cos, sin, gq, gk, ctx.eps, ctx.key)
        return dq, dk, None, None, None


def qk_norm_rope_3d(q, k, cos, sin, eps=torch.finfo(torch.float32).eps):
    """Normalize and rotate both Q/K without cloning their strided QKV views.

    Input gradients are supported. Positional cos/sin tables are fixed inputs,
    matching the existing RoPE kernel's contract.
    """
    if q.shape != k.shape or q.ndim != 4 or not q.is_cuda or k.device != q.device:
        raise ValueError(
            "Q/K must be CUDA tensors of the same [N,S,H,D] shape and device"
        )
    if q.dtype != k.dtype or q.dtype not in (
        torch.float32,
        torch.bfloat16,
        torch.float16,
    ):
        raise ValueError("Q/K must have the same supported floating dtype")
    if (
        cos.shape != sin.shape
        or cos.ndim != 3
        or cos.shape[0] not in (1, q.shape[0])
        or cos.shape[1] != q.shape[1]
    ):
        raise ValueError("cos/sin must have shape [1 or N,S,HALF]")
    if (
        cos.device != q.device
        or sin.device != q.device
        or cos.shape[-1] < 1
        or 2 * cos.shape[-1] > q.shape[-1]
    ):
        raise ValueError("invalid rotary prefix or positional-table device")
    if cos.requires_grad or sin.requires_grad:
        raise ValueError("positional cos/sin tables must be constants")
    return _QKNormRoPE.apply(q, k, cos, sin, eps)
