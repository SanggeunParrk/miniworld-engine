# vendored from team-gm origin/miniworld@7c3c67e : src/team_gm/modules/kernels/triangle_attention_pair_bias.py
from miniworld_engine.autotune.configs import configs_for
import torch

from miniworld_engine.kernels._compile import opaque
from miniworld_engine import settings
import triton
import triton.language as tl

import os

from einops import rearrange, repeat, reduce

from miniworld_engine.autotune.shape_key import token_key


AUTOTUNE = settings.current().autotunes("tri_attention")


def get_seq_group(L) -> int:
    """Delegates to canonical size-bucketing (autotune.buckets)."""
    from miniworld_engine.autotune.buckets import bucket_squared
    return bucket_squared(L * L)


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,  #
    k_ptr,
    v_ptr,  #
    start_m,
    qk_scale,  #
    b_ptr,  # <--- ADDED
    stride_kn,
    stride_vn,
    stride_bm,
    stride_bn,  #
    BLOCK_M1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    N_CTX,
    EVEN_N: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    lo, hi = 0, N_CTX
    # loop over k, v and update accumulator
    offset_m = start_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    offset_k = tl.arange(0, HEAD_DIM_PAD)

    dtype = k_ptr.dtype.element_ty

    for start_n in range(lo, hi, BLOCK_M2):
        # start_n = tl.multiple_of(start_n, BLOCK_M2)
        # -- compute qk ----
        if EVEN_N:
            k = tl.load(k_ptr, mask=offset_k[:, None] < HEAD_DIM, other=0.0)
            bias_val = tl.load(b_ptr)
            v = tl.load(v_ptr, mask=offset_k[None, :] < HEAD_DIM, other=0.0)
        else:
            offset_n = start_n + tl.arange(0, BLOCK_M2)
            bias_mask = (offset_m[:, None] < N_CTX) & (offset_n[None, :] < N_CTX)
            bias_val = tl.load(b_ptr, mask=bias_mask, other=float("-inf"))
            k = tl.load(k_ptr, mask=offset_n[None, :] < N_CTX, other=0.0)
            v = tl.load(v_ptr, mask=offset_n[:, None] < N_CTX, other=0.0)

        qk = tl.dot(q, k, allow_tf32=False) + bias_val / (qk_scale / 1.44269504)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        p = tl.maximum(p, 0.0)

        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        alpha = tl.maximum(alpha, 0.0)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        acc = tl.dot(p.to(dtype), v, acc, allow_tf32=False)
        # update m_i and l_i
        m_i = m_ij
        b_ptr += BLOCK_M2 * stride_bn
        k_ptr += BLOCK_M2 * stride_kn
        v_ptr += BLOCK_M2 * stride_vn
    return acc, l_i, m_i


# fmt: off


@triton.autotune(configs=configs_for("triangle_attention_fwd_contig_triton"),
                 key=['shape_key', 'H', 'HEAD_DIM'])
@triton.jit
def _attn_fwd(
    Q, K, V, Bias, sm_scale,
    M, Out,
    stride_qz, stride_qh, stride_qt, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kt, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vt, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_ot, stride_om, stride_on,
    stride_bz, stride_bh, stride_bm, stride_bn,
    stride_mz, stride_mh, stride_mt, stride_mm,
    Z, H: tl.constexpr, N_CTX, HEAD_DIM: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_M2: tl.constexpr, HEAD_DIM_PAD: tl.constexpr,
    shape_key,
):
    tl.static_assert(HEAD_DIM_PAD >= HEAD_DIM)
    start_m = tl.program_id(0).to(tl.int64)
    off_hz = tl.program_id(1).to(tl.int64)
    off_t = tl.program_id(2).to(tl.int64)
    off_z = off_hz // H
    off_h = off_hz % H
    qkv_offset = (
        off_z.to(tl.int64) * stride_qz
        + off_h.to(tl.int64) * stride_qh
        + off_t.to(tl.int64) * stride_qt
    )
    bias_offset = off_z * stride_bz + off_h * stride_bh

    offset_m = start_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    offset_n = tl.arange(0, BLOCK_M2)
    offset_k = tl.arange(0, HEAD_DIM_PAD)

    q_ptr = (
        Q + qkv_offset + offset_m[:, None] * stride_qm + offset_k[None, :] * stride_qk
    )
    k_ptr = (
        K + qkv_offset + offset_n[None, :] * stride_kn + offset_k[:, None] * stride_kk
    )
    v_ptr = (
        V + qkv_offset + offset_n[:, None] * stride_vn + offset_k[None, :] * stride_vk
    )
    o_ptr = (
        Out + qkv_offset + offset_m[:, None] * stride_om + offset_k[None, :] * stride_on
    )
    bias_ptr = (
        Bias
        + bias_offset
        + offset_m[:, None] * stride_bm
        + offset_n[None, :] * stride_bn
    )

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M1], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M1, HEAD_DIM_PAD], dtype=tl.float32)

    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    EVEN_N = (N_CTX % BLOCK_M1 == 0) & (N_CTX % BLOCK_M2 == 0)
    EVEN_D = HEAD_DIM % HEAD_DIM_PAD == 0

    # load q: it will stay in SRAM throughout
    if EVEN_N and EVEN_D:
        q = tl.load(q_ptr)
    elif not EVEN_N and EVEN_D:
        q = tl.load(q_ptr, mask=offset_m[:, None] < N_CTX, other=0.0)
    elif EVEN_N and not EVEN_D:
        q = tl.load(q_ptr, mask=offset_k[None, :] < HEAD_DIM, other=0.0)
    else:
        q_mask = (offset_m[:, None] < N_CTX) & (offset_k[None, :] < HEAD_DIM)
        q = tl.load(q_ptr, mask=q_mask, other=0.0)

    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        k_ptr,
        v_ptr,  #
        start_m,
        qk_scale,
        bias_ptr,  #
        stride_kn,
        stride_vn,  #
        stride_bm,
        stride_bn,  #
        BLOCK_M1,
        BLOCK_M2,
        HEAD_DIM_PAD,
        HEAD_DIM,
        N_CTX,
        EVEN_N,
        EVEN_D,
    )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = (
        M
        + off_z * stride_mz
        + off_h * stride_mh
        + off_t * stride_mt
        + start_m * BLOCK_M1
        + tl.arange(0, BLOCK_M1)
    )

    if EVEN_N and EVEN_D:
        tl.store(m_ptrs, m_i)
        tl.store(o_ptr, acc.to(Out.type.element_ty))
    elif not EVEN_N and EVEN_D:
        tl.store(m_ptrs, m_i, mask=offset_m < N_CTX)
        tl.store(o_ptr, acc.to(Out.type.element_ty), mask=offset_m[:, None] < N_CTX)
    elif EVEN_N and not EVEN_D:
        tl.store(m_ptrs, m_i)
        tl.store(o_ptr, acc.to(Out.type.element_ty), mask=offset_k[None, :] < HEAD_DIM)
    else:
        tl.store(m_ptrs, m_i, mask=offset_m < N_CTX)
        out_mask = (offset_m[:, None] < N_CTX) & (offset_k[None, :] < HEAD_DIM)
        tl.store(o_ptr, acc.to(Out.type.element_ty), mask=out_mask)
# fmt: on


# fmt: off


# The launcher hands this kernel H*L, not H, because q is the "B (H L) L2 D" rearrangement. The
# parameter was named H and keyed on: token-derived, so it partitioned the cache per sequence length
# and defeated the shape_key bucket sitting next to it -- for a value the body never reads. Dropped.
@triton.autotune(configs=configs_for("triangle_attention_bwd_pre_contig_triton"),
                 key=['shape_key', 'HEAD_DIM'])
@triton.jit
def _attn_bwd_preprocess(
    O, DO, Delta,
    Z, N_CTX, HEAD_DIM: tl.constexpr,
    BLOCK_M1: tl.constexpr, HEAD_DIM_PAD: tl.constexpr,
    shape_key,
):
    off_m = tl.program_id(0).to(tl.int64) * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    off_hz = tl.program_id(1).to(tl.int64)
    mask_delta = off_m < N_CTX

    # delta[m] = sum_d o[m,d]*do[m,d]: a reduction over d, so HEAD_DIM_PAD tiles it and the partials
    # accumulate. The accumulator is fp32 (the per-tile tl.sum already promoted to fp32 before);
    # summation is associative, so the tiled result matches the single-tile one.
    delta = tl.zeros([BLOCK_M1], dtype=tl.float32)
    for d0 in range(0, HEAD_DIM, HEAD_DIM_PAD):
        off_n = d0 + tl.arange(0, HEAD_DIM_PAD)
        o_ptr = O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
        do_ptr = DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
        mask_m = (off_m[:, None] < N_CTX) & (off_n[None, :] < HEAD_DIM)
        ot = tl.load(o_ptr, mask=mask_m, other=0.0)
        dot = tl.load(do_ptr, mask=mask_m, other=0.0)
        delta += tl.sum(ot * dot, axis=1)

    delta_ptr = Delta + off_hz * N_CTX + off_m
    tl.store(delta_ptr, delta, mask=mask_delta)
# fmt: on


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dqdkdv(
    dk,
    dv,
    DBias,
    Q,
    k,
    v,
    Bias,
    qk_scale,  #
    DO,
    DQ,  #
    M,
    D,  #
    EVEN_N,
    EVEN_D,  #
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,  #
    N_CTX,
    BLOCK_M1: tl.constexpr,  #
    BLOCK_M2: tl.constexpr,  #
    HEAD_DIM_PAD: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    # Filled in by the wrapper.
    start_n,
    start_m,
    stride_bm,
    stride_bn,
):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_M2)
    offs_k = tl.arange(0, HEAD_DIM_PAD)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dqT_ptrs = DQ + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    biasT_ptrs = Bias + offs_m[None, :] * stride_bm + offs_n[:, None] * stride_bn
    dbiasT_ptrs = DBias + offs_m[None, :] * stride_bm + offs_n[:, None] * stride_bn
    m_ptrs = M + offs_m
    d_ptrs = D + offs_m

    dtype = DO.dtype.element_ty

    lo, hi = 0, N_CTX
    for start_m in range(lo, hi, BLOCK_M1):
        offs_m = start_m + tl.arange(0, BLOCK_M1)
        qT_mask = (offs_m[None, :] < N_CTX) & (offs_k[:, None] < HEAD_DIM)
        do_mask = (offs_m[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM)
        qT = tl.load(qT_ptrs, mask=qT_mask, other=0.0)

        bias_mask = (offs_m[None, :] < N_CTX) & (offs_n[:, None] < N_CTX)
        biasT = tl.load(biasT_ptrs, mask=bias_mask, other=float("-inf"))
        do = tl.load(do_ptrs, mask=do_mask, other=0.0)
        m = tl.load(m_ptrs, mask=offs_m < N_CTX, other=0.0)
        Di = tl.load(d_ptrs, mask=offs_m < N_CTX, other=0.0)

        # qT = qT.to(tl.bfloat16)
        qkT = tl.dot(k, qT, allow_tf32=False) + biasT / (qk_scale / 1.44269504)
        qkT = qkT * qk_scale - m[None, :]
        # qkT = qkT * qk_scale - m[:, None]

        pT = tl.math.exp2(qkT)
        # Compute dV.

        dv = tl.dot(pT.to(dtype), do, dv, allow_tf32=False)

        # D (= delta) is pre-divided by ds_scale.
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do), allow_tf32=False)
        # dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - Di[None, :])

        if DBias is not None:  # or just skip the check if always passing DBias
            if EVEN_N:
                tl.store(dbiasT_ptrs, dsT)
            else:
                mask = (offs_m[None, :] < N_CTX) & (offs_n[:, None] < N_CTX)
                tl.store(dbiasT_ptrs, dsT, mask=mask)

        dsT = dsT.to(dtype)
        dk = tl.dot(dsT, tl.trans(qT), dk, allow_tf32=False)
        dqT_to_add = tl.dot(tl.trans(k), dsT, allow_tf32=False) * (qk_scale / 1.44269504)
        tl.atomic_add(dqT_ptrs, dqT_to_add, qT_mask)

        # Increment pointers.
        qT_ptrs += BLOCK_M1 * stride_tok
        do_ptrs += BLOCK_M1 * stride_tok
        dqT_ptrs += BLOCK_M1 * stride_tok
        biasT_ptrs += BLOCK_M1 * stride_bm
        dbiasT_ptrs += BLOCK_M1 * stride_bm
        m_ptrs += BLOCK_M1
        d_ptrs += BLOCK_M1
    return dk, dv


# fmt: off


# HL, not H: `bhid` runs over B*HL because q is the "B (H L) L2 D" rearrangement, so the decode
# below divides by H*L. Keying it would partition the cache per sequence length and make shape_key
# redundant, so only the bucket is keyed.
@triton.autotune(configs=configs_for("triangle_attention_bwd_atomic_triton"),
                 key=['shape_key', 'HEAD_DIM'],
                 reset_to_zero=['DQ'])
@triton.jit
def _attn_bwd(
    Q, K, V, Bias, sm_scale,
    DO, DQ, DK, DV, DBias,
    M, D,
    stride_z, stride_h, stride_tok, stride_d,
    bias_stride_z, bias_stride_h, bias_stride_m, bias_stride_n,
    HL, N_CTX, HEAD_DIM: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_M2: tl.constexpr, HEAD_DIM_PAD: tl.constexpr,
    shape_key,
):
    tl.static_assert(HEAD_DIM_PAD >= HEAD_DIM)

    bhid = tl.program_id(2).to(tl.int64)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % HL) + stride_z * (bhid // HL)).to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    # Also offset DBias so it points to start of [bhid, :, :].
    # If DBias has shape [B*H, N_CTX, N_CTX], do:
    # offset_bias = (bhid * N_CTX * N_CTX).to(tl.int64)
    offset_bias = (bias_stride_h * (bhid % HL) + bias_stride_z * (bhid // HL)).to(tl.int64)
    Bias = Bias + offset_bias
    DBias = DBias + offset_bias

    # load scales
    offs_k = tl.arange(0, HEAD_DIM_PAD)

    start_n = pid * BLOCK_M2
    offs_n = start_n + tl.arange(0, BLOCK_M2)

    dv = tl.zeros([BLOCK_M2, HEAD_DIM_PAD], dtype=tl.float32)
    dk = tl.zeros([BLOCK_M2, HEAD_DIM_PAD], dtype=tl.float32)

    EVEN_N = (N_CTX % BLOCK_M2 == 0) & (N_CTX % BLOCK_M1 == 0)
    EVEN_D = HEAD_DIM % HEAD_DIM_PAD == 0

    # load K and V: they stay in SRAM throughout the inner loop.
    k_ptr = K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    v_ptr = V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d

    if EVEN_N and EVEN_D:
        k = tl.load(k_ptr)
        v = tl.load(v_ptr)
    elif not EVEN_N and EVEN_D:
        k = tl.load(k_ptr, mask=offs_n[:, None] < N_CTX, other=0.0)
        v = tl.load(v_ptr, mask=offs_n[:, None] < N_CTX, other=0.0)
    elif EVEN_N and not EVEN_D:
        k = tl.load(k_ptr, mask=offs_k[None, :] < HEAD_DIM, other=0.0)
        v = tl.load(v_ptr, mask=offs_k[None, :] < HEAD_DIM, other=0.0)
    else:
        k = tl.load(
            k_ptr,
            mask=(offs_n[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM),
            other=0.0,
        )
        v = tl.load(
            v_ptr,
            mask=(offs_n[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM),
            other=0.0,
        )

    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    # load

    # Compute dK and dV for non-masked blocks.
    dk, dv = _attn_bwd_dqdkdv(  #
        dk,
        dv,
        DBias,
        Q,
        k,
        v,
        Bias,
        qk_scale,  #
        DO,
        DQ,  #
        M,
        D,  #
        EVEN_N,
        EVEN_D,  #
        stride_tok,
        stride_d,  #
        N_CTX,  #
        BLOCK_M1,
        BLOCK_M2,
        HEAD_DIM_PAD,
        HEAD_DIM,  #
        start_n,
        start_m=0,
        stride_bm=bias_stride_m,
        stride_bn=bias_stride_n,
    )

    # offs_q = pid * BLOCK_M2 + tl.arange(0, BLOCK_M1)
    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    dk = dk * sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d

    if EVEN_N and EVEN_D:
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)
    elif not EVEN_N and EVEN_D:
        tl.store(dv_ptrs, dv, mask=offs_n[:, None] < N_CTX)
        tl.store(dk_ptrs, dk, mask=offs_n[:, None] < N_CTX)
    elif EVEN_N and not EVEN_D:
        tl.store(dv_ptrs, dv, mask=offs_k[None, :] < HEAD_DIM)
        tl.store(dk_ptrs, dk, mask=offs_k[None, :] < HEAD_DIM)
    else:
        tl.store(
            dv_ptrs, dv, mask=(offs_n[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM)
        )
        tl.store(
            dk_ptrs, dk, mask=(offs_n[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM)
        )
# fmt: on


def _atomic_fwd_fake(q, k, v, bias, shape_key):
    """``(o, M)``: ``o`` like ``q``; ``M`` is the ``(B, H, L, L)`` log2-space logsumexp and stays
    fp32 even though the forward saves the activations as bf16 -- the backward recomputes
    ``p = exp2(qk*scale - M)`` from it, so bf16's digits would land in an exponent.
    """
    B, H, L, _L2, _D = q.shape
    return torch.empty_like(q), q.new_empty((B, H, L, L), dtype=torch.float32)


@opaque(fake=_atomic_fwd_fake, name="triangle_attention_atomic_fwd")
def _atomic_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor,
                shape_key: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The attention launch -> ``(o, M)``. Every tensor arrives contiguous and shape-checked.

    Split out of the Function so the validation, the contiguous() calls and
    ``save_for_backward`` stay traceable -- see ``kernels._compile``.
    """
    B, H, L, _L2, D = q.shape
    sm_scale = D**-0.5
    o = torch.empty_like(q)
    M = torch.empty(B, H, L, L, device=q.device, dtype=torch.float32)

    # fmt: off
    grid = lambda META: [triton.cdiv(L, META["BLOCK_M1"]), B * H, L]
    _attn_fwd[grid](
        q, k, v, bias, sm_scale,
        M, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3), q.stride(4),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3), k.stride(4),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3), v.stride(4),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3), o.stride(4),
        bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3),
        M.stride(0), M.stride(1), M.stride(2), M.stride(3),
        B, H, L, D,
        shape_key=shape_key,
        HEAD_DIM_PAD=triton.next_power_of_2(D),
    )
    # fmt: on

    return o, M


def _atomic_bwd_fake(q, k, v, bias, o, M, grad_output, shape_key):
    """The four gradients in the SPLIT-head layout the body rearranges back to.

    The arguments arrive merged -- ``q`` is ``(B, H*L, L2, D)``, ``bias`` the repeated
    ``(B, H*L3, L, L2)`` -- but the returns are not: the body rearranges dq/dk/dv to
    ``(B, H, L, L2, D)``, casts dq back to ``q``'s dtype, and reduces dbias over the repeated row
    axis to ``(B, H, L, L2)``. A fake echoing the INPUT shapes has the right arity and the wrong
    metadata, so nothing catches it until a compiled run does.
    """
    B, HL, L2, D = q.shape
    L = L2
    H = HL // L
    return (
        q.new_empty((B, H, L, L2, D)),
        k.new_empty((B, H, L, L2, D)),
        v.new_empty((B, H, L, L2, D)),
        bias.new_empty((B, H, L, L2)),
    )


@opaque(fake=_atomic_bwd_fake, name="triangle_attention_atomic_bwd")
def _atomic_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor,
                o: torch.Tensor, M: torch.Tensor, grad_output: torch.Tensor,
                shape_key: int) -> tuple[
                    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The two backward launches -> ``(dq, dk, dv, dbias)``; ``dq`` comes back fp32."""
    B, HL, L, D = q.shape
    sm_scale = D**-0.5
    delta = torch.empty_like(M)

    # fmt: off
    grid = lambda META: [triton.cdiv(L, META["BLOCK_M1"]), B * HL, 1]
    _attn_bwd_preprocess[grid](
        o, grad_output, delta,
        B, L, D,
        shape_key=shape_key,
        HEAD_DIM_PAD=triton.next_power_of_2(D),
    )
    # fmt: on

    dq = torch.zeros_like(q).to(torch.float32)
    dv = torch.empty_like(v)
    dk = torch.empty_like(k)
    dbias = torch.empty_like(bias)

    # fmt: off
    grid = lambda META: [triton.cdiv(L, META["BLOCK_M2"]), 1, B * HL]
    _attn_bwd[grid](
        q, k, v, bias, sm_scale,
        grad_output, dq, dk, dv, dbias,
        M, delta,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3),
        HL, L, D,
        HEAD_DIM_PAD=triton.next_power_of_2(D),
        shape_key=shape_key,
    )
    # fmt: on

    # q was cast to the Function's op_dtype before it got here, so this IS that dtype --
    # named off an argument instead of a caller local, which was a NameError.
    dq = rearrange(dq, "B (H L) L2 D -> B H L L2 D", L=L).to(q.dtype)
    dk = rearrange(dk, "B (H L) L2 D -> B H L L2 D", L=L)
    dv = rearrange(dv, "B (H L) L2 D -> B H L L2 D", L=L)
    dbias = reduce(dbias, "B (H L3) L L2 -> B H L L2", "sum", L3=L)
    return dq, dk, dv, dbias


class TritonTriangleAttentionPairBiasFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
    ):
        op_dtype = q.dtype
        if not q.dtype == k.dtype == v.dtype == bias.dtype == op_dtype:
            raise ValueError(
                f"q, k, v, and bias must have the same dtype, "
                f"but got {q.dtype=}, {k.dtype=}, {v.dtype=}, {bias.dtype=}"
            )
        if not q.shape == k.shape == v.shape:
            raise ValueError(
                f"q, k, v must have the same shape, "
                f"but got {q.shape=}, {k.shape=}, {v.shape=}"
            )
        if q.ndim != 5:
            raise ValueError(f"q, k, v must have 5D, but got {q.ndim=}D")

        q, k, v, bias = [x.contiguous() for x in (q, k, v, bias)]
        B, H, L, L2, D = q.shape
        if L != L2:
            raise ValueError(f"q, k, v must have square shape, but got {L=}, {L2=}")
        if D != 32:
            raise ValueError(f"Only support D=32, but got {D=}")
        if bias.shape != (B, H, L, L):
            raise ValueError(f"bias must have shape {B, H, L, L}, but got {bias.shape=}")

        o, M = _atomic_fwd(q, k, v, bias, token_key(L))

        # M is a log2-space logsumexp, not an activation: backward recomputes
        # p = exp2(qk*scale - M) from it, so bf16's ~3 significant digits land in an
        # exponent. main.py keeps it fp32 and so does this path.
        ctx.save_for_backward(
            q.to(torch.bfloat16),
            k.to(torch.bfloat16),
            v.to(torch.bfloat16),
            bias.to(torch.bfloat16),
            o.to(torch.bfloat16),
            M,
        )
        ctx.op_dtype = op_dtype
        return o

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        op_dtype = ctx.op_dtype
        # This file's kernels take no stride arguments -- `_attn_bwd_preprocess` addresses O/DO as
        # off_hz*HEAD_DIM*N_CTX + off_m*HEAD_DIM + off_n. forward already enforces that on its
        # inputs (see the .contiguous() below the shape checks); backward did not, so an upstream
        # grad that is a stride-0 expanded view -- what `out.sum().backward()` hands us -- had one
        # element of storage and the kernel read L*D past it. main.py is immune because its
        # preprocess takes explicit 5D strides.
        grad_output = grad_output.contiguous()
        q, k, v, bias, o = [x.to(op_dtype) for x in ctx.saved_tensors[:-1]]
        M = ctx.saved_tensors[-1]                       # stays fp32, so delta does too
        q, k, v, o, grad_output = [
            rearrange(x, "B H L L2 D -> B (H L) L2 D") for x in (q, k, v, o, grad_output)
        ]
        bias = repeat(bias, "B H L L2 -> B (H L3) L L2", L3=bias.shape[2])
        M = rearrange(M, "B H L L2 -> B (H L) L2")

        dq, dk, dv, dbias = _atomic_bwd(q, k, v, bias, o, M, grad_output, token_key(L))
        return dq, dk, dv, dbias


triton_triangle_attention_pair_bias = TritonTriangleAttentionPairBiasFunction.apply
