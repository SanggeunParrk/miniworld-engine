# vendored from team-gm origin/perf/trimul@3fbb02b : src/team_gm/modules/kernels/triangle_attention_pair_bias.py
import os

import torch
import triton
import triton.language as tl
from einops import rearrange, reduce, repeat
from jaxtyping import Float

from miniworld_kernels._typecheck import typecheck

AUTOTUNE = os.getenv("TRITON_AUTOTUNE", "0").lower() == "tri_attention"

if AUTOTUNE:
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": n}, w, s)
        for m in [16, 32, 64]
        for n in [16, 32, 64]
        for w in [4, 8]
        for s in [2, 3]
    ]
    pre_configs = [
        triton.Config({"BLOCK_M": m}, w, s)
        for m in [16, 32, 64, 128, 256]
        for w in [4, 8]
        for s in [1, 2, 3]
    ]
    bwd_configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": n}, w, s)
        for m in [16, 32, 64]
        for n in [16, 32, 64]
        for w in [4, 8]
        for s in [2, 3]
    ]
else:
    configs = [
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, 4, 2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 16}, 4, 1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, 4, 1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, 4, 3),
    ]
    pre_configs = [
        triton.Config({"BLOCK_M": 128}, 4, 1),
        triton.Config({"BLOCK_M": 64}, 4, 1),
        triton.Config({"BLOCK_M": 64}, 4, 2),
    ]
    bwd_configs = [
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, 4, 2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, 4, 3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, 4, 2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, 4, 3),
    ]


def get_seq_group(length: int) -> int:
    """Get sequence group based on length."""
    GROUP_LENGTHS = [32, 64, 128, 256, 512]
    for group_idx, group_length in enumerate(GROUP_LENGTHS):
        if length <= group_length:
            return group_idx
    return len(GROUP_LENGTHS) - 1


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    k_ptr,
    v_ptr,
    start_m,
    qk_scale,
    b_ptr,
    stride_kn,
    stride_vn,
    stride_bm,
    stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    N_CTX,
):
    offset_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    for start_n in range(0, N_CTX, BLOCK_N):
        offset_n = start_n + tl.arange(0, BLOCK_N)
        bias_mask = (offset_m[:, None] < N_CTX) & (offset_n[None, :] < N_CTX)
        bias_val = tl.load(b_ptr, mask=bias_mask, other=float("-inf"))
        k = tl.load(k_ptr, mask=offset_n[None, :] < N_CTX, other=0.0)
        v = tl.load(v_ptr, mask=offset_n[:, None] < N_CTX, other=0.0)

        qk = tl.dot(q, k)
        qk = qk + bias_val / (qk_scale / 1.44269504)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        m_ij = tl.maximum(m_ij, -1e38)  # Prevent -inf to avoid NaN in subtraction
        qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        p = tl.maximum(p, 0.0)

        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        alpha = tl.maximum(alpha, 0.0)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij
        b_ptr += BLOCK_N * stride_bn
        k_ptr += BLOCK_N * stride_kn
        v_ptr += BLOCK_N * stride_vn
    return acc, l_i, m_i


@triton.autotune(configs=configs, key=["GROUP_N", "H", "HEAD_DIM"])
@triton.jit
def _attn_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    bias_ptr,
    sm_scale,
    m_ptr,
    out_ptr,
    stride_qz,
    stride_qh,
    stride_qt,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kt,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vt,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_ot,
    stride_om,
    stride_on,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_mz,
    stride_mh,
    stride_mt,
    stride_mm,
    Z,
    H: tl.constexpr,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_N: tl.constexpr,
):
    tl.static_assert(BLOCK_D >= HEAD_DIM)
    start_m = tl.program_id(0).to(tl.int64)
    off_hz = tl.program_id(1).to(tl.int64)
    off_t = tl.program_id(2).to(tl.int64)
    off_z = off_hz // H
    off_h = off_hz % H
    qkv_offset = off_z * stride_qz + off_h * stride_qh + off_t * stride_qt
    bias_offset = off_z * stride_bz + off_h * stride_bh

    offset_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = tl.arange(0, BLOCK_N)
    offset_k = tl.arange(0, BLOCK_D)

    q_ptr = (
        q_ptr
        + qkv_offset
        + offset_m[:, None] * stride_qm
        + offset_k[None, :] * stride_qk
    )
    k_ptr = (
        k_ptr
        + qkv_offset
        + offset_n[None, :] * stride_kn
        + offset_k[:, None] * stride_kk
    )
    v_ptr = (
        v_ptr
        + qkv_offset
        + offset_n[:, None] * stride_vn
        + offset_k[None, :] * stride_vk
    )
    o_ptr = (
        out_ptr
        + qkv_offset
        + offset_m[:, None] * stride_om
        + offset_k[None, :] * stride_on
    )
    bias_ptr = (
        bias_ptr
        + bias_offset
        + offset_m[:, None] * stride_bm
        + offset_n[None, :] * stride_bn
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    q_mask = (offset_m[:, None] < N_CTX) & (offset_k[None, :] < HEAD_DIM)
    q = tl.load(q_ptr, mask=q_mask, other=0.0)

    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        k_ptr,
        v_ptr,
        start_m,
        qk_scale,
        bias_ptr,
        stride_kn,
        stride_vn,
        stride_bm,
        stride_bn,
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
        HEAD_DIM,
        N_CTX,
    )
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptr = (
        m_ptr
        + off_z * stride_mz
        + off_h * stride_mh
        + off_t * stride_mt
        + start_m * BLOCK_M
        + tl.arange(0, BLOCK_M)
    )

    tl.store(m_ptr, m_i, mask=offset_m < N_CTX)
    out_mask = (offset_m[:, None] < N_CTX) & (offset_k[None, :] < HEAD_DIM)
    tl.store(o_ptr, acc.to(out_ptr.type.element_ty), mask=out_mask)


@triton.autotune(configs=pre_configs, key=["GROUP_N", "H", "HEAD_DIM"])
@triton.jit
def _attn_bwd_preprocess(
    o,
    DO,
    Delta,
    Z,
    H: tl.constexpr,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_N: tl.constexpr,
):
    off_m = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1).to(tl.int64)
    off_n = tl.arange(0, BLOCK_D)

    o_ptr = o + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
    do_ptr = DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]

    mask_m = (off_m[:, None] < N_CTX) & (off_n[None, :] < HEAD_DIM)
    mask_delta = off_m < N_CTX

    o = tl.load(o_ptr, mask=mask_m, other=0.0).to(tl.float32)
    do = tl.load(do_ptr, mask=mask_m, other=0.0).to(tl.float32)

    delta = tl.sum(o * do, axis=1)

    delta_ptr = Delta + off_hz * N_CTX + off_m
    tl.store(delta_ptr, delta, mask=mask_delta)


@triton.jit
def _attn_bwd_dqdkdv(
    dk,
    dv,
    DBias,
    Q,
    k,
    v,
    Bias,
    qk_scale,
    DO,
    DQ,
    M,
    D,
    stride_tok,
    stride_d,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    start_n,
    start_m,
    stride_bm,
    stride_bn,
):
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_D)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dqT_ptrs = DQ + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    biasT_ptrs = Bias + offs_m[None, :] * stride_bm + offs_n[:, None] * stride_bn
    dbiasT_ptrs = DBias + offs_m[None, :] * stride_bm + offs_n[:, None] * stride_bn
    m_ptrs = M + offs_m
    d_ptrs = D + offs_m

    for inner_start_m in range(0, N_CTX, BLOCK_M):
        offs_m = inner_start_m + tl.arange(0, BLOCK_M)
        qT_mask = (offs_m[None, :] < N_CTX) & (offs_k[:, None] < HEAD_DIM)
        do_mask = (offs_m[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM)
        qT = tl.load(qT_ptrs, mask=qT_mask, other=0.0)

        bias_mask = (offs_m[None, :] < N_CTX) & (offs_n[:, None] < N_CTX)
        biasT = tl.load(biasT_ptrs, mask=bias_mask, other=0.0)
        do = tl.load(do_ptrs, mask=do_mask, other=0.0)
        m = tl.load(m_ptrs, mask=offs_m < N_CTX, other=0.0)
        Di = tl.load(d_ptrs, mask=offs_m < N_CTX, other=0.0)

        qkT = tl.dot(k, qT)
        qkT = qkT + biasT / (qk_scale / 1.44269504)
        m_safe = tl.maximum(m, -1e38)  # Prevent -inf to avoid NaN in subtraction
        qkT = qkT * qk_scale - m_safe[None, :]

        pT = tl.math.exp2(qkT)
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - Di[None, :])
        dv = tl.dot(pT.to(do.dtype), do, dv)

        mask = (offs_m[None, :] < N_CTX) & (offs_n[:, None] < N_CTX)
        tl.store(dbiasT_ptrs, dsT.to(DBias.dtype.element_ty), mask=mask)

        dsT = dsT.to(do.dtype)
        dk = tl.dot(dsT, tl.trans(qT), dk)
        dqT_to_add = tl.dot(tl.trans(k), dsT)
        dqT_to_add = dqT_to_add * (qk_scale / 1.44269504)
        tl.atomic_add(dqT_ptrs, dqT_to_add, qT_mask)

        qT_ptrs += BLOCK_M * stride_tok
        do_ptrs += BLOCK_M * stride_tok
        dqT_ptrs += BLOCK_M * stride_tok
        biasT_ptrs += BLOCK_M * stride_bm
        dbiasT_ptrs += BLOCK_M * stride_bm
        m_ptrs += BLOCK_M
        d_ptrs += BLOCK_M
    return dk, dv


@triton.autotune(
    configs=bwd_configs,
    key=["GROUP_N", "H", "HEAD_DIM"],
    reset_to_zero=["dq_ptr"],
)
@triton.jit
def _attn_bwd(
    q_ptr,
    k_ptr,
    v_ptr,
    bias_ptr,
    sm_scale,
    do_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    dbias_ptr,
    m_ptr,
    d_ptr,
    stride_z,
    stride_h,
    stride_tok,
    stride_d,
    bias_stride_z,
    bias_stride_h,
    bias_stride_m,
    bias_stride_n,
    H: tl.constexpr,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_N: tl.constexpr,
):
    tl.static_assert(BLOCK_D >= HEAD_DIM)

    bhid = tl.program_id(2).to(tl.int64)
    off_chz = bhid * N_CTX
    adj = stride_h * (bhid % H) + stride_z * (bhid // H)
    pid = tl.program_id(0).to(tl.int64)

    q_ptr += adj
    k_ptr += adj
    v_ptr += adj
    do_ptr += adj
    dq_ptr += adj
    dk_ptr += adj
    dv_ptr += adj
    m_ptr += off_chz
    d_ptr += off_chz

    offset_bias = bias_stride_h * (bhid % H) + bias_stride_z * (bhid // H)
    bias_ptr = bias_ptr + offset_bias
    dbias_ptr = dbias_ptr + offset_bias

    offs_k = tl.arange(0, BLOCK_D)

    start_n = pid * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)

    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    k_ptr = k_ptr + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    v_ptr = v_ptr + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d

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

    dk, dv = _attn_bwd_dqdkdv(
        dk,
        dv,
        dbias_ptr,
        q_ptr,
        k,
        v,
        bias_ptr,
        qk_scale,
        do_ptr,
        dq_ptr,
        m_ptr,
        d_ptr,
        stride_tok,
        stride_d,
        H,
        N_CTX,
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
        HEAD_DIM,
        start_n,
        start_m=0,
        stride_bm=bias_stride_m,
        stride_bn=bias_stride_n,
    )

    dv_ptrs = dv_ptr + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    dk = dk * sm_scale
    dk_ptrs = dk_ptr + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d

    tl.store(
        dv_ptrs,
        dv.to(dv_ptr.dtype.element_ty),
        mask=(offs_n[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM),
    )
    tl.store(
        dk_ptrs,
        dk.to(dk_ptr.dtype.element_ty),
        mask=(offs_n[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM),
    )


class TritonTriangleAttentionPairBiasFunction(torch.autograd.Function):
    @typecheck
    @staticmethod
    @torch.compiler.disable()
    def forward(
        ctx,
        q: Float[torch.Tensor, "B H L L D"],
        k: Float[torch.Tensor, "B H L L D"],
        v: Float[torch.Tensor, "B H L L D"],
        bias: Float[torch.Tensor, "B H L L"],
    ) -> Float[torch.Tensor, "B H L L D"]:
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
            q = q.to(dtype)
            k = k.to(dtype)
            v = v.to(dtype)
            bias = bias.to(dtype)

        q, k, v, bias = [x.contiguous() for x in (q, k, v, bias)]
        B, H, L, _, D = q.shape

        sm_scale = D**-0.5
        out = torch.empty_like(q)
        m = torch.empty(B, H, L, L, device=q.device, dtype=torch.float32)

        grid = lambda META: [triton.cdiv(L, META["BLOCK_M"]), B * H, L]
        _attn_fwd[grid](
            q,
            k,
            v,
            bias,
            sm_scale,
            m,
            out,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *out.stride(),
            *bias.stride(),
            *m.stride(),
            B,
            H,
            L,
            D,
            BLOCK_D=triton.next_power_of_2(D),
            GROUP_N=get_seq_group(L),
        )

        ctx.save_for_backward(q, k, v, bias, m, out)
        return out

    @staticmethod
    @torch.compiler.disable()
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, bias, m, out = ctx.saved_tensors

        if grad_output.dtype != q.dtype:
            grad_output = grad_output.to(q.dtype)

        q, k, v, out, grad_output = [
            rearrange(x, "B H L L2 D -> B (H L) L2 D")
            for x in (q, k, v, out, grad_output)
        ]
        bias = repeat(bias, "B H L L2 -> B (H L3) L L2", L3=bias.shape[2])
        m = rearrange(m, "B H L L2 -> B (H L) L2")

        B, HL, L, D = q.shape
        sm_scale = D**-0.5
        delta = torch.empty_like(m)

        grid = lambda META: [triton.cdiv(L, META["BLOCK_M"]), B * HL, 1]
        _attn_bwd_preprocess[grid](
            out,
            grad_output,
            delta,
            B,
            HL,
            L,
            D,
            BLOCK_D=triton.next_power_of_2(D),
            GROUP_N=get_seq_group(L),
        )

        dq = torch.zeros_like(q, dtype=torch.float32)
        dv = torch.empty_like(v)
        dk = torch.empty_like(k)
        dbias = torch.empty_like(bias)

        grid = lambda META: [triton.cdiv(L, META["BLOCK_N"]), 1, B * HL]
        _attn_bwd[grid](
            q,
            k,
            v,
            bias,
            sm_scale,
            grad_output,
            dq,
            dk,
            dv,
            dbias,
            m,
            delta,
            *q.stride(),
            *bias.stride(),
            HL,
            L,
            D,
            BLOCK_D=triton.next_power_of_2(D),
            GROUP_N=get_seq_group(L),
        )

        dq = rearrange(dq, "B (H L) L2 D -> B H L L2 D", L=L)
        dk = rearrange(dk, "B (H L) L2 D -> B H L L2 D", L=L)
        dv = rearrange(dv, "B (H L) L2 D -> B H L L2 D", L=L)
        dbias = reduce(dbias, "B (H L3) L L2 -> B H L L2", "sum", L3=L)
        return dq, dk, dv, dbias


triton_triangle_attention_pair_bias = TritonTriangleAttentionPairBiasFunction.apply
