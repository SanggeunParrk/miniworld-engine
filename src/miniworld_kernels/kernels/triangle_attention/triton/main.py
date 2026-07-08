# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/triangle_attention_pair_bias.py
import os

import torch
import triton
import triton.language as tl
from einops import rearrange, reduce, repeat
from jaxtyping import Float

from miniworld_kernels._typecheck import typecheck

AUTOTUNE = os.getenv("TRITON_AUTOTUNE", "0").lower() == "tri_attention"


def get_seq_group(length: int) -> int:
    """Get sequence group based on length."""
    GROUP_LENGTHS = [32, 64, 128, 256, 384, 512]
    for group_idx, group_length in enumerate(GROUP_LENGTHS):
        if length <= group_length:
            return group_idx
    return len(GROUP_LENGTHS) - 1


fwd_configs = [
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, 4, 2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, 4, 3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, 4, 2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, 4, 3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, 4, 2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, 8, 3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, 4, 3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, 8, 3),
]

bwd_preprocess_configs = [
    triton.Config({"BLOCK_M": 32}, 4, 3),  # 128,384 at H100
    triton.Config({"BLOCK_M": 32}, 4, 2),  # 256 at H100
    triton.Config({"BLOCK_M": 64}, 8, 2),  # 512 at H100
]

bwd_configs = [
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, 4, 3),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, 4, 3),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, 8, 3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, 4, 3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, 4, 3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, 8, 3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, 4, 3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, 8, 3),
]


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


@triton.autotune(configs=fwd_configs, key=["GROUP_N", "H", "HEAD_DIM"])
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
    start_m = tl.program_id(0)
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


@triton.autotune(configs=bwd_preprocess_configs, key=["GROUP_N", "H", "HEAD_DIM"])
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
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
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


@triton.autotune(configs=bwd_configs, key=["GROUP_N", "H", "HEAD_DIM"])
@triton.jit
def _attn_bwd_dkdv(
    q_ptr,
    k_ptr,
    v_ptr,
    bias_ptr,
    sm_scale,
    do_ptr,
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
    # FA2 dK/dV kernel: fix a key/value block of BLOCK_N, loop over all query rows,
    # accumulate dk/dv in registers, store dbias (=dS) per (query,key). No dq path, no
    # atomics (dq is a separate kernel) -> lower register pressure than the old fused
    # dq+dk+dv kernel, so occupancy is no longer register-walled.
    tl.static_assert(BLOCK_D >= HEAD_DIM)
    bhid = tl.program_id(2).to(tl.int64)
    off_chz = bhid * N_CTX
    adj = stride_h * (bhid % H) + stride_z * (bhid // H)
    pid = tl.program_id(0)
    q_ptr += adj
    k_ptr += adj
    v_ptr += adj
    do_ptr += adj
    dk_ptr += adj
    dv_ptr += adj
    m_ptr += off_chz
    d_ptr += off_chz
    offset_bias = bias_stride_h * (bhid % H) + bias_stride_z * (bhid // H)
    bias_ptr += offset_bias
    dbias_ptr += offset_bias

    offs_k = tl.arange(0, BLOCK_D)
    start_n = pid * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)

    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    kv_mask = (offs_n[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM)
    k = tl.load(
        k_ptr + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d,
        mask=kv_mask, other=0.0,
    )
    v = tl.load(
        v_ptr + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d,
        mask=kv_mask, other=0.0,
    )

    qk_scale = sm_scale * 1.44269504  # 1/log(2)

    offs_m0 = tl.arange(0, BLOCK_M)
    qT_ptrs = q_ptr + offs_m0[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = do_ptr + offs_m0[:, None] * stride_tok + offs_k[None, :] * stride_d
    biasT_ptrs = bias_ptr + offs_m0[None, :] * bias_stride_m + offs_n[:, None] * bias_stride_n
    dbiasT_ptrs = dbias_ptr + offs_m0[None, :] * bias_stride_m + offs_n[:, None] * bias_stride_n
    m_ptrs = m_ptr + offs_m0
    d_ptrs = d_ptr + offs_m0

    for start_m in range(0, N_CTX, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        qT_mask = (offs_m[None, :] < N_CTX) & (offs_k[:, None] < HEAD_DIM)
        do_mask = (offs_m[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM)
        bias_mask = (offs_m[None, :] < N_CTX) & (offs_n[:, None] < N_CTX)
        qT = tl.load(qT_ptrs, mask=qT_mask, other=0.0)
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

        tl.store(dbiasT_ptrs, dsT.to(dbias_ptr.dtype.element_ty), mask=bias_mask)

        dsT = dsT.to(do.dtype)
        dk = tl.dot(dsT, tl.trans(qT), dk)

        qT_ptrs += BLOCK_M * stride_tok
        do_ptrs += BLOCK_M * stride_tok
        biasT_ptrs += BLOCK_M * bias_stride_m
        dbiasT_ptrs += BLOCK_M * bias_stride_m
        m_ptrs += BLOCK_M
        d_ptrs += BLOCK_M

    dk = dk * sm_scale
    tl.store(
        dv_ptr + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d,
        dv.to(dv_ptr.dtype.element_ty), mask=kv_mask,
    )
    tl.store(
        dk_ptr + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d,
        dk.to(dk_ptr.dtype.element_ty), mask=kv_mask,
    )


@triton.autotune(configs=bwd_configs, key=["GROUP_N", "H", "HEAD_DIM"])
@triton.jit
def _attn_bwd_dq(
    q_ptr,
    k_ptr,
    v_ptr,
    bias_ptr,
    sm_scale,
    do_ptr,
    dq_ptr,
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
    # FA2 dQ kernel: fix a query block of BLOCK_M, loop over all key/value blocks,
    # accumulate dq in registers, store once. Recomputes P/dS (cheap) instead of the old
    # atomic_add. Single [BLOCK_M, D] accumulator -> low register pressure -> high occupancy.
    tl.static_assert(BLOCK_D >= HEAD_DIM)
    bhid = tl.program_id(2).to(tl.int64)
    off_chz = bhid * N_CTX
    adj = stride_h * (bhid % H) + stride_z * (bhid // H)
    pid = tl.program_id(0)
    q_ptr += adj
    k_ptr += adj
    v_ptr += adj
    do_ptr += adj
    dq_ptr += adj
    m_ptr += off_chz
    d_ptr += off_chz
    offset_bias = bias_stride_h * (bhid % H) + bias_stride_z * (bhid // H)
    bias_ptr += offset_bias

    offs_k = tl.arange(0, BLOCK_D)
    start_m = pid * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)

    q_mask = (offs_m[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM)
    q = tl.load(
        q_ptr + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d,
        mask=q_mask, other=0.0,
    )
    do = tl.load(
        do_ptr + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d,
        mask=q_mask, other=0.0,
    )
    m = tl.load(m_ptr + offs_m, mask=offs_m < N_CTX, other=0.0)
    Di = tl.load(d_ptr + offs_m, mask=offs_m < N_CTX, other=0.0)
    m_safe = tl.maximum(m, -1e38)

    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504  # 1/log(2)

    offs_n0 = tl.arange(0, BLOCK_N)
    kT_ptrs = k_ptr + offs_n0[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = v_ptr + offs_n0[None, :] * stride_tok + offs_k[:, None] * stride_d
    bias_ptrs = bias_ptr + offs_m[:, None] * bias_stride_m + offs_n0[None, :] * bias_stride_n

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_mask = (offs_n[None, :] < N_CTX) & (offs_k[:, None] < HEAD_DIM)
        bias_mask = (offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX)
        kT = tl.load(kT_ptrs, mask=kv_mask, other=0.0)
        vT = tl.load(vT_ptrs, mask=kv_mask, other=0.0)
        bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)

        qk = tl.dot(q, kT)
        qk = qk + bias / (qk_scale / 1.44269504)
        qk = qk * qk_scale - m_safe[:, None]
        p = tl.math.exp2(qk)
        dp = tl.dot(do, vT)
        ds = p * (dp - Di[:, None])
        ds = ds.to(kT.dtype)
        dq = tl.dot(ds, tl.trans(kT), dq)

        kT_ptrs += BLOCK_N * stride_tok
        vT_ptrs += BLOCK_N * stride_tok
        bias_ptrs += BLOCK_N * bias_stride_n

    dq = dq * sm_scale
    tl.store(
        dq_ptr + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d,
        dq.to(dq_ptr.dtype.element_ty), mask=q_mask,
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
        # if torch.is_autocast_enabled():
        #     dtype = torch.get_autocast_dtype("cuda")
        #     q = q.to(dtype)
        #     k = k.to(dtype)
        #     v = v.to(dtype)
        #     bias = bias.to(dtype)

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

        # FA2 split backward: separate dK/dV and dQ kernels (register-accumulated dq,
        # no atomics) to escape the fused kernel's 255-reg wall. dq is fully written by
        # _attn_bwd_dq, so it needs no zero-init.
        dq = torch.empty_like(q, dtype=torch.float32)
        dv = torch.empty_like(v)
        dk = torch.empty_like(k)
        dbias = torch.empty_like(bias)

        grid_kv = lambda META: [triton.cdiv(L, META["BLOCK_N"]), 1, B * HL]
        _attn_bwd_dkdv[grid_kv](
            q, k, v, bias, sm_scale, grad_output,
            dk, dv, dbias, m, delta,
            *q.stride(),
            *bias.stride(),
            HL, L, D,
            BLOCK_D=triton.next_power_of_2(D),
            GROUP_N=get_seq_group(L),
        )
        grid_q = lambda META: [triton.cdiv(L, META["BLOCK_M"]), 1, B * HL]
        _attn_bwd_dq[grid_q](
            q, k, v, bias, sm_scale, grad_output,
            dq, m, delta,
            *q.stride(),
            *bias.stride(),
            HL, L, D,
            BLOCK_D=triton.next_power_of_2(D),
            GROUP_N=get_seq_group(L),
        )
        dq = rearrange(dq, "B (H L) L2 D -> B H L L2 D", L=L)
        dk = rearrange(dk, "B (H L) L2 D -> B H L L2 D", L=L)
        dv = rearrange(dv, "B (H L) L2 D -> B H L L2 D", L=L)
        dbias = reduce(dbias, "B (H L3) L L2 -> B H L L2", "sum", L3=L)
        return dq, dk, dv, dbias


triton_triangle_attention_pair_bias = TritonTriangleAttentionPairBiasFunction.apply
