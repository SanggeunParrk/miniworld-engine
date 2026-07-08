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
    q_stride_tok,     # Q (q-group, strided) token/d strides
    q_stride_d,
    o_stride_tok,     # DO/DQ (o-group, contiguous) token/d strides
    o_stride_d,
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
    qT_ptrs = Q + offs_m[None, :] * q_stride_tok + offs_k[:, None] * q_stride_d
    do_ptrs = DO + offs_m[:, None] * o_stride_tok + offs_k[None, :] * o_stride_d
    dqT_ptrs = DQ + offs_m[None, :] * o_stride_tok + offs_k[:, None] * o_stride_d
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

        qT_ptrs += BLOCK_M * q_stride_tok
        do_ptrs += BLOCK_M * o_stride_tok
        dqT_ptrs += BLOCK_M * o_stride_tok
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
    qs_z,          # q/k/v STRIDED 5D strides (B, H, Lrow, tok=Lseq, D)
    qs_h,
    qs_lrow,
    qs_tok,
    qs_d,
    os_z,          # o-group merged-contiguous strides (B, HL, tok, D)
    os_h,
    os_tok,
    os_d,
    bias_stride_z,
    bias_stride_h,
    bias_stride_m,
    bias_stride_n,
    dbias_hl,      # dbias merged HL-dim stride (per-row output, contiguous B,HL,L,L)
    H: tl.constexpr,
    N_CTX,
    HL,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_N: tl.constexpr,
):
    tl.static_assert(BLOCK_D >= HEAD_DIM)

    bhid = tl.program_id(2).to(tl.int64)
    off_chz = bhid * N_CTX
    # decompose merged batch bhid in [0, B*HL) -> (b, h, i_row); HL = H*N_CTX (square L)
    b = bhid // HL
    hl = bhid % HL
    h = hl // N_CTX
    i_row = hl % N_CTX
    adj_q = b * qs_z + h * qs_h + i_row * qs_lrow   # q/k/v strided 5D offset
    adj_o = bhid * os_h                             # o-group contiguous merged offset
    pid = tl.program_id(0)

    q_ptr += adj_q
    k_ptr += adj_q
    v_ptr += adj_q
    do_ptr += adj_o
    dq_ptr += adj_o
    dk_ptr += adj_o
    dv_ptr += adj_o
    m_ptr += off_chz
    d_ptr += off_chz

    bias_ptr = bias_ptr + b * bias_stride_z + h * bias_stride_h   # broadcast over row
    dbias_ptr = dbias_ptr + bhid * dbias_hl                       # per-row output

    offs_k = tl.arange(0, BLOCK_D)

    start_n = pid * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)

    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    k_ptr = k_ptr + offs_n[:, None] * qs_tok + offs_k[None, :] * qs_d
    v_ptr = v_ptr + offs_n[:, None] * qs_tok + offs_k[None, :] * qs_d

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
        qs_tok,        # Q (q-group) token/d strides
        qs_d,
        os_tok,        # DO/DQ (o-group) token/d strides
        os_d,
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

    dv_ptrs = dv_ptr + offs_n[:, None] * os_tok + offs_k[None, :] * os_d
    dk = dk * sm_scale
    dk_ptrs = dk_ptr + offs_n[:, None] * os_tok + offs_k[None, :] * os_d

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
        # if torch.is_autocast_enabled():
        #     dtype = torch.get_autocast_dtype("cuda")
        #     q = q.to(dtype)
        #     k = k.to(dtype)
        #     v = v.to(dtype)
        #     bias = bias.to(dtype)

        # Strided-friendly: _attn_fwd indexes q/k/v/bias via explicit strides (passed below),
        # so we consume the module's strided (B,H,L,L2,D) transpose views directly — no
        # .contiguous() copy. Output/m are freshly allocated contiguous.
        B, H, L, _, D = q.shape

        sm_scale = D**-0.5
        out = torch.empty(B, H, L, L, D, device=q.device, dtype=q.dtype)
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

        # q/k/v stay STRIDED 5D (transpose views) — consumed via explicit 5D strides in-kernel
        # (no (H L) merge copy). The o-group (out/grad_output/dq/dk/dv) is contiguous, so its
        # (H L) merge is a FREE view. bias is read BROADCAST over the query-row (no `repeat`).
        B, H, L, _, D = q.shape                       # square pair: rows == seq == L
        HL = H * L
        sm_scale = D**-0.5
        grad_output = grad_output.contiguous()
        out_m = rearrange(out, "B H L L2 D -> B (H L) L2 D")          # free view (contiguous)
        do_m = rearrange(grad_output, "B H L L2 D -> B (H L) L2 D")   # free view (contiguous)
        m_m = rearrange(m, "B H L L2 -> B (H L) L2")
        delta = torch.empty_like(m_m)

        grid = lambda META: [triton.cdiv(L, META["BLOCK_M"]), B * HL, 1]
        _attn_bwd_preprocess[grid](           # out_m/do_m contiguous -> unchanged kernel
            out_m, do_m, delta, B, HL, L, D,
            BLOCK_D=triton.next_power_of_2(D), GROUP_N=get_seq_group(L),
        )

        dq = torch.zeros(B, HL, L, D, device=q.device, dtype=torch.float32)  # o-group contiguous
        dk = torch.empty(B, HL, L, D, device=q.device, dtype=v.dtype)
        dv = torch.empty(B, HL, L, D, device=q.device, dtype=v.dtype)
        dbias = torch.empty(B, HL, L, L, device=q.device, dtype=bias.dtype)  # per-row, reduced below

        grid = lambda META: [triton.cdiv(L, META["BLOCK_N"]), 1, B * HL]
        _attn_bwd[grid](
            q, k, v, bias, sm_scale, do_m, dq, dk, dv, dbias, m_m, delta,
            *q.stride(),        # q 5D strides: (B, H, Lrow, Lseq/tok, D)  [k,v share pattern]
            *do_m.stride(),     # o-group merged strides: (B, HL, tok, D)  [dq/dk/dv share]
            *bias.stride(),     # bias 5D-ish strides: (B, H, m, n)  broadcast over row
            L * L,              # dbias HL-dim stride (contiguous B,HL,L,L)
            H, L, HL, D,
            BLOCK_D=triton.next_power_of_2(D), GROUP_N=get_seq_group(L),
        )

        dq = rearrange(dq, "B (H L) L2 D -> B H L L2 D", L=L)
        dk = rearrange(dk, "B (H L) L2 D -> B H L L2 D", L=L)
        dv = rearrange(dv, "B (H L) L2 D -> B H L L2 D", L=L)
        dbias = reduce(dbias, "B (H L3) L L2 -> B H L L2", "sum", L3=L)
        return dq, dk, dv, dbias


triton_triangle_attention_pair_bias = TritonTriangleAttentionPairBiasFunction.apply
