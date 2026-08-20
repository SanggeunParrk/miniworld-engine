
from miniworld_engine.kernels._compile import opaque
# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/bias_only_attention.py

from miniworld_engine.autotune.configs import configs_for
import torch
import triton
import triton.language as tl

from einops import rearrange, reduce, repeat
from jaxtyping import Float

from miniworld_engine.autotune import key_bucket_of, tensor_dtype_of
from miniworld_engine._typecheck import typecheck

# HEAD_DIM_PAD was a launch constant (next_power_of_2(HEAD_DIM)). delta = sum_d(o*do) is a plain
# reduction over d, so it tiles with an accumulating loop and HEAD_DIM_PAD joins the sweep.


def get_seq_group(length) -> int:
    """Delegates to canonical size-bucketing (autotune.buckets)."""
    from miniworld_engine.autotune.buckets import bucket_squared
    return bucket_squared(length * length)


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    v_ptr,
    start_m,
    b_ptr,
    stride_vn,
    stride_bn,
    BLOCK_M1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    N_CTX,
):
    offset_m = start_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    offset_k = tl.arange(0, HEAD_DIM_PAD)
    for start_n in range(0, N_CTX, BLOCK_M2):
        offset_n = start_n + tl.arange(0, BLOCK_M2)
        bias_mask = (offset_m[:, None] < N_CTX) & (offset_n[None, :] < N_CTX)
        bias_val = tl.load(b_ptr, mask=bias_mask, other=float("-inf"))
        v_mask = (offset_n[:, None] < N_CTX) & (offset_k[None, :] < HEAD_DIM)
        v = tl.load(v_ptr, mask=v_mask, other=0.0)

        logits = bias_val * 1.44269504  # 1/log(2)
        m_ij = tl.maximum(m_i, tl.max(logits, 1))
        m_ij = tl.maximum(m_ij, -1e38)  # Prevent -inf to avoid NaN in subtraction
        logits = logits - m_ij[:, None]

        p = tl.math.exp2(logits)
        p = tl.maximum(p, 0.0)

        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        alpha = tl.maximum(alpha, 0.0)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij
        b_ptr += BLOCK_M2 * stride_bn
        v_ptr += BLOCK_M2 * stride_vn
    return acc, l_i, m_i




@triton.autotune(configs=configs_for("bias_only_attention_fwd_triton"),
                 key=['GROUP_N', 'H', 'HEAD_DIM'])
@triton.jit
def _attn_fwd(
    v_ptr,
    bias_ptr,
    m_ptr,
    out_ptr,
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
    BLOCK_M1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
    GROUP_N,
):
    tl.static_assert(HEAD_DIM_PAD >= HEAD_DIM)
    start_m = tl.program_id(0).to(tl.int64)
    off_hz = tl.program_id(1).to(tl.int64)
    off_t = tl.program_id(2).to(tl.int64)
    off_z = off_hz // H
    off_h = off_hz % H
    value_offset = off_z * stride_vz + off_h * stride_vh + off_t * stride_vt
    output_offset = off_z * stride_oz + off_h * stride_oh + off_t * stride_ot
    bias_offset = off_z * stride_bz + off_h * stride_bh

    offset_m = start_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    offset_n = tl.arange(0, BLOCK_M2)
    offset_k = tl.arange(0, HEAD_DIM_PAD)

    v_ptr = (
        v_ptr
        + value_offset
        + offset_n[:, None] * stride_vn
        + offset_k[None, :] * stride_vk
    )
    o_ptr = (
        out_ptr
        + output_offset
        + offset_m[:, None] * stride_om
        + offset_k[None, :] * stride_on
    )
    bias_ptr = (
        bias_ptr
        + bias_offset
        + offset_m[:, None] * stride_bm
        + offset_n[None, :] * stride_bn
    )

    m_i = tl.zeros([BLOCK_M1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M1], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M1, HEAD_DIM_PAD], dtype=tl.float32)

    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        v_ptr,
        start_m,
        bias_ptr,
        stride_vn,
        stride_bn,
        BLOCK_M1,
        BLOCK_M2,
        HEAD_DIM_PAD,
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
        + start_m * BLOCK_M1
        + tl.arange(0, BLOCK_M1)
    )

    tl.store(m_ptr, m_i, mask=offset_m < N_CTX)
    out_mask = (offset_m[:, None] < N_CTX) & (offset_k[None, :] < HEAD_DIM)
    tl.store(o_ptr, acc.to(out_ptr.type.element_ty), mask=out_mask)




# The launcher hands the backward H*L, not H: v is the "B (H L) L2 D" rearrangement. Naming it H and
# keying on it partitioned the cache per sequence length -- defeating the GROUP_N bucket next to it --
# for a value this kernel never reads. Dropped.
@triton.autotune(configs=configs_for("bias_only_attention_bwd_pre_triton"),
                 key=['GROUP_N', 'HEAD_DIM'])
@triton.jit
def _attn_bwd_preprocess(
    o,
    DO,
    Delta,
    Z,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
    GROUP_N,
):
    off_m = tl.program_id(0).to(tl.int64) * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    off_hz = tl.program_id(1).to(tl.int64)
    mask_delta = off_m < N_CTX

    # delta[m] = sum_d o[m,d]*do[m,d]: a reduction over d, so HEAD_DIM_PAD tiles it and the fp32
    # partials accumulate (summation is associative -> exact for any tile size).
    delta = tl.zeros([BLOCK_M1], dtype=tl.float32)
    for d0 in range(0, HEAD_DIM, HEAD_DIM_PAD):
        off_n = d0 + tl.arange(0, HEAD_DIM_PAD)
        o_ptr = o + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
        do_ptr = DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
        mask_m = (off_m[:, None] < N_CTX) & (off_n[None, :] < HEAD_DIM)
        ot = tl.load(o_ptr, mask=mask_m, other=0.0).to(tl.float32)
        dot = tl.load(do_ptr, mask=mask_m, other=0.0).to(tl.float32)
        delta += tl.sum(ot * dot, axis=1)

    delta_ptr = Delta + off_hz * N_CTX + off_m
    tl.store(delta_ptr, delta, mask=mask_delta)


@triton.jit
def _attn_bwd_dvdbias(
    dv,
    DBias,
    v,
    Bias,
    DO,
    M,
    D,
    stride_tok,
    stride_d,
    N_CTX,
    BLOCK_M1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    start_n,
    stride_bm,
    stride_bn,
):
    offs_n = start_n + tl.arange(0, BLOCK_M2)
    offs_k = tl.arange(0, HEAD_DIM_PAD)
    do_ptrs = DO + offs_k[None, :] * stride_d
    biasT_ptrs = Bias + offs_n[:, None] * stride_bn
    dbiasT_ptrs = DBias + offs_n[:, None] * stride_bn
    m_ptrs = M
    d_ptrs = D

    for inner_start_m in range(0, N_CTX, BLOCK_M1):
        offs_m = inner_start_m + tl.arange(0, BLOCK_M1)
        do_mask = (offs_m[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM)
        bias_mask = (offs_m[None, :] < N_CTX) & (offs_n[:, None] < N_CTX)

        biasT = tl.load(
            biasT_ptrs + offs_m[None, :] * stride_bm,
            mask=bias_mask,
            other=float("-inf"),
        )
        do = tl.load(do_ptrs + offs_m[:, None] * stride_tok, mask=do_mask, other=0.0)
        m = tl.load(m_ptrs + offs_m, mask=offs_m < N_CTX, other=0.0)
        Di = tl.load(d_ptrs + offs_m, mask=offs_m < N_CTX, other=0.0)

        logitsT = biasT * 1.44269504  # 1/log(2)
        m_safe = tl.maximum(m, -1e38)
        logitsT = logitsT - m_safe[None, :]

        pT = tl.math.exp2(logitsT)
        pT = tl.maximum(pT, 0.0)
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - Di[None, :])
        dv = tl.dot(pT.to(do.dtype), do, dv)

        tl.store(
            dbiasT_ptrs + offs_m[None, :] * stride_bm,
            dsT.to(DBias.dtype.element_ty),
            mask=bias_mask,
        )
    return dv




# HL, not H: `bhid` runs over B*HL, so the decode below divides by H*L. Keyed on GROUP_N only --
# keying the raw product would partition the cache per sequence length.
@triton.autotune(configs=configs_for("bias_only_attention_bwd_triton"),
                 key=['GROUP_N', 'HEAD_DIM'])
@triton.jit
def _attn_bwd(
    v_ptr,
    bias_ptr,
    do_ptr,
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
    HL,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
    GROUP_N,
):
    tl.static_assert(HEAD_DIM_PAD >= HEAD_DIM)

    bhid = tl.program_id(2).to(tl.int64)
    off_chz = bhid * N_CTX
    adj = stride_h * (bhid % HL) + stride_z * (bhid // HL)
    pid = tl.program_id(0).to(tl.int64)

    v_ptr += adj
    do_ptr += adj
    dv_ptr += adj
    m_ptr += off_chz
    d_ptr += off_chz

    offset_bias = bias_stride_h * (bhid % HL) + bias_stride_z * (bhid // HL)
    bias_ptr = bias_ptr + offset_bias
    dbias_ptr = dbias_ptr + offset_bias

    offs_k = tl.arange(0, HEAD_DIM_PAD)

    start_n = pid * BLOCK_M2
    offs_n = start_n + tl.arange(0, BLOCK_M2)

    dv = tl.zeros([BLOCK_M2, HEAD_DIM_PAD], dtype=tl.float32)

    v_ptrs = v_ptr + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    dv_ptrs = dv_ptr + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    v = tl.load(
        v_ptrs,
        mask=(offs_n[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM),
        other=0.0,
    )

    dv = _attn_bwd_dvdbias(
        dv,
        dbias_ptr,
        v,
        bias_ptr,
        do_ptr,
        m_ptr,
        d_ptr,
        stride_tok,
        stride_d,
        N_CTX,
        BLOCK_M1,
        BLOCK_M2,
        HEAD_DIM_PAD,
        HEAD_DIM,
        start_n,
        bias_stride_m,
        bias_stride_n,
    )

    tl.store(
        dv_ptrs,
        dv.to(dv_ptr.dtype.element_ty),
        mask=(offs_n[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM),
    )


class TritonBiasOnlyAttentionFunction(torch.autograd.Function):
    @typecheck
    @staticmethod
    @opaque()
    def forward(
        ctx,
        v: Float[torch.Tensor, "B H L L D"],
        bias: Float[torch.Tensor, "B H L L"],
    ) -> Float[torch.Tensor, "B H L L D"]:
        v, bias = [x.contiguous() for x in (v, bias)]
        B, H, L, _, D = v.shape

        out = torch.empty_like(v)
        m = torch.empty(B, H, L, L, device=v.device, dtype=torch.float32)

        grid = lambda META: [triton.cdiv(L, META["BLOCK_M1"]), B * H, L]
        _attn_fwd[grid](
            v,
            bias,
            m,
            out,
            *v.stride(),
            *out.stride(),
            *bias.stride(),
            *m.stride(),
            B,
            H,
            L,
            D,
            HEAD_DIM_PAD=triton.next_power_of_2(D),
            GROUP_N=get_seq_group(L),
        )

        ctx.save_for_backward(v, bias, m, out)
        return out

    @staticmethod
    @opaque()
    def backward(ctx, grad_output: torch.Tensor):
        v, bias, m, out = ctx.saved_tensors

        if grad_output.dtype != v.dtype:
            grad_output = grad_output.to(v.dtype)
        # `_attn_bwd_preprocess` below takes no stride arguments -- it addresses o/DO by a
        # contiguous formula. An upstream grad that is a stride-0 expanded view, which is what
        # `out.sum().backward()` hands us, has one element of storage and the kernel reads L*D
        # past it. triton/atomic.py had the identical defect; compute-sanitizer reported 14
        # invalid global reads there before the same one-line fix.
        grad_output = grad_output.contiguous()

        v, out, grad_output = [
            rearrange(x, "B H L L2 D -> B (H L) L2 D")
            for x in (v, out, grad_output)
        ]
        bias = repeat(bias, "B H L L2 -> B (H L3) L L2", L3=bias.shape[2])
        m = rearrange(m, "B H L L2 -> B (H L) L2")

        B, HL, L, D = v.shape
        delta = torch.empty_like(m)

        grid = lambda META: [triton.cdiv(L, META["BLOCK_M1"]), B * HL, 1]
        _attn_bwd_preprocess[grid](
            out,
            grad_output,
            delta,
            B,
            L,
            D,
            GROUP_N=get_seq_group(L),
            HEAD_DIM_PAD=triton.next_power_of_2(D),
        )

        dv = torch.empty_like(v)
        dbias = torch.empty_like(bias)

        grid = lambda META: [triton.cdiv(L, META["BLOCK_M2"]), 1, B * HL]
        _attn_bwd[grid](
            v,
            bias,
            grad_output,
            dv,
            dbias,
            m,
            delta,
            *v.stride(),
            *bias.stride(),
            HL,
            L,
            D,
            HEAD_DIM_PAD=triton.next_power_of_2(D),
            GROUP_N=get_seq_group(L),
        )

        dv = rearrange(dv, "B (H L) L2 D -> B H L L2 D", L=L)
        dbias = reduce(dbias, "B (H L3) L L2 -> B H L L2", "sum", L3=L)
        return dv, dbias


triton_bias_only_attention = TritonBiasOnlyAttentionFunction.apply
