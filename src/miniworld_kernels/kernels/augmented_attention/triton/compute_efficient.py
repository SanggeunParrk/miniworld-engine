# vendored from team-gm origin/exp/miniworld@32e3897 : src/team_gm/modules/kernels/augmented_attention_pair_bias_compute_efficient.py
import os

import torch
import triton
import triton.language as tl
from jaxtyping import Bool, Float

from miniworld_kernels._typecheck import typecheck

AUTOTUNE = os.getenv("TRITON_AUTOTUNE", "0").lower() == "augmented_attention"


def get_seq_group(length: int) -> int:
    """Get sequence group based on length."""
    GROUP_LENGTHS = [32, 64, 128, 256, 384, 512, 1024, 2048, 3072, 4096]
    for group_idx, group_length in enumerate(GROUP_LENGTHS):
        if length <= group_length:
            return group_idx
    return len(GROUP_LENGTHS) - 1


fwd_configs = [
    triton.Config({"BLOCK_M": m, "BLOCK_N": n}, w, s)
    for m in [32, 64]
    for n in [64, 128]
    for w in [2, 4, 8]
    for s in [2, 4]
]
bwd_pre_configs = [
    triton.Config({"BLOCK_M": m}, w, s)
    for m in [16, 32]
    for w in [1, 2]
    for s in [4, 6, 8]
]
bwd_configs = [
    triton.Config({"BLOCK_M": m, "BLOCK_N": n}, w, s)
    for m in [64, 128]
    for n in [64, 128]
    for w in [2, 4, 8]
    for s in [2, 4]
    if m >= n
]

# dq_expand reduce 시 사용할 최소 BLOCK_N (최대 split 수 계산용)
_BWD_MIN_BLOCK_N = min(c.kwargs["BLOCK_N"] for c in bwd_configs)


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    k_ptr,
    v_ptr,
    start_m,
    start_d,
    qk_scale,
    b_ptr,
    mask_ptr,
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
    offset_k = start_d * BLOCK_D + tl.arange(0, BLOCK_D)
    for start_n in range(0, N_CTX, BLOCK_N):
        offset_n = start_n + tl.arange(0, BLOCK_N)
        bias_mask = (offset_m[:, None] < N_CTX) & (offset_n[None, :] < N_CTX)
        bias_val = tl.load(b_ptr, mask=bias_mask, other=float("-inf"))

        key_mask = tl.load(mask_ptr + offset_n, mask=offset_n < N_CTX, other=False)
        bias_val = tl.where(key_mask[None, :], bias_val, float("-inf"))

        k_mask = (offset_n[None, :] < N_CTX) & (offset_k[:, None] < HEAD_DIM)
        v_mask = (offset_n[:, None] < N_CTX) & (offset_k[None, :] < HEAD_DIM)
        k = tl.load(k_ptr, mask=k_mask, other=0.0)
        v = tl.load(v_ptr, mask=v_mask, other=0.0)

        qk = tl.dot(q, k) + bias_val / (qk_scale / 1.44269504)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        m_ij = tl.maximum(m_ij, -1e38)
        qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
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
    Q,
    K,
    V,
    Bias,
    Mask,
    sm_scale,
    M,
    Out,
    stride_qz,
    stride_qm,
    stride_qh,
    stride_qk,
    stride_kz,
    stride_kn,
    stride_kh,
    stride_kk,
    stride_vz,
    stride_vn,
    stride_vh,
    stride_vk,
    stride_oz,
    stride_om,
    stride_oh,
    stride_on,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_maska,
    stride_maskb,
    A: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_N: tl.constexpr,
):
    start_m = tl.program_id(0).to(tl.int64)
    off_hz = tl.program_id(1).to(tl.int64)
    start_d = tl.program_id(2).to(tl.int64)
    off_z = off_hz // H
    off_h = off_hz % H
    off_a = off_z // B
    off_b = off_z % B
    qvk_offset = off_z * stride_qz + off_h * stride_qh
    bias_offset = off_b * stride_bz + off_h * stride_bh
    mask_offset = off_a * stride_maska + off_b * stride_maskb

    offset_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = tl.arange(0, BLOCK_N)
    offset_k = start_d * BLOCK_D + tl.arange(0, BLOCK_D)

    q_ptr = (
        Q + qvk_offset + offset_m[:, None] * stride_qm + offset_k[None, :] * stride_qk
    )
    k_ptr = (
        K + qvk_offset + offset_n[None, :] * stride_kn + offset_k[:, None] * stride_kk
    )
    v_ptr = (
        V + qvk_offset + offset_n[:, None] * stride_vn + offset_k[None, :] * stride_vk
    )
    o_ptr = (
        Out + qvk_offset + offset_m[:, None] * stride_om + offset_k[None, :] * stride_on
    )
    bias_ptr = (
        Bias
        + bias_offset
        + offset_m[:, None] * stride_bm
        + offset_n[None, :] * stride_bn
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.44269504

    q_mask = (offset_m[:, None] < N_CTX) & (offset_k[None, :] < HEAD_DIM)
    q = tl.load(q_ptr, mask=q_mask, other=0.0)

    mask_ptr = Mask + mask_offset

    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        k_ptr,
        v_ptr,
        start_m,
        start_d,
        qk_scale,
        bias_ptr,
        mask_ptr,
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

    l_i = tl.maximum(l_i, 1e-30)  # guard fully-masked rows: finite 0 output, no 0/0 NaN
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    tl.store(m_ptrs, m_i, mask=offset_m < N_CTX)
    out_mask = (offset_m[:, None] < N_CTX) & (offset_k[None, :] < HEAD_DIM)
    tl.store(o_ptr, acc.to(Out.type.element_ty), mask=out_mask)


@triton.autotune(configs=bwd_pre_configs, key=["GROUP_N", "H", "HEAD_DIM"])
@triton.jit
def _attn_bwd_preprocess(
    O,
    DO,
    Delta,
    A: tl.constexpr,
    B: tl.constexpr,
    N_CTX,
    stride_z: tl.constexpr,
    stride_m: tl.constexpr,
    stride_h: tl.constexpr,
    stride_k: tl.constexpr,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_N: tl.constexpr,
):
    off_m = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_z = tl.program_id(1).to(tl.int64)
    off_h = tl.program_id(2).to(tl.int64)

    offset = off_z * stride_z + off_h * stride_h
    off_k = tl.arange(0, BLOCK_D)

    o_ptr = O + offset + off_m[:, None] * stride_m + off_k[None, :] * stride_k
    do_ptr = DO + offset + off_m[:, None] * stride_m + off_k[None, :] * stride_k

    mask_m = (off_m[:, None] < N_CTX) & (off_k[None, :] < HEAD_DIM)
    mask_delta = off_m < N_CTX

    o = tl.load(o_ptr, mask=mask_m, other=0.0).to(tl.float32)
    do = tl.load(do_ptr, mask=mask_m, other=0.0).to(tl.float32)

    delta = tl.sum(o * do, axis=1)
    delta_ptr = Delta + off_z * H * N_CTX + off_h * N_CTX + off_m
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
    mask_ptr,
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

    key_mask = tl.load(mask_ptr + offs_n, mask=offs_n < N_CTX, other=False)

    for _start_m in range(0, N_CTX, BLOCK_M):
        offs_m = _start_m + tl.arange(0, BLOCK_M)
        qT_mask = (offs_m[None, :] < N_CTX) & (offs_k[:, None] < HEAD_DIM)
        do_mask = (offs_m[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM)
        qT = tl.load(qT_ptrs, mask=qT_mask, other=0.0)

        bias_mask = (offs_m[None, :] < N_CTX) & (offs_n[:, None] < N_CTX)
        biasT = tl.load(biasT_ptrs, mask=bias_mask, other=float("-inf"))
        biasT = tl.where(key_mask[:, None], biasT, float("-inf"))
        do = tl.load(do_ptrs, mask=do_mask, other=0.0)
        m = tl.load(m_ptrs, mask=offs_m < N_CTX, other=0.0)
        Di = tl.load(d_ptrs, mask=offs_m < N_CTX, other=0.0)

        qkT = tl.dot(k, qT) + biasT / (qk_scale / 1.44269504)
        m_safe = tl.maximum(m, -1e38)
        qkT = qkT * qk_scale - m_safe[None, :]

        pT = tl.math.exp2(qkT)
        dv = tl.dot(pT.to(do.dtype), do, dv)

        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - Di[None, :])

        tl.store(dbiasT_ptrs, dsT, mask=bias_mask)

        dsT = dsT.to(qT.dtype)
        dk = tl.dot(dsT, tl.trans(qT), dk)

        # --- 변경: atomic_add → store (각 split이 자기 슬롯에 직접 write) ---
        dqT_to_add = tl.dot(tl.trans(k), dsT) * (qk_scale / 1.44269504)
        tl.store(dqT_ptrs, dqT_to_add, mask=qT_mask)

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
    reset_to_zero=["DQ", "DBias"],
)
@triton.jit
def _attn_bwd(
    Q,
    K,
    V,
    Bias,
    Mask,
    sm_scale,
    DO,
    DQ,
    DK,
    DV,
    DBias,
    M,
    D,
    stride_a,
    stride_z,
    stride_n,
    stride_h,
    stride_d,
    stride_dq_split,  # NEW: dq_expand의 split 차원 stride
    bias_stride_a,
    bias_stride_z,
    bias_stride_h,
    bias_stride_m,
    bias_stride_n,
    stride_maska,
    stride_maskb,
    A: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_N: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    aid = tl.program_id(1).to(tl.int64)
    bhid = tl.program_id(2).to(tl.int64)
    bid = bhid // H
    hid = bhid % H
    qkv_offset = aid * stride_a + bid * stride_z + hid * stride_h
    M_offset = aid * (B * H * N_CTX) + bhid * N_CTX

    Q += qkv_offset
    K += qkv_offset
    V += qkv_offset
    DO += qkv_offset
    # --- 변경: DQ에 split offset 추가 (pid별 독립 슬롯) ---
    DQ += pid * stride_dq_split + qkv_offset
    DK += qkv_offset
    DV += qkv_offset
    M += M_offset
    D += M_offset

    offset_bias = bid * bias_stride_z + hid * bias_stride_h
    Bias += offset_bias
    DBias += aid * bias_stride_a + offset_bias

    mask_offset = aid * stride_maska + bid * stride_maskb
    mask_ptr = Mask + mask_offset

    offs_k = tl.arange(0, BLOCK_D)
    start_n = pid * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)

    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    k_ptr = K + offs_n[:, None] * stride_n + offs_k[None, :] * stride_d
    v_ptr = V + offs_n[:, None] * stride_n + offs_k[None, :] * stride_d

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

    qk_scale = sm_scale * 1.44269504
    dk, dv = _attn_bwd_dqdkdv(
        dk,
        dv,
        DBias,
        Q,
        k,
        v,
        Bias,
        mask_ptr,
        qk_scale,
        DO,
        DQ,
        M,
        D,
        stride_n,
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

    dv_ptrs = DV + offs_n[:, None] * stride_n + offs_k[None, :] * stride_d
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_n + offs_k[None, :] * stride_d

    tl.store(
        dv_ptrs,
        dv.to(dv_ptrs.dtype.element_ty),
        mask=(offs_n[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM),
    )
    tl.store(
        dk_ptrs,
        dk.to(dk_ptrs.dtype.element_ty),
        mask=(offs_n[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM),
    )


# ── dq_expand reduction 커널 ──────────────────────────────────────────────
@triton.jit
def _dq_reduce(
    DQ_Expand,  # (num_splits, A, B, L, H, D)
    DQ_Out,  # (A, B, L, H, D)
    num_splits,
    stride_split,
    stride_inner,  # = A*B*L*H*D 방향이 아니라 element 단위 stride
    N_ELEM,  # 한 split 내 총 element 수 (= A*B*L*H*D)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_ELEM

    stride_split = stride_split.to(tl.int64)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for s in range(num_splits):
        ptr = DQ_Expand + s * stride_split + offs
        val = tl.load(ptr, mask=mask, other=0.0)
        acc += val

    out_ptr = DQ_Out + offs
    tl.store(out_ptr, acc, mask=mask)


class TritonAugmentedAttentionFunction(torch.autograd.Function):
    @typecheck
    @staticmethod
    @torch.compiler.disable()
    def forward(
        ctx,
        q: Float[torch.Tensor, "A B L H D"],
        k: Float[torch.Tensor, "A B L H D"],
        v: Float[torch.Tensor, "A B L H D"],
        bias: Float[torch.Tensor, "B L L H"],
        mask: Bool[torch.Tensor, "A B L"] | None = None,
    ) -> Float[torch.Tensor, "A B L H D"]:
        A, B, L, H, D = q.shape
        if D > 64:
            msg = f"Only support HEAD_DIM <= 64, but got {D}."
            raise ValueError(msg)

        bias = bias.permute(0, 3, 1, 2)
        q, k, v, bias = [x.contiguous() for x in (q, k, v, bias)]

        if mask is None:
            mask = torch.ones(A, B, L, dtype=torch.bool, device=q.device)
        mask = mask.contiguous()
        q, k, v = [x.view(A * B, L, H, D) for x in (q, k, v)]

        sm_scale = D**-0.5
        out = torch.empty_like(q)
        m = torch.empty(A, B, H, L, device=q.device, dtype=torch.float32)

        grid = lambda META: (
            triton.cdiv(L, META["BLOCK_M"]),
            A * B * H,
            triton.cdiv(D, META["BLOCK_D"]),
        )
        _attn_fwd[grid](
            q,
            k,
            v,
            bias,
            mask,
            sm_scale,
            m,
            out,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *out.stride(),
            *bias.stride(),
            *mask.stride()[:2],
            A,
            B,
            H,
            L,
            D,
            BLOCK_D=triton.next_power_of_2(D),
            GROUP_N=get_seq_group(L),
        )

        q, k, v, out = [x.view(A, B, L, H, D) for x in (q, k, v, out)]

        ctx.save_for_backward(q, k, v, bias, mask, out, m)
        return out

    @staticmethod
    @torch.compiler.disable()
    def backward(ctx, *grad_output: torch.Tensor):
        (dy,) = grad_output
        q, k, v, bias, mask, o, m = ctx.saved_tensors

        if dy.dtype != q.dtype:
            dy = dy.to(q.dtype)

        dy = dy.contiguous()

        A, B, L, H, D = q.shape
        sm_scale = D**-0.5
        delta = torch.empty_like(m)

        grid = lambda META: (triton.cdiv(L, META["BLOCK_M"]), A * B, H)
        _attn_bwd_preprocess[grid](
            o,
            dy,
            delta,
            A,
            B,
            L,
            q.stride(1),
            q.stride(2),
            q.stride(3),
            q.stride(4),
            H,
            D,
            BLOCK_D=triton.next_power_of_2(D),
            GROUP_N=get_seq_group(L),
        )

        # ── 변경: dq_expand 할당 (num_splits, A, B, L, H, D) ──
        num_splits = triton.cdiv(L, _BWD_MIN_BLOCK_N)
        dq_expand = torch.zeros(
            int(num_splits), A, B, L, H, D, device=q.device, dtype=torch.float32
        )

        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dbias = torch.zeros(A, B, H, L, L, device=q.device, dtype=torch.float32)

        grid = lambda META: (
            triton.cdiv(L, META["BLOCK_N"]),
            A,
            B * H,
        )
        _attn_bwd[grid](
            q,
            k,
            v,
            bias,
            mask,
            sm_scale,
            dy,
            dq_expand,  # DQ → dq_expand 전달
            dk,
            dv,
            dbias,
            m,
            delta,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            q.stride(4),
            dq_expand.stride(0),  # stride_dq_split
            dbias.stride(0),
            dbias.stride(1),
            dbias.stride(2),
            dbias.stride(3),
            dbias.stride(4),
            *mask.stride()[:2],
            A,
            B,
            H,
            L,
            D,
            BLOCK_D=triton.next_power_of_2(D),
            GROUP_N=get_seq_group(L),
        )

        # ── 변경: split들을 합산하여 최종 dq 생성 ──
        dq = torch.empty(A, B, L, H, D, device=q.device, dtype=torch.float32)
        n_elem = A * B * L * H * D
        BLOCK_SIZE = 1024
        grid_reduce = (triton.cdiv(n_elem, BLOCK_SIZE),)
        _dq_reduce[grid_reduce](
            dq_expand,
            dq,
            int(num_splits),
            dq_expand.stride(0),
            1,  # element stride (contiguous)
            n_elem,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        dbias = dbias.sum(dim=0)
        dbias = dbias.permute(0, 2, 3, 1).contiguous()

        return dq, dk, dv, dbias, None


triton_augmented_attention_pair_bias = TritonAugmentedAttentionFunction.apply
