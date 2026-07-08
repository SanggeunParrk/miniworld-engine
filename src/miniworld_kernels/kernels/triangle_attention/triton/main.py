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


@triton.autotune(configs=fwd_configs, key=["GROUP_N", "H", "HEAD_DIM"])
@triton.jit
def _attn_fwd(
    q_ptr, k_ptr, v_ptr, bias_ptr, sm_scale, m_ptr, out_ptr,
    qs_z, qs_h, qs_lrow, qs_tok, qs_d,
    os_z, os_h, os_lrow, os_tok, os_d,
    bs_z, bs_h, bs_m, bs_n,
    ms_z, ms_h, ms_lrow, ms_tok,
    Z, H: tl.constexpr, N_CTX, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, GROUP_N: tl.constexpr,
):
    tl.static_assert(BLOCK_D >= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1).to(tl.int64)
    off_t = tl.program_id(2).to(tl.int64)
    off_z = off_hz // H
    off_h = off_hz % H
    q_base = q_ptr + off_z * qs_z + off_h * qs_h + off_t * qs_lrow
    k_base = k_ptr + off_z * qs_z + off_h * qs_h + off_t * qs_lrow
    v_base = v_ptr + off_z * qs_z + off_h * qs_h + off_t * qs_lrow
    o_base = out_ptr + off_z * os_z + off_h * os_h + off_t * os_lrow
    b_base = bias_ptr + off_z * bs_z + off_h * bs_h

    Q_bp = tl.make_block_ptr(base=q_base, shape=(N_CTX, HEAD_DIM), strides=(qs_tok, qs_d),
                             offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_D), order=(1, 0))
    K_bp = tl.make_block_ptr(base=k_base, shape=(N_CTX, HEAD_DIM), strides=(qs_tok, qs_d),
                             offsets=(0, 0), block_shape=(BLOCK_N, BLOCK_D), order=(1, 0))
    V_bp = tl.make_block_ptr(base=v_base, shape=(N_CTX, HEAD_DIM), strides=(qs_tok, qs_d),
                             offsets=(0, 0), block_shape=(BLOCK_N, BLOCK_D), order=(1, 0))
    B_bp = tl.make_block_ptr(base=b_base, shape=(N_CTX, N_CTX), strides=(bs_m, bs_n),
                             offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    O_bp = tl.make_block_ptr(base=o_base, shape=(N_CTX, HEAD_DIM), strides=(os_tok, os_d),
                             offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_D), order=(1, 0))

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_i = tl.zeros([BLOCK_M], tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_bp, boundary_check=(0, 1), padding_option="zero")

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K_bp, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_bp, boundary_check=(0, 1), padding_option="zero")
        bias = tl.load(B_bp, boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(q, tl.trans(k))
        qk = qk + bias / (qk_scale / 1.44269504)
        qk = tl.where(offs_n[None, :] < N_CTX, qk, -1e38)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        m_ij = tl.maximum(m_ij, -1e38)
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
        K_bp = tl.advance(K_bp, (BLOCK_N, 0))
        V_bp = tl.advance(V_bp, (BLOCK_N, 0))
        B_bp = tl.advance(B_bp, (0, BLOCK_N))

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_base = m_ptr + off_z * ms_z + off_h * ms_h + off_t * ms_lrow
    tl.store(m_base + offs_m * ms_tok, m_i, mask=offs_m < N_CTX)
    tl.store(O_bp, acc.to(out_ptr.type.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=bwd_preprocess_configs, key=["GROUP_N", "H", "HEAD_DIM"])
@triton.jit
def _attn_bwd_preprocess(
    o, DO, Delta,
    os_z, os_h, os_lrow, os_tok, os_d,
    ds_z, ds_h, ds_lrow, ds_tok, ds_d,
    HL, Z, H: tl.constexpr, N_CTX, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr, GROUP_N: tl.constexpr,
):
    # B+C: both `o` (fwd out) and `DO` (grad_output) are STRIDED 5D (B,H,Lrow,tok,D) views
    # over projection-layout [B,L,L2,H*D] -> read via explicit strides, no .contiguous().
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1).to(tl.int64)
    off_n = tl.arange(0, BLOCK_D)
    b = off_hz // HL
    hl = off_hz % HL
    h = hl // N_CTX
    i_row = hl % N_CTX
    o_base = o + b * os_z + h * os_h + i_row * os_lrow
    o_ptr = o_base + off_m[:, None] * os_tok + off_n[None, :] * os_d
    do_base = DO + b * ds_z + h * ds_h + i_row * ds_lrow
    do_ptr = do_base + off_m[:, None] * ds_tok + off_n[None, :] * ds_d
    mask_m = (off_m[:, None] < N_CTX) & (off_n[None, :] < HEAD_DIM)
    oo = tl.load(o_ptr, mask=mask_m, other=0.0).to(tl.float32)
    do = tl.load(do_ptr, mask=mask_m, other=0.0).to(tl.float32)
    delta = tl.sum(oo * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + off_m, delta, mask=off_m < N_CTX)


@triton.autotune(configs=bwd_configs, key=["GROUP_N", "H", "HEAD_DIM"])
@triton.jit
def _attn_bwd_dkdv(
    q_ptr, k_ptr, v_ptr, bias_ptr, sm_scale, do_ptr, dk_ptr, dv_ptr, dbias_ptr, m_ptr, d_ptr,
    qs_z, qs_h, qs_lrow, qs_tok, qs_d,
    ds_z, ds_h, ds_lrow, ds_tok, ds_d,
    bs_z, bs_h, bs_m, bs_n,
    dbias_hl,
    H: tl.constexpr, N_CTX, HL, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, GROUP_N: tl.constexpr,
):
    # v4b: minimize tl.trans (was 4/iter -> 2/iter) by working in [key, query] space like the
    # raw-ptr v2 kernel, but with make_block_ptr strided loads. qT & biasT loaded transposed.
    tl.static_assert(BLOCK_D >= HEAD_DIM)
    bhid = tl.program_id(2).to(tl.int64)
    off_chz = bhid * N_CTX
    b = bhid // HL
    hl = bhid % HL
    h = hl // N_CTX
    i_row = hl % N_CTX
    adj_q = b * qs_z + h * qs_h + i_row * qs_lrow
    adj_do = b * ds_z + h * ds_h + i_row * ds_lrow
    pid = tl.program_id(0)
    q_base = q_ptr + adj_q
    k_base = k_ptr + adj_q
    v_base = v_ptr + adj_q
    do_base = do_ptr + adj_do
    dk_base = dk_ptr + adj_q   # A: dq/dk/dv written in projection layout (q-group strided)
    dv_base = dv_ptr + adj_q
    b_base = bias_ptr + b * bs_z + h * bs_h
    dbias_base = dbias_ptr + bhid * dbias_hl
    m_ptr += off_chz
    d_ptr += off_chz
    start_n = pid * BLOCK_N

    K_bp = tl.make_block_ptr(base=k_base, shape=(N_CTX, HEAD_DIM), strides=(qs_tok, qs_d),
                             offsets=(start_n, 0), block_shape=(BLOCK_N, BLOCK_D), order=(1, 0))
    V_bp = tl.make_block_ptr(base=v_base, shape=(N_CTX, HEAD_DIM), strides=(qs_tok, qs_d),
                             offsets=(start_n, 0), block_shape=(BLOCK_N, BLOCK_D), order=(1, 0))
    # qT transposed [D, BLOCK_M]
    QT_bp = tl.make_block_ptr(base=q_base, shape=(HEAD_DIM, N_CTX), strides=(qs_d, qs_tok),
                              offsets=(0, 0), block_shape=(BLOCK_D, BLOCK_M), order=(0, 1))
    DO_bp = tl.make_block_ptr(base=do_base, shape=(N_CTX, HEAD_DIM), strides=(ds_tok, ds_d),
                              offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_D), order=(1, 0))
    # biasT transposed [BLOCK_N key, BLOCK_M query]: bias mem is [query, key] (bs_m, bs_n)
    BT_bp = tl.make_block_ptr(base=b_base, shape=(N_CTX, N_CTX), strides=(bs_n, bs_m),
                              offsets=(start_n, 0), block_shape=(BLOCK_N, BLOCK_M), order=(0, 1))
    # dbiasT [BLOCK_N key, BLOCK_M query] view into dbias[query,key] (strides swapped)
    DBT_bp = tl.make_block_ptr(base=dbias_base, shape=(N_CTX, N_CTX), strides=(bs_n, bs_m),
                               offsets=(start_n, 0), block_shape=(BLOCK_N, BLOCK_M), order=(0, 1))
    DK_bp = tl.make_block_ptr(base=dk_base, shape=(N_CTX, HEAD_DIM), strides=(qs_tok, qs_d),
                              offsets=(start_n, 0), block_shape=(BLOCK_N, BLOCK_D), order=(1, 0))
    DV_bp = tl.make_block_ptr(base=dv_base, shape=(N_CTX, HEAD_DIM), strides=(qs_tok, qs_d),
                              offsets=(start_n, 0), block_shape=(BLOCK_N, BLOCK_D), order=(1, 0))

    offs_n = start_n + tl.arange(0, BLOCK_N)
    dv = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
    k = tl.load(K_bp, boundary_check=(0, 1), padding_option="zero")   # [BN, D]
    v = tl.load(V_bp, boundary_check=(0, 1), padding_option="zero")   # [BN, D]
    qk_scale = sm_scale * 1.44269504

    m_ptrs = m_ptr + tl.arange(0, BLOCK_M)
    d_ptrs = d_ptr + tl.arange(0, BLOCK_M)
    for start_m in range(0, N_CTX, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        qT = tl.load(QT_bp, boundary_check=(0, 1), padding_option="zero")   # [D, BM]
        do = tl.load(DO_bp, boundary_check=(0, 1), padding_option="zero")   # [BM, D]
        biasT = tl.load(BT_bp, boundary_check=(0, 1), padding_option="zero")  # [BN, BM]
        mq = tl.load(m_ptrs, mask=offs_m < N_CTX, other=0.0)
        Di = tl.load(d_ptrs, mask=offs_m < N_CTX, other=0.0)

        qkT = tl.dot(k, qT)                          # [BN key, BM query]
        qkT = qkT + biasT / (qk_scale / 1.44269504)
        m_safe = tl.maximum(mq, -1e38)
        qkT = qkT * qk_scale - m_safe[None, :]
        pT = tl.math.exp2(qkT)
        pT = tl.where((offs_n[:, None] < N_CTX) & (offs_m[None, :] < N_CTX), pT, 0.0)
        dpT = tl.dot(v, tl.trans(do))                # [BN, BM]  (1 trans)
        dsT = pT * (dpT - Di[None, :])               # [BN, BM]
        dv = tl.dot(pT.to(do.dtype), do, dv)         # [BN, D]
        dk = tl.dot(dsT.to(do.dtype), tl.trans(qT), dk)  # [BN, D] = dsT[BN,BM] @ q[BM,D] (1 trans)
        tl.store(DBT_bp, dsT.to(dbias_ptr.dtype.element_ty), boundary_check=(0, 1))

        QT_bp = tl.advance(QT_bp, (0, BLOCK_M))
        DO_bp = tl.advance(DO_bp, (BLOCK_M, 0))
        BT_bp = tl.advance(BT_bp, (0, BLOCK_M))
        DBT_bp = tl.advance(DBT_bp, (0, BLOCK_M))
        m_ptrs += BLOCK_M
        d_ptrs += BLOCK_M

    dk = dk * sm_scale
    tl.store(DV_bp, dv.to(dv_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(DK_bp, dk.to(dk_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=bwd_configs, key=["GROUP_N", "H", "HEAD_DIM"])
@triton.jit
def _attn_bwd_dq(
    q_ptr, k_ptr, v_ptr, bias_ptr, sm_scale, do_ptr, dq_ptr, m_ptr, d_ptr,
    qs_z, qs_h, qs_lrow, qs_tok, qs_d,
    ds_z, ds_h, ds_lrow, ds_tok, ds_d,
    bs_z, bs_h, bs_m, bs_n,
    H: tl.constexpr, N_CTX, HL, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, GROUP_N: tl.constexpr,
):
    tl.static_assert(BLOCK_D >= HEAD_DIM)
    bhid = tl.program_id(2).to(tl.int64)
    off_chz = bhid * N_CTX
    b = bhid // HL
    hl = bhid % HL
    h = hl // N_CTX
    i_row = hl % N_CTX
    adj_q = b * qs_z + h * qs_h + i_row * qs_lrow
    adj_do = b * ds_z + h * ds_h + i_row * ds_lrow
    pid = tl.program_id(0)
    q_base = q_ptr + adj_q
    k_base = k_ptr + adj_q
    v_base = v_ptr + adj_q
    do_base = do_ptr + adj_do
    dq_base = dq_ptr + adj_q   # A: dq written in projection layout (q-group strided)
    b_base = bias_ptr + b * bs_z + h * bs_h
    m_ptr += off_chz
    d_ptr += off_chz
    start_m = pid * BLOCK_M

    Q_bp = tl.make_block_ptr(base=q_base, shape=(N_CTX, HEAD_DIM), strides=(qs_tok, qs_d),
                             offsets=(start_m, 0), block_shape=(BLOCK_M, BLOCK_D), order=(1, 0))
    DO_bp = tl.make_block_ptr(base=do_base, shape=(N_CTX, HEAD_DIM), strides=(ds_tok, ds_d),
                              offsets=(start_m, 0), block_shape=(BLOCK_M, BLOCK_D), order=(1, 0))
    DQ_bp = tl.make_block_ptr(base=dq_base, shape=(N_CTX, HEAD_DIM), strides=(qs_tok, qs_d),
                              offsets=(start_m, 0), block_shape=(BLOCK_M, BLOCK_D), order=(1, 0))
    K_bp = tl.make_block_ptr(base=k_base, shape=(N_CTX, HEAD_DIM), strides=(qs_tok, qs_d),
                             offsets=(0, 0), block_shape=(BLOCK_N, BLOCK_D), order=(1, 0))
    V_bp = tl.make_block_ptr(base=v_base, shape=(N_CTX, HEAD_DIM), strides=(qs_tok, qs_d),
                             offsets=(0, 0), block_shape=(BLOCK_N, BLOCK_D), order=(1, 0))
    B_bp = tl.make_block_ptr(base=b_base, shape=(N_CTX, N_CTX), strides=(bs_m, bs_n),
                             offsets=(start_m, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))

    offs_m = start_m + tl.arange(0, BLOCK_M)
    q = tl.load(Q_bp, boundary_check=(0, 1), padding_option="zero")
    do = tl.load(DO_bp, boundary_check=(0, 1), padding_option="zero")
    mq = tl.load(m_ptr + offs_m, mask=offs_m < N_CTX, other=0.0)
    Di = tl.load(d_ptr + offs_m, mask=offs_m < N_CTX, other=0.0)
    m_safe = tl.maximum(mq, -1e38)
    dq = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    qk_scale = sm_scale * 1.44269504

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K_bp, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_bp, boundary_check=(0, 1), padding_option="zero")
        bias = tl.load(B_bp, boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(q, tl.trans(k))
        qk = qk + bias / (qk_scale / 1.44269504)
        qk = qk * qk_scale - m_safe[:, None]
        p = tl.math.exp2(qk)
        p = tl.where((offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX), p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - Di[:, None])
        dq = tl.dot(ds.to(k.dtype), k, dq)
        K_bp = tl.advance(K_bp, (BLOCK_N, 0))
        V_bp = tl.advance(V_bp, (BLOCK_N, 0))
        B_bp = tl.advance(B_bp, (0, BLOCK_N))

    dq = dq * sm_scale
    tl.store(DQ_bp, dq.to(dq_ptr.dtype.element_ty), boundary_check=(0, 1))


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
        # v4 stride-native: q/k/v stay strided (transpose views) -> no .contiguous() copy.
        # bias IS contiguous (its strided key-dim stride=H is a fully-gappy load = the whole
        # 2.3x penalty; the copy is tiny, ~1MB). out/m are freshly-allocated contiguous.
        bias = bias.contiguous()
        B, H, L, _, D = q.shape
        sm_scale = D**-0.5
        # B: write `out` into PROJECTION layout [B,L,L2,H*D] via a strided (B,H,L,L2,D) view
        # (head_dim stride-1 -> coalesced write, free like q/k/v). The module's rearrange-back
        # "B H L L2 D -> B L L2 (H D)" is then a FREE view -> kills the ~0.2 ms fwd out copy.
        out_proj = torch.empty(B, L, L, H * D, device=q.device, dtype=q.dtype)
        out = rearrange(out_proj, "B L L2 (H D) -> B H L L2 D", H=H)
        m = torch.empty(B, H, L, L, device=q.device, dtype=torch.float32)

        grid = lambda META: [triton.cdiv(L, META["BLOCK_M"]), B * H, L]
        _attn_fwd[grid](
            q, k, v, bias, sm_scale, m, out,
            *q.stride(),      # q strided 5D (B,H,Lrow,tok,D)  [k,v share pattern]
            *out.stride(),    # o-group STRIDED 5D (projection layout, head_dim stride-1)
            *bias.stride(),   # bias contiguous (B,H,m,n)
            *m.stride(),
            B, H, L, D, BLOCK_D=triton.next_power_of_2(D), GROUP_N=get_seq_group(L),
        )
        ctx.save_for_backward(q, k, v, bias, m, out)
        return out

    @staticmethod
    @torch.compiler.disable()
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, bias, m, out = ctx.saved_tensors   # q/k/v strided; bias/m/out contiguous
        if grad_output.dtype != q.dtype:
            grad_output = grad_output.to(q.dtype)

        B, H, L, _, D = q.shape
        HL = H * L
        sm_scale = D**-0.5
        # B+C: consume both out and grad_output STRIDED 5D (no .contiguous); preprocess reads
        # each via its own explicit strides.
        m_m = rearrange(m, "B H L L2 -> B (H L) L2")
        delta = torch.empty_like(m_m)

        grid = lambda META: [triton.cdiv(L, META["BLOCK_M"]), B * HL, 1]
        _attn_bwd_preprocess[grid](
            out, grad_output, delta,
            *out.stride(),           # out STRIDED 5D (projection layout)
            *grad_output.stride(), HL, B, HL, L, D,
            BLOCK_D=triton.next_power_of_2(D), GROUP_N=get_seq_group(L),
        )

        # A: allocate dq/dk/dv in the PROJECTION layout [B,L,L2,H*D] and hand the kernels
        # strided (B,H,L,L2,D) views of them. The kernels write strided (q-group), so the
        # module's rearrange-backward becomes a FREE view -> no grad-transpose copy.
        dq_proj = torch.empty(B, L, L, H * D, device=q.device, dtype=torch.float32)
        dk_proj = torch.empty(B, L, L, H * D, device=q.device, dtype=v.dtype)
        dv_proj = torch.empty(B, L, L, H * D, device=q.device, dtype=v.dtype)
        dq = rearrange(dq_proj, "B L L2 (H D) -> B H L L2 D", H=H)
        dk = rearrange(dk_proj, "B L L2 (H D) -> B H L L2 D", H=H)
        dv = rearrange(dv_proj, "B L L2 (H D) -> B H L L2 D", H=H)
        dbias = torch.empty(B, HL, L, L, device=q.device, dtype=bias.dtype)  # per-row, reduced below

        grid_kv = lambda META: [triton.cdiv(L, META["BLOCK_N"]), 1, B * HL]
        _attn_bwd_dkdv[grid_kv](
            q, k, v, bias, sm_scale, grad_output, dk, dv, dbias, m_m, delta,
            *q.stride(),        # q strided 5D  [k,v share]
            *grad_output.stride(),  # do STRIDED 5D (B,H,Lrow,tok,D)
            *bias.stride(),     # bias contiguous (B,H,m,n)  broadcast over row
            L * L,              # dbias HL-dim stride
            HL, L, HL, D,
            BLOCK_D=triton.next_power_of_2(D), GROUP_N=get_seq_group(L),
        )
        grid_q = lambda META: [triton.cdiv(L, META["BLOCK_M"]), 1, B * HL]
        _attn_bwd_dq[grid_q](
            q, k, v, bias, sm_scale, grad_output, dq, m_m, delta,
            *q.stride(),
            *grad_output.stride(),
            *bias.stride(),
            HL, L, HL, D,
            BLOCK_D=triton.next_power_of_2(D), GROUP_N=get_seq_group(L),
        )

        # dq/dk/dv are already (B,H,L,L2,D) strided views over [B,L,L2,H*D] -> returned as-is;
        # the module's rearrange-backward is a free view (data already in projection layout).
        dbias = reduce(dbias, "B (H L3) L L2 -> B H L L2", "sum", L3=L)
        return dq, dk, dv, dbias


triton_triangle_attention_pair_bias = TritonTriangleAttentionPairBiasFunction.apply
