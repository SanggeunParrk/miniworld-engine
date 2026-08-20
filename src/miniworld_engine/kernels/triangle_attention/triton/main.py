
from miniworld_engine.kernels._compile import opaque
# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/triangle_attention_pair_bias.py

from miniworld_engine.autotune.configs import configs_for
import torch
import triton
import triton.language as tl

from einops import rearrange, reduce, repeat
from jaxtyping import Float

from miniworld_engine._typecheck import typecheck
from miniworld_engine.autotune import key_bucket_of, tensor_dtype_of


def get_seq_group(length) -> int:
    """Delegates to canonical size-bucketing (autotune.buckets)."""
    from miniworld_engine.autotune.buckets import bucket_squared
    return bucket_squared(length * length)



# HEAD_DIM_PAD was a launch constant (next_power_of_2(HEAD_DIM)). delta = sum_d(o*do) is a plain
# reduction over d, so it tiles with an accumulating loop and HEAD_DIM_PAD joins the sweep.





@triton.autotune(configs=configs_for("triangle_attention_fwd_triton"), key=['shape_key', 'H', 'HEAD_DIM'])
@triton.jit
def _attn_fwd(
    q_ptr, k_ptr, v_ptr, bias_ptr, sm_scale, m_ptr, out_ptr,
    qs_z, qs_h, qs_lrow, qs_tok, qs_d,
    os_z, os_h, os_lrow, os_tok, os_d,
    bs_z, bs_h, bs_m, bs_n,
    ms_z, ms_h, ms_lrow, ms_tok,
    Z, H: tl.constexpr, N_CTX, HEAD_DIM: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_M2: tl.constexpr, HEAD_DIM_PAD: tl.constexpr, shape_key,
):
    tl.static_assert(HEAD_DIM_PAD >= HEAD_DIM)
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
                             offsets=(start_m * BLOCK_M1, 0), block_shape=(BLOCK_M1, HEAD_DIM_PAD), order=(1, 0))
    K_bp = tl.make_block_ptr(base=k_base, shape=(N_CTX, HEAD_DIM), strides=(qs_tok, qs_d),
                             offsets=(0, 0), block_shape=(BLOCK_M2, HEAD_DIM_PAD), order=(1, 0))
    V_bp = tl.make_block_ptr(base=v_base, shape=(N_CTX, HEAD_DIM), strides=(qs_tok, qs_d),
                             offsets=(0, 0), block_shape=(BLOCK_M2, HEAD_DIM_PAD), order=(1, 0))
    B_bp = tl.make_block_ptr(base=b_base, shape=(N_CTX, N_CTX), strides=(bs_m, bs_n),
                             offsets=(start_m * BLOCK_M1, 0), block_shape=(BLOCK_M1, BLOCK_M2), order=(1, 0))
    O_bp = tl.make_block_ptr(base=o_base, shape=(N_CTX, HEAD_DIM), strides=(os_tok, os_d),
                             offsets=(start_m * BLOCK_M1, 0), block_shape=(BLOCK_M1, HEAD_DIM_PAD), order=(1, 0))

    offs_m = start_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    m_i = tl.zeros([BLOCK_M1], tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M1], tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M1, HEAD_DIM_PAD], tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_bp, boundary_check=(0, 1), padding_option="zero")

    for start_n in range(0, N_CTX, BLOCK_M2):
        offs_n = start_n + tl.arange(0, BLOCK_M2)
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
        K_bp = tl.advance(K_bp, (BLOCK_M2, 0))
        V_bp = tl.advance(V_bp, (BLOCK_M2, 0))
        B_bp = tl.advance(B_bp, (0, BLOCK_M2))

    l_i = tl.where(l_i > 0.0, l_i, 1.0)  # guard fully-masked rows (l_i=0): finite 0 output, no 0/0 NaN
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_base = m_ptr + off_z * ms_z + off_h * ms_h + off_t * ms_lrow
    tl.store(m_base + offs_m * ms_tok, m_i, mask=offs_m < N_CTX)
    tl.store(O_bp, acc.to(out_ptr.type.element_ty), boundary_check=(0, 1))




# These three take H*L already, in the `HL` parameter the body uses. A second parameter named `H`
# was receiving the SAME H*L value and being keyed on it: token-derived, so it partitioned the cache
# per sequence length and made the shape_key bucket beside it redundant -- for a duplicate the body
# never read. Both the parameter and the key entry are gone.
@triton.autotune(configs=configs_for("triangle_attention_bwd_pre_triton"),
                 key=['shape_key', 'HEAD_DIM'])
@triton.jit
def _attn_bwd_preprocess(
    o, DO, Delta,
    os_z, os_h, os_lrow, os_tok, os_d,
    ds_z, ds_h, ds_lrow, ds_tok, ds_d,
    HL, Z, N_CTX, HEAD_DIM: tl.constexpr,
    BLOCK_M1: tl.constexpr, HEAD_DIM_PAD: tl.constexpr, shape_key,
):
    # B+C: both `o` (fwd out) and `DO` (grad_output) are STRIDED 5D (B,H,Lrow,tok,D) views
    # over projection-layout [B,L,L2,H*D] -> read via explicit strides, no .contiguous().
    off_m = tl.program_id(0) * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    off_hz = tl.program_id(1).to(tl.int64)
    b = off_hz // HL
    hl = off_hz % HL
    h = hl // N_CTX
    i_row = hl % N_CTX
    o_base = o + b * os_z + h * os_h + i_row * os_lrow
    do_base = DO + b * ds_z + h * ds_h + i_row * ds_lrow
    # delta[m] = sum_d o[m,d]*do[m,d]: a reduction over d, so HEAD_DIM_PAD tiles it and the fp32
    # partials accumulate (summation is associative -> exact for any tile size).
    delta = tl.zeros([BLOCK_M1], dtype=tl.float32)
    for d0 in range(0, HEAD_DIM, HEAD_DIM_PAD):
        off_n = d0 + tl.arange(0, HEAD_DIM_PAD)
        o_ptr = o_base + off_m[:, None] * os_tok + off_n[None, :] * os_d
        do_ptr = do_base + off_m[:, None] * ds_tok + off_n[None, :] * ds_d
        mask_m = (off_m[:, None] < N_CTX) & (off_n[None, :] < HEAD_DIM)
        oo = tl.load(o_ptr, mask=mask_m, other=0.0).to(tl.float32)
        do = tl.load(do_ptr, mask=mask_m, other=0.0).to(tl.float32)
        delta += tl.sum(oo * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + off_m, delta, mask=off_m < N_CTX)


def _next_pow2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length() if x > 1 else 1


# Static-shared-memory estimate for the dkdv/dq backward configs, used by the device-smem
# prune to drop unlaunchable configs BEFORE the autotuner tries (and dies on) them. The
# kernel's smem is dominated by the num_stages-pipelined [BLOCK_M1 x HEAD_DIM_PAD] q/do loads plus
# the [BLOCK_N x BLOCK_M1] biasT and the persistent [BLOCK_N x HEAD_DIM_PAD] k/v, with HEAD_DIM_PAD >=
# HEAD_DIM. The 2.8x factor calibrates the raw tile bytes to Triton's actual allocation
# (measured on sm_86: 147456 B at num_stages=3, HEAD_DIM=64, BLOCK_M1=32, BLOCK_N=64). It only
# needs to RANK configs by size well enough to drop the over-limit ones; make_device_smem_prune
# always keeps the smallest as a last-resort candidate, so an imperfect estimate never empties
# the grid.
_SMEM_FUDGE = 2.8


@triton.autotune(configs=configs_for("triangle_attention_bwd_dkdv_triton"),
                 key=['shape_key', 'HEAD_DIM'])
@triton.jit
def _attn_bwd_dkdv(
    q_ptr, k_ptr, v_ptr, bias_ptr, sm_scale, do_ptr, dk_ptr, dv_ptr, dbias_ptr, m_ptr, d_ptr,
    qs_z, qs_h, qs_lrow, qs_tok, qs_d,
    gs_z, gs_h, gs_lrow, gs_tok, gs_d,
    ds_z, ds_h, ds_lrow, ds_tok, ds_d,
    bs_z, bs_h, bs_m, bs_n,
    dbias_hl,
    N_CTX, HL, HEAD_DIM: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_M2: tl.constexpr, HEAD_DIM_PAD: tl.constexpr, shape_key,
):
    # v4b: minimize tl.trans (was 4/iter -> 2/iter) by working in [key, query] space like the
    # raw-ptr v2 kernel, but with make_block_ptr strided loads. qT & biasT loaded transposed.
    tl.static_assert(HEAD_DIM_PAD >= HEAD_DIM)
    bhid = tl.program_id(2).to(tl.int64)
    off_chz = bhid * N_CTX
    b = bhid // HL
    hl = bhid % HL
    h = hl // N_CTX
    i_row = hl % N_CTX
    adj_q = b * qs_z + h * qs_h + i_row * qs_lrow
    adj_g = b * gs_z + h * gs_h + i_row * gs_lrow   # grad-group layout (may differ from q)
    adj_do = b * ds_z + h * ds_h + i_row * ds_lrow
    pid = tl.program_id(0)
    q_base = q_ptr + adj_q
    k_base = k_ptr + adj_q
    v_base = v_ptr + adj_q
    do_base = do_ptr + adj_do
    dk_base = dk_ptr + adj_g   # dq/dk/dv written in grad-group layout (decoupled from q strides)
    dv_base = dv_ptr + adj_g
    b_base = bias_ptr + b * bs_z + h * bs_h
    dbias_base = dbias_ptr + bhid * dbias_hl
    m_ptr += off_chz
    d_ptr += off_chz
    start_n = pid * BLOCK_M2

    K_bp = tl.make_block_ptr(base=k_base, shape=(N_CTX, HEAD_DIM), strides=(qs_tok, qs_d),
                             offsets=(start_n, 0), block_shape=(BLOCK_M2, HEAD_DIM_PAD), order=(1, 0))
    V_bp = tl.make_block_ptr(base=v_base, shape=(N_CTX, HEAD_DIM), strides=(qs_tok, qs_d),
                             offsets=(start_n, 0), block_shape=(BLOCK_M2, HEAD_DIM_PAD), order=(1, 0))
    # qT transposed [D, BLOCK_M1]
    QT_bp = tl.make_block_ptr(base=q_base, shape=(HEAD_DIM, N_CTX), strides=(qs_d, qs_tok),
                              offsets=(0, 0), block_shape=(HEAD_DIM_PAD, BLOCK_M1), order=(0, 1))
    DO_bp = tl.make_block_ptr(base=do_base, shape=(N_CTX, HEAD_DIM), strides=(ds_tok, ds_d),
                              offsets=(0, 0), block_shape=(BLOCK_M1, HEAD_DIM_PAD), order=(1, 0))
    # biasT transposed [BLOCK_M2 key, BLOCK_M1 query]: bias mem is [query, key] (bs_m, bs_n)
    BT_bp = tl.make_block_ptr(base=b_base, shape=(N_CTX, N_CTX), strides=(bs_n, bs_m),
                              offsets=(start_n, 0), block_shape=(BLOCK_M2, BLOCK_M1), order=(0, 1))
    # dbiasT [BLOCK_M2 key, BLOCK_M1 query] view into dbias[query,key] (strides swapped)
    DBT_bp = tl.make_block_ptr(base=dbias_base, shape=(N_CTX, N_CTX), strides=(bs_n, bs_m),
                               offsets=(start_n, 0), block_shape=(BLOCK_M2, BLOCK_M1), order=(0, 1))
    DK_bp = tl.make_block_ptr(base=dk_base, shape=(N_CTX, HEAD_DIM), strides=(gs_tok, gs_d),
                              offsets=(start_n, 0), block_shape=(BLOCK_M2, HEAD_DIM_PAD), order=(1, 0))
    DV_bp = tl.make_block_ptr(base=dv_base, shape=(N_CTX, HEAD_DIM), strides=(gs_tok, gs_d),
                              offsets=(start_n, 0), block_shape=(BLOCK_M2, HEAD_DIM_PAD), order=(1, 0))

    offs_n = start_n + tl.arange(0, BLOCK_M2)
    dv = tl.zeros([BLOCK_M2, HEAD_DIM_PAD], tl.float32)
    dk = tl.zeros([BLOCK_M2, HEAD_DIM_PAD], tl.float32)
    k = tl.load(K_bp, boundary_check=(0, 1), padding_option="zero")   # [BN, D]
    v = tl.load(V_bp, boundary_check=(0, 1), padding_option="zero")   # [BN, D]
    qk_scale = sm_scale * 1.44269504

    m_ptrs = m_ptr + tl.arange(0, BLOCK_M1)
    d_ptrs = d_ptr + tl.arange(0, BLOCK_M1)
    for start_m in range(0, N_CTX, BLOCK_M1):
        offs_m = start_m + tl.arange(0, BLOCK_M1)
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

        QT_bp = tl.advance(QT_bp, (0, BLOCK_M1))
        DO_bp = tl.advance(DO_bp, (BLOCK_M1, 0))
        BT_bp = tl.advance(BT_bp, (0, BLOCK_M1))
        DBT_bp = tl.advance(DBT_bp, (0, BLOCK_M1))
        m_ptrs += BLOCK_M1
        d_ptrs += BLOCK_M1

    dk = dk * sm_scale
    tl.store(DV_bp, dv.to(dv_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(DK_bp, dk.to(dk_ptr.dtype.element_ty), boundary_check=(0, 1))




@triton.autotune(configs=configs_for("triangle_attention_bwd_dq_triton"),
                 key=['shape_key', 'HEAD_DIM'])
@triton.jit
def _attn_bwd_dq(
    q_ptr, k_ptr, v_ptr, bias_ptr, sm_scale, do_ptr, dq_ptr, m_ptr, d_ptr,
    qs_z, qs_h, qs_lrow, qs_tok, qs_d,
    gs_z, gs_h, gs_lrow, gs_tok, gs_d,
    ds_z, ds_h, ds_lrow, ds_tok, ds_d,
    bs_z, bs_h, bs_m, bs_n,
    N_CTX, HL, HEAD_DIM: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_M2: tl.constexpr, HEAD_DIM_PAD: tl.constexpr, shape_key,
):
    tl.static_assert(HEAD_DIM_PAD >= HEAD_DIM)
    bhid = tl.program_id(2).to(tl.int64)
    off_chz = bhid * N_CTX
    b = bhid // HL
    hl = bhid % HL
    h = hl // N_CTX
    i_row = hl % N_CTX
    adj_q = b * qs_z + h * qs_h + i_row * qs_lrow
    adj_g = b * gs_z + h * gs_h + i_row * gs_lrow   # grad-group layout (may differ from q)
    adj_do = b * ds_z + h * ds_h + i_row * ds_lrow
    pid = tl.program_id(0)
    q_base = q_ptr + adj_q
    k_base = k_ptr + adj_q
    v_base = v_ptr + adj_q
    do_base = do_ptr + adj_do
    dq_base = dq_ptr + adj_g   # dq written in grad-group layout (decoupled from q strides)
    b_base = bias_ptr + b * bs_z + h * bs_h
    m_ptr += off_chz
    d_ptr += off_chz
    start_m = pid * BLOCK_M1

    Q_bp = tl.make_block_ptr(base=q_base, shape=(N_CTX, HEAD_DIM), strides=(qs_tok, qs_d),
                             offsets=(start_m, 0), block_shape=(BLOCK_M1, HEAD_DIM_PAD), order=(1, 0))
    DO_bp = tl.make_block_ptr(base=do_base, shape=(N_CTX, HEAD_DIM), strides=(ds_tok, ds_d),
                              offsets=(start_m, 0), block_shape=(BLOCK_M1, HEAD_DIM_PAD), order=(1, 0))
    DQ_bp = tl.make_block_ptr(base=dq_base, shape=(N_CTX, HEAD_DIM), strides=(gs_tok, gs_d),
                              offsets=(start_m, 0), block_shape=(BLOCK_M1, HEAD_DIM_PAD), order=(1, 0))
    K_bp = tl.make_block_ptr(base=k_base, shape=(N_CTX, HEAD_DIM), strides=(qs_tok, qs_d),
                             offsets=(0, 0), block_shape=(BLOCK_M2, HEAD_DIM_PAD), order=(1, 0))
    V_bp = tl.make_block_ptr(base=v_base, shape=(N_CTX, HEAD_DIM), strides=(qs_tok, qs_d),
                             offsets=(0, 0), block_shape=(BLOCK_M2, HEAD_DIM_PAD), order=(1, 0))
    B_bp = tl.make_block_ptr(base=b_base, shape=(N_CTX, N_CTX), strides=(bs_m, bs_n),
                             offsets=(start_m, 0), block_shape=(BLOCK_M1, BLOCK_M2), order=(1, 0))

    offs_m = start_m + tl.arange(0, BLOCK_M1)
    q = tl.load(Q_bp, boundary_check=(0, 1), padding_option="zero")
    do = tl.load(DO_bp, boundary_check=(0, 1), padding_option="zero")
    mq = tl.load(m_ptr + offs_m, mask=offs_m < N_CTX, other=0.0)
    Di = tl.load(d_ptr + offs_m, mask=offs_m < N_CTX, other=0.0)
    m_safe = tl.maximum(mq, -1e38)
    dq = tl.zeros([BLOCK_M1, HEAD_DIM_PAD], tl.float32)
    qk_scale = sm_scale * 1.44269504

    for start_n in range(0, N_CTX, BLOCK_M2):
        offs_n = start_n + tl.arange(0, BLOCK_M2)
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
        K_bp = tl.advance(K_bp, (BLOCK_M2, 0))
        V_bp = tl.advance(V_bp, (BLOCK_M2, 0))
        B_bp = tl.advance(B_bp, (0, BLOCK_M2))

    dq = dq * sm_scale
    tl.store(DQ_bp, dq.to(dq_ptr.dtype.element_ty), boundary_check=(0, 1))


class TritonTriangleAttentionPairBiasFunction(torch.autograd.Function):
    @typecheck
    @staticmethod
    @opaque()
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

        grid = lambda META: [triton.cdiv(L, META["BLOCK_M1"]), B * H, L]
        _attn_fwd[grid](
            q, k, v, bias, sm_scale, m, out,
            *q.stride(),      # q strided 5D (B,H,Lrow,tok,D)  [k,v share pattern]
            *out.stride(),    # o-group STRIDED 5D (projection layout, head_dim stride-1)
            *bias.stride(),   # bias contiguous (B,H,m,n)
            *m.stride(),
            B, H, L, D, HEAD_DIM_PAD=triton.next_power_of_2(D), shape_key=get_seq_group(L),
        )
        ctx.save_for_backward(q, k, v, bias, m, out)
        return out

    @staticmethod
    @opaque()
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

        grid = lambda META: [triton.cdiv(L, META["BLOCK_M1"]), B * HL, 1]
        _attn_bwd_preprocess[grid](
            out, grad_output, delta,
            *out.stride(),           # out STRIDED 5D (projection layout)
            *grad_output.stride(), HL, B, L, D,
            shape_key=get_seq_group(L),
            HEAD_DIM_PAD=triton.next_power_of_2(D),
        )

        # A: allocate dq/dk/dv in the PROJECTION layout [B,L,L2,H*D] and hand the kernels
        # strided (B,H,L,L2,D) views of them. The kernels write strided (q-group), so the
        # module's rearrange-backward becomes a FREE view -> no grad-transpose copy.
        # dq stored bf16 (was fp32 — a vestige of the original atomic-add dq, which needed an
        # fp32 accumulator buffer; FA2 split removed the atomics). The kernel accumulates dq in
        # fp32 registers and stores once, exactly like dk/dv, so bf16 out is precision-neutral
        # (matches the cuBLAS/pytorch ref which also stores bf16 grads). Kills the fp32->bf16
        # cast copy of the returned query grad.
        dq_proj = torch.empty(B, L, L, H * D, device=q.device, dtype=v.dtype)
        dk_proj = torch.empty(B, L, L, H * D, device=q.device, dtype=v.dtype)
        dv_proj = torch.empty(B, L, L, H * D, device=q.device, dtype=v.dtype)
        dq = rearrange(dq_proj, "B L L2 (H D) -> B H L L2 D", H=H)
        dk = rearrange(dk_proj, "B L L2 (H D) -> B H L L2 D", H=H)
        dv = rearrange(dv_proj, "B L L2 (H D) -> B H L L2 D", H=H)
        dbias = torch.empty(B, HL, L, L, device=q.device, dtype=bias.dtype)  # per-row, reduced below

        grid_kv = lambda META: [triton.cdiv(L, META["BLOCK_M2"]), 1, B * HL]
        _attn_bwd_dkdv[grid_kv](
            q, k, v, bias, sm_scale, grad_output, dk, dv, dbias, m_m, delta,
            *q.stride(),        # q strided 5D  [k,v share] — READ layout
            *dk.stride(),       # grad-group WRITE layout (may differ from q, e.g. concat split-view)
            *grad_output.stride(),  # do STRIDED 5D (B,H,Lrow,tok,D)
            *bias.stride(),     # bias contiguous (B,H,m,n)  broadcast over row
            L * L,              # dbias HL-dim stride
            L, HL, D,
            HEAD_DIM_PAD=triton.next_power_of_2(D), shape_key=get_seq_group(L),
        )
        grid_q = lambda META: [triton.cdiv(L, META["BLOCK_M1"]), 1, B * HL]
        _attn_bwd_dq[grid_q](
            q, k, v, bias, sm_scale, grad_output, dq, m_m, delta,
            *q.stride(),
            *dq.stride(),       # grad-group WRITE layout
            *grad_output.stride(),
            *bias.stride(),
            L, HL, D,
            HEAD_DIM_PAD=triton.next_power_of_2(D), shape_key=get_seq_group(L),
        )

        # dq/dk/dv are already (B,H,L,L2,D) strided views over [B,L,L2,H*D] -> returned as-is;
        # the module's rearrange-backward is a free view (data already in projection layout).
        dbias = reduce(dbias, "B (H L3) L L2 -> B H L L2", "sum", L3=L)
        return dq, dk, dv, dbias


triton_triangle_attention_pair_bias = TritonTriangleAttentionPairBiasFunction.apply
