"""Inference attention core of the fused token DiT: the engine's forward kernel (``augmented_attention/triton/main.py``
``_attn_fwd`` + ``_attn_fwd_inner``), with four changes and the same arithmetic.

1. Grid order. The engine launches (m-block, sample*head, d-split) with the m-block fastest, so the S programs that
   read the same bias tile (same m-block, same head, different sample) are L/BLOCK_M1 * H programs apart in launch
   order -- 192 at L=768 -- and each re-reads the tile from HBM: 94 MB of bias per call instead of 19. Here the
   sample index is the fastest axis, so those S programs run back to back and share the tile through L2.
2. The output gate. ``o * sigmoid(g)`` is applied before the store (``g`` is the fourth column block of the same
   GEMM output, so it has q's strides); the separate gate pass is gone.
3. Head dim 48 as 32 + 16, not padded to 64 (a third of every MMA was zeros).
4. Optional pre-scaled logits: with sm_scale*log2(e) folded into Wq, bq and log2(e) into the pair-bias weights,
   the inner loop drops two multiplies and a division per logit.
5. Written over q, with q, k, v, g all strided views of one [M, 4D] buffer: each program reads its own q tile
   before it writes that tile, and no other program reads it. No logsumexp is stored -- nothing runs a backward.
"""
import triton
import triton.language as tl


def _cfgs():
    return [triton.Config({"BLOCK_M1": m1, "BLOCK_M2": m2}, num_warps=w, num_stages=s)
            for m1 in (32, 64, 128) for m2 in (32, 64, 128) for w in (4, 8) for s in (2, 3, 4)]


# restore_value: the output overwrites q, and autotuning runs the kernel once per config.
@triton.autotune(configs=_cfgs(), key=["N_CTX", "H", "HEAD_DIM", "PREC"], restore_value=["Q"])
@triton.jit
def _attn_fwd_gated(Q, K, V, G, Bias, Mask, sm_scale,
                    stride_qz, stride_qm, stride_qh, stride_qk,
                    stride_bh, stride_bm, stride_bn, stride_mz,
                    H: tl.constexpr, N_CTX, HEAD_DIM: tl.constexpr, HEAD_DIM_PAD: tl.constexpr,
                    D1: tl.constexpr, D2: tl.constexpr, PRESCALED: tl.constexpr, PREC: tl.constexpr,
                    BLOCK_M1: tl.constexpr, BLOCK_M2: tl.constexpr):
    off_z = tl.program_id(0).to(tl.int64)          # sample: fastest, so the S readers of one bias tile are adjacent
    start_m = tl.program_id(1).to(tl.int64)
    off_h = tl.program_id(2).to(tl.int64)
    base = off_z * stride_qz + off_h * stride_qh     # q, k, v, g and the output share one layout
    offset_m = start_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    offset_n = tl.arange(0, BLOCK_M2)
    # head dim as D1 + D2 (48 = 32 + 16): the engine pads it to 64 and spends a third of every MMA on zeros
    k1 = tl.arange(0, D1)
    k2 = D1 + tl.arange(0, D2)
    mrow = offset_m[:, None] < N_CTX
    qrow = Q + base + offset_m[:, None] * stride_qm
    q1 = tl.load(qrow + k1[None, :] * stride_qk, mask=mrow, other=0.0)
    q2 = tl.load(qrow + k2[None, :] * stride_qk, mask=mrow, other=0.0)
    kb = K + base + offset_n[None, :] * stride_qm
    vb = V + base + offset_n[:, None] * stride_qm
    b_ptr = Bias + off_h * stride_bh + offset_m[:, None] * stride_bm + offset_n[None, :] * stride_bn
    mask_ptr = Mask + off_z * stride_mz
    qk_scale = sm_scale * 1.44269504
    m_i = tl.zeros([BLOCK_M1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M1], dtype=tl.float32) + 1.0
    acc1 = tl.zeros([BLOCK_M1, D1], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_M1, D2], dtype=tl.float32)
    for start_n in range(0, N_CTX, BLOCK_M2):
        on = start_n + offset_n
        nk = on[None, :] < N_CTX
        nv = on[:, None] < N_CTX
        bias_val = tl.load(b_ptr, mask=mrow & nk, other=float("-inf"))
        key_mask = tl.load(mask_ptr + on, mask=on < N_CTX, other=False)
        bias_val = tl.where(key_mask[None, :], bias_val, float("-inf"))
        kk1 = tl.load(kb + k1[:, None] * stride_qk, mask=nk, other=0.0)
        kk2 = tl.load(kb + k2[:, None] * stride_qk, mask=nk, other=0.0)
        v1 = tl.load(vb + k1[None, :] * stride_qk, mask=nv, other=0.0)
        v2 = tl.load(vb + k2[None, :] * stride_qk, mask=nv, other=0.0)
        qk = tl.dot(q1, kk1, input_precision=PREC)
        if PRESCALED:
            # q carries sm_scale*log2(e) and the bias carries log2(e), both folded into weights when they were
            # packed: the logits are already in the exp2 domain, no per-element scale or bias division
            qk = tl.dot(q2, kk2, qk, input_precision=PREC) + bias_val
            m_ij = tl.maximum(tl.maximum(m_i, tl.max(qk, 1)), -1e38)
            qk = qk - m_ij[:, None]
        else:
            qk = tl.dot(q2, kk2, qk, input_precision=PREC) + bias_val / (qk_scale / 1.44269504)
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            m_ij = tl.maximum(m_ij, -1e38)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        pb = p.to(v1.dtype)
        acc1 = tl.dot(pb, v1, acc1 * alpha[:, None], input_precision=PREC)
        acc2 = tl.dot(pb, v2, acc2 * alpha[:, None], input_precision=PREC)
        m_i = m_ij
        b_ptr += BLOCK_M2 * stride_bn
        kb += BLOCK_M2 * stride_qm
        vb += BLOCK_M2 * stride_qm
    inv = 1.0 / tl.maximum(l_i, 1e-30)
    g1 = tl.load(G + base + offset_m[:, None] * stride_qm + k1[None, :] * stride_qk, mask=mrow, other=0.0).to(tl.float32)
    g2 = tl.load(G + base + offset_m[:, None] * stride_qm + k2[None, :] * stride_qk, mask=mrow, other=0.0).to(tl.float32)
    tl.store(qrow + k1[None, :] * stride_qk, (acc1 * inv[:, None] * tl.sigmoid(g1)).to(Q.dtype.element_ty), mask=mrow)
    tl.store(qrow + k2[None, :] * stride_qk, (acc2 * inv[:, None] * tl.sigmoid(g2)).to(Q.dtype.element_ty), mask=mrow)


def attention_gated_in_place(q, k, v, g, bias, mask, prescaled=False, precision="tf32"):
    """q, k, v, g: [S, L, H, D] views with identical strides; bias [H, L, L]; mask [S, L] bool. Writes sigmoid(g)*o over q.

    ``precision`` is the MMA precision for fp32 operands ("tf32" -- MiniWorld's fp32 recipe runs TF32 --, "tf32x3" or
    "ieee"); bf16 operands ignore it."""
    S, L, H, D = q.shape
    assert k.stride() == q.stride() and v.stride() == q.stride() and g.stride() == q.stride()
    d1 = 1 << (D.bit_length() - 1)                    # largest power of two <= D
    d2 = D - d1
    assert d2 == 0 or (d2 >= 16 and d2 & (d2 - 1) == 0), f"head dim {D} is not a sum of two powers of two >= 16"
    assert d2 > 0, "power-of-two head dims: use the engine kernel"
    grid = lambda c: (S, triton.cdiv(L, c["BLOCK_M1"]), H)
    _attn_fwd_gated[grid](q, k, v, g, bias, mask, D ** -0.5, *q.stride(), *bias.stride(), mask.stride(0),
                          H=H, N_CTX=L, HEAD_DIM=D, HEAD_DIM_PAD=max(16, triton.next_power_of_2(D)), D1=d1, D2=d2,
                          PRESCALED=prescaled, PREC=precision)
    return q
