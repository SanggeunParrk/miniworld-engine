"""The packaged v5 core with the hoisted bias stored in fp8 (e4m3) plus one scale per (block, head).

The bias is 18.9 MB per block at L768 and every sample reads all of it: 94 MB from L2, which the ablation prices at
11.3 us of the core's 50.0. Halving the bytes is the only way left to cut that -- sharing the tile between the samples
needs a cluster, which Triton cannot express here and which a hand-written core did not pay for.

The key mask cannot ride along: it is folded into the bf16 bias as -inf, which e4m3 has no room for. A masked run would
need the mask as its own [L] vector added in the loop (cheap, not done here), so these numbers stand for the unmasked case.
"""
import os

import triton
import triton.language as tl

CTAS = int(os.environ.get("ATTN_CTAS", "1"))                       # cluster size: the S same-bias programs are grid dim 0


def _desc_pre_hook(nargs):
    nargs["Bdesc"].block_shape = [nargs["BLOCK_M1"], nargs["BLOCK_M2"]]


# num_ctas > 1 launches the programs that share a bias tile (they differ only in the sample, grid dim 0) as one cluster,
# so the tile is TMA-multicast into all of them: the bias is read from L2 once per cluster instead of once per sample.
def _cfgs2():
    return [triton.Config({"BLOCK_M1": m1, "BLOCK_M2": m2}, num_warps=w, num_stages=s, num_ctas=c,
                          pre_hook=_desc_pre_hook)
            for m1 in (64, 128) for m2 in (32, 64, 128) for w in (4, 8) for s in (2, 3, 4) for c in (CTAS,)]


# restore_value: the output overwrites q, and autotuning runs the kernel once per config.
@triton.autotune(configs=_cfgs2(), key=["N_CTX", "H", "HEAD_DIM", "PREC", "NO_BIAS", "NO_GATE"], restore_value=["Q"])
@triton.jit
def _attn_fp8(Q, K, V, G, Bdesc, BSCALE, brow0,
                     stride_qz, stride_qm, stride_qh, stride_qk,
                     H: tl.constexpr, N_CTX, HEAD_DIM: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr,
                     EVEN: tl.constexpr, PREC: tl.constexpr, NO_BIAS: tl.constexpr, NO_GATE: tl.constexpr,
                     BLOCK_M1: tl.constexpr, BLOCK_M2: tl.constexpr):
    off_z = tl.program_id(0).to(tl.int64)
    start_m = tl.program_id(1)
    off_h = tl.program_id(2)
    base = off_z * stride_qz + off_h.to(tl.int64) * stride_qh
    offset_m = start_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    offset_n = tl.arange(0, BLOCK_M2)
    k1 = tl.arange(0, D1)
    k2 = D1 + tl.arange(0, D2)
    mrow = offset_m[:, None] < N_CTX
    qrow = Q + base + offset_m[:, None].to(tl.int64) * stride_qm
    if EVEN:
        q1 = tl.load(qrow + k1[None, :] * stride_qk)
        q2 = tl.load(qrow + k2[None, :] * stride_qk)
    else:
        q1 = tl.load(qrow + k1[None, :] * stride_qk, mask=mrow, other=0.0)
        q2 = tl.load(qrow + k2[None, :] * stride_qk, mask=mrow, other=0.0)
    kvrow = base + offset_n[:, None].to(tl.int64) * stride_qm
    brow = brow0 + off_h * N_CTX + start_m * BLOCK_M1
    bscale = tl.load(BSCALE + brow0 // N_CTX + off_h).to(tl.float32)   # one scale per (block, head)
    m_i = tl.full([BLOCK_M1], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M1], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_M1, D1], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_M1, D2], dtype=tl.float32)
    for start_n in range(0, N_CTX, BLOCK_M2):
        ro = start_n.to(tl.int64) * stride_qm
        if EVEN:
            kk1 = tl.load(K + kvrow + ro + k1[None, :] * stride_qk)
            kk2 = tl.load(K + kvrow + ro + k2[None, :] * stride_qk)
        else:
            nm = (start_n + offset_n)[:, None] < N_CTX
            kk1 = tl.load(K + kvrow + ro + k1[None, :] * stride_qk, mask=nm, other=0.0)
            kk2 = tl.load(K + kvrow + ro + k2[None, :] * stride_qk, mask=nm, other=0.0)
        qk = tl.dot(q1, tl.trans(kk1), input_precision=PREC)
        qk = tl.dot(q2, tl.trans(kk2), qk, input_precision=PREC)
        # logits already in the exp2 domain: sm_scale*log2(e) is folded into q, log2(e) into the bias
        sc = qk if NO_BIAS else qk + Bdesc.load([brow, start_n]).to(tl.float32) * bscale
        if not EVEN:
            sc = tl.where(((start_n + offset_n) < N_CTX)[None, :], sc, -float("inf"))
        m_new = tl.maximum(tl.maximum(m_i, tl.max(sc, 1)), -1e38)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new
        if EVEN:
            v1 = tl.load(V + kvrow + ro + k1[None, :] * stride_qk)
            v2 = tl.load(V + kvrow + ro + k2[None, :] * stride_qk)
        else:
            v1 = tl.load(V + kvrow + ro + k1[None, :] * stride_qk, mask=nm, other=0.0)
            v2 = tl.load(V + kvrow + ro + k2[None, :] * stride_qk, mask=nm, other=0.0)
        pb = p.to(v1.dtype)
        acc1 = tl.dot(pb, v1, acc1 * alpha[:, None], input_precision=PREC)
        acc2 = tl.dot(pb, v2, acc2 * alpha[:, None], input_precision=PREC)
    inv = 1.0 / tl.maximum(l_i, 1e-30)
    grow = G + base + offset_m[:, None].to(tl.int64) * stride_qm
    if EVEN:
        g1 = tl.load(grow + k1[None, :] * stride_qk).to(tl.float32)
        g2 = tl.load(grow + k2[None, :] * stride_qk).to(tl.float32)
        tl.store(qrow + k1[None, :] * stride_qk, (acc1 * inv[:, None] * (1.0 if NO_GATE else tl.sigmoid(g1))).to(Q.dtype.element_ty))
        tl.store(qrow + k2[None, :] * stride_qk, (acc2 * inv[:, None] * (1.0 if NO_GATE else tl.sigmoid(g2))).to(Q.dtype.element_ty))
    else:
        g1 = tl.load(grow + k1[None, :] * stride_qk, mask=mrow, other=0.0).to(tl.float32)
        g2 = tl.load(grow + k2[None, :] * stride_qk, mask=mrow, other=0.0).to(tl.float32)
        tl.store(qrow + k1[None, :] * stride_qk, (acc1 * inv[:, None] * (1.0 if NO_GATE else tl.sigmoid(g1))).to(Q.dtype.element_ty), mask=mrow)
        tl.store(qrow + k2[None, :] * stride_qk, (acc2 * inv[:, None] * (1.0 if NO_GATE else tl.sigmoid(g2))).to(Q.dtype.element_ty), mask=mrow)


def bias_descriptor(bias_all):
    """One TMA descriptor over every block's hoisted bias, viewed [nb*H*L, L]; the block shape is set per config."""
    from triton.tools.tensor_descriptor import TensorDescriptor
    NBH, L, _ = bias_all.shape
    return TensorDescriptor(bias_all, [NBH * L, L], [L, 1], [64, 64])


def attention_fp8(q, k, v, g, bdesc, bscale, block, precision="tf32", no_bias=False, no_gate=False):
    """As ``attention_gated_in_place`` with pre-scaled logits, the key mask already folded into the bias, and the bias
    read through ``bdesc`` (``bias_descriptor``) at block ``block``'s head rows."""
    S, L, H, D = q.shape
    assert k.stride() == q.stride() and v.stride() == q.stride() and g.stride() == q.stride()
    d1 = 1 << (D.bit_length() - 1)
    d2 = D - d1
    assert d2 >= 16 and d2 & (d2 - 1) == 0, f"head dim {D} is not a power of two plus a power of two >= 16"
    grid = lambda c: (S, triton.cdiv(L, c["BLOCK_M1"]), H)
    _attn_fp8[grid](q, k, v, g, bdesc, bscale, block * H * L, *q.stride(), H=H, N_CTX=L, HEAD_DIM=D, D1=d1, D2=d2,
                   EVEN=(L % 128 == 0), PREC=precision, NO_BIAS=no_bias, NO_GATE=no_gate)
    return q


def quantize_bias(bias):
    """bias [nb*H, L, L] bf16 -> (fp8 e4m3 tensor, per-row-block scale [nb*H] fp32) and its TMA descriptor."""
    import torch
    from triton.tools.tensor_descriptor import TensorDescriptor
    amax = bias.float().abs().amax(dim=(1, 2)).clamp_min(1e-30)
    scale = (amax / 448.0)
    q = (bias.float() / scale[:, None, None]).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
    nbh, L, _ = q.shape
    return q, scale.contiguous(), TensorDescriptor(q, [nbh * L, L], [L, 1], [64, 64])
