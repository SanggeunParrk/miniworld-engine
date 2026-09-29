"""Forward Triton kernels of the fused ESMFold2 SWA atom DiT block (kernel family ``swa_dit``), bf16.

Moved from team-gm ``src/team_gm/modules/blocks/swa_fused_triton.py`` (commit 14f2c73, "research(swa): preserve opt-in
fused atom transformer work"; copied from team-gm 4fafa83). The kernel bodies are byte-for-byte the team-gm ones. What
changed is only the plumbing the engine requires: the symbols carry a ``_swa_`` prefix (four other kernels in this repo
are called ``_attn_fwd``), the key-only parameter ``SB`` (log2 of the row count) is the engine's ``shape_key``, and the
config lists come from ``configs_for("<op>")`` -- ``autotune/configs/default/<op>.csv`` holds exactly the configs the
team-gm decorators enumerated.

Per row r of the flattened [N = A*B, S] atom sequence (the adaLN modulation depends only on (b, atom) and is hoisted):

    x   = rmsnorm(q) * (1 + scale_a) + shift_a                                                    (bf16)
    q_h, k_h, v_h = split_heads(x @ Wqkv^T);  q_h, k_h = rope(rmsnorm_D(q_h)), rope(rmsnorm_D(k_h))
    o   = sliding-window attention (|i - j| <= HW, keys / queries >= seqused masked, padding rows 0)
    a   = (sigmoid(x @ Wg^T) * o) @ Wo^T ;  q = q + gate_a * a
    y   = rmsnorm(q) * (1 + scale_f) + shift_f ;  q = q + gate_f * ((silu(y Wu1^T) * (y Wu2^T)) @ Wd^T)

Three kernels: qkvg (norm + modulate + 4 projections + qk-norm + RoPE), window attention, out-proj + gated residual + FFN.
Launched from ``kernels/swa_dit/dispatch.py``, which also picks the hand-CUDA twins of the first and last on sm_90.
"""
import triton
import triton.language as tl

from miniworld_engine.autotune.configs import configs_for


@triton.jit
def _rope(x, cs, sn, BR: tl.constexpr, H: tl.constexpr, D: tl.constexpr):
    """x [BR, H*D] fp32 (per head: halves x1 | x2) -> x*cos + rotate_half(x)*sin, cos / sin [BR, D/2] shared by the heads."""
    HALF: tl.constexpr = D // 2
    x4 = tl.reshape(x, (BR, H, 2, HALF))
    x1, x2 = tl.split(tl.permute(x4, (0, 1, 3, 2)))                                  # [BR, H, HALF] each
    c = cs[:, None, :]; s = sn[:, None, :]
    y1 = x1 * c - x2 * s
    y2 = x2 * c + x1 * s
    return tl.reshape(tl.permute(tl.join(y1, y2), (0, 1, 3, 2)), (BR, H * D))


@triton.jit
def _head_rms(x, eps, BR: tl.constexpr, H: tl.constexpr, D: tl.constexpr):
    x3 = tl.reshape(x, (BR, H, D))
    r = 1.0 / tl.sqrt(tl.sum(x3 * x3, axis=2) / D + eps)
    return tl.reshape(x3 * r[:, :, None], (BR, H * D))


# ================================================================ ① qkvg ===============================================================
@triton.autotune(configs=configs_for("swa_dit_inproj_fwd_triton"), key=["shape_key", "SAVE"])
@triton.jit(do_not_specialize=["NROWS", "S", "B"])
def _swa_qkvg_fwd_kernel(Q, MOD, COS, SIN, WQKV, WG, QO, KO, VO, GO, RSTD, XS, PQS, PKS, NROWS, S, B, eps, qk_eps, shape_key,
                         C: tl.constexpr, H: tl.constexpr, D: tl.constexpr, MODW: tl.constexpr, SAVE: tl.constexpr, BR: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BR + tl.arange(0, BR).to(tl.int64); ok = rows < NROWS
    cc = tl.arange(0, C).to(tl.int64)
    n = rows // S; s = rows - n * S; mrow = (n % B) * S + s                          # (b, atom) row of the hoisted modulation
    m2 = ok[:, None]
    q = tl.load(Q + rows[:, None] * C + cc[None, :], mask=m2, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(q * q, axis=1) / C + eps)
    tl.store(RSTD + rows, rstd, mask=ok)
    shift = tl.load(MOD + mrow[:, None] * MODW + cc[None, :], mask=m2, other=0.0).to(tl.float32)
    scale = tl.load(MOD + mrow[:, None] * MODW + C + cc[None, :], mask=m2, other=0.0).to(tl.float32)
    x = (q * rstd[:, None] * (1.0 + scale) + shift).to(tl.bfloat16)
    if SAVE:
        tl.store(XS + rows[:, None] * C + cc[None, :], x, mask=m2)
    HALF: tl.constexpr = D // 2
    hc = tl.arange(0, HALF).to(tl.int64)
    cs = tl.load(COS + mrow[:, None] * HALF + hc[None, :], mask=m2, other=1.0)
    sn = tl.load(SIN + mrow[:, None] * HALF + hc[None, :], mask=m2, other=0.0)
    wofs = cc[None, :] * C + cc[:, None]                                              # W^T tile [in][out]
    # q, k: projection -> bf16 -> per-head RMSNorm (fp32 eps) -> bf16 -> RoPE -> bf16
    # head-major destinations [N, H, S, D]: element (row, h*D + d) -> ((n*H + h)*S + s)*D + d
    hm = ((n[:, None] * H + (cc[None, :] // D)) * S + s[:, None]) * D + (cc[None, :] % D)
    pqb = tl.dot(x, tl.load(WQKV + wofs)).to(tl.bfloat16)
    if SAVE:
        tl.store(PQS + rows[:, None] * C + cc[None, :], pqb, mask=m2)
    pq = _head_rms(pqb.to(tl.float32), qk_eps, BR, H, D).to(tl.bfloat16).to(tl.float32)
    tl.store(QO + hm, _rope(pq, cs, sn, BR, H, D).to(tl.bfloat16), mask=m2)
    pkb = tl.dot(x, tl.load(WQKV + C * C + wofs)).to(tl.bfloat16)
    if SAVE:
        tl.store(PKS + rows[:, None] * C + cc[None, :], pkb, mask=m2)
    pk = _head_rms(pkb.to(tl.float32), qk_eps, BR, H, D).to(tl.bfloat16).to(tl.float32)
    tl.store(KO + hm, _rope(pk, cs, sn, BR, H, D).to(tl.bfloat16), mask=m2)
    pv = tl.dot(x, tl.load(WQKV + 2 * C * C + wofs))
    tl.store(VO + hm, pv.to(tl.bfloat16), mask=m2)
    pg = tl.dot(x, tl.load(WG + wofs))
    tl.store(GO + rows[:, None] * C + cc[None, :], pg.to(tl.bfloat16), mask=m2)


# ================================================================ ② window attention (one head per program) ==========================
@triton.autotune(configs=configs_for("swa_dit_softmax_fwd_triton"), key=["shape_key", "HW"])
@triton.jit(do_not_specialize=["S"])
def _swa_attn_fwd_kernel(QH, KH, VH, SEQU, O, LSE, S, scale, shape_key,
                         C: tl.constexpr, H: tl.constexpr, D: tl.constexpr, HW: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """Program = (query tile, n*H + h).  Q / K / V head-major [N, H, S, D]; keys j with |i - j| <= HW and j < seqused[n].
    O [N, S, C] (bf16, head h in columns h*D..), padding query rows 0; LSE [N, H, S]."""
    t = tl.program_id(0); nh = tl.program_id(1).to(tl.int64)
    n = nh // H; h = nh - n * H
    su = tl.load(SEQU + n)
    i0 = t * BM
    qi = i0 + tl.arange(0, BM); qok = qi < su
    dc = tl.arange(0, D).to(tl.int64)
    hb = nh * S * D
    qh = tl.load(QH + hb + qi[:, None] * D + dc[None, :], mask=qok[:, None], other=0.0)
    m = tl.full((BM,), -float("inf"), tl.float32); l = tl.zeros((BM,), tl.float32); acc = tl.zeros((BM, D), tl.float32)
    lo = tl.maximum(i0 - HW, 0) // BN * BN
    hi = tl.minimum(i0 + BM + HW, su)
    for j0 in range(lo, hi, BN):
        kj = j0 + tl.arange(0, BN); kok = kj < su
        kh = tl.load(KH + hb + kj[:, None] * D + dc[None, :], mask=kok[:, None], other=0.0)
        vh = tl.load(VH + hb + kj[:, None] * D + dc[None, :], mask=kok[:, None], other=0.0)
        sc = tl.dot(qh, tl.trans(kh)) * scale
        dij = qi[:, None] - kj[None, :]
        sc = tl.where((dij <= HW) & (dij >= -HW) & kok[None, :], sc, -float("inf"))
        mn = tl.maximum(m, tl.max(sc, axis=1))
        mn_s = tl.where(mn == -float("inf"), 0.0, mn)
        p = tl.exp(sc - mn_s[:, None])
        alpha = tl.exp(m - mn_s)
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vh)
        m = mn
    l_s = tl.where(l == 0.0, 1.0, l)
    o = tl.where(qok[:, None], acc / l_s[:, None], 0.0)
    qs = qi < S
    tl.store(LSE + nh * S + qi, tl.where(l > 0.0, m + tl.log(l_s), 0.0), mask=qs)
    tl.store(O + (n * S + qi)[:, None] * C + h * D + dc[None, :], o.to(tl.bfloat16), mask=qs[:, None])


# ================================================================ ③ out-proj + gated residual + FFN ===================================
@triton.autotune(configs=configs_for("swa_dit_output_swiglu_fwd_triton"), key=["shape_key", "SAVE"])
@triton.jit(do_not_specialize=["NROWS", "S", "B"])
def _swa_oproj_ffn_fwd_kernel(QI, O, G, MOD, WO, WU, WD, Q1, OUT, RSTD, ATS, YS, ABS, FFS, NROWS, S, B, eps, shape_key,
                              C: tl.constexpr, NHID: tl.constexpr, MODW: tl.constexpr, SAVE: tl.constexpr, BR: tl.constexpr, NH: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BR + tl.arange(0, BR).to(tl.int64); ok = rows < NROWS
    cc = tl.arange(0, C).to(tl.int64)
    n = rows // S; s = rows - n * S; mrow = (n % B) * S + s
    m2 = ok[:, None]
    ofs = rows[:, None] * C + cc[None, :]
    g = tl.load(G + ofs, mask=m2, other=0.0).to(tl.float32)
    gated = (tl.sigmoid(g) * tl.load(O + ofs, mask=m2, other=0.0).to(tl.float32)).to(tl.bfloat16)
    att = tl.dot(gated, tl.load(WO + cc[None, :] * C + cc[:, None])).to(tl.bfloat16)
    ga = tl.load(MOD + mrow[:, None] * MODW + 2 * C + cc[None, :], mask=m2, other=0.0).to(tl.bfloat16)
    q1 = tl.load(QI + ofs, mask=m2, other=0.0) + ga * att                             # bf16
    if SAVE:
        tl.store(Q1 + ofs, q1, mask=m2)
        tl.store(ATS + ofs, att, mask=m2)
    q = q1.to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(q * q, axis=1) / C + eps)
    if SAVE:
        tl.store(RSTD + rows, rstd, mask=ok)
    shift = tl.load(MOD + mrow[:, None] * MODW + 3 * C + cc[None, :], mask=m2, other=0.0).to(tl.float32)
    scale = tl.load(MOD + mrow[:, None] * MODW + 4 * C + cc[None, :], mask=m2, other=0.0).to(tl.float32)
    y = (q * rstd[:, None] * (1.0 + scale) + shift).to(tl.bfloat16)
    if SAVE:
        tl.store(YS + ofs, y, mask=m2)
    nh = tl.arange(0, NH).to(tl.int64)
    acc = tl.zeros((BR, C), dtype=tl.float32)
    for j0 in range(0, NHID, NH):
        hid = j0 + nh
        a = tl.dot(y, tl.load(WU + hid[None, :] * C + cc[:, None]))
        b = tl.dot(y, tl.load(WU + (NHID + hid)[None, :] * C + cc[:, None]))
        hh = (a * tl.sigmoid(a) * b).to(tl.bfloat16)
        acc += tl.dot(hh, tl.load(WD + cc[None, :] * NHID + hid[:, None]))
    gf = tl.load(MOD + mrow[:, None] * MODW + 5 * C + cc[None, :], mask=m2, other=0.0).to(tl.bfloat16)
    ffn = acc.to(tl.bfloat16)
    if SAVE:
        tl.store(FFS + ofs, ffn, mask=m2)
    tl.store(OUT + ofs, q1 + gf * ffn, mask=m2)
