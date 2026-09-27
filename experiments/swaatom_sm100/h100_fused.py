"""Fused ESMFold2 SWA atom DiT block (team_gm SWAAtomBlock, block_style "esmfold2"), bf16, Triton.

Block (per row r of the flattened [N = A*B, S] atom sequence; the adaLN modulation depends only on (b, atom) and is hoisted):
    mod = silu(c) @ Wmod^T -> shift_a | scale_a | gate_a | shift_f | scale_f | gate_f           (per (b, atom), once)
    x   = rmsnorm(q) * (1 + scale_a) + shift_a                                                    (bf16)
    q_h, k_h, v_h = split_heads(x @ Wqkv^T);  q_h, k_h = rope(rmsnorm_D(q_h)), rope(rmsnorm_D(k_h))
    o   = sliding-window attention (|i - j| <= 64, keys / queries >= seqused masked, padding rows 0)
    a   = (sigmoid(x @ Wg^T) * o) @ Wo^T ;  q = q + gate_a * a
    y   = rmsnorm(q) * (1 + scale_f) + shift_f ;  q = q + gate_f * ((silu(y Wu1^T) * (y Wu2^T)) @ Wd^T)
Forward kernels: qkvg (norm + modulate + 4 projections + qk-norm + RoPE), window attention, out-proj + gated residual + FFN.
On Hopper the qkvg / out-proj+FFN forward and the FFN backward run as CUDA bf16-wgmma kernels (swa_cuda/ or cuda/ next to this file,
built on first use; env SWA_QKVG_FWD / SWA_FFN_FWD / SWA_FFN_BWD = triton selects the Triton kernel), everything else in Triton."""
import math
import os
import torch
import triton
import triton.language as tl

LAST = {}
FP32_EPS = float(torch.finfo(torch.float32).eps)


def _bucket(n):
    return int(math.log2(max(int(n), 1)))


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
@triton.autotune(configs=[triton.Config({"BR": br}, num_warps=w, num_stages=st) for br in (32, 64, 128) for w in (4, 8) for st in (1, 2)],
                 key=["SB", "SAVE"])
@triton.jit(do_not_specialize=["NROWS", "S", "B"])
def _qkvg_fwd(Q, MOD, COS, SIN, WQKV, WG, QO, KO, VO, GO, RSTD, XS, PQS, PKS, NROWS, S, B, eps, qk_eps, SB,
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
@triton.autotune(configs=[triton.Config({"BM": bm, "BN": bn}, num_warps=w, num_stages=st) for bm, bn in ((64, 64), (128, 64), (64, 32), (128, 32))
                          for w in (4, 8) for st in (1, 2, 3)], key=["SB", "HW"])
@triton.jit(do_not_specialize=["S"])
def _attn_fwd(QH, KH, VH, SEQU, O, LSE, S, scale, SB,
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
@triton.autotune(configs=[triton.Config({"BR": br, "NH": nh}, num_warps=w, num_stages=st) for br in (32, 64, 128) for nh in (64, 128)
                          for w in (4, 8) for st in (1, 2)] + [triton.Config({"BR": 16, "NH": 64}, num_warps=2, num_stages=2)], key=["SB", "SAVE"])
@triton.jit(do_not_specialize=["NROWS", "S", "B"])
def _oproj_ffn_fwd(QI, O, G, MOD, WO, WU, WD, Q1, OUT, RSTD, ATS, YS, ABS, FFS, NROWS, S, B, eps, SB,
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


# ================================================================ host =================================================================
def block_fwd(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window=64, eps=FP32_EPS, save=False):
    """q [N, S, C] bf16; mod [B*S, 6C] fp32 (hoisted adaLN modulation of this block); cos / sin [B*S, D/2] fp32; seqused [N] int32."""
    N, S, C = q.shape; H = 4; D = C // H; M = N * S
    qf = q.reshape(M, C)
    Qh = torch.empty(N, H, S, D, device=q.device, dtype=q.dtype); Kh = torch.empty_like(Qh); Vh = torch.empty_like(Qh); G = torch.empty_like(qf)
    r1 = torch.empty(M, device=q.device); sb = _bucket(M)
    if save:
        Xs = torch.empty_like(qf); PQs = torch.empty_like(qf); PKs = torch.empty_like(qf)
    else:
        Xs = PQs = PKs = G
    if _qkvg_fwd_cuda_ok(C, H, q.dtype):
        Qh, Kh, Vh, G, Xs, PQs, PKs = _cuda_ext("qkvg").qkvg_fwd(qf, mod, cos, sin, torch.cat([wqkv, wg]), N // B, B, S, eps, FP32_EPS, save)
    else:
        LAST["qkvg"] = _qkvg_fwd[lambda m: (triton.cdiv(M, m["BR"]),)](qf, mod, cos, sin, wqkv, wg, Qh, Kh, Vh, G, r1, Xs, PQs, PKs, M, S, B, eps, FP32_EPS,
                                                                       sb, C=C, H=H, D=D, MODW=6 * C, SAVE=save)
    O = torch.empty_like(qf); lse = torch.empty(N, H, S, device=q.device)
    LAST["attn"] = _attn_fwd[lambda m: (triton.cdiv(S, m["BM"]), N * H)](Qh, Kh, Vh, seqused, O, lse, S, D ** -0.5, sb, C=C, H=H, D=D, HW=half_window)
    NHID = wd.shape[1]
    if _ffn_fwd_cuda_ok(C, NHID, q.dtype):
        q2, q1, Att, Ys, FFs = _cuda_ext("fwd").ffn_fwd(qf, G, O, mod, wo, _pack_ffn64(wu), wd, N // B, B, S, eps, save)
        out = q2.view(N, S, C)
        if save:
            return out, dict(Qh=Qh, Kh=Kh, Vh=Vh, G=G, O=O, lse=lse, q1=q1, X=Xs, PQ=PQs, PK=PKs, Att=Att, Y=Ys, FF=FFs)
        return out
    q2 = torch.empty_like(qf)
    q1 = torch.empty_like(qf) if save else q2; r2 = torch.empty(M, device=q.device) if save else r1
    if save:
        Att = torch.empty_like(qf); Ys = torch.empty_like(qf); ABs = Ys; FFs = torch.empty_like(qf)
    else:
        Att = Ys = ABs = FFs = q2
    LAST["ffn"] = _oproj_ffn_fwd[lambda m: (triton.cdiv(M, m["BR"]),)](qf, O, G, mod, wo, wu, wd, q1, q2, r2, Att, Ys, ABs, FFs, M, S, B, eps, sb,
                                                                       C=C, NHID=NHID, MODW=6 * C, SAVE=save)
    out = q2.view(N, S, C)
    if save:
        return out, dict(Qh=Qh, Kh=Kh, Vh=Vh, G=G, O=O, lse=lse, q1=q1, X=Xs, PQ=PQs, PK=PKs, Att=Att, Y=Ys, FF=FFs)
    return out


def hoist_mod(c1, wmod):
    """c1 [B, S, d_cond] (augment-invariant conditioning), wmod [6C, d_cond] -> [B*S, 6C] fp32 = silu(c1) (in c1's dtype) @ wmod^T,
    fp32 accumulate and kept fp32 (as the engine's rmsnorm_adamod keeps its projections in registers)."""
    a = torch.nn.functional.silu(c1).float()
    return (a.reshape(-1, a.shape[-1]) @ wmod.float().t()).contiguous()


# ================================================================ backward ============================================================
# Tiles for the kernels that reduce the (augment-invariant) modulation gradient: SP augments x AT atoms of one batch element b;
# rows r = ((a*B + b)*S + s).  Pre-summed over the tile's augments, then one atomic add per (atom, channel).
@triton.jit
def _tile_rows(S, A, B, SP: tl.constexpr, AT: tl.constexpr):
    ab = tl.program_id(0).to(tl.int64); ag = tl.program_id(1).to(tl.int64); b = tl.program_id(2).to(tl.int64)
    R: tl.constexpr = SP * AT
    r = tl.arange(0, R).to(tl.int64)
    a = ag * SP + r // AT; s = ab * AT + r % AT
    ok = (a < A) & (s < S)
    rows = (a * B + b) * S + s
    return rows, ok, b * S + s, s


@triton.jit
def _presum_add(DMOD, x, col0, ab_idx, b, S, ok_at, C: tl.constexpr, MODW: tl.constexpr, SP: tl.constexpr, AT: tl.constexpr):
    """x [SP*AT, C] (zero on invalid rows) -> sum over the SP augments -> atomic add into DMOD[(b*S + atom), col0 + c]."""
    cc = tl.arange(0, C).to(tl.int64)
    ar = ab_idx * AT + tl.arange(0, AT).to(tl.int64)
    tl.atomic_add(DMOD + (b * S + ar)[:, None] * MODW + col0 + cc[None, :], tl.sum(tl.reshape(x, (SP, AT, C)), axis=0),
                  mask=ok_at[:, None], sem="relaxed")


@triton.autotune(configs=[triton.Config({"SP": sp, "AT": at, "NH": nh}, num_warps=w, num_stages=1)
                          for sp, at in ((2, 16), (4, 8), (2, 32), (4, 16), (1, 32)) for nh in (64, 128) for w in (4, 8)]
                         + [triton.Config({"SP": sp, "AT": at, "NH": 64}, num_warps=4, num_stages=1) for sp, at in ((16, 1), (2, 8), (8, 2))],   # NH=32 with 8 warps, SP=2, AT=32 faults (Triton codegen)
                 key=["SB"], restore_value=["DMOD"])
@triton.jit(do_not_specialize=["S", "A", "B"])
def _ffn_bwd(DQ2, Q1, MOD, WU, WD, YS, FFS, DQ1, DAB, HH, DFFN, DMOD, S, A, B, eps, SB,
             C: tl.constexpr, NHID: tl.constexpr, MODW: tl.constexpr, DWOPS: tl.constexpr, SP: tl.constexpr, AT: tl.constexpr, NH: tl.constexpr):
    """FFN backward from the saved a|b and FFN output: dq1 = dq2 + RMSNorm/adaLN backward; dW operands; d shift_f / scale_f / gate_f."""
    rows, ok, mrow, s_ = _tile_rows(S, A, B, SP, AT)
    b = tl.program_id(2).to(tl.int64); ab_idx = tl.program_id(0).to(tl.int64)
    ok_at = (ab_idx * AT + tl.arange(0, AT)) < S
    cc = tl.arange(0, C).to(tl.int64)
    m2 = ok[:, None]
    ofs = rows[:, None] * C + cc[None, :]
    dq2 = tl.load(DQ2 + ofs, mask=m2, other=0.0).to(tl.float32)
    dffn = (dq2 * tl.load(MOD + mrow[:, None] * MODW + 5 * C + cc[None, :], mask=m2, other=0.0).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    if DWOPS:
        tl.store(DFFN + ofs, dffn, mask=m2)
    _presum_add(DMOD, tl.where(m2, dq2 * tl.load(FFS + ofs, mask=m2, other=0.0).to(tl.float32), 0.0), 5 * C, ab_idx, b, S, ok_at, C, MODW, SP, AT)
    nh = tl.arange(0, NH).to(tl.int64)
    dy = tl.zeros((SP * AT, C), dtype=tl.float32)
    y = tl.load(YS + ofs, mask=m2, other=0.0)
    for j0 in range(0, NHID, NH):
        hid = j0 + nh
        a = tl.dot(y, tl.load(WU + hid[None, :] * C + cc[:, None]))                          # recomputed (not saved in the forward)
        bb = tl.dot(y, tl.load(WU + (NHID + hid)[None, :] * C + cc[:, None]))
        sa = tl.sigmoid(a)
        if DWOPS:
            tl.store(HH + rows[:, None] * NHID + hid[None, :], (a * sa * bb).to(tl.bfloat16), mask=m2)
        dh = tl.dot(dffn, tl.load(WD + cc[:, None] * NHID + hid[None, :]))                 # [R][NH] = dffn Wd[:, hid]
        da = (dh * bb * sa * (1.0 + a * (1.0 - sa))).to(tl.bfloat16)
        db = (dh * a * sa).to(tl.bfloat16)
        dy += tl.dot(da, tl.load(WU + hid[:, None] * C + cc[None, :])) + tl.dot(db, tl.load(WU + (NHID + hid)[:, None] * C + cc[None, :]))
        if DWOPS:
            tl.store(DAB + rows[:, None] * (2 * NHID) + hid[None, :], da, mask=m2)
            tl.store(DAB + rows[:, None] * (2 * NHID) + NHID + hid[None, :], db, mask=m2)
    q1 = tl.load(Q1 + ofs, mask=m2, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(q1 * q1, axis=1) / C + eps)
    xh = q1 * rstd[:, None]
    _presum_add(DMOD, tl.where(m2, dy * xh, 0.0), 4 * C, ab_idx, b, S, ok_at, C, MODW, SP, AT)                                    # d scale_f
    _presum_add(DMOD, tl.where(m2, dy, 0.0), 3 * C, ab_idx, b, S, ok_at, C, MODW, SP, AT)                                         # d shift_f
    dxh = dy * (1.0 + tl.load(MOD + mrow[:, None] * MODW + 4 * C + cc[None, :], mask=m2, other=0.0))
    tl.store(DQ1 + ofs, (dq2 + rstd[:, None] * (dxh - xh * (tl.sum(dxh * xh, axis=1) / C)[:, None])).to(DQ1.dtype.element_ty), mask=m2)


@triton.autotune(configs=[triton.Config({"BR": br, "HS": hs}, num_warps=w, num_stages=st) for br in (32, 64, 128) for hs in (32, 64)
                          for w in (4, 8) for st in (1, 2)], key=["SB"], reset_to_zero=["DWU", "DWD"])
@triton.jit(do_not_specialize=["M", "S", "B"])
def _ffn_dw(DQ2, MOD, WU, WD, YS, DWU, DWD, M, S, B, SB, NSPLIT,
            C: tl.constexpr, NHID: tl.constexpr, MODW: tl.constexpr, BR: tl.constexpr, HS: tl.constexpr):
    """dW_up / dW_down without materialising [da|db] / h: program (split, hidden slice) recomputes a, b, dh for its HS hidden units
    over the rows t = split, split + NSPLIT, ... and keeps dWu[slice] (a and b halves) and dWd[:, slice] in registers."""
    sp = tl.program_id(0); hs = tl.program_id(1).to(tl.int64)
    cc = tl.arange(0, C).to(tl.int64); hid = hs * HS + tl.arange(0, HS).to(tl.int64)
    wa = tl.load(WU + hid[None, :] * C + cc[:, None]); wb = tl.load(WU + (NHID + hid)[None, :] * C + cc[:, None])   # [C][HS]
    wd = tl.load(WD + cc[:, None] * NHID + hid[None, :])                                                             # [C][HS]
    acc_a = tl.zeros((HS, C), dtype=tl.float32); acc_b = tl.zeros((HS, C), dtype=tl.float32); acc_d = tl.zeros((C, HS), dtype=tl.float32)
    for t in range(sp, tl.cdiv(M, BR), tl.num_programs(0)):
        rows = t * BR + tl.arange(0, BR).to(tl.int64); ok = rows < M
        n = rows // S; mrow = (n % B) * S + (rows - n * S)
        m2 = ok[:, None]; ofs = rows[:, None] * C + cc[None, :]
        y = tl.load(YS + ofs, mask=m2, other=0.0)
        dq2 = tl.load(DQ2 + ofs, mask=m2, other=0.0).to(tl.float32)
        dffn = (dq2 * tl.load(MOD + mrow[:, None] * MODW + 5 * C + cc[None, :], mask=m2, other=0.0).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        a = tl.dot(y, wa); bb = tl.dot(y, wb); dh = tl.dot(dffn, wd)
        sa = tl.sigmoid(a)
        hh = (a * sa * bb).to(tl.bfloat16)
        da = (dh * bb * sa * (1.0 + a * (1.0 - sa))).to(tl.bfloat16)
        db = (dh * a * sa).to(tl.bfloat16)
        acc_a += tl.dot(tl.trans(da), y); acc_b += tl.dot(tl.trans(db), y); acc_d += tl.dot(tl.trans(dffn), hh)
    hr = tl.arange(0, HS).to(tl.int64)
    tl.atomic_add(DWU + (hs * HS + hr)[:, None] * C + cc[None, :], acc_a, sem="relaxed")
    tl.atomic_add(DWU + (NHID + hs * HS + hr)[:, None] * C + cc[None, :], acc_b, sem="relaxed")
    tl.atomic_add(DWD + cc[:, None] * NHID + (hs * HS + hr)[None, :], acc_d, sem="relaxed")


@triton.autotune(configs=[triton.Config({"SP": sp, "AT": at}, num_warps=w, num_stages=1)
                          for sp, at in ((2, 16), (4, 8), (2, 32), (4, 16), (1, 32)) for w in (4, 8)]
                         + [triton.Config({"SP": sp, "AT": at}, num_warps=w, num_stages=1) for sp, at in ((16, 1), (4, 4), (2, 8)) for w in (2, 4)],
                 key=["SB"], restore_value=["DMOD"])
@triton.jit(do_not_specialize=["S", "A", "B"])
def _oproj_bwd(DQ1, O, G, MOD, WO, ATS, DOUT_, DG, DVv, DATT, GATED, DMOD, S, A, B, SB,
               C: tl.constexpr, H: tl.constexpr, D: tl.constexpr, MODW: tl.constexpr, SP: tl.constexpr, AT: tl.constexpr):
    """q1 = q + ga * ((sigmoid(g) * O) Wo^T): dO, dG, D = per-head rowsum(dO * O), d gate_a, dWo operands."""
    rows, ok, mrow, s_ = _tile_rows(S, A, B, SP, AT)
    b = tl.program_id(2).to(tl.int64); ab_idx = tl.program_id(0).to(tl.int64)
    ok_at = (ab_idx * AT + tl.arange(0, AT)) < S
    cc = tl.arange(0, C).to(tl.int64)
    m2 = ok[:, None]
    ofs = rows[:, None] * C + cc[None, :]
    dq1 = tl.load(DQ1 + ofs, mask=m2, other=0.0)
    g = tl.load(G + ofs, mask=m2, other=0.0).to(tl.float32); so = tl.sigmoid(g)
    o = tl.load(O + ofs, mask=m2, other=0.0).to(tl.float32)
    gated = (so * o).to(tl.bfloat16)
    tl.store(GATED + ofs, gated, mask=m2)
    att = tl.load(ATS + ofs, mask=m2, other=0.0).to(tl.float32)
    ga = tl.load(MOD + mrow[:, None] * MODW + 2 * C + cc[None, :], mask=m2, other=0.0).to(tl.bfloat16).to(tl.float32)
    _presum_add(DMOD, tl.where(m2, dq1 * att, 0.0), 2 * C, ab_idx, b, S, ok_at, C, MODW, SP, AT)    # d gate_a
    datt = (dq1 * ga).to(tl.bfloat16)
    tl.store(DATT + ofs, datt, mask=m2)
    dgated = tl.dot(datt, tl.load(WO + cc[:, None] * C + cc[None, :]))                           # datt Wo
    do_ = (dgated * so).to(tl.bfloat16)
    tl.store(DOUT_ + ofs, do_, mask=m2)
    tl.store(DG + ofs, (dgated * o * so * (1.0 - so)).to(tl.bfloat16), mask=m2)
    dvv = tl.sum(tl.reshape(do_.to(tl.float32) * o, (SP * AT, H, D)), axis=2)
    hh_ = tl.arange(0, H).to(tl.int64)
    n = rows // S
    tl.store(DVv + (n[:, None] * H + hh_[None, :]) * S + s_[:, None], dvv, mask=m2)


@triton.autotune(configs=[triton.Config({"BM": bm, "BN": bn}, num_warps=w, num_stages=st) for bm, bn in ((64, 64), (64, 32), (128, 64), (32, 32))
                          for w in (4, 8) for st in (1, 2)], key=["SB", "HW"])
@triton.jit(do_not_specialize=["S"])
def _attn_bwd_dq(QH, KH, VH, DOUT_, LSE, DVv, SEQU, DQH, S, scale, SB,
                 C: tl.constexpr, H: tl.constexpr, D: tl.constexpr, HW: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    t = tl.program_id(0); nh = tl.program_id(1).to(tl.int64)
    n = nh // H; h = nh - n * H
    su = tl.load(SEQU + n)
    i0 = t * BM
    qi = i0 + tl.arange(0, BM); qok = qi < su
    dc = tl.arange(0, D).to(tl.int64)
    hb = nh * S * D
    q = tl.load(QH + hb + qi[:, None] * D + dc[None, :], mask=qok[:, None], other=0.0)
    do_ = tl.load(DOUT_ + (n * S + qi)[:, None] * C + h * D + dc[None, :], mask=qok[:, None], other=0.0)
    lse = tl.load(LSE + nh * S + qi, mask=qok, other=0.0)
    dv = tl.load(DVv + nh * S + qi, mask=qok, other=0.0)
    dq = tl.zeros((BM, D), tl.float32)
    lo = tl.maximum(i0 - HW, 0) // BN * BN
    hi = tl.minimum(i0 + BM + HW, su)
    for j0 in range(lo, hi, BN):
        kj = j0 + tl.arange(0, BN); kok = kj < su
        k = tl.load(KH + hb + kj[:, None] * D + dc[None, :], mask=kok[:, None], other=0.0)
        v = tl.load(VH + hb + kj[:, None] * D + dc[None, :], mask=kok[:, None], other=0.0)
        dij = qi[:, None] - kj[None, :]
        allow = (dij <= HW) & (dij >= -HW) & kok[None, :] & qok[:, None]
        p = tl.where(allow, tl.exp(tl.dot(q, tl.trans(k)) * scale - lse[:, None]), 0.0)
        dp = tl.dot(do_, tl.trans(v))
        ds = (p * (dp - dv[:, None])).to(tl.bfloat16)
        dq += tl.dot(ds, k)
    tl.store(DQH + hb + qi[:, None] * D + dc[None, :], (dq * scale).to(tl.bfloat16), mask=(qi < S)[:, None])


@triton.autotune(configs=[triton.Config({"BM": bm, "BN": bn}, num_warps=w, num_stages=st) for bm, bn in ((64, 64), (32, 64), (64, 128), (32, 32))
                          for w in (4, 8) for st in (1, 2)], key=["SB", "HW"])
@triton.jit(do_not_specialize=["S"])
def _attn_bwd_dkv(QH, KH, VH, DOUT_, LSE, DVv, SEQU, DKH, DVH, S, scale, SB,
                  C: tl.constexpr, H: tl.constexpr, D: tl.constexpr, HW: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """Program = (key tile of BN keys, n*H + h); loops the query tiles (BM) whose windows reach the tile."""
    t = tl.program_id(0); nh = tl.program_id(1).to(tl.int64)
    n = nh // H; h = nh - n * H
    su = tl.load(SEQU + n)
    j0 = t * BN
    kj = j0 + tl.arange(0, BN); kok = kj < su
    dc = tl.arange(0, D).to(tl.int64)
    hb = nh * S * D
    k = tl.load(KH + hb + kj[:, None] * D + dc[None, :], mask=kok[:, None], other=0.0)
    v = tl.load(VH + hb + kj[:, None] * D + dc[None, :], mask=kok[:, None], other=0.0)
    dk = tl.zeros((BN, D), tl.float32); dv = tl.zeros((BN, D), tl.float32)
    lo = tl.maximum(j0 - HW, 0) // BM * BM
    hi = tl.minimum(j0 + BN + HW, su)
    for i0 in range(lo, hi, BM):
        qi = i0 + tl.arange(0, BM); qok = qi < su
        q = tl.load(QH + hb + qi[:, None] * D + dc[None, :], mask=qok[:, None], other=0.0)
        do_ = tl.load(DOUT_ + (n * S + qi)[:, None] * C + h * D + dc[None, :], mask=qok[:, None], other=0.0)
        lse = tl.load(LSE + nh * S + qi, mask=qok, other=0.0)
        dvv = tl.load(DVv + nh * S + qi, mask=qok, other=0.0)
        dij = qi[None, :] - kj[:, None]                                                  # [BN keys][BM queries]
        allow = (dij <= HW) & (dij >= -HW) & kok[:, None] & qok[None, :]
        pT = tl.where(allow, tl.exp(tl.dot(k, tl.trans(q)) * scale - lse[None, :]), 0.0)
        dv += tl.dot(pT.to(tl.bfloat16), do_)
        dpT = tl.dot(v, tl.trans(do_))
        dsT = (pT * (dpT - dvv[None, :])).to(tl.bfloat16)
        dk += tl.dot(dsT, q)
    km = (kj < S)[:, None]
    tl.store(DKH + hb + kj[:, None] * D + dc[None, :], (dk * scale).to(tl.bfloat16), mask=km)
    tl.store(DVH + hb + kj[:, None] * D + dc[None, :], dv.to(tl.bfloat16), mask=km)


@triton.jit
def _unrope(dy, cs, sn, BR: tl.constexpr, H: tl.constexpr, D: tl.constexpr):
    """Transpose of _rope: dx1 = dy1*c + dy2*s ; dx2 = dy2*c - dy1*s."""
    HALF: tl.constexpr = D // 2
    y4 = tl.reshape(dy, (BR, H, 2, HALF))
    y1, y2 = tl.split(tl.permute(y4, (0, 1, 3, 2)))
    c = cs[:, None, :]; s = sn[:, None, :]
    return tl.reshape(tl.permute(tl.join(y1 * c + y2 * s, y2 * c - y1 * s), (0, 1, 3, 2)), (BR, H * D))


@triton.jit
def _head_rms_bwd(p, dy, eps, BR: tl.constexpr, H: tl.constexpr, D: tl.constexpr):
    """y = p * r (r = rsqrt(mean_D p^2 + eps)) -> dp = r * (dy - yhat * mean_D(dy * yhat)), yhat = p * r."""
    p3 = tl.reshape(p, (BR, H, D)); d3 = tl.reshape(dy, (BR, H, D))
    r = 1.0 / tl.sqrt(tl.sum(p3 * p3, axis=2) / D + eps)
    yh = p3 * r[:, :, None]
    return tl.reshape(r[:, :, None] * (d3 - yh * (tl.sum(d3 * yh, axis=2) / D)[:, :, None]), (BR, H * D))


@triton.autotune(configs=[triton.Config({"SP": sp, "AT": at}, num_warps=w, num_stages=1)
                          for sp, at in ((2, 16), (4, 8), (2, 32), (4, 16), (8, 4), (16, 2)) for w in (4, 8)],
                 key=["SB"], restore_value=["DMOD"])
@triton.jit(do_not_specialize=["S", "A", "B"])
def _qkvg_bwd(QI, MOD, COS, SIN, WQKV, WG, PQS, PKS, DQH, DKH, DVH, DG, DQ1, DQ, DP, DMOD, S, A, B, eps, qk_eps, SB,
              C: tl.constexpr, H: tl.constexpr, D: tl.constexpr, MODW: tl.constexpr, SP: tl.constexpr, AT: tl.constexpr):
    rows, ok, mrow, s_ = _tile_rows(S, A, B, SP, AT)
    b = tl.program_id(2).to(tl.int64); ab_idx = tl.program_id(0).to(tl.int64)
    ok_at = (ab_idx * AT + tl.arange(0, AT)) < S
    R: tl.constexpr = SP * AT
    cc = tl.arange(0, C).to(tl.int64)
    m2 = ok[:, None]
    ofs = rows[:, None] * C + cc[None, :]
    q = tl.load(QI + ofs, mask=m2, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(q * q, axis=1) / C + eps)
    xh = q * rstd[:, None]
    scale = tl.load(MOD + mrow[:, None] * MODW + C + cc[None, :], mask=m2, other=0.0)
    HALF: tl.constexpr = D // 2
    hc = tl.arange(0, HALF).to(tl.int64)
    cs = tl.load(COS + mrow[:, None] * HALF + hc[None, :], mask=m2, other=1.0)
    sn = tl.load(SIN + mrow[:, None] * HALF + hc[None, :], mask=m2, other=0.0)
    n = rows // S
    hm = ((n[:, None] * H + (cc[None, :] // D)) * S + s_[:, None]) * D + (cc[None, :] % D)
    # q: the pre-norm projection, back through RoPE and the per-head RMSNorm
    pq = tl.load(PQS + ofs, mask=m2, other=0.0).to(tl.float32)
    dpq = _head_rms_bwd(pq, _unrope(tl.load(DQH + hm, mask=m2, other=0.0).to(tl.float32), cs, sn, R, H, D), qk_eps, R, H, D)
    dpq_b = dpq.to(tl.bfloat16)
    tl.store(DP + rows[:, None] * (4 * C) + cc[None, :], dpq_b, mask=m2)
    dx = tl.dot(dpq_b, tl.load(WQKV + cc[:, None] * C + cc[None, :]))
    pk = tl.load(PKS + ofs, mask=m2, other=0.0).to(tl.float32)
    dpk = _head_rms_bwd(pk, _unrope(tl.load(DKH + hm, mask=m2, other=0.0).to(tl.float32), cs, sn, R, H, D), qk_eps, R, H, D)
    dpk_b = dpk.to(tl.bfloat16)
    tl.store(DP + rows[:, None] * (4 * C) + C + cc[None, :], dpk_b, mask=m2)
    dx += tl.dot(dpk_b, tl.load(WQKV + C * C + cc[:, None] * C + cc[None, :]))
    dpv_b = tl.load(DVH + hm, mask=m2, other=0.0).to(tl.bfloat16)
    tl.store(DP + rows[:, None] * (4 * C) + 2 * C + cc[None, :], dpv_b, mask=m2)
    dx += tl.dot(dpv_b, tl.load(WQKV + 2 * C * C + cc[:, None] * C + cc[None, :]))
    dpg_b = tl.load(DG + ofs, mask=m2, other=0.0)
    tl.store(DP + rows[:, None] * (4 * C) + 3 * C + cc[None, :], dpg_b, mask=m2)
    dx += tl.dot(dpg_b, tl.load(WG + cc[:, None] * C + cc[None, :]))
    _presum_add(DMOD, tl.where(m2, dx * xh, 0.0), C, ab_idx, b, S, ok_at, C, MODW, SP, AT)          # d scale_a
    _presum_add(DMOD, tl.where(m2, dx, 0.0), 0, ab_idx, b, S, ok_at, C, MODW, SP, AT)               # d shift_a
    dxh = dx * (1.0 + scale)
    dq = tl.load(DQ1 + ofs, mask=m2, other=0.0) + rstd[:, None] * (dxh - xh * (tl.sum(dxh * xh, axis=1) / C)[:, None])
    tl.store(DQ + ofs, dq.to(tl.bfloat16), mask=m2)


_CUDA = {}


def _cuda_ext(which="bwd"):
    """CUDA (sm_90a bf16 wgmma) kernels next to this file: cuda/swa_ffn_{fwd,bwd}.cu.  None when unavailable (not Hopper / build failure)."""
    if which not in _CUDA:
        _CUDA[which] = None
        try:
            if torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0):
                from torch.utils.cpp_extension import load
                here = os.path.dirname(os.path.abspath(__file__))
                d = next(p for p in (os.path.join(here, "swa_cuda"), os.path.join(here, "cuda")) if os.path.isdir(p))
                src = {"bwd": "swa_ffn_bwd", "fwd": "swa_ffn_fwd", "qkvg": "swa_qkvg_fwd"}[which]
                kw = {}
                if os.environ.get("SWA_CUDA_BUILD"):                # default: torch's extension cache (file-locked across ranks)
                    kw["build_directory"] = os.path.join(os.environ["SWA_CUDA_BUILD"], src); os.makedirs(kw["build_directory"], exist_ok=True)
                _CUDA[which] = load(name=src + "_ext", sources=[os.path.join(d, src + ".cu")], extra_include_paths=[d],
                                    extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math"], verbose=False, **kw)
        except Exception as ex:                             # noqa: BLE001 -- fall back to Triton
            _CUDA["err_" + which] = repr(ex)
    return _CUDA[which]


def _ffn_bwd_cuda_ok(C, NHID, dtype):
    return os.environ.get("SWA_FFN_BWD", "cuda") == "cuda" and C == 128 and NHID == 256 and dtype == torch.bfloat16 and _cuda_ext("bwd") is not None


def _ffn_fwd_cuda_ok(C, NHID, dtype):
    return os.environ.get("SWA_FFN_FWD", "cuda") == "cuda" and C == 128 and NHID == 256 and dtype == torch.bfloat16 and _cuda_ext("fwd") is not None


def _qkvg_fwd_cuda_ok(C, H, dtype):
    return os.environ.get("SWA_QKVG_FWD", "cuda") == "cuda" and C == 128 and H == 4 and dtype == torch.bfloat16 and _cuda_ext("qkvg") is not None


def _pack_ffn64(wu):
    """rows per 64-wide hidden chunk j: [Wu[64j..64j+63] ; Wu[NHID + 64j..]] (the forward kernel's a|b tile)."""
    NH = wu.shape[0] // 2; C = wu.shape[1]
    return wu.view(2, NH // 64, 64, C).permute(1, 0, 2, 3).reshape(2 * NH, C).contiguous()


def _pack_ffn(wu, wd):
    """wab: per 32-wide hidden chunk j rows [Wu[32j..] ; Wu[NHID + 32j..]]; wdt = Wd^T; wabt = wab^T."""
    NH = wd.shape[1]; C = wd.shape[0]
    wab = wu.view(2, NH // 32, 32, C).permute(1, 0, 2, 3).reshape(2 * NH, C).contiguous()
    return wab, wd.t().contiguous(), wab.t().contiguous()


FFN_DW = os.environ.get("SWA_FFN_DW", "mat")   # "fused": _ffn_dw recompute kernel (correct, slower in Triton: 255 regs)
_SMS = torch.cuda.get_device_properties(0).multi_processor_count if torch.cuda.is_available() else 132
DQ1_DTYPE = torch.bfloat16 if os.environ.get("SWA_DQ1", "bf16") == "bf16" else torch.float32


def block_bwd(dy, q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, sv, half_window=64, eps=FP32_EPS):
    """Backward of block_fwd.  dy [N, S, C] bf16.  Returns dq [N,S,C] bf16, dmod [B*S, 6C] fp32, dWqkv, dWg, dWo, dWu, dWd (bf16)."""
    N, S, C = q.shape; H = 4; D = C // H; M = N * S; A = N // B; NHID = wd.shape[1]
    dev = q.device; sb = _bucket(M)
    dmod = torch.zeros(B * S, 6 * C, device=dev)
    dq1 = torch.empty(M, C, device=dev, dtype=DQ1_DTYPE); dO = torch.empty(M, C, device=dev, dtype=torch.bfloat16); dG = torch.empty_like(dO)
    Dv = torch.empty(N, H, S, device=dev)
    fused_dw = FFN_DW == "fused"
    if fused_dw:
        dab = hh = dffn = dO
    else:
        dab = torch.empty(M, 2 * NHID, device=dev, dtype=torch.bfloat16); hh = torch.empty(M, NHID, device=dev, dtype=torch.bfloat16)
        dffn = torch.empty_like(dO)
    datt = torch.empty_like(dO); gated = torch.empty_like(dO)
    grid_t = lambda m: (triton.cdiv(S, m["AT"]), triton.cdiv(A, m["SP"]), B)
    if _ffn_bwd_cuda_ok(C, NHID, q.dtype) and not fused_dw:
        dq1, dffn, hh, dab = _cuda_ext("bwd").ffn_bwd(dy.reshape(M, C), sv["q1"], sv["Y"], sv["FF"], mod, dmod, *_pack_ffn(wu, wd), A, B, S, eps, 8 if A % 8 == 0 else 0)
    else:
        LAST["ffn_bwd"] = _ffn_bwd[grid_t](dy.reshape(M, C), sv["q1"], mod, wu, wd, sv["Y"], sv["FF"], dq1, dab, hh, dffn, dmod, S, A, B, eps, sb,
                                           C=C, NHID=NHID, MODW=6 * C, DWOPS=not fused_dw)
        if fused_dw:
            dWu32 = torch.zeros(2 * NHID, C, device=dev); dWd32 = torch.zeros(C, NHID, device=dev)
            ns = lambda m: max(1, min(triton.cdiv(M, m["BR"]), 2 * _SMS // (NHID // m["HS"])))
            LAST["ffn_dw"] = _ffn_dw[lambda m: (ns(m), NHID // m["HS"])](dy.reshape(M, C), mod, wu, wd, sv["Y"], dWu32, dWd32, M, S, B, sb, 0,
                                                                      C=C, NHID=NHID, MODW=6 * C)
    LAST["oproj_bwd"] = _oproj_bwd[grid_t](dq1, sv["O"], sv["G"], mod, wo, sv["Att"], dO, dG, Dv, datt, gated, dmod, S, A, B, sb, C=C, H=H, D=D, MODW=6 * C)
    dQh = torch.empty(N, H, S, D, device=dev, dtype=torch.bfloat16); dKh = torch.empty_like(dQh); dVh = torch.empty_like(dQh)
    LAST["dq"] = _attn_bwd_dq[lambda m: (triton.cdiv(S, m["BM"]), N * H)](sv["Qh"], sv["Kh"], sv["Vh"], dO, sv["lse"], Dv, seqused, dQh, S, D ** -0.5, sb,
                                                                          C=C, H=H, D=D, HW=half_window)
    LAST["dkv"] = _attn_bwd_dkv[lambda m: (triton.cdiv(S, m["BN"]), N * H)](sv["Qh"], sv["Kh"], sv["Vh"], dO, sv["lse"], Dv, seqused, dKh, dVh, S,
                                                                            D ** -0.5, sb, C=C, H=H, D=D, HW=half_window)
    dq = torch.empty(M, C, device=dev, dtype=torch.bfloat16); dP = torch.empty(M, 4 * C, device=dev, dtype=torch.bfloat16)
    LAST["qkvg_bwd"] = _qkvg_bwd[grid_t](q.reshape(M, C), mod, cos, sin, wqkv, wg, sv["PQ"], sv["PK"], dQh, dKh, dVh, dG, dq1, dq, dP, dmod, S, A, B, eps,
                                         FP32_EPS, sb, C=C, H=H, D=D, MODW=6 * C)
    xn = sv["X"]
    dWqkv = dP[:, :3 * C].t() @ xn; dWg = dP[:, 3 * C:].t() @ xn
    dWo = datt.t() @ gated
    if fused_dw:
        dWu = dWu32.to(wu.dtype); dWd = dWd32.to(wd.dtype)
    else:
        dWu = dab.t() @ sv["Y"]; dWd = dffn.t() @ hh
    return dq.view(N, S, C), dmod, dWqkv, dWg, dWo, dWu, dWd


class SWABlockFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window):
        train = any(ctx.needs_input_grad)
        q = q.contiguous()
        if not train:
            return block_fwd(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window)
        out, sv = block_fwd(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window, save=True)
        ctx.sv_keys = list(sv.keys())
        ctx.save_for_backward(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, *sv.values())
        ctx.B, ctx.hw = B, half_window
        return out

    @staticmethod
    def backward(ctx, dy):
        t = ctx.saved_tensors
        q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd = t[:10]
        sv = dict(zip(ctx.sv_keys, t[10:]))
        dq, dmod, dWqkv, dWg, dWo, dWu, dWd = block_bwd(dy.contiguous(), q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, ctx.B, sv, ctx.hw)
        return dq, dmod, None, None, None, dWqkv, dWg, dWo, dWu, dWd, None, None


def swa_block(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window=64):
    """Differentiable fused ESMFold2 SWA atom block.  q [N, S, C] bf16, mod [B*S, 6C] fp32 (differentiable; hoisted modulation)."""
    return SWABlockFn.apply(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window)
