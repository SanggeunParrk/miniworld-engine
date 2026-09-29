"""Backward Triton kernels of the fused ESMFold2 SWA atom DiT block (kernel family ``swa_dit``), bf16.

Moved from team-gm ``src/team_gm/modules/blocks/swa_fused_triton.py`` (commit 14f2c73; copied from team-gm 4fafa83). The
kernel bodies are byte-for-byte the team-gm ones; see ``triton/forward.py`` for what changed around them (``_swa_``
symbol prefix, ``SB`` -> ``shape_key``, configs from ``configs_for``). ``restore_value`` / ``reset_to_zero`` are kept.

The three row-tiled kernels that reduce the (augment-invariant) modulation gradient take one more thing than team-gm's
decorators: an ``early_config_prune`` that drops a tile of fewer than 16 rows (``SP * AT`` is the M extent of every
``tl.dot`` there). No config team-gm enumerated -- i.e. nothing in ``configs/default`` -- is below it; it only keeps the
Cartesian ``configs/grid`` ladders from offering tiles ``tl.dot`` cannot compile.
"""
import triton
import triton.language as tl

from miniworld_engine.autotune.configs import configs_for


def _rows_at_least_16(configs, named_args, **kwargs):
    """``early_config_prune``: SP augments x AT atoms is the row extent of every ``tl.dot`` in the kernel, which needs 16.
    A correctness exclusion only; performance is the autotuner's to decide."""
    keep = [c for c in configs if c.kwargs["SP"] * c.kwargs["AT"] >= 16]
    if not keep:
        raise ValueError("no config has SP * AT >= 16 rows")
    return keep


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


# NH=32 with 8 warps, SP=2, AT=32 faults (Triton codegen) -- team-gm's note; no config set offers NH=32.
@triton.autotune(configs=configs_for("swa_dit_swiglu_bwd_triton"), key=["shape_key", "DWOPS"], restore_value=["DMOD"],
                 prune_configs_by={"early_config_prune": _rows_at_least_16})
@triton.jit(do_not_specialize=["S", "A", "B"])
def _swa_ffn_bwd_kernel(DQ2, Q1, MOD, WU, WD, YS, FFS, DQ1, DAB, HH, DFFN, DMOD, S, A, B, eps, shape_key,
                        C: tl.constexpr, NHID: tl.constexpr, MODW: tl.constexpr, DWOPS: tl.constexpr, SP: tl.constexpr, AT: tl.constexpr,
                        NH: tl.constexpr):
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


@triton.autotune(configs=configs_for("swa_dit_swiglu_dw_triton"), key=["shape_key"], reset_to_zero=["DWU", "DWD"])
@triton.jit(do_not_specialize=["M", "S", "B"])
def _swa_ffn_dw_kernel(DQ2, MOD, WU, WD, YS, DWU, DWD, M, S, B, shape_key, NSPLIT,
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


@triton.autotune(configs=configs_for("swa_dit_output_bwd_triton"), key=["shape_key"], restore_value=["DMOD"],
                 prune_configs_by={"early_config_prune": _rows_at_least_16})
@triton.jit(do_not_specialize=["S", "A", "B"])
def _swa_oproj_bwd_kernel(DQ1, O, G, MOD, WO, ATS, DOUT_, DG, DVv, DATT, GATED, DMOD, S, A, B, shape_key,
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


@triton.autotune(configs=configs_for("swa_dit_softmax_bwd_dq_triton"), key=["shape_key", "HW"])
@triton.jit(do_not_specialize=["S"])
def _swa_attn_bwd_dq_kernel(QH, KH, VH, DOUT_, LSE, DVv, SEQU, DQH, S, scale, shape_key,
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


@triton.autotune(configs=configs_for("swa_dit_softmax_bwd_dkdv_triton"), key=["shape_key", "HW"])
@triton.jit(do_not_specialize=["S"])
def _swa_attn_bwd_dkv_kernel(QH, KH, VH, DOUT_, LSE, DVv, SEQU, DKH, DVH, S, scale, shape_key,
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


@triton.autotune(configs=configs_for("swa_dit_inproj_bwd_triton"), key=["shape_key"], restore_value=["DMOD"],
                 prune_configs_by={"early_config_prune": _rows_at_least_16})
@triton.jit(do_not_specialize=["S", "A", "B"])
def _swa_qkvg_bwd_kernel(QI, MOD, COS, SIN, WQKV, WG, PQS, PKS, DQH, DKH, DVH, DG, DQ1, DQ, DP, DMOD, S, A, B, eps, qk_eps, shape_key,
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
