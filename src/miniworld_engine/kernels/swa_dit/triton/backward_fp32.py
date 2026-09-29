"""fp32 backward Triton kernels of the fused SWA atom DiT block (kernel family ``swa_dit``).

The fp32 twins of ``backward.py``'s FFN, out-projection and qkvg kernels; the attention backward is
``backward._swa_attn_bwd_dq_kernel`` / ``_swa_attn_bwd_dkv_kernel`` unchanged, on bf16 operands, as the per-op path's
FlashAttention-4 backward runs it (bf16 q / k / v / dO, fp32 accumulate, bf16 dq / dk / dv widened afterwards). Every
elementwise step and every gradient that leaves a kernel is fp32, except the two that FA4 hands over in bf16: dO (the
attention output's gradient, cast to bf16 before the attention backward) and dQ / dK / dV.

GEMM precision: the forward's group constants (``forward_fp32.QKVG_DOT`` / ``OUT_DOT`` / ``FFN_DOT``), for the measured
reasons given there:

  FFN: a, b recomputed from y (y Wu^T), dh = dffn Wd, dy = da Wu_a + db Wu_b     FFN_DOT  = "tf32x3"
  out-projection: dgated = datt Wo                                              OUT_DOT  = "tf32"
  qkvg: dx = dq Wq + dk Wk + dv Wv + dg Wg                                      QKVG_DOT = "tf32"
  weight gradients (dWqkv, dWg, dWo, dWu, dWd)   torch GEMMs on the saved fp32 operands in ``dispatch``, so they follow
                                                 the process's matmul precision exactly as the per-op path's autograd does.

The FFN backward always materialises [da | db] and h for cuBLAS weight gradients (``swa_dit_ffn_dw="mat"``); the fused-dW
variant is bf16-only.
"""
import triton
import triton.language as tl

from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.kernels.swa_dit.triton.backward import (
    _head_rms_bwd,
    _presum_add,
    _rows_at_least_16,
    _tile_rows,
    _unrope,
)
from miniworld_engine.kernels.swa_dit.triton.forward_fp32 import FFN_DOT, OUT_DOT, QKVG_DOT


@triton.autotune(configs=configs_for("swa_dit_swiglu_bwd_fp32_triton"), key=["shape_key"], restore_value=["DMOD"],
                 prune_configs_by={"early_config_prune": _rows_at_least_16})
@triton.jit(do_not_specialize=["S", "A", "B"])
def _swa_ffn_bwd_fp32_kernel(DQ2, Q1, MOD, WU, WD, YS, FFS, DQ1, DAB, HH, DFFN, DMOD, S, A, B, eps, shape_key,
                             C: tl.constexpr, NHID: tl.constexpr, MODW: tl.constexpr, SP: tl.constexpr, AT: tl.constexpr,
                             NH: tl.constexpr):
    """FFN backward from the saved y and FFN output: dq1 = dq2 + RMSNorm/adaLN backward; [da | db], h and dffn for the
    weight gradients; d shift_f / scale_f / gate_f pre-summed over the tile's augments."""
    rows, ok, mrow, s_ = _tile_rows(S, A, B, SP, AT)
    b = tl.program_id(2).to(tl.int64); ab_idx = tl.program_id(0).to(tl.int64)
    ok_at = (ab_idx * AT + tl.arange(0, AT)) < S
    cc = tl.arange(0, C).to(tl.int64)
    m2 = ok[:, None]
    ofs = rows[:, None] * C + cc[None, :]
    dq2 = tl.load(DQ2 + ofs, mask=m2, other=0.0)
    dffn = dq2 * tl.load(MOD + mrow[:, None] * MODW + 5 * C + cc[None, :], mask=m2, other=0.0)
    tl.store(DFFN + ofs, dffn, mask=m2)
    _presum_add(DMOD, tl.where(m2, dq2 * tl.load(FFS + ofs, mask=m2, other=0.0), 0.0), 5 * C, ab_idx, b, S, ok_at, C, MODW, SP, AT)
    nh = tl.arange(0, NH).to(tl.int64)
    dy = tl.zeros((SP * AT, C), dtype=tl.float32)
    y = tl.load(YS + ofs, mask=m2, other=0.0)
    for j0 in range(0, NHID, NH):
        hid = j0 + nh
        a = tl.dot(y, tl.load(WU + hid[None, :] * C + cc[:, None]), input_precision=FFN_DOT)          # recomputed
        bb = tl.dot(y, tl.load(WU + (NHID + hid)[None, :] * C + cc[:, None]), input_precision=FFN_DOT)
        sa = tl.sigmoid(a)
        tl.store(HH + rows[:, None] * NHID + hid[None, :], a * sa * bb, mask=m2)
        dh = tl.dot(dffn, tl.load(WD + cc[:, None] * NHID + hid[None, :]), input_precision=FFN_DOT)    # [R][NH] = dffn Wd[:, hid]
        da = dh * bb * sa * (1.0 + a * (1.0 - sa))
        db = dh * a * sa
        dy += tl.dot(da, tl.load(WU + hid[:, None] * C + cc[None, :]), input_precision=FFN_DOT) + \
            tl.dot(db, tl.load(WU + (NHID + hid)[:, None] * C + cc[None, :]), input_precision=FFN_DOT)
        tl.store(DAB + rows[:, None] * (2 * NHID) + hid[None, :], da, mask=m2)
        tl.store(DAB + rows[:, None] * (2 * NHID) + NHID + hid[None, :], db, mask=m2)
    q1 = tl.load(Q1 + ofs, mask=m2, other=0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(q1 * q1, axis=1) / C + eps)
    xh = q1 * rstd[:, None]
    _presum_add(DMOD, tl.where(m2, dy * xh, 0.0), 4 * C, ab_idx, b, S, ok_at, C, MODW, SP, AT)                                    # d scale_f
    _presum_add(DMOD, tl.where(m2, dy, 0.0), 3 * C, ab_idx, b, S, ok_at, C, MODW, SP, AT)                                         # d shift_f
    dxh = dy * (1.0 + tl.load(MOD + mrow[:, None] * MODW + 4 * C + cc[None, :], mask=m2, other=0.0))
    tl.store(DQ1 + ofs, dq2 + rstd[:, None] * (dxh - xh * (tl.sum(dxh * xh, axis=1) / C)[:, None]), mask=m2)


@triton.autotune(configs=configs_for("swa_dit_output_bwd_fp32_triton"), key=["shape_key"], restore_value=["DMOD"],
                 prune_configs_by={"early_config_prune": _rows_at_least_16})
@triton.jit(do_not_specialize=["S", "A", "B"])
def _swa_oproj_bwd_fp32_kernel(DQ1, O, G, MOD, WO, ATS, DOUT_, DG, DVv, DATT, GATED, DMOD, S, A, B, shape_key,
                               C: tl.constexpr, H: tl.constexpr, D: tl.constexpr, MODW: tl.constexpr, SP: tl.constexpr,
                               AT: tl.constexpr):
    """q1 = q + ga * ((sigmoid(g) * O) Wo^T): dO (bf16, the attention backward's operand), dG, D = per-head
    rowsum(dO * O), d gate_a, and the dWo operands datt / gated (fp32)."""
    rows, ok, mrow, s_ = _tile_rows(S, A, B, SP, AT)
    b = tl.program_id(2).to(tl.int64); ab_idx = tl.program_id(0).to(tl.int64)
    ok_at = (ab_idx * AT + tl.arange(0, AT)) < S
    cc = tl.arange(0, C).to(tl.int64)
    m2 = ok[:, None]
    ofs = rows[:, None] * C + cc[None, :]
    dq1 = tl.load(DQ1 + ofs, mask=m2, other=0.0)
    g = tl.load(G + ofs, mask=m2, other=0.0); so = tl.sigmoid(g)
    o = tl.load(O + ofs, mask=m2, other=0.0).to(tl.float32)
    gated = so * o
    tl.store(GATED + ofs, gated, mask=m2)
    att = tl.load(ATS + ofs, mask=m2, other=0.0)
    ga = tl.load(MOD + mrow[:, None] * MODW + 2 * C + cc[None, :], mask=m2, other=0.0)
    _presum_add(DMOD, tl.where(m2, dq1 * att, 0.0), 2 * C, ab_idx, b, S, ok_at, C, MODW, SP, AT)    # d gate_a
    datt = dq1 * ga
    tl.store(DATT + ofs, datt, mask=m2)
    dgated = tl.dot(datt, tl.load(WO + cc[:, None] * C + cc[None, :]), input_precision=OUT_DOT)   # datt Wo
    do_ = (dgated * so).to(tl.bfloat16)
    tl.store(DOUT_ + ofs, do_, mask=m2)
    tl.store(DG + ofs, dgated * o * so * (1.0 - so), mask=m2)
    dvv = tl.sum(tl.reshape(do_.to(tl.float32) * o, (SP * AT, H, D)), axis=2)
    hh_ = tl.arange(0, H).to(tl.int64)
    n = rows // S
    tl.store(DVv + (n[:, None] * H + hh_[None, :]) * S + s_[:, None], dvv, mask=m2)


@triton.autotune(configs=configs_for("swa_dit_inproj_bwd_fp32_triton"), key=["shape_key"], restore_value=["DMOD"],
                 prune_configs_by={"early_config_prune": _rows_at_least_16})
@triton.jit(do_not_specialize=["S", "A", "B"])
def _swa_qkvg_bwd_fp32_kernel(QI, MOD, COS, SIN, WQKV, WG, PQS, PKS, DQH, DKH, DVH, DG, DQ1, DQ, DP, DMOD, S, A, B, eps, qk_eps,
                              shape_key, C: tl.constexpr, H: tl.constexpr, D: tl.constexpr, MODW: tl.constexpr, SP: tl.constexpr,
                              AT: tl.constexpr):
    """From the bf16 dQ / dK / dV of the attention backward and dG: back through RoPE and the per-head RMSNorm (on the
    saved fp32 pre-norm projections), the four projections, and the adaLN RMSNorm; dP [N*S, 4C] for the weight
    gradients, d shift_a / scale_a, dq = dq1 + the RMSNorm backward (fp32)."""
    rows, ok, mrow, s_ = _tile_rows(S, A, B, SP, AT)
    b = tl.program_id(2).to(tl.int64); ab_idx = tl.program_id(0).to(tl.int64)
    ok_at = (ab_idx * AT + tl.arange(0, AT)) < S
    R: tl.constexpr = SP * AT
    cc = tl.arange(0, C).to(tl.int64)
    m2 = ok[:, None]
    ofs = rows[:, None] * C + cc[None, :]
    q = tl.load(QI + ofs, mask=m2, other=0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(q * q, axis=1) / C + eps)
    xh = q * rstd[:, None]
    scale = tl.load(MOD + mrow[:, None] * MODW + C + cc[None, :], mask=m2, other=0.0)
    HALF: tl.constexpr = D // 2
    hc = tl.arange(0, HALF).to(tl.int64)
    cs = tl.load(COS + mrow[:, None] * HALF + hc[None, :], mask=m2, other=1.0)
    sn = tl.load(SIN + mrow[:, None] * HALF + hc[None, :], mask=m2, other=0.0)
    n = rows // S
    hm = ((n[:, None] * H + (cc[None, :] // D)) * S + s_[:, None]) * D + (cc[None, :] % D)
    pq = tl.load(PQS + ofs, mask=m2, other=0.0)
    dpq = _head_rms_bwd(pq, _unrope(tl.load(DQH + hm, mask=m2, other=0.0).to(tl.float32), cs, sn, R, H, D), qk_eps, R, H, D)
    tl.store(DP + rows[:, None] * (4 * C) + cc[None, :], dpq, mask=m2)
    dx = tl.dot(dpq, tl.load(WQKV + cc[:, None] * C + cc[None, :]), input_precision=QKVG_DOT)
    pk = tl.load(PKS + ofs, mask=m2, other=0.0)
    dpk = _head_rms_bwd(pk, _unrope(tl.load(DKH + hm, mask=m2, other=0.0).to(tl.float32), cs, sn, R, H, D), qk_eps, R, H, D)
    tl.store(DP + rows[:, None] * (4 * C) + C + cc[None, :], dpk, mask=m2)
    dx += tl.dot(dpk, tl.load(WQKV + C * C + cc[:, None] * C + cc[None, :]), input_precision=QKVG_DOT)
    dpv = tl.load(DVH + hm, mask=m2, other=0.0).to(tl.float32)
    tl.store(DP + rows[:, None] * (4 * C) + 2 * C + cc[None, :], dpv, mask=m2)
    dx += tl.dot(dpv, tl.load(WQKV + 2 * C * C + cc[:, None] * C + cc[None, :]), input_precision=QKVG_DOT)
    dpg = tl.load(DG + ofs, mask=m2, other=0.0)
    tl.store(DP + rows[:, None] * (4 * C) + 3 * C + cc[None, :], dpg, mask=m2)
    dx += tl.dot(dpg, tl.load(WG + cc[:, None] * C + cc[None, :]), input_precision=QKVG_DOT)
    _presum_add(DMOD, tl.where(m2, dx * xh, 0.0), C, ab_idx, b, S, ok_at, C, MODW, SP, AT)          # d scale_a
    _presum_add(DMOD, tl.where(m2, dx, 0.0), 0, ab_idx, b, S, ok_at, C, MODW, SP, AT)               # d shift_a
    dxh = dx * (1.0 + scale)
    dq = tl.load(DQ1 + ofs, mask=m2, other=0.0) + rstd[:, None] * (dxh - xh * (tl.sum(dxh * xh, axis=1) / C)[:, None])
    tl.store(DQ + ofs, dq, mask=m2)
