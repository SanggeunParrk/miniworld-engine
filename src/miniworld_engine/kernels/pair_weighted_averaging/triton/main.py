"""MSAPairWeightedAveraging in Triton, forward and backward -- the portable path (any GPU Triton supports).

The module's statements: y = LN_m(msa); v = y Wv^T; logits = LN_z(pair) Wb^T (masked keys -> bf16 min); w = softmax_j;
o_h = w_h v_h; u = sigmoid(y Wg^T) * o; out = msa + dropout(u Wo^T). The decomposition keeps the two big contractions in cuBLAS:

  forward   layernorm_gemm_softmax  per row i: LN_z -> Wb -> key mask -> softmax over j (two passes)         w [NH, L, L] bf16
            layernorm_gemm          LN_m -> Wv, written head-major                                          v [NH, L, S*C]
            o                       o_h = w_h v_h  (cuBLAS)                                                 o [NH, L, S*C]
            gate_gemm               LN_m -> gate, u = sigmoid(g) o, out-projection summed over heads, dropout, residual
  backward  bwd_gate_gemm           per (head, tile range): g recomputed, du = dout' Wo_h, do = du g (head-major),
                                    dgp = du o g (1 - g); dWo_h, dWg_h in registers
            dv, dw                  dv_h = w_h^T do_h, dw_h = do_h v_h^T  (cuBLAS)
            bwd_layernorm_gemm_dw   per head: dWv_h = dv_h^T y
            bwd_layernorm_gemm_dx_dlnw  dy = dgp Wg + dv Wv, LayerNorm backward + dres -> dmsa, dgamma_m, dbeta_m
            bwd_layernorm_gemm_softmax  dlogit = w (dw - sum_j w dw) (0 on masked keys), dWb, LN_z backward -> dpair, dgamma_z, dbeta_z

Dropout: the module's Dropout(broadcast_dim=1) keeps one mask per (token, channel) shared over the MSA rows; the caller draws that
[L, D] keep-mask and the kernels apply x * keep / (1 - p).

Every tile, warp count, stage count and persistent program count is an autotune axis (``autotune/configs/grid/<op>.csv``);
the constants in the kernels are the model dimensions (constexpr). Offsets into the head-major [NH, L, S C] buffers and the
[S L, NH C] buffers are 64-bit. d_msa and d_hidden must be powers of two >= 16 (spanned by one tile); d_pair is padded to
the next power of two inside the pair kernels, so any d_pair >= 16 is served.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import length_of, token_key
from miniworld_engine.kernels._compile import opaque

NEG = tl.constexpr(-3.3895313892515355e38)   # torch.finfo(torch.bfloat16).min: the module's masked_fill value


def _divides(**axes):
    """``early_config_prune``: keep the configs whose tile axis ``ax`` divides the kernel argument ``axes[ax]`` names (a tile over
    an unmasked extent must divide it). A correctness exclusion only; performance is the autotuner's to decide."""
    def prune(configs, named_args, **kw):
        args = {**named_args, **kw}
        keep = [c for c in configs if all(args[dim] % c.kwargs[ax] == 0 for ax, dim in axes.items())]
        if not keep:
            raise ValueError(f"no config tiles {', '.join(f'{d}={args[d]}' for d in axes.values())}")
        return keep
    return prune


def _nsm(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


# ---------------------------------------------------------------------------------------------------- forward kernels
@triton.autotune(configs=configs_for("pair_weighted_averaging_layernorm_gemm_softmax_triton"),
                 key=["shape_key", "DZP", "HP", "HAS_MASK"])   # DZP / HP: padding implied by DZ / NH (the key folds <= 3 axes)
@triton.jit
def _pwa_layernorm_gemm_softmax_kernel(Z, MASK, LNW, LNB, WB, W, L, eps, shape_key,
                                       DZ: tl.constexpr, DZP: tl.constexpr, NH: tl.constexpr, HP: tl.constexpr,
                                       HAS_MASK: tl.constexpr, BJ: tl.constexpr):
    # one program per query row i: pass 0 keeps the running max / sum per head, pass 1 writes the normalized weights
    i = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, DZP)
    cm = c < DZ
    hh = tl.arange(0, HP)
    gz = tl.load(LNW + c, mask=cm, other=0.0)
    bz = tl.load(LNB + c, mask=cm, other=0.0)
    wbt = tl.load(WB + hh[None, :] * DZ + c[:, None], mask=(hh < NH)[None, :] & cm[:, None], other=0.0)   # [DZP, HP] = Wb^T, padded
    m_run = tl.full((HP,), -float("inf"), tl.float32)
    l_run = tl.zeros((HP,), tl.float32)
    for pas in tl.static_range(2):
        for j0 in range(0, L, BJ):
            j = j0 + tl.arange(0, BJ)
            jm = j < L
            z = tl.load(Z + (i * L + j)[:, None] * DZ + c[None, :], mask=jm[:, None] & cm[None, :], other=0.0).to(tl.float32)
            mu = tl.sum(z, axis=1) / DZ
            zc = tl.where(cm[None, :], z - mu[:, None], 0.0)
            rs = tl.rsqrt(tl.sum(zc * zc, axis=1) / DZ + eps)
            zy = (zc * rs[:, None] * gz[None, :] + bz[None, :]).to(tl.bfloat16)
            lg = tl.dot(zy, wbt)                                                              # [BJ, HP]
            if HAS_MASK:
                km = tl.load(MASK + j, mask=jm, other=0) != 0
                lg = tl.where(km[:, None], lg, NEG)
            lg = tl.where(jm[:, None], lg, -float("inf"))
            if pas == 0:
                m_new = tl.maximum(m_run, tl.max(lg, axis=0))
                l_run = l_run * tl.exp(m_run - m_new) + tl.sum(tl.exp(lg - m_new[None, :]), axis=0)
                m_run = m_new
            else:
                pr = tl.exp(lg - m_run[None, :]) / l_run[None, :]
                tl.store(W + (hh[None, :] * L + i) * L + j[:, None], pr.to(tl.bfloat16), mask=jm[:, None] & (hh < NH)[None, :])


@triton.autotune(configs=configs_for("pair_weighted_averaging_layernorm_gemm_triton"), key=["shape_key"],
                 prune_configs_by={"early_config_prune": _divides(BN="HC")})
@triton.jit
def _pwa_layernorm_gemm_kernel(M, LNW, LNB, WV, V, S, L, eps, shape_key,
                               D: tl.constexpr, HC: tl.constexpr, C: tl.constexpr, BS: tl.constexpr, BN: tl.constexpr):
    # program = (token j, block of BS MSA rows, block of BN of the NH C outputs): v[h][j][s C + c] = LN(msa[s, j]) . Wv[h C + c]
    j = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1).to(tl.int64) * BS + tl.arange(0, BS)
    sm = s < S
    k = tl.arange(0, D)
    x = tl.load(M + (s * L + j)[:, None] * D + k[None, :], mask=sm[:, None], other=0.0).to(tl.float32)
    mu = tl.sum(x, axis=1) / D
    xc = x - mu[:, None]
    rs = tl.rsqrt(tl.sum(xc * xc, axis=1) / D + eps)
    y = (xc * rs[:, None] * tl.load(LNW + k)[None, :] + tl.load(LNB + k)[None, :]).to(tl.bfloat16)
    n = tl.program_id(2) * BN + tl.arange(0, BN)
    wvt = tl.load(WV + n[None, :] * D + k[:, None])                                           # [D, BN]
    v = tl.dot(y, wvt).to(tl.bfloat16)                                                        # [BS, BN]
    h = n // C
    cc = n % C
    tl.store(V + (h[None, :].to(tl.int64) * L + j) * (S * C) + s[:, None] * C + cc[None, :], v, mask=sm[:, None])


@triton.autotune(configs=configs_for("pair_weighted_averaging_gate_gemm_triton"), key=["shape_key", "HAS_KEEP"])
@triton.jit
def _pwa_gate_gemm_kernel(M, O, LNW, LNB, WG, WO, KEEP, OUT, S, L, eps, scale, shape_key,
                          D: tl.constexpr, NH: tl.constexpr, C: tl.constexpr, HAS_KEEP: tl.constexpr, BS: tl.constexpr):
    # program = (token i, BS MSA rows): out = msa + keep / (1 - p) * sum_h (sigmoid(y Wg_h^T) * o_h) Wo_h^T (runtime head loop)
    i = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1).to(tl.int64) * BS + tl.arange(0, BS)
    sm = s < S
    k = tl.arange(0, D)
    cc = tl.arange(0, C)
    rowp = (s * L + i)[:, None] * D + k[None, :]
    x = tl.load(M + rowp, mask=sm[:, None], other=0.0).to(tl.float32)
    mu = tl.sum(x, axis=1) / D
    xc = x - mu[:, None]
    rs = tl.rsqrt(tl.sum(xc * xc, axis=1) / D + eps)
    y = (xc * rs[:, None] * tl.load(LNW + k)[None, :] + tl.load(LNB + k)[None, :]).to(tl.bfloat16)
    acc = tl.zeros((BS, D), tl.float32)
    for h in range(NH):
        wgt = tl.load(WG + (h * C + cc)[None, :] * D + k[:, None])                           # [D, C]
        g = tl.sigmoid(tl.dot(y, wgt))
        o = tl.load(O + (h * L + i) * (S * C) + s[:, None] * C + cc[None, :], mask=sm[:, None], other=0.0).to(tl.float32)
        u = (g * o).to(tl.bfloat16)
        wot = tl.load(WO + k[None, :] * (NH * C) + h * C + cc[:, None])                      # [C, D] = Wo_h^T
        acc += tl.dot(u, wot)
    if HAS_KEEP:
        acc = acc * tl.load(KEEP + i * D + k).to(tl.float32)[None, :] * scale
    tl.store(OUT + rowp, (x + acc).to(tl.bfloat16), mask=sm[:, None])


# ---------------------------------------------------------------------------------------------------- backward kernels
@triton.autotune(configs=configs_for("pair_weighted_averaging_bwd_gate_gemm_triton"), key=["shape_key", "HAS_KEEP"],
                 reset_to_zero=["DWO", "DWG"])
@triton.jit
def _pwa_bwd_gate_gemm_kernel(M, DR, O, LNW, LNB, WG, WO, KEEP, DO, DGP, DWO, DWG, S, L, eps, scale, shape_key,
                              D: tl.constexpr, NH: tl.constexpr, C: tl.constexpr, HAS_KEEP: tl.constexpr, BS: tl.constexpr,
                              PPS: tl.constexpr):
    # persistent over (token, BS rows) tiles, one head per grid column; dWo_h / dWg_h stay in registers until the end
    pid = tl.program_id(0)
    h = tl.program_id(1)
    nprog = tl.num_programs(0)
    k = tl.arange(0, D)
    cc = tl.arange(0, C)
    gam = tl.load(LNW + k)
    bet = tl.load(LNB + k)
    wgt = tl.load(WG + (h * C + cc)[None, :] * D + k[:, None])                               # [D, C]
    woh = tl.load(WO + k[:, None] * (NH * C) + h * C + cc[None, :])                          # [D, C] = Wo[:, h]
    dwo = tl.zeros((D, C), tl.float32)
    dwg = tl.zeros((C, D), tl.float32)
    ntiles = L * tl.cdiv(S, BS)
    for t in range(pid, ntiles, nprog):
        i = (t % L).to(tl.int64)
        s = (t // L).to(tl.int64) * BS + tl.arange(0, BS)
        sm = s < S
        rowp = (s * L + i)[:, None] * D + k[None, :]
        x = tl.load(M + rowp, mask=sm[:, None], other=0.0).to(tl.float32)
        mu = tl.sum(x, axis=1) / D
        xc = x - mu[:, None]
        rs = tl.rsqrt(tl.sum(xc * xc, axis=1) / D + eps)
        y = (xc * rs[:, None] * gam[None, :] + bet[None, :]).to(tl.bfloat16)
        dr = tl.load(DR + rowp, mask=sm[:, None], other=0.0).to(tl.float32)
        if HAS_KEEP:
            dr = dr * tl.load(KEEP + i * D + k).to(tl.float32)[None, :] * scale
        drb = dr.to(tl.bfloat16)
        g = tl.sigmoid(tl.dot(y, wgt))                                                        # [BS, C]
        hp = (h.to(tl.int64) * L + i) * (S * C) + s[:, None] * C + cc[None, :]
        o = tl.load(O + hp, mask=sm[:, None], other=0.0).to(tl.float32)
        du = tl.dot(drb, woh)                                                                 # [BS, C]
        tl.store(DO + hp, (du * g).to(tl.bfloat16), mask=sm[:, None])
        gp = (du * o * g * (1.0 - g)).to(tl.bfloat16)
        tl.store(DGP + (s * L + i)[:, None] * (NH * C) + h * C + cc[None, :], gp, mask=sm[:, None])
        dwo += tl.dot(tl.trans(drb), (g * o).to(tl.bfloat16))
        dwg += tl.dot(tl.trans(gp), y)
    tl.atomic_add(DWO + k[:, None] * (NH * C) + h * C + cc[None, :], dwo)
    tl.atomic_add(DWG + (h * C + cc)[:, None] * D + k[None, :], dwg)


@triton.autotune(configs=configs_for("pair_weighted_averaging_bwd_layernorm_gemm_dw_triton"), key=["shape_key"], reset_to_zero=["DWV"])
@triton.jit
def _pwa_bwd_layernorm_gemm_dw_kernel(M, DV, LNW, LNB, DWV, S, L, eps, shape_key,
                                      D: tl.constexpr, C: tl.constexpr, BS: tl.constexpr, PPS: tl.constexpr):
    # program = (tile range, head): dWv[h C + c, :] += sum_(s, j) dv[h][j][s C + c] y[s, j, :]
    pid = tl.program_id(0)
    h = tl.program_id(1)
    nprog = tl.num_programs(0)
    k = tl.arange(0, D)
    cc = tl.arange(0, C)
    gam = tl.load(LNW + k)
    bet = tl.load(LNB + k)
    acc = tl.zeros((C, D), tl.float32)
    ntiles = L * tl.cdiv(S, BS)
    for t in range(pid, ntiles, nprog):
        j = (t % L).to(tl.int64)
        s = (t // L).to(tl.int64) * BS + tl.arange(0, BS)
        sm = s < S
        x = tl.load(M + (s * L + j)[:, None] * D + k[None, :], mask=sm[:, None], other=0.0).to(tl.float32)
        mu = tl.sum(x, axis=1) / D
        xc = x - mu[:, None]
        rs = tl.rsqrt(tl.sum(xc * xc, axis=1) / D + eps)
        y = (xc * rs[:, None] * gam[None, :] + bet[None, :]).to(tl.bfloat16)
        dv = tl.load(DV + (h.to(tl.int64) * L + j) * (S * C) + s[:, None] * C + cc[None, :], mask=sm[:, None], other=0.0)
        acc += tl.dot(tl.trans(dv), y)
    tl.atomic_add(DWV + (h * C + cc)[:, None] * D + k[None, :], acc)


@triton.autotune(configs=configs_for("pair_weighted_averaging_bwd_layernorm_gemm_dx_dlnw_triton"), key=["shape_key"],
                 reset_to_zero=["DG", "DB"])
@triton.jit
def _pwa_bwd_layernorm_gemm_dx_dlnw_kernel(M, DR, DGP, DV, LNW, WG, WV, DM, DG, DB, S, L, eps, shape_key,
                                           D: tl.constexpr, NH: tl.constexpr, C: tl.constexpr, BS: tl.constexpr, PPS: tl.constexpr):
    # persistent over (token j, BS MSA rows) tiles: dy = sum_h (dgp_h Wg_h + dv_h Wv_h); LayerNorm backward + dres -> dmsa.
    # One head's [C, D] weight slices per step (not the whole [NH C, D] weights resident): small shared memory on any GPU.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    k = tl.arange(0, D)
    cc = tl.arange(0, C)
    gam = tl.load(LNW + k)
    dgs = tl.zeros((D,), tl.float32)
    dbs = tl.zeros((D,), tl.float32)
    ntiles = L * tl.cdiv(S, BS)
    for t in range(pid, ntiles, nprog):
        j = (t % L).to(tl.int64)
        s = (t // L).to(tl.int64) * BS + tl.arange(0, BS)
        sm = s < S
        rowp = (s * L + j)[:, None] * D + k[None, :]
        x = tl.load(M + rowp, mask=sm[:, None], other=0.0).to(tl.float32)
        mu = tl.sum(x, axis=1) / D
        xc = x - mu[:, None]
        rs = tl.rsqrt(tl.sum(xc * xc, axis=1) / D + eps)
        xh = xc * rs[:, None]
        dy = tl.zeros((BS, D), tl.float32)
        for h in range(NH):
            gp = tl.load(DGP + (s * L + j)[:, None] * (NH * C) + h * C + cc[None, :], mask=sm[:, None], other=0.0)
            dv = tl.load(DV + (h * L + j) * (S * C) + s[:, None] * C + cc[None, :], mask=sm[:, None], other=0.0)
            wgh = tl.load(WG + (h * C + cc)[:, None] * D + k[None, :])                       # [C, D]
            wvh = tl.load(WV + (h * C + cc)[:, None] * D + k[None, :])
            dy += tl.dot(gp, wgh) + tl.dot(dv, wvh)
        dgs += tl.sum(tl.where(sm[:, None], dy * xh, 0.0), axis=0)
        dbs += tl.sum(tl.where(sm[:, None], dy, 0.0), axis=0)
        dxh = dy * gam[None, :]
        m1 = tl.sum(dxh, axis=1) / D
        m2 = tl.sum(dxh * xh, axis=1) / D
        dx = rs[:, None] * (dxh - m1[:, None] - xh * m2[:, None])
        dx += tl.load(DR + rowp, mask=sm[:, None], other=0.0).to(tl.float32)
        tl.store(DM + rowp, dx.to(tl.bfloat16), mask=sm[:, None])
    tl.atomic_add(DG + k, dgs)
    tl.atomic_add(DB + k, dbs)


@triton.autotune(configs=configs_for("pair_weighted_averaging_bwd_layernorm_gemm_softmax_triton"),
                 key=["shape_key", "DZP", "HP", "HAS_MASK"],
                 reset_to_zero=["DWB", "DGZ", "DBZ"])
@triton.jit
def _pwa_bwd_layernorm_gemm_softmax_kernel(Z, W, DW, MASK, LNW, LNB, WB, DZ, DWB, DGZ, DBZ, L, eps, shape_key,
                                           DZC: tl.constexpr, DZP: tl.constexpr, NH: tl.constexpr, HP: tl.constexpr,
                                           HAS_MASK: tl.constexpr, BJ: tl.constexpr):
    # one program per query row i: s_h = sum_j w dw first, then dlogit, dWb, and the pair LayerNorm backward per key block
    i = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, DZP)
    cm = c < DZC
    hh = tl.arange(0, HP)
    hm = hh < NH
    gz = tl.load(LNW + c, mask=cm, other=0.0)
    bz = tl.load(LNB + c, mask=cm, other=0.0)
    wb = tl.load(WB + hh[:, None] * DZC + c[None, :], mask=hm[:, None] & cm[None, :], other=0.0)   # [HP, DZP]
    ssum = tl.zeros((HP,), tl.float32)
    for j0 in range(0, L, BJ):
        j = j0 + tl.arange(0, BJ)
        msk = (j < L)[:, None] & hm[None, :]
        pp = (hh[None, :] * L + i) * L + j[:, None]
        wv = tl.load(W + pp, mask=msk, other=0.0).to(tl.float32)
        dv = tl.load(DW + pp, mask=msk, other=0.0).to(tl.float32)
        ssum += tl.sum(wv * dv, axis=0)
    dwb = tl.zeros((HP, DZP), tl.float32)
    dgs = tl.zeros((DZP,), tl.float32)
    dbs = tl.zeros((DZP,), tl.float32)
    for j0 in range(0, L, BJ):
        j = j0 + tl.arange(0, BJ)
        jm = j < L
        msk = jm[:, None] & hm[None, :]
        pp = (hh[None, :] * L + i) * L + j[:, None]
        wv = tl.load(W + pp, mask=msk, other=0.0).to(tl.float32)
        dv = tl.load(DW + pp, mask=msk, other=0.0).to(tl.float32)
        dl = wv * (dv - ssum[None, :])                                                       # [BJ, HP]
        if HAS_MASK:                                                                         # a masked logit is a constant
            km = tl.load(MASK + j, mask=jm, other=0) != 0
            dl = tl.where(km[:, None], dl, 0.0)
        z = tl.load(Z + (i * L + j)[:, None] * DZC + c[None, :], mask=jm[:, None] & cm[None, :], other=0.0).to(tl.float32)
        mu = tl.sum(z, axis=1) / DZC
        zc = tl.where(cm[None, :], z - mu[:, None], 0.0)
        rs = tl.rsqrt(tl.sum(zc * zc, axis=1) / DZC + eps)
        zh = zc * rs[:, None]
        zy = (zh * gz[None, :] + bz[None, :]).to(tl.bfloat16)
        dlb = dl.to(tl.bfloat16)
        dwb += tl.dot(tl.trans(dlb), zy)
        dzy = tl.dot(dlb, wb.to(tl.bfloat16))                                               # [BJ, DZP]
        dgs += tl.sum(dzy * zh, axis=0)
        dbs += tl.sum(dzy, axis=0)
        dxh = dzy * gz[None, :]
        m1 = tl.sum(dxh, axis=1) / DZC
        m2 = tl.sum(dxh * zh, axis=1) / DZC
        dz = rs[:, None] * (dxh - m1[:, None] - zh * m2[:, None])
        tl.store(DZ + (i * L + j)[:, None] * DZC + c[None, :], dz.to(tl.bfloat16), mask=jm[:, None] & cm[None, :])
    tl.atomic_add(DWB + hh[:, None] * DZC + c[None, :], dwb, mask=hm[:, None] & cm[None, :])
    tl.atomic_add(DGZ + c, dgs, mask=cm)
    tl.atomic_add(DBZ + c, dbs, mask=cm)


# ---------------------------------------------------------------------------------------------------- launches (opaque to Dynamo)
def _pow2_16(n: int) -> bool:
    return n >= 16 and n & (n - 1) == 0


def refusal(msa: torch.Tensor, pair: torch.Tensor, n_head: int, d_hidden: int) -> str | None:
    """None if the Triton path can run this MSAPairWeightedAveraging call, else why it cannot. Never raises."""
    try:
        if not msa.is_cuda:
            return "the input is not on a CUDA device"
        if msa.dtype != torch.bfloat16 or pair.dtype != torch.bfloat16:
            return f"the Triton path is bf16, got {msa.dtype} / {pair.dtype}"
        d_msa, d_pair = msa.shape[-1], pair.shape[-1]
        if not (_pow2_16(d_msa) and _pow2_16(d_hidden)):
            return f"d_msa and d_hidden must be powers of two >= 16 (spanned by one tile), got {d_msa}, {d_hidden}"
        if d_pair < 16:
            return f"d_pair must be >= 16 (one tl.dot K step), got {d_pair}"
        if n_head < 1:
            return f"n_head must be positive, got {n_head}"
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _hp(nh: int) -> int:
    return max(16, triton.next_power_of_2(nh))


def _fwd_fake(msa, pair, mask, lnm_w, lnm_b, wv, wg, lnz_w, lnz_b, wb, wo, keep, eps_m, eps_z, scale):
    """Shapes of `_fwd`'s outputs: out like msa, w [NH, L, L], v / o [NH, L, S C] bf16."""
    S, L, _ = msa.shape
    NH, HC = wb.shape[0], wv.shape[0]
    bf = torch.bfloat16
    return [torch.empty_like(msa), msa.new_empty((NH, L, L), dtype=bf), msa.new_empty((NH, L, S * (HC // NH)), dtype=bf),
            msa.new_empty((NH, L, S * (HC // NH)), dtype=bf)]


@opaque(fake=_fwd_fake, name="pair_weighted_averaging_triton_fwd")
def _fwd(msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor, lnm_w: torch.Tensor, lnm_b: torch.Tensor, wv: torch.Tensor,
         wg: torch.Tensor, lnz_w: torch.Tensor, lnz_b: torch.Tensor, wb: torch.Tensor, wo: torch.Tensor, keep: torch.Tensor,
         eps_m: float, eps_z: float, scale: float) -> list[torch.Tensor]:
    """One stack: msa [S, L, D], pair [L, L, DZ], mask [L] bool (or empty), keep [L, D] (or empty: no dropout).
    Returns [out [S, L, D], w [NH, L, L], v, o [NH, L, S C]]; the last three are what the backward reads."""
    S, L, D = msa.shape
    NH, HC = wb.shape[0], wv.shape[0]
    C = HC // NH
    DZ = pair.shape[-1]
    bf = torch.bfloat16
    has_mask, has_keep = mask.numel() > 0, keep.numel() > 0
    x = msa.reshape(S * L, D).contiguous()
    z = pair.reshape(L * L, DZ).contiguous()
    m8 = mask.reshape(L).contiguous().view(torch.uint8) if has_mask else x
    lnm = (lnm_w.float().contiguous(), lnm_b.float().contiguous())
    w = msa.new_empty((NH, L, L), dtype=bf)
    _pwa_layernorm_gemm_softmax_kernel[(L,)](
        z, m8, lnz_w.float().contiguous(), lnz_b.float().contiguous(), wb.to(bf).contiguous(), w, L, eps_z,
        shape_key=token_key(length_of(pair.shape), DZ=DZ, NH=NH),
        DZ=DZ, DZP=triton.next_power_of_2(DZ), NH=NH, HP=_hp(NH), HAS_MASK=has_mask)
    v = msa.new_empty((NH, L, S * C), dtype=bf)
    _pwa_layernorm_gemm_kernel[lambda meta: (L, triton.cdiv(S, meta["BS"]), triton.cdiv(HC, meta["BN"]))](
        x, *lnm, wv.to(bf).contiguous(), v, S, L, eps_m, shape_key=token_key(length_of(msa.shape), D=D, HC=HC, C=C), D=D, HC=HC, C=C)
    o = torch.bmm(w, v)                                                                       # [NH, L, S*C]
    out = torch.empty_like(x)
    _pwa_gate_gemm_kernel[lambda meta: (L, triton.cdiv(S, meta["BS"]))](
        x, o, *lnm, wg.to(bf).contiguous(), wo.to(bf).contiguous(), keep if has_keep else x, out, S, L, eps_m, scale,
        shape_key=token_key(length_of(msa.shape), D=D, NH=NH, C=C), D=D, NH=NH, C=C, HAS_KEEP=has_keep)
    return [out.view(S, L, D), w, v, o]


def _bwd_fake(dres, msa, pair, mask, w, v, o, lnm_w, lnm_b, wv, wg, lnz_w, lnz_b, wb, wo, keep, eps_m, eps_z, scale):
    """Shapes of `_bwd`'s outputs: dmsa / dpair like their inputs, every parameter gradient fp32 in its parameter's shape."""
    f32 = torch.float32
    return [torch.empty_like(msa), torch.empty_like(pair)] + [t.new_empty(t.shape, dtype=f32)
                                                              for t in (lnm_w, lnm_b, wv, wg, lnz_w, lnz_b, wb, wo)]


@opaque(fake=_bwd_fake, name="pair_weighted_averaging_triton_bwd")
def _bwd(dres: torch.Tensor, msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor, w: torch.Tensor, v: torch.Tensor,
         o: torch.Tensor, lnm_w: torch.Tensor, lnm_b: torch.Tensor, wv: torch.Tensor, wg: torch.Tensor, lnz_w: torch.Tensor,
         lnz_b: torch.Tensor, wb: torch.Tensor, wo: torch.Tensor, keep: torch.Tensor, eps_m: float, eps_z: float,
         scale: float) -> list[torch.Tensor]:
    """Gradients of one stack: [dmsa, dpair, dgamma_m, dbeta_m, dWv, dWg, dgamma_z, dbeta_z, dWb, dWo] (fp32 weights)."""
    S, L, D = msa.shape
    NH, HC = wb.shape[0], wv.shape[0]
    C = HC // NH
    DZ = pair.shape[-1]
    bf = torch.bfloat16
    f32 = torch.float32
    has_mask, has_keep = mask.numel() > 0, keep.numel() > 0
    x = msa.reshape(S * L, D).contiguous()
    z = pair.reshape(L * L, DZ).contiguous()
    m8 = mask.reshape(L).contiguous().view(torch.uint8) if has_mask else x
    dr = dres.reshape(S * L, D).contiguous()
    lnm = (lnm_w.float().contiguous(), lnm_b.float().contiguous())
    wg16, wv16 = wg.to(bf).contiguous(), wv.to(bf).contiguous()
    nsm = _nsm(msa.device)
    do = msa.new_empty((NH, L, S * C), dtype=bf)
    dgp = msa.new_empty((S * L, HC), dtype=bf)
    dwo = msa.new_zeros((D, HC), dtype=f32)
    dwg = msa.new_zeros((HC, D), dtype=f32)
    dwv = msa.new_zeros((HC, D), dtype=f32)
    dgm, dbm = msa.new_zeros((D,), dtype=f32), msa.new_zeros((D,), dtype=f32)
    dwb = msa.new_zeros((NH, DZ), dtype=f32)
    dgz, dbz = msa.new_zeros((DZ,), dtype=f32), msa.new_zeros((DZ,), dtype=f32)
    # persistent (programs per SM, head) grids: the per-head weight gradients live in the program's registers
    _pwa_bwd_gate_gemm_kernel[lambda meta: (max(1, min(L * triton.cdiv(S, meta["BS"]), meta["PPS"] * nsm // NH)), NH)](
        x, dr, o, *lnm, wg16, wo.to(bf).contiguous(), keep if has_keep else x, do, dgp, dwo, dwg, S, L, eps_m, scale,
        shape_key=token_key(length_of(msa.shape), D=D, NH=NH, C=C), D=D, NH=NH, C=C, HAS_KEEP=has_keep)
    dv = torch.bmm(w.transpose(1, 2), do)                                                     # [NH, L(j), S*C]
    dw = torch.bmm(do, v.transpose(1, 2))                                                     # [NH, L(i), L(j)]
    del do
    _pwa_bwd_layernorm_gemm_dw_kernel[lambda meta: (max(1, min(L * triton.cdiv(S, meta["BS"]), meta["PPS"] * nsm // NH)), NH)](
        x, dv, *lnm, dwv, S, L, eps_m, shape_key=token_key(length_of(msa.shape), D=D, C=C), D=D, C=C)
    dm = torch.empty_like(x)
    _pwa_bwd_layernorm_gemm_dx_dlnw_kernel[lambda meta: (max(1, min(L * triton.cdiv(S, meta["BS"]), meta["PPS"] * nsm)),)](
        x, dr, dgp, dv, lnm[0], wg16, wv16, dm, dgm, dbm, S, L, eps_m,
        shape_key=token_key(length_of(msa.shape), D=D, NH=NH, C=C), D=D, NH=NH, C=C)
    dz = torch.empty_like(z)
    _pwa_bwd_layernorm_gemm_softmax_kernel[(L,)](
        z, w, dw, m8, lnz_w.float().contiguous(), lnz_b.float().contiguous(), wb.to(bf).contiguous(), dz, dwb, dgz, dbz, L, eps_z,
        shape_key=token_key(length_of(pair.shape), DZC=DZ, NH=NH),
        DZC=DZ, DZP=triton.next_power_of_2(DZ), NH=NH, HP=_hp(NH), HAS_MASK=has_mask)
    return [dm.view(S, L, D), dz.view(L, L, DZ), dgm, dbm, dwv, dwg, dgz, dbz, dwb, dwo]


class _PairWeightedAveragingTriton(torch.autograd.Function):
    """One MSA stack (no batch axis). ``mask`` / ``keep`` are empty tensors when absent."""

    @staticmethod
    def forward(ctx, msa, pair, mask, lnm_w, lnm_b, wv, wg, lnz_w, lnz_b, wb, wo, keep, eps_m, eps_z, scale):
        out, w, v, o = _fwd(msa, pair, mask, lnm_w, lnm_b, wv, wg, lnz_w, lnz_b, wb, wo, keep, eps_m, eps_z, scale)
        ctx.save_for_backward(msa, pair, mask, w, v, o, lnm_w, lnm_b, wv, wg, lnz_w, lnz_b, wb, wo, keep)
        ctx.meta = (eps_m, eps_z, scale)
        return out

    @staticmethod
    def backward(ctx, dres):
        msa, pair, mask, w, v, o, lnm_w, lnm_b, wv, wg, lnz_w, lnz_b, wb, wo, keep = ctx.saved_tensors
        g = _bwd(dres.contiguous(), msa, pair, mask, w, v, o, lnm_w, lnm_b, wv, wg, lnz_w, lnz_b, wb, wo, keep, *ctx.meta)
        dm, dz = g[0], g[1]
        params = (lnm_w, lnm_b, wv, wg, lnz_w, lnz_b, wb, wo)
        return (dm, dz, None, *(gi.to(p.dtype) for gi, p in zip(g[2:], params, strict=True)), None, None, None, None)


def triton_pair_weighted_averaging(msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, ln_msa_weight: torch.Tensor,
                                   ln_msa_bias: torch.Tensor, w_value: torch.Tensor, w_gate: torch.Tensor, ln_pair_weight: torch.Tensor,
                                   ln_pair_bias: torch.Tensor, w_bias: torch.Tensor, w_out: torch.Tensor, *, eps_msa: float = 1e-5,
                                   eps_pair: float = 1e-5, keep: torch.Tensor | None = None, p_drop: float = 0.0) -> torch.Tensor:
    """``msa + dropout(PWA(msa, pair))``, differentiable. msa [B, S, L, D] bf16, pair [B, L, L, DZ] bf16, mask [B, L] bool.

    ``keep`` [B, L, D] is the row-broadcast dropout keep-mask (the module's Dropout(broadcast_dim=1)), applied as keep / (1 - p_drop);
    None applies no dropout. Stacks run one at a time. Check :func:`refusal` first."""
    empty = msa.new_empty((0,))
    scale = 1.0 / (1.0 - p_drop) if keep is not None else 1.0
    outs = [_PairWeightedAveragingTriton.apply(msa[bi], pair[bi], empty if mask is None else mask[bi], ln_msa_weight, ln_msa_bias,
                                               w_value, w_gate, ln_pair_weight, ln_pair_bias, w_bias, w_out,
                                               empty if keep is None else keep[bi].to(torch.bfloat16).contiguous(),
                                               float(eps_msa), float(eps_pair), float(scale))
            for bi in range(msa.shape[0])]
    return outs[0].unsqueeze(0) if len(outs) == 1 else torch.stack(outs, 0)
