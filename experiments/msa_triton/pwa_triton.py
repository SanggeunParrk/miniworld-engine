"""MSAPairWeightedAveraging in Triton (portable: any GPU Triton supports), forward + backward.

Reference statements (module): y = LN_m(msa); v = y Wv^T; logits = LN_z(pair) Wb^T (masked keys -> bf16 min); w = softmax_j;
o_h = w_h v_h; u = sigmoid(y Wg^T) * o; out = msa + dropout(u Wo^T). Decomposition (the big contractions stay in cuBLAS):

  forward   pair   (Triton)  per row i: LN_z -> Wb -> key mask -> softmax over j (two passes)      w [H, L, L] bf16
            value  (Triton)  LN_m -> Wv, written head-major                                         v [H, L, S*C]
            o      (cuBLAS)  o_h = w_h v_h                                                          o [H, L, S*C]
            final  (Triton)  LN_m -> gate, u = sigmoid(g) o, out-projection summed over heads, dropout, residual
  backward  glue   (Triton)  one program per (head, tile range): g recomputed, du = dout' Wo_h, do = du g (head-major),
                             dgp = du o g (1 - g) ([T, H*C]); dWo_h, dWg_h in registers
            dv, dw (cuBLAS)  dv_h = w_h^T do_h, dw_h = do_h v_h^T
            dWv    (Triton)  per head: dv_h^T y
            proj   (Triton)  dy = dgp Wg + dv Wv, LayerNorm backward + dres -> dmsa, dgamma_m, dbeta_m
            pair   (Triton)  dlogit = w (dw - sum_j w dw) (0 on masked keys), dWb, LN_z backward -> dpair, dgamma_z, dbeta_z

Dropout: the module's Dropout(broadcast_dim=1) keeps one mask per (token i, channel) shared over the MSA rows; the training path
draws its own [L, D] keep-mask (same distribution, different RNG stream) and applies x * keep / (1 - p).

Tiles, warps, stages and persistent program counts are autotune axes (``_tuning.SPECS``, the engine's GRID SPEC form); the only
constants in the kernels are the model dimensions (constexpr). Offsets into the head-major [H, L, S C] buffers and the [S L, H C]
buffers are 64-bit. Unsupported calls (non-bf16 input, dimensions that are not powers of two >= 16) are refused with the reason.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

try:
    from . import _tuning as T
except ImportError:                       # run as a script from this directory
    import _tuning as T

NEG = tl.constexpr(-3.3895313892515355e38)   # torch.finfo(torch.bfloat16).min: the module's masked_fill value


def _nsm():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


# ---------------------------------------------------------------------------------------------------- forward kernels
@triton.autotune(configs=T.configs("pwa_pair_fwd_triton"), key=["shape_key", "DZ", "H", "HAS_MASK"])
@triton.jit
def _pair_fwd_kernel(Z, MASK, LNW, LNB, WB, W, L, eps, shape_key,
                     DZ: tl.constexpr, H: tl.constexpr, HP: tl.constexpr, HAS_MASK: tl.constexpr, BJ: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, DZ)
    hh = tl.arange(0, HP)
    gz = tl.load(LNW + c)
    bz = tl.load(LNB + c)
    wbt = tl.load(WB + hh[None, :] * DZ + c[:, None], mask=(hh < H)[None, :], other=0.0)     # [DZ, HP] = Wb^T, padded heads
    m_run = tl.full((HP,), -float("inf"), tl.float32)
    l_run = tl.zeros((HP,), tl.float32)
    for pas in tl.static_range(2):
        for j0 in range(0, L, BJ):
            j = j0 + tl.arange(0, BJ)
            jm = j < L
            z = tl.load(Z + (i * L + j)[:, None] * DZ + c[None, :], mask=jm[:, None], other=0.0).to(tl.float32)
            mu = tl.sum(z, axis=1) / DZ
            zc = z - mu[:, None]
            rs = tl.rsqrt(tl.sum(zc * zc, axis=1) / DZ + eps)
            zy = (zc * rs[:, None] * gz[None, :] + bz[None, :]).to(tl.bfloat16)
            lg = tl.dot(zy, wbt)                                                                  # [BJ, HP]
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
                tl.store(W + (hh[None, :] * L + i) * L + j[:, None], pr.to(tl.bfloat16), mask=jm[:, None] & (hh < H)[None, :])


@triton.autotune(configs=T.configs("pwa_value_fwd_triton"), key=["shape_key", "D", "HC", "C"],
                 prune_configs_by={"early_config_prune": T.prune(lambda c, a: c["BN"] <= a["HC"])})
@triton.jit
def _value_fwd_kernel(M, LNW, LNB, WV, V, S, L, eps, shape_key,
                      D: tl.constexpr, HC: tl.constexpr, C: tl.constexpr, BS: tl.constexpr, BN: tl.constexpr):
    # program = (token j, block of BS MSA rows, block of BN of the H C outputs): v[h][j][s C + c] = LN(msa[s, j]) . Wv[h C + c]
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
    wvt = tl.load(WV + n[None, :] * D + k[:, None])                                               # [D, BN]
    v = tl.dot(y, wvt).to(tl.bfloat16)                                                            # [BS, BN]
    h = n // C
    cc = n % C
    ptr = V + (h[None, :].to(tl.int64) * L + j) * (S * C) + s[:, None] * C + cc[None, :]
    tl.store(ptr, v, mask=sm[:, None])


@triton.autotune(configs=T.configs("pwa_final_fwd_triton"), key=["shape_key", "D", "H", "C", "HAS_KEEP"])
@triton.jit
def _final_fwd_kernel(M, O, LNW, LNB, WG, WO, KEEP, OUT, S, L, eps, scale, shape_key,
                      D: tl.constexpr, H: tl.constexpr, C: tl.constexpr, HAS_KEEP: tl.constexpr, BS: tl.constexpr):
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
    for h in range(H):
        wgt = tl.load(WG + (h * C + cc)[None, :] * D + k[:, None])                               # [D, C]
        g = tl.sigmoid(tl.dot(y, wgt))
        o = tl.load(O + (h * L + i) * (S * C) + s[:, None] * C + cc[None, :], mask=sm[:, None], other=0.0).to(tl.float32)
        u = (g * o).to(tl.bfloat16)
        wot = tl.load(WO + k[None, :] * (H * C) + h * C + cc[:, None])                           # [C, D] = Wo_h^T
        acc += tl.dot(u, wot)
    if HAS_KEEP:
        acc = acc * tl.load(KEEP + i * D + k).to(tl.float32)[None, :] * scale
    tl.store(OUT + rowp, (x + acc).to(tl.bfloat16), mask=sm[:, None])


# ---------------------------------------------------------------------------------------------------- backward kernels
@triton.autotune(configs=T.configs("pwa_glue_bwd_triton"), key=["shape_key", "D", "H", "C", "HAS_KEEP"],
                 restore_value=["DWO", "DWG"])
@triton.jit
def _glue_bwd_kernel(M, DR, O, LNW, LNB, WG, WO, KEEP, DO, DGP, DWO, DWG, S, L, eps, scale, shape_key,
                     D: tl.constexpr, H: tl.constexpr, C: tl.constexpr, HAS_KEEP: tl.constexpr, BS: tl.constexpr, PPS: tl.constexpr):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    nprog = tl.num_programs(0)
    k = tl.arange(0, D)
    cc = tl.arange(0, C)
    gam = tl.load(LNW + k)
    bet = tl.load(LNB + k)
    wgt = tl.load(WG + (h * C + cc)[None, :] * D + k[:, None])                                   # [D, C]
    woh = tl.load(WO + k[:, None] * (H * C) + h * C + cc[None, :])                                # [D, C] = Wo[:, h]
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
        g = tl.sigmoid(tl.dot(y, wgt))                                                            # [BS, C]
        hp = (h.to(tl.int64) * L + i) * (S * C) + s[:, None] * C + cc[None, :]
        o = tl.load(O + hp, mask=sm[:, None], other=0.0).to(tl.float32)
        du = tl.dot(drb, woh)                                                                     # [BS, C]
        tl.store(DO + hp, (du * g).to(tl.bfloat16), mask=sm[:, None])
        gp = (du * o * g * (1.0 - g)).to(tl.bfloat16)
        tl.store(DGP + (s * L + i)[:, None] * (H * C) + h * C + cc[None, :], gp, mask=sm[:, None])
        dwo += tl.dot(tl.trans(drb), (g * o).to(tl.bfloat16))
        dwg += tl.dot(tl.trans(gp), y)
    tl.atomic_add(DWO + k[:, None] * (H * C) + h * C + cc[None, :], dwo)
    tl.atomic_add(DWG + (h * C + cc)[:, None] * D + k[None, :], dwg)


@triton.autotune(configs=T.configs("pwa_dwv_triton"), key=["shape_key", "D", "C"], restore_value=["DWV"])
@triton.jit
def _dwv_kernel(M, DV, LNW, LNB, DWV, S, L, eps, shape_key,
                D: tl.constexpr, C: tl.constexpr, BS: tl.constexpr, PPS: tl.constexpr):
    # program = (head, tile range): dWv[h C + c, :] += sum_(s, j) dv[h][j][s C + c] y[s, j, :]
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


@triton.autotune(configs=T.configs("pwa_proj_bwd_triton"), key=["shape_key", "D", "H", "C"], restore_value=["DG", "DB"])
@triton.jit
def _proj_bwd_kernel(M, DR, DGP, DV, LNW, LNB, WG, WV, DM, DG, DB, S, L, eps, shape_key,
                     D: tl.constexpr, H: tl.constexpr, C: tl.constexpr, BS: tl.constexpr, PPS: tl.constexpr):
    # program walks tiles (token j, BS MSA rows): dy = sum_h (dgp_h Wg_h + dv_h Wv_h); LayerNorm backward + dres -> dmsa.
    # One head's [C, D] weight slices per step (not the whole [H C, D] weights resident): small shared memory on any GPU.
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
        for h in range(H):
            gp = tl.load(DGP + (s * L + j)[:, None] * (H * C) + h * C + cc[None, :], mask=sm[:, None], other=0.0)
            dv = tl.load(DV + (h * L + j) * (S * C) + s[:, None] * C + cc[None, :], mask=sm[:, None], other=0.0)          # h * L + j: int64 (j)
            wgh = tl.load(WG + (h * C + cc)[:, None] * D + k[None, :])                           # [C, D]
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


@triton.autotune(configs=T.configs("pwa_pair_bwd_triton"), key=["shape_key", "DZC", "H", "HAS_MASK"],
                 restore_value=["DWB", "DGZ", "DBZ"])
@triton.jit
def _pair_bwd_kernel(Z, W, DW, MASK, LNW, LNB, WB, DZ, DWB, DGZ, DBZ, L, eps, shape_key,
                     DZC: tl.constexpr, H: tl.constexpr, HP: tl.constexpr, HAS_MASK: tl.constexpr, BJ: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, DZC)
    hh = tl.arange(0, HP)
    hm = hh < H
    gz = tl.load(LNW + c)
    bz = tl.load(LNB + c)
    wb = tl.load(WB + hh[:, None] * DZC + c[None, :], mask=hm[:, None], other=0.0)            # [HP, DZ]
    ssum = tl.zeros((HP,), tl.float32)
    for j0 in range(0, L, BJ):                                                                   # s_h = sum_j w dw
        j = j0 + tl.arange(0, BJ)
        msk = (j < L)[:, None] & hm[None, :]
        pp = (hh[None, :] * L + i) * L + j[:, None]
        wv = tl.load(W + pp, mask=msk, other=0.0).to(tl.float32)
        dv = tl.load(DW + pp, mask=msk, other=0.0).to(tl.float32)
        ssum += tl.sum(wv * dv, axis=0)
    dwb = tl.zeros((HP, DZC), tl.float32)
    dgs = tl.zeros((DZC,), tl.float32)
    dbs = tl.zeros((DZC,), tl.float32)
    for j0 in range(0, L, BJ):
        j = j0 + tl.arange(0, BJ)
        jm = j < L
        msk = jm[:, None] & hm[None, :]
        pp = (hh[None, :] * L + i) * L + j[:, None]
        wv = tl.load(W + pp, mask=msk, other=0.0).to(tl.float32)
        dv = tl.load(DW + pp, mask=msk, other=0.0).to(tl.float32)
        dl = wv * (dv - ssum[None, :])                                                           # [BJ, HP]
        if HAS_MASK:                                                                             # a masked logit is a constant
            km = tl.load(MASK + j, mask=jm, other=0) != 0
            dl = tl.where(km[:, None], dl, 0.0)
        z = tl.load(Z + (i * L + j)[:, None] * DZC + c[None, :], mask=jm[:, None], other=0.0).to(tl.float32)
        mu = tl.sum(z, axis=1) / DZC
        zc = z - mu[:, None]
        rs = tl.rsqrt(tl.sum(zc * zc, axis=1) / DZC + eps)
        zh = zc * rs[:, None]
        zy = (zh * gz[None, :] + bz[None, :]).to(tl.bfloat16)
        dlb = dl.to(tl.bfloat16)
        dwb += tl.dot(tl.trans(dlb), zy)
        dzy = tl.dot(dlb, wb.to(tl.bfloat16))                                                   # [BJ, DZ]
        dgs += tl.sum(dzy * zh, axis=0)
        dbs += tl.sum(dzy, axis=0)
        dxh = dzy * gz[None, :]
        m1 = tl.sum(dxh, axis=1) / DZC
        m2 = tl.sum(dxh * zh, axis=1) / DZC
        dz = rs[:, None] * (dxh - m1[:, None] - zh * m2[:, None])
        tl.store(DZ + (i * L + j)[:, None] * DZC + c[None, :], dz.to(tl.bfloat16), mask=jm[:, None])
    tl.atomic_add(DWB + hh[:, None] * DZC + c[None, :], dwb, mask=hm[:, None])
    tl.atomic_add(DGZ + c, dgs)
    tl.atomic_add(DBZ + c, dbs)


# ---------------------------------------------------------------------------------------------------- autograd
def refusal(module, msa, pair, mask=None) -> str | None:
    """None if the Triton path serves this MSAPairWeightedAveraging call, else why not."""
    if not msa.is_cuda:
        return "the input is not on a CUDA device"
    if msa.dtype != torch.bfloat16 or pair.dtype != torch.bfloat16:
        return f"the Triton path is bf16, got {msa.dtype} / {pair.dtype}"
    hc = module.to_value.weight.shape[0]
    return T.pow2_at_least_16(d_msa=msa.shape[-1], d_pair=pair.shape[-1], d_hidden=hc // module.n_head, n_head_x_d_hidden=hc)


def _forward(msa, pair, mask, lnm_w, lnm_b, wv, wg, lnz_w, lnz_b, wb, wo, eps_m, eps_z, keep, scale):
    S, L, D = msa.shape
    H = wb.shape[0]
    HC = wv.shape[0]
    C = HC // H
    DZ = pair.shape[-1]
    dev = msa.device
    bf = torch.bfloat16
    x = msa.reshape(S * L, D).contiguous()
    z = pair.reshape(L * L, DZ).contiguous()
    m8 = mask.reshape(L).contiguous().view(torch.uint8) if mask is not None else x
    key = T.shape_key(S, L)
    w = torch.empty(H, L, L, device=dev, dtype=bf)
    _pair_fwd_kernel[(L,)](z, m8, lnz_w, lnz_b, wb, w, L, eps_z, key, DZ=DZ, H=H, HP=max(16, triton.next_power_of_2(H)),
                           HAS_MASK=mask is not None)
    v = torch.empty(H, L, S * C, device=dev, dtype=bf)
    _value_fwd_kernel[lambda M_: (L, triton.cdiv(S, M_["BS"]), triton.cdiv(HC, M_["BN"]))](x, lnm_w, lnm_b, wv, v, S, L, eps_m, key,
                                                                                          D=D, HC=HC, C=C)
    o = torch.bmm(w, v)                                                                           # [H, L, S*C]
    out = torch.empty_like(x)
    _final_fwd_kernel[lambda M_: (L, triton.cdiv(S, M_["BS"]))](x, o, lnm_w, lnm_b, wg, wo, keep if keep is not None else x, out, S, L,
                                                               eps_m, scale, key, D=D, H=H, C=C, HAS_KEEP=keep is not None)
    return out.view(S, L, D), (x, z, m8 if mask is not None else None, w, v, o)


class PWATriton(torch.autograd.Function):
    """One MSA stack (no batch dim): msa [S, L, D], pair [L, L, DZ], mask [L] or None."""

    @staticmethod
    def forward(ctx, msa, pair, mask, lnm_w, lnm_b, wv, wg, lnz_w, lnz_b, wb, wo, eps_m, eps_z, p_drop):
        bf = torch.bfloat16
        f32 = lambda t: t.detach().float().contiguous()  # noqa: E731
        b16 = lambda t: t.detach().to(bf).contiguous()  # noqa: E731
        L, D = msa.shape[1], msa.shape[2]
        keep, scale = None, 1.0
        if p_drop > 0:
            keep = (torch.rand(L, D, device=msa.device) > p_drop).to(bf)
            scale = 1.0 / (1.0 - p_drop)
        args = (f32(lnm_w), f32(lnm_b), b16(wv), b16(wg), f32(lnz_w), f32(lnz_b), b16(wb), b16(wo))
        out, (x, z, m8, w, v, o) = _forward(msa, pair, mask, *args, eps_m, eps_z, keep, scale)
        ctx.save_for_backward(x, z, w, v, o, *args, *([m8] if m8 is not None else []), *([keep] if keep is not None else []))
        ctx.has_mask, ctx.has_keep = m8 is not None, keep is not None
        ctx.meta = (msa.shape, pair.shape, eps_m, eps_z, scale)
        ctx.dtypes = (lnm_w.dtype, lnm_b.dtype, wv.dtype, wg.dtype, lnz_w.dtype, lnz_b.dtype, wb.dtype, wo.dtype)
        return out

    @staticmethod
    def backward(ctx, dres):
        sv = ctx.saved_tensors
        x, z, w, v, o, lnm_w, lnm_b, wv, wg, lnz_w, lnz_b, wb, wo = sv[:13]
        rest = list(sv[13:])
        m8 = rest.pop(0) if ctx.has_mask else None
        keep = rest.pop(0) if ctx.has_keep else None
        (S, L, D), pshape, eps_m, eps_z, scale = ctx.meta
        H = wb.shape[0]
        HC = wv.shape[0]
        C = HC // H
        DZ = pshape[-1]
        dev = x.device
        bf = torch.bfloat16
        dr = dres.reshape(S * L, D).contiguous()
        key = T.shape_key(S, L)
        nsm = _nsm()
        do = torch.empty(H, L, S * C, device=dev, dtype=bf)
        dgp = torch.empty(S * L, HC, device=dev, dtype=bf)
        red = torch.zeros(D * HC + HC * D + HC * D + 2 * D + H * DZ + 2 * DZ, device=dev, dtype=torch.float32)
        o_ = 0
        dwo = red[o_:o_ + D * HC].view(D, HC); o_ += D * HC
        dwg = red[o_:o_ + HC * D].view(HC, D); o_ += HC * D
        dwv = red[o_:o_ + HC * D].view(HC, D); o_ += HC * D
        dgm, dbm = red[o_:o_ + D], red[o_ + D:o_ + 2 * D]; o_ += 2 * D
        dwb = red[o_:o_ + H * DZ].view(H, DZ); o_ += H * DZ
        dgz, dbz = red[o_:o_ + DZ], red[o_ + DZ:o_ + 2 * DZ]
        # persistent (programs per SM, head) grids: the per-head weight gradients live in the program's registers
        pgrid = lambda M_: (max(1, min(L * triton.cdiv(S, M_["BS"]), M_["PPS"] * nsm // H)), H)  # noqa: E731
        _glue_bwd_kernel[pgrid](x, dr, o, lnm_w, lnm_b, wg, wo, keep if keep is not None else x, do, dgp, dwo, dwg, S, L, eps_m,
                                scale, key, D=D, H=H, C=C, HAS_KEEP=keep is not None)
        dv = torch.bmm(w.transpose(1, 2), do)                                                     # [H, L(j), S*C]
        dw = torch.bmm(do, v.transpose(1, 2))                                                     # [H, L(i), L(j)]
        _dwv_kernel[pgrid](x, dv, lnm_w, lnm_b, dwv, S, L, eps_m, key, D=D, C=C)
        dm = torch.empty_like(x)
        _proj_bwd_kernel[lambda M_: (max(1, min(L * triton.cdiv(S, M_["BS"]), M_["PPS"] * nsm)),)](
            x, dr, dgp, dv, lnm_w, lnm_b, wg, wv, dm, dgm, dbm, S, L, eps_m, key, D=D, H=H, C=C)
        dz = torch.empty_like(z)
        _pair_bwd_kernel[(L,)](z, w, dw, m8 if m8 is not None else x, lnz_w, lnz_b, wb, dz, dwb, dgz, dbz, L, eps_z, key, DZC=DZ, H=H,
                               HP=max(16, triton.next_power_of_2(H)), HAS_MASK=m8 is not None)
        t = ctx.dtypes
        return (dm.view(S, L, D), dz.view(pshape), None, dgm.to(t[0]), dbm.to(t[1]), dwv.to(t[2]), dwg.to(t[3]), dgz.to(t[4]),
                dbz.to(t[5]), dwb.to(t[6]), dwo.to(t[7]), None, None, None)


def pair_weighted_averaging(module, msa, pair, mask=None):
    """msa + PWA(msa, pair, key mask) with the module's dropout (training only) through the Triton path; differentiable. Batches are
    run per stack. Raises NotImplementedError with the reason for an unsupported call."""
    why = refusal(module, msa, pair, mask)
    if why is not None:
        raise NotImplementedError(why)
    p_drop = float(module.drop_msa.p_drop) if module.training else 0.0
    args = (module.ln_msa.weight, module.ln_msa.bias, module.to_value.weight, module.to_gate.weight, module.ln_pair.weight,
            module.ln_pair.bias, module.to_bias.weight, module.to_out.weight, float(module.ln_msa.eps), float(module.ln_pair.eps), p_drop)
    outs = [PWATriton.apply(msa[bi], pair[bi], None if mask is None else mask[bi], *args) for bi in range(msa.shape[0])]
    return torch.stack(outs, 0)
