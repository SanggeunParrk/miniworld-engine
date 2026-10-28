"""OuterProductMean in Triton, forward and backward -- the portable path (any GPU Triton supports).

The decomposition keeps the module's statements and fuses only where the PyTorch path pays extra memory passes:

  forward   layernorm_gemm   LN(msa) -> left / right projections -> mask                       a, b [S, L, CH] bf16
            outer product    O = a^T b in the grouped layout [(i, d), (j, e)]  (cuBLAS: the one layout that is a matmul)
            counts           n = mask^T mask (fp32, clamp 1)
            epilogue         out[i, j] = W_out . vec(O_ij) / n_ij + bias (+ residual)   the [i,j,d,e] permute folded into the loads
  backward  bwd_epilogue_dx  dzn = dz / n, dO = dzn . W_out, written in the grouped layout
            dA, dB           dA = b . dO^T, dB = a . dO  (cuBLAS)
            bwd_epilogue_dw  dW_out = sum_ij dzn_ij (x) O_ij off the kept grouped O  (M = c, N = (d, e), K split over i)
            bwd_layernorm_gemm   mask, dy = [dA | dB] W, dW_left / dW_right, LayerNorm backward -> dmsa, dgamma, dbeta

Every tile, warp count, stage count and persistent program count is an autotune axis (``autotune/configs/grid/<op>.csv``);
the constants in the kernels are the model dimensions (constexpr). Offsets that scale with L^2 CH^2 or S L are 64-bit.
``normalize_before_proj=True`` (AF3 order) divides before the projection; False (ESMFold2) divides after it, bias included.

d_msa and d_hidden must be powers of two >= 16 (``tl.arange`` over the whole row); d_pair only has to be a multiple of 32,
because it is tiled, never spanned (the smallest output tile of the ladder is 32). ``refusal`` says which.

Two variants ride the same kernels (``kernels.outer_product_mean.reference`` is their oracle):

  projection biases (RoseTTAFold3)  b_left / b_right are added in the projection kernel's epilogue before the mask (a masked row loses
                                    its bias too); the backward kernel sums the masked dA / dB into db_left / db_right.
  groups G > 1 (Chai-1)             the hidden channels are G halves of K = CH / G; the projection kernel stores them group-major
                                    [G, S L, K], so the outer product is ONE batched GEMM O[g] = a_g^T b_g ([G, L K, L K]) and the
                                    G x (CH/G)^2 products of a pair (i, j) are a gather from it. Chai-1 LayerNorms those products
                                    (``product_norm``, its own eps) and projects them; there is no division by the pair counts. The
                                    ``*_epilogue_ln_*`` kernels are the row-wise pass over the pair map (gather + LayerNorm -> y [L L, P];
                                    the LayerNorm backward + dO scatter) with the projection and dy = dz W_out in cuBLAS, and dW_out a
                                    split-K kernel over (dz, y); dA / dB are two batched GEMMs and the (shared) projection backward
                                    kernel closes the chain.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import length_of, token_key
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._device import num_sms


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




# ---------------------------------------------------------------------------------------------------- kernels
@triton.autotune(configs=configs_for("outer_product_mean_layernorm_gemm_triton"), key=["shape_key", "HAS_MASK", "HAS_PBIAS", "K"])
@triton.jit
def _opm_layernorm_gemm_kernel(M, MASK, LNW, LNB, WL, WR, BL, BR, A, B, T_, eps, shape_key,
                               CM: tl.constexpr, CH: tl.constexpr, K: tl.constexpr, HAS_MASK: tl.constexpr,
                               HAS_PBIAS: tl.constexpr, BT: tl.constexpr):
    # a[t] = mask[t] * (LN(msa[t]) . W_left^T [+ b_left]), b likewise; one program = BT rows of the flattened [S L] tokens.
    # Stored group-major [G, T, K] (G = CH / K groups of K channels; K = CH is the ungrouped [T, CH] layout, same formula), so each
    # group's [S, L K] operand of the outer product is a plain strided matrix.
    rows = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    rm = rows < T_
    cols = tl.arange(0, CM)
    x = tl.load(M + rows[:, None] * CM + cols[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    mu = tl.sum(x, axis=1) / CM
    xc = x - mu[:, None]
    rs = tl.rsqrt(tl.sum(xc * xc, axis=1) / CM + eps)
    y = (xc * rs[:, None] * tl.load(LNW + cols)[None, :] + tl.load(LNB + cols)[None, :]).to(tl.bfloat16)
    ch = tl.arange(0, CH)
    wl = tl.load(WL + ch[None, :] * CM + cols[:, None])                                   # [CM, CH] = W_left^T
    wr = tl.load(WR + ch[None, :] * CM + cols[:, None])
    a = tl.dot(y, wl)
    b = tl.dot(y, wr)
    if HAS_PBIAS:
        a += tl.load(BL + ch)[None, :]
        b += tl.load(BR + ch)[None, :]
    a = a.to(tl.bfloat16)
    b = b.to(tl.bfloat16)
    if HAS_MASK:                                                                          # a masked row loses its bias too
        mk = tl.load(MASK + rows, mask=rm, other=0) != 0
        a = tl.where(mk[:, None], a, 0.0)
        b = tl.where(mk[:, None], b, 0.0)
    dst = ((ch // K).to(tl.int64)[None, :] * T_ + rows[:, None]) * K + (ch % K)[None, :]
    tl.store(A + dst, a, mask=rm[:, None])
    tl.store(B + dst, b, mask=rm[:, None])


@triton.autotune(configs=configs_for("outer_product_mean_epilogue_triton"), key=["shape_key", "HAS_RES", "NORM_FIRST"],
                 prune_configs_by={"early_config_prune": _divides(BN="CZ", BKD="CH")})
@triton.jit
def _opm_epilogue_kernel(O, WOT, BIAS, NORM, RES, OUT, L, shape_key,
                         CH: tl.constexpr, CZ: tl.constexpr, HAS_RES: tl.constexpr, NORM_FIRST: tl.constexpr,
                         TI: tl.constexpr, BJ: tl.constexpr, BN: tl.constexpr, BKD: tl.constexpr):
    # tile = TI rows i x BJ columns j (M = TI BJ pairs, so W_out is re-read once per TI BJ pairs) x BN outputs; K = CH^2 in steps of
    # BKD values of d (a K chunk of BKD CH): A[(i, j), (d, e)] = O[i CH + d, j CH + e]
    r = tl.arange(0, TI * BJ)
    i = tl.program_id(0) * TI + r // BJ
    j = tl.program_id(1) * BJ + r % BJ
    rm = (i < L) & (j < L)
    n = tl.program_id(2) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BKD * CH)
    dd = kk // CH
    e = kk % CH
    ldo = L.to(tl.int64) * CH
    arow = i.to(tl.int64) * CH * ldo + j.to(tl.int64) * CH
    acc = tl.zeros((TI * BJ, BN), dtype=tl.float32)
    for d0 in range(0, CH, BKD):
        a = tl.load(O + arow[:, None] + ((d0 + dd).to(tl.int64) * ldo + e)[None, :], mask=rm[:, None], other=0.0)
        w = tl.load(WOT + ((d0 + dd) * CH + e)[:, None] * CZ + n[None, :])
        acc += tl.dot(a, w)
    nrm = tl.load(NORM + i.to(tl.int64) * L + j, mask=rm, other=1.0)
    bias = tl.load(BIAS + n)
    if NORM_FIRST:
        z = acc / nrm[:, None] + bias[None, :]
    else:
        z = (acc + bias[None, :]) / nrm[:, None]
    ptr = (i.to(tl.int64) * L + j)[:, None] * CZ + n[None, :]
    if HAS_RES:
        z += tl.load(RES + ptr, mask=rm[:, None], other=0.0).to(tl.float32)
    tl.store(OUT + ptr, z.to(tl.bfloat16), mask=rm[:, None])


@triton.autotune(configs=configs_for("outer_product_mean_bwd_epilogue_dx_triton"), key=["shape_key"],
                 prune_configs_by={"early_config_prune": _divides(BKC="CZ", BN="CH2")})
@triton.jit
def _opm_bwd_epilogue_dx_kernel(DZ, NORM, WO, DZN, DO, L, CH2, shape_key,
                                CH: tl.constexpr, CZ: tl.constexpr, TI: tl.constexpr, BJ: tl.constexpr, BN: tl.constexpr,
                                BKC: tl.constexpr):
    # dzn = dz / n (either order: the projection is linear), dO[(i, d), (j, e)] = sum_c dzn[i, j, c] W_out[c, d CH + e].
    # tile = TI i x BJ j pairs x BN of the CH^2 (d, e) outputs, K = CZ in BKC steps; the n-block-0 programs also store dzn.
    r = tl.arange(0, TI * BJ)
    i = tl.program_id(0) * TI + r // BJ
    j = tl.program_id(1) * BJ + r % BJ
    rm = (i < L) & (j < L)
    pn = tl.program_id(2)
    n = pn * BN + tl.arange(0, BN)
    d = n // CH
    e = n % CH
    nrm = tl.load(NORM + i.to(tl.int64) * L + j, mask=rm, other=1.0)
    prow = (i.to(tl.int64) * L + j) * CZ
    acc = tl.zeros((TI * BJ, BN), dtype=tl.float32)
    for c0 in range(0, CZ, BKC):
        c = c0 + tl.arange(0, BKC)
        dz = tl.load(DZ + prow[:, None] + c[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        dzn = (dz / nrm[:, None]).to(tl.bfloat16)
        if pn == 0:
            tl.store(DZN + prow[:, None] + c[None, :], dzn, mask=rm[:, None])
        w = tl.load(WO + c[:, None] * (CH * CH) + n[None, :])                                # [BKC, BN]
        acc += tl.dot(dzn, w)
    ldo = L.to(tl.int64) * CH
    orow = i.to(tl.int64) * CH * ldo + j.to(tl.int64) * CH
    tl.store(DO + orow[:, None] + (d.to(tl.int64) * ldo + e)[None, :], acc.to(tl.bfloat16), mask=rm[:, None])


@triton.autotune(configs=configs_for("outer_product_mean_bwd_epilogue_dw_triton"), key=["shape_key"], reset_to_zero=["DWO"],
                 prune_configs_by={"early_config_prune": _divides(BM="CZ", BN="CH2")})
@triton.jit
def _opm_bwd_epilogue_dw_kernel(DZN, O, DWO, L, CH2, shape_key,
                                CH: tl.constexpr, CZ: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                                SPLIT: tl.constexpr):
    # dWo[c, (d, e)] = sum_(i, j) dzn[i, j, c] O[i CH + d, j CH + e]: an M = CZ, N = CH^2, K = L^2 GEMM, tile BM x BN, K split over
    # SPLIT ranges of i (fp32 atomics once per program)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    d = n // CH
    e = n % CH
    sp = tl.program_id(2)
    i_per = tl.cdiv(L, SPLIT)
    ldo = L.to(tl.int64) * CH
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for ii in range(i_per):
        i = sp * i_per + ii
        if i < L:
            for j0 in range(0, L, BK):
                j = j0 + kk
                jm = j < L
                a = tl.load(DZN + (i.to(tl.int64) * L + j)[:, None] * CZ + m[None, :], mask=jm[:, None], other=0.0)     # [BK, BM]
                b = tl.load(O + i.to(tl.int64) * CH * ldo + j[:, None].to(tl.int64) * CH + (d.to(tl.int64) * ldo + e)[None, :],
                            mask=jm[:, None], other=0.0)                                                             # [BK, BN]
                acc += tl.dot(tl.trans(a), b)
    tl.atomic_add(DWO + m[:, None] * (CH * CH) + n[None, :], acc)


@triton.autotune(configs=configs_for("outer_product_mean_bwd_layernorm_gemm_triton"), key=["shape_key", "HAS_MASK", "HAS_PBIAS", "K"],
                 reset_to_zero=["DWL", "DWR", "DG", "DBETA", "DBL", "DBR"])
@triton.jit
def _opm_bwd_layernorm_gemm_kernel(M, MASK, LNW, LNB, WL, WR, DA, DB, DM, DWL, DWR, DG, DBETA, DBL, DBR, T_, eps, shape_key,
                                   CM: tl.constexpr, CH: tl.constexpr, K: tl.constexpr, HAS_MASK: tl.constexpr,
                                   HAS_PBIAS: tl.constexpr, BT: tl.constexpr, PPS: tl.constexpr):
    # persistent (PPS programs per SM): the program walks token blocks pid, pid + nprog, ..., keeping dW_left / dW_right / dgamma /
    # dbeta in registers, and adds them once at the end
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    cols = tl.arange(0, CM)
    ch = tl.arange(0, CH)
    g = tl.load(LNW + cols)
    bta = tl.load(LNB + cols)
    wl = tl.load(WL + ch[:, None] * CM + cols[None, :])                                   # [CH, CM]
    wr = tl.load(WR + ch[:, None] * CM + cols[None, :])
    dwl = tl.zeros((CH, CM), dtype=tl.float32)
    dwr = tl.zeros((CH, CM), dtype=tl.float32)
    dgs = tl.zeros((CM,), dtype=tl.float32)
    dbs = tl.zeros((CM,), dtype=tl.float32)
    dbl = tl.zeros((CH,), dtype=tl.float32)
    dbr = tl.zeros((CH,), dtype=tl.float32)
    src_ch = ((ch // K).to(tl.int64) * T_)[None, :] * K + (ch % K)[None, :]               # group-major [G, T, K] columns
    nblk = tl.cdiv(T_, BT)
    for blk in range(pid, nblk, nprog):
        rows = blk.to(tl.int64) * BT + tl.arange(0, BT)
        rm = rows < T_
        x = tl.load(M + rows[:, None] * CM + cols[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        mu = tl.sum(x, axis=1) / CM
        xc = x - mu[:, None]
        rs = tl.rsqrt(tl.sum(xc * xc, axis=1) / CM + eps)
        xh = xc * rs[:, None]
        y = (xh * g[None, :] + bta[None, :]).to(tl.bfloat16)
        da = tl.load(DA + rows[:, None] * K + src_ch, mask=rm[:, None], other=0.0)
        db = tl.load(DB + rows[:, None] * K + src_ch, mask=rm[:, None], other=0.0)
        if HAS_MASK:
            mk = tl.load(MASK + rows, mask=rm, other=0) != 0
            da = tl.where(mk[:, None], da, 0.0).to(tl.bfloat16)
            db = tl.where(mk[:, None], db, 0.0).to(tl.bfloat16)
        if HAS_PBIAS:
            dbl += tl.sum(da.to(tl.float32), axis=0)
            dbr += tl.sum(db.to(tl.float32), axis=0)
        dwl += tl.dot(tl.trans(da), y)
        dwr += tl.dot(tl.trans(db), y)
        dy = tl.dot(da, wl) + tl.dot(db, wr)                                                 # [BT, CM] fp32
        dgs += tl.sum(dy * xh, axis=0)
        dbs += tl.sum(dy, axis=0)
        dxh = dy * g[None, :]
        m1 = tl.sum(dxh, axis=1) / CM
        m2 = tl.sum(dxh * xh, axis=1) / CM
        dx = rs[:, None] * (dxh - m1[:, None] - xh * m2[:, None])
        tl.store(DM + rows[:, None] * CM + cols[None, :], dx.to(tl.bfloat16), mask=rm[:, None])
    tl.atomic_add(DWL + ch[:, None] * CM + cols[None, :], dwl)
    tl.atomic_add(DWR + ch[:, None] * CM + cols[None, :], dwr)
    tl.atomic_add(DG + cols, dgs)
    tl.atomic_add(DBETA + cols, dbs)
    if HAS_PBIAS:
        tl.atomic_add(DBL + ch, dbl)
        tl.atomic_add(DBR + ch, dbr)


# ---------------------------------------------------------------------------------------------------- grouped kernels (Chai-1)
# ``groups`` G > 1: per group g an outer product of K = CH / G wide halves, O_g = a_g^T b_g ([L K, L K], one batched GEMM over g), and
# for the pair (i, j) the G K^2 products p[g, k, l] = O_g[i K + k, j K + l]. Chai-1 LayerNorms those P = G K^2 products (``product_norm``,
# its own eps) and projects them; there is no division by the pair counts. The gather + LayerNorm is the only part that is not a GEMM: it
# is a row-wise pass over the pair map (kernels below), the projection and its gradient dy = dz W_out are cuBLAS, dW_out a split-K kernel.
@triton.autotune(configs=configs_for("outer_product_mean_epilogue_ln_triton"), key=["shape_key", "P"])
@triton.jit
def _opm_grouped_epilogue_kernel(O, GAM, BET, Y, L, eps, shape_key,
                                 K: tl.constexpr, P: tl.constexpr, BJ: tl.constexpr):
    # program = (row i, BJ columns j): the (j, l) run of every (g, k) is contiguous in O, so the tile is loaded as [G K, BJ K] (coalesced),
    # LayerNorm'd over (g k, l) per j, turned into BJ rows of P and stored as the rows of y [L L, P]
    GK: tl.constexpr = P // K
    i = tl.program_id(0).to(tl.int64)
    j0 = tl.program_id(1) * BJ
    gk = tl.arange(0, GK)
    jl = tl.arange(0, BJ * K)
    lk = L.to(tl.int64) * K
    ptr = O + (gk // K).to(tl.int64)[:, None] * lk * lk + ((gk % K) + i * K).to(tl.int64)[:, None] * lk + (j0 * K + jl)[None, :]
    x = tl.load(ptr, mask=((j0 + jl // K) < L)[None, :], other=0.0).to(tl.float32)
    x3 = tl.reshape(x, [GK, BJ, K])
    mu = tl.sum(tl.sum(x3, axis=2), axis=0) / P
    xc = x3 - mu[None, :, None]
    rs = tl.rsqrt(tl.sum(tl.sum(xc * xc, axis=2), axis=0) / P + eps)
    l = tl.arange(0, K)
    pidx = gk[:, None, None] * K + l[None, None, :]
    y3 = xc * rs[None, :, None] * tl.load(GAM + pidx) + tl.load(BET + pidx)
    y = tl.reshape(tl.permute(y3, (1, 0, 2)), [BJ, P])
    j = j0 + tl.arange(0, BJ)
    tl.store(Y + (i * L + j)[:, None] * P + tl.arange(0, P)[None, :], y.to(tl.bfloat16), mask=(j < L)[:, None])


@triton.autotune(configs=configs_for("outer_product_mean_bwd_epilogue_ln_dx_triton"), key=["shape_key", "P"],
                 reset_to_zero=["DGAM", "DBET"])
@triton.jit
def _opm_grouped_bwd_epilogue_dx_kernel(DY, O, GAM, DO, DGAM, DBET, L, eps, shape_key,
                                        K: tl.constexpr, P: tl.constexpr, BJ: tl.constexpr, PPS: tl.constexpr):
    # persistent over (row i, BJ columns j) tiles: the product LayerNorm's backward from dy = dz W_out ([L L, P], cuBLAS) and the recomputed
    # statistics of O -> dO scattered into the grouped layout (coalesced along (j, l)); dgamma / dbeta stay in registers until the end
    GK: tl.constexpr = P // K
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    gk = tl.arange(0, GK)
    jl = tl.arange(0, BJ * K)
    l = tl.arange(0, K)
    pidx = gk[:, None, None] * K + l[None, None, :]
    gam = tl.load(GAM + pidx)
    lk = L.to(tl.int64) * K
    goff = (gk // K).to(tl.int64)[:, None] * lk * lk
    dgs = tl.zeros((GK, K), tl.float32)
    dbs = tl.zeros((GK, K), tl.float32)
    nj = tl.cdiv(L, BJ)
    for t in range(pid, L * nj, nprog):
        i = (t // nj).to(tl.int64)
        j0 = (t % nj) * BJ
        jm = ((j0 + jl // K) < L)[None, :]
        off = goff + ((gk % K) + i * K).to(tl.int64)[:, None] * lk + (j0 * K + jl)[None, :]
        x3 = tl.reshape(tl.load(O + off, mask=jm, other=0.0).to(tl.float32), [GK, BJ, K])
        mu = tl.sum(tl.sum(x3, axis=2), axis=0) / P
        xc = x3 - mu[None, :, None]
        rs = tl.rsqrt(tl.sum(tl.sum(xc * xc, axis=2), axis=0) / P + eps)
        xh = xc * rs[None, :, None]
        j = j0 + tl.arange(0, BJ)
        dyr = tl.load(DY + (i * L + j)[:, None] * P + tl.arange(0, P)[None, :], mask=(j < L)[:, None], other=0.0).to(tl.float32)
        dy = tl.permute(tl.reshape(dyr, [BJ, GK, K]), (1, 0, 2))
        dgs += tl.sum(dy * xh, axis=1)
        dbs += tl.sum(dy, axis=1)
        dxh = dy * gam
        m1 = tl.sum(tl.sum(dxh, axis=2), axis=0) / P
        m2 = tl.sum(tl.sum(dxh * xh, axis=2), axis=0) / P
        dp = rs[None, :, None] * (dxh - m1[None, :, None] - xh * m2[None, :, None])
        tl.store(DO + off, tl.reshape(dp, [GK, BJ * K]).to(tl.bfloat16), mask=jm)
    pg = gk[:, None] * K + l[None, :]
    tl.atomic_add(DGAM + pg, dgs)
    tl.atomic_add(DBET + pg, dbs)


@triton.autotune(configs=configs_for("outer_product_mean_bwd_epilogue_ln_dw_triton"), key=["shape_key", "P"], reset_to_zero=["DWO"],
                 prune_configs_by={"early_config_prune": _divides(BM="CZ", BN="P")})
@triton.jit
def _opm_grouped_bwd_epilogue_dw_kernel(DZ, Y, DWO, NPAIR, shape_key,
                                        P: tl.constexpr, CZ: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                                        SPLIT: tl.constexpr):
    # dWo[c, p] = sum_pairs dz[pair, c] y[pair, p]: M = CZ, N = P, K = L^2 pairs split over SPLIT ranges (fp32 atomics once per program)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    per = tl.cdiv(NPAIR, SPLIT)
    q0 = tl.program_id(2) * per
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for qq in range(0, per, BK):
        q = q0 + qq + kk
        qm = (q < NPAIR) & (qq + kk < per)
        a = tl.load(DZ + q[:, None].to(tl.int64) * CZ + m[None, :], mask=qm[:, None], other=0.0)         # [BK, BM]
        b = tl.load(Y + q[:, None].to(tl.int64) * P + n[None, :], mask=qm[:, None], other=0.0)           # [BK, BN]
        acc += tl.dot(tl.trans(a), b)
    tl.atomic_add(DWO + m[:, None] * P + n[None, :], acc)


# ---------------------------------------------------------------------------------------------------- launches (opaque to Dynamo)
def _pow2_16(n: int) -> bool:
    return n >= 16 and n & (n - 1) == 0


def refusal(msa: torch.Tensor, d_hidden: int, d_pair: int, *, interchain: bool, residual: torch.Tensor | None = None,
            groups: int = 1) -> str | None:
    """None if the Triton path can run this OuterProductMean call, else why it cannot. Never raises."""
    try:
        if interchain:
            return "mask_interchain is applied after the projection and is not fused here"
        if not msa.is_cuda:
            return "the input is not on a CUDA device"
        if msa.dtype != torch.bfloat16:
            return f"the Triton path is bf16, got {msa.dtype}"
        if residual is not None and (residual.dtype != torch.bfloat16 or residual.shape != (msa.shape[0], msa.shape[2], msa.shape[2], d_pair)):
            return "the pair residual must be bf16 and [B, L, L, d_pair] to be fused"
        d_msa = msa.shape[-1]
        if not (_pow2_16(d_msa) and _pow2_16(d_hidden)):
            return f"d_msa and d_hidden must be powers of two >= 16 (spanned by one tile), got {d_msa}, {d_hidden}"
        if d_pair % 32:
            return f"d_pair must be a multiple of 32 (the smallest output tile), got {d_pair}"
        if groups > 1:
            if d_hidden % groups:
                return f"d_hidden ({d_hidden}) must be divisible by groups ({groups})"
            products = groups * (d_hidden // groups) ** 2
            if not _pow2_16(products) or products > 1024 or groups & (groups - 1):
                return (f"the grouped products (groups * (d_hidden / groups)^2 = {products}) must be a power of two in 16..1024 (a tile "
                        f"spans them) and groups a power of two, got groups {groups}")
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _fwd_fake(msa, mask, lnw, lnb, wl, wr, bl, br, wo, bo, residual, eps, norm_first):
    """Shapes of `_fwd`'s outputs: out [L, L, CZ], a / b [S L, CH], O [L CH, L CH] bf16, n [L, L] fp32."""
    S, L, _ = msa.shape
    CH, CZ = wl.shape[0], wo.shape[0]
    new = lambda *shape, dt=torch.bfloat16: msa.new_empty(shape, dtype=dt)  # noqa: E731
    return [new(L, L, CZ), new(S * L, CH), new(S * L, CH), new(L * CH, L * CH), new(L, L, dt=torch.float32)]


@opaque(fake=_fwd_fake, name="outer_product_mean_triton_fwd")
def _fwd(msa: torch.Tensor, mask: torch.Tensor, lnw: torch.Tensor, lnb: torch.Tensor, wl: torch.Tensor, wr: torch.Tensor,
         bl: torch.Tensor, br: torch.Tensor, wo: torch.Tensor, bo: torch.Tensor, residual: torch.Tensor, eps: float,
         norm_first: bool) -> list[torch.Tensor]:
    """One stack: msa [S, L, CM], mask [S, L] bool (or empty: no mask), residual [L, L, CZ] (or empty), bl / br the projections'
    biases [CH] (or empty). Returns [out [L, L, CZ], a, b [S L, CH], O [L CH, L CH], n [L, L]]; the last four are what the backward reads."""
    S, L, CM = msa.shape
    CH, CZ = wl.shape[0], wo.shape[0]
    T_ = S * L
    bf = torch.bfloat16
    has_mask, has_res, has_pbias = mask.numel() > 0, residual.numel() > 0, bl.numel() > 0
    x = msa.reshape(T_, CM).contiguous()
    m8 = mask.reshape(T_).contiguous().view(torch.uint8) if has_mask else x
    w_l, w_r = wl.to(bf).contiguous(), wr.to(bf).contiguous()
    b_l, b_r = (bl.float().contiguous(), br.float().contiguous()) if has_pbias else (x, x)
    a = msa.new_empty((T_, CH), dtype=bf)
    b = msa.new_empty((T_, CH), dtype=bf)
    _opm_layernorm_gemm_kernel[lambda meta: (triton.cdiv(T_, meta["BT"]),)](
        x, m8, lnw.float().contiguous(), lnb.float().contiguous(), w_l, w_r, b_l, b_r, a, b, T_, eps,
        shape_key=token_key(length_of(msa.shape), CM=CM, CH=CH), CM=CM, CH=CH, K=CH, HAS_MASK=has_mask, HAS_PBIAS=has_pbias)
    o = torch.mm(a.view(S, L * CH).t(), b.view(S, L * CH))                                  # [(i, d), (j, e)]
    if has_mask:
        mf = mask.reshape(S, L).float()
        norm = (mf.t() @ mf).clamp_(min=1)
    else:
        norm = msa.new_full((L, L), float(S), dtype=torch.float32)
    out = msa.new_empty((L, L, CZ), dtype=bf)
    res = residual.reshape(L * L, CZ).contiguous() if has_res else out
    _opm_epilogue_kernel[lambda meta: (triton.cdiv(L, meta["TI"]), triton.cdiv(L, meta["BJ"]), triton.cdiv(CZ, meta["BN"]))](
        o, wo.to(bf).t().contiguous(), bo.float().contiguous(), norm, res, out, L,
        shape_key=token_key(length_of(msa.shape), CH=CH, CZ=CZ), CH=CH, CZ=CZ, HAS_RES=has_res, NORM_FIRST=norm_first)
    return [out, a, b, o, norm]


def _bwd_fake(dz, msa, mask, a, b, o, norm, lnw, lnb, wl, wr, bl, br, wo, eps, norm_first):
    """Shapes of `_bwd`'s outputs: dmsa like msa, every parameter gradient fp32 in its parameter's shape."""
    f32 = torch.float32
    return [torch.empty_like(msa), lnw.new_empty(lnw.shape, dtype=f32), lnb.new_empty(lnb.shape, dtype=f32),
            wl.new_empty(wl.shape, dtype=f32), wr.new_empty(wr.shape, dtype=f32), wo.new_empty(wo.shape, dtype=f32),
            wo.new_empty((wo.shape[0],), dtype=f32), wl.new_empty((wl.shape[0],), dtype=f32), wl.new_empty((wl.shape[0],), dtype=f32)]


@opaque(fake=_bwd_fake, name="outer_product_mean_triton_bwd")
def _bwd(dz: torch.Tensor, msa: torch.Tensor, mask: torch.Tensor, a: torch.Tensor, b: torch.Tensor, o: torch.Tensor,
         norm: torch.Tensor, lnw: torch.Tensor, lnb: torch.Tensor, wl: torch.Tensor, wr: torch.Tensor, bl: torch.Tensor,
         br: torch.Tensor, wo: torch.Tensor, eps: float, norm_first: bool) -> list[torch.Tensor]:
    """Gradients of one stack: [dmsa, dgamma, dbeta, dW_left, dW_right, dW_out, db_out, db_left, db_right] (fp32 but dmsa; the last
    two are zeros when the projections have no bias)."""
    S, L, CM = msa.shape
    CH, CZ = wl.shape[0], wo.shape[0]
    T_ = S * L
    bf = torch.bfloat16
    has_mask, has_pbias = mask.numel() > 0, bl.numel() > 0
    x = msa.reshape(T_, CM).contiguous()
    m8 = mask.reshape(T_).contiguous().view(torch.uint8) if has_mask else x
    dzc = dz.reshape(L * L, CZ).contiguous()
    dzn = dz.new_empty((L * L, CZ), dtype=bf)
    dO = dz.new_empty((L * CH, L * CH), dtype=bf)
    _opm_bwd_epilogue_dx_kernel[lambda meta: (triton.cdiv(L, meta["TI"]), triton.cdiv(L, meta["BJ"]), triton.cdiv(CH * CH, meta["BN"]))](
        dzc, norm, wo.to(bf).contiguous(), dzn, dO, L, CH * CH,
        shape_key=token_key(length_of(msa.shape), CH=CH, CZ=CZ), CH=CH, CZ=CZ)
    # the bias gradient is a column sum over the L L pairs: one fp32-accumulated GEMV (ones . dz), not an ATen axis-0 reduction (66 us at L = 256,
    # 140 us with the fp32 division of the ESMFold2 order, where the bias is divided too: it sums dz / n, which dzn already holds)
    src = dzc if norm_first else dzn
    dbo = torch.mm(src.new_ones((1, L * L)), src, out_dtype=torch.float32).reshape(-1)
    dA = torch.mm(b.view(S, L * CH), dO.t())                                                # [S, (i, d)]
    dB = torch.mm(a.view(S, L * CH), dO)                                                    # [S, (j, e)]
    del dO
    dwo = dz.new_zeros((CZ, CH * CH), dtype=torch.float32)
    _opm_bwd_epilogue_dw_kernel[lambda meta: (triton.cdiv(CZ, meta["BM"]), triton.cdiv(CH * CH, meta["BN"]), meta["SPLIT"])](
        dzn, o, dwo, L, CH * CH, shape_key=token_key(length_of(msa.shape), CH=CH, CZ=CZ), CH=CH, CZ=CZ)
    dm, dg, dbeta, dwl, dwr, dbl, dbr = _ln_gemm_bwd(x, m8, lnw, lnb, wl, wr, dA, dB, eps, S, L, CM, CH, CH, has_mask, has_pbias)
    return [dm, dg, dbeta, dwl, dwr, dwo, dbo, dbl, dbr]


def _ln_gemm_bwd(x, m8, lnw, lnb, wl, wr, dA, dB, eps, S, L, CM, CH, K, has_mask, has_pbias):
    """The shared tail of both backwards: mask, dy = [dA | dB] W, dW_left / dW_right (and the biases' gradients), LayerNorm backward.
    dA / dB are [G, S L, K] (K = CH: [S L, CH]). Returns [dmsa [S L, CM], dgamma, dbeta, dW_left, dW_right, db_left, db_right]."""
    bf = torch.bfloat16
    T_ = S * L
    dm = torch.empty_like(x)
    red = x.new_zeros((2 * CH * CM + 2 * CM + 2 * CH,), dtype=torch.float32)
    dwl, dwr = red[:CH * CM].view(CH, CM), red[CH * CM:2 * CH * CM].view(CH, CM)
    dg, dbeta = red[2 * CH * CM:2 * CH * CM + CM], red[2 * CH * CM + CM:2 * CH * CM + 2 * CM]
    dbl, dbr = red[2 * CH * CM + 2 * CM:2 * CH * CM + 2 * CM + CH], red[2 * CH * CM + 2 * CM + CH:]
    nsm = num_sms(x.device)
    _opm_bwd_layernorm_gemm_kernel[lambda meta: (min(triton.cdiv(T_, meta["BT"]), meta["PPS"] * nsm),)](
        x, m8, lnw.float().contiguous(), lnb.float().contiguous(), wl.to(bf).contiguous(), wr.to(bf).contiguous(),
        dA, dB, dm, dwl, dwr, dg, dbeta, dbl, dbr, T_, eps,
        shape_key=token_key(length_of((S, L, CM)), CM=CM, CH=CH, **({} if K == CH else {"G": CH // K})), CM=CM, CH=CH, K=K,
        HAS_MASK=has_mask, HAS_PBIAS=has_pbias)
    return dm.view(S, L, CM), dg.clone(), dbeta.clone(), dwl.clone(), dwr.clone(), dbl.clone(), dbr.clone()


# ---------------------------------------------------------------------------------------------------- grouped launches
def _fwd_grouped_fake(msa, mask, lnw, lnb, wl, wr, bl, br, gam, bet, wo, bo, residual, eps, eps_prod, groups):
    """Shapes of `_fwd_grouped`'s outputs: out [L, L, CZ], a / b [S L, CH] (group-major), O [G, L K, L K], y [L L, P] bf16."""
    S, L, _ = msa.shape
    CH, CZ = wl.shape[0], wo.shape[0]
    K = CH // groups
    new = lambda *shape: msa.new_empty(shape, dtype=torch.bfloat16)  # noqa: E731
    return [new(L, L, CZ), new(S * L, CH), new(S * L, CH), new(groups, L * K, L * K), new(L * L, groups * K * K)]


@opaque(fake=_fwd_grouped_fake, name="outer_product_mean_triton_fwd_grouped")
def _fwd_grouped(msa: torch.Tensor, mask: torch.Tensor, lnw: torch.Tensor, lnb: torch.Tensor, wl: torch.Tensor, wr: torch.Tensor,
                 bl: torch.Tensor, br: torch.Tensor, gam: torch.Tensor, bet: torch.Tensor, wo: torch.Tensor, bo: torch.Tensor,
                 residual: torch.Tensor, eps: float, eps_prod: float, groups: int) -> list[torch.Tensor]:
    """One stack of the grouped (Chai-1) variant. Returns [out [L, L, CZ], a, b [S L, CH] (stored [G, S L, K]), O [G, L K, L K],
    y [L L, P]: the LayerNorm'd products, which dW_out reads]."""
    S, L, CM = msa.shape
    CH, CZ = wl.shape[0], wo.shape[0]
    K = CH // groups
    P = groups * K * K
    T_ = S * L
    bf = torch.bfloat16
    has_mask, has_res, has_pbias = mask.numel() > 0, residual.numel() > 0, bl.numel() > 0
    x = msa.reshape(T_, CM).contiguous()
    m8 = mask.reshape(T_).contiguous().view(torch.uint8) if has_mask else x
    b_l, b_r = (bl.float().contiguous(), br.float().contiguous()) if has_pbias else (x, x)
    a = msa.new_empty((T_, CH), dtype=bf)
    b = msa.new_empty((T_, CH), dtype=bf)
    _opm_layernorm_gemm_kernel[lambda meta: (triton.cdiv(T_, meta["BT"]),)](
        x, m8, lnw.float().contiguous(), lnb.float().contiguous(), wl.to(bf).contiguous(), wr.to(bf).contiguous(), b_l, b_r, a, b, T_, eps,
        shape_key=token_key(length_of(msa.shape), CM=CM, CH=CH, G=groups), CM=CM, CH=CH, K=K, HAS_MASK=has_mask, HAS_PBIAS=has_pbias)
    o = torch.bmm(a.view(groups, S, L * K).transpose(1, 2), b.view(groups, S, L * K))      # [G, (i, k), (j, l)]
    y = msa.new_empty((L * L, P), dtype=bf)
    _opm_grouped_epilogue_kernel[lambda meta: (L, triton.cdiv(L, meta["BJ"]))](
        o, gam.float().contiguous(), bet.float().contiguous(), y, L, eps_prod,
        shape_key=token_key(length_of(msa.shape), K=K, G=groups), K=K, P=P)
    out = torch.nn.functional.linear(y, wo.to(bf), bo.to(bf))                                # cuBLAS, the bias in its epilogue
    if has_res:
        out.add_(residual.reshape(L * L, CZ))
    return [out.view(L, L, CZ), a, b, o, y]


def _bwd_grouped_fake(dz, msa, mask, a, b, o, y, lnw, lnb, wl, wr, bl, br, gam, bet, wo, eps, eps_prod, groups):
    """Shapes of `_bwd_grouped`'s outputs: dmsa like msa, every parameter gradient fp32 in its parameter's shape."""
    f32 = torch.float32
    return [torch.empty_like(msa)] + [t.new_empty(t.shape, dtype=f32) for t in (lnw, lnb, wl, wr, wo, gam, bet)] + \
        [wo.new_empty((wo.shape[0],), dtype=f32), wl.new_empty((wl.shape[0],), dtype=f32), wl.new_empty((wl.shape[0],), dtype=f32)]


@opaque(fake=_bwd_grouped_fake, name="outer_product_mean_triton_bwd_grouped")
def _bwd_grouped(dz: torch.Tensor, msa: torch.Tensor, mask: torch.Tensor, a: torch.Tensor, b: torch.Tensor, o: torch.Tensor,
                 y: torch.Tensor, lnw: torch.Tensor, lnb: torch.Tensor, wl: torch.Tensor, wr: torch.Tensor, bl: torch.Tensor, br: torch.Tensor,
                 gam: torch.Tensor, bet: torch.Tensor, wo: torch.Tensor, eps: float, eps_prod: float,
                 groups: int) -> list[torch.Tensor]:
    """Gradients of one grouped stack: [dmsa, dgamma, dbeta, dW_left, dW_right, dW_out, dgamma_prod, dbeta_prod, db_out, db_left, db_right]."""
    S, L, CM = msa.shape
    CH, CZ = wl.shape[0], wo.shape[0]
    K = CH // groups
    P = groups * K * K
    T_ = S * L
    bf = torch.bfloat16
    f32 = torch.float32
    has_mask, has_pbias = mask.numel() > 0, bl.numel() > 0
    x = msa.reshape(T_, CM).contiguous()
    m8 = mask.reshape(T_).contiguous().view(torch.uint8) if has_mask else x
    dzc = dz.reshape(L * L, CZ).contiguous()
    dO = dz.new_empty((groups, L * K, L * K), dtype=bf)
    dgam, dbet = dz.new_zeros((P,), dtype=f32), dz.new_zeros((P,), dtype=f32)
    nsm = num_sms(msa.device)
    dy = torch.mm(dzc, wo.to(bf))                                                           # [L L, P], cuBLAS
    _opm_grouped_bwd_epilogue_dx_kernel[lambda meta: (max(1, min(L * triton.cdiv(L, meta["BJ"]), meta["PPS"] * nsm)),)](
        dy, o, gam.float().contiguous(), dO, dgam, dbet, L, eps_prod,
        shape_key=token_key(length_of(msa.shape), K=K, G=groups), K=K, P=P)
    del dy
    dbo = dzc.sum(0, dtype=torch.float32)
    dA = torch.bmm(b.view(groups, S, L * K), dO.transpose(1, 2))                            # [G, S, (i, k)]
    dB = torch.bmm(a.view(groups, S, L * K), dO)                                            # [G, S, (j, l)]
    del dO
    dwo = dz.new_zeros((CZ, P), dtype=f32)
    _opm_grouped_bwd_epilogue_dw_kernel[lambda meta: (triton.cdiv(CZ, meta["BM"]), triton.cdiv(P, meta["BN"]), meta["SPLIT"])](
        dzc, y, dwo, L * L, shape_key=token_key(length_of(msa.shape), K=K, G=groups, CZ=CZ), P=P, CZ=CZ)
    dm, dg, dbeta, dwl, dwr, dbl, dbr = _ln_gemm_bwd(x, m8, lnw, lnb, wl, wr, dA, dB, eps, S, L, CM, CH, K, has_mask, has_pbias)
    return [dm, dg, dbeta, dwl, dwr, dwo, dgam, dbet, dbo, dbl, dbr]


class _OuterProductMeanTriton(torch.autograd.Function):
    """One MSA stack (no batch axis). ``mask`` / ``residual`` / ``bl`` / ``br`` are empty tensors when absent."""

    @staticmethod
    def forward(ctx, msa, mask, lnw, lnb, wl, wr, bl, br, wo, bo, residual, eps, norm_first):
        out, a, b, o, norm = _fwd(msa, mask, lnw, lnb, wl, wr, bl, br, wo, bo, residual, eps, norm_first)
        ctx.save_for_backward(msa, mask, a, b, o, norm, lnw, lnb, wl, wr, bl, br, wo, bo)
        ctx.eps, ctx.norm_first, ctx.has_res = eps, norm_first, residual.numel() > 0
        return out

    @staticmethod
    def backward(ctx, dz):
        msa, mask, a, b, o, norm, lnw, lnb, wl, wr, bl, br, wo, bo = ctx.saved_tensors
        dm, dg, dbeta, dwl, dwr, dwo, dbo, dbl, dbr = _bwd(dz.contiguous(), msa, mask, a, b, o, norm, lnw, lnb, wl, wr, bl, br, wo,
                                                           ctx.eps, ctx.norm_first)
        return (dm, None, dg.to(lnw.dtype), dbeta.to(lnb.dtype), dwl.to(wl.dtype), dwr.to(wr.dtype),
                dbl.to(bl.dtype) if bl.numel() else None, dbr.to(br.dtype) if br.numel() else None, dwo.to(wo.dtype),
                dbo.to(bo.dtype), dz if ctx.has_res else None, None, None)


class _OuterProductMeanGroupedTriton(torch.autograd.Function):
    """One MSA stack of the grouped (Chai-1) variant: ``groups`` outer products, LayerNorm'd products, no pair-count division."""

    @staticmethod
    def forward(ctx, msa, mask, lnw, lnb, wl, wr, bl, br, gam, bet, wo, bo, residual, eps, eps_prod, groups):
        out, a, b, o, y = _fwd_grouped(msa, mask, lnw, lnb, wl, wr, bl, br, gam, bet, wo, bo, residual, eps, eps_prod, groups)
        ctx.save_for_backward(msa, mask, a, b, o, y, lnw, lnb, wl, wr, bl, br, gam, bet, wo, bo)
        ctx.meta, ctx.has_res = (eps, eps_prod, groups), residual.numel() > 0
        return out

    @staticmethod
    def backward(ctx, dz):
        msa, mask, a, b, o, y, lnw, lnb, wl, wr, bl, br, gam, bet, wo, bo = ctx.saved_tensors
        dm, dg, dbeta, dwl, dwr, dwo, dgam, dbet, dbo, dbl, dbr = _bwd_grouped(dz.contiguous(), msa, mask, a, b, o, y, lnw, lnb, wl, wr, bl,
                                                                               br, gam, bet, wo, *ctx.meta)
        return (dm, None, dg.to(lnw.dtype), dbeta.to(lnb.dtype), dwl.to(wl.dtype), dwr.to(wr.dtype),
                dbl.to(bl.dtype) if bl.numel() else None, dbr.to(br.dtype) if br.numel() else None, dgam.to(gam.dtype),
                dbet.to(bet.dtype), dwo.to(wo.dtype), dbo.to(bo.dtype), dz if ctx.has_res else None, None, None, None)


def triton_outer_product_mean(msa: torch.Tensor, mask: torch.Tensor | None, ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                              w_left: torch.Tensor, w_right: torch.Tensor, w_out: torch.Tensor, b_out: torch.Tensor,
                              residual: torch.Tensor | None = None, *, eps: float = 1e-5, normalize_before_proj: bool = True,
                              b_left: torch.Tensor | None = None, b_right: torch.Tensor | None = None, groups: int = 1,
                              product_norm_weight: torch.Tensor | None = None, product_norm_bias: torch.Tensor | None = None,
                              product_norm_eps: float = 0.1) -> torch.Tensor:
    """``residual + OPM(msa)`` (``OPM(msa)`` without a residual), differentiable. msa [B, S, L, CM] bf16, mask [B, S, L] bool.

    ``b_left`` / ``b_right`` (RoseTTAFold3): the projections' biases, masked rows lose them too. ``groups`` > 1 (Chai-1): the hidden
    channels split into ``groups`` halves with one outer product each, their ``groups * (CH / groups)^2`` products LayerNorm'd
    (``product_norm_*``, eps ``product_norm_eps``), projected, and NOT divided by the pair counts (``normalize_before_proj`` is moot).

    Stacks run one at a time: the mask counts and the grouped outer product are per stack. Check :func:`refusal` first."""
    def empty():                                           # one tensor per absent input: Dynamo refuses an autograd.Function
        return msa.new_empty((0,))                         # applied to the same tensor twice

    bl, br = (empty(), empty()) if b_left is None else (b_left, b_right)
    outs = []
    for bi in range(msa.shape[0]):
        m = empty() if mask is None else mask[bi]
        r = empty() if residual is None else residual[bi]
        if groups > 1:
            outs.append(_OuterProductMeanGroupedTriton.apply(msa[bi], m, ln_weight, ln_bias, w_left, w_right, bl, br, product_norm_weight,
                                                             product_norm_bias, w_out, b_out, r, float(eps), float(product_norm_eps),
                                                             int(groups)))
        else:
            outs.append(_OuterProductMeanTriton.apply(msa[bi], m, ln_weight, ln_bias, w_left, w_right, bl, br, w_out, b_out, r,
                                                      float(eps), bool(normalize_before_proj)))
    return outs[0].unsqueeze(0) if len(outs) == 1 else torch.stack(outs, 0)
