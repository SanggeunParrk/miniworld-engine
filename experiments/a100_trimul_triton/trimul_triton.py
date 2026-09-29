"""TriMul (single / bidirectional) forward + training backward in Triton: the fusion of the A100 CUDA kernels (../a100_trimul_fwd, the
sm_90 algorithm), with Triton-specific tiling.

forward   _k1   LN_in(z) -> x_n; (g, p) = x_n . [Wg | Wp] in one dot per output tile (interleaved columns); a = sigmoid(g) p m_i m_j
                -> channel-major planes a | b; saves the LN_in (mean, rstd) of every token
          contraction  cuBLAS per plane channel (outgoing A B^T, incoming A^T B; bidirectional: one half each)
          _k3   LN_out(X) . W_o^T  *  sigmoid(x_n . W_og^T)  (* ds[j])  + z, output width tiled; training also saves the four LN statistics + x_n
backward  _b1e  (the forward's statistics and x_n) o, g recomputed per output tile; d_o, d_g, A_o = d_o r_o, the fold vectors
          _b1d  dx_hat = (d_o . W_o) diag g_o as a GEMM, the LN_out backward in its epilogue -> dX planes
          weight gradients: W_o (split-K G = A_o^T X^T + rank-1 LayerNorm folds), W_og = d_g^T x_n -- cuBLAS on a side stream under the
                contraction backward
          contraction backward (cuBLAS)
          input side (split, default): _src (GEMM grid: (g, p) recomputed, dg / dp -> the operand buffer) + dW = dgp^T x_n (split-K cuBLAS,
                side stream) + _con (dx_n = [dg | dp | d_g] . [W_in ; W_og], LN_in backward + residual -> dz)
          input side (joint, TT_JOINT=1): _b7j, one launch of source + consumer programs meeting in an L2 ring (release / acquire flags).
                Triton has no cooperative launch, so co-residency rests on an occupancy estimate; measured slower than the split path.

Autotune: every kernel carries the repo-width ladder (TT_AUTOTUNE=dev selects a small development set); configs impossible for the live
shape (a tile wider than, or not dividing, the dimension it tiles) are pruned per call.  Plane offsets are 64-bit; every kernel masks the
token tail.
"""
import itertools
import os

import torch
import triton
import triton.language as tl

PARAMS = ("ln_pair.weight", "ln_pair.bias", "to_left_gate.weight", "to_left.weight", "to_right_gate.weight", "to_right.weight",
          "ln_out.weight", "ln_out.bias", "to_gate.weight", "to_out.weight")
RESERVED_SMEM_PER_BLOCK = 1024     # sm_80+ (cudaDevAttrReservedSharedMemoryPerBlock), for the joint kernel's residency estimate
_OVERLAP = os.environ.get("TT_OVERLAP", "1") == "1"
_JOINT = os.environ.get("TT_JOINT", "0") == "1"
_DEV = os.environ.get("TT_AUTOTUNE", "") == "dev"
# the joint kernel's static configuration (not autotuned: its grid must fit on the device at once)
JOINT = {256: dict(SW=32, BM=128, BK=64, num_warps=8, num_stages=2, C=8, rings=8),
         128: dict(SW=32, BM=128, BK=64, num_warps=8, num_stages=2, C=6, rings=8)}
GEMM_TK = dict(chunks=(32, 16, 8, 4, 2), min_chunk=256)


# ============================================================================================================ autotune spaces
def _space(wide, dev):
    """triton.Config list from {axis: values}; num_warps / num_stages are launch options.  The wide ladder is the default (repo rule: the
    shipped space is not hand-narrowed); TT_AUTOTUNE=dev selects the development set."""
    axes = dev if _DEV else wide
    out = []
    for vals in itertools.product(*axes.values()):
        kw = dict(zip(axes, vals))
        nw, ns = kw.pop("num_warps"), kw.pop("num_stages")
        out.append(triton.Config(kw, num_warps=nw, num_stages=ns))
    return out


def _fits(**limits):
    """early_config_prune: drop configs whose tile exceeds, or does not divide, the live dimension it tiles."""
    def prune(configs, named_args, **kwargs):
        args = {**named_args, **kwargs}
        keep = [c for c in configs if all(c.kwargs[k] <= args[dim] and args[dim] % c.kwargs[k] == 0 for k, dim in limits.items())]
        return keep or configs[:1]
    return {"early_config_prune": prune}


# wide ladders (the breadth of the repo's grid sets for the same kinds of kernel) and the development sets
W_GEMM_GATE = dict(BM=(32, 64, 128), BN=(16, 32, 64), num_warps=(1, 2, 4, 8), num_stages=(1, 2, 3, 4, 5, 6, 8, 10))
W_OUTPROJ = dict(BM=(32, 64, 128), BN=(32, 64, 128), num_warps=(1, 2, 4, 8), num_stages=(1, 2, 3, 4))
W_LN_GEMM = dict(BM=(16, 32, 64, 128), BK=(16, 32, 64, 128), num_warps=(1, 2, 4, 8), num_stages=(1, 2, 3, 4, 5, 6))
W_SRC = dict(BM=(32, 64, 128), SW=(16, 32, 64, 128), num_warps=(1, 2, 4, 8), num_stages=(1, 2, 3, 4, 5, 6))
W_CON = dict(BM=(32, 64, 128), BK=(16, 32, 64, 128), WAVES=(1, 2, 4), num_warps=(1, 2, 4, 8), num_stages=(1, 2, 3, 4, 5, 6))
D_GEMM_GATE = dict(BM=(64, 128), BN=(32, 64), num_warps=(4, 8), num_stages=(1, 2, 3))
D_OUTPROJ = dict(BM=(32, 64), BN=(32, 64), num_warps=(4, 8), num_stages=(1,))
D_B1E = dict(BM=(32, 64, 128), BN=(32, 64), num_warps=(4, 8), num_stages=(1, 2))
D_LN_GEMM = dict(BM=(32, 64, 128), BK=(32, 64, 128), num_warps=(4, 8), num_stages=(1, 2, 3))
D_SRC = dict(BM=(64, 128), SW=(32, 64), num_warps=(4, 8), num_stages=(1, 2))
D_CON = dict(BM=(64, 128), BK=(32, 64), WAVES=(1, 2), num_warps=(4, 8), num_stages=(2, 3, 4))


@triton.jit
def _tanh(x):
    """tanh.approx.f32 (the CUDA kernels' sigmoid: s = 1/2 + tanh(g/2)/2, s (1 - s) = (1 - tanh(g/2)^2) / 4); tl.sigmoid compiles to an
    IEEE division."""
    return tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1)


# ============================================================================================================ forward
@triton.autotune(configs=_space(W_GEMM_GATE, D_GEMM_GATE), key=["T", "NP"], prune_configs_by=_fits(BN="NP"))
@triton.jit
def _k1(z, mask, w1i, gin, bin_, ab, zst, T, L, eps,
        D: tl.constexpr, NP: tl.constexpr, HAS_MASK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """LN_in + gated projections; w1i [D, 2 NP] with a plane channel's gate / projection columns interleaved (2c, 2c + 1)."""
    pid = tl.program_id(0)
    rt = pid.to(tl.int64) * BM + tl.arange(0, BM)
    rc = tl.arange(0, D)
    ok = rt < T
    x = tl.load(z + rt[:, None] * D + rc[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, 1) / D
    xc = x - mean[:, None]
    rstd = tl.rsqrt(tl.sum(xc * xc, 1) / D + eps)
    tl.store(zst + rt * 2, mean, mask=ok)
    tl.store(zst + rt * 2 + 1, rstd, mask=ok)
    xn = (xc * rstd[:, None] * tl.load(gin + rc)[None, :] + tl.load(bin_ + rc)[None, :]).to(tl.bfloat16)
    m = ok.to(tl.float32)
    if HAS_MASK:
        i = rt // L
        j = rt - i * L
        m = m * ((tl.load(mask + i, mask=ok, other=0) != 0) & (tl.load(mask + j, mask=ok, other=0) != 0)).to(tl.float32)
    c2 = tl.arange(0, 2 * BN)
    rn = tl.arange(0, BN)
    for n0 in range(0, NP, BN):
        acc = tl.dot(xn, tl.load(w1i + rc[:, None] * (2 * NP) + (2 * n0 + c2)[None, :]))       # [BM, 2 BN]: (g, p) interleaved
        g, p = tl.split(tl.reshape(acc, (BM, BN, 2)))
        a = (0.5 + 0.5 * _tanh(0.5 * g)) * p * m[:, None]
        tl.store(ab + (n0 + rn).to(tl.int64)[:, None] * T + rt[None, :], tl.trans(a).to(tl.bfloat16), mask=ok[None, :])


@triton.autotune(configs=_space(W_OUTPROJ, D_OUTPROJ), key=["T", "CH", "TRAIN"], prune_configs_by=_fits(BN="D"))
@triton.jit
def _k3(x, z, zst, wot, go, bo, wogt, gin, bin_, ds, out, st, xn_out, T, L, eps,
        D: tl.constexpr, CH: tl.constexpr, HAS_DS: tl.constexpr, TRAIN: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """LN_out + projection + gate + residual; wot = W_o^T [CH, D], wogt = W_og^T [D, D]; the output width tiled by BN."""
    pid = tl.program_id(0)
    rt = pid.to(tl.int64) * BM + tl.arange(0, BM)
    ok = rt < T
    rch = tl.arange(0, CH)
    rc = tl.arange(0, D)
    X = tl.load(x + rch.to(tl.int64)[None, :] * T + rt[:, None], mask=ok[:, None], other=0.0).to(tl.float32)   # [BM, CH]
    mu_o = tl.sum(X, 1) / CH
    Xc = X - mu_o[:, None]
    r_o = tl.rsqrt(tl.sum(Xc * Xc, 1) / CH + eps)
    y = (Xc * r_o[:, None] * tl.load(go + rch)[None, :] + tl.load(bo + rch)[None, :]).to(tl.bfloat16)
    mu_i = tl.load(zst + rt * 2, mask=ok, other=0.0)
    r_i = tl.load(zst + rt * 2 + 1, mask=ok, other=1.0)
    zt = tl.load(z + rt[:, None] * D + rc[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    xn = ((zt - mu_i[:, None]) * r_i[:, None] * tl.load(gin + rc)[None, :] + tl.load(bin_ + rc)[None, :]).to(tl.bfloat16)
    if TRAIN:
        tl.store(st + rt * 4, mu_o, mask=ok)
        tl.store(st + rt * 4 + 1, r_o, mask=ok)
        tl.store(st + rt * 4 + 2, mu_i, mask=ok)
        tl.store(st + rt * 4 + 3, r_i, mask=ok)
        tl.store(xn_out + rt[:, None] * D + rc[None, :], xn, mask=ok[:, None])
    rn = tl.arange(0, BN)
    for n0 in tl.static_range(0, D, BN):
        proj = tl.dot(y, tl.load(wot + rch[:, None] * D + (n0 + rn)[None, :]))
        g = tl.dot(xn, tl.load(wogt + rc[:, None] * D + (n0 + rn)[None, :]))
        upd = (0.5 + 0.5 * _tanh(0.5 * g)) * proj
        if HAS_DS:
            j = rt % L
            upd = upd.to(tl.bfloat16).to(tl.float32) * tl.load(ds + j[:, None] * D + (n0 + rn)[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        res = tl.load(z + rt[:, None] * D + (n0 + rn)[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        tl.store(out + rt[:, None] * D + (n0 + rn)[None, :], (res + upd).to(tl.bfloat16), mask=ok[:, None])


# ============================================================================================================ backward, output side
@triton.autotune(configs=_space(W_OUTPROJ, D_B1E), key=["T", "CH"], reset_to_zero=["vs"], prune_configs_by=_fits(BN="D"))
@triton.jit
def _b1e(x, xn, st, dy, ds, wot, go, bo, wogt, do_out, ao, dg, ldg, vs, T, L,
         D: tl.constexpr, CH: tl.constexpr, HAS_DS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """B1, per BN-wide output tile: o, g recomputed; d_o (for the dx GEMM), d_g (row stride ldg), A_o, the fold vectors (atomic)."""
    pid = tl.program_id(0)
    rt = pid.to(tl.int64) * BM + tl.arange(0, BM)
    ok = rt < T
    rch = tl.arange(0, CH)
    rc = tl.arange(0, D)
    rn = tl.arange(0, BN)
    mu = tl.load(st + rt * 4, mask=ok, other=0.0)
    r = tl.load(st + rt * 4 + 1, mask=ok, other=0.0)
    y = ((tl.load(x + rch.to(tl.int64)[None, :] * T + rt[:, None], mask=ok[:, None], other=0.0).to(tl.float32) - mu[:, None]) * r[:, None]
         * tl.load(go + rch)[None, :] + tl.load(bo + rch)[None, :]).to(tl.bfloat16)
    xnt = tl.load(xn + rt[:, None] * D + rc[None, :], mask=ok[:, None], other=0.0)
    for n0 in range(0, D, BN):
        o = tl.dot(y, tl.load(wot + rch[:, None] * D + (n0 + rn)[None, :]))
        th = _tanh(0.5 * tl.dot(xnt, tl.load(wogt + rc[:, None] * D + (n0 + rn)[None, :])))
        s = 0.5 + 0.5 * th
        du = tl.load(dy + rt[:, None] * D + (n0 + rn)[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        if HAS_DS:
            j = rt % L
            du = du * tl.load(ds + j[:, None] * D + (n0 + rn)[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        d_o = du * s
        off = rt[:, None] * D + (n0 + rn)[None, :]
        tl.store(do_out + off, d_o.to(tl.bfloat16), mask=ok[:, None])
        tl.store(dg + rt[:, None] * ldg + (n0 + rn)[None, :], (du * o * 0.25 * (1.0 - th * th)).to(tl.bfloat16), mask=ok[:, None])
        a_o = (d_o * r[:, None]).to(tl.bfloat16)
        tl.store(ao + off, a_o, mask=ok[:, None])
        tl.atomic_add(vs + n0 + rn, tl.sum(a_o.to(tl.float32) * mu[:, None], 0))
        tl.atomic_add(vs + D + n0 + rn, tl.sum(d_o, 0))


@triton.autotune(configs=_space(W_LN_GEMM, D_LN_GEMM), key=["T", "CH"], prune_configs_by=_fits(BK="D"))
@triton.jit
def _b1d(do_in, wo, go, x, st, dx, T, D: tl.constexpr, CH: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
    """B1, LN_out backward: dx_hat = (d_o . W_o) diag g_o as a GEMM (K = D), the LayerNorm backward in its epilogue -> dX planes."""
    pid = tl.program_id(0)
    rt = pid.to(tl.int64) * BM + tl.arange(0, BM)
    ok = rt < T
    rch = tl.arange(0, CH)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, CH), dtype=tl.float32)
    for k0 in range(0, D, BK):
        a = tl.load(do_in + rt[:, None] * D + (k0 + rk)[None, :], mask=ok[:, None], other=0.0)
        acc = tl.dot(a, tl.load(wo + (k0 + rk)[:, None] * CH + rch[None, :]), acc)
    mu = tl.load(st + rt * 4, mask=ok, other=0.0)
    r = tl.load(st + rt * 4 + 1, mask=ok, other=0.0)
    dxh = acc * tl.load(go + rch)[None, :]
    xh = (tl.load(x + rch.to(tl.int64)[None, :] * T + rt[:, None], mask=ok[:, None], other=0.0).to(tl.float32) - mu[:, None]) * r[:, None]
    m1 = tl.sum(dxh, 1) / CH
    m2 = tl.sum(dxh * xh, 1) / CH
    tl.store(dx + rch.to(tl.int64)[None, :] * T + rt[:, None], (r[:, None] * (dxh - m1[:, None] - xh * m2[:, None])).to(tl.bfloat16),
             mask=ok[:, None])


# ============================================================================================================ backward, input side
@triton.autotune(configs=_space(W_SRC, D_SRC), key=["T", "NP"], prune_configs_by=_fits(SW="NP"))
@triton.jit
def _src(xn, wgt, wpt, dab, mask, dgp, T, L, D: tl.constexpr, NP: tl.constexpr, HAS_MASK: tl.constexpr, BM: tl.constexpr, SW: tl.constexpr):
    """Source, GEMM grid (token tile x SW plane channels): g = x_n . Wg^T, p = x_n . Wp^T (wgt / wpt = [D, NP]); dg -> operand columns
    2 c0 .. 2 c0 + SW, dp -> 2 c0 + SW .. 2 c0 + 2 SW (the consumer weight uses the same blocked order for this SW)."""
    K: tl.constexpr = 2 * NP + D
    c0 = tl.program_id(1) * SW
    tok = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    ok = tok < T
    rc = tl.arange(0, D)
    rs = tl.arange(0, SW)
    xt = tl.load(xn + tok[:, None] * D + rc[None, :], mask=ok[:, None], other=0.0)
    gp = tl.dot(xt, tl.load(wgt + rc[:, None] * NP + (c0 + rs)[None, :]))
    pp = tl.dot(xt, tl.load(wpt + rc[:, None] * NP + (c0 + rs)[None, :]))
    dA = tl.load(dab + (c0 + rs).to(tl.int64)[None, :] * T + tok[:, None], mask=ok[:, None], other=0.0).to(tl.float32)
    if HAS_MASK:
        i = tok // L
        j = tok - i * L
        dA = dA * ((tl.load(mask + i, mask=ok, other=0) != 0) & (tl.load(mask + j, mask=ok, other=0) != 0)).to(tl.float32)[:, None]
    th = _tanh(0.5 * gp)
    hA = 0.5 * dA
    D0 = dgp + tok[:, None] * K + 2 * c0 + rs[None, :]
    tl.store(D0, (hA * pp * 0.5 * (1.0 - th * th)).to(tl.bfloat16), mask=ok[:, None])      # dg = dA p s (1 - s)
    tl.store(D0 + SW, (hA + hA * th).to(tl.bfloat16), mask=ok[:, None])                     # dp = dA s


@triton.jit
def _con_epilogue(acc, z, dy, st, gv, dz, t0, T, D: tl.constexpr, BM: tl.constexpr):
    """dx_n -> LN_in backward + residual -> dz (rows past T neither stored nor counted); returns the dgamma / dbeta contributions."""
    tok = t0 + tl.arange(0, BM)
    ok = tok < T
    rc = tl.arange(0, D)
    zt = tl.load(z + tok[:, None] * D + rc[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    mu = tl.load(st + tok * 4 + 2, mask=ok, other=0.0)
    rs = tl.load(st + tok * 4 + 3, mask=ok, other=0.0)
    zh = (zt - mu[:, None]) * rs[:, None]
    dxh = acc * gv[None, :]
    m1 = tl.sum(dxh, 1) / D
    m2 = tl.sum(dxh * zh, 1) / D
    dzv = rs[:, None] * (dxh - m1[:, None] - zh * m2[:, None]) + tl.load(dy + tok[:, None] * D + rc[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    tl.store(dz + tok[:, None] * D + rc[None, :], dzv.to(tl.bfloat16), mask=ok[:, None])
    return tl.sum(acc * zh, 0), tl.sum(acc, 0)          # rows past T: acc = 0 (zero A rows), zh = 0


@triton.autotune(configs=_space(W_CON, D_CON), key=["T", "K"], reset_to_zero=["ln"], prune_configs_by=_fits(BK="K"))
@triton.jit
def _con(dgp, wdx, z, dy, st, gin, dz, ln, T, n_sm, D: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr, WAVES: tl.constexpr):
    """Consumer, persistent over the token tiles: dx_n = operand . wdx, LN_in backward + residual; dgamma / dbeta added atomically into ln."""
    pid = tl.program_id(0)
    rt = tl.arange(0, BM)
    rc = tl.arange(0, D)
    rk = tl.arange(0, BK)
    gv = tl.load(gin + rc)
    dgam = tl.zeros((D,), dtype=tl.float32)
    dbet = tl.zeros((D,), dtype=tl.float32)
    for tile in range(pid, tl.cdiv(T, BM), tl.num_programs(0)):
        t0 = tile.to(tl.int64) * BM
        tok = t0 + rt
        ok = tok < T
        acc = tl.zeros((BM, D), dtype=tl.float32)
        for k0 in range(0, K, BK):
            a = tl.load(dgp + tok[:, None] * K + (k0 + rk)[None, :], mask=ok[:, None], other=0.0)
            acc = tl.dot(a, tl.load(wdx + (k0 + rk)[:, None] * D + rc[None, :]), acc)
        pg, pb = _con_epilogue(acc, z, dy, st, gv, dz, t0, T, D, BM)
        dgam += pg
        dbet += pb
    tl.atomic_add(ln + rc, dgam)
    tl.atomic_add(ln + D + rc, dbet)


# ---- joint input side (TT_JOINT=1)
@triton.jit
def _src_tile(xn, dab, mask, wgb, wpb, dwg, dwp, t0, T, L, D: tl.constexpr, SW: tl.constexpr, c0, HAS_MASK: tl.constexpr, BM: tl.constexpr):
    """One joint-source tile: (g, p) of SW plane channels over BM tokens, dg / dp (0 past T), the dW updates."""
    tok = t0 + tl.arange(0, BM)
    ok = tok < T
    rc = tl.arange(0, D)
    rs = tl.arange(0, SW)
    xt = tl.load(xn + tok[:, None] * D + rc[None, :], mask=ok[:, None], other=0.0)
    gp = tl.dot(xt, tl.trans(wgb))
    pp = tl.dot(xt, tl.trans(wpb))
    dA = tl.load(dab + (c0 + rs).to(tl.int64)[None, :] * T + tok[:, None], mask=ok[:, None], other=0.0).to(tl.float32)   # [BM, SW]
    if HAS_MASK:
        i = tok // L
        j = tok - i * L
        dA = dA * ((tl.load(mask + i, mask=ok, other=0) != 0) & (tl.load(mask + j, mask=ok, other=0) != 0)).to(tl.float32)[:, None]
    th = _tanh(0.5 * gp)
    hA = 0.5 * dA
    dpv = (hA + hA * th).to(tl.bfloat16)
    dgv = (hA * pp * 0.5 * (1.0 - th * th)).to(tl.bfloat16)
    dwg += tl.dot(tl.trans(dgv), xt)
    dwp += tl.dot(tl.trans(dpv), xt)
    return dgv, dpv, dwg, dwp


@triton.jit
def _wait_geq(ptr, want):
    while tl.atomic_add(ptr, 0, sem="acquire", scope="gpu") < want:
        pass


@triton.jit
def _b7j(xn, wg, wp, dab, mask, wdx, dg, z, dy, st, gin, ring, prod, cons, dz, dwpart, lnpart,
         T, L, num_tiles, S, C, G, RINGS,
         D: tl.constexpr, SW: tl.constexpr, NP: tl.constexpr, HAS_MASK: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
    """Joint input side: G groups x (S sources + C consumers).  A source keeps one SW-channel block's dW in registers and publishes each
    [BM, 2 SW] derivative tile into its group's ring slot (release flag); a consumer acquires the S flags, runs the dx_n GEMM + LN_in
    backward + residual, and releases the slot."""
    K4: tl.constexpr = 2 * NP
    pid = tl.program_id(0)
    per = S + C
    g = pid // per
    r = pid % per
    rt = tl.arange(0, BM)
    rc = tl.arange(0, D)
    rs = tl.arange(0, SW)
    if r < S:
        b = r
        c0 = SW * b
        wgb = tl.load(wg + (c0 + rs)[:, None] * D + rc[None, :])
        wpb = tl.load(wp + (c0 + rs)[:, None] * D + rc[None, :])
        dwg = tl.zeros((SW, D), dtype=tl.float32)
        dwp = tl.zeros((SW, D), dtype=tl.float32)
        n_iter = tl.cdiv(num_tiles - g, G)
        for it in range(0, n_iter):
            t0 = (g + it * G).to(tl.int64) * BM
            slot = it % RINGS
            dgv, dpv, dwg, dwp = _src_tile(xn, dab, mask, wgb, wpb, dwg, dwp, t0, T, L, D, SW, c0, HAS_MASK, BM)
            if it >= RINGS:
                _wait_geq(cons + g * RINGS + slot, it // RINGS)
            R = ring + (g * RINGS + slot).to(tl.int64) * BM * K4 + rt[:, None] * K4 + 2 * SW * b
            tl.store(R + rs[None, :], dgv)
            tl.store(R + SW + rs[None, :], dpv)
            tl.debug_barrier()
            tl.atomic_xchg(prod + (g * RINGS + slot) * S + b, it + 1, sem="release", scope="gpu")
        base = dwpart + g.to(tl.int64) * 2 * NP * D
        tl.store(base + (c0 + rs)[:, None] * D + rc[None, :], dwg)
        tl.store(base + NP * D + (c0 + rs)[:, None] * D + rc[None, :], dwp)
    else:
        c = r - S
        gv = tl.load(gin + rc)
        dgam = tl.zeros((D,), dtype=tl.float32)
        dbet = tl.zeros((D,), dtype=tl.float32)
        rk = tl.arange(0, BK)
        n_seq = tl.cdiv(num_tiles - g, G)
        for k in range(c, n_seq, C):
            t0 = (g + k * G).to(tl.int64) * BM
            slot = k % RINGS
            tok = t0 + rt
            for sidx in range(0, S):
                _wait_geq(prod + (g * RINGS + slot) * S + sidx, k + 1)
            R = ring + (g * RINGS + slot).to(tl.int64) * BM * K4
            acc = tl.zeros((BM, D), dtype=tl.float32)
            for k0 in range(0, K4, BK):
                a = tl.load(R + rt[:, None] * K4 + (k0 + rk)[None, :], cache_modifier=".cg")
                acc = tl.dot(a, tl.load(wdx + (k0 + rk)[:, None] * D + rc[None, :]), acc)
            tl.debug_barrier()
            tl.atomic_xchg(cons + g * RINGS + slot, k // RINGS + 1, sem="release", scope="gpu")
            dgt = tl.load(dg + tok[:, None] * D + rc[None, :], mask=(tok < T)[:, None], other=0.0)
            acc = tl.dot(dgt, tl.load(wdx + (K4 + rc)[:, None] * D + rc[None, :]), acc)
            pg, pb = _con_epilogue(acc, z, dy, st, gv, dz, t0, T, D, BM)
            dgam += pg
            dbet += pb
        tl.store(lnpart + (g * C + c) * 2 * D + rc, dgam)
        tl.store(lnpart + (g * C + c) * 2 * D + D + rc, dbet)


# ============================================================================================================ host side
def _props(dev):
    return torch.cuda.get_device_properties(dev)


@torch.no_grad()
def pack(m):
    """The module's weights in the kernels' layouts (cached on the module by parameter version)."""
    key = tuple((t.data_ptr(), t._version) for t in m.parameters())
    if getattr(m, "_tt_key", None) == key:
        return m._tt_pk
    bf = lambda t: t.detach().to(torch.bfloat16).contiguous()  # noqa: E731
    f = lambda t: t.detach().float().contiguous()  # noqa: E731
    wg = torch.cat([bf(m.to_left_gate.weight), bf(m.to_right_gate.weight)], 0).contiguous()   # [NP, D]
    wp = torch.cat([bf(m.to_left.weight), bf(m.to_right.weight)], 0).contiguous()
    NP, D = wg.shape
    assert D % 16 == 0 and (D & (D - 1)) == 0, "D a power of two (tl.arange)"
    pk = dict(D=D, ch=NP // 2, NP=NP, bidir=m.__class__.__name__.startswith("Bidirectional"), outgoing=getattr(m, "outgoing", True),
              wg=wg, wp=wp, wgt=wg.t().contiguous(), wpt=wp.t().contiguous(), wdx={},
              w1i=torch.stack([wg, wp], 1).reshape(2 * NP, D).t().contiguous(),                    # [D, 2 NP], (g, p) interleaved
              wo=bf(m.to_out.weight), wot=bf(m.to_out.weight).t().contiguous(), wog=bf(m.to_gate.weight),
              wogt=bf(m.to_gate.weight).t().contiguous(), gin=f(m.ln_pair.weight), bin=f(m.ln_pair.bias), go=f(m.ln_out.weight),
              bo=f(m.ln_out.bias), eps_in=float(m.ln_pair.eps), eps_out=float(m.ln_out.eps))
    m._tt_pk, m._tt_key = pk, key
    return pk


def _wdx(pk, sw):
    """The consumer's weight for source width sw: per source block [its sw gate rows ; its sw projection rows], then W_og  ([2 NP + D, D])."""
    if sw not in pk["wdx"]:
        wg, wp, NP = pk["wg"], pk["wp"], pk["NP"]
        assert NP % sw == 0, "plane channels in whole source blocks"
        blocks = [torch.cat([wg[sw * b:sw * (b + 1)], wp[sw * b:sw * (b + 1)]], 0) for b in range(NP // sw)]
        pk["wdx"][sw] = torch.cat(blocks + [pk["wog"]], 0).contiguous()
    return pk["wdx"][sw]


def _halves(pk):
    """Channels [0, h) outgoing, [h, CH) incoming."""
    ch = pk["ch"]
    return ch // 2 if pk["bidir"] else (ch if pk["outgoing"] else 0)


def _contract(ab, x, pk, L):
    ch, h = pk["ch"], _halves(pk)
    A, B = ab[:ch].view(ch, L, L), ab[ch:].view(ch, L, L)
    if h:
        torch.bmm(A[:h], B[:h].transpose(1, 2), out=x[:h])
    if h < ch:
        torch.bmm(A[h:].transpose(1, 2), B[h:], out=x[h:])


def _forward(pk, z, mask, ds, train):
    B, L, _, D = z.shape
    assert B == 1 and D == pk["D"]
    T, ch, NP, dev = L * L, pk["ch"], pk["NP"], z.device
    zf = z.reshape(T, D).contiguous()
    mk = zf
    if mask is not None:                         # a bool mask is read as uint8 in place (no cast kernel); the kernels test != 0
        mk = mask.reshape(L)
        mk = mk.view(torch.uint8) if mk.dtype == torch.bool and mk.is_contiguous() else (mk != 0).to(torch.uint8)
    ab = torch.empty(NP, T, device=dev, dtype=torch.bfloat16)
    zst = torch.empty(T, 2, device=dev, dtype=torch.float32)
    _k1[lambda a: (triton.cdiv(T, a["BM"]),)](zf, mk, pk["w1i"], pk["gin"], pk["bin"], ab, zst, T, L, pk["eps_in"], D=D, NP=NP,
                                              HAS_MASK=mask is not None)
    x = torch.empty(ch, L, L, device=dev, dtype=torch.bfloat16)
    _contract(ab, x, pk, L)
    out = torch.empty_like(zf)
    st = torch.empty(T, 4, device=dev, dtype=torch.float32) if train else zst
    xn = torch.empty_like(zf) if train else zf
    dsv = ds.reshape(L, D).contiguous() if ds is not None else zf
    _k3[lambda a: (triton.cdiv(T, a["BM"]),)](x.view(ch, T), zf, zst, pk["wot"], pk["go"], pk["bo"], pk["wogt"], pk["gin"], pk["bin"], dsv,
                                              out, st, xn, T, L, pk["eps_out"], D=D, CH=ch, HAS_DS=ds is not None, TRAIN=train)
    return out.view(B, L, L, D), (zf, ab, x, dsv if ds is not None else None, mk if mask is not None else None, st, xn)


def forward(m, z, mask=None):
    """Inference: y = z + TriMul(z)."""
    return _forward(pack(m), z, mask, None, False)[0]


def _mm32(a, b):
    return torch.mm(a, b, out_dtype=torch.float32)


def _gemm_tk(a, b):
    """a^T b over K = T in fp32, K split into chunks (cuBLAS picks a non-split kernel for these shapes)."""
    T = a.shape[0]
    S = next((s for s in GEMM_TK["chunks"] if T % s == 0 and T // s >= GEMM_TK["min_chunk"]), 1)
    if S == 1:
        return _mm32(a.t(), b)
    return torch.bmm(a.unflatten(0, (S, T // S)).transpose(1, 2), b.unflatten(0, (S, T // S)), out_dtype=torch.float32).sum(0)


_SIDE = {}
_JOINT_CAP = {}


def _side(dev, k):
    return _SIDE.setdefault((dev, k), torch.cuda.Stream(device=dev))


def _joint_capacity(dev, args, meta):
    """Co-resident programs of the joint kernel (its sources and consumers wait on each other: the grid must fit at once)."""
    key = (dev, tuple(sorted(meta.items())))
    if key not in _JOINT_CAP:
        k = _b7j.warmup(*args, grid=(1,), **meta)
        k._init_handles()
        p = _props(dev)
        threads = 32 * meta["num_warps"]
        by_regs = p.regs_per_multiprocessor // max(1, k.n_regs * threads)
        by_smem = p.shared_memory_per_multiprocessor // (k.metadata.shared + RESERVED_SMEM_PER_BLOCK)
        by_thr = p.max_threads_per_multi_processor // threads
        _JOINT_CAP[key] = max(0, min(by_regs, by_smem, by_thr)) * p.multi_processor_count
    return _JOINT_CAP[key]


class TriMulTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, mask, ds, m, *params):
        pk = pack(m)
        y, saved = _forward(pk, z, mask, ds, True)
        zf, ab, x, dsv, mk, st, xn = saved
        ctx.save_for_backward(zf, ab, x, st, xn, *(t for t in (dsv, mk) if t is not None))
        ctx.has_ds, ctx.has_mask, ctx.pk, ctx.m, ctx.L = dsv is not None, mk is not None, pk, m, z.shape[1]
        return y

    @staticmethod
    def backward(ctx, dy):
        sv = list(ctx.saved_tensors)
        zf, ab, x, st, xn = sv[:5]
        rest = sv[5:]
        dsv = rest.pop(0) if ctx.has_ds else None
        mk = rest.pop(0) if ctx.has_mask else None
        pk, m, L = ctx.pk, ctx.m, ctx.L
        ch, NP, D, T, dev = pk["ch"], pk["NP"], pk["D"], L * L, zf.device
        dyf = dy.reshape(T, D).contiguous().to(torch.bfloat16)
        mkp = mk if mk is not None else zf
        nsm = _props(dev).multi_processor_count
        main = torch.cuda.current_stream()
        # ---- B1 (split path: d_g goes straight into the input-side operand buffer [dg | dp (planes) | d_g])
        dX = torch.empty(ch, L, L, device=dev, dtype=torch.bfloat16)
        ao = torch.empty(T, D, device=dev, dtype=torch.bfloat16)
        d_o = torch.empty(T, D, device=dev, dtype=torch.bfloat16)
        Kd = 2 * NP + D
        if _JOINT:
            dg = torch.empty(T, D, device=dev, dtype=torch.bfloat16)
        else:
            dgp = torch.empty(T, Kd, device=dev, dtype=torch.bfloat16)
            dg = dgp[:, 2 * NP:]
        vs = torch.zeros(2, D, device=dev, dtype=torch.float32)
        _b1e[lambda a: (triton.cdiv(T, a["BM"]),)](x.view(ch, T), xn, st, dyf, dsv if dsv is not None else dyf, pk["wot"], pk["go"], pk["bo"],
                                                   pk["wogt"], d_o, ao, dg, dg.stride(0), vs, T, L, D=D, CH=ch, HAS_DS=dsv is not None)
        _b1d[lambda a: (triton.cdiv(T, a["BM"]),)](d_o, pk["wo"], pk["go"], x.view(ch, T), st, dX.view(ch, T), T, D=D, CH=ch)

        def weight_grads():
            v_o, S_o = vs.unbind(0)
            H = _gemm_tk(ao, x.view(ch, T).t()) - v_o[:, None]
            g_out, b_out = m.ln_out.weight.detach().float(), m.ln_out.bias.detach().float()
            Wo = m.to_out.weight.detach().float()
            return (H * g_out[None, :] + S_o[:, None] * b_out[None, :], (Wo * H).sum(0), (Wo * S_o[:, None]).sum(0), _mm32(dg.t(), xn))

        # the weight-gradient GEMMs are DRAM-bound, the contraction backward is tensor-bound: side by side
        if _OVERLAP:
            side = _side(dev, 1)
            side.wait_stream(main)
            for t in (vs, ao, x, dg, xn):
                t.record_stream(side)
            with torch.cuda.stream(side):
                d_wo, d_gout, d_bout, d_wog = weight_grads()
        else:
            d_wo, d_gout, d_bout, d_wog = weight_grads()
        # ---- contraction backward
        dab = torch.empty(NP, L, L, device=dev, dtype=torch.bfloat16)
        A, Bp, dA, dB = ab[:ch].view(ch, L, L), ab[ch:].view(ch, L, L), dab[:ch], dab[ch:]
        h = _halves(pk)
        if h:
            torch.bmm(dX[:h], Bp[:h], out=dA[:h])
            torch.bmm(dX[:h].transpose(1, 2), A[:h], out=dB[:h])
        if h < ch:
            torch.bmm(Bp[h:], dX[h:].transpose(1, 2), out=dA[h:])
            torch.bmm(A[h:], dX[h:], out=dB[h:])
        # ---- input side
        dz = torch.empty_like(zf)
        if _JOINT:
            cj = dict(JOINT[ch])
            Cc, rings, BM = cj.pop("C"), cj.pop("rings"), cj["BM"]
            S = NP // cj["SW"]
            num_tiles = triton.cdiv(T, BM)
            wdx = _wdx(pk, cj["SW"])
            meta = dict(D=D, NP=NP, HAS_MASK=mk is not None, **cj)
            probe = (xn, pk["wg"], pk["wp"], dab, mkp, wdx, dg, zf, dyf, st, pk["gin"], dab, vs, vs, dz, vs, vs, T, L, num_tiles, S, Cc, 1, rings)
            Gn = _joint_capacity(dev, probe, meta) // (S + Cc)
            assert Gn >= 1, "one group of the joint kernel does not fit on the device at once"
            ring = torch.empty(Gn * rings * BM * 2 * NP, device=dev, dtype=torch.bfloat16)
            flags = torch.zeros(Gn * rings * (S + 1), device=dev, dtype=torch.int32)
            dwp = torch.empty(Gn, 2, NP, D, device=dev, dtype=torch.float32)
            part = torch.empty(Gn * Cc, 2, D, device=dev, dtype=torch.float32)
            _b7j[(Gn * (S + Cc),)](xn, pk["wg"], pk["wp"], dab.view(NP, T), mkp, wdx, dg, zf, dyf, st, pk["gin"], ring,
                                   flags[:Gn * rings * S], flags[Gn * rings * S:], dz, dwp, part, T, L, num_tiles, S, Cc, Gn, rings, **meta)
            dw = dwp.sum(0)                                                  # [2 (gate | proj), NP, D]
            d_gin, d_bin = part.sum(0).unbind(0)
        else:
            _src[lambda a: (triton.cdiv(T, a["BM"]), NP // a["SW"])](xn, pk["wgt"], pk["wpt"], dab.view(NP, T), mkp, dgp, T, L, D=D, NP=NP,
                                                                      HAS_MASK=mk is not None)
            sw = _src.best_config.kwargs["SW"]
            # dW = dgp^T x_n (split-K, DRAM-bound) on a second side stream, under the tensor-bound consumer
            side2 = _side(dev, 2)
            side2.wait_stream(main)
            dgp.record_stream(side2)
            xn.record_stream(side2)
            with torch.cuda.stream(side2):
                dwb = _gemm_tk(dgp[:, :2 * NP], xn)
            ln = torch.zeros(2, D, device=dev, dtype=torch.float32)
            _con[lambda a: (min(triton.cdiv(T, a["BM"]), a["WAVES"] * nsm),)](dgp, _wdx(pk, sw), zf, dyf, st, pk["gin"], dz, ln, T, nsm,
                                                                              D=D, K=Kd)
            main.wait_stream(side2)
            dwb.record_stream(main)
            dw = dwb.view(NP // sw, 2, sw, D).transpose(0, 1).reshape(2, NP, D)   # rows [blocks][gate sw | proj sw] -> [2, NP, D]
            d_gin, d_bin = ln.unbind(0)
        if _OVERLAP:
            main.wait_stream(side)
            for t in (d_wo, d_gout, d_bout, d_wog):
                t.record_stream(main)
        out = {"ln_pair.weight": d_gin, "ln_pair.bias": d_bin, "to_left_gate.weight": dw[0, :ch], "to_right_gate.weight": dw[0, ch:],
               "to_left.weight": dw[1, :ch], "to_right.weight": dw[1, ch:], "ln_out.weight": d_gout, "ln_out.bias": d_bout,
               "to_gate.weight": d_wog, "to_out.weight": d_wo}
        grads = [out[n].to(dict(m.named_parameters())[n].dtype) for n in PARAMS]
        return (dz.view(1, L, L, D), None, None, None, *grads)


def forward_train(m, z, mask, ds):
    params = [dict(m.named_parameters())[n] for n in PARAMS]
    return TriMulTriton.apply(z, mask, ds, m, *params)
