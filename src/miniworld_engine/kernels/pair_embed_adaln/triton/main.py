"""La-Proteina's initial pair builder (``pair_embed_adaln``) as portable kernels: the contract of the H100 CUDA kernels on any card.

    pair[b,i,j] = pm_ij * ( LN(h) * gamma[b] + beta[b] ),   h = LN_affine( T[sep] + T[nS + dist_xt] + T[nS + nX + dist_sc] )

The one-hots of the sequence separation and of the two CA-distance bins times the Linear are three rows of the fp32 table ``T = W^T`` [NT, 256];
the bins are computed in registers, both LayerNorms and the per-sample adaptive scale / shift run on the 256-channel row in registers, the output is bf16.

* forward ``_adaln_fwd``: one program per (batch, row i, block of columns j); it also writes the packed bin word of every pair (``bins`` int32:
  sequence-separation bin | xt bin << 8 | sc bin << 16 | has-sc flag | bit 31 = padding), which the backward replays.
* backward, no atomics (bitwise reproducible for a given launch configuration): ``_adaln_bwd`` recomputes the pair, writes d(table sum) as bf16 (the batch is
  processed in chunks so that scratch stays small) and one fp32 partial of ``[dgamma | dbeta | dln_w | dln_b]`` per (sample, block of RB rows);
  ``_adaln_dw`` forms the table gradient as ``onehot^T @ dx`` on the tensor cores (bf16 one-hot, fp32 accumulation, a program owns one channel slice of a FIXED
  contiguous slice of the pairs, tiles in order); ``_adaln_bwd_reduce`` adds the partials in fixed order.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import token_key

C = 256
NBLK = 192                   # pair slices of the table-gradient kernel (fixed)
RB = 8                       # rows of pairs per program of the backward row kernel (one partial per program)
DX_CHUNK_BYTES = 1 << 29     # scratch for d(table sum): at most this many bytes of pairs at a time
_HAS_SC = 1 << 24


@triton.jit
def _dist3(X, b, n, i, j, ok):
    pi = (b * n + i) * 3
    pj = (b * n + j) * 3
    dx = tl.load(X + pi) - tl.load(X + pj, mask=ok, other=0.0)
    dy = tl.load(X + pi + 1) - tl.load(X + pj + 1, mask=ok, other=0.0)
    dz = tl.load(X + pi + 2) - tl.load(X + pj + 2, mask=ok, other=0.0)
    return tl.sqrt_rn(dx * dx + dy * dy + dz * dz)


@triton.jit
def _bucket(v, lim, N: tl.constexpr):
    c = tl.zeros(v.shape, dtype=tl.int32)
    for k in tl.static_range(N):
        c += (tl.load(lim + k) < v).to(tl.int32)
    return c


@triton.jit
def _row(TBL, r, m, c):
    return tl.load(TBL + r[:, None] * 256 + c[None, :], mask=m[:, None], other=0.0)


@triton.jit
def _forward_tile(TBL, w, live, lw, lb, c, nS: tl.constexpr, nX: tl.constexpr, e1, e2):
    """The two LayerNorms of a [BLOCK, 256] tile of pairs from their bin words: (yhat1, rstd1, yhat2, rstd2)."""
    x = _row(TBL, w & 255, live, c)
    x += _row(TBL, nS + ((w >> 8) & 255), live, c)
    hs = live & ((w & (1 << 24)) != 0)
    x += _row(TBL, nS + nX + ((w >> 16) & 255), hs, c)
    mean = tl.sum(x, axis=1) / 256
    xc = x - mean[:, None]
    r1 = tl.math.rsqrt(tl.sum(xc * xc, axis=1) / 256 + e1)
    yh1 = xc * r1[:, None]
    y = yh1 * lw[None, :] + lb[None, :]
    mean2 = tl.sum(y, axis=1) / 256
    yc = y - mean2[:, None]
    r2 = tl.math.rsqrt(tl.sum(yc * yc, axis=1) / 256 + e2)
    return yh1, r1, yc * r2[:, None], r2


@triton.autotune(configs=configs_for("pair_embed_adaln_fwd_triton"), key=["shape_key", "HAS_SC"])
@triton.jit
def _adaln_fwd(TBL, LNW, LNB, GAMMA, BETA, XT, SC, IDX, MASK, XLIM, CLIM, OUT, BINS, n, e1, e2,
               HAS_SC: tl.constexpr, nS: tl.constexpr, nX: tl.constexpr, NXL: tl.constexpr, NCL: tl.constexpr, BLOCK_J: tl.constexpr, shape_key):
    jb = tl.program_id(0)
    i = tl.program_id(1).to(tl.int64)
    b = tl.program_id(2).to(tl.int64)
    j = (jb * BLOCK_J + tl.arange(0, BLOCK_J)).to(tl.int64)
    jin = j < n
    ri = b * n + i
    rj = b * n + j
    live = jin & (tl.load(MASK + ri) != 0) & (tl.load(MASK + rj, mask=jin, other=0) != 0)
    s = tl.minimum(tl.maximum(tl.load(IDX + ri) - tl.load(IDX + rj, mask=jin, other=0) + (nS - 1) // 2, 0), nS - 1)
    w = s | (_bucket(_dist3(XT, b, n, i, j, live), XLIM, NXL) << 8)
    if HAS_SC:
        w = w | (1 << 24) | (_bucket(_dist3(SC, b, n, i, j, live), CLIM, NCL) << 16)
    w = tl.where(live, w, -2147483648)
    tl.store(BINS + ri * n + j, w, mask=jin)
    c = tl.arange(0, 256)
    lw = tl.load(LNW + c)
    lb = tl.load(LNB + c)
    _, _, yh2, _ = _forward_tile(TBL, w, live, lw, lb, c, nS, nX, e1, e2)
    o = yh2 * tl.load(GAMMA + b * 256 + c)[None, :] + tl.load(BETA + b * 256 + c)[None, :]
    tl.store(OUT + (ri * n + j)[:, None] * 256 + c[None, :], tl.where(live[:, None], o, 0.0).to(tl.bfloat16), mask=jin[:, None])


@triton.autotune(configs=configs_for("pair_embed_adaln_bwd_triton"), key=["shape_key"])
@triton.jit
def _adaln_bwd(TBL, LNW, LNB, GAMMA, BINS, DOUT, DX, PART, b0, n, nrb, e1, e2,
               nS: tl.constexpr, nX: tl.constexpr, RB: tl.constexpr, BLOCK_J: tl.constexpr, shape_key):
    """d(table sum) per pair (bf16) and the partial [dgamma | dbeta | dln_w | dln_b] of the RB rows (rb*RB ..) of sample b0 + program_id(1)."""
    rb = tl.program_id(0)
    bl = tl.program_id(1).to(tl.int64)
    b = b0 + bl
    c = tl.arange(0, 256)
    lw = tl.load(LNW + c)
    lb = tl.load(LNB + c)
    gm = tl.load(GAMMA + b * 256 + c)
    dg = tl.zeros([256], dtype=tl.float32)
    dbt = tl.zeros([256], dtype=tl.float32)
    dw = tl.zeros([256], dtype=tl.float32)
    dl = tl.zeros([256], dtype=tl.float32)
    for r in range(RB):
        i = (rb * RB + r).to(tl.int64)
        for j0 in range(0, n, BLOCK_J):
            j = (j0 + tl.arange(0, BLOCK_J)).to(tl.int64)
            jin = (j < n) & (i < n)
            pr = (b * n + i) * n + j
            w = tl.load(BINS + pr, mask=jin, other=-2147483648)
            live = jin & (w >= 0)
            yh1, r1, yh2, r2 = _forward_tile(TBL, w, live, lw, lb, c, nS, nX, e1, e2)
            g = tl.load(DOUT + pr[:, None] * 256 + c[None, :], mask=live[:, None], other=0.0).to(tl.float32)
            dbt += tl.sum(g, axis=0)
            dg += tl.sum(g * yh2, axis=0)
            d2 = g * gm[None, :]
            p2 = tl.sum(d2, axis=1) / 256
            a2 = tl.sum(d2 * yh2, axis=1) / 256
            dy = r2[:, None] * (d2 - p2[:, None] - yh2 * a2[:, None])
            dl += tl.sum(dy, axis=0)
            dw += tl.sum(dy * yh1, axis=0)
            d1 = dy * lw[None, :]
            p1 = tl.sum(d1, axis=1) / 256
            a1 = tl.sum(d1 * yh1, axis=1) / 256
            dxv = r1[:, None] * (d1 - p1[:, None] - yh1 * a1[:, None])
            tl.store(DX + ((bl * n + i) * n + j)[:, None] * 256 + c[None, :], tl.where(live[:, None], dxv, 0.0).to(tl.bfloat16), mask=jin[:, None])
    base = PART + (b * nrb + rb) * 1024
    tl.store(base + c, dg)
    tl.store(base + 256 + c, dbt)
    tl.store(base + 512 + c, dw)
    tl.store(base + 768 + c, dl)


@triton.autotune(configs=configs_for("pair_embed_adaln_dw_triton"), key=["shape_key"])
@triton.jit
def _adaln_dw(DX, BINS, PART, P, per, nS: tl.constexpr, nX: tl.constexpr, nC: tl.constexpr, RS: tl.constexpr, RX: tl.constexpr, RC: tl.constexpr,
              BLOCK_P: tl.constexpr, shape_key):
    """PART[t, row, channels of slice q] = sum over the pairs of slice t of onehot(row) * dx[pair, channels]; pair tiles in order."""
    q = tl.program_id(0)
    t = tl.program_id(1).to(tl.int64)
    p0 = t * per
    p1 = tl.minimum(P, p0 + per)
    rs = tl.arange(0, RS)
    rx = tl.arange(0, RX)
    rc = tl.arange(0, RC)
    cc = tl.arange(0, 32)
    acc_s = tl.zeros([RS, 32], dtype=tl.float32)
    acc_x = tl.zeros([RX, 32], dtype=tl.float32)
    acc_c = tl.zeros([RC, 32], dtype=tl.float32)
    for s0 in range(p0, p1, BLOCK_P):
        p = s0 + tl.arange(0, BLOCK_P)
        pm = p < p1
        w = tl.load(BINS + p, mask=pm, other=-2147483648)
        live = pm & (w >= 0)
        g = tl.load(DX + p[:, None] * 256 + q * 32 + cc[None, :], mask=live[:, None], other=0.0)
        acc_s = tl.dot(((rs[:, None] == (w & 255)[None, :]) & live[None, :]).to(tl.bfloat16), g, acc_s)
        acc_x = tl.dot(((rx[:, None] == ((w >> 8) & 255)[None, :]) & live[None, :]).to(tl.bfloat16), g, acc_x)
        hs = live & ((w & (1 << 24)) != 0)
        acc_c = tl.dot(((rc[:, None] == ((w >> 16) & 255)[None, :]) & hs[None, :]).to(tl.bfloat16), g, acc_c)
    base = PART + t * (nS + nX + nC) * 256 + q * 32
    tl.store(base + rs[:, None] * 256 + cc[None, :], acc_s, mask=(rs < nS)[:, None])
    tl.store(base + (nS + rx)[:, None] * 256 + cc[None, :], acc_x, mask=(rx < nX)[:, None])
    tl.store(base + (nS + nX + rc)[:, None] * 256 + cc[None, :], acc_c, mask=(rc < nC)[:, None])


@triton.autotune(configs=configs_for("pair_embed_adaln_bwd_reduce_triton"), key=["shape_key"])
@triton.jit
def _adaln_bwd_reduce(PART, OUT, total, count, stride, off, BLOCK: tl.constexpr, shape_key):
    """OUT[g, k] = sum over r < count of PART[(g * count + r) * stride + off + k], in the order r = 0, 1, ..."""
    g = tl.program_id(1).to(tl.int64)
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = k < total
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for r in range(count):
        acc += tl.load(PART + (g * count + r) * stride + off + k, mask=m, other=0.0)
    tl.store(OUT + g * total + k, acc, mask=m)


def _reduce(part, groups, count, total, stride, off, key):
    out = torch.empty(groups, total, device=part.device, dtype=torch.float32)
    _adaln_bwd_reduce[lambda meta: (triton.cdiv(total, meta["BLOCK"]), groups)](part, out, total, count, stride, off, shape_key=key)
    return out


def portable_forward(tbl, lnw, lnb, gamma, beta, xt, sc, has_sc, idx, mask, xlim, clim, nS, nX, eps1, eps2):
    """(pair [B, n, n, 256] bf16, bins [B, n, n] int32), as ``cuda.sm90.forward``."""
    B, n = idx.shape
    out = torch.empty(B, n, n, C, device=tbl.device, dtype=torch.bfloat16)
    bins = torch.empty(B, n, n, device=tbl.device, dtype=torch.int32)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_J"]), n, B)
    _adaln_fwd[grid](tbl, lnw, lnb, gamma, beta, xt, sc, idx, mask, xlim, clim, out, bins, n, float(eps1), float(eps2),
                     HAS_SC=bool(has_sc), nS=int(nS), nX=int(nX), NXL=int(xlim.numel()), NCL=int(clim.numel()), shape_key=token_key(n))
    return out, bins


def portable_backward(tbl, lnw, lnb, gamma, bins, dout, nS, nX, eps1, eps2):
    """(dtbl [NT, 256], dln_w, dln_b [256], dgamma, dbeta [B, 256]) fp32 from d pair = ``dout`` [B, n, n, 256] bf16, as ``cuda.sm90.backward``."""
    B, n = bins.shape[:2]
    NT = tbl.shape[0]
    nS, nX = int(nS), int(nX)
    nC = NT - nS - nX
    dev = tbl.device
    key = token_key(n)
    chunk_b = max(1, min(B, DX_CHUNK_BYTES // (n * n * C * 2)))
    nchunk = -(-B // chunk_b)
    nrb = -(-n // RB)
    dx = torch.empty(chunk_b, n, n, C, device=dev, dtype=torch.bfloat16)
    part = torch.empty(B * nrb * 1024, device=dev, dtype=torch.float32)
    tpart = torch.empty(nchunk * NBLK, NT, C, device=dev, dtype=torch.float32)
    pow2 = triton.next_power_of_2
    for ch in range(nchunk):
        b0 = ch * chunk_b
        Bc = min(chunk_b, B - b0)
        _adaln_bwd[(nrb, Bc)](tbl, lnw, lnb, gamma, bins, dout, dx, part, b0, n, nrb, float(eps1), float(eps2),
                              nS=nS, nX=nX, RB=RB, shape_key=key)
        P = Bc * n * n
        _adaln_dw[(C // 32, NBLK)](dx, bins[b0:b0 + Bc], tpart[ch * NBLK:(ch + 1) * NBLK], P, triton.cdiv(P, NBLK), nS=nS, nX=nX, nC=nC,
                                   RS=max(16, pow2(nS)), RX=max(16, pow2(nX)), RC=max(16, pow2(nC)), shape_key=key)
    dtbl = _reduce(tpart, 1, nchunk * NBLK, NT * C, NT * C, 0, key).view(NT, C)
    dgb = _reduce(part, B, nrb, 512, 1024, 0, key)
    dl = _reduce(part, 1, B * nrb, 512, 1024, 512, key)
    return dtbl, dl[0, :C].contiguous(), dl[0, C:].contiguous(), dgb[:, :C].contiguous(), dgb[:, C:].contiguous()
