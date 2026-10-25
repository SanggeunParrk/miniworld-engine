"""Backbone pair embedding (``pair_embed_geom`` / ``pair_embed_ca``) as portable kernels: the contract of the H100 CUDA kernels on any card.

    pair[b,i,j] = M_ij * ( T[sep] + (ca_i ca_j) (T[d_CA-N] + T[d_CA-CA] + T[d_CA-C] + T[d_CA-CB])
                           + (ca_i ca_j cb_i) (T[theta_12] + T[theta_21] + T[phi_12] + T[phi_21] + T[omega]) + chain_i chain_j T[chain] )

A one-hot times a Linear is a row gather from the table ``T`` [317 (decoder: 157), 256] (bf16, the autocast weight): the bins of a pair are computed in
registers (distances, dihedrals and bond angles as torch computes them), up to ten table rows are summed in fp32 and the sum is rounded to bf16
(``ln_eps``: first normalised over the 256 channels, no affine, on the bf16-rounded sum -- what torch's ``layer_norm`` of the Linear output sees).

* forward, two kernels. ``_geom_bins`` / ``_ca_bins`` compute the bins of every pair (one pair per thread) and write the 16-byte code of the pair (word 0:
  sequence-separation bin | four distance bins | flags, bit 31 = live pair; word 1: the five angle bins; word 2: the chain product) -- the same words as the CUDA
  kernel, which the backward replays. ``_geom_fwd_gather`` reads the codes and sums the table rows (``MODE`` 0: encoder table, 1: decoder table), optionally
  followed by the LayerNorm; it writes the only large tensor. (One kernel doing both is 5-6x slower: the geometry of a [BLOCK_J] vector is replicated over the
  threads that hold the [BLOCK_J, 256] tile.)
* backward: the table gradient is ``onehot^T @ d pair``: ``_geom_dw`` builds the bf16 one-hot of each feature group from the codes (nothing is materialised) and
  multiplies it with a [pairs x CH channels] tile of the gradient on the tensor cores, accumulating in fp32; a program owns one channel slice of a FIXED
  contiguous slice of the pairs and the pair tiles are visited in order, so the partial sums do not depend on the launch configuration; ``_geom_dw_reduce``
  adds the NBLK slice partials in slice order and rounds to bf16 once. No atomics: bitwise reproducible. ``_geom_ln_bwd`` is the LayerNorm backward with the
  pair recomputed from the codes (``ln_eps``).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import token_key

C = 256
NT = 317                      # 127 sep + 4 x 21 dist + 5 x 21 angle + 1 chain
NT_CA = 157                   # 127 sep + 30 CA distance
NBLK = 192                    # slices of the pairs in the table-gradient kernel (fixed: the summation order does not depend on the card)
_F_DIST = 1 << 28
_F_ORI = 1 << 29


# ---------------------------------------------------------------------------------------------------------------------------------------
# geometry, written as torch computes it (fp32, IEEE division and square root)
# ---------------------------------------------------------------------------------------------------------------------------------------
@triton.jit
def _unit(x, y, z):
    n = tl.maximum(tl.sqrt_rn(x * x + y * y + z * z), 1e-8)
    return tl.math.div_rn(x, n), tl.math.div_rn(y, n), tl.math.div_rn(z, n)


@triton.jit
def _cross(ax, ay, az, bx, by, bz):
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


@triton.jit
def _dihedral(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz):
    b0x, b0y, b0z = bx - ax, by - ay, bz - az
    b1x, b1y, b1z = cx - bx, cy - by, cz - bz
    b2x, b2y, b2z = dx - cx, dy - cy, dz - cz
    c1x, c1y, c1z = _cross(b0x, b0y, b0z, b1x, b1y, b1z)
    n1x, n1y, n1z = _unit(c1x, c1y, c1z)
    c2x, c2y, c2z = _cross(b1x, b1y, b1z, b2x, b2y, b2z)
    n2x, n2y, n2z = _unit(c2x, c2y, c2z)
    cosv = n1x * n2x + n1y * n2y + n1z * n2z
    xx, xy, xz = _cross(n1x, n1y, n1z, n2x, n2y, n2z)
    s = xx * b1x + xy * b1y + xz * b1z
    sg = tl.where(s > 0.0, 1.0, tl.where(s < 0.0, -1.0, 0.0))
    return libdevice.atan2(sg * tl.sqrt_rn(xx * xx + xy * xy + xz * xz), cosv)


@triton.jit
def _bond(ax, ay, az, bx, by, bz, cx, cy, cz):
    b0x, b0y, b0z = _unit(bx - ax, by - ay, bz - az)
    b1x, b1y, b1z = _unit(cx - ax, cy - ay, cz - az)
    xx, xy, xz = _cross(b0x, b0y, b0z, b1x, b1y, b1z)
    return libdevice.atan2(tl.sqrt_rn(xx * xx + xy * xy + xz * xz), b0x * b1x + b0y * b1y + b0z * b1z)


@triton.jit
def _bucket(v, lim, N: tl.constexpr):
    """torch.bucketize: the number of limits strictly below ``v``."""
    c = tl.zeros(v.shape, dtype=tl.int32)
    for k in tl.static_range(N):
        c += (tl.load(lim + k) < v).to(tl.int32)
    return c


@triton.jit
def _atoms(BB, b, n, j, a: tl.constexpr, ok):
    base = ((b * n + j) * 4 + a) * 3
    return (tl.load(BB + base, mask=ok, other=0.0), tl.load(BB + base + 1, mask=ok, other=0.0), tl.load(BB + base + 2, mask=ok, other=0.0))


@triton.jit
def _row(TBL, r, m, c):
    """The table rows ``r`` [BLOCK] (masked by ``m``) as fp32 [BLOCK, 256]."""
    return tl.load(TBL + r[:, None] * 256 + c[None, :], mask=m[:, None], other=0.0).to(tl.float32)


@triton.jit
def _gather_tile(TBL, w0, w1, cvb, live, c, MODE: tl.constexpr):
    """The sum of the table rows of every pair of a tile, from its code words: [BLOCK, 256] fp32 (zeros for padding)."""
    x = _row(TBL, w0 & 127, live, c)
    if MODE == 0:
        cv = tl.where(live, cvb.to(tl.float32, bitcast=True), 0.0)
        hd = live & ((w0 & (1 << 28)) != 0)
        ho = live & ((w0 & (1 << 29)) != 0)
        x += _row(TBL, 127 + ((w0 >> 8) & 31), hd, c)
        x += _row(TBL, 148 + ((w0 >> 13) & 31), hd, c)
        x += _row(TBL, 169 + ((w0 >> 18) & 31), hd, c)
        x += _row(TBL, 190 + ((w0 >> 23) & 31), hd, c)
        x += _row(TBL, 211 + (w1 & 31), ho, c)
        x += _row(TBL, 232 + ((w1 >> 5) & 31), ho, c)
        x += _row(TBL, 253 + ((w1 >> 10) & 31), ho, c)
        x += _row(TBL, 274 + ((w1 >> 15) & 31), ho, c)
        x += _row(TBL, 295 + ((w1 >> 20) & 31), ho, c)
        x += cv[:, None] * _row(TBL, tl.zeros_like(w0) + 316, live, c)
    else:
        x += _row(TBL, 127 + ((w0 >> 8) & 31), live, c)
    return x


@triton.autotune(configs=configs_for("pair_embed_geom_bins_triton"), key=["shape_key"])
@triton.jit
def _geom_bins(BB, CA, CB, MASK, IDX, CHN, DLIM, ALIM, CODES, n, BLOCK_J: tl.constexpr, shape_key):
    """The 16-byte code of every pair (b, i, j): word 0 = sequence-separation bin | four distance bins << 8, 13, 18, 23 | flags (bit 28 distances, bit 29
    orientations) | bit 31 = live pair; word 1 = the five angle bins (5 bits each); word 2 = the chain product (float bits); word 3 = 0."""
    jb = tl.program_id(0)
    i = tl.program_id(1).to(tl.int64)
    b = tl.program_id(2).to(tl.int64)
    j = (jb * BLOCK_J + tl.arange(0, BLOCK_J)).to(tl.int64)
    jin = j < n
    ri = b * n + i
    rj = b * n + j
    mi = tl.load(MASK + ri) != 0
    cai = tl.load(CA + ri) != 0
    cbi = tl.load(CB + ri) != 0
    mj = tl.load(MASK + rj, mask=jin, other=0) != 0
    caj = tl.load(CA + rj, mask=jin, other=0) != 0
    cbj = tl.load(CB + rj, mask=jin, other=0) != 0
    live = jin & mi & mj
    dv = live & cai & caj
    ori = dv & cbi
    s = tl.minimum(tl.maximum(tl.load(IDX + ri) - tl.load(IDX + rj, mask=jin, other=0) + 63, 0), 126)
    cv = tl.where(live, (tl.load(CHN + ri) * tl.load(CHN + rj, mask=jin, other=0)).to(tl.float32), 0.0)
    cax, cay, caz = _atoms(BB, b, n, i, 1, mi)
    d0 = tl.zeros([BLOCK_J], dtype=tl.int32)
    d1 = tl.zeros([BLOCK_J], dtype=tl.int32)
    d2 = tl.zeros([BLOCK_J], dtype=tl.int32)
    d3 = tl.zeros([BLOCK_J], dtype=tl.int32)
    a0 = tl.zeros([BLOCK_J], dtype=tl.int32)
    a1 = tl.zeros([BLOCK_J], dtype=tl.int32)
    a2 = tl.zeros([BLOCK_J], dtype=tl.int32)
    a3 = tl.zeros([BLOCK_J], dtype=tl.int32)
    a4 = tl.zeros([BLOCK_J], dtype=tl.int32)
    if tl.max(dv.to(tl.int32), axis=0) > 0:
        njx, njy, njz = _atoms(BB, b, n, j, 0, dv)
        cjx, cjy, cjz = _atoms(BB, b, n, j, 1, dv)
        ccx, ccy, ccz = _atoms(BB, b, n, j, 2, dv)
        bjx, bjy, bjz = _atoms(BB, b, n, j, 3, dv)
        dd0 = tl.sqrt_rn((cax - njx) * (cax - njx) + (cay - njy) * (cay - njy) + (caz - njz) * (caz - njz))
        dd1 = tl.sqrt_rn((cax - cjx) * (cax - cjx) + (cay - cjy) * (cay - cjy) + (caz - cjz) * (caz - cjz))
        dd2 = tl.sqrt_rn((cax - ccx) * (cax - ccx) + (cay - ccy) * (cay - ccy) + (caz - ccz) * (caz - ccz))
        dd3 = tl.where(cbj, tl.sqrt_rn((cax - bjx) * (cax - bjx) + (cay - bjy) * (cay - bjy) + (caz - bjz) * (caz - bjz)), 0.0)
        d0 = _bucket(dd0, DLIM, 20)
        d1 = _bucket(dd1, DLIM, 20)
        d2 = _bucket(dd2, DLIM, 20)
        d3 = _bucket(dd3, DLIM, 20)
        if tl.max(ori.to(tl.int32), axis=0) > 0:
            nix, niy, niz = _atoms(BB, b, n, i, 0, mi)
            bix, biy, biz = _atoms(BB, b, n, i, 3, mi)
            t0 = _dihedral(nix, niy, niz, cax, cay, caz, bix, biy, biz, bjx, bjy, bjz)
            t1 = _dihedral(njx, njy, njz, cjx, cjy, cjz, bjx, bjy, bjz, bix, biy, biz)
            t2 = _bond(cax, cay, caz, bix, biy, biz, bjx, bjy, bjz)
            t3 = _bond(cjx, cjy, cjz, bjx, bjy, bjz, bix, biy, biz)
            t4 = _dihedral(cax, cay, caz, bix, biy, biz, bjx, bjy, bjz, cjx, cjy, cjz)
            a0 = _bucket(t0, ALIM, 20)
            a1 = _bucket(t1, ALIM, 20)
            a2 = _bucket(t2, ALIM, 20)
            a3 = _bucket(t3, ALIM, 20)
            a4 = _bucket(t4, ALIM, 20)
    w0 = s | tl.where(dv, (1 << 28) | (d0 << 8) | (d1 << 13) | (d2 << 18) | (d3 << 23), 0) | tl.where(ori, 1 << 29, 0)
    w0 = tl.where(live, w0, 0) | (live.to(tl.int32) << 31)
    w1 = tl.where(ori, a0 | (a1 << 5) | (a2 << 10) | (a3 << 15) | (a4 << 20), 0)
    cb_ = ((b * n + i) * n + j) * 4
    tl.store(CODES + cb_, w0, mask=jin)
    tl.store(CODES + cb_ + 1, w1, mask=jin)
    tl.store(CODES + cb_ + 2, cv.to(tl.int32, bitcast=True), mask=jin)
    tl.store(CODES + cb_ + 3, tl.zeros([BLOCK_J], dtype=tl.int32), mask=jin)


@triton.autotune(configs=configs_for("pair_embed_ca_bins_triton"), key=["shape_key"])
@triton.jit
def _ca_bins(CAX, MASK, IDX, DLIM, CODES, n, ND: tl.constexpr, BLOCK_J: tl.constexpr, shape_key):
    """The decoder pair's code: word 0 = sequence-separation bin | CA-distance bin << 8 | bit 31 = live pair; words 1 .. 3 = 0."""
    jb = tl.program_id(0)
    i = tl.program_id(1).to(tl.int64)
    b = tl.program_id(2).to(tl.int64)
    j = (jb * BLOCK_J + tl.arange(0, BLOCK_J)).to(tl.int64)
    jin = j < n
    ri = b * n + i
    rj = b * n + j
    live = jin & (tl.load(MASK + ri) != 0) & (tl.load(MASK + rj, mask=jin, other=0) != 0)
    s = tl.minimum(tl.maximum(tl.load(IDX + ri) - tl.load(IDX + rj, mask=jin, other=0) + 63, 0), 126)
    xi = tl.load(CAX + ri * 3)
    yi = tl.load(CAX + ri * 3 + 1)
    zi = tl.load(CAX + ri * 3 + 2)
    dx = xi - tl.load(CAX + rj * 3, mask=live, other=0.0)
    dy = yi - tl.load(CAX + rj * 3 + 1, mask=live, other=0.0)
    dz = zi - tl.load(CAX + rj * 3 + 2, mask=live, other=0.0)
    bn = _bucket(tl.sqrt_rn(dx * dx + dy * dy + dz * dz), DLIM, ND)
    w0 = tl.where(live, s | (bn << 8), 0) | (live.to(tl.int32) << 31)
    cb_ = ((b * n + i) * n + j) * 4
    tl.store(CODES + cb_, w0, mask=jin)
    tl.store(CODES + cb_ + 1, tl.zeros([BLOCK_J], dtype=tl.int32), mask=jin)
    tl.store(CODES + cb_ + 2, tl.zeros([BLOCK_J], dtype=tl.int32), mask=jin)
    tl.store(CODES + cb_ + 3, tl.zeros([BLOCK_J], dtype=tl.int32), mask=jin)


@triton.autotune(configs=configs_for("pair_embed_geom_fwd_gather_triton"), key=["shape_key", "MODE", "LN"])
@triton.jit
def _geom_fwd_gather(TBL, CODES, OUT, P, eps, MODE: tl.constexpr, LN: tl.constexpr, BLOCK_P: tl.constexpr, shape_key):
    """pair = the sum of the table rows the codes name, rounded to bf16 (LN: the LayerNorm of the bf16-rounded sum over the 256 channels, no affine)."""
    p = tl.program_id(0).to(tl.int64) * BLOCK_P + tl.arange(0, BLOCK_P)
    pm = p < P
    w0 = tl.load(CODES + p * 4, mask=pm, other=0)
    live = pm & (w0 < 0)
    c = tl.arange(0, 256)
    if MODE == 0:
        w1 = tl.load(CODES + p * 4 + 1, mask=pm, other=0)
        cvb = tl.load(CODES + p * 4 + 2, mask=pm, other=0)
    else:
        w1 = tl.zeros_like(w0)
        cvb = tl.zeros_like(w0)
    x = _gather_tile(TBL, w0, w1, cvb, live, c, MODE)
    if LN:
        x = x.to(tl.bfloat16).to(tl.float32)
        mean = tl.sum(x, axis=1) / 256
        xc = x - mean[:, None]
        rstd = tl.math.rsqrt(tl.sum(xc * xc, axis=1) / 256 + eps)
        x = xc * rstd[:, None]
    tl.store(OUT + p[:, None] * 256 + c[None, :], x.to(tl.bfloat16), mask=pm[:, None])


# ---------------------------------------------------------------------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------------------------------------------------------------------
@triton.autotune(configs=configs_for("pair_embed_geom_dw_triton"), key=["shape_key", "MODE"])
@triton.jit
def _geom_dw(G, CODES, PART, P, per, MODE: tl.constexpr, CH: tl.constexpr, BLOCK_P: tl.constexpr, shape_key):
    """PART[t, row, channels of slice q] = sum over the pairs of slice t of onehot(row) * G[pair, channels]; the pair tiles are visited in order.

    A program owns one block of 128 table rows (``blk``: 0 the sequence separations (+ the chain row), 1 the distance bins (decoder: the CA-distance bins), 2 the
    angle bins), one slice of CH channels and one slice t of the pairs, so each program holds one [128, CH] accumulator and several programs share an SM."""
    NTAB: tl.constexpr = 317 if MODE == 0 else 157
    q = tl.program_id(0)
    blk = tl.program_id(1)
    t = tl.program_id(2).to(tl.int64)
    p0 = t * per
    p1 = tl.minimum(P, p0 + per)
    r = tl.arange(0, 128)
    cc = tl.arange(0, CH)
    acc = tl.zeros([128, CH], dtype=tl.float32)
    acc_h = tl.zeros([CH], dtype=tl.float32)
    for s0 in range(p0, p1, BLOCK_P):
        p = s0 + tl.arange(0, BLOCK_P)
        pm = p < p1
        w0 = tl.load(CODES + p * 4, mask=pm, other=0)
        live = pm & (w0 < 0)
        g = tl.load(G + p[:, None] * 256 + q * CH + cc[None, :], mask=live[:, None], other=0.0)
        if MODE == 0:
            w1 = tl.load(CODES + p * 4 + 1, mask=pm, other=0)
            hd = live & ((w0 & (1 << 28)) != 0) & (blk == 1)
            ho = live & ((w0 & (1 << 29)) != 0) & (blk == 2)
            k0 = tl.where(blk == 0, w0 & 127, tl.where(blk == 1, (w0 >> 8) & 31, w1 & 31))
            v0 = (live & (blk == 0)) | hd | ho
            k1 = tl.where(blk == 1, 21 + ((w0 >> 13) & 31), 21 + ((w1 >> 5) & 31))
            k2 = tl.where(blk == 1, 42 + ((w0 >> 18) & 31), 42 + ((w1 >> 10) & 31))
            k3 = tl.where(blk == 1, 63 + ((w0 >> 23) & 31), 63 + ((w1 >> 15) & 31))
            k4 = 84 + ((w1 >> 20) & 31)
            v1 = hd | ho
            m = (r[:, None] == tl.where(v0, k0, -1)[None, :]) | (r[:, None] == tl.where(v1, k1, -1)[None, :])
            m |= (r[:, None] == tl.where(v1, k2, -1)[None, :]) | (r[:, None] == tl.where(v1, k3, -1)[None, :]) | (r[:, None] == tl.where(ho, k4, -1)[None, :])
            cv = tl.load(CODES + p * 4 + 2, mask=pm, other=0).to(tl.float32, bitcast=True)
            acc_h += tl.sum(tl.where(live & (blk == 0), cv, 0.0)[:, None] * g.to(tl.float32), axis=0)
        else:
            k0 = tl.where(blk == 0, w0 & 127, (w0 >> 8) & 31)
            m = r[:, None] == tl.where(live, k0, -1)[None, :]
        acc = tl.dot(m.to(tl.bfloat16), g, acc)
    nrow = tl.where(blk == 0, 127, tl.where(blk == 1, 84 if MODE == 0 else 30, 105))
    off = tl.where(blk == 0, 0, tl.where(blk == 1, 127, 211))
    base = PART + t * NTAB * 256 + q * CH
    tl.store(base + (off + r)[:, None] * 256 + cc[None, :], acc, mask=(r < nrow)[:, None])
    if MODE == 0:
        if blk == 0:
            tl.store(base + 316 * 256 + cc, acc_h)


@triton.autotune(configs=configs_for("pair_embed_geom_bwd_reduce_triton"), key=["shape_key"])
@triton.jit
def _geom_dw_reduce(PART, DTBL, total, NBLK: tl.constexpr, BLOCK: tl.constexpr, shape_key):
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = k < total
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for t in range(NBLK):
        acc += tl.load(PART + t * total + k, mask=m, other=0.0)
    tl.store(DTBL + k, acc.to(tl.bfloat16), mask=m)


@triton.autotune(configs=configs_for("pair_embed_geom_bwd_ln_triton"), key=["shape_key", "MODE"])
@triton.jit
def _geom_ln_bwd(TBL, CODES, DH, DX, P, eps, MODE: tl.constexpr, BLOCK_P: tl.constexpr, shape_key):
    """dx = rstd * (dh - mean(dh) - hat * mean(dh * hat)) with hat = LN(bf16(sum of the table rows of the pair)); zeros for padding."""
    p = tl.program_id(0).to(tl.int64) * BLOCK_P + tl.arange(0, BLOCK_P)
    pm = p < P
    w0 = tl.load(CODES + p * 4, mask=pm, other=0)
    live = pm & (w0 < 0)
    c = tl.arange(0, 256)
    if MODE == 0:
        w1 = tl.load(CODES + p * 4 + 1, mask=pm, other=0)
        cvb = tl.load(CODES + p * 4 + 2, mask=pm, other=0)
    else:
        w1 = tl.zeros_like(w0)
        cvb = tl.zeros_like(w0)
    x = _gather_tile(TBL, w0, w1, cvb, live, c, MODE)
    x = x.to(tl.bfloat16).to(tl.float32)
    mean = tl.sum(x, axis=1) / 256
    xc = x - mean[:, None]
    rstd = tl.math.rsqrt(tl.sum(xc * xc, axis=1) / 256 + eps)
    hat = xc * rstd[:, None]
    g = tl.load(DH + p[:, None] * 256 + c[None, :], mask=live[:, None], other=0.0).to(tl.float32)
    a = tl.sum(g, axis=1) / 256
    bs = tl.sum(g * hat, axis=1) / 256
    dx = rstd[:, None] * (g - a[:, None] - hat * bs[:, None])
    tl.store(DX + p[:, None] * 256 + c[None, :], tl.where(live[:, None], dx, 0.0).to(tl.bfloat16), mask=pm[:, None])


# ---------------------------------------------------------------------------------------------------------------------------------------
# launchers (the signatures of ``cuda.sm90``)
# ---------------------------------------------------------------------------------------------------------------------------------------
def portable_forward(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, with_codes=False, ln_eps=None):
    """pair [B, n, n, 256] bf16 (``with_codes``: and the codes [B, n, n, 4] int32); ``ln_eps``: LayerNorm(pair) instead of the pair.
    Two kernels: the bins of every pair (written as the 16-byte codes), then the gather of the table rows."""
    B, n = idx.shape
    dev = tbl.device
    key = token_key(n)
    codes = torch.empty(B, n, n, 4, device=dev, dtype=torch.int32)
    _geom_bins[lambda meta: (triton.cdiv(n, meta["BLOCK_J"]), n, B)](bb, ca, cb, mask, idx, chains, dlim, alim, codes, n, shape_key=key)
    out = torch.empty(B, n, n, C, device=dev, dtype=torch.bfloat16)
    P = B * n * n
    _geom_fwd_gather[lambda meta: (triton.cdiv(P, meta["BLOCK_P"]),)](tbl, codes, out, P, float(ln_eps or 0.0), MODE=0, LN=ln_eps is not None, shape_key=key)
    return (out, codes) if with_codes else out


def portable_forward_ca(tbl, ca, mask, idx, dlim, with_codes=False, ln_eps=None):
    B, n = idx.shape
    dev = tbl.device
    key = token_key(n)
    codes = torch.empty(B, n, n, 4, device=dev, dtype=torch.int32)
    _ca_bins[lambda meta: (triton.cdiv(n, meta["BLOCK_J"]), n, B)](ca, mask, idx, dlim, codes, n, ND=int(dlim.numel()), shape_key=key)
    out = torch.empty(B, n, n, C, device=dev, dtype=torch.bfloat16)
    P = B * n * n
    _geom_fwd_gather[lambda meta: (triton.cdiv(P, meta["BLOCK_P"]),)](tbl, codes, out, P, float(ln_eps or 0.0), MODE=1, LN=ln_eps is not None, shape_key=key)
    return (out, codes) if with_codes else out


def portable_backward(g, codes, mode):
    """Table gradient [317 (mode 0) or 157 (mode 1), 256] bf16 from d pair ``g`` [B, n, n, 256] bf16: fixed-order partial sums, no atomics."""
    B, n = g.shape[:2]
    P = B * n * n
    ntab = NT if mode == 0 else NT_CA
    per = triton.cdiv(P, NBLK)
    part = torch.empty(NBLK, ntab, C, device=g.device, dtype=torch.float32)
    key = token_key(n)
    _geom_dw[lambda meta: (C // meta["CH"], 3 if mode == 0 else 2, NBLK)](g, codes, part, P, per, MODE=mode, shape_key=key)
    dtbl = torch.empty(ntab, C, device=g.device, dtype=torch.bfloat16)
    total = ntab * C
    _geom_dw_reduce[lambda meta: (triton.cdiv(total, meta["BLOCK"]),)](part, dtbl, total, NBLK=NBLK, shape_key=key)
    return dtbl


def portable_ln_backward(tbl, codes, dh, mode, eps):
    """d pair [B, n, n, 256] bf16 from d LayerNorm(pair) (the pair is recomputed from the codes)."""
    P = dh.shape[0] * dh.shape[1] * dh.shape[2]
    dx = torch.empty_like(dh)
    _geom_ln_bwd[lambda meta: (triton.cdiv(P, meta["BLOCK_P"]),)](tbl, codes, dh, dx, P, float(eps), MODE=mode, shape_key=token_key(dh.shape[1]))
    return dx
