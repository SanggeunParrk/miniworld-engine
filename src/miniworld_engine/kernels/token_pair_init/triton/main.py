"""AF3-style token-pair initialisation (``token_pair_init``) as portable kernels: the contract of the H100 / B200 CUDA kernels on any card.

    z[b,i,j] = left[b,i] + right[b,j] + Linear_rel(relative_position_one_hot(i, j)) + Linear_bond(one_hot(bond[b,i,j]))

The 139-wide one-hot times ``W_rel`` is a gather of up to four table rows (``T = [W_rel^T ; W_bond^T]`` [n_rel + 2, 128]): the three relative-position
classes of a token pair (residue offset, token offset, chain offset, each clamped), the same-entity flag and the bond row. fp32 throughout, the sum in the
CUDA kernel's order (left + right, class rows, entity row, bond row), so the forward is bit-identical to it.

* forward ``_tpi_fwd``: one program per (batch, row i, block of columns j); z is the only large tensor written.
* backward, no atomics (bitwise reproducible): ``dleft`` / ``dright`` are the sums of dz over j / over i, summed in a fixed order by ``_tpi_dx``;
  the table gradient is ``onehot^T @ dz`` -- ``_tpi_dw`` builds the one-hot of each tile of pairs from the ids (exact in bf16) and multiplies it with dz split into
  three bf16 terms (hi + mid + lo = the fp32 value, so the product is fp32-accurate) on the tensor cores; a program owns one 32-channel slice of a FIXED
  contiguous slice of the pairs and visits the tiles in order; ``_tpi_dw_reduce`` adds the slice partials in slice order.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import token_key

P = 128
NBLK = 192                   # pair slices of the table-gradient kernel (fixed)


def n_rel(r_max: int, s_max: int) -> int:
    return 2 * (2 * r_max + 2) + (2 * s_max + 2) + 1


@triton.jit
def _ids(IDS, BL, idx, ok):
    return (tl.load(IDS + idx, mask=ok, other=0), tl.load(IDS + BL + idx, mask=ok, other=0), tl.load(IDS + 2 * BL + idx, mask=ok, other=0),
            tl.load(IDS + 3 * BL + idx, mask=ok, other=0), tl.load(IDS + 4 * BL + idx, mask=ok, other=0))


@triton.jit
def _bins(ai, ri, ti, ei, si, aj, rj, tj, ej, sj, rmax: tl.constexpr, smax: tl.constexpr):
    """The three relative-position table rows of a token pair and its same-entity flag (AlphaFold 3 SI 2.8)."""
    nr: tl.constexpr = 2 * rmax + 2
    ns: tl.constexpr = 2 * smax + 2
    same_chain = ai == aj
    same_res = ri == rj
    same_ent = ei == ej
    dres = tl.minimum(tl.maximum(ri - rj + rmax, 0), 2 * rmax)
    dtok = tl.minimum(tl.maximum(ti - tj + rmax, 0), 2 * rmax)
    dch = tl.minimum(tl.maximum(si - sj + smax, 0), 2 * smax)
    b1 = tl.where(same_chain, dres, nr - 1)
    b2 = nr + tl.where(same_chain & same_res, dtok, nr - 1)
    b3 = 2 * nr + tl.where(same_ent, dch, ns - 1)
    return b1, b2, b3, same_ent


@triton.jit
def _row(T, r, m, c):
    return tl.load(T + r[:, None] * 128 + c[None, :], mask=m[:, None], other=0.0)


@triton.autotune(configs=configs_for("token_pair_init_fwd_triton"), key=["shape_key"])
@triton.jit
def _tpi_fwd(LEFT, RIGHT, TBL, IDS, BOND, OUT, L, BL, rmax: tl.constexpr, smax: tl.constexpr, BLOCK_J: tl.constexpr, shape_key):
    jb = tl.program_id(0)
    i = tl.program_id(1).to(tl.int64)
    b = tl.program_id(2).to(tl.int64)
    j = (jb * BLOCK_J + tl.arange(0, BLOCK_J)).to(tl.int64)
    jin = j < L
    ri = b * L + i
    rj = b * L + j
    c = tl.arange(0, 128)
    nrel: tl.constexpr = 2 * (2 * rmax + 2) + (2 * smax + 2) + 1
    ai, rsi, ti, ei, si = _ids(IDS, BL, ri, ri >= 0)
    aj, rsj, tj, ej, sj = _ids(IDS, BL, rj, jin)
    b1, b2, b3, se = _bins(ai, rsi, ti, ei, si, aj, rsj, tj, ej, sj, rmax, smax)
    bond = (tl.load(BOND + ri * L + j, mask=jin, other=0) != 0).to(tl.int32)
    o = tl.load(LEFT + ri * 128 + c)[None, :] + tl.load(RIGHT + rj[:, None] * 128 + c[None, :], mask=jin[:, None], other=0.0)
    o += _row(TBL, b1, jin, c)
    o += _row(TBL, b2, jin, c)
    o += _row(TBL, b3, jin, c)
    o += _row(TBL, tl.full([BLOCK_J], nrel - 1, tl.int32), jin & se, c)
    o += _row(TBL, nrel + bond, jin, c)
    tl.store(OUT + (ri * L + j)[:, None] * 128 + c[None, :], o, mask=jin[:, None])


@triton.autotune(configs=configs_for("token_pair_init_dw_triton"), key=["shape_key"])
@triton.jit
def _tpi_dw(G, IDS, BOND, PART, L, BL, P_total, per, rmax: tl.constexpr, smax: tl.constexpr, RBB: tl.constexpr, CH: tl.constexpr, BLOCK_P: tl.constexpr, shape_key):
    """PART[t, row, channels of slice q] = sum over the pairs of slice t of onehot(row) * dz[pair, channels]; pair tiles in order, dz split into 3 bf16 terms.

    The table has n_rel + 2 rows: rows 0 .. 127 (accumulators ``a*``) and the rest (``b*``, RBB rows); each bf16 term of dz has its own accumulator so that the
    summation order of every term does not depend on the tile."""
    nrel: tl.constexpr = 2 * (2 * rmax + 2) + (2 * smax + 2) + 1
    q = tl.program_id(0)
    t = tl.program_id(1).to(tl.int64)
    p0 = t * per
    p1 = tl.minimum(P_total, p0 + per)
    ra = tl.arange(0, 128)
    rb = 128 + tl.arange(0, RBB)
    cc = tl.arange(0, CH)
    a1 = tl.zeros([128, CH], dtype=tl.float32)
    a2 = tl.zeros([128, CH], dtype=tl.float32)
    a3 = tl.zeros([128, CH], dtype=tl.float32)
    b1_ = tl.zeros([RBB, CH], dtype=tl.float32)
    b2_ = tl.zeros([RBB, CH], dtype=tl.float32)
    b3_ = tl.zeros([RBB, CH], dtype=tl.float32)
    for s0 in range(p0, p1, BLOCK_P):
        p = s0 + tl.arange(0, BLOCK_P)
        pm = p < p1
        b = p // (L * L)
        rem = p % (L * L)
        i = rem // L
        j = rem % L
        ai, rsi, ti, ei, si = _ids(IDS, BL, b * L + i, pm)
        aj, rsj, tj, ej, sj = _ids(IDS, BL, b * L + j, pm)
        c1, c2, c3, se = _bins(ai, rsi, ti, ei, si, aj, rsj, tj, ej, sj, rmax, smax)
        bond = (tl.load(BOND + p, mask=pm, other=0) != 0).to(tl.int32)
        r4 = tl.where(se, nrel - 1, -1)
        r5 = nrel + bond
        pmr = pm[None, :]
        oha = (((ra[:, None] == c1[None, :]) | (ra[:, None] == c2[None, :]) | (ra[:, None] == c3[None, :]) | (ra[:, None] == r4[None, :])
                | (ra[:, None] == r5[None, :])) & pmr).to(tl.bfloat16)
        ohb = (((rb[:, None] == c1[None, :]) | (rb[:, None] == c2[None, :]) | (rb[:, None] == c3[None, :]) | (rb[:, None] == r4[None, :])
                | (rb[:, None] == r5[None, :])) & pmr).to(tl.bfloat16)
        g = tl.load(G + p[:, None] * 128 + q * CH + cc[None, :], mask=pm[:, None], other=0.0)
        g1 = g.to(tl.bfloat16)
        r1 = g - g1.to(tl.float32)
        g2 = r1.to(tl.bfloat16)
        g3 = (r1 - g2.to(tl.float32)).to(tl.bfloat16)
        a1 = tl.dot(oha, g1, a1)
        b1_ = tl.dot(ohb, g1, b1_)
        a2 = tl.dot(oha, g2, a2)
        b2_ = tl.dot(ohb, g2, b2_)
        a3 = tl.dot(oha, g3, a3)
        b3_ = tl.dot(ohb, g3, b3_)
    acc_a = a1 + (a2 + a3)           # three bf16 terms of dz, the small ones first
    acc_b = b1_ + (b2_ + b3_)
    nbin: tl.constexpr = nrel + 2
    base = PART + t * nbin * 128 + q * CH
    tl.store(base + ra[:, None] * 128 + cc[None, :], acc_a, mask=(ra < nbin)[:, None])
    tl.store(base + rb[:, None] * 128 + cc[None, :], acc_b, mask=(rb < nbin)[:, None])


@triton.autotune(configs=configs_for("token_pair_init_dw_reduce_triton"), key=["shape_key"])
@triton.jit
def _tpi_dw_reduce(PART, OUT, total, NBLK: tl.constexpr, BLOCK: tl.constexpr, shape_key):
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = k < total
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for t in range(NBLK):
        acc += tl.load(PART + t * total + k, mask=m, other=0.0)
    tl.store(OUT + k, acc, mask=m)


@triton.autotune(configs=configs_for("token_pair_init_bwd_dx_triton"), key=["shape_key", "OVER_I"])
@triton.jit
def _tpi_dx(G, OUT, L, OVER_I: tl.constexpr, BLOCK_J: tl.constexpr, shape_key):
    """OVER_I = 0: OUT[b, i, :] = sum over j of G[b, i, j, :]; OVER_I = 1: OUT[b, j, :] = sum over i of G[b, i, j, :]. Fixed order (j blocks / i rows in turn)."""
    c = tl.arange(0, 128)
    if OVER_I:
        b = tl.program_id(1).to(tl.int64)
        j = (tl.program_id(0) * BLOCK_J + tl.arange(0, BLOCK_J)).to(tl.int64)
        jin = j < L
        acc = tl.zeros([BLOCK_J, 128], dtype=tl.float32)
        for i in range(L):
            acc += tl.load(G + ((b * L + i) * L + j)[:, None] * 128 + c[None, :], mask=jin[:, None], other=0.0)
        tl.store(OUT + (b * L + j)[:, None] * 128 + c[None, :], acc, mask=jin[:, None])
    else:
        b = tl.program_id(1).to(tl.int64)
        i = tl.program_id(0).to(tl.int64)
        acc = tl.zeros([128], dtype=tl.float32)
        for j0 in range(0, L, BLOCK_J):
            j = (j0 + tl.arange(0, BLOCK_J)).to(tl.int64)
            jin = j < L
            tile = tl.load(G + ((b * L + i) * L + j)[:, None] * 128 + c[None, :], mask=jin[:, None], other=0.0)
            acc += tl.sum(tile, axis=0)
        tl.store(OUT + (b * L + i) * 128 + c, acc)


def portable_forward(left, right, tbl, ids, bond, r_max: int, s_max: int) -> torch.Tensor:
    """z [B, L, L, 128] fp32 from left / right [B, L, 128], the table [n_rel + 2, 128], ids [5, B*L] int32, bond [B*L*L] uint8."""
    B, L, _ = left.shape
    out = torch.empty(B, L, L, P, device=left.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(L, meta["BLOCK_J"]), L, B)
    _tpi_fwd[grid](left, right, tbl, ids, bond, out, L, B * L, rmax=int(r_max), smax=int(s_max), shape_key=token_key(L))
    return out


def portable_backward(g, ids, bond, r_max: int, s_max: int):
    """(dleft [B, L, 128], dright [B, L, 128], dtbl [n_rel + 2, 128]) of :func:`forward` from dz = ``g`` [B, L, L, 128] fp32."""
    B, L = g.shape[:2]
    dev = g.device
    nbin = n_rel(r_max, s_max) + 2
    key = token_key(L)
    dleft = torch.empty(B, L, P, device=dev, dtype=torch.float32)
    dright = torch.empty(B, L, P, device=dev, dtype=torch.float32)
    _tpi_dx[lambda meta: (L, B)](g, dleft, L, OVER_I=0, shape_key=key)
    _tpi_dx[lambda meta: (triton.cdiv(L, meta["BLOCK_J"]), B)](g, dright, L, OVER_I=1, shape_key=key)
    total_pairs = B * L * L
    part = torch.empty(NBLK, nbin, P, device=dev, dtype=torch.float32)
    rbb = max(16, triton.next_power_of_2(max(nbin - 128, 1)))
    _tpi_dw[lambda meta: (P // meta["CH"], NBLK)](g, ids, bond, part, L, B * L, total_pairs, triton.cdiv(total_pairs, NBLK), rmax=int(r_max), smax=int(s_max),
                                                  RBB=rbb, shape_key=key)
    dtbl = torch.empty(nbin, P, device=dev, dtype=torch.float32)
    total = nbin * P
    _tpi_dw_reduce[lambda meta: (triton.cdiv(total, meta["BLOCK"]),)](part, dtbl, total, NBLK=NBLK, shape_key=key)
    return dleft, dright, dtbl
