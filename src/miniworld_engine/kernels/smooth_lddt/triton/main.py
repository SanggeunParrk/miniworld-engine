"""Smooth lDDT (AF3 Algorithm 27) in portable kernels: forward value and analytic coordinate gradient, fp32 compute.

The same op as ``cuda/sm100/smooth_lddt.cu`` -- no [N, N] float tensor, no per-augment pair tensor -- for the cards that have no CUDA
kernel. A tile is a dense block of pairs (not a walk over set bits) and the augments share one per-call pair-set summary.

* ``smooth_lddt_prep_triton`` (grid B x NB x NB): for every 32 x 32 tile of the ground-truth pair matrix one flag byte (bit 0: some pair
  of the loss lies in the tile, bit 1: some pair carrying gradient to either atom does) and the number of pairs of the loss in it.
  The augments are rigid motions of one structure, so this is built once per call, from ``gt`` only.
* ``smooth_lddt_fwd_triton`` (grid A*B x N / BLOCK_I): one program owns BLOCK_I rows ``i``; it visits the BLOCK_J-column tiles whose flags say "pairs here", recomputes
  the ground-truth distance of each pair (3 FMA, cheaper than a bit-mask load per augment), masks with ``mask``, ``i != j`` and the radius
  of atom ``i``, and sums ``mean_t sigmoid(t - |d_pred - d_gt|)``. The four sigmoids are one rational function of ``w = exp(-|.|)``.
* ``smooth_lddt_bwd_triton``: the same walk; every atom collects the gradient of BOTH roles it plays in a pair (weight
  ``[g < cut_i] + [g < cut_j]``), so no atomic exists anywhere and the result is deterministic. The reductions are fixed-order
  (``tl.sum`` over a tile, the tiles in loop order, the per-row scores added by torch in a fixed order).

Layout: x [A, B, N, 3] fp32 or bf16 (loaded and computed in fp32; the gradient is stored in x's dtype), g [B, N, 3] fp32, nuc / mask [B, N] int8; N % 128 == 0 (the caller pads). Flags and counts: [B, N / 32, N / 32].
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import atom_key
from miniworld_engine.kernels._compile import opaque

BLK = 32   # atoms per flag tile edge, fixed; the kernels' own tiles (BLOCK_I x BLOCK_J, 32 / 64 / 128) are tuned over it
NUC_CUT, OTHER_CUT = 30.0, 15.0


@triton.jit
def _score(w):
    """sum over t in (.5, 1, 2, 4) of sigmoid(t - x) / 4, for w = exp(-x), x >= 0: ``w * Q'(w) / (4 Q(w))``, Q = prod_t (w + exp(-t))."""
    q = w + 1.12806102
    q = q * w + 0.375327923
    q = q * w + 0.036699476
    q = q * w + 0.00055308437
    dq = 4.0 * w + 3.38418307
    dq = dq * w + 0.750655846
    dq = dq * w + 0.036699476
    return 0.25 * w * dq / q


@triton.jit
def _dscore(w):
    """-d(score)/dx for w = exp(-x): ``w N(w) / (4 Q(w)^2)``."""
    q = w + 1.12806102
    q = q * w + 0.375327923
    q = q * w + 0.036699476
    q = q * w + 0.00055308437
    n = 1.12806102 * w + 1.50131169
    n = n * w + 0.753688085
    n = n * w + 0.174446344
    n = n * w + 0.0193895544
    n = n * w + 0.000830352031
    n = n * w + 2.02979066e-05
    return 0.25 * w * n / (q * q)


@triton.autotune(configs=configs_for("smooth_lddt_prep_triton"), key=["shape_key"])
@triton.jit
def smooth_lddt_prep_kernel(
    G, NUC, M, TILE, CNT,
    N, NF, shape_key,
    BLOCK_P: tl.constexpr,
):
    """One program: a BLOCK_P x BLOCK_P block of the pair matrix, written as (BLOCK_P / 32)^2 flag tiles."""
    b = tl.program_id(0)
    gb = G + b * N * 3
    for si in tl.static_range(BLOCK_P // 32):
        ri = (tl.program_id(1) * (BLOCK_P // 32) + si) * 32 + tl.arange(0, 32)
        gi0 = tl.load(gb + ri * 3)
        gi1 = tl.load(gb + ri * 3 + 1)
        gi2 = tl.load(gb + ri * 3 + 2)
        mi = tl.load(M + b * N + ri) != 0
        ci = tl.where(tl.load(NUC + b * N + ri) != 0, 30.0, 15.0)
        for sj in tl.static_range(BLOCK_P // 32):
            rj = (tl.program_id(2) * (BLOCK_P // 32) + sj) * 32 + tl.arange(0, 32)
            gj0 = tl.load(gb + rj * 3)
            gj1 = tl.load(gb + rj * 3 + 1)
            gj2 = tl.load(gb + rj * 3 + 2)
            mj = tl.load(M + b * N + rj) != 0
            cj = tl.where(tl.load(NUC + b * N + rj) != 0, 30.0, 15.0)
            h0 = gi0[:, None] - gj0[None, :]
            h1 = gi1[:, None] - gj1[None, :]
            h2 = gi2[:, None] - gj2[None, :]
            g2 = h0 * h0 + h1 * h1 + h2 * h2
            ok = mi[:, None] & mj[None, :] & (ri[:, None] != rj[None, :])
            f1 = ok & (g2 < (ci * ci)[:, None])
            f2 = ok & (g2 < (cj * cj)[None, :])
            n1 = tl.sum(f1.to(tl.int32))
            nu = tl.sum((f1 | f2).to(tl.int32))
            flag = (n1 > 0).to(tl.int32) | ((nu > 0).to(tl.int32) << 1)
            t = (b * NF + (tl.program_id(1) * (BLOCK_P // 32) + si)) * NF + (tl.program_id(2) * (BLOCK_P // 32) + sj)
            tl.store(TILE + t, flag.to(tl.int8))
            tl.store(CNT + t, n1)


@triton.jit
def _tile_has(TILE, b, NF, i0, j0, BIT: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    """Does any 32 x 32 flag tile inside rows [i0, i0 + BLOCK_I) x columns [j0, j0 + BLOCK_J) have bit ``BIT``?"""
    fi = i0 // 32 + tl.arange(0, BLOCK_I // 32)
    fj = j0 // 32 + tl.arange(0, BLOCK_J // 32)
    fl = tl.load(TILE + (b * NF + fi)[:, None] * NF + fj[None, :])
    return tl.max(tl.max(((fl >> BIT) & 1).to(tl.int32), axis=1), axis=0) != 0


@triton.autotune(configs=configs_for("smooth_lddt_fwd_triton"), key=["shape_key"])
@triton.jit
def smooth_lddt_fwd_kernel(
    X, G, NUC, M, TILE, OUT,
    B, N, NF, shape_key,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr,
):
    """OUT[ab, i] = sum over the pairs (i, j) of the loss of the pair score; one program owns BLOCK_I rows."""
    ab = tl.program_id(0)
    b = ab % B
    i0 = tl.program_id(1) * BLOCK_I
    ri = i0 + tl.arange(0, BLOCK_I)
    xb = X + ab * N * 3
    gb = G + b * N * 3
    xi0 = tl.load(xb + ri * 3).to(tl.float32)
    xi1 = tl.load(xb + ri * 3 + 1).to(tl.float32)
    xi2 = tl.load(xb + ri * 3 + 2).to(tl.float32)
    gi0 = tl.load(gb + ri * 3)
    gi1 = tl.load(gb + ri * 3 + 1)
    gi2 = tl.load(gb + ri * 3 + 2)
    mi = tl.load(M + b * N + ri) != 0
    ci2 = tl.where(tl.load(NUC + b * N + ri) != 0, 900.0, 225.0)
    acc = tl.zeros([BLOCK_I], dtype=tl.float32)
    for j0 in range(0, N, BLOCK_J):
        if _tile_has(TILE, b, NF, i0, j0, 0, BLOCK_I, BLOCK_J):
            rj = j0 + tl.arange(0, BLOCK_J)
            xj0 = tl.load(xb + rj * 3).to(tl.float32)
            xj1 = tl.load(xb + rj * 3 + 1).to(tl.float32)
            xj2 = tl.load(xb + rj * 3 + 2).to(tl.float32)
            gj0 = tl.load(gb + rj * 3)
            gj1 = tl.load(gb + rj * 3 + 1)
            gj2 = tl.load(gb + rj * 3 + 2)
            mj = tl.load(M + b * N + rj) != 0
            h0 = gi0[:, None] - gj0[None, :]
            h1 = gi1[:, None] - gj1[None, :]
            h2 = gi2[:, None] - gj2[None, :]
            g2 = h0 * h0 + h1 * h1 + h2 * h2
            e0 = xi0[:, None] - xj0[None, :]
            e1 = xi1[:, None] - xj1[None, :]
            e2 = xi2[:, None] - xj2[None, :]
            d2 = e0 * e0 + e1 * e1 + e2 * e2
            keep = mi[:, None] & mj[None, :] & (ri[:, None] != rj[None, :]) & (g2 < ci2[:, None])
            diff = tl.abs(tl.sqrt(d2) - tl.sqrt(g2))
            s = _score(tl.exp(-diff))
            acc += tl.sum(tl.where(keep, s, 0.0), axis=1)
    tl.store(OUT + ab * N + ri, acc)


@triton.autotune(configs=configs_for("smooth_lddt_bwd_triton"), key=["shape_key"])
@triton.jit
def smooth_lddt_bwd_kernel(
    X, G, NUC, M, TILE, GO, GX,
    B, N, NF, shape_key,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr,
):
    ab = tl.program_id(0)
    b = ab % B
    i0 = tl.program_id(1) * BLOCK_I
    ri = i0 + tl.arange(0, BLOCK_I)
    xb = X + ab * N * 3
    gb = G + b * N * 3
    xi0 = tl.load(xb + ri * 3).to(tl.float32)
    xi1 = tl.load(xb + ri * 3 + 1).to(tl.float32)
    xi2 = tl.load(xb + ri * 3 + 2).to(tl.float32)
    gi0 = tl.load(gb + ri * 3)
    gi1 = tl.load(gb + ri * 3 + 1)
    gi2 = tl.load(gb + ri * 3 + 2)
    mi = tl.load(M + b * N + ri) != 0
    ci2 = tl.where(tl.load(NUC + b * N + ri) != 0, 900.0, 225.0)
    a0 = tl.zeros([BLOCK_I], dtype=tl.float32)
    a1 = tl.zeros([BLOCK_I], dtype=tl.float32)
    a2 = tl.zeros([BLOCK_I], dtype=tl.float32)
    for j0 in range(0, N, BLOCK_J):
        if _tile_has(TILE, b, NF, i0, j0, 1, BLOCK_I, BLOCK_J):
            rj = j0 + tl.arange(0, BLOCK_J)
            xj0 = tl.load(xb + rj * 3).to(tl.float32)
            xj1 = tl.load(xb + rj * 3 + 1).to(tl.float32)
            xj2 = tl.load(xb + rj * 3 + 2).to(tl.float32)
            gj0 = tl.load(gb + rj * 3)
            gj1 = tl.load(gb + rj * 3 + 1)
            gj2 = tl.load(gb + rj * 3 + 2)
            mj = tl.load(M + b * N + rj) != 0
            cj2 = tl.where(tl.load(NUC + b * N + rj) != 0, 900.0, 225.0)
            h0 = gi0[:, None] - gj0[None, :]
            h1 = gi1[:, None] - gj1[None, :]
            h2 = gi2[:, None] - gj2[None, :]
            g2 = h0 * h0 + h1 * h1 + h2 * h2
            e0 = xi0[:, None] - xj0[None, :]
            e1 = xi1[:, None] - xj1[None, :]
            e2 = xi2[:, None] - xj2[None, :]
            d = tl.sqrt(e0 * e0 + e1 * e1 + e2 * e2)
            df = d - tl.sqrt(g2)
            ok = mi[:, None] & mj[None, :] & (ri[:, None] != rj[None, :])
            wgt = tl.where(ok & (g2 < ci2[:, None]), 1.0, 0.0) + tl.where(ok & (g2 < cj2[None, :]), 1.0, 0.0)
            sgn = tl.where(df > 0.0, 1.0, tl.where(df < 0.0, -1.0, 0.0))
            coef = -wgt * sgn * _dscore(tl.exp(-tl.abs(df))) / tl.maximum(d, 1e-6)
            a0 += tl.sum(coef * e0, axis=1)
            a1 += tl.sum(coef * e1, axis=1)
            a2 += tl.sum(coef * e2, axis=1)
    go = tl.load(GO + ab)
    gx = GX + ab * N * 3
    tl.store(gx + ri * 3, (a0 * go).to(GX.dtype.element_ty))
    tl.store(gx + ri * 3 + 1, (a1 * go).to(GX.dtype.element_ty))
    tl.store(gx + ri * 3 + 2, (a2 * go).to(GX.dtype.element_ty))


def _smooth_lddt_prep_portable_fake(gt, nuc, mask):
    """[tile flags, pair counts per tile], both [B, N / 32, N / 32]."""
    b, n = mask.shape
    nf = n // BLK
    return [torch.empty((b, nf, nf), dtype=torch.int8, device=gt.device), torch.empty((b, nf, nf), dtype=torch.int32, device=gt.device)]


@opaque(fake=_smooth_lddt_prep_portable_fake, name="smooth_lddt_prep_portable")
def smooth_lddt_prep_portable(gt: torch.Tensor, nuc: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
    """[tile flags, pair counts per tile], both [B, N / 32, N / 32], of gt [B, N, 3] fp32 (N % 128 == 0)."""
    b, n = mask.shape
    nf = n // BLK
    tile = torch.empty((b, nf, nf), dtype=torch.int8, device=gt.device)
    cnt = torch.empty((b, nf, nf), dtype=torch.int32, device=gt.device)
    grid = lambda meta: (b, nf // (meta["BLOCK_P"] // BLK), nf // (meta["BLOCK_P"] // BLK))
    smooth_lddt_prep_kernel[grid](gt, nuc, mask, tile, cnt, n, nf, shape_key=atom_key(n))
    return [tile, cnt]


def _smooth_lddt_fwd_portable_fake(x, gt, nuc, mask, tile):
    """The per-row pair scores [A * B, N] fp32."""
    return torch.empty((x.shape[0] * x.shape[1], x.shape[2]), dtype=torch.float32, device=x.device)


@opaque(fake=_smooth_lddt_fwd_portable_fake, name="smooth_lddt_fwd_portable")
def smooth_lddt_fwd_portable(x: torch.Tensor, gt: torch.Tensor, nuc: torch.Tensor, mask: torch.Tensor, tile: torch.Tensor) -> torch.Tensor:
    """The pair scores summed per row i, [A * B, N] fp32 (the caller adds the rows up)."""
    a, b, n, _ = x.shape
    out = torch.empty((a * b, n), dtype=torch.float32, device=x.device)
    grid = lambda meta: (a * b, n // meta["BLOCK_I"])
    smooth_lddt_fwd_kernel[grid](x, gt, nuc, mask, tile, out, b, n, n // BLK, shape_key=atom_key(n))
    return out


def _smooth_lddt_bwd_portable_fake(x, gt, nuc, mask, tile, go):
    """The coordinate gradient, the shape and dtype of x."""
    return torch.empty_like(x)


@opaque(fake=_smooth_lddt_bwd_portable_fake, name="smooth_lddt_bwd_portable")
def smooth_lddt_bwd_portable(x: torch.Tensor, gt: torch.Tensor, nuc: torch.Tensor, mask: torch.Tensor, tile: torch.Tensor,
                             go: torch.Tensor) -> torch.Tensor:
    """d(sum of the pair scores)/dx [A, B, N, 3] fp32 for the upstream gradient ``go`` [A * B] fp32."""
    a, b, n, _ = x.shape
    gx = torch.empty_like(x)
    grid = lambda meta: (a * b, n // meta["BLOCK_I"])
    smooth_lddt_bwd_kernel[grid](x, gt, nuc, mask, tile, go, gx, b, n, n // BLK, shape_key=atom_key(n))
    return gx
