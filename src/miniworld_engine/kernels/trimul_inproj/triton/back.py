"""Fused trimul back-half in Triton: LN_out + proj-gemm + gate-gemm + mul, one kernel.

Computes, per pair row (gate is computed IN the back, not the front — no gate
materialization, no separate mul pass):

    proj = LayerNorm_D(tri) @ Wp           # tri: bmm output, LN over channel D
    gate = sigmoid(x_n @ Wg)               # x_n: the LN_in'd pair (front input)
    y    = gate * proj                     # [B, L, L, D]

Reads tri + x_n, writes y (3T, no intermediate materialized). tri comes in bdll
as a (D, M) contiguous view (channel-major, m strided by 1 within a plane);
x_n is (M, D) blld contiguous. B=1, K=N=D=128, bf16.
"""

from __future__ import annotations
from miniworld_engine.autotune.configs import configs_for

import torch
import triton
import triton.language as tl


from miniworld_engine.autotune import key_bucket_of, tensor_dtype_of
from miniworld_engine.autotune.shape_key import token_key


# B200 (sm_100) pruned set. Swept BM in {32,64,128,256,512} x warps {4,8,16}
# x stages {2,3,4,5} for L in {384,512,768,1024}. Findings:
#   - BM>=128 fails ptxas register allocation (255 regs) on sm_100 when the two
#     128-wide GEMM accumulators (proj, gate) are both live at full N.
#   - BM=512 exceeds tensor memory (OutOfResources).
#   - There is no K-loop, so num_stages does not pipeline; it is effectively noise.
#   - The output dim N is tiled by BN (static_range over N): each program reuses
#     the single LN-normed row tile and xn tile across the N-subtiles, but only
#     keeps a (BM, BN) accumulator pair live at a time. BN=64 halves the live
#     accumulator registers vs the old full-N kernel, lifting occupancy and giving
#     ~13-15% over the prior BM=64 full-N winner at every L. Casting `norm` to
#     bf16 up front (instead of inside each dot) further trims register pressure
#     under the N-tiling and is faster here (it was a wash without N-tiling).
#   - BLOCK_M1=64, BLOCK_N=64, num_warps=4 is the winner for every L.
#
# REGISTER CEILING, and the CSV is what has to respect it: this kernel holds a live
# (BLOCK_M1, BLOCK_N) accumulator PAIR (proj + gate) at full K=N=128 with no K-loop, so wide tiles
# blow the 255-register budget. Rows at 256 tiles, num_warps=16, or num_stages>=4 make ptxas
# spill/thrash for 20+ MINUTES per config. Nothing filters such a row out any more -- a config set
# that contains one stalls the run.

# BK tiles the contraction / LN-reduce axis, which used to be the raw shape constant K
# (`tl.arange(0, K)`, a whole [BM, K] row pinned on-chip). It is a CSV tile rather
# than the narrow BLOCK_K so the sweep can still express "one tile holds the whole row" -- that
# schedule is what makes this kernel a single-pass LN, and the narrow set (<=128) would have
# forced a multi-pass at every d_pair > 128. The k-loops below make the smaller candidates
# correct; the pruned BM/BN box above is unchanged.


@triton.autotune(configs=configs_for("trimul_outproj_layernorm_gemm_gate_triton"), key=['shape_key', 'K', 'ADD_RESIDUAL'])
@triton.jit
def _back_kernel(
    tri_ptr,  # (D, M) channel-major: tri[k, m] at k*M + m
    xn_ptr,   # (M, D) row-major
    wp_ptr, wg_ptr,    # (D, D) = to_out.weight.T, to_gate.weight.T  (K=in, N=out)
    lnw_ptr, lnb_ptr,  # (D,)
    y_ptr,    # (M, D) row-major
    res_ptr,  # (M, D) row-major residual (== the module input pair); read iff ADD_RESIDUAL
    M, eps,
    K: tl.constexpr, N: tl.constexpr, BLOCK_M1: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    shape_key, ADD_RESIDUAL: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    rm = pid * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    mmask = rm[:, None] < M

    if BLOCK_K >= K:
        # COVERING TILE (BLOCK_K and K are both tl.constexpr -> this branch is selected at COMPILE
        # time and only one of the two is ever emitted). One tile holds the whole LN row, so
        # read `tri` ONCE, keep the fp32 centered row and the bf16 `norm` in registers, and
        # reuse them across every N-subtile. This is exactly the pre-tiling single-pass
        # schedule; the k-tiled `else` below is the general (BLOCK_K < K) form. Numerics are
        # identical to the else-branch at BLOCK_K >= K: the loops there are single-trip and the
        # arithmetic is written to match term for term.
        rk = tl.arange(0, BLOCK_K)
        kmask1 = rk < K
        kmask = kmask1[None, :]
        tri = tl.load(tri_ptr + rk[None, :] * M + rm[:, None],
                      mask=mmask & kmask, other=0.0).to(tl.float32)
        mean = tl.sum(tri, axis=1) / K
        xc = tl.where(kmask, tri - mean[:, None], 0.0)
        var = tl.sum(xc * xc, axis=1) / K
        rstd = 1.0 / tl.sqrt(var + eps)
        lnw = tl.load(lnw_ptr + rk, mask=kmask1, other=0.0).to(tl.float32)
        lnb = tl.load(lnb_ptr + rk, mask=kmask1, other=0.0).to(tl.float32)
        norm = tl.where(
            kmask, (xc * rstd[:, None]) * lnw[None, :] + lnb[None, :], 0.0,
        ).to(tl.bfloat16)
        xn = tl.load(xn_ptr + rm[:, None] * K + rk[None, :], mask=mmask & kmask, other=0.0)
        for j in tl.static_range(0, N, BLOCK_N):
            rn = j + tl.arange(0, BLOCK_N)
            nmask = rn[None, :] < N
            wp = tl.load(wp_ptr + rk[:, None] * N + rn[None, :],
                         mask=kmask1[:, None] & nmask, other=0.0)
            wg = tl.load(wg_ptr + rk[:, None] * N + rn[None, :],
                         mask=kmask1[:, None] & nmask, other=0.0)
            proj = tl.dot(norm, wp)                              # (BLOCK_M1, BLOCK_N)
            gate = tl.sigmoid(tl.dot(xn, wg))                    # (BLOCK_M1, BLOCK_N)
            acc = proj * gate
            if ADD_RESIDUAL:
                res = tl.load(res_ptr + rm[:, None] * N + rn[None, :],
                              mask=mmask & nmask, other=0.0).to(tl.float32)
                acc = acc + res
            tl.store(y_ptr + rm[:, None] * N + rn[None, :],
                     acc.to(y_ptr.dtype.element_ty), mask=mmask & nmask)
    else:
        # --- LN row statistics over K-tiles. Two sweeps (mean, then CENTERED variance) so the fp32
        # algebra at BLOCK_K >= K is exactly the original single-tile one. tri is (D, M) channel-major, so
        # each k-slice is a coalesced run over m. ---
        s = tl.zeros([BLOCK_M1], dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            kmask = rk[None, :] < K
            tri = tl.load(tri_ptr + rk[None, :] * M + rm[:, None],
                          mask=mmask & kmask, other=0.0).to(tl.float32)
            s += tl.sum(tri, axis=1)
        mean = s / K
        s = tl.zeros([BLOCK_M1], dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            kmask = rk[None, :] < K
            tri = tl.load(tri_ptr + rk[None, :] * M + rm[:, None],
                          mask=mmask & kmask, other=0.0).to(tl.float32)
            xc = tl.where(kmask, tri - mean[:, None], 0.0)
            s += tl.sum(xc * xc, axis=1)
        var = s / K
        rstd = 1.0 / tl.sqrt(var + eps)

        # Tile the output dim N: keep only a (BLOCK_M1, BLOCK_N) accumulator pair live at a time.
        # rn is masked against N so BLOCK_N need not divide N (a config with BLOCK_N>N — e.g. the
        # full-grid autotune fallback on a stale cache — reads/writes only in-bounds; the
        # out-of-range weight columns load as 0 and are dropped in the masked store, so the
        # result is identical for every BLOCK_N. Without this mask BLOCK_N>N faults (illegal address).
        for j in tl.static_range(0, N, BLOCK_N):
            rn = j + tl.arange(0, BLOCK_N)
            nmask = rn[None, :] < N
            pacc = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
            gacc = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K):
                rk = k0 + tl.arange(0, BLOCK_K)
                kmask1 = rk < K
                kmask = kmask1[None, :]
                tri = tl.load(tri_ptr + rk[None, :] * M + rm[:, None],
                              mask=mmask & kmask, other=0.0).to(tl.float32)
                lnw = tl.load(lnw_ptr + rk, mask=kmask1, other=0.0).to(tl.float32)
                lnb = tl.load(lnb_ptr + rk, mask=kmask1, other=0.0).to(tl.float32)
                xc = tl.where(kmask, tri - mean[:, None], 0.0)
                # LN-normed tile, cast to bf16 once (matches the original operand dtype).
                norm = tl.where(
                    kmask, (xc * rstd[:, None]) * lnw[None, :] + lnb[None, :], 0.0,
                ).to(tl.bfloat16)
                xn = tl.load(xn_ptr + rm[:, None] * K + rk[None, :],
                             mask=mmask & kmask, other=0.0)
                wp = tl.load(wp_ptr + rk[:, None] * N + rn[None, :],
                             mask=kmask1[:, None] & nmask, other=0.0)
                wg = tl.load(wg_ptr + rk[:, None] * N + rn[None, :],
                             mask=kmask1[:, None] & nmask, other=0.0)
                pacc = tl.dot(norm, wp, pacc)
                gacc = tl.dot(xn, wg, gacc)
            proj = pacc                                              # (BLOCK_M1, BLOCK_N)
            gate = tl.sigmoid(gacc)                                  # (BLOCK_M1, BLOCK_N)
            acc = proj * gate
            if ADD_RESIDUAL:
                # Fuse the pairformer residual add y = pair + trimul(pair): the residual is the
                # module's own (pre-LN) input, added in the same coalesced store. No dropout here
                # (inference: dropout is identity; training uses the v6 kernel).
                res = tl.load(res_ptr + rm[:, None] * N + rn[None, :], mask=mmask & nmask, other=0.0).to(tl.float32)
                acc = acc + res
            y = acc.to(y_ptr.dtype.element_ty)
            tl.store(y_ptr + rm[:, None] * N + rn[None, :], y, mask=mmask & nmask)


def trimul_back_triton(tri_bdll, x_n, Wp, Wg, ln_w, ln_b, eps=1e-5, residual=None):
    """tri_bdll:(B,D,L,L), x_n:(B,L,L,D), Wp/Wg:(D,D)=weight.T -> y:(B,L,L,D). B=1.

    ``residual`` (optional, [B,L,L,D] == the module input pair): fuses the pairformer
    residual add ``y = pair + trimul(pair)`` into the store epilogue. None -> plain trimul.
    """
    # B==1 by design; B>1 works via a per-batch loop, which is faster than a batched single
    # launch (L2 thrashing of large bdll intermediates). See front.py's note and
    # docs/kernel-optimization/trimul_batch_generalization.
    B, D, L, L2 = tri_bdll.shape
    assert B == 1 and L == L2
    M = L * L
    tri_dm = tri_bdll.reshape(D, M)            # (D, M) contiguous, channel-major
    xn_flat = x_n.reshape(M, D)
    y = torch.empty(M, D, device=x_n.device, dtype=x_n.dtype)
    add_residual = residual is not None
    res_flat = residual.reshape(M, D).contiguous() if add_residual else y  # dummy ptr when off
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M1"]),)  # noqa: E731
    _back_kernel[grid](tri_dm, xn_flat, Wp.contiguous(), Wg.contiguous(),
                       ln_w.contiguous(), ln_b.contiguous(), y, res_flat, M, float(eps),
                       K=D, N=D, shape_key=token_key(L), ADD_RESIDUAL=add_residual)
    return y.view(B, L, L, D)
