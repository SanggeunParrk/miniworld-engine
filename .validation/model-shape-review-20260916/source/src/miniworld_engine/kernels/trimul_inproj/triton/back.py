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

from miniworld_engine.kernels._compile import opaque
import triton
import triton.language as tl


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


# N is constexpr but deliberately NOT in the key: trimul_back_triton is the only launch site and it
# passes ``K=D, N=D`` (Wp/Wg are (D, D)), so N == K and the K entry already covers it. The
# BLOCK_K >= K covering-tile branch is likewise selected by K, which is folded into shape_key.
# There is no ADD_RESIDUAL. The pairformer residual is part of what this op IS, not an option
# it offers, so `residual` is required and the add is unconditional -- see gate_elem.py for the
# measurement (1.27x at both production lengths, and the more accurate of the two forms).
@triton.autotune(configs=configs_for("trimul_outproj_layernorm_gemm_gate_triton"), key=['shape_key'])
@triton.jit
def _back_kernel(
    tri_ptr,  # (K, M) channel-major: tri[k, m] at k*M + m
    xn_ptr,   # (M, KG) row-major
    wp_ptr,   # (K, N)  = to_out.weight.T
    wg_ptr,   # (KG, N) = to_gate.weight.T
    lnw_ptr, lnb_ptr,  # (K,)
    y_ptr,    # (M, N) row-major
    res_ptr,  # (M, N) row-major residual (== the module input pair); always read
    M, eps,
    K: tl.constexpr, N: tl.constexpr, KG: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    shape_key,
):
    # K is the axis LN normalises over and the proj GEMM contracts; KG is the axis the GATE
    # contracts. They were ONE axis, which is what made this kernel refuse the bidirectional
    # trimul: there the tri block is 2*d_hidden wide while x_n stays d_pair, so a single K cannot
    # describe both and the module fell back to a split back half -- `_te_forward` (LN+GEMM) then
    # `gate_elem_infer`, two passes over M x N with a materialised (M, N) proj between them.
    # Measured on an A6000 at L=1024, d_pair=128: split 5.06 ms (2.79 + 2.27) against 1.68 ms for
    # this kernel at K = N = 128, while the whole module sat 1.2 ms behind cuequivariance.
    # KG == K reproduces the old kernel exactly: every loop below is single-trip in that case.
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
        for j in tl.static_range(0, N, BLOCK_N):
            rn = j + tl.arange(0, BLOCK_N)
            nmask = rn[None, :] < N
            wp = tl.load(wp_ptr + rk[:, None] * N + rn[None, :],
                         mask=kmask1[:, None] & nmask, other=0.0)
            proj = tl.dot(norm, wp)                              # (BLOCK_M1, BLOCK_N)
            # The gate contracts over KG, which is x_n's width and Wg's first axis. Its own loop,
            # because KG is not K on the bidirectional shape; at KG == K <= BLOCK_K it is one trip
            # over the same range the LN just used, which is the original single `tl.dot`.
            gacc = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
            for g0 in range(0, KG, BLOCK_K):
                rg = g0 + tl.arange(0, BLOCK_K)
                gmask1 = rg < KG
                gmask = gmask1[None, :]
                xn = tl.load(xn_ptr + rm[:, None] * KG + rg[None, :],
                             mask=mmask & gmask, other=0.0)
                wg = tl.load(wg_ptr + rg[:, None] * N + rn[None, :],
                             mask=gmask1[:, None] & nmask, other=0.0)
                gacc = tl.dot(xn, wg, gacc)
            gate = tl.sigmoid(gacc)                              # (BLOCK_M1, BLOCK_N)
            acc = proj * gate
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
                wp = tl.load(wp_ptr + rk[:, None] * N + rn[None, :],
                             mask=kmask1[:, None] & nmask, other=0.0)
                pacc = tl.dot(norm, wp, pacc)
            # The gate's contraction is KG, not K. Separate loop for the same reason as above; at
            # KG == K it visits the identical k-tiles the proj loop just did, in the same order,
            # so the accumulation order -- and the result -- is unchanged for the square case.
            for g0 in range(0, KG, BLOCK_K):
                rg = g0 + tl.arange(0, BLOCK_K)
                gmask1 = rg < KG
                gmask = gmask1[None, :]
                xn = tl.load(xn_ptr + rm[:, None] * KG + rg[None, :],
                             mask=mmask & gmask, other=0.0)
                wg = tl.load(wg_ptr + rg[:, None] * N + rn[None, :],
                             mask=gmask1[:, None] & nmask, other=0.0)
                gacc = tl.dot(xn, wg, gacc)
            proj = pacc                                              # (BLOCK_M1, BLOCK_N)
            gate = tl.sigmoid(gacc)                                  # (BLOCK_M1, BLOCK_N)
            acc = proj * gate
            # The pairformer residual add y = pair + trimul(pair): the residual is the module's
            # own (pre-LN) input, added in the same coalesced store. No dropout here (inference:
            # dropout is identity; training uses the v6 kernel).
            res = tl.load(res_ptr + rm[:, None] * N + rn[None, :], mask=mmask & nmask, other=0.0).to(tl.float32)
            acc = acc + res
            y = acc.to(y_ptr.dtype.element_ty)
            tl.store(y_ptr + rm[:, None] * N + rn[None, :], y, mask=mmask & nmask)


def _trimul_back_triton_fake(tri_bdll, x_n, Wp, Wg, ln_w, ln_b, eps, residual):
    """y [B, L, L, D]: the back half is shape-preserving, so it matches x_n's shape/dtype."""
    return x_n.new_empty(x_n.shape)


@opaque(fake=_trimul_back_triton_fake, name="trimul_back_fused")
def trimul_back_triton(tri_bdll: torch.Tensor, x_n: torch.Tensor, Wp: torch.Tensor,
                       Wg: torch.Tensor, ln_w: torch.Tensor, ln_b: torch.Tensor,
                       eps: float, residual: torch.Tensor) -> torch.Tensor:
    """tri_bdll:(B,D,L,L), x_n:(B,L,L,D), Wp/Wg:(D,D)=weight.T -> y:(B,L,L,D). B=1.

    ``residual`` ([B,L,L,D] == the module input pair) is REQUIRED: this op is
    ``y = pair + trimul(pair)``, with the add fused into the store epilogue.
    """
    # B==1 by design; B>1 works via a per-batch loop, which is faster than a batched single
    # launch (L2 thrashing of large bdll intermediates). See front.py's note and
    # notes/trimul_batch_generalization.
    B, K, L, L2 = tri_bdll.shape
    assert B == 1 and L == L2
    M = L * L
    N = x_n.shape[-1]

    # VALIDATE BEFORE LAUNCHING. The kernel indexes `wp_ptr`/`wg_ptr` as (K, N) and `xn_ptr` as
    # (M, K); it has no way to notice that the tensor it was handed is smaller. This launcher used
    # to take ONE `D = tri_bdll.shape[1]` and pass it as both K and N, so a caller whose to_out
    # maps d_hidden -> d_pair with d_hidden != d_pair handed in a (d_hidden, d_pair) weight while
    # the kernel read (d_pair, d_pair) -- an out-of-bounds READ of exactly the missing rows.
    #
    # That read does not reliably crash. It lands inside whatever the caching allocator has next,
    # so it usually returns another tensor's bytes and corrupts the result silently; it only traps
    # when it clears the mapping, which is why it showed up as an "illegal memory access" that
    # came and went with `torch.cuda.empty_cache()`. Reported by both the A100 and A6000 sessions,
    # surfacing at whatever kernel ran next (`layernorm_transpose` in one replay), never here.
    #
    # A ValueError is the right outcome, not a silent widen: this kernel folds LN(tri) and the
    # gate on x_n into ONE pass, so it needs tri's channel axis and x_n's width to be the same
    # axis. When they are not, there is no correct thing for it to compute.
    def _dims(t):
        return tuple(t.shape)
    # x_n's width is the GATE's contraction axis and it no longer has to equal K. It used to: the
    # kernel gated over the same axis it normalised, so the bidirectional trimul -- tri 2*d_hidden
    # wide, x_n d_pair -- was refused here and used a split back half instead.
    KG = x_n.shape[-1]
    if _dims(Wp) != (K, N):
        msg = (f"trimul_back_triton: Wp is {_dims(Wp)}, expected (K={K}, N={N}). "
               f"Pass the TRANSPOSED weight ((in, out)); a mismatch here is an "
               f"out-of-bounds read inside the kernel, not a shape error it can detect.")
        raise ValueError(msg)
    if _dims(Wg) != (KG, N):
        msg = (f"trimul_back_triton: Wg is {_dims(Wg)}, expected (KG={KG}, N={N}) -- KG is x_n's "
               f"width, which is what the gate contracts over. Pass the TRANSPOSED weight.")
        raise ValueError(msg)
    for name, v in (("ln_w", ln_w), ("ln_b", ln_b)):
        if v.numel() != K:
            msg = f"trimul_back_triton: {name} has {v.numel()} elements, expected K={K}"
            raise ValueError(msg)
    if residual.shape[-1] != N or residual.numel() != M * N:
        msg = (f"trimul_back_triton: residual is {_dims(residual)}, expected (..., {N}) "
               f"with {M * N} elements")
        raise ValueError(msg)

    tri_dm = tri_bdll.reshape(K, M)            # (K, M) contiguous, channel-major
    xn_flat = x_n.reshape(M, KG)
    y = torch.empty(M, N, device=x_n.device, dtype=x_n.dtype)
    res_flat = residual.reshape(M, N).contiguous()
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M1"]),)  # noqa: E731
    # KG is in the key: it is a constexpr the kernel tiles over, so two launches that differ only
    # in the gate's contraction width compile to different code and must not share a tuned config.
    _back_kernel[grid](tri_dm, xn_flat, Wp.contiguous(), Wg.contiguous(),
                       ln_w.contiguous(), ln_b.contiguous(), y, res_flat, M, float(eps),
                       K=K, N=N, KG=KG, shape_key=token_key(L, K=K, KG=KG))
    return y.view(B, L, L, N)
