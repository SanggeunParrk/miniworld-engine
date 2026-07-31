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

import torch
import triton
import triton.language as tl

from miniworld_kernels.autotune import key_bucket_of, make_cache_prune, tensor_dtype_of
from miniworld_kernels.kernels.trimul_inproj.triton._autotune import get_seq_group


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
#   - BM=64, BN=64, num_warps=4 is the winner for every L.
# Kept a tiny pruned set around the winner plus safe fallbacks.
_trimul_back_prune = make_cache_prune(
    "trimul_back", dtype_of=tensor_dtype_of("tri_ptr"),
    bucket_of=key_bucket_of("GROUP_M", "K", "N", "ADD_RESIDUAL"),
)


@triton.autotune(
    configs=[
        triton.Config({"BM": 64, "BN": 64}, num_warps=4, num_stages=2),
        triton.Config({"BM": 64, "BN": 64}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64, "BN": 32}, num_warps=4, num_stages=2),
        triton.Config({"BM": 64, "BN": 64}, num_warps=8, num_stages=2),
    ],
    key=["GROUP_M", "K", "N", "ADD_RESIDUAL"],
    prune_configs_by={"early_config_prune": _trimul_back_prune},
)
@triton.jit
def _back_kernel(
    tri_ptr,  # (D, M) channel-major: tri[k, m] at k*M + m
    xn_ptr,   # (M, D) row-major
    wp_ptr, wg_ptr,    # (D, D) = to_out.weight.T, to_gate.weight.T  (K=in, N=out)
    lnw_ptr, lnb_ptr,  # (D,)
    y_ptr,    # (M, D) row-major
    res_ptr,  # (M, D) row-major residual (== the module input pair); read iff ADD_RESIDUAL
    M, eps,
    K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    GROUP_M: tl.constexpr, ADD_RESIDUAL: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    rm = pid * BM + tl.arange(0, BM)
    rk = tl.arange(0, K)
    mmask = rm[:, None] < M

    # tri (BM, K) — channel-major load (k strided by M)
    tri = tl.load(tri_ptr + rk[None, :] * M + rm[:, None], mask=mmask, other=0.0).to(tl.float32)
    mean = tl.sum(tri, axis=1) / K
    xc = tri - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / K
    rstd = 1.0 / tl.sqrt(var + eps)
    lnw = tl.load(lnw_ptr + rk).to(tl.float32)
    lnb = tl.load(lnb_ptr + rk).to(tl.float32)
    # LN-normed row tile, cast to bf16 once (reused across all N-subtiles).
    norm = ((xc * rstd[:, None]) * lnw[None, :] + lnb[None, :]).to(tl.bfloat16)  # (BM, K)
    xn = tl.load(xn_ptr + rm[:, None] * K + rk[None, :], mask=mmask, other=0.0)  # (BM, K)

    # Tile the output dim N: keep only a (BM, BN) accumulator pair live at a time.
    for j in tl.static_range(0, N, BN):
        rn = j + tl.arange(0, BN)
        wp = tl.load(wp_ptr + rk[:, None] * N + rn[None, :])
        wg = tl.load(wg_ptr + rk[:, None] * N + rn[None, :])
        proj = tl.dot(norm, wp)                                  # (BM, BN)
        gate = tl.sigmoid(tl.dot(xn, wg))                        # (BM, BN)
        acc = proj * gate
        if ADD_RESIDUAL:
            # Fuse the pairformer residual add y = pair + trimul(pair): the residual is the
            # module's own (pre-LN) input, added in the same coalesced store. No dropout here
            # (inference: dropout is identity; training uses the v6 kernel).
            res = tl.load(res_ptr + rm[:, None] * N + rn[None, :], mask=mmask, other=0.0).to(tl.float32)
            acc = acc + res
        y = acc.to(y_ptr.dtype.element_ty)
        tl.store(y_ptr + rm[:, None] * N + rn[None, :], y, mask=mmask)


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
    grid = lambda meta: (triton.cdiv(M, meta["BM"]),)  # noqa: E731
    _back_kernel[grid](tri_dm, xn_flat, Wp.contiguous(), Wg.contiguous(),
                       ln_w.contiguous(), ln_b.contiguous(), y, res_flat, M, float(eps),
                       K=D, N=D, GROUP_M=get_seq_group(M), ADD_RESIDUAL=add_residual)
    return y.view(B, L, L, D)
