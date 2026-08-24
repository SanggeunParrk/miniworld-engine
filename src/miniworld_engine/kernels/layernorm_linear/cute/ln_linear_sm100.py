"""SM100 (B200) LayerNormLinear — CUEQUIV-FREE two-kernel milestone.

The H100 design (``gemm_layernorm_linear_fused.py``) forks quack's ``GemmSm90``
(WGMMA) and folds LayerNorm stats into the GEMM epilogue — it does NOT run on
Blackwell. This module is the sm100 port, as the explicitly-blessed two-kernel
milestone:

  ① LN_out over the K(=d_hidden) channel of the einsum output ``tri`` [B,K,L,L],
     read M-major (no transpose copy) by a custom Triton kernel, written as a
     contiguous (M, K) bf16 ``LNout``  —  ``_ln_transpose_dbn_kernel``, imported from
     ``layernorm/triton/transpose.py`` (the copy that used to live here was the same
     program and bitwise equal on Y).
  ② proj = LNout @ Wp.T   on OUR tm1 tcgen05 CUTLASS Blackwell collective
     (``GatedPersistentGemmKernel``) in ``proj_only`` mode (single effective B;
     the dummy second-B TMA load is negligible since B=(N,K) is tiny).

Math is the standard LayerNorm-then-Linear (bf16 in / fp32 acc / bf16 out); the
LN reduction is fp32. No cuequiv, no quack. B=1.

A later round can fuse ① into ② (raw X@W2 with an fp32 rstd/S/B2 correction
epilogue on the collective — the H100 fold ported to tcgen05) to drop the LNout
materialize round-trip; this two-kernel version already beats cuequiv on B200.
"""

from __future__ import annotations
# This file used to carry its own copy of the m-major LayerNorm forward. It was the same
# program as layernorm/triton/transpose.py's (identical node and line counts) and bitwise
# equal on Y when both were handed the same arguments (.bench/eq_*.out), so it is imported
# now. Note it is NOT interchangeable with the row-major forward: feeding it row-major input
# gives rel=1.442, because reading the M-major view is part of what the kernel means.
from miniworld_engine.kernels.layernorm.triton.transpose import _ln_transpose_dbn_kernel

import torch

from miniworld_engine.kernels._compile import opaque
import triton
import triton.language as tl


import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutils
from cutlass import Float32
from cutlass.cute.runtime import from_dlpack
# Memoized occupancy query (raw HardwareInfo probe JIT-recompiles every call). Same int
# returned -> identical launch/numerics; removes the per-call eager compile overhead.
from quack.cute_dsl_utils import get_max_active_clusters

from miniworld_engine.kernels.tm1.cute.sm100_gate_gemm_collective import (
    GatedPersistentGemmKernel,
)


# --------------------------------------------------------------------------- #
# ① LN over K, reading tri [B,K,L,L] M-major (channel strided by M=L*L), writing
#    a contiguous (M, K) output. No transpose copy (mirrors the H100 LNL, which
#    feeds an M-major A view straight into the GEMM). The K axis is now a tuned,
#    MASKED tile (BLOCK_K from the sweep), so K need not be a power of two and the
#    tail columns contribute nothing to the variance.
# --------------------------------------------------------------------------- #


# BLOCK_K is the REDUCE axis (mean/var over K), so it is a CSV tile rather than the narrow
# canonical BLOCK_K: it used to arrive as BLOCK_K=next_power_of_2(K) from the launcher and K is
# d_hidden (128..1024), so a set that stopped at 128 would force a multi-pass over K on every
# shape. A row at or above the extent keeps the whole-row single-pass schedule.
from miniworld_engine.autotune.buckets import bucket_mixed as _bucket
from miniworld_engine.autotune.shape_key import both_key


def get_seq_group(rows) -> int:
    """Delegates to canonical size-bucketing (autotune.buckets)."""
    return _bucket(rows)


@opaque(fake=lambda tri_bkll, w, b, eps: tri_bkll.new_empty(
            (tri_bkll.shape[2] * tri_bkll.shape[3], tri_bkll.shape[1])),
        name="ln_out_mmajor_sm100")
def ln_out_mmajor(tri_bkll: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
                  eps: float) -> torch.Tensor:
    """LayerNorm over K of ``tri`` [B,K,L,L] (B=1), read M-major. Returns (M, K)
    bf16 contiguous (M = L*L)."""
    B, K, L, L2 = tri_bkll.shape
    assert B == 1 and L == L2
    M = L * L
    x = tri_bkll.reshape(K, M)  # X[m,k] = x[k, m] (M-major)
    Y = torch.empty(M, K, device=tri_bkll.device, dtype=tri_bkll.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M1"]),)  # noqa: E731
    # M = L*L, the row count this launch iterates. It used to pass L, from when the key was a
    # length; a `level=both` kernel keys on rows now (see BOTH_ROWS) precisely so that a pair
    # L=1024 and an atom A=1024 stop sharing a bucket. (The kernel parameter is `shape_key`;
    # `GROUP_M` was a stale name from before the rename.)
    _ln_transpose_dbn_kernel[grid](x, Y, w, b, M, float(eps), D=K, shape_key=both_key(M))
    return Y


# --------------------------------------------------------------------------- #
# ② proj = A @ Wp.T on the tm1 tcgen05 collective, proj_only (single-B GEMM).
# --------------------------------------------------------------------------- #
_PROJ_CACHE: dict = {}


def proj_gemm_sm100(A: torch.Tensor, Wp: torch.Tensor) -> torch.Tensor:
    """proj = A @ Wp.T.  A:(M,K) bf16, Wp:(N,K) bf16 (== nn.Linear weight) -> (M,N) bf16.

    cutlass-dsl 4.5.2 migration: the previous path drove our vendored tm1 Blackwell
    collective (``GatedPersistentGemmKernel`` proj_only), whose 4.4.2-era launch is
    incompatible with the 4.5.2 launch ABI (host crash in cuLaunchKernelEx). A plain
    projection is exactly quack's maintained, 4.5.2-native GEMM, so we call that directly
    (``gemm(A, B)`` computes ``A @ B`` with ``B`` shaped ``(K, N)``, hence ``Wp.t()``).
    Reached lazily via the quack-compat shim (applies the RoundingMode fix)."""
    from miniworld_engine.kernels._quack_compat import gemm as _quack_gemm

    return _quack_gemm(A, Wp.t())


def layernorm_linear_sm100(tri_bkll: torch.Tensor, ln_w: torch.Tensor,
                           ln_b: torch.Tensor, Wp_nn: torch.Tensor,
                           eps: float = 1e-5) -> torch.Tensor:
    """proj = LayerNorm_K(tri) @ Wp.T. tri:[B,K,L,L] (B=1), Wp_nn:(N,K)=to_out.weight
    (nn.Linear form). Returns (M=L*L, N) bf16."""
    lnout = ln_out_mmajor(tri_bkll, ln_w, ln_b, eps)   # (M, K)
    return proj_gemm_sm100(lnout, Wp_nn)               # (M, N)
