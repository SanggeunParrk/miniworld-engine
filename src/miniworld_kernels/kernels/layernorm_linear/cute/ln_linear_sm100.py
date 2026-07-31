"""SM100 (B200) LayerNormLinear — CUEQUIV-FREE two-kernel milestone.

The H100 design (``gemm_layernorm_linear_fused.py``) forks quack's ``GemmSm90``
(WGMMA) and folds LayerNorm stats into the GEMM epilogue — it does NOT run on
Blackwell. This module is the sm100 port, as the explicitly-blessed two-kernel
milestone:

  ① LN_out over the K(=d_hidden) channel of the einsum output ``tri`` [B,K,L,L],
     read M-major (no transpose copy) by a custom Triton kernel, written as a
     contiguous (M, K) bf16 ``LNout``  —  ``_ln_mmajor_kernel``.
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

import torch
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

from miniworld_kernels.autotune import key_bucket_of, make_cache_prune, tensor_dtype_of
from miniworld_kernels.kernels.tm1.cute.sm100_gate_gemm_collective import (
    GatedPersistentGemmKernel,
)


# --------------------------------------------------------------------------- #
# ① LN over K, reading tri [B,K,L,L] M-major (channel strided by M=L*L), writing
#    a contiguous (M, K) output. No transpose copy (mirrors the H100 LNL, which
#    feeds an M-major A view straight into the GEMM). K is a power of two here so
#    BLOCK_K == K and there are no masked (variance-corrupting) columns.
# --------------------------------------------------------------------------- #
_layernorm_linear_cute_sm100_ln_mmajor_prune = make_cache_prune(
    "layernorm_linear_cute_sm100_ln_mmajor", dtype_of=tensor_dtype_of("X"),
    bucket_of=key_bucket_of("K"),
)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm}, num_warps=nw, num_stages=ns)
        for bm in (16, 32, 64)
        for nw in (4, 8)
        for ns in (2, 3, 4)
    ],
    key=["K"],
    prune_configs_by={"early_config_prune": _layernorm_linear_cute_sm100_ln_mmajor_prune},
)
@triton.jit
def _ln_mmajor_kernel(X, Y, W, B, M, K: tl.constexpr, eps,
                      BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    rm = row * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    mmask = rm < M
    # X[m,k] at m*1 + k*M  (M-major view of the [K, L, L] planes)
    xoff = rm[:, None] + rk[None, :] * M
    x = tl.load(X + xoff, mask=mmask[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / K
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / K
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + rk).to(tl.float32)
    b = tl.load(B + rk).to(tl.float32)
    y = xc * rstd[:, None] * w[None, :] + b[None, :]
    yoff = rm[:, None] * K + rk[None, :]
    tl.store(Y + yoff, y.to(Y.dtype.element_ty), mask=mmask[:, None])


def ln_out_mmajor(tri_bkll: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
                  eps: float) -> torch.Tensor:
    """LayerNorm over K of ``tri`` [B,K,L,L] (B=1), read M-major. Returns (M, K)
    bf16 contiguous (M = L*L)."""
    B, K, L, L2 = tri_bkll.shape
    assert B == 1 and L == L2
    M = L * L
    x = tri_bkll.reshape(K, M)  # X[m,k] = x[k, m] (M-major)
    Y = torch.empty(M, K, device=tri_bkll.device, dtype=tri_bkll.dtype)
    BLOCK_K = triton.next_power_of_2(K)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)  # noqa: E731
    _ln_mmajor_kernel[grid](x, Y, w, b, M, K=K, eps=float(eps), BLOCK_K=BLOCK_K)
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
    from miniworld_kernels.kernels._quack_compat import gemm as _quack_gemm

    return _quack_gemm(A, Wp.t())


def layernorm_linear_sm100(tri_bkll: torch.Tensor, ln_w: torch.Tensor,
                           ln_b: torch.Tensor, Wp_nn: torch.Tensor,
                           eps: float = 1e-5) -> torch.Tensor:
    """proj = LayerNorm_K(tri) @ Wp.T. tri:[B,K,L,L] (B=1), Wp_nn:(N,K)=to_out.weight
    (nn.Linear form). Returns (M=L*L, N) bf16."""
    lnout = ln_out_mmajor(tri_bkll, ln_w, ln_b, eps)   # (M, K)
    return proj_gemm_sm100(lnout, Wp_nn)               # (M, N)
