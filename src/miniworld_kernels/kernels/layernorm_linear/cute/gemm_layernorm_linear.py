"""Fused LayerNormLinear forward on quack's SM90 BF16 GEMM.

We do NOT write a GEMM. We start from quack's trusted warp-specialized SM90 GEMM
(`GemmSm90` + the composable `GemmDefaultEpiMixin`) and only swap the epilogue.

Folded formulation (see ../reference.py / README):

    acc[m,n] = X @ W2          (W2 = gamma ⊙ W, fed as the GEMM B operand)
    Y[m,n]   = rstd[m]*acc[m,n] - c1[m]*S[n] + B2[n]

where, per row m, ``rstd[m]`` and ``c1[m]=mean[m]*rstd[m]`` come from a separate
stats pass over X; per col n, ``S[n]=sum_k W2[k,n]`` and ``B2[n]=sum_k beta*W+bias``
come from the prologue. Raw X@W2 means LayerNorm(X) [M,K] is never materialized.

The epilogue reuses the default ``mColVecBroadcast`` (=rstd, per-m) and
``mRowVecBroadcast`` (=S, per-n) broadcast loaders and adds two more —
``mC1`` (per-m) and ``mB2`` (per-n). Every broadcast fragment is aligned to the
accumulator fragment index ``i``, so the epilogue is one fused line.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr

from quack.cute_dsl_utils import (
    mlir_namedtuple,
    torch2cute_dtype_map,
    get_device_capacity,
    get_max_active_clusters,
)
from quack.epi_ops import RowVecLoad, ColVecLoad
from quack.gemm_sm90 import GemmSm90
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.rounding import RoundingMode
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cache_utils import jit_cache
from quack.gemm_interface import default_config
from quack.gemm_tvm_ffi_utils import (
    get_majors,
    get_dtypes,
    perm3d,
    make_scheduler_args,
    make_varlen_args,
    make_fake_scheduler_args,
    make_fake_varlen_args,
    make_fake_gemm_tensors,
    compile_gemm_kernel,
)


class GemmLayerNormLinearMixin(GemmDefaultEpiMixin):
    """Epilogue: Y = rstd[m]*acc - c1[m]*S[n] + B2[n]."""

    _epi_ops = (*GemmDefaultEpiMixin._epi_ops, ColVecLoad("mC1"), RowVecLoad("mB2"))

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None  # S[n]
        mColVecBroadcast: Optional[cute.Tensor] = None  # rstd[m]
        mC1: Optional[cute.Tensor] = None               # c1[m] = mean[m]*rstd[m]
        mB2: Optional[cute.Tensor] = None               # B2[n]
        add_to_output: cutlass.Constexpr[bool] = False
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[cute.Tensor] = None

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = args.rounding_mode
        d = self._epi_ops_to_params_dict(args)
        return self.EpilogueParams(**d)

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        rstd = epi_loop_tensors["mColVecBroadcast"]
        c1 = epi_loop_tensors["mC1"]
        S = epi_loop_tensors["mRowVecBroadcast"]
        B2 = epi_loop_tensors["mB2"]
        # Y[i] = rstd[i]*acc[i] - c1[i]*S[i] + B2[i]; all fragments broadcast-aligned to i.
        for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
            tRS_rD[i] = rstd[i] * tRS_rD[i] - c1[i] * S[i] + B2[i]
        return None


class GemmLayerNormLinearSm90(GemmLayerNormLinearMixin, GemmSm90):
    pass


@jit_cache
def _compile_gemm_lnl(
    a_dtype,
    b_dtype,
    d_dtype,
    a_major,
    b_major,
    d_major,
    vec_dtype,
    tile_shape_mn,
    cluster_shape_mnk,
    pingpong,
    persistent,
    is_dynamic_persistent,
    device_capacity,
):
    GemmCls = GemmLayerNormLinearSm90
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype, b_dtype, d_dtype, None, a_major, b_major, d_major, None
    )
    mRowVec = fake_tensor(vec_dtype, (l, n), leading_dim=1, divisibility=4)  # S
    mB2 = fake_tensor(vec_dtype, (l, n), leading_dim=1, divisibility=4)
    mColVec = fake_tensor(vec_dtype, (l, m), leading_dim=1, divisibility=4)  # rstd
    mC1 = fake_tensor(vec_dtype, (l, m), leading_dim=1, divisibility=4)
    epi_args = GemmCls.EpilogueArguments(
        mRowVecBroadcast=mRowVec,
        mColVecBroadcast=mColVec,
        mC1=mC1,
        mB2=mB2,
    )
    scheduler_args = make_fake_scheduler_args(
        (is_dynamic_persistent and device_capacity[0] == 9), False, l
    )
    varlen_args = make_fake_varlen_args(False, False, False, None)
    return compile_gemm_kernel(
        GemmCls,
        a_dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        False,
        is_dynamic_persistent,
        device_capacity,
        mA,
        mB,
        mD,
        mC,
        epi_args,
        scheduler_args,
        varlen_args,
    )


def gemm_layernorm_linear(
    A: Tensor,     # (M, K) bf16, k-major
    B: Tensor,     # (N, K) bf16, k-major — W2 = gamma ⊙ W in (N,K) layout
    D: Tensor,     # (M, N) bf16, n-major — output Y
    rstd: Tensor,  # (1, M) fp32
    c1: Tensor,    # (1, M) fp32  = mean*rstd
    S: Tensor,     # (1, N) fp32
    B2: Tensor,    # (1, N) fp32
    *,
    config=None,
) -> None:
    """Run the fused GEMM+LayerNorm epilogue, writing Y into D."""
    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] == 9, "first version targets SM90 (H100)"
    if config is None:
        config = default_config(A.device)

    # quack's low-level GEMM expects 3D (l, m, k) / (l, n, k) / (l, m, n).
    A3 = A.unsqueeze(0) if A.dim() == 2 else A
    B3 = B.unsqueeze(0) if B.dim() == 2 else B
    D3 = D.unsqueeze(0) if D.dim() == 2 else D
    A_p, B_p, D_p, _ = perm3d(A3, B3, D3, None)
    a_major, b_major, d_major, _ = get_majors(A_p, B_p, D_p, None)
    a_dtype, b_dtype, d_dtype, _ = get_dtypes(A, B, D, None)
    vec_dtype = torch2cute_dtype_map[rstd.dtype]
    is_dynamic_persistent = config.is_dynamic_persistent

    compiled_fn = _compile_gemm_lnl(
        a_dtype,
        b_dtype,
        d_dtype,
        a_major,
        b_major,
        d_major,
        vec_dtype,
        (config.tile_m, config.tile_n),
        (config.cluster_m, config.cluster_n, 1),
        config.pingpong,
        True,
        is_dynamic_persistent,
        device_capacity,
    )

    from quack.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY:
        return

    max_active_clusters = get_max_active_clusters(config.cluster_m * config.cluster_n)
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=A.device)
        if (is_dynamic_persistent and device_capacity[0] == 9)
        else None
    )
    epi_args = GemmLayerNormLinearMixin.EpilogueArguments(
        mRowVecBroadcast=S,
        mColVecBroadcast=rstd,
        mC1=c1,
        mB2=B2,
        add_to_output=None,   # Constexpr, pass None at runtime
        rounding_mode=None,   # Constexpr, pass None at runtime
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters, config.max_swizzle_size, tile_count_semaphore
    )
    varlen_args = make_varlen_args(None, None, None)
    compiled_fn(A_p, B_p, D_p, None, epi_args, scheduler_args, varlen_args, None)


# ---------------------------------------------------------------------------
# Prologue (fold) + stats + GEMM — the user-facing entry.
# ---------------------------------------------------------------------------


def fold_for_gemm(
    weight: Tensor,     # (N, K)  nn.Linear layout
    ln_weight: Tensor,  # (K,) gamma
    ln_bias: Tensor,    # (K,) beta
    bias: Tensor | None,  # (N,)
    *,
    w2_dtype: torch.dtype = torch.bfloat16,
) -> tuple[Tensor, Tensor, Tensor]:
    """Prologue: B=(N,K) GEMM operand, S=(N,), B2=(N,). Cache for fixed weights.

    B[n,k] = gamma[k]*weight[n,k]   (= W2 in (N,K) layout, the GEMM B operand)
    S[n]   = sum_k B[n,k]           (FP32, reduced from the *stored* bf16 B)
    B2[n]  = sum_k beta[k]*weight[n,k] + bias[n]   (FP32)
    """
    Bw = (weight.float() * ln_weight.float()[None, :]).to(w2_dtype).contiguous()
    S = Bw.float().sum(dim=1)                       # from stored B (cancellation-consistent)
    B2 = weight.float() @ ln_bias.float()
    if bias is not None:
        B2 = B2 + bias.float()
    return Bw, S.contiguous(), B2.contiguous()


# `_stats` recompiles per (M, K) shape; allow enough cache entries for a sweep.
# (A hand-written Triton stats kernel — no recompiles — is the next perf step.)
torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)


@torch.compile(fullgraph=True, dynamic=False)
def _stats(x: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    """rstd[m], c1[m]=mean*rstd — single fused reduction pass over X (Milestone 1).

    `torch.compile` fuses this into one kernel that reads X (bf16) once; an eager
    `x.float().mean()`/`(x*x).mean()` instead materializes an FP32 [M,K] copy and
    runs several passes, which dominates runtime at large M. A hand-written
    Triton stats kernel (or folding stats into the GEMM mainloop, M2) is next.
    """
    xf = x.float()
    mean = xf.mean(dim=1)
    var = (xf * xf).mean(dim=1) - mean * mean
    rstd = torch.rsqrt(var + eps)
    return rstd, mean * rstd


def layernorm_linear_cute(
    x: Tensor,          # (M, K) bf16
    ln_weight: Tensor,  # (K,)
    ln_bias: Tensor,    # (K,)
    weight: Tensor,     # (N, K)
    bias: Tensor | None,  # (N,)
    eps: float = 1e-5,
    *,
    prefolded: tuple[Tensor, Tensor, Tensor] | None = None,
) -> Tensor:
    """Fused forward LayerNormLinear (Milestone 1: fused stats + fused GEMM epilogue)."""
    assert x.is_cuda and x.dim() == 2
    M, K = x.shape
    N = weight.shape[0]
    Bw, S, B2 = prefolded if prefolded is not None else fold_for_gemm(
        weight, ln_weight, ln_bias, bias, w2_dtype=x.dtype
    )

    rstd, c1 = _stats(x, eps)
    rstd2 = rstd.contiguous().view(1, M)
    c12 = c1.contiguous().view(1, M)
    S2 = S.float().contiguous().view(1, N)
    B22 = B2.float().contiguous().view(1, N)
    Y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    gemm_layernorm_linear(x, Bw, Y, rstd2, c12, S2, B22)
    return Y
