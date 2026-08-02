"""Fused LayerNorm + dual-projection + SwiGLU (the Transition "expand") on quack SM90.

The Transition front half is

    x = LayerNorm(x);  a = x @ Wa^T;  b = x @ Wb^T;  h = silu(a) * b      # (M, n*d)

We do NOT fork the GEMM mainloop. We compose two trusted quack pieces:

  * ``GemmGatedMixin`` — a single WGMMA GEMM whose B operand is the **stacked**
    [gate | up] weight (N_gated = 2*n*d), with a register-permute epilogue that pairs
    each gate column with its up column and applies a 2-arg gate ``act_fn(gate, up)``.
    quack already solved the cross-column pairing (``permute_gated_Cregs_b16``).

  * the Milestone-1 LayerNorm fold (see ../../layernorm_linear/cute): instead of
    materializing LayerNorm(x), feed raw ``x @ W2`` (W2 = gamma ⊙ W) and recover
    ``rstd[m]*acc - c1[m]*S[n] + B2[n]`` in the epilogue (rstd/c1 from a separate
    stats pass; S=Σ_k W2, B2=Σ_k beta*W + bias from the prologue).

``GemmLnGatedMixin`` = the gated mixin with the epilogue swapped to apply the LN-fold
affine to BOTH halves of the accumulator first, then gate. So one kernel emits the gated
``expand`` (M, n*d); the ``squeeze`` (n*d -> d) stays a plain ``torch.matmul`` (a clean
GEMM whose fusion would blow the accumulator register budget — see triton/fused.py).

B / S / B2 are PRE-INTERLEAVED here (row 2j = gate_j, 2j+1 = up_j) so the stock gated
pairing ``act_fn(rD[2i], rD[2i+1])`` lines gate up with up — no concat_layout needed.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import torch
from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

import quack.layout_utils as layout_utils
from quack.cute_dsl_utils import (
    mlir_namedtuple,
    torch2cute_dtype_map,
    get_device_capacity,
    get_max_active_clusters,
)
from quack.epi_ops import RowVecLoad, ColVecLoad
from quack.gemm_sm90 import GemmSm90
from quack.gemm_act import GemmGatedMixin
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.activation import gate_fn_map
from quack.rounding import RoundingMode
from quack.compile_utils import make_fake_tensor as fake_tensor
from miniworld_engine.kernels._quack_compat import jit_cache
from quack.gemm_config import GemmConfig
from quack.gemm_tvm_ffi_utils import (
    perm3d_single,
    get_major,
    make_scheduler_args,
    make_varlen_args,
    make_fake_scheduler_args,
    make_fake_varlen_args,
    div_for_dtype,
    make_fake_gemm_tensors,
    compile_gemm_kernel,
)


class GemmLnGatedMixin(GemmGatedMixin):
    """Epilogue: gate(rstd*acc - c1*S + B2) over the stacked [gate|up] accumulator.

    Adds ``mC1`` (per-m) and ``mB2`` (per-n) on top of the gated mixin's
    ``mPostAct`` / ``act_fn`` / ``mRowVecBroadcast`` (=S) / ``mColVecBroadcast`` (=rstd).
    """

    _epi_ops = (*GemmGatedMixin._epi_ops, ColVecLoad("mC1"), RowVecLoad("mB2"))

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor
        act_fn: cutlass.Constexpr[Optional[object]] = None
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None  # S[n]   (stacked 2N, interleaved)
        mColVecBroadcast: Optional[cute.Tensor] = None  # rstd[m]
        mC1: Optional[cute.Tensor] = None               # c1[m] = mean*rstd
        mB2: Optional[cute.Tensor] = None               # B2[n]  (stacked 2N, interleaved)
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        # Mirror GemmGatedMixin: set up the gated postact tile (N//2) + dtype/layout.
        assert args.mPostAct.element_type.width == 16, "gated postact must be 16-bit"
        assert cutlass.utils.LayoutEnum.from_tensor(args.mPostAct).is_n_major_c()
        if self.arch == 90:
            assert self.cta_tile_shape_mnk[1] % 32 == 0, "gated SM90 needs tileN % 32 == 0"
        self.rounding_mode = args.rounding_mode
        self.postact_dtype = args.mPostAct.element_type
        self.postact_layout = cutlass.utils.LayoutEnum.from_tensor(args.mPostAct)
        self.cta_tile_shape_postact_mn = (
            self.cta_tile_shape_mnk[0],
            self.cta_tile_shape_mnk[1] // 2,
        )
        d = self._epi_ops_to_params_dict(args)
        d["act_fn"] = args.act_fn
        return self.EpilogueParams(**d)

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        rstd = epi_loop_tensors["mColVecBroadcast"]
        c1 = epi_loop_tensors["mC1"]
        S = epi_loop_tensors["mRowVecBroadcast"]
        B2 = epi_loop_tensors["mB2"]
        # LN-fold affine on the full stacked accumulator: acc' = rstd*acc - c1*S + B2.
        for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
            tRS_rD[i] = rstd[i] * tRS_rD[i] - c1[i] * S[i] + B2[i]
        # Gated activation: pair gate (even) with up (odd) -> silu(gate)*up. (SM90 path.)
        tRS_rPostAct_layout = cute.recast_layout(2, 1, tRS_rD.layout)
        tRS_rPostAct = cute.make_rmem_tensor(tRS_rPostAct_layout.shape, self.acc_dtype)
        for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
            tRS_rPostAct[i] = params.act_fn(tRS_rD[2 * i], tRS_rD[2 * i + 1])
        return tRS_rPostAct

    # epi_convert_postact (which calls permute_gated_Cregs_b16 on SM90) is inherited.


class GemmLnGatedSm90(GemmLnGatedMixin, GemmSm90):
    pass


@jit_cache
def _compile_gemm_ln_swiglu(
    a_dtype, b_dtype, postact_dtype,
    a_major, b_major, postact_major,
    vec_dtype,
    tile_shape_mn, cluster_shape_mnk,
    pingpong, is_dynamic_persistent, device_capacity,
    act_fn=None,
):
    if act_fn is None:
        act_fn = gate_fn_map["swiglu"]
    GemmCls = GemmLnGatedSm90
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype, b_dtype, None, None, a_major, b_major, None, None
    )
    pa_n = cute.sym_int()  # gated postact width = n // 2
    div_pa = div_for_dtype(postact_dtype)
    mPostAct = fake_tensor(postact_dtype, (m, pa_n, l), leading_dim=1, divisibility=div_pa)
    mRowVec = fake_tensor(vec_dtype, (l, n), leading_dim=1, divisibility=4)  # S (2N)
    mB2 = fake_tensor(vec_dtype, (l, n), leading_dim=1, divisibility=4)
    mColVec = fake_tensor(vec_dtype, (l, m), leading_dim=1, divisibility=4)  # rstd
    mC1 = fake_tensor(vec_dtype, (l, m), leading_dim=1, divisibility=4)
    epi_args = GemmCls.EpilogueArguments(
        mPostAct,
        act_fn,
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
        GemmCls, a_dtype, tile_shape_mn, cluster_shape_mnk,
        pingpong, True, False, is_dynamic_persistent, device_capacity,
        mA, mB, mD, mC, epi_args, scheduler_args, varlen_args,
    )


def gemm_ln_swiglu(
    A: Tensor,        # (M, K) bf16, k-major  = x
    B: Tensor,        # (2N, K) bf16, k-major = interleaved [gate|up] W2 = gamma ⊙ [Wa,Wb]
    PostAct: Tensor,  # (M, N) bf16, n-major  = gated expand output
    rstd: Tensor,     # (1, M) fp32
    c1: Tensor,       # (1, M) fp32
    S: Tensor,        # (1, 2N) fp32  interleaved
    B2: Tensor,       # (1, 2N) fp32  interleaved
    *,
    config: GemmConfig | None = None,
    act_fn=None,      # default swiglu; override to isolate activation cost
) -> None:
    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] == 9, "SM90 (H100) only"
    if config is None:
        # d-aware best from the K-sweep (tile_n caps at 128 under pingpong; gated needs %32):
        #   K=128  -> 256x128 NON-pingpong cluster(1,2)  (~1.1x vs triton at large M)
        #   K>=256 -> 192x128 pingpong     cluster(1,2)  (1.6x at K=256, 2.6x at K=512)
        # cute beats the tuned triton expand everywhere and the win GROWS with K, because
        # triton's BLOCK_K=next_pow2(K) full-row load scales badly with K while WGMMA stays
        # near FLOP-linear. K = A.shape[-1].
        K = A.shape[-1]
        if K <= 128:
            config = GemmConfig(
                tile_m=256, tile_n=128, pingpong=False, is_dynamic_persistent=False,
                cluster_m=1, cluster_n=2, swap_ab=False, max_swizzle_size=8, device_capacity=9,
            )
        else:
            config = GemmConfig(
                tile_m=192, tile_n=128, pingpong=True, is_dynamic_persistent=False,
                cluster_m=1, cluster_n=2, swap_ab=False, max_swizzle_size=8, device_capacity=9,
            )

    A3 = A.unsqueeze(0) if A.dim() == 2 else A
    B3 = B.unsqueeze(0) if B.dim() == 2 else B
    PA3 = PostAct.unsqueeze(0) if PostAct.dim() == 2 else PostAct
    A_p = perm3d_single(A3)
    B_p = perm3d_single(B3)
    PA_p = perm3d_single(PA3)
    a_major = get_major(A_p, "m", "k")
    b_major = get_major(B_p, "n", "k")
    postact_major = get_major(PA_p, "m", "n")
    a_dtype = torch2cute_dtype_map[A.dtype]
    b_dtype = torch2cute_dtype_map[B.dtype]
    postact_dtype = torch2cute_dtype_map[PostAct.dtype]
    vec_dtype = torch2cute_dtype_map[rstd.dtype]
    is_dynamic_persistent = config.is_dynamic_persistent

    compiled_fn = _compile_gemm_ln_swiglu(
        a_dtype, b_dtype, postact_dtype,
        a_major, b_major, postact_major,
        vec_dtype,
        (config.tile_m, config.tile_n),
        (config.cluster_m, config.cluster_n, 1),
        config.pingpong, is_dynamic_persistent, device_capacity,
        act_fn,
    )

    from miniworld_engine.kernels._quack_compat import is_compile_only

    if is_compile_only():
        return

    max_active_clusters = get_max_active_clusters(config.cluster_m * config.cluster_n)
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=A.device)
        if (is_dynamic_persistent and device_capacity[0] == 9)
        else None
    )
    epi_args = GemmLnGatedMixin.EpilogueArguments(
        PA_p,
        None,            # act_fn is Constexpr, baked at compile
        mRowVecBroadcast=S,
        mColVecBroadcast=rstd,
        mC1=c1,
        mB2=B2,
        rounding_mode=None,
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters, config.max_swizzle_size, tile_count_semaphore
    )
    varlen_args = make_varlen_args(None, None, None)
    compiled_fn(A_p, B_p, None, None, epi_args, scheduler_args, varlen_args, None)


# ---------------------------------------------------------------------------
# Prologue (fold + interleave) — cache for fixed weights.
# ---------------------------------------------------------------------------


def fold_swiglu(
    Wa: Tensor,         # (N, K) = expand_a.weight,  N = n*d
    Wb: Tensor,         # (N, K) = expand_b.weight
    ln_weight: Tensor,  # (K,) gamma
    ln_bias: Tensor,    # (K,) beta
    *,
    w2_dtype: torch.dtype = torch.bfloat16,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build the interleaved gated GEMM operands.

    B[2j]   = gamma ⊙ Wa[j],   B[2j+1] = gamma ⊙ Wb[j]    -> (2N, K)
    S[2j]   = Σ_k B_gate[j,k],  B2[2j] = Σ_k beta*Wa[j,k]  (and the up half at 2j+1)
    """
    g = ln_weight.float()
    Wa2 = (Wa.float() * g[None, :]).to(w2_dtype)
    Wb2 = (Wb.float() * g[None, :]).to(w2_dtype)
    N, K = Wa.shape
    B = torch.empty(2 * N, K, dtype=w2_dtype, device=Wa.device)
    B[0::2] = Wa2
    B[1::2] = Wb2
    Sa = Wa2.float().sum(dim=1)
    Sb = Wb2.float().sum(dim=1)
    B2a = Wa.float() @ ln_bias.float()
    B2b = Wb.float() @ ln_bias.float()
    S = torch.empty(2 * N, dtype=torch.float32, device=Wa.device)
    B2 = torch.empty(2 * N, dtype=torch.float32, device=Wa.device)
    S[0::2], S[1::2] = Sa, Sb
    B2[0::2], B2[1::2] = B2a, B2b
    return B.contiguous(), S.contiguous(), B2.contiguous()


def transition_expand_swiglu_cute(
    x2: Tensor,         # (M, K) bf16, contiguous
    ln_weight: Tensor,  # (K,)
    ln_bias: Tensor,    # (K,)
    Wa: Tensor,         # (N, K)
    Wb: Tensor,         # (N, K)
    eps: float,
    *,
    prefolded: tuple[Tensor, Tensor, Tensor] | None = None,
    stats: tuple[Tensor, Tensor] | None = None,
    config: GemmConfig | None = None,
) -> Tensor:
    """LayerNorm(x) -> SwiGLU(x@Wa, x@Wb) -> expand (M, N). Stats via the triton pass."""
    from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton

    assert x2.is_cuda and x2.dim() == 2
    M, K = x2.shape
    N = Wa.shape[0]
    if prefolded is not None:
        B, S, B2 = prefolded
    else:
        # Fused single-kernel fold (~28us) instead of the launch-bound torch fold (~141us).
        # Weights change each optimizer step in training, so this fold can't be cached across
        # steps; inference can still pass prefolded= to skip it. See transition-b2b-forward-verdict.
        from miniworld_engine.kernels.transition.triton.fold import fold_swiglu_triton

        B, S, B2 = fold_swiglu_triton(Wa, Wb, ln_weight, ln_bias, w2_dtype=x2.dtype)
    rstd, c1 = stats if stats is not None else stats_triton(x2, eps)
    rstd2 = rstd.contiguous().view(1, M)
    c12 = c1.contiguous().view(1, M)
    S2 = S.contiguous().view(1, 2 * N)
    B22 = B2.contiguous().view(1, 2 * N)
    expand = torch.empty(M, N, device=x2.device, dtype=x2.dtype)
    gemm_ln_swiglu(x2, B, expand, rstd2, c12, S2, B22, config=config)
    return expand
