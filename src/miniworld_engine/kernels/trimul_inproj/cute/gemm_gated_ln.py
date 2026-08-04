"""LN-folded gated GEMM that writes BDLL directly — the trimul front, optimized.

Composes:
  * transition's ``GemmLnGatedMixin`` (GemmGated + M1 LN-fold epilogue:
    ``glu(rstd*acc - c1*S + B2)`` over the stacked [gate|up] accumulator), and
  * the bdll (M-major postact) policy (drop the n-major-c assert + leading_dim
    follows the M-major layout) — so left/right land in ``[B, 2D, L, L]`` with NO
    separate LN kernel and NO ``x_n`` materialization. Only stats (rstd, c1) are
    produced (cheap, and exactly what the backward needs).

Front forward becomes:  stats(x) -> this kernel(raw x, folded W2) -> left,right (bdll).
B = (4D, D) k-major = interleaved [gate|up] folded weights; S,B2 = (4D,) fold vecs.
B=1, D=128, SM90.
"""

from __future__ import annotations

import torch
import cutlass
import cutlass.cute as cute
from cutlass import const_expr  # noqa: F401

from quack.activation import gate_fn_map
from miniworld_engine.kernels._quack_compat import jit_cache, is_compile_only
from quack.cute_dsl_utils import (
    torch2cute_dtype_map, get_device_capacity, get_max_active_clusters,
)
from quack.gemm_sm90 import GemmSm90
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.gemm_tvm_ffi_utils import (
    perm3d_single, get_major, make_scheduler_args, make_varlen_args,
    make_fake_scheduler_args, make_fake_varlen_args, div_for_dtype,
    make_fake_gemm_tensors, compile_gemm_kernel,
)

from miniworld_engine.kernels.transition.cute.gemm_transition_swiglu import GemmLnGatedMixin
from miniworld_engine.kernels.layernorm_linear.cute.gemm_layernorm_linear import fold_for_gemm
from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton


class _BdllLnGatedMixin(GemmLnGatedMixin):
    """GemmLnGatedMixin, but accept an M-major (bdll) gated postact (no n-major assert)."""

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        assert args.mAuxOut.element_type.width == 16, "gated postact must be 16-bit"
        # (n-major assert dropped — bdll postact is M-major)
        if self.arch == 90:
            assert self.cta_tile_shape_mnk[1] % 32 == 0, "gated SM90 needs tileN % 32 == 0"
        self.rounding_mode = args.rounding_mode
        self.aux_out_dtype = args.mAuxOut.element_type
        self.aux_out_layout = cutlass.utils.LayoutEnum.from_tensor(args.mAuxOut)
        self.cta_tile_shape_aux_out_mn = (
            self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1] // 2,
        )
        d = self._epi_ops_to_params_dict(args)
        d["act_fn"] = args.act_fn
        return self.EpilogueParams(**d)


class GemmGatedLNBdllSm90(_BdllLnGatedMixin, GemmSm90):
    pass


@jit_cache
def _compile_gated_ln(a_dtype, b_dtype, postact_dtype, a_major, b_major, postact_major,
                      vec_dtype, tile_shape_mn, cluster_shape_mnk, pingpong,
                      is_dyn, device_capacity, act_fn):
    GemmCls = GemmGatedLNBdllSm90
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype, b_dtype, None, None, a_major, b_major, None, None
    )
    pa_n = cute.sym_int()  # gated postact width = n // 2
    pa_leading = 0 if postact_major == "m" else 1  # bdll = M-major
    mPostAct = fake_tensor(postact_dtype, (m, pa_n, l), leading_dim=pa_leading,
                           divisibility=div_for_dtype(postact_dtype))
    mRowVec = fake_tensor(vec_dtype, (l, n), leading_dim=1, divisibility=4)   # S (n=2N)
    mB2 = fake_tensor(vec_dtype, (l, n), leading_dim=1, divisibility=4)
    mColVec = fake_tensor(vec_dtype, (l, m), leading_dim=1, divisibility=4)   # rstd
    mC1 = fake_tensor(vec_dtype, (l, m), leading_dim=1, divisibility=4)
    epi_args = GemmCls.EpilogueArguments(
        mPostAct, act_fn, mRowVecBroadcast=mRowVec, mColVecBroadcast=mColVec, mC1=mC1, mB2=mB2,
    )
    scheduler_args = make_fake_scheduler_args((is_dyn and device_capacity[0] == 9), False, l)
    varlen_args = make_fake_varlen_args(False, False, False, None)
    return compile_gemm_kernel(
        GemmCls, a_dtype, tile_shape_mn, cluster_shape_mnk,
        pingpong, True, False, is_dyn, device_capacity,
        mA, mB, mD, mC, epi_args, scheduler_args, varlen_args,
    )


def prepack_front_folded(WL, WLg, WR, WRg, ln_w, ln_b):
    """nn.Linear weights (D,D) -> (Bf (4D,D), S (4D,), B2 (4D,)) interleaved [gate|up], folded.

    Built ONCE per fixed weights. Bf[2o]=γ⊙gate_o, Bf[2o+1]=γ⊙up_o; left block then right.
    """
    D = WL.shape[0]

    def il_rows(A, B):
        return torch.stack([A, B], dim=1).reshape(2 * D, -1)

    def il_vec(a, b):
        return torch.stack([a, b], dim=1).reshape(2 * D)

    BwLg, SLg, B2Lg = fold_for_gemm(WLg, ln_w, ln_b, None)
    BwL, SL, B2L = fold_for_gemm(WL, ln_w, ln_b, None)
    BwRg, SRg, B2Rg = fold_for_gemm(WRg, ln_w, ln_b, None)
    BwR, SR, B2R = fold_for_gemm(WR, ln_w, ln_b, None)
    Bf = torch.cat([il_rows(BwLg, BwL), il_rows(BwRg, BwR)], 0).contiguous()
    S = torch.cat([il_vec(SLg, SL), il_vec(SRg, SR)]).float().contiguous()
    B2 = torch.cat([il_vec(B2Lg, B2L), il_vec(B2Rg, B2R)]).float().contiguous()
    return Bf, S, B2


def trimul_front_lnfold(x, Bf, S, B2, eps=1e-5, *, config, act_fn=None):
    """x: (B,L,L,D) RAW pair. Returns (left, right) in (B,D,L,L) bdll, LN_in folded.

    config: GemmConfig. Bf/S/B2 from prepack_front_folded. stats computed inside."""
    B, L, _, D = x.shape
    assert B == 1
    M = L * L
    dev, dt = x.device, x.dtype
    xf = x.reshape(M, D)
    rstd, c1 = stats_triton(xf, eps)                         # (M,) fp32 each
    if act_fn is None:
        act_fn = gate_fn_map["glu"]

    lr = torch.empty(B, 2 * D, L, L, device=dev, dtype=dt)
    lr_view = lr.view(2 * D, M).T                            # (M, 2D) M-major (strides 1, L*L)

    A_p = perm3d_single(xf.unsqueeze(0))
    B_p = perm3d_single(Bf.unsqueeze(0))
    PA_p = perm3d_single(lr_view.unsqueeze(0))
    a_major = get_major(A_p, "m", "k")
    b_major = get_major(B_p, "n", "k")
    postact_major = get_major(PA_p, "m", "n")
    cap = get_device_capacity(dev)
    compiled = _compile_gated_ln(
        torch2cute_dtype_map[dt], torch2cute_dtype_map[dt], torch2cute_dtype_map[dt],
        a_major, b_major, postact_major, torch2cute_dtype_map[torch.float32],
        (config.tile_m, config.tile_n), (config.cluster_m, config.cluster_n, 1),
        config.pingpong, config.is_dynamic_persistent, cap, act_fn,
    )
    if is_compile_only():
        return lr[:, :D], lr[:, D:]
    max_clusters = get_max_active_clusters(config.cluster_m * config.cluster_n)
    epi_args = GemmGatedLNBdllSm90.EpilogueArguments(
        PA_p, None,
        mRowVecBroadcast=S.view(1, 4 * D), mColVecBroadcast=rstd.view(1, M),
        mC1=c1.view(1, M), mB2=B2.view(1, 4 * D),
        rounding_mode=None,  # Constexpr baked at compile -> must be None at call
    )
    scheduler_args = make_scheduler_args(max_clusters, config.max_swizzle_size, None)
    varlen_args = make_varlen_args(None, None, None)
    compiled(A_p, B_p, None, None, epi_args, scheduler_args, varlen_args, None)
    return lr[:, :D], lr[:, D:]
