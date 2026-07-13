"""Cute dual-gemm fused back-half: ONE gated GEMM that computes proj (LN-folded)
and gate (sigmoid) and multiplies — gate NOT materialized.

Kernel = quack GemmGatedSm90 (glu-style halving: acc 2D -> postact D) with a
CUSTOM act that injects the LN correction on the proj half:

    A = [tri | x_n]  (M, 2D),  B interleaved block-diag [W2 | Wg] (2D, 2D)
    acc[:, 2j] = tri @ W2_j   (proj logit, W2 = gamma*Wp)
    acc[:, 2j+1] = x_n @ Wg_j (gate logit)
    out[j] = (rstd[m]*acc[2j] - c1[m]*S[j] + B2[j]) * sigmoid(acc[2j+1])

rstd/c1 are PRECOMPUTED per-row stats of tri (passed as col-vecs); S/B2 are the
fold_for_gemm row-vecs (placed at the proj/even positions of a 2D vector). This
avoids an in-kernel stats reduction.  B=1, D=128, bf16, SM90.

FIRST CUT — composable-epilogue plumbing is intricate; expect to iterate.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple, Optional

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from quack.cute_dsl_utils import mlir_namedtuple
from quack.epi_ops import ColVecLoad, RowVecLoad, Scalar, TileStore
from quack.gemm_act import GemmGatedMixin, _gated_epi_tile_fn
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.gemm_sm90 import GemmSm90
from quack.rounding import RoundingMode
from quack.activation import sigmoid as cute_sigmoid
import quack.layout_utils as layout_utils
import os as _os
_DG_DBG = int(_os.environ.get("DG_DBG", "0"))  # 1=proj 2=gate 3=raw_p 4=raw_g


class GemmGatedLNMixin(GemmGatedMixin):
    """Gated halving + LN-correct(proj) × sigmoid(gate) custom act."""

    _epi_ops = (
        Scalar("alpha"),
        Scalar("beta"),
        Scalar("sr_seed", dtype=Int32),
        ColVecLoad("mRstd"),
        ColVecLoad("mC1"),
        RowVecLoad("mSv"),
        RowVecLoad("mB2v"),
        TileStore("mPostAct", epi_tile_fn=_gated_epi_tile_fn),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor
        mRstd: Optional[cute.Tensor] = None
        mC1: Optional[cute.Tensor] = None
        mSv: Optional[cute.Tensor] = None
        mB2v: Optional[cute.Tensor] = None
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        assert args.mPostAct.element_type.width == 16
        assert cutlass.utils.LayoutEnum.from_tensor(args.mPostAct).is_n_major_c()
        if self.arch == 90:
            assert self.cta_tile_shape_mnk[1] % 32 == 0
        self.rounding_mode = args.rounding_mode
        self.postact_dtype = args.mPostAct.element_type
        self.postact_layout = cutlass.utils.LayoutEnum.from_tensor(args.mPostAct)
        self.cta_tile_shape_postact_mn = (
            self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1] // 2,
        )
        d = self._epi_ops_to_params_dict(args)
        return self.EpilogueParams(**d)

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        rstd = epi_loop_tensors["mRstd"]
        c1 = epi_loop_tensors["mC1"]
        S = epi_loop_tensors["mSv"]
        B2 = epi_loop_tensors["mB2v"]
        post_layout = cute.recast_layout(2, 1, tRS_rD.layout)
        tRS_rPostAct = cute.make_rmem_tensor(post_layout.shape, self.acc_dtype)
        for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
            p = tRS_rD[2 * i]       # proj logit  = tri @ W2_j
            g = tRS_rD[2 * i + 1]   # gate logit  = x_n @ Wg_j
            proj = rstd[2 * i] * p - c1[2 * i] * S[2 * i] + B2[2 * i]
            if const_expr(_DG_DBG == 1):
                tRS_rPostAct[i] = proj
            elif const_expr(_DG_DBG == 2):
                tRS_rPostAct[i] = cute_sigmoid(g)
            elif const_expr(_DG_DBG == 3):
                tRS_rPostAct[i] = p
            elif const_expr(_DG_DBG == 4):
                tRS_rPostAct[i] = g
            else:
                tRS_rPostAct[i] = proj * cute_sigmoid(g)
        return tRS_rPostAct


class GemmGatedLNSm90(GemmGatedLNMixin, GemmSm90):
    pass


# ── vendored compile + launch (adapted from quack.gemm_act) ──────────────────
from miniworld_kernels.kernels._quack_compat import jit_cache, is_compile_only  # noqa: E402
from quack.compile_utils import make_fake_tensor as fake_tensor  # noqa: E402
from quack.cute_dsl_utils import (  # noqa: E402
    get_device_capacity, get_max_active_clusters, torch2cute_dtype_map,
)
from quack.gemm_tvm_ffi_utils import (  # noqa: E402
    get_major, perm3d_single, make_scheduler_args, make_varlen_args,
    make_fake_scheduler_args, make_fake_varlen_args, div_for_dtype,
    make_fake_gemm_tensors, compile_gemm_kernel,
)

_CFG = dict(tile_m=128, tile_n=256, cluster_m=1, cluster_n=1, pingpong=False)


@jit_cache
def _compile_dualgemm(a_dtype, b_dtype, postact_dtype, vec_dtype,
                      a_major, b_major, postact_major, device_capacity):
    cfg = _CFG
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype, b_dtype, None, None, a_major, b_major, None, None
    )
    pa_n = cute.sym_int()  # gated halves N -> N/2
    mPostAct = fake_tensor(postact_dtype, (m, pa_n, l), leading_dim=1,
                           divisibility=div_for_dtype(postact_dtype))
    mRstd = fake_tensor(vec_dtype, (l, m), leading_dim=1, divisibility=4)
    mC1 = fake_tensor(vec_dtype, (l, m), leading_dim=1, divisibility=4)
    mSv = fake_tensor(vec_dtype, (l, n), leading_dim=1, divisibility=4)
    mB2v = fake_tensor(vec_dtype, (l, n), leading_dim=1, divisibility=4)
    epi_args = GemmGatedLNSm90.EpilogueArguments(
        mPostAct, mRstd=mRstd, mC1=mC1, mSv=mSv, mB2v=mB2v,
        rounding_mode=RoundingMode.RN, sr_seed=None,
    )
    scheduler_args = make_fake_scheduler_args(False, False, l)
    varlen_args = make_fake_varlen_args(False, False, False, None)
    return compile_gemm_kernel(
        GemmGatedLNSm90, a_dtype, (cfg["tile_m"], cfg["tile_n"]),
        (cfg["cluster_m"], cfg["cluster_n"], 1),
        cfg["pingpong"], True, False, False, device_capacity,
        mA, mB, mD, mC, epi_args, scheduler_args, varlen_args,
    )


def prepack_dualgemm(Wp, Wg, ln_w, ln_b, *, dtype, device, eps=1e-5):
    """Weight-derived operands (built ONCE, reused across forwards): (Bk, S2, B22).

    Bk: (N=2D, K=2D) interleaved block-diag, kernel-ready (N,K). S2/B22: (2D,) with
    the per-proj-col fold values at the even (proj) positions.
    """
    from .dualgemm_back import build_dualgemm_operands

    D = Wp.shape[0]
    dummy = torch.zeros(1, D, 1, 1, device=device, dtype=dtype)
    _A, Bm, S, B2, _ = build_dualgemm_operands(
        None, torch.zeros(1, 1, 1, D, device=device, dtype=dtype), dummy,
        Wp, Wg, ln_w, ln_b, eps)
    S2 = torch.zeros(2 * D, device=device, dtype=torch.float32)
    B22 = torch.zeros(2 * D, device=device, dtype=torch.float32)
    S2[0::2] = S.float()
    B22[0::2] = B2.float()
    return Bm.t().contiguous(), S2, B22  # Bk (N,K)


def dualgemm_back_cute(tri_bdll, x_n, Wp, Wg, ln_w, ln_b, eps=1e-5, *, prepacked=None):
    """One gated GEMM: y = (LN_D(tri)@Wp) ⊙ sigmoid(x_n@Wg), gate not materialized."""
    B, D, L, _ = tri_bdll.shape
    assert B == 1
    M = L * L
    dev, dt = x_n.device, x_n.dtype
    tri_md = tri_bdll.reshape(D, M).t().contiguous()   # (M, D)
    A = torch.cat([tri_md, x_n.reshape(M, D)], dim=1)  # (M, 2D)
    if prepacked is None:
        Bk, S2, B22 = prepack_dualgemm(Wp, Wg, ln_w, ln_b, dtype=dt, device=dev, eps=eps)
    else:
        Bk, S2, B22 = prepacked

    # per-row stats of tri over D
    trif = tri_md.float()
    mean = trif.mean(dim=1)
    var = trif.var(dim=1, unbiased=False)
    rstd = (1.0 / torch.sqrt(var + eps)).contiguous()        # (M,)
    c1 = (mean * rstd).contiguous()                          # (M,)

    Y = torch.empty(M, D, device=dev, dtype=dt)

    A_p = perm3d_single(A.unsqueeze(0))        # (1,M,2D) -> (M,2D,1)
    B_p = perm3d_single(Bk.unsqueeze(0))       # (1,N,K)  -> (N,K,1)
    PostAct_p = perm3d_single(Y.unsqueeze(0))  # (1,M,D)  -> (M,D,1)
    a_major = get_major(A_p, "m", "k")
    b_major = get_major(B_p, "n", "k")
    postact_major = get_major(PostAct_p, "m", "n")
    cap = get_device_capacity(dev)
    compiled = _compile_dualgemm(
        torch2cute_dtype_map[dt], torch2cute_dtype_map[dt], torch2cute_dtype_map[dt],
        torch2cute_dtype_map[torch.float32], a_major, b_major, postact_major, cap,
    )
    if is_compile_only():
        return Y.view(B, L, L, D)
    epi_args = GemmGatedLNSm90.EpilogueArguments(
        PostAct_p, mRstd=rstd.view(1, M), mC1=c1.view(1, M),
        mSv=S2.view(1, 2 * D), mB2v=B22.view(1, 2 * D),
        rounding_mode=None, sr_seed=None,
    )
    max_clusters = get_max_active_clusters(1)
    scheduler_args = make_scheduler_args(max_clusters, 8, None)
    varlen_args = make_varlen_args(None, None, None)
    compiled(A_p, B_p, None, None, epi_args, scheduler_args, varlen_args, None)
    return Y.view(B, L, L, D)
