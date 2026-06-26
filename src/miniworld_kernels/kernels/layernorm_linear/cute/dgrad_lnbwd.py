"""cute 1+4: fused dgrad GEMM (dY@W) + LN-backward epilogue.  [BRING-UP / iteration 1]

See FUSED_DGRAD_LNBWD_DESIGN.md. Forks M1's composable epilogue (GemmSm90 + GemmDefaultEpiMixin)
on a `dY @ W` GEMM. tile_N = K so the full K-row is in ONE epilogue subtile → single-pass
reduce+apply. x̂ is fed as the C operand (tRS_rC). v1 computes dx only (the novel N-reduction);
dγ/dβ (M-reduction + atomic) added once dx verifies.

    A = dY (M,N)   B = W (N,K)   C = x̂ (M,K)   D = dx (M,K)
    dx̂ = acc·γ ;  c2 = meanₖ(dx̂) ;  c1 = meanₖ(dx̂·x̂) ;  dx = rstd·(dx̂ − c2 − x̂·c1)

BRING-UP STATUS (iter 5): COMPILES + RUNS, structure follows gemm_dact's GemmDGatedMixin
template — two ColVecReduce ops ("mC2red","mC1red") give per-m accumulators via epi_loop_tensors,
filled by `colvec_reduce_accumulate`, then warp_reduction(threads_in_group=4) finalized IN-visit
(single subtile, tile_N=K), then dx applied. ColVecReduce param shape = (l,M,n_tiles) leading=2.
STILL WRONG: dx cos≈0 (numerics off, |max|~38). Suspects to debug next (hardware-in-loop):
  1. lanes_in_N hardcoded 4 — may differ for this tiled_copy (ColVecReduce derives it from
     `_get_lane_warp_layouts`); if not 4 the N-lane reduction groups wrong lanes.
  2. c2buf / dxhat flat-index alignment in colvec_reduce_accumulate (sizes/layout must match).
  3. c2buf[i] read in the apply after filter_zeros+warp_reduction (broadcast read of the per-m
     reduced value) — verify the reduced value lands where the apply reads.
Debug via a DEBUG mode dumping c2/c1 vs a torch reference per row. Then add dγ/dβ. NOT wired.
"""

from __future__ import annotations

import operator
from typing import NamedTuple, Optional

import torch
from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from quack.cute_dsl_utils import mlir_namedtuple, torch2cute_dtype_map, get_device_capacity, get_max_active_clusters
from quack.epi_ops import RowVecLoad, ColVecLoad, ColVecReduce, colvec_reduce_accumulate
from quack.gemm_sm90 import GemmSm90
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.rounding import RoundingMode
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cache_utils import jit_cache
from quack.gemm_interface import default_config
from quack.gemm_tvm_ffi_utils import (
    get_majors, get_dtypes, perm3d, make_scheduler_args, make_varlen_args,
    make_fake_scheduler_args, make_fake_varlen_args, make_fake_gemm_tensors, compile_gemm_kernel,
)

_LANES_IN_N = 4  # SM90 WGMMA epilogue: 4 lanes share an M row (ColVecReduce.end)


class _DgradLNBwdMixin(GemmDefaultEpiMixin):
    """dx = rstd·(acc·γ − meanₖ(acc·γ) − x̂·meanₖ(acc·γ·x̂)).  γ=RowVec(n), rstd=ColVec(m), x̂=C."""

    # mC2red/mC1red: ColVecReduce gives correctly (m,n)→m -laid-out per-row accumulators via
    # epi_loop_tensors (begin allocates + zeros). We accumulate Σdx̂ and Σ(dx̂·x̂) in visit, then
    # — because tile_N=K is a SINGLE subtile — finalize the N-lane warp reduction IN visit (the
    # framework's ColVecReduce.end would only write to gmem, too late for the dx apply).
    _epi_ops = (*GemmDefaultEpiMixin._epi_ops, RowVecLoad("mGamma"),
                ColVecReduce("mC2red"), ColVecReduce("mC1red"))
    _extra_param_fields = (("inv_k", Float32, Float32(1.0)),)

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None   # unused slot (default)
        mColVecBroadcast: Optional[cute.Tensor] = None   # rstd[m]
        mGamma: Optional[cute.Tensor] = None             # gamma[n] (= K output col)
        mC2red: Optional[cute.Tensor] = None             # scratch (M,) for Σdx̂ (gmem ignored)
        mC1red: Optional[cute.Tensor] = None             # scratch (M,) for Σdx̂·x̂
        sr_seed: Optional[cute.Tensor] = None
        inv_k: Optional[Float32] = None
        add_to_output: cutlass.Constexpr[bool] = False
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = args.rounding_mode
        d = self._epi_ops_to_params_dict(args)
        d["inv_k"] = args.inv_k
        return self.EpilogueParams(**d)

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        rstd = epi_loop_tensors["mColVecBroadcast"]   # per-element (broadcast over n)
        gamma = epi_loop_tensors["mGamma"]            # per-element (broadcast over m)
        c2buf = epi_loop_tensors["mC2red"]            # per-m accumulator (stride (1,0))
        c1buf = epi_loop_tensors["mC1red"]
        inv_k = params.inv_k
        nfrag = cute.size(tRS_rD)
        dxhat = cute.make_rmem_tensor_like(tRS_rD, Float32)
        xh = cute.make_rmem_tensor_like(tRS_rD, Float32)
        for i in cutlass.range(nfrag, unroll_full=True):
            dxhat[i] = tRS_rD[i].to(Float32) * gamma[i].to(Float32)
            xh[i] = tRS_rC[i].to(Float32)
        # per-m partial sums (n collapsed via the colvec stride-(1,0) layout)
        colvec_reduce_accumulate(self, c2buf, dxhat)
        colvec_reduce_accumulate(self, c1buf, dxhat, rScale=xh)
        # finalize across the N lanes, in register (single subtile → end is too late)
        c2f = cute.filter_zeros(c2buf)
        c1f = cute.filter_zeros(c1buf)
        for j in cutlass.range(cute.size(c2f), unroll_full=True):
            c2f[j] = cute.arch.warp_reduction(c2f[j], operator.add, threads_in_group=_LANES_IN_N)
            c1f[j] = cute.arch.warp_reduction(c1f[j], operator.add, threads_in_group=_LANES_IN_N)
        for i in cutlass.range(nfrag, unroll_full=True):
            tRS_rD[i] = rstd[i].to(Float32) * (dxhat[i] - c2buf[i] * inv_k - xh[i] * c1buf[i] * inv_k)
        return None


class _DgradLNBwdSm90(_DgradLNBwdMixin, GemmSm90):
    pass


@jit_cache
def _compile(a_dtype, b_dtype, d_dtype, c_dtype, a_major, b_major, d_major, c_major,
             vec_dtype, tile_mn, cluster_mnk, pingpong, persistent, is_dyn, device_capacity):
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype, b_dtype, d_dtype, c_dtype, a_major, b_major, d_major, c_major
    )
    mColVec = fake_tensor(vec_dtype, (l, m), leading_dim=1, divisibility=4)  # rstd
    mGamma = fake_tensor(vec_dtype, (l, n), leading_dim=1, divisibility=4)   # gamma over output K
    n_tiles = cute.sym_int()
    mC2red = fake_tensor(Float32, (l, m, n_tiles), leading_dim=2, divisibility=1)   # scratch (l,M,n_tiles)
    mC1red = fake_tensor(Float32, (l, m, n_tiles), leading_dim=2, divisibility=1)
    epi_args = _DgradLNBwdSm90.EpilogueArguments(
        mColVecBroadcast=mColVec, mGamma=mGamma, mC2red=mC2red, mC1red=mC1red, inv_k=Float32(1.0))
    sched = make_fake_scheduler_args((is_dyn and device_capacity[0] == 9), False, l)
    varlen = make_fake_varlen_args(False, False, False, None)
    return compile_gemm_kernel(
        _DgradLNBwdSm90, a_dtype, tile_mn, cluster_mnk, pingpong, persistent, False, is_dyn,
        device_capacity, mA, mB, mD, mC, epi_args, sched, varlen,
    )


def dgrad_lnbwd_cute(dY: Tensor, W: Tensor, xhat: Tensor, _gamma: Tensor, rstd: Tensor):
    """dx (M,K) = LN-backward(dY@W). dY (M,N), W (N,K), xhat=(x-mean)*rstd (M,K), gamma (K,), rstd (M,)."""
    dev = get_device_capacity(dY.device)
    assert dev[0] == 9, "SM90 only"
    M, N = dY.shape
    K = W.shape[1]
    cfg = default_config(dY.device)
    # tile_N must cover K (single-subtile reduction): use K (<=256) as tile_n.
    tile_mn = (cfg.tile_m, K)
    dx = torch.empty(M, K, device=dY.device, dtype=dY.dtype)
    A, B, C, Dt = dY.unsqueeze(0), W.unsqueeze(0), xhat.unsqueeze(0), dx.unsqueeze(0)
    A_p, B_p, D_p, C_p = perm3d(A, B, Dt, C)
    a_maj, b_maj, d_maj, c_maj = get_majors(A_p, B_p, D_p, C_p)
    a_dt, b_dt, d_dt, c_dt = get_dtypes(dY, W, dx, xhat)
    vec_dt = torch2cute_dtype_map[rstd.dtype]
    fn = _compile(a_dt, b_dt, d_dt, c_dt, a_maj, b_maj, d_maj, c_maj, vec_dt,
                  tile_mn, (cfg.cluster_m, cfg.cluster_n, 1), cfg.pingpong, True,
                  cfg.is_dynamic_persistent, dev)
    from quack.cache_utils import COMPILE_ONLY
    if COMPILE_ONLY:
        return dx
    mac = get_max_active_clusters(cfg.cluster_m * cfg.cluster_n)
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=dY.device)
        if (cfg.is_dynamic_persistent and dev[0] == 9) else None
    )
    rstd2 = rstd.float().contiguous().view(1, M)
    gamma2 = _gamma.float().contiguous().view(1, K)
    c2scratch = torch.empty(1, M, 1, dtype=torch.float32, device=dY.device)  # (l, M, n_tiles=1)
    c1scratch = torch.empty(1, M, 1, dtype=torch.float32, device=dY.device)
    epi_args = _DgradLNBwdSm90.EpilogueArguments(
        mColVecBroadcast=rstd2, mGamma=gamma2, mC2red=c2scratch, mC1red=c1scratch,
        inv_k=Float32(1.0 / K), add_to_output=None, rounding_mode=None,
    )
    sched = make_scheduler_args(mac, cfg.max_swizzle_size, tile_count_semaphore)
    varlen = make_varlen_args(None, None, None)
    fn(A_p, B_p, D_p, C_p, epi_args, sched, varlen, None)
    return dx
