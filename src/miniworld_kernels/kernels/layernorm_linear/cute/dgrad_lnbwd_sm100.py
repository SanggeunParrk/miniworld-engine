"""sm100 (B200) port of the H100 fused dgrad GEMM (dY@W) + LN-norm-backward epilogue -> dx.

Faithful port of `layernorm_linear/cute/dgrad_lnbwd.py` (H100 GemmSm90 WGMMA) onto the
tcgen05 Blackwell collective (GemmSm100). Same math / precision (bf16 in, fp32 acc, fp32 LN
stats & row reductions, bf16 out):

    A = dY (M,N)   B = W^T (K,N)   C = xhat (M,K)   D = dx (M,K)
    dxhat = acc*g ;  c2 = mean_k(dxhat) ;  c1 = mean_k(dxhat*xhat) ;
    dx = rstd*(dxhat - c2 - xhat*c1)

The H100 kernel overrides 3 sm90-internal hooks that do NOT port by base-class swap. This
module re-derives them the sm100 way:

  (1) `_compute_stages`: NOT overridden -- the sm90 override (10-arg) mismatches the sm100
      collective's 17-arg call. We inherit `GemmSm100._compute_stages` (returns 4 values incl.
      num_acc_stage). The sm90 override was a smem-reclaim perf tweak, not a correctness need.

  (2) Full-N single epilogue subtile (so the per-row LN reduction over K is single-pass): the
      sm90 hook `_sm90_compute_tile_shape_or_override` is never called by tcgen05. Instead we
      override `_setup_attributes` and force `self.epi_tile` to the full CTA-N via a scoped
      monkeypatch of `compute_epilogue_tile_shape` (its natural bf16 result would be epi_N=32 ->
      4 subtiles -> partial c1/c2 -> cos~0.5, the same class of bug as the H100 bring-up).

  (3) tcgen05 lane reduction: the H100 finalizes c1/c2 in-visit with a HARDCODED 4-lane
      `warp_reduction` matching the WGMMA t2r layout. On tcgen05 the t2r fragment->lane mapping
      is different. We derive `lanes_in_N` / `warps_in_N` generically from `tiled_copy_t2r`
      (via quack's `_get_lane_warp_layouts`, exactly as `ColVecReduce.end` does) in
      `epi_setup_postact`, then reduce with the derived lane count in-visit (skipped when
      lanes_in_N==1, i.e. the whole row is already in one thread's registers).

Produces ONLY dx. dgamma/dbeta/dW are derived in the composed backward from T = dY^T@xhat.
"""

from __future__ import annotations

import operator
from typing import NamedTuple, Optional

import torch
from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr

import cutlass.utils.blackwell_helpers as _bh

from quack.cute_dsl_utils import (
    mlir_namedtuple, torch2cute_dtype_map, get_device_capacity, get_max_active_clusters,
)
from quack.epi_ops import RowVecLoad, ColVecReduce, colvec_reduce_accumulate, _get_lane_warp_layouts
from quack.gemm_sm100 import GemmSm100
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.rounding import RoundingMode
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cache_utils import jit_cache, COMPILE_ONLY
from quack.gemm_interface import default_config
from quack.gemm_tvm_ffi_utils import (
    get_majors, get_dtypes, perm3d, make_scheduler_args, make_varlen_args,
    make_fake_scheduler_args, make_fake_varlen_args, make_fake_gemm_tensors, compile_gemm_kernel,
)


class _DgradLNBwdSm100(GemmDefaultEpiMixin, GemmSm100):
    """dx = rstd*(acc*g - mean_k(acc*g) - xhat*mean_k(acc*g*xhat)). g=RowVec(n), rstd=ColVec(m), xhat=C.

    LN-backward row reduction (c1,c2 over full K = GEMM-N) + apply happen in ONE full-N epi
    subtile (forced in `_setup_attributes`). Lane reduction is derived from the tcgen05 t2r
    layout (blocker 3), not hardcoded.
    """

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
        mC2red: Optional[cute.Tensor] = None             # scratch (M,) for sum dxhat (gmem ignored)
        mC1red: Optional[cute.Tensor] = None             # scratch (M,) for sum dxhat*xhat
        sr_seed: Optional[cute.Tensor] = None
        inv_k: Optional[Float32] = None
        add_to_output: cutlass.Constexpr[bool] = False
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = args.rounding_mode
        d = self._epi_ops_to_params_dict(args)
        d["inv_k"] = args.inv_k
        return self.EpilogueParams(**d)

    # -- blocker 2: force a single full-N epilogue subtile (tcgen05 has no sm90 hook) --
    def _setup_attributes(self, epilogue_args, varlen_args):
        _orig = _bh.compute_epilogue_tile_shape

        def _forced(cta_tile_shape, use_2cta_instrs, layout_d, elem_ty_d,
                    *, layout_c=None, elem_ty_c=None, loc=None, ip=None):
            cta_m, cta_n = cta_tile_shape[:2]
            warp_m, warp_n = (2, 2) if (cta_m == 64 and use_2cta_instrs) else (4, 1)
            tile_m = min(cta_m, 32 * warp_m)
            tile_n = cta_n  # full N -> single subtile (whole K-row visible to the reduce)
            tile_m_layout = cute.make_layout(tile_m, loc=loc, ip=ip)
            tile_n_layout = cute.make_layout(
                (tile_n // warp_n, warp_n), stride=(1, cta_n // warp_n), loc=loc, ip=ip)
            return (tile_m_layout, cute.coalesce(tile_n_layout, loc=loc, ip=ip))

        _bh.compute_epilogue_tile_shape = _forced
        try:
            super()._setup_attributes(epilogue_args, varlen_args)
        finally:
            _bh.compute_epilogue_tile_shape = _orig

    # -- blocker 3: derive the tcgen05 lane/warp layout for the row reduction --
    def epi_setup_postact(self, params, epi_smem_tensors, tiled_copy_r2s, tiled_copy_t2r,
                          tile_coord_mnkl, varlen_manager, tidx):
        tiled_copy = tiled_copy_t2r if tiled_copy_t2r is not None else tiled_copy_r2s
        reference_src = tiled_copy_t2r is None
        lane_layout_MN, warp_layout_MN = _get_lane_warp_layouts(tiled_copy, reference_src)
        self._lanes_in_N = cute.size(lane_layout_MN, mode=[1])
        self._warps_in_N = cute.size(warp_layout_MN, mode=[1])
        assert self._warps_in_N == 1, (
            f"_DgradLNBwdSm100 needs warps_in_N==1 for in-visit reduction, got {self._warps_in_N}")
        return None

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
        # per-m partial sums over this thread's N fragment (arch-aware packed path inside)
        colvec_reduce_accumulate(self, c2buf, dxhat)
        colvec_reduce_accumulate(self, c1buf, dxhat, rScale=xh)
        # finalize across the N lanes in-register (single subtile -> ColVecReduce.end too late).
        # lanes_in_N derived from the tcgen05 t2r layout; ==1 => whole row already in one thread.
        if const_expr(self._lanes_in_N > 1):
            c2f = cute.filter_zeros(c2buf)
            c1f = cute.filter_zeros(c1buf)
            for j in cutlass.range(cute.size(c2f), unroll_full=True):
                c2f[j] = cute.arch.warp_reduction(
                    c2f[j], operator.add, threads_in_group=self._lanes_in_N)
                c1f[j] = cute.arch.warp_reduction(
                    c1f[j], operator.add, threads_in_group=self._lanes_in_N)
        for i in cutlass.range(nfrag, unroll_full=True):
            tRS_rD[i] = rstd[i].to(Float32) * (
                dxhat[i] - c2buf[i] * inv_k - xh[i] * c1buf[i] * inv_k)
        return None


@jit_cache
def _compile(a_dtype, b_dtype, d_dtype, c_dtype, a_major, b_major, d_major, c_major,
             vec_dtype, tile_mn, cluster_mnk, is_dyn, device_capacity):
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype, b_dtype, d_dtype, c_dtype, a_major, b_major, d_major, c_major
    )
    mColVec = fake_tensor(vec_dtype, (l, m), leading_dim=1, divisibility=4)  # rstd
    mGamma = fake_tensor(vec_dtype, (l, n), leading_dim=1, divisibility=4)   # gamma over output K
    n_tiles = cute.sym_int()
    mC2red = fake_tensor(Float32, (l, m, n_tiles), leading_dim=2, divisibility=1)
    mC1red = fake_tensor(Float32, (l, m, n_tiles), leading_dim=2, divisibility=1)
    epi_args = _DgradLNBwdSm100.EpilogueArguments(
        mColVecBroadcast=mColVec, mGamma=mGamma, mC2red=mC2red, mC1red=mC1red, inv_k=Float32(1.0))
    sched = make_fake_scheduler_args((is_dyn and device_capacity[0] == 9), False, l)
    varlen = make_fake_varlen_args(False, False, False, None)
    return compile_gemm_kernel(
        _DgradLNBwdSm100, a_dtype, tile_mn, cluster_mnk, False, True, False, is_dyn,
        device_capacity, mA, mB, mD, mC, epi_args, sched, varlen,
    )


def dgrad_lnbwd_sm100(dY: Tensor, W: Tensor, xhat: Tensor, _gamma: Tensor, rstd: Tensor):
    """dx (M,K) = LN-backward(dY@W). dY (M,N), W (N,K), xhat=(x-mean)*rstd (M,K), gamma (K,), rstd (M,)."""
    dev = get_device_capacity(dY.device)
    assert dev[0] == 10, "SM100 only"
    M, N = dY.shape
    K = W.shape[1]
    cfg = default_config(dY.device)
    # CTA tile_n must cover K (single-subtile reduction). tile_m=128 -> 4 epi warps over M.
    tile_m = 128
    tile_mn = (tile_m, K)
    dx = torch.empty(M, K, device=dY.device, dtype=dY.dtype)
    # GEMM computes A @ B^T; want dx_normed = dY @ W (contract N) => B = W^T (K,N).
    Wt = W.t().contiguous()  # (K, N)
    A, B, C, Dt = dY.unsqueeze(0), Wt.unsqueeze(0), xhat.unsqueeze(0), dx.unsqueeze(0)
    A_p, B_p, D_p, C_p = perm3d(A, B, Dt, C)
    a_maj, b_maj, d_maj, c_maj = get_majors(A_p, B_p, D_p, C_p)
    a_dt, b_dt, d_dt, c_dt = get_dtypes(dY, W, dx, xhat)
    vec_dt = torch2cute_dtype_map[rstd.dtype]
    cluster_mnk = (1, 1, 1)
    fn = _compile(a_dt, b_dt, d_dt, c_dt, a_maj, b_maj, d_maj, c_maj, vec_dt,
                  tile_mn, cluster_mnk, cfg.is_dynamic_persistent, dev)
    if COMPILE_ONLY:
        return dx
    mac = get_max_active_clusters(1)
    rstd2 = rstd.float().contiguous().view(1, M)
    gamma2 = _gamma.float().contiguous().view(1, K)
    c2scratch = torch.empty(1, M, 1, dtype=torch.float32, device=dY.device)  # (l, M, n_tiles=1)
    c1scratch = torch.empty(1, M, 1, dtype=torch.float32, device=dY.device)
    epi_args = _DgradLNBwdSm100.EpilogueArguments(
        mColVecBroadcast=rstd2, mGamma=gamma2, mC2red=c2scratch, mC1red=c1scratch,
        inv_k=Float32(1.0 / K), add_to_output=None, rounding_mode=None,
    )
    sched = make_scheduler_args(mac, cfg.max_swizzle_size, None)
    varlen = make_varlen_args(None, None, None)
    try:
        fn(A_p, B_p, D_p, C_p, epi_args, sched, varlen, None)              # sm90 arity
    except TypeError:
        fn(A_p, B_p, D_p, C_p, epi_args, sched, varlen, None, None, None)  # sm100 (mSFA,mSFB,trace)
    return dx
