"""cute 1+4: fused dgrad GEMM (dY@W) + LN-backward epilogue.  [BRING-UP / iteration 1]

See FUSED_DGRAD_LNBWD_DESIGN.md. Forks M1's composable epilogue (GemmSm90 + GemmDefaultEpiMixin)
on a `dY @ W` GEMM. tile_N = K so the full K-row is in ONE epilogue subtile → single-pass
reduce+apply. x̂ is fed as the C operand (tRS_rC). v1 computes dx only (the novel N-reduction);
dγ/dβ (M-reduction + atomic) added once dx verifies.

    A = dY (M,N)   B = W (N,K)   C = x̂ (M,K)   D = dx (M,K)
    dx̂ = acc·γ ;  c2 = meanₖ(dx̂) ;  c1 = meanₖ(dx̂·x̂) ;  dx = rstd·(dx̂ − c2 − x̂·c1)

BRING-UP STATUS (iter 3): COMPILES + RUNS (fork/epilogue/C-operand=x̂/inv_k-via-params all
work). But dx cos=0.004 — WRONG: the per-row reduction is naive. The fragment groups (m,n) and
a thread holds elements for MULTIPLE m rows, so the flat accumulate + a single
`warp_reduction(threads_in_group=4)` is NOT row-aligned.
NEXT: build per-m partials via `layout_utils.convert_layout_zero_stride(frag, colvec_layout)`
(the `colvec_reduce_accumulate` pattern, epi_ops.py:517-536) → per-m s2,s1; warp_reduction(4)
per m; then apply per (m,n). Then add dγ/dβ (M-reduction + atomic). Then verify + bench.
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
from quack.epi_ops import RowVecLoad, ColVecLoad
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

    _epi_ops = (*GemmDefaultEpiMixin._epi_ops, RowVecLoad("mGamma"))
    _extra_param_fields = (("inv_k", Float32, Float32(1.0)),)

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None   # unused slot (default)
        mColVecBroadcast: Optional[cute.Tensor] = None   # rstd[m]
        mGamma: Optional[cute.Tensor] = None             # gamma[n] (= K output col)
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
        inv_k = params.inv_k
        nfrag = cute.size(tRS_rD)
        # dx̂ = acc·γ ; keep x̂ = C
        # per-thread partials of Σ dx̂ and Σ dx̂·x̂ across this thread's n-fragment, per row.
        # NOTE(bring-up): fragment groups (m,n); a thread holds a few of each. We reduce all
        # local elements then warp_reduction over the 4 N-lanes. If a thread spans >1 m this is
        # WRONG and needs the (m,n) split — fix after first run tells us the layout.
        s2 = Float32(0.0)
        s1 = Float32(0.0)
        for i in cutlass.range(nfrag, unroll_full=True):
            dxh = tRS_rD[i] * gamma[i]
            s2 += dxh
            s1 += dxh * tRS_rC[i].to(Float32)
        c2 = cute.arch.warp_reduction(s2, operator.add, threads_in_group=_LANES_IN_N) * inv_k
        c1 = cute.arch.warp_reduction(s1, operator.add, threads_in_group=_LANES_IN_N) * inv_k
        for i in cutlass.range(nfrag, unroll_full=True):
            dxh = tRS_rD[i] * gamma[i]
            tRS_rD[i] = rstd[i] * (dxh - c2 - tRS_rC[i].to(Float32) * c1)
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
    epi_args = _DgradLNBwdSm90.EpilogueArguments(mColVecBroadcast=mColVec, mGamma=mGamma, inv_k=Float32(1.0))
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
    epi_args = _DgradLNBwdSm90.EpilogueArguments(
        mColVecBroadcast=rstd2, mGamma=gamma2, inv_k=Float32(1.0 / K),
        add_to_output=None, rounding_mode=None,
    )
    sched = make_scheduler_args(mac, cfg.max_swizzle_size, tile_count_semaphore)
    varlen = make_varlen_args(None, None, None)
    fn(A_p, B_p, D_p, C_p, epi_args, sched, varlen, None)
    return dx
