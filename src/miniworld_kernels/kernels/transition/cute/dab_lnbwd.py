"""Transition-specific fused dAB@Wab + LayerNorm backward for SM90.

This differs from the generic LayerNormLinear dgrad+LN-bwd kernel: the C operand is the
raw Transition input x, not a pre-materialized xhat. The epilogue reconstructs
``xhat = x * rstd - c1`` in registers and applies LN backward directly to the GEMM
accumulator ``d_xn = dAB @ Wab``.
"""

from __future__ import annotations

import operator
from typing import NamedTuple, Optional

import torch
from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import Float32

from miniworld_kernels.kernels._quack_compat import is_compile_only, jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import (
    get_device_capacity,
    get_max_active_clusters,
    mlir_namedtuple,
    torch2cute_dtype_map,
)
from quack.epi_ops import ColVecLoad, ColVecReduce, RowVecLoad, colvec_reduce_accumulate
from quack.gemm_default_epi import GemmDefaultEpiMixin
from miniworld_kernels.kernels._quack_compat import default_config
from quack.gemm_sm90 import GemmSm90
from quack.gemm_tvm_ffi_utils import (
    compile_gemm_kernel,
    get_dtypes,
    get_majors,
    make_fake_gemm_tensors,
    make_fake_scheduler_args,
    make_fake_varlen_args,
    make_scheduler_args,
    make_varlen_args,
    perm3d,
)
from quack.rounding import RoundingMode

_LANES_IN_N = 4


class _TransitionDabLNBwdMixin(GemmDefaultEpiMixin):
    """D = LNBackward(dAB @ Wab), using raw x as the C operand."""

    @classmethod
    def _compute_stages(
        cls,
        cta_tile_shape_mnk,
        epi_tile,
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        epilogue_args,
        smem_capacity,
        occupancy,
    ):
        epi_stage = 1
        epi_c_stage = 1
        d_bytes_per_stage = cute.size(epi_tile) * d_dtype.width // 8
        epi_bytes_per_stage = d_bytes_per_stage + cls.epi_smem_bytes_per_stage(
            epilogue_args,
            cta_tile_shape_mnk,
            epi_tile,
        )
        epi_bytes = epi_bytes_per_stage * epi_stage
        epi_bytes += cute.size(epi_tile) * c_dtype.width // 8 * epi_c_stage
        a_shape = cute.slice_(cta_tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(cta_tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        remaining_bytes = smem_capacity // occupancy - 1024 - epi_bytes
        ab_stage = remaining_bytes // ab_bytes_per_stage
        return ab_stage, epi_stage, epi_c_stage

    @staticmethod
    def _sm90_compute_tile_shape_or_override(
        cta_tile_shape_mnk,
        atom_layout_mnk,
        element_type=None,
        epi_tile_override=None,
    ):
        assert atom_layout_mnk[0] == 1 and atom_layout_mnk[1] == 1, (
            "Transition dAB+LN-bwd needs atom_layout 1x1 for one full-N epilogue subtile"
        )
        return (cta_tile_shape_mnk[0], cta_tile_shape_mnk[1])

    _epi_ops = (
        *GemmDefaultEpiMixin._epi_ops,
        RowVecLoad("mGamma"),
        ColVecLoad("mC1"),
        ColVecReduce("mC2red"),
        ColVecReduce("mC1red"),
    )
    _extra_param_fields = (("inv_k", Float32, Float32(1.0)),)

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mGamma: Optional[cute.Tensor] = None
        mC1: Optional[cute.Tensor] = None
        mC2red: Optional[cute.Tensor] = None
        mC1red: Optional[cute.Tensor] = None
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
        gamma = epi_loop_tensors["mGamma"]
        rstd = epi_loop_tensors["mColVecBroadcast"]
        c1 = epi_loop_tensors["mC1"]
        c2buf = epi_loop_tensors["mC2red"]
        c1buf = epi_loop_tensors["mC1red"]
        inv_k = params.inv_k
        nfrag = cute.size(tRS_rD)
        dxhat = cute.make_rmem_tensor_like(tRS_rD, Float32)
        xhat = cute.make_rmem_tensor_like(tRS_rD, Float32)
        for i in cutlass.range(nfrag, unroll_full=True):
            dxhat[i] = tRS_rD[i].to(Float32) * gamma[i].to(Float32)
            xhat[i] = tRS_rC[i].to(Float32) * rstd[i].to(Float32) - c1[i].to(Float32)
        colvec_reduce_accumulate(self, c2buf, dxhat)
        colvec_reduce_accumulate(self, c1buf, dxhat, rScale=xhat)
        c2f = cute.filter_zeros(c2buf)
        c1f = cute.filter_zeros(c1buf)
        for j in cutlass.range(cute.size(c2f), unroll_full=True):
            c2f[j] = cute.arch.warp_reduction(c2f[j], operator.add, threads_in_group=_LANES_IN_N)
            c1f[j] = cute.arch.warp_reduction(c1f[j], operator.add, threads_in_group=_LANES_IN_N)
        for i in cutlass.range(nfrag, unroll_full=True):
            tRS_rD[i] = rstd[i].to(Float32) * (
                dxhat[i] - c2buf[i] * inv_k - xhat[i] * c1buf[i] * inv_k
            )
        return None


class _TransitionDabLNBwdSm90(_TransitionDabLNBwdMixin, GemmSm90):
    pass


@jit_cache
def _compile(
    abi_tag,
    a_dtype,
    b_dtype,
    d_dtype,
    c_dtype,
    a_major,
    b_major,
    d_major,
    c_major,
    vec_dtype,
    tile_mn,
    cluster_mnk,
    pingpong,
    persistent,
    is_dyn,
    device_capacity,
):
    _ = abi_tag
    mA, mB, mD, mC, m, n, _k, l = make_fake_gemm_tensors(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
    )
    mGamma = fake_tensor(vec_dtype, (l, n), leading_dim=1, divisibility=4)
    mRstd = fake_tensor(vec_dtype, (l, m), leading_dim=1, divisibility=4)
    mC1 = fake_tensor(vec_dtype, (l, m), leading_dim=1, divisibility=4)
    n_tiles = cute.sym_int()
    mC2red = fake_tensor(Float32, (l, m, n_tiles), leading_dim=2, divisibility=1)
    mC1red = fake_tensor(Float32, (l, m, n_tiles), leading_dim=2, divisibility=1)
    epi_args = _TransitionDabLNBwdSm90.EpilogueArguments(
        mColVecBroadcast=mRstd,
        mGamma=mGamma,
        mC1=mC1,
        mC2red=mC2red,
        mC1red=mC1red,
        inv_k=Float32(1.0),
    )
    sched = make_fake_scheduler_args((is_dyn and device_capacity[0] == 9), False, l)
    varlen = make_fake_varlen_args(False, False, False, None)
    return compile_gemm_kernel(
        _TransitionDabLNBwdSm90,
        a_dtype,
        tile_mn,
        cluster_mnk,
        pingpong,
        persistent,
        False,
        is_dyn,
        device_capacity,
        mA,
        mB,
        mD,
        mC,
        epi_args,
        sched,
        varlen,
    )


def transition_dab_lnbwd_cute(
    dAB: Tensor,
    w_ab: Tensor,
    x: Tensor,
    gamma: Tensor,
    rstd: Tensor,
    c1: Tensor,
) -> Tensor:
    """Return dx = LNBackward(dAB @ w_ab), without materializing d_xn."""
    dev = get_device_capacity(dAB.device)
    assert dev[0] == 9, "SM90 only"
    M, _n = dAB.shape
    K = w_ab.shape[1]
    cfg = default_config(dAB.device)
    tile_mn = (64, K)
    pingpong = True
    dx = torch.empty(M, K, device=dAB.device, dtype=dAB.dtype)
    wt = w_ab.t().contiguous()
    a3 = dAB.contiguous().unsqueeze(0)
    b3 = wt.unsqueeze(0)
    d3 = dx.unsqueeze(0)
    c3 = x.contiguous().unsqueeze(0)
    a_p, b_p, d_p, c_p = perm3d(a3, b3, d3, c3)
    a_major, b_major, d_major, c_major = get_majors(a_p, b_p, d_p, c_p)
    a_dt, b_dt, d_dt, c_dt = get_dtypes(dAB, w_ab, dx, x)
    vec_dt = torch2cute_dtype_map[rstd.dtype]
    fn = _compile(
        1,
        a_dt,
        b_dt,
        d_dt,
        c_dt,
        a_major,
        b_major,
        d_major,
        c_major,
        vec_dt,
        tile_mn,
        (1, 1, 1),
        pingpong,
        True,
        cfg.is_dynamic_persistent,
        dev,
    )
    if is_compile_only():
        return dx
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=dAB.device)
        if (cfg.is_dynamic_persistent and dev[0] == 9)
        else None
    )
    gamma2 = gamma.float().contiguous().view(1, K)
    rstd2 = rstd.float().contiguous().view(1, M)
    c12 = c1.float().contiguous().view(1, M)
    c2scratch = torch.empty(1, M, 1, dtype=torch.float32, device=dAB.device)
    c1scratch = torch.empty(1, M, 1, dtype=torch.float32, device=dAB.device)
    epi_args = _TransitionDabLNBwdSm90.EpilogueArguments(
        mColVecBroadcast=rstd2,
        mGamma=gamma2,
        mC1=c12,
        mC2red=c2scratch,
        mC1red=c1scratch,
        inv_k=Float32(1.0 / K),
        add_to_output=None,
        rounding_mode=None,
    )
    sched = make_scheduler_args(
        get_max_active_clusters(1),
        cfg.max_swizzle_size,
        tile_count_semaphore,
    )
    varlen = make_varlen_args(None, None, None)
    fn(a_p, b_p, d_p, c_p, epi_args, sched, varlen, None)
    return dx
