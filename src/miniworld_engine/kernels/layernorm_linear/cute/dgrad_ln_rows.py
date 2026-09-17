"""Projection-aware LayerNorm dgrad: external row corrections remove full-N tile restriction."""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import Float32

from quack.cute_dsl_utils import (
    mlir_namedtuple,
    get_device_capacity,
    get_max_active_clusters,
)
from quack.epi_ops import RowVecLoad, ColVecLoad
from quack.gemm_sm90 import GemmSm90
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.rounding import RoundingMode
from quack.compile_utils import make_fake_tensor as fake_tensor
from miniworld_engine.kernels._quack_compat import jit_cache
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


class _DgradLNRowsMixin(GemmDefaultEpiMixin):
    """Apply precomputed row corrections after the projection dgrad GEMM.

    Each output-column tile is independent: c1/c2 are supplied as row vectors,
    so ordinary cooperative/pingpong GEMM tiles and epilogue subtiles are valid.
    C holds saved xhat; D is channel-major for the following contractions.
    """

    _epi_ops = (
        *GemmDefaultEpiMixin._epi_ops,
        RowVecLoad("mGamma"),
        ColVecLoad("mC2red"),
        ColVecLoad("mC1red"),
    )
    _extra_param_fields = (("inv_k", Float32, Float32(1.0)),)

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None  # unused slot (default)
        mColVecBroadcast: Optional[cute.Tensor] = None  # rstd[m]
        mGamma: Optional[cute.Tensor] = None  # gamma[n] (= K output col)
        mC2red: Optional[cute.Tensor] = None  # mean_K(dxhat)
        mC1red: Optional[cute.Tensor] = None  # mean_K(dxhat*xhat)
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
        rs = epi_loop_tensors["mColVecBroadcast"]
        gamma = epi_loop_tensors["mGamma"]
        c2 = epi_loop_tensors["mC2red"]
        c1 = epi_loop_tensors["mC1red"]
        for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
            tRS_rD[i] = rs[i].to(Float32) * (
                tRS_rD[i].to(Float32) * gamma[i].to(Float32)
                - c2[i].to(Float32)
                - tRS_rC[i].to(Float32) * c1[i].to(Float32)
            )
        return None


class _DgradLNRowsSm90(_DgradLNRowsMixin, GemmSm90):
    pass


@jit_cache
def _compile(
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
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype, b_dtype, d_dtype, c_dtype, a_major, b_major, d_major, c_major
    )
    mColVec = fake_tensor(vec_dtype, (l, m), leading_dim=1, divisibility=4)  # rstd
    mGamma = fake_tensor(
        vec_dtype, (l, n), leading_dim=1, divisibility=4
    )  # gamma over output K
    mC2red = fake_tensor(Float32, (l, m), leading_dim=1, divisibility=4)
    mC1red = fake_tensor(Float32, (l, m), leading_dim=1, divisibility=4)
    epi_args = _DgradLNRowsSm90.EpilogueArguments(
        mColVecBroadcast=mColVec,
        mGamma=mGamma,
        mC2red=mC2red,
        mC1red=mC1red,
        inv_k=Float32(1.0),
    )
    sched = make_fake_scheduler_args((is_dyn and device_capacity[0] == 9), False, l)
    varlen = make_fake_varlen_args(False, False, False, None)
    return compile_gemm_kernel(
        _DgradLNRowsSm90,
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


def dgrad_ln_rows(
    dY: Tensor,
    W: Tensor,
    xhat: Tensor,
    gamma: Tensor,
    rstd: Tensor,
    c1: Tensor,
    c2: Tensor,
    *,
    config=None,
):
    from miniworld_engine.autotune.cute_config import (
        plain_sm90_candidates,
        resolve_config,
    )
    from miniworld_engine.autotune.native import tensor_key

    dev = get_device_capacity(dY.device)
    assert dev[0] == 9
    M, N = dY.shape
    K = W.shape[1]
    if (
        dY.dtype != torch.bfloat16
        or W.dtype != dY.dtype
        or xhat.dtype != dY.dtype
        or N != 128
        or M % 8 != 0
        or tuple(W.shape) != (128, 256)
        or xhat.shape != (M, K)
    ):
        raise ValueError("projection-aware dgrad requires BF16 (M,128) @ (128,256)")
    # The epilogue always loads FP32 gamma. Canonicalize before keying so
    # BF16 driver weights and FP32 production LN parameters share that ABI.
    gamma = gamma.float().contiguous()
    candidates = plain_sm90_candidates()
    cfg = config
    if cfg is None:
        cfg = resolve_config(
            "trimul_output_bwd_rows_sm90_cute",
            candidates,
            dtype=str(dY.dtype),
            bucket=tensor_key(dY, W, xhat, gamma, rstd, c1, c2),
            device_index=dY.device.index,
            run=lambda c: dgrad_ln_rows(dY, W, xhat, gamma, rstd, c1, c2, config=c),
        )
    if cfg not in candidates:
        raise ValueError("configuration outside declared projection-aware dgrad grid")
    dx = torch.empty(K, M, device=dY.device, dtype=dY.dtype).t()
    Wt = W.t().contiguous()
    A_p, B_p, D_p, C_p = perm3d(
        dY.unsqueeze(0), Wt.unsqueeze(0), dx.unsqueeze(0), xhat.unsqueeze(0)
    )
    fn = _compile(
        *get_dtypes(dY, W, dx, xhat),
        *get_majors(A_p, B_p, D_p, C_p),
        Float32,
        (cfg.tile_m, cfg.tile_n),
        (cfg.cluster_m, cfg.cluster_n, 1),
        cfg.pingpong,
        True,
        cfg.is_dynamic_persistent,
        dev,
    )
    from miniworld_engine.kernels._quack_compat import is_compile_only

    if is_compile_only():
        return dx
    args = _DgradLNRowsSm90.EpilogueArguments(
        mColVecBroadcast=rstd.float().contiguous().view(1, M),
        mGamma=gamma.float().contiguous().view(1, K),
        mC2red=c2.view(1, M),
        mC1red=c1.view(1, M),
        inv_k=Float32(1.0 / K),
        add_to_output=None,
        rounding_mode=None,
    )
    semaphore = (
        torch.zeros(1, device=dY.device, dtype=torch.int32)
        if cfg.is_dynamic_persistent
        else None
    )
    sched = make_scheduler_args(
        get_max_active_clusters(cfg.cluster_m * cfg.cluster_n),
        cfg.max_swizzle_size,
        semaphore,
    )
    fn(A_p, B_p, D_p, C_p, args, sched, make_varlen_args(None, None, None), None)
    return dx
