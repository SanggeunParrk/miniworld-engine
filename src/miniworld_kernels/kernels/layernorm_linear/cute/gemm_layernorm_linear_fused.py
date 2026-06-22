"""Milestone 2 (WIP): fused LayerNormLinear in ONE main kernel.

Forks quack's `GemmSm90` (no smaller seam — confirmed by reading the source).
Overrides 4 methods:
  - `__call__()`     : adds a [2*BLK_M] FP32 `sStats` smem buffer to SharedStorage.
  - `kernel()`       : faithful copy + per-row stat regs, calls our `mma` with sA,
                       finalizes mean/var/rstd/c1 -> sStats after the K loop, syncs.
  - `mma()`          : consumer WGMMA loop + per-row reduction of sA (=X) on CUDA
                       cores, in parallel with the in-flight tensor-core mma.
  - `epi_visit_acc()`: Y = rstd[m]*acc - c1[m]*S[n] + B2[n] via C-tile coords;
                       rstd/c1 from sStats, S/B2 from gmem params.

First version uses a NON-pingpong config so `self.epilogue_barrier` syncs the math
warps between the stats write and the epilogue read. Reduction: thread t owns rows
{t, t+NT, ...}; reads sA[m, 0:BLK_K] by logical index (cute resolves the swizzle),
holding the COMPLETE per-row sums (no cross-thread reduction).
"""

import math
from functools import partial
from typing import Callable, NamedTuple, Optional

import torch
from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32, Float32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup

import quack.utils as utils
import quack.copy_utils as copy_utils
import quack.sm90_utils as quack_sm90_utils
from quack.cute_dsl_utils import (
    mlir_namedtuple, torch2cute_dtype_map, get_device_capacity, get_max_active_clusters,
)
from quack.gemm_sm90 import GemmSm90
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.epi_ops import ColVecLoad, RowVecLoad, Scalar
from quack.rounding import RoundingMode
from quack.gemm_config import GemmConfig
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cache_utils import jit_cache
from cutlass.utils import LayoutEnum
from quack.varlen_utils import VarlenManager, VarlenArguments
from quack.pipeline import make_pipeline_state
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass import pipeline
import quack.layout_utils as layout_utils
from quack.gemm_tvm_ffi_utils import (
    get_majors, get_dtypes, perm3d, make_scheduler_args, make_varlen_args,
    make_fake_scheduler_args, make_fake_varlen_args, make_fake_gemm_tensors, compile_gemm_kernel,
)


import os
_DEBUG_MODE = int(os.environ.get("LNL_DEBUG", "0"))
_WS_DEBUG = int(os.environ.get("LNL_WS_DEBUG", "0"))  # warp-specialized stats bring-up
_WS = int(os.environ.get("LNL_WS", "0"))  # warp-specialized stats PRODUCTION path


@cute.jit
def _reduce_gmem_row(mX, gm, len_k):
    """Sum and sum-of-squares of X[gm, 0:len_k] read straight from gmem (mX is a
    plain (M,K) tensor passed via EpilogueArguments — element-indexable, unlike the
    TMA tensor). Decoupled from the GEMM's sA smem pipeline → no recycle races."""
    s = Float32(0.0)
    ssq = Float32(0.0)
    for k in cutlass.range(len_k):
        v = mX[gm, k].to(Float32)
        s += v
        ssq += v * v
    return s, ssq


@cute.jit
def _reduce_gmem_coop(mX, m_base, s_rstd, s_c1, wg_off, tidx, len_k, len_k_f, eps,
                      blk_m: cutlass.Constexpr[int], nwarps: cutlass.Constexpr[int]):
    """COALESCED LN stats from gmem: one warp per row. The 32 lanes of a warp read
    X[gm, lane], X[gm, lane+32], ... so lane k-strides are contiguous in memory
    (fully coalesced), then a butterfly warp-reduce sums them. Each warp owns
    rows {warp_id, warp_id+nwarps, ...}; lane 0 writes the per-row rstd/c1 to smem
    at s_rstd[wg_off + r] (wg_off lets each pingpong WG fill its own half).
    Replaces the one-thread-per-row reducer whose stride-K reads were uncoalesced
    and ~1000x slower than the available HBM bandwidth. Requires len_k % 32 == 0."""
    warp_id = tidx // 32
    lane = cute.arch.lane_idx()
    nstep = len_k // 32
    # ceil so non-divisor nwarps (e.g. 3 stats warps over 128 rows) still cover all rows.
    rows_per_warp = const_expr((blk_m + nwarps - 1) // nwarps)
    for ri in cutlass.range_constexpr(rows_per_warp):
        r = ri * nwarps + warp_id
        if r < blk_m:
            gm = m_base + r
            s = Float32(0.0)
            ssq = Float32(0.0)
            for j in cutlass.range(nstep):
                v = mX[gm, j * 32 + lane].to(Float32)
                s += v
                ssq += v * v
            for i in cutlass.range_constexpr(5):  # log2(32) butterfly ALL-reduce
                s = s + cute.arch.shuffle_sync_bfly(s, offset=1 << i)
                ssq = ssq + cute.arch.shuffle_sync_bfly(ssq, offset=1 << i)
            # After the butterfly ALL-reduce every lane holds the full s/ssq, so ALL 32
            # lanes write the SAME value to s_rstd[wg_off+r] (a benign redundant store).
            # This keeps the warp CONVERGED — an `if lane == 0` write left the warp
            # divergent at the next named barrier (synccheck "Divergent thread(s)") and
            # deadlocked under pingpong.
            mean = s / len_k_f
            var = ssq / len_k_f - mean * mean
            rstd = cute.math.rsqrt(var + eps, fastmath=True)
            s_rstd[wg_off + r] = rstd
            s_c1[wg_off + r] = mean * rstd


@cute.jit
def _stats_dump_gmem(mX, mDbg, m_base, stidx, len_k, len_k_f, eps,
                     blk_m: cutlass.Constexpr[int], nwarps: cutlass.Constexpr[int]):
    """[bring-up scaffold] warp-specialized stats: the idle load-WG warps reduce
    X[m_base:m_base+blk_m] from gmem (coalesced warp-per-row) and dump rstd to the
    debug gmem buffer mDbg[m_base+r]. `stidx` is the LOCAL stats-thread index (0..95)."""
    warp_id = stidx // 32
    lane = cute.arch.lane_idx()
    nstep = len_k // 32
    rows_per_warp = const_expr((blk_m + nwarps - 1) // nwarps)
    for ri in cutlass.range_constexpr(rows_per_warp):
        r = ri * nwarps + warp_id
        if r < blk_m:
            gm = m_base + r
            s = Float32(0.0)
            ssq = Float32(0.0)
            for j in cutlass.range(nstep):
                v = mX[gm, j * 32 + lane].to(Float32)
                s += v
                ssq += v * v
            for i in cutlass.range_constexpr(5):
                s = s + cute.arch.shuffle_sync_bfly(s, offset=1 << i)
                ssq = ssq + cute.arch.shuffle_sync_bfly(ssq, offset=1 << i)
            mean = s / len_k_f
            var = ssq / len_k_f - mean * mean
            mDbg[gm] = cute.math.rsqrt(var + eps, fastmath=True)


@cute.jit
def _reduce_sA_rows(sA_stage, red_sum, red_sumsq, tidx, blk_k, wg_rows, tpwg):
    """Each warpgroup reduces the rows IT owns in the epilogue (M-split: WG g owns
    rows [g*wg_rows, (g+1)*wg_rows)). Thread (wg, lane<wg_rows) owns one row, so
    there is NO cross-warpgroup stats sharing — avoids the persistent 2-WG race."""
    wg = tidx // tpwg
    lane = tidx % tpwg
    if lane < wg_rows:
        m = wg * wg_rows + lane
        s = Float32(0.0)
        ssq = Float32(0.0)
        for k in cutlass.range(blk_k, unroll_full=True):
            v = sA_stage[m, k].to(Float32)
            s += v
            ssq += v * v
        red_sum[0] = red_sum[0] + s
        red_sumsq[0] = red_sumsq[0] + ssq


class SmemColVec(ColVecLoad):
    """Like ColVecLoad but the per-m smem buffer is PRE-FILLED by the kernel
    (no gmem load). Used to broadcast in-kernel-computed rstd[m] / c1[m] onto the
    epilogue accumulator fragment with the same proven alignment as ColVecLoad."""

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        return {self.name: None}  # no gmem source; smem is filled by the kernel

    def needs_async_fence(self):
        return False

    def smem_struct_field(self, gemm, params):
        # pingpong runs two output tiles at once -> one half of the buffer per WG.
        mult = 2 if gemm.pingpong else 1
        size = mult * self._tile_size(gemm.cta_tile_shape_mnk)
        return (f"s_{self.name}", cute.struct.Align[cute.struct.MemRange[Float32, size], 16])

    def get_smem_tensor(self, gemm, params, storage_epi):
        mult = 2 if gemm.pingpong else 1
        size = mult * self._tile_size(gemm.cta_tile_shape_mnk)
        return getattr(storage_epi, f"s_{self.name}").get_tensor(cute.make_layout(size))

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        # In pingpong each WG reads its own half (offset warp_group_idx*tile_M).
        # thread_idx is the true hw idx (pingpong reassigns the local tidx upstream).
        if const_expr(gemm.pingpong):
            off = (cute.arch.thread_idx()[0] // gemm.num_threads_per_warp_group) * ctx.tile_M
        else:
            off = 0
        tDsV = ctx.partition_for_epilogue_fn(
            cute.make_tensor(
                smem_tensor.iterator + off,
                cute.make_layout((ctx.tile_M, ctx.tile_N), stride=self._broadcast_stride()),
            )
        )
        if const_expr(ctx.tiled_copy_t2r is not None):
            tDsV = ctx.tiled_copy_r2s.retile(tDsV)
        tDsV_sub = cute.group_modes(tDsV, 3, cute.rank(tDsV))[None, None, None, 0]
        tDrV_cvt = cute.make_rmem_tensor(tDsV_sub.layout, gemm.acc_dtype)
        return [tDsV, tDrV_cvt]


class _LNLEpiMixin(GemmDefaultEpiMixin):
    """Epilogue: Y = rstd[m]*acc - c1[m]*S[n] + B2[n].
    rstd/c1 from kernel-filled smem (SmemColVec), S/B2 from gmem (RowVecLoad)."""

    _epi_ops = (
        Scalar("alpha"),
        Scalar("beta"),
        Scalar("sr_seed", dtype=Int32),
        SmemColVec("mRstd"),
        SmemColVec("mC1"),
        RowVecLoad("mS"),
        RowVecLoad("mB2"),
    )
    _extra_param_fields = (
        ("eps", Float32, Float32(1e-5)), ("mX", object, None), ("mDbg", object, None),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mS: Optional[cute.Tensor] = None
        mB2: Optional[cute.Tensor] = None
        eps: Float32 = Float32(1e-5)
        mX: Optional[cute.Tensor] = None  # plain (M,K) X for the stats reduction
        mDbg: Optional[cute.Tensor] = None  # [M] gmem: warp-specialized-stats rstd dump
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRstd: Optional[cute.Tensor] = None
        mC1: Optional[cute.Tensor] = None
        add_to_output: cutlass.Constexpr[bool] = False
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = args.rounding_mode
        d = self._epi_ops_to_params_dict(args)
        d["eps"] = args.eps
        d["mX"] = args.mX
        d["mDbg"] = args.mDbg
        return self.EpilogueParams(**d)

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        rstd = epi_loop_tensors["mRstd"]
        c1 = epi_loop_tensors["mC1"]
        S = epi_loop_tensors["mS"]
        B2 = epi_loop_tensors["mB2"]
        for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
            if const_expr(_DEBUG_MODE == 1 or _DEBUG_MODE == 3):
                tRS_rD[i] = rstd[i]        # debug: output rstd[m] broadcast over n
            elif const_expr(_DEBUG_MODE == 2):
                tRS_rD[i] = c1[i]          # debug: output c1[m]
            elif const_expr(_DEBUG_MODE == 4):
                pass                       # debug: identity, output = acc (x@W2^T)
            else:
                tRS_rD[i] = rstd[i] * tRS_rD[i] - c1[i] * S[i] + B2[i]
        return None


class GemmLNLFusedSm90(_LNLEpiMixin, GemmSm90):
    @cute.jit
    def mma(self, ab_pipeline, ab_read_state, mma_fn, acc, acc_slow, k_tile_cnt,
            warp_group_idx, sA=None, red_sum=None, red_sumsq=None, tidx=Int32(0),
            blk_k: cutlass.Constexpr[int] = 64, wg_rows: cutlass.Constexpr[int] = 64,
            tpwg: cutlass.Constexpr[int] = 128, do_reduce=Boolean(True)):
        k_pipe_mmas = 1
        ab_release_state = ab_read_state.clone()
        num_prologue_mma = min(k_pipe_mmas, k_tile_cnt)
        peek = Boolean(True)
        if 0 < k_tile_cnt:
            peek = ab_pipeline.consumer_try_wait(ab_read_state)
        zero_init = Boolean(True)
        for k_tile in cutlass.range(num_prologue_mma):
            ab_pipeline.consumer_wait(ab_read_state, peek)
            idx = ab_read_state.index
            # Issue the WGMMA FIRST (async), then reduce sA on CUDA cores in PARALLEL
            # with the tensor-core compute (overlap). sync_warp: the pipeline empty
            # barrier needs an arrive from every consumer warp (lane 0 = signalling
            # thread), so each warp's lane 0 must wait for its lanes' sA reads before the
            # stage can recycle. fence: TMA async-proxy write -> generic ld.shared read.
            mma_fn(A_idx=idx, B_idx=idx, zero_init=zero_init)
            zero_init = Boolean(False)
            if const_expr(sA is not None):
                if do_reduce:  # uniform across the WG (same m-tile for all threads)
                    cute.arch.fence_view_async_shared()
                    _reduce_sA_rows(sA[None, None, idx], red_sum, red_sumsq, tidx, blk_k, wg_rows, tpwg)
                    cute.arch.sync_warp()
            ab_read_state.advance()
            peek = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek = ab_pipeline.consumer_try_wait(ab_read_state)
        for k_tile in cutlass.range(num_prologue_mma, k_tile_cnt, unroll=1):
            ab_pipeline.consumer_wait(ab_read_state, peek)
            idx = ab_read_state.index
            mma_fn(A_idx=idx, B_idx=idx, zero_init=zero_init)
            zero_init = Boolean(False)
            if const_expr(sA is not None):
                if do_reduce:
                    cute.arch.fence_view_async_shared()
                    _reduce_sA_rows(sA[None, None, idx], red_sum, red_sumsq, tidx, blk_k, wg_rows, tpwg)
                    cute.arch.sync_warp()
            warpgroup.wait_group(k_pipe_mmas)
            ab_pipeline.consumer_release(ab_release_state)
            ab_read_state.advance()
            ab_release_state.advance()
            peek = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek = ab_pipeline.consumer_try_wait(ab_read_state)
        if const_expr(self.pingpong):
            # Cue the other WG's MMA to start (release the tensor cores).
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        warpgroup.wait_group(0)
        for k_tile in cutlass.range(num_prologue_mma, unroll=1):
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
        return ab_read_state

    @cute.kernel
    def kernel(
        self, tiled_mma, tma_atom_a, mA_mkl, tma_atom_b, mB_nkl, tma_atom_d, mD_mnl,
        tma_atom_c, mC_mnl, epilogue_params, varlen_params, cluster_layout_mnk,
        a_smem_layout, b_smem_layout, epi_smem_layout, epi_c_smem_layout,
        tile_sched_params, TileSchedulerCls: cutlass.Constexpr[Callable],
        trace_ptr: Optional[cutlass.Int64] = None,
    ):
        from quack.trace import TraceContext
        tctx = TraceContext.create(trace_ptr)
        varlen_m = const_expr(varlen_params.cu_seqlens_m is not None)
        varlen_k = const_expr(varlen_params.cu_seqlens_k is not None)
        has_D = const_expr(mD_mnl is not None)
        has_C = const_expr(mC_mnl is not None)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == self.ab_load_warp_id:
            for tma_atom in (tma_atom_a, tma_atom_b, tma_atom_d, tma_atom_c):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        ab_pipeline = self.make_ab_pipeline(
            tiled_mma=tiled_mma,
            cluster_layout_vmnk=cute.make_layout((1, *cluster_layout_mnk.shape)),
            ab_pipeline_mbar_ptr=storage.ab_pipeline_array_ptr.data_ptr(),
        )
        epi_pipeline = None
        if const_expr(has_C):
            epi_pipeline = self.make_epi_pipeline(
                c_smem_layout=cute.slice_(epi_c_smem_layout, (None, None, 0)),
                epi_pipeline_mbar_ptr=storage.epi_pipeline_array_ptr.data_ptr(),
            )
        sched_pipeline = None
        sched_data = None
        if const_expr(self.is_persistent):
            sched_pipeline = self.make_sched_pipeline(
                cluster_layout_mnk,
                sched_pipeline_mbar_ptr=storage.sched_pipeline_array_ptr.data_ptr(),
                varlen_k=varlen_k,
            )
            sched_data = storage.sched_data.get_tensor((4, self.sched_stage))
        if const_expr(_WS):
            # init the stats handshake mbarriers (Full[g], Empty[g]) as part of the main
            # barrier-init protocol: ONE thread per barrier (thread t inits slot t), then
            # fence; the cluster pipeline_init_wait below publishes them. (All 32 threads
            # of a warp calling mbarrier_init on the same barrier is illegal — re-init.)
            sStat_mbar = storage.sStatMbar.data_ptr()
            if cute.arch.thread_idx()[0] < 2 * self.mma_warp_groups:
                cute.arch.mbarrier_init(sStat_mbar + cute.arch.thread_idx()[0], 1)
            cute.arch.mbarrier_init_fence()
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk[:-1], is_relaxed=True)
        sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sD = None
        if const_expr(has_D):
            sD = storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)
        sC = None
        if const_expr(has_C):
            sC = storage.sC.get_tensor(epi_c_smem_layout.outer, swizzle=epi_c_smem_layout.inner)
        epi_smem_tensors = self.epi_get_smem_tensors(epilogue_params, storage)
        varlen_manager = VarlenManager.create(
            varlen_params,
            len_m_static=Int32(
                cute.size(mA_mkl, mode=[0])
                if varlen_k or varlen_params.mAIdx is None
                else varlen_params.mAIdx.shape[0]
            ),
            len_k_static=Int32(cute.size(mA_mkl, mode=[1])),
        )
        TileSchedulerCls = partial(
            TileSchedulerCls.create, tile_sched_params, sched_data, sched_pipeline
        )
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk[:-1])
        if const_expr(_WS):
            # Empty[g] is now live (published by pipeline_init_wait). Pre-arrive it (ONE
            # thread per barrier: thread g arrives Empty[g]) so the producer's first
            # acquire passes; the producer's wait acquires this cross-warp via the mbar.
            if cute.arch.thread_idx()[0] < self.mma_warp_groups:
                cute.arch.mbarrier_arrive(sStat_mbar + self.mma_warp_groups + cute.arch.thread_idx()[0])

        if warp_idx >= self.ab_load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            if (warp_idx >= self.ab_load_warp_id
                    and warp_idx < self.ab_load_warp_id + self.num_ab_load_warps):
                if const_expr(self.use_pdl):
                    cute.arch.griddepcontrol_wait()
                cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                block_in_cluster_coord_mnk = cluster_layout_mnk.get_flat_coord(cta_rank_in_cluster)
                a_mcast_mask = cute.make_layout_image_mask(cluster_layout_mnk, block_in_cluster_coord_mnk, mode=1)
                b_mcast_mask = cute.make_layout_image_mask(cluster_layout_mnk, block_in_cluster_coord_mnk, mode=0)
                a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
                b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0
                is_scheduler_warp = self.num_ab_load_warps == 1 or warp_idx == self.ab_load_warp_id
                if const_expr(cute.size(cluster_layout_mnk) > 1):
                    is_scheduler_warp = is_scheduler_warp and cute.arch.block_idx_in_cluster() == 0
                tile_scheduler = TileSchedulerCls()
                work_tile = tile_scheduler.initial_work_tile_info()
                ab_producer_state = make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stage)
                while work_tile.is_valid_tile:
                    tile_coord_mnkl = work_tile.tile_idx
                    batch_idx = tile_coord_mnkl[3]
                    mA_mk = varlen_manager.offset_batch_A(mA_mkl, batch_idx)
                    gA_mk = cute.local_tile(mA_mk, cute.select(self.cta_tile_shape_mnk, [0, 2]), (tile_coord_mnkl[0], None))
                    copy_A, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_a, cta_coord=block_in_cluster_coord_mnk[1],
                        cta_layout=cute.make_layout(cute.slice_(cluster_layout_mnk, (0, None, 0)).shape),
                        src_tensor=gA_mk, dst_tensor=sA, mcast_mask=a_mcast_mask,
                    )
                    gB_nk = cute.local_tile(varlen_manager.offset_batch_B(mB_nkl, batch_idx), cute.select(self.cta_tile_shape_mnk, [1, 2]), (tile_coord_mnkl[1], None))
                    copy_B, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_b, cta_coord=block_in_cluster_coord_mnk[0],
                        cta_layout=cute.make_layout(cute.slice_(cluster_layout_mnk, (None, 0, 0)).shape),
                        src_tensor=gB_nk, dst_tensor=sB, mcast_mask=b_mcast_mask,
                    )
                    len_k = varlen_manager.len_k(batch_idx)
                    k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                    ab_producer_state = self.load_AB(ab_pipeline, ab_producer_state, copy_A, copy_B, k_tile_cnt)
                    tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                    work_tile = tile_scheduler.get_current_work()
                if const_expr(self.pingpong and not varlen_k):
                    # pingpong: hand the next work-tile to the other WG via smem.
                    if is_scheduler_warp:
                        tile_scheduler.write_work_tile_to_smem(work_tile)
                    work_tile = tile_scheduler.get_current_work()
                if warp_idx == self.ab_load_warp_id:
                    ab_pipeline.producer_tail(ab_producer_state)
                if is_scheduler_warp:
                    tile_scheduler.producer_tail()
            elif const_expr(_WS_DEBUG or _WS):
                # STATS WARPS (the idle load-WG warps 9-11): independently replicate the
                # STATIC tile sequence (work_idx = cluster_idx, += grid.z; _delinearize
                # handles the swizzle) and reduce X[m-tile] from gmem ON THESE WARPS, in
                # parallel with the math WGs' WGMMA (no MMA-throughput theft, no sA
                # recycle race). _WS_DEBUG dumps to mDbg; _WS feeds the math WGs via the
                # per-WG single-stage handshake (Empty[g]/Full[g] mbarriers).
                BLK_M_s = const_expr(self.cta_tile_shape_mnk[0])
                N_STAT_WARPS = const_expr(
                    self.threads_per_cta // 32 - (self.ab_load_warp_id + self.num_ab_load_warps)
                )
                MWG = const_expr(self.mma_warp_groups)
                stidx = cute.arch.thread_idx()[0] - (self.ab_load_warp_id + self.num_ab_load_warps) * 32
                s_rstd = epi_smem_tensors[self._epi_smem_map["mRstd"]]
                s_c1 = epi_smem_tensors[self._epi_smem_map["mC1"]]
                s_sched = TileSchedulerCls()
                s_widx = s_sched._current_work_idx
                gz = Int32(cute.arch.grid_dim()[2])
                s_count = Int32(0)
                s_wt = s_sched._delinearize_work_idx(s_widx)
                while s_wt.is_valid_tile:
                    s_mc = s_wt.tile_idx
                    s_lenk = varlen_manager.len_k(s_mc[3])
                    if const_expr(_WS):
                        g = s_count % MWG               # which math WG owns this tile
                        ph = (s_count // MWG) % 2        # handshake phase for this g
                        cute.arch.mbarrier_wait(sStat_mbar + MWG + g, ph)   # Empty[g]
                        _reduce_gmem_coop(
                            epilogue_params.mX, s_mc[0] * BLK_M_s, s_rstd, s_c1, g * BLK_M_s,
                            stidx, s_lenk, Float32(s_lenk), epilogue_params.eps,
                            blk_m=BLK_M_s, nwarps=N_STAT_WARPS,
                        )
                        cute.arch.barrier(barrier_id=10, number_of_threads=N_STAT_WARPS * 32)
                        cute.arch.fence_view_async_shared()
                        if stidx == 0:
                            cute.arch.mbarrier_arrive(sStat_mbar + g)       # Full[g]
                    else:
                        _stats_dump_gmem(
                            epilogue_params.mX, epilogue_params.mDbg, s_mc[0] * BLK_M_s, stidx,
                            s_lenk, Float32(s_lenk), epilogue_params.eps,
                            blk_m=BLK_M_s, nwarps=N_STAT_WARPS,
                        )
                    s_count = s_count + Int32(1)
                    s_widx = s_widx + gz
                    s_wt = s_sched._delinearize_work_idx(s_widx)

        if warp_idx < self.ab_load_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_mma)
            is_tma_warp = Boolean(
                (not self.pingpong and warp_idx == 0)
                or (self.pingpong and (warp_idx == 0 or warp_idx == 4))
            )
            tidx, _, _ = cute.arch.thread_idx()
            warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
            if const_expr(self.pingpong):
                tidx = tidx % self.num_threads_per_warp_group
            warp_group_thread_layout = cute.make_layout(
                self.mma_warp_groups if const_expr(not self.pingpong) else 1,
                stride=self.num_threads_per_warp_group,
            )
            thr_mma = tiled_mma.get_slice(
                warp_group_thread_layout(warp_group_idx if not self.pingpong else 0)
            )
            acc, tCrA, tCrB = quack_sm90_utils.partition_fragment_ABC(thr_mma, self.cta_tile_shape_mnk, sA, sB)
            mma_fn = partial(quack_sm90_utils.gemm_w_idx, tiled_mma, acc, tCrA, tCrB)
            # fused LN reduction state
            BLK_M = const_expr(self.cta_tile_shape_mnk[0])
            BLK_K = const_expr(self.cta_tile_shape_mnk[2])
            TPWG = const_expr(self.num_threads_per_warp_group)
            NT = const_expr(self.mma_warp_groups * TPWG)
            # warps cooperating on the LN reduction: pingpong -> each WG reduces its
            # OWN tile alone (TPWG/32 warps, local tidx); non-pingpong -> both WGs
            # share one tile (NT/32 warps, global tidx).
            STAT_WARPS = const_expr((TPWG // 32) if self.pingpong else (NT // 32))

            if const_expr(self.pingpong):
                if warp_group_idx == 0:
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="mma")
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="epi")

            k_tile_cnt_static = cute.ceil_div(cute.size(mA_mkl, mode=[1]), self.cta_tile_shape_mnk[2])
            c_tile_cnt = cute.size(cute.ceil_div(self.cta_tile_shape_mnk[:2], self.epi_tile))
            ab_read_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            epi_store_pipeline = self.make_epi_store_pipeline()
            epi_read_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.epi_c_stage)
            epi_producer_state = make_pipeline_state(pipeline.PipelineUserType.Producer, self.epi_c_stage)
            # SmemColVec buffers, sized [2*BLK_M]: each WG fills/reads its own half
            # (offset warp_group_idx*BLK_M) since the two WGs run different tiles whose
            # mma/epilogue overlap.
            s_rstd = epi_smem_tensors[self._epi_smem_map["mRstd"]]
            s_c1 = epi_smem_tensors[self._epi_smem_map["mC1"]]
            wg_off = (warp_group_idx * BLK_M) if const_expr(self.pingpong) else Int32(0)
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            # LN stats are identical for every n-tile of a given m-tile, so we reduce
            # ONCE per m-tile and reuse the s_rstd/s_c1 smem for that m-tile's other
            # n-tiles (the AlongN raster gives a WG consecutive same-m tiles). prev_m
            # tracks the last m-tile this WG reduced.
            prev_m = Int32(-1)
            # warp-specialized consumer handshake: this WG's per-tile phase (mbar ptr
            # = the hoisted sStat_mbar from the init block, reused to avoid touching the
            # SharedStorage object inside a flattened dynamic-if branch).
            ws_phase = Int32(0)
            if const_expr(self.pingpong):
                if warp_idx >= 4:
                    epi_read_state.advance_iters(c_tile_cnt)
                    epi_producer_state.advance_iters(c_tile_cnt)
                    ab_read_state.advance_iters(k_tile_cnt_static)
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
            while work_tile.is_valid_tile:
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx = tile_coord_mnkl[3]
                len_k = varlen_manager.len_k(batch_idx)
                k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                len_k_f = Float32(len_k)
                m_tile = tile_coord_mnkl[0]
                do_reduce = Boolean(True)  # TODO reuse disabled (was buggy at BLK_N>128)
                if const_expr(_WS):
                    # WARP-SPECIALIZED: stats are produced by the idle load-WG warps and
                    # land in s_rstd/s_c1[wg half] via the Full[g] mbarrier. The math WG
                    # does a PLAIN GEMM (full WGMMA pipelining, no reduction stealing
                    # throughput) and just waits for its stats before the epilogue.
                    self.pingpong_barrier_sync(warp_group_idx, stage="mma")
                    ab_read_state = self.mma(
                        ab_pipeline, ab_read_state, mma_fn, acc, None, k_tile_cnt, warp_group_idx,
                    )
                    cute.arch.mbarrier_wait(sStat_mbar + warp_group_idx, ws_phase)  # Full[g]
                    self.pingpong_barrier_sync(warp_group_idx, stage="epi")
                elif const_expr(self.pingpong):
                    # --- LN stats from the GEMM's sA smem (already loaded by the TMA, so
                    # NO extra gmem traffic) reduced ON CUDA CORES in parallel with the
                    # WGMMA. pingpong: each WG runs a full tile, thread t owns row t, so
                    # red_sum is per-thread and every thread writes its OWN row (converged
                    # -> no divergent-barrier deadlock). Skipped (reused) when this m-tile
                    # was already reduced on a previous n-tile. ---
                    red_sum = cute.make_rmem_tensor(1, Float32)
                    red_sumsq = cute.make_rmem_tensor(1, Float32)
                    red_sum[0] = Float32(0.0)
                    red_sumsq[0] = Float32(0.0)
                    self.pingpong_barrier_sync(warp_group_idx, stage="mma")
                    ab_read_state = self.mma(
                        ab_pipeline, ab_read_state, mma_fn, acc, None, k_tile_cnt, warp_group_idx,
                        sA=sA, red_sum=red_sum, red_sumsq=red_sumsq, tidx=tidx,
                        blk_k=BLK_K, wg_rows=BLK_M, tpwg=TPWG, do_reduce=do_reduce,
                    )
                    if do_reduce:
                        mean = red_sum[0] / len_k_f
                        var = red_sumsq[0] / len_k_f - mean * mean
                        rstd = cute.math.rsqrt(var + epilogue_params.eps, fastmath=True)
                        s_rstd[wg_off + tidx] = rstd
                        s_c1[wg_off + tidx] = mean * rstd
                        prev_m = m_tile
                    self.pingpong_barrier_sync(warp_group_idx, stage="epi")
                else:
                    # non-pingpong fallback: coalesced gmem reduction (re-reads X) +
                    # CTA-wide publish barrier.
                    m_base = tile_coord_mnkl[0] * BLK_M
                    _reduce_gmem_coop(
                        epilogue_params.mX, m_base, s_rstd, s_c1, wg_off, tidx, len_k, len_k_f,
                        epilogue_params.eps, blk_m=BLK_M, nwarps=STAT_WARPS,
                    )
                    cute.arch.barrier(barrier_id=8, number_of_threads=NT)
                    ab_read_state = self.mma(
                        ab_pipeline, ab_read_state, mma_fn, acc, None, k_tile_cnt, warp_group_idx,
                    )

                copy_D = None
                if const_expr(has_D):
                    copy_D, _, _ = self.epilog_gmem_copy_and_partition(
                        tma_atom_d, varlen_manager.offset_batch_epi(mD_mnl, batch_idx),
                        self.cta_tile_shape_mnk[:2], self.epi_tile, sD, tile_coord_mnkl,
                    )
                d_dtype_for_layout = self.d_dtype if self.d_dtype is not None else cutlass.BFloat16
                tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
                    tiled_mma, self.d_layout, d_dtype_for_layout, sD, tidx
                )
                tRS_rAcc = self.epi_retile_acc(acc, tRS_rD, tiled_copy_r2s)
                load_acc_subtile = partial(self.epi_load_acc_subtile, tRS_rAcc)
                epi_read_state, epi_producer_state = self.epilogue(
                    epilogue_params, epi_smem_tensors, epi_pipeline, epi_store_pipeline,
                    epi_read_state, epi_producer_state, self.epi_tile, load_acc_subtile,
                    tRS_rD, None, None, tiled_copy_r2s, tRS_sD, None, None, None, copy_D, None,
                    tile_coord_mnkl, varlen_manager, self.epilogue_barrier, tile_scheduler, tidx, is_tma_warp,
                )
                if const_expr(_WS):
                    # epilogue done reading s_rstd[wg half] -> free the stage (Empty[g])
                    # for the producer's next tile; advance this WG's handshake phase.
                    if tidx == 0:
                        cute.arch.mbarrier_arrive(sStat_mbar + self.mma_warp_groups + warp_group_idx)
                    ws_phase = ws_phase ^ Int32(1)
                if const_expr(self.pingpong):
                    if is_tma_warp:
                        epi_store_pipeline.producer_tail()
                    self.pingpong_barrier_arrive(1 - warp_group_idx, stage="epi")
                    epi_read_state.advance_iters(c_tile_cnt)
                    epi_producer_state.advance_iters(c_tile_cnt)
                    ab_read_state.advance_iters(k_tile_cnt_static)
                    tile_scheduler.advance_to_next_work(advance_count=self.mma_warp_groups)
                    work_tile = tile_scheduler.get_current_work()
                else:
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()
            if const_expr(not self.pingpong):
                if is_tma_warp:
                    epi_store_pipeline.producer_tail()
        tctx.flush()

    @cute.jit
    def __call__(self, mA, mB, mD, mC, epilogue_args, scheduler_args, varlen_args,
                 stream, trace_ptr=None):
        # Faithful copy of GemmSm90.__call__ + a [2*BLK_M] FP32 sStats smem field.
        mA, mB, mD, mC = [
            layout_utils.concat_to_interleave(mT, 1 - mT.leading_dim)
            if const_expr(name in self.concat_layout and mT is not None) else mT
            for name, mT in [("A", mA), ("B", mB), ("out", mD), ("C", mC)]
        ]
        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.d_dtype = mD.element_type if mD is not None else None
        self.c_dtype = mC.element_type if mC is not None else None
        self.a_layout = LayoutEnum.from_tensor(mA)
        self.b_layout = LayoutEnum.from_tensor(mB)
        self.d_layout = LayoutEnum.from_tensor(mD) if mD is not None else None
        self.c_layout = LayoutEnum.from_tensor(mC) if mC is not None else None
        if const_expr(varlen_args is None):
            varlen_args = VarlenArguments()
        varlen_m = varlen_args.mCuSeqlensM is not None
        varlen_k = varlen_args.mCuSeqlensK is not None
        self._setup_attributes(epilogue_args)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            mA, a_smem_layout, (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[2]),
            self.cluster_shape_mnk[1],
        )
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            mB, b_smem_layout, (self.cta_tile_shape_mnk[1], self.cta_tile_shape_mnk[2]),
            self.cluster_shape_mnk[0],
        )
        self.num_tma_load_bytes = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes += cute.size_in_bytes(self.a_dtype, a_smem_layout)
        tma_atom_d, tma_tensor_d = None, None
        if const_expr(mD is not None):
            tma_atom_d, tma_tensor_d = self._make_tma_epi_atoms_and_tensors(
                mD, self.epi_smem_layout_staged, self.epi_tile, op_type="store",
            )
        tma_atom_c, tma_tensor_c = None, None
        epilogue_params = self.epi_to_underlying_arguments(epilogue_args)
        varlen_params = VarlenManager.to_underlying_arguments(varlen_args)
        TileSchedulerCls = self.get_scheduler_class(varlen_m=varlen_m)
        tile_sched_args = self.get_scheduler_arguments(mA, mB, mD, scheduler_args, varlen_args, epilogue_args)
        tile_sched_params = TileSchedulerCls.to_underlying_arguments(tile_sched_args)
        grid = TileSchedulerCls.get_grid_shape(tile_sched_params, scheduler_args.max_active_clusters)
        epi_smem_size = cute.cosize(self.epi_smem_layout_staged) if mD is not None else 0
        epi_c_smem_size = cute.cosize(self.epi_c_smem_layout_staged) if mC is not None else 0
        blk_m = self.cta_tile_shape_mnk[0]

        @cute.struct
        class SharedStorage:
            ab_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            epi_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.epi_c_stage * 2]
            sched_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.sched_stage * 2]
            sched_data: cute.struct.MemRange[Int32, self.sched_stage * 4]
            # warp-specialized stats: 2 mbarriers per math WG (Full, Empty) for the
            # single-stage producer(stats warps)->consumer(math WG) handshake.
            sStatMbar: cute.struct.MemRange[cutlass.Int64, 2 * self.mma_warp_groups]
            sD: cute.struct.Align[
                cute.struct.MemRange[self.d_dtype if self.d_dtype is not None else Int32, epi_smem_size],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype if self.c_dtype is not None else Int32, epi_c_smem_size],
                self.buffer_align_bytes,
            ]
            epi: self.epi_get_smem_struct(epilogue_params)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            self.tiled_mma, tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b,
            tma_atom_d, tma_tensor_d, tma_atom_c, tma_tensor_c, epilogue_params,
            varlen_params, self.cluster_layout_mnk, self.a_smem_layout_staged,
            self.b_smem_layout_staged, self.epi_smem_layout_staged,
            self.epi_c_smem_layout_staged, tile_sched_params, TileSchedulerCls, trace_ptr,
        ).launch(
            grid=grid, block=[self.threads_per_cta, 1, 1], cluster=self.cluster_shape_mnk,
            stream=stream, min_blocks_per_mp=1, use_pdl=self.use_pdl,
        )
        return


# ---------------------------------------------------------------------------
# Launch wrapper + user-facing entry.
# ---------------------------------------------------------------------------

# SHIPPING CONFIG: PINGPONG + PERSISTENT (quack's proven SM90 path). Each math WG runs a
# full output tile by itself (leapfrog), so its mma overlaps the other WG's epilogue. LN
# stats are reduced from the GEMM's sA smem ON CUDA CORES during the WGMMA (no extra gmem
# traffic), thread t owning row t and writing its OWN row of the [2*BLK_M] per-WG s_rstd/
# s_c1 smem (converged write — an `if lane==0` write deadlocked the pingpong named
# barrier with "Divergent thread(s) in warp"; all-lanes / own-row writes avoid it).
# Correct cos=0.999997 on all 18 shapes; FASTEST at small/mid d (d<=256: beats M1 +
# torch.compile + TE, e.g. d=128 M=262144 0.060ms vs M1 0.094). Still ~2x behind M1 at
# large d (512/768) — the per-k-tile sA reduce + sync_warp competes with the WGMMA and
# red_sum register pressure spills the 128-row acc; that's the open optimization.
# (non-pingpong+persistent is a quack acc-reuse hazard; non-pingpong+NON-persistent also
# works via the gmem-reduction fallback below but loses everywhere — see git history.)
_FUSED_CONFIG = dict(tile_m=128, tile_n=128, cluster_m=1, cluster_n=1, pingpong=True)


@jit_cache
def _compile_fused(a_dtype, b_dtype, d_dtype, a_major, b_major, d_major, vec_dtype, device_capacity):
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype, b_dtype, d_dtype, None, a_major, b_major, d_major, None
    )
    mS = fake_tensor(vec_dtype, (l, n), leading_dim=1, divisibility=4)
    mB2 = fake_tensor(vec_dtype, (l, n), leading_dim=1, divisibility=4)
    mX = fake_tensor(a_dtype, (m, k), leading_dim=1, divisibility=8)  # plain (M,K) X
    # mDbg only exists in the warp-specialized bring-up; gated so the shipping signature
    # has None at this slot (else the runtime None mismatches a compiled Tensor arg).
    mDbg = fake_tensor(Float32, (m,), leading_dim=0, divisibility=4) if _WS_DEBUG else None
    epi_args = GemmLNLFusedSm90.EpilogueArguments(mS=mS, mB2=mB2, eps=Float32(1e-5), mX=mX, mDbg=mDbg)
    scheduler_args = make_fake_scheduler_args(False, False, l)
    varlen_args = make_fake_varlen_args(False, False, False, None)
    return compile_gemm_kernel(
        GemmLNLFusedSm90, a_dtype,
        (_FUSED_CONFIG["tile_m"], _FUSED_CONFIG["tile_n"]),
        (_FUSED_CONFIG["cluster_m"], _FUSED_CONFIG["cluster_n"], 1),
        _FUSED_CONFIG["pingpong"], _FUSED_CONFIG["pingpong"], False, False, device_capacity,  # persistent tied to pingpong
        mA, mB, mD, mC, epi_args, scheduler_args, varlen_args,
    )


def gemm_lnl_fused(A, B, D, S, B2, eps: float = 1e-5, mDbg=None):
    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] == 9, "SM90 (H100) only"
    A3, B3, D3 = A.unsqueeze(0), B.unsqueeze(0), D.unsqueeze(0)
    A_p, B_p, D_p, _ = perm3d(A3, B3, D3, None)
    a_major, b_major, d_major, _ = get_majors(A_p, B_p, D_p, None)
    a_dtype, b_dtype, d_dtype, _ = get_dtypes(A, B, D, None)
    vec_dtype = torch2cute_dtype_map[S.dtype]
    compiled_fn = _compile_fused(a_dtype, b_dtype, d_dtype, a_major, b_major, d_major, vec_dtype, device_capacity)
    from quack.cache_utils import COMPILE_ONLY
    if COMPILE_ONLY:
        return
    max_active_clusters = get_max_active_clusters(1)
    epi_args = GemmLNLFusedSm90.EpilogueArguments(
        mS=S, mB2=B2, eps=Float32(eps), mX=A, mDbg=mDbg,
        add_to_output=None, rounding_mode=None,
    )
    # NOTE on the prev_m per-m-tile stats reuse in kernel(): it only triggers when a WG
    # gets consecutive same-m tiles, which the PERSISTENT scheduler does NOT provide (it
    # strides tiles across CTAs, so an m-tile's n-tiles land on different CTAs). So the
    # reuse is currently a no-op and the reduction runs once per (m,n) tile. Eliminating
    # that redundancy needs cross-CTA sharing (gmem + sync) — left as future work. swizzle
    # kept at the GEMM default (8) for L2 reuse (disabling it gave no speedup here).
    scheduler_args = make_scheduler_args(max_active_clusters, 8, None)
    varlen_args = make_varlen_args(None, None, None)
    compiled_fn(A_p, B_p, D_p, None, epi_args, scheduler_args, varlen_args, None)


def layernorm_linear_cute_fused(x, ln_weight, ln_bias, weight, bias, eps: float = 1e-5, *, prefolded=None):
    """Fused forward LayerNormLinear — stats computed inside the GEMM (Milestone 2)."""
    from .gemm_layernorm_linear import fold_for_gemm
    assert x.is_cuda and x.dim() == 2
    M, K = x.shape
    N = weight.shape[0]
    Bw, S, B2 = prefolded if prefolded is not None else fold_for_gemm(
        weight, ln_weight, ln_bias, bias, w2_dtype=x.dtype
    )
    S2 = S.float().contiguous().view(1, N)
    B22 = B2.float().contiguous().view(1, N)
    Y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    gemm_lnl_fused(x, Bw, Y, S2, B22, eps)
    return Y
