"""TRUE fused back-to-back (b2b) transition forward for sm100 (B200), cutlass.cute DSL.

Fused, precomputed-normalized input (LN done upstream):
    a  = xn @ wa^T            xn:(M,K) bf16, wa:(ND,K) bf16   -> a:(M,ND)
    b  = xn @ wb^T            wb:(ND,K) bf16                   -> b:(M,ND)
    h  = silu(a) * b          = a*sigmoid(a)*b  (fp32 -> bf16)  (M,ND)
    out = h @ ws^T            ws:(D,ND) bf16                   -> out:(M,D) bf16

A-from-TMEM squeeze (Blackwell analog of H100's register-source RS squeeze):
h is NOT round-tripped through shared memory. The epilogue computes silu(a)*b in
registers and stores it straight back into TMEM (r2t, tcgen05.st), and the squeeze
MMA reads its A operand directly from TMEM (a_source=OperandSource.TMEM). This keeps
h on-chip with no smem bank-conflict store path and takes the epilogue off the
critical smem-dependency chain that capped the smem-round-trip version at ~16% SM.

Persistent over M-tiles ONLY. Warp specialization: epilogue warps (0..3), mma warp (4),
tma warp (5). Expand K streamed in KT tiles so wa/wb/ws fit in smem.

TMEM budget (512 cols): a_acc(BN) + b_acc(BN) + o_acc(D) + h(BN bf16 ~BN/2 cols).
BN=64 -> d=128: 64+64+128+~32=288; d=256: 64+64+256+~32=416. Both fit (single-stage).
"""

from __future__ import annotations

import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch
from quack.cute_dsl_utils import get_max_active_clusters
from cutlass import Float32, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.blackwell_helpers import get_tmem_load_op, get_smem_store_op, make_smem_layout
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout, epilogue_tmem_copy_and_partition

from miniworld_kernels.kernels.tm1.cute._blackwell_dense_gemm import (
    PersistentDenseGemmKernel,
)


class TransitionB2BFusedKernel(PersistentDenseGemmKernel):
    """Fused expand+SwiGLU+squeeze b2b GEMM, persistent over M-tiles, A-from-TMEM squeeze."""

    def __init__(self, cta_m=128, bn=64, kt=128):
        self.acc_dtype = Float32
        self.use_2cta_instrs = False
        self.cluster_shape_mn = (1, 1)
        self.use_tma_store = False
        self.arch = "sm_100"
        self.cta_group = tcgen05.CtaGroup.ONE
        self.occupancy = 1
        self.epilogue_warp_id = (0, 1, 2, 3)   # silu group 0
        self.silu1_warp_id = (4, 5, 6, 7)      # silu group 1 (ping-pong)
        self.mma_warp_id = 8
        self.tma_warp_id = 9
        self.threads_per_cta = 32 * 10
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3
        self.phase_sync_bar_id = 4
        self.silu1_sync_bar_id = 6
        self.CTA_M = cta_m
        self.BN = bn
        self.KT = kt
        self.K = None
        self.ND = None
        self.D = None
        # Stage depths; decided/TMEM-clamped by the wrapper before compile.
        self.num_acc_stage = 2
        self.num_h_stage = 2
        self.num_out_stage = 1
        self.defer = False
        self.pingpong = False

    def _tiled_mma(self, mn):
        return utils.sm100.make_trivial_tiled_mma(
            self.ab_dtype, self.a_major_mode, self.b_major_mode,
            self.acc_dtype, self.cta_group, mn,
        )

    def _tiled_mma_squeeze(self, mn):
        # Squeeze reads A (=h) from TMEM. h contraction dim is BN (the ND chunk),
        # laid out K-major in TMEM.
        return utils.sm100.make_trivial_tiled_mma(
            self.ab_dtype, tcgen05.OperandMajorMode.K, self.b_major_mode,
            self.acc_dtype, self.cta_group, mn,
            a_source=tcgen05.OperandSource.TMEM,
        )

    def _setup(self):
        K, ND, D, BN, KT, CTA_M = self.K, self.ND, self.D, self.BN, self.KT, self.CTA_M
        self.NCHUNK = ND // BN
        self.n_kt = K // KT
        self.mma_expand = self._tiled_mma((CTA_M, BN))
        self.mma_squeeze = self._tiled_mma_squeeze((CTA_M, D))
        self.xn_layout = utils.sm100.make_smem_layout_a(
            self.mma_expand, (CTA_M, BN, K), self.ab_dtype, 1)
        self.wab_layout = utils.sm100.make_smem_layout_b(
            self.mma_expand, (CTA_M, BN, KT), self.ab_dtype, self.num_wab_stage)
        self.ws_layout = utils.sm100.make_smem_layout_b(
            self.mma_squeeze, (CTA_M, D, BN), self.ab_dtype, self.num_ws_stage)
        # full-tile epilogue (no subtiling): whole (CTA_M, BN) h chunk per pass.
        # Must be a proper cute tiler (layouts), matching compute_epilogue_tile_shape
        # format, so product_each/flat_divide collapse correctly (sub_h == 1).
        self.epi_tile_h = (cute.make_layout(CTA_M), cute.make_layout(BN))
        self.epi_tile_out = utils.sm100.compute_epilogue_tile_shape(
            (CTA_M, D, BN), self.use_2cta_instrs, self.c_layout, self.c_dtype)
        self.cta_tile_shape_mnk = (CTA_M, D, BN)

    @cute.jit
    def __call__(self, xn, wa, wb, ws, out, max_active_clusters, stream):
        xn = cute.make_tensor(xn.iterator, cute.select(xn.layout, mode=[1, 2, 0]))   # (M,K,L)
        wa = cute.make_tensor(wa.iterator, cute.select(wa.layout, mode=[1, 2, 0]))   # (ND,K,L)
        wb = cute.make_tensor(wb.iterator, cute.select(wb.layout, mode=[1, 2, 0]))   # (ND,K,L)
        ws = cute.make_tensor(ws.iterator, cute.select(ws.layout, mode=[1, 2, 0]))   # (D,ND,L)
        out = cute.make_tensor(out.iterator, cute.select(out.layout, mode=[1, 2, 0]))  # (M,D,L)

        self.ab_dtype = xn.element_type
        self.c_dtype = out.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(xn).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(wa).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(out)

        _e = self.ab_dtype.width // 8
        _fixed = self.CTA_M * self.K * _e   # sXn only (h lives in TMEM now)
        _pw = 2 * self.BN * self.KT * _e   # sWa + sWb per stage
        _psw = self.D * self.BN * _e       # sWs per stage
        _budget = 227 * 1024 - 4096
        self.num_wab_stage = 3 if _fixed + 3 * _pw + 2 * _psw <= _budget else (
            2 if _fixed + 2 * _pw + _psw <= _budget else 1)
        self.num_ws_stage = 2 if _fixed + self.num_wab_stage * _pw + 2 * _psw <= _budget else 1
        # num_acc_stage / num_h_stage are decided (and TMEM-clamped) in the wrapper
        # (host Python) and set as attributes on the op BEFORE cute.compile, so they
        # are plain host ints here. num_out_stage stays 1.
        # TMEM budget (512 cols, fp32-col units):
        #   a_acc + b_acc = 2*num_acc_stage*BN ; o_acc = D ; h = num_h_stage*(BN/2).
        # Deeper acc lookahead hides the squeeze's tcgen05.ld (t2r) latency by keeping
        # num_acc_stage-1 independent expand chunks in flight in the MMA warp.
        self.num_out_stage = 1
        self._setup()

        mma_e, mma_s = self.mma_expand, self.mma_squeeze
        cl_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (mma_e.thr_id.shape,))

        a_op = utils.sm100.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, mma_e.thr_id)
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, mma_e.thr_id)
        bs_op = utils.sm100.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, mma_s.thr_id)

        xn_sl = cute.slice_(self.xn_layout, (None, None, None, 0))
        tma_xn, txn = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, xn, xn_sl, (self.CTA_M, self.BN, self.K), mma_e, cl_vmnk.shape)
        wab_sl = cute.slice_(self.wab_layout, (None, None, None, 0))
        tma_wa, twa = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, wa, wab_sl, (self.CTA_M, self.BN, self.KT), mma_e, cl_vmnk.shape)
        tma_wb, twb = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, wb, wab_sl, (self.CTA_M, self.BN, self.KT), mma_e, cl_vmnk.shape)
        ws_sl = cute.slice_(self.ws_layout, (None, None, None, 0))
        tma_ws, tws = cute.nvgpu.make_tiled_tma_atom_B(
            bs_op, ws, ws_sl, (self.CTA_M, self.D, self.BN), mma_s, cl_vmnk.shape)

        tile_sched_params, grid = self._compute_grid(
            out, self.cta_tile_shape_mnk, self.cluster_shape_mn, max_active_clusters)

        self.kernel(
            mma_e, mma_s, cl_vmnk,
            tma_xn, txn, tma_wa, twa, tma_wb, twb, tma_ws, tws, out,
            self.xn_layout, self.wab_layout, self.ws_layout,
            self.epi_tile_h, self.epi_tile_out,
            tile_sched_params,
        ).launch(
            grid=grid, block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1), stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mma_e: cute.TiledMma, mma_s: cute.TiledMma, cl_vmnk: cute.Layout,
        tma_xn: cute.CopyAtom, mXn: cute.Tensor,
        tma_wa: cute.CopyAtom, mWa: cute.Tensor,
        tma_wb: cute.CopyAtom, mWb: cute.Tensor,
        tma_ws: cute.CopyAtom, mWs: cute.Tensor,
        mOut: cute.Tensor,
        xn_layout, wab_layout, ws_layout,
        epi_tile_h: cute.Tile, epi_tile_out: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_xn)
            cpasync.prefetch_descriptor(tma_wa)
            cpasync.prefetch_descriptor(tma_wb)
            cpasync.prefetch_descriptor(tma_ws)

        tidx, _, _ = cute.arch.thread_idx()
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        blk = cl_vmnk.get_flat_coord(cta_rank)

        CTA_M, BN, KT, D, K = self.CTA_M, self.BN, self.KT, self.D, self.K
        NCHUNK, n_kt = self.NCHUNK, self.n_kt
        _wb = self.ab_dtype.width // 8
        xn_bytes = self.CTA_M * self.K * _wb
        wab_bytes = self.BN * self.KT * _wb
        ws_bytes = self.D * self.BN * _wb

        @cute.struct
        class Shared:
            xn_full: cute.struct.MemRange[cutlass.Int64, 2]
            wab_full: cute.struct.MemRange[cutlass.Int64, self.num_wab_stage * 2]
            ws_full: cute.struct.MemRange[cutlass.Int64, self.num_ws_stage * 2]
            acc_full: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            h_full: cute.struct.MemRange[cutlass.Int64, self.num_h_stage * 2]
            acc_full0: cute.struct.MemRange[cutlass.Int64, 2]
            acc_full1: cute.struct.MemRange[cutlass.Int64, 2]
            h_full0: cute.struct.MemRange[cutlass.Int64, 2]
            h_full1: cute.struct.MemRange[cutlass.Int64, 2]
            out_full: cute.struct.MemRange[cutlass.Int64, self.num_out_stage * 2]
            tmem_dealloc_mbar: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: cute.struct.MemRange[cutlass.Int32, 1]

        smem = utils.SmemAllocator()
        storage = smem.allocate(Shared)

        thr_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        one_cons = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        epi_cons = pipeline.CooperativeGroup(pipeline.Agent.Thread, len(self.epilogue_warp_id))
        epi_prod = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 32 * len(self.epilogue_warp_id))

        xn_pipe = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.xn_full.data_ptr(), num_stages=1,
            producer_group=thr_grp, consumer_group=one_cons,
            tx_count=xn_bytes, cta_layout_vmnk=None, defer_sync=True)
        wab_pipe = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.wab_full.data_ptr(), num_stages=self.num_wab_stage,
            producer_group=thr_grp, consumer_group=one_cons,
            tx_count=2 * wab_bytes, cta_layout_vmnk=None, defer_sync=True)
        ws_pipe = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ws_full.data_ptr(), num_stages=self.num_ws_stage,
            producer_group=thr_grp, consumer_group=one_cons,
            tx_count=ws_bytes, cta_layout_vmnk=None, defer_sync=True)
        acc_pipe = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full.data_ptr(), num_stages=self.num_acc_stage,
            producer_group=thr_grp, consumer_group=epi_cons,
            cta_layout_vmnk=None, defer_sync=True)
        out_pipe = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.out_full.data_ptr(), num_stages=self.num_out_stage,
            producer_group=thr_grp, consumer_group=epi_cons,
            cta_layout_vmnk=None, defer_sync=True)
        h_pipe = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.h_full.data_ptr(), num_stages=self.num_h_stage,
            producer_group=epi_prod, consumer_group=one_cons,
            cta_layout_vmnk=None, defer_sync=True)
        acc_pipe0 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full0.data_ptr(), num_stages=1,
            producer_group=thr_grp, consumer_group=epi_cons,
            cta_layout_vmnk=None, defer_sync=True)
        acc_pipe1 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full1.data_ptr(), num_stages=1,
            producer_group=thr_grp, consumer_group=epi_cons,
            cta_layout_vmnk=None, defer_sync=True)
        h_pipe0 = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.h_full0.data_ptr(), num_stages=1,
            producer_group=epi_prod, consumer_group=one_cons,
            cta_layout_vmnk=None, defer_sync=True)
        h_pipe1 = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.h_full1.data_ptr(), num_stages=1,
            producer_group=epi_prod, consumer_group=one_cons,
            cta_layout_vmnk=None, defer_sync=True)

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * (len((self.mma_warp_id, *self.epilogue_warp_id))
                              + (len(self.silu1_warp_id) if self.pingpong else 0)))
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.data_ptr(),
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0], is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.data_ptr())

        pipeline_init_arrive(cluster_shape_mn=cl_vmnk, is_relaxed=True)

        sXn = smem.allocate_tensor(self.ab_dtype, xn_layout.outer, 128, xn_layout.inner)
        sWa = smem.allocate_tensor(self.ab_dtype, wab_layout.outer, 128, wab_layout.inner)
        sWb = smem.allocate_tensor(self.ab_dtype, wab_layout.outer, 128, wab_layout.inner)
        sWs = smem.allocate_tensor(self.ab_dtype, ws_layout.outer, 128, ws_layout.inner)

        gXn = cute.local_tile(mXn, cute.slice_((CTA_M, BN, K), (None, 0, None)), (None, None, None))
        gWa = cute.local_tile(mWa, cute.slice_((CTA_M, BN, KT), (0, None, None)), (None, None, None))
        gWb = cute.local_tile(mWb, cute.slice_((CTA_M, BN, KT), (0, None, None)), (None, None, None))
        gWs = cute.local_tile(mWs, cute.slice_((CTA_M, D, BN), (0, None, None)), (None, None, None))
        gOut = cute.local_tile(mOut, cute.slice_((CTA_M, D, BN), (None, None, 0)), (None, None, None))

        thr_e = mma_e.get_slice(0)
        thr_s = mma_s.get_slice(0)
        tCgXn = thr_e.partition_A(gXn)
        tCgWa = thr_e.partition_B(gWa)
        tCgWb = thr_e.partition_B(gWb)
        tCgWs = thr_s.partition_B(gWs)

        a_cta = cute.make_layout(cute.slice_(cl_vmnk, (0, 0, None, 0)).shape)
        b_cta = cute.make_layout(cute.slice_(cl_vmnk, (0, None, 0, 0)).shape)
        tXsX, tXgX = cpasync.tma_partition(
            tma_xn, blk[2], a_cta, cute.group_modes(sXn, 0, 3), cute.group_modes(tCgXn, 0, 3))
        tWasWa, tWagWa = cpasync.tma_partition(
            tma_wa, blk[1], b_cta, cute.group_modes(sWa, 0, 3), cute.group_modes(tCgWa, 0, 3))
        tWbsWb, tWbgWb = cpasync.tma_partition(
            tma_wb, blk[1], b_cta, cute.group_modes(sWb, 0, 3), cute.group_modes(tCgWb, 0, 3))
        tWssWs, tWsgWs = cpasync.tma_partition(
            tma_ws, blk[1], b_cta, cute.group_modes(sWs, 0, 3), cute.group_modes(tCgWs, 0, 3))

        tCrXn = mma_e.make_fragment_A(sXn)
        tCrWa = mma_e.make_fragment_B(sWa)
        tCrWb = mma_e.make_fragment_B(sWb)
        tCrWs = mma_s.make_fragment_B(sWs)

        acc_e_shape = mma_e.partition_shape_C((CTA_M, BN))
        acc_s_shape = mma_s.partition_shape_C((CTA_M, D))
        a_sq_shape = mma_s.partition_shape_A((CTA_M, BN))
        tAcc_e_fake = mma_e.make_fragment_C(cute.append(acc_e_shape, self.num_acc_stage))
        tAcc_s_fake = mma_s.make_fragment_C(cute.append(acc_s_shape, 1))
        tH_fake = mma_s.make_fragment_A(cute.append(a_sq_shape, self.num_h_stage))

        kb_kt = KT // 16
        kb_bn = BN // 16

        pipeline_init_wait(cluster_shape_mn=cl_vmnk)

        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
        work_tile = tile_sched.initial_work_tile_info()

        # ============================ TMA warp ============================
        if warp_idx == self.tma_warp_id:
            xn_ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            wab_ps = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_wab_stage)
            ws_ps = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ws_stage)
            while work_tile.is_valid_tile:
                m = work_tile.tile_idx[0]
                xn_pipe.producer_acquire(xn_ps)
                cute.copy(tma_xn, tXgX[(None, m, 0, 0)], tXsX[(None, xn_ps.index)],
                          tma_bar_ptr=xn_pipe.producer_get_barrier(xn_ps))
                xn_ps.advance()
                if const_expr(self.pingpong):
                    # ping-pong: the MMA interleaves expand(2 chunks ahead) with
                    # squeeze(2 current chunks), so the TMA MUST match that exact order
                    # (prologue: wab chunk0,1; then per pair: wab chunk cp+2,cp+3 then
                    # ws chunk cp,cp+1). Emitting all wab then all ws deadlocks because
                    # the MMA blocks on ws(chunk0) after only expanding chunks 0-3 while
                    # the TMA still owes wab 4-7 and won't emit ws until they're done.
                    for c in range(2):
                        for kt in range(n_kt):
                            wab_pipe.producer_acquire(wab_ps)
                            bar = wab_pipe.producer_get_barrier(wab_ps)
                            cute.copy(tma_wa, tWagWa[(None, c, kt, 0)],
                                      tWasWa[(None, wab_ps.index)], tma_bar_ptr=bar)
                            cute.copy(tma_wb, tWbgWb[(None, c, kt, 0)],
                                      tWbsWb[(None, wab_ps.index)], tma_bar_ptr=bar)
                            wab_ps.advance()
                    for cp in range(0, NCHUNK, 2):
                        if cp + 2 < NCHUNK:
                            for c in range(cp + 2, cp + 4):
                                for kt in range(n_kt):
                                    wab_pipe.producer_acquire(wab_ps)
                                    bar = wab_pipe.producer_get_barrier(wab_ps)
                                    cute.copy(tma_wa, tWagWa[(None, c, kt, 0)],
                                              tWasWa[(None, wab_ps.index)], tma_bar_ptr=bar)
                                    cute.copy(tma_wb, tWbgWb[(None, c, kt, 0)],
                                              tWbsWb[(None, wab_ps.index)], tma_bar_ptr=bar)
                                    wab_ps.advance()
                        for c in range(cp, cp + 2):
                            ws_pipe.producer_acquire(ws_ps)
                            cute.copy(tma_ws, tWsgWs[(None, 0, c, 0)], tWssWs[(None, ws_ps.index)],
                                      tma_bar_ptr=ws_pipe.producer_get_barrier(ws_ps))
                            ws_ps.advance()
                elif const_expr(self.defer):
                    # deferred: emit ALL wab first, THEN all ws (matches the deferred
                    # MMA: all expands then all squeezes).
                    for c in range(NCHUNK):
                        for kt in range(n_kt):
                            wab_pipe.producer_acquire(wab_ps)
                            bar = wab_pipe.producer_get_barrier(wab_ps)
                            cute.copy(tma_wa, tWagWa[(None, c, kt, 0)],
                                      tWasWa[(None, wab_ps.index)], tma_bar_ptr=bar)
                            cute.copy(tma_wb, tWbgWb[(None, c, kt, 0)],
                                      tWbsWb[(None, wab_ps.index)], tma_bar_ptr=bar)
                            wab_ps.advance()
                    for c in range(NCHUNK):
                        ws_pipe.producer_acquire(ws_ps)
                        cute.copy(tma_ws, tWsgWs[(None, 0, c, 0)], tWssWs[(None, ws_ps.index)],
                                  tma_bar_ptr=ws_pipe.producer_get_barrier(ws_ps))
                        ws_ps.advance()
                else:
                    for c in range(NCHUNK):
                        for kt in range(n_kt):
                            wab_pipe.producer_acquire(wab_ps)
                            bar = wab_pipe.producer_get_barrier(wab_ps)
                            cute.copy(tma_wa, tWagWa[(None, c, kt, 0)],
                                      tWasWa[(None, wab_ps.index)], tma_bar_ptr=bar)
                            cute.copy(tma_wb, tWbgWb[(None, c, kt, 0)],
                                      tWbsWb[(None, wab_ps.index)], tma_bar_ptr=bar)
                            wab_ps.advance()
                        ws_pipe.producer_acquire(ws_ps)
                        cute.copy(tma_ws, tWsgWs[(None, 0, c, 0)], tWssWs[(None, ws_ps.index)],
                                  tma_bar_ptr=ws_pipe.producer_get_barrier(ws_ps))
                        ws_ps.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            xn_pipe.producer_tail(xn_ps)
            wab_pipe.producer_tail(wab_ps)
            ws_pipe.producer_tail(ws_ps)

        # ============================ MMA warp ============================
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            a_acc_b = cute.make_tensor(tmem_ptr, tAcc_e_fake.layout)
            off = const_expr(tcgen05.find_tmem_tensor_col_offset(a_acc_b))
            b_acc_b = cute.make_tensor(tmem_ptr + off, tAcc_e_fake.layout)
            if const_expr(self.defer):
                # deferred: h at [2*off, 2*off + NCHUNK*hcols); o_acc ALIASES a_acc at
                # offset 0 (a_acc is dead after Phase 1; a Phase1/2 barrier guards it).
                h_tmem = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + 2 * off, dtype=self.ab_dtype),
                    tH_fake.layout)
                o_acc_b = cute.make_tensor(tmem_ptr, tAcc_s_fake.layout)
            else:
                o_acc_b = cute.make_tensor(tmem_ptr + 2 * off, tAcc_s_fake.layout)
                off_s = const_expr(tcgen05.find_tmem_tensor_col_offset(o_acc_b))
                h_tmem = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + 2 * off + off_s, dtype=self.ab_dtype),
                    tH_fake.layout)

            xn_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            wab_cs = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_wab_stage)
            ws_cs = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ws_stage)
            acc_ps = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage)
            h_cs = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_h_stage)
            out_ps = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_out_stage)

            # K-deep expand lookahead: issue num_acc_stage expands before the first
            # squeeze so the tensor core has num_acc_stage-1 independent expand chunks
            # in flight to hide the squeeze's tcgen05.ld (t2r) latency on h.
            # H>=1 makes this deadlock-free: when the h buffer is full the epilogue
            # has released >= that many acc stages, so an acc stage is always free
            # for the MMA warp's next expand acquire (see analysis).
            # host-time python int; DSL forbids closures in dynamic control flow, so the
            # expand body is inlined (via a range==1 python loop) in both spots below.
            K_AHEAD = self.num_acc_stage
            phase_bar = pipeline.NamedBarrier(
                barrier_id=self.phase_sync_bar_id,
                num_threads=32 * (1 + len(self.epilogue_warp_id)))

            if const_expr(self.pingpong):
                acc0_ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                acc1_ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                h0_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
                h1_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
                while work_tile.is_valid_tile:
                    o_acc = o_acc_b[(None, None, None, 0)]
                    xn_pipe.consumer_wait(xn_cs)
                    out_pipe.producer_acquire(out_ps)
                    acc_pipe0.producer_acquire(acc0_ps)
                    a_acc = a_acc_b[(None, None, None, 0)]
                    b_acc = b_acc_b[(None, None, None, 0)]
                    for kt in range(n_kt):
                        wab_pipe.consumer_wait(wab_cs)
                        wi = wab_cs.index
                        mma_e.set(tcgen05.Field.ACCUMULATE, kt != 0)
                        for j in cutlass.range_constexpr(kb_kt):
                            gkb = kt * kb_kt + j
                            cute.gemm(mma_e, a_acc, tCrXn[(None, None, gkb, 0)],
                                      tCrWa[(None, None, j, wi)], a_acc)
                            mma_e.set(tcgen05.Field.ACCUMULATE, True)
                        mma_e.set(tcgen05.Field.ACCUMULATE, kt != 0)
                        for j in cutlass.range_constexpr(kb_kt):
                            gkb = kt * kb_kt + j
                            cute.gemm(mma_e, b_acc, tCrXn[(None, None, gkb, 0)],
                                      tCrWb[(None, None, j, wi)], b_acc)
                            mma_e.set(tcgen05.Field.ACCUMULATE, True)
                        wab_pipe.consumer_release(wab_cs)
                        wab_cs.advance()
                    acc_pipe0.producer_commit(acc0_ps)
                    acc0_ps.advance()
                    acc_pipe1.producer_acquire(acc1_ps)
                    a_acc = a_acc_b[(None, None, None, 1)]
                    b_acc = b_acc_b[(None, None, None, 1)]
                    for kt in range(n_kt):
                        wab_pipe.consumer_wait(wab_cs)
                        wi = wab_cs.index
                        mma_e.set(tcgen05.Field.ACCUMULATE, kt != 0)
                        for j in cutlass.range_constexpr(kb_kt):
                            gkb = kt * kb_kt + j
                            cute.gemm(mma_e, a_acc, tCrXn[(None, None, gkb, 0)],
                                      tCrWa[(None, None, j, wi)], a_acc)
                            mma_e.set(tcgen05.Field.ACCUMULATE, True)
                        mma_e.set(tcgen05.Field.ACCUMULATE, kt != 0)
                        for j in cutlass.range_constexpr(kb_kt):
                            gkb = kt * kb_kt + j
                            cute.gemm(mma_e, b_acc, tCrXn[(None, None, gkb, 0)],
                                      tCrWb[(None, None, j, wi)], b_acc)
                            mma_e.set(tcgen05.Field.ACCUMULATE, True)
                        wab_pipe.consumer_release(wab_cs)
                        wab_cs.advance()
                    acc_pipe1.producer_commit(acc1_ps)
                    acc1_ps.advance()
                    for cp in range(0, NCHUNK, 2):
                        if cp + 2 < NCHUNK:
                            acc_pipe0.producer_acquire(acc0_ps)
                            a_acc = a_acc_b[(None, None, None, 0)]
                            b_acc = b_acc_b[(None, None, None, 0)]
                            for kt in range(n_kt):
                                wab_pipe.consumer_wait(wab_cs)
                                wi = wab_cs.index
                                mma_e.set(tcgen05.Field.ACCUMULATE, kt != 0)
                                for j in cutlass.range_constexpr(kb_kt):
                                    gkb = kt * kb_kt + j
                                    cute.gemm(mma_e, a_acc, tCrXn[(None, None, gkb, 0)],
                                              tCrWa[(None, None, j, wi)], a_acc)
                                    mma_e.set(tcgen05.Field.ACCUMULATE, True)
                                mma_e.set(tcgen05.Field.ACCUMULATE, kt != 0)
                                for j in cutlass.range_constexpr(kb_kt):
                                    gkb = kt * kb_kt + j
                                    cute.gemm(mma_e, b_acc, tCrXn[(None, None, gkb, 0)],
                                              tCrWb[(None, None, j, wi)], b_acc)
                                    mma_e.set(tcgen05.Field.ACCUMULATE, True)
                                wab_pipe.consumer_release(wab_cs)
                                wab_cs.advance()
                            acc_pipe0.producer_commit(acc0_ps)
                            acc0_ps.advance()
                            acc_pipe1.producer_acquire(acc1_ps)
                            a_acc = a_acc_b[(None, None, None, 1)]
                            b_acc = b_acc_b[(None, None, None, 1)]
                            for kt in range(n_kt):
                                wab_pipe.consumer_wait(wab_cs)
                                wi = wab_cs.index
                                mma_e.set(tcgen05.Field.ACCUMULATE, kt != 0)
                                for j in cutlass.range_constexpr(kb_kt):
                                    gkb = kt * kb_kt + j
                                    cute.gemm(mma_e, a_acc, tCrXn[(None, None, gkb, 0)],
                                              tCrWa[(None, None, j, wi)], a_acc)
                                    mma_e.set(tcgen05.Field.ACCUMULATE, True)
                                mma_e.set(tcgen05.Field.ACCUMULATE, kt != 0)
                                for j in cutlass.range_constexpr(kb_kt):
                                    gkb = kt * kb_kt + j
                                    cute.gemm(mma_e, b_acc, tCrXn[(None, None, gkb, 0)],
                                              tCrWb[(None, None, j, wi)], b_acc)
                                    mma_e.set(tcgen05.Field.ACCUMULATE, True)
                                wab_pipe.consumer_release(wab_cs)
                                wab_cs.advance()
                            acc_pipe1.producer_commit(acc1_ps)
                            acc1_ps.advance()
                        h_pipe0.consumer_wait(h0_cs)
                        ws_pipe.consumer_wait(ws_cs)
                        si = ws_cs.index
                        tCrH = h_tmem[(None, None, None, 0)]
                        mma_s.set(tcgen05.Field.ACCUMULATE, cp != 0)
                        for j in cutlass.range_constexpr(kb_bn):
                            cute.gemm(mma_s, o_acc, tCrH[(None, None, j)],
                                      tCrWs[(None, None, j, si)], o_acc)
                            mma_s.set(tcgen05.Field.ACCUMULATE, True)
                        ws_pipe.consumer_release(ws_cs)
                        ws_cs.advance()
                        h_pipe0.consumer_release(h0_cs)
                        h0_cs.advance()
                        h_pipe1.consumer_wait(h1_cs)
                        ws_pipe.consumer_wait(ws_cs)
                        si = ws_cs.index
                        tCrH = h_tmem[(None, None, None, 1)]
                        mma_s.set(tcgen05.Field.ACCUMULATE, True)
                        for j in cutlass.range_constexpr(kb_bn):
                            cute.gemm(mma_s, o_acc, tCrH[(None, None, j)],
                                      tCrWs[(None, None, j, si)], o_acc)
                            mma_s.set(tcgen05.Field.ACCUMULATE, True)
                        ws_pipe.consumer_release(ws_cs)
                        ws_cs.advance()
                        h_pipe1.consumer_release(h1_cs)
                        h1_cs.advance()
                    out_pipe.producer_commit(out_ps)
                    out_ps.advance()
                    xn_pipe.consumer_release(xn_cs)
                    xn_cs.advance()
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
                acc_pipe0.producer_tail(acc0_ps)
                acc_pipe1.producer_tail(acc1_ps)
                out_pipe.producer_tail(out_ps)
            else:
                while work_tile.is_valid_tile:
                    o_acc = o_acc_b[(None, None, None, 0)]

                    xn_pipe.consumer_wait(xn_cs)
                    out_pipe.producer_acquire(out_ps)

                    if const_expr(self.defer):
                        # Phase 1: ALL expands+silu (no squeeze); epilogue fills NCHUNK h.
                        for c in range(NCHUNK):
                            acc_pipe.producer_acquire(acc_ps)
                            ai = acc_ps.index
                            a_acc = a_acc_b[(None, None, None, ai)]
                            b_acc = b_acc_b[(None, None, None, ai)]
                            for kt in range(n_kt):
                                wab_pipe.consumer_wait(wab_cs)
                                wi = wab_cs.index
                                mma_e.set(tcgen05.Field.ACCUMULATE, kt != 0)
                                for j in cutlass.range_constexpr(kb_kt):
                                    gkb = kt * kb_kt + j
                                    cute.gemm(mma_e, a_acc, tCrXn[(None, None, gkb, 0)],
                                              tCrWa[(None, None, j, wi)], a_acc)
                                    mma_e.set(tcgen05.Field.ACCUMULATE, True)
                                mma_e.set(tcgen05.Field.ACCUMULATE, kt != 0)
                                for j in cutlass.range_constexpr(kb_kt):
                                    gkb = kt * kb_kt + j
                                    cute.gemm(mma_e, b_acc, tCrXn[(None, None, gkb, 0)],
                                              tCrWb[(None, None, j, wi)], b_acc)
                                    mma_e.set(tcgen05.Field.ACCUMULATE, True)
                                wab_pipe.consumer_release(wab_cs)
                                wab_cs.advance()
                            acc_pipe.producer_commit(acc_ps)
                            acc_ps.advance()
                        # every h committed & every a_acc read -> safe to alias o onto a
                        phase_bar.arrive_and_wait()
                        # Phase 2: ALL squeezes back-to-back (every h already resident)
                        for c in range(NCHUNK):
                            h_pipe.consumer_wait(h_cs)
                            ws_pipe.consumer_wait(ws_cs)
                            hi = h_cs.index
                            si = ws_cs.index
                            tCrH = h_tmem[(None, None, None, hi)]
                            mma_s.set(tcgen05.Field.ACCUMULATE, c != 0)
                            for j in cutlass.range_constexpr(kb_bn):
                                cute.gemm(mma_s, o_acc, tCrH[(None, None, j)],
                                          tCrWs[(None, None, j, si)], o_acc)
                                mma_s.set(tcgen05.Field.ACCUMULATE, True)
                            ws_pipe.consumer_release(ws_cs)
                            ws_cs.advance()
                            h_pipe.consumer_release(h_cs)
                            h_cs.advance()
                    else:
                        # ---- prologue: issue K_AHEAD-1 expands ahead (chunks 0..K_AHEAD-2) ----
                        for _pc in range(min(K_AHEAD - 1, NCHUNK)):
                            acc_pipe.producer_acquire(acc_ps)
                            ai = acc_ps.index
                            a_acc = a_acc_b[(None, None, None, ai)]
                            b_acc = b_acc_b[(None, None, None, ai)]
                            for kt in range(n_kt):
                                wab_pipe.consumer_wait(wab_cs)
                                wi = wab_cs.index
                                mma_e.set(tcgen05.Field.ACCUMULATE, kt != 0)
                                for j in cutlass.range_constexpr(kb_kt):
                                    gkb = kt * kb_kt + j
                                    cute.gemm(mma_e, a_acc, tCrXn[(None, None, gkb, 0)],
                                              tCrWa[(None, None, j, wi)], a_acc)
                                    mma_e.set(tcgen05.Field.ACCUMULATE, True)
                                mma_e.set(tcgen05.Field.ACCUMULATE, kt != 0)
                                for j in cutlass.range_constexpr(kb_kt):
                                    gkb = kt * kb_kt + j
                                    cute.gemm(mma_e, b_acc, tCrXn[(None, None, gkb, 0)],
                                              tCrWb[(None, None, j, wi)], b_acc)
                                    mma_e.set(tcgen05.Field.ACCUMULATE, True)
                                wab_pipe.consumer_release(wab_cs)
                                wab_cs.advance()
                            acc_pipe.producer_commit(acc_ps)
                            acc_ps.advance()

                        for c in range(NCHUNK):
                            # ---- expand chunk c+K_AHEAD-1 (ahead: overlaps epilogue+squeeze) ----
                            for _ec in range(1 if c + K_AHEAD - 1 < NCHUNK else 0):
                                acc_pipe.producer_acquire(acc_ps)
                                ai = acc_ps.index
                                a_acc = a_acc_b[(None, None, None, ai)]
                                b_acc = b_acc_b[(None, None, None, ai)]
                                for kt in range(n_kt):
                                    wab_pipe.consumer_wait(wab_cs)
                                    wi = wab_cs.index
                                    mma_e.set(tcgen05.Field.ACCUMULATE, kt != 0)
                                    for j in cutlass.range_constexpr(kb_kt):
                                        gkb = kt * kb_kt + j
                                        cute.gemm(mma_e, a_acc, tCrXn[(None, None, gkb, 0)],
                                                  tCrWa[(None, None, j, wi)], a_acc)
                                        mma_e.set(tcgen05.Field.ACCUMULATE, True)
                                    mma_e.set(tcgen05.Field.ACCUMULATE, kt != 0)
                                    for j in cutlass.range_constexpr(kb_kt):
                                        gkb = kt * kb_kt + j
                                        cute.gemm(mma_e, b_acc, tCrXn[(None, None, gkb, 0)],
                                                  tCrWb[(None, None, j, wi)], b_acc)
                                        mma_e.set(tcgen05.Field.ACCUMULATE, True)
                                    wab_pipe.consumer_release(wab_cs)
                                    wab_cs.advance()
                                acc_pipe.producer_commit(acc_ps)
                                acc_ps.advance()

                            # ---- squeeze chunk c: A (=h) read straight from TMEM ----
                            h_pipe.consumer_wait(h_cs)
                            ws_pipe.consumer_wait(ws_cs)
                            hi = h_cs.index
                            si = ws_cs.index
                            tCrH = h_tmem[(None, None, None, hi)]
                            mma_s.set(tcgen05.Field.ACCUMULATE, c != 0)
                            for j in cutlass.range_constexpr(kb_bn):
                                cute.gemm(mma_s, o_acc, tCrH[(None, None, j)],
                                          tCrWs[(None, None, j, si)], o_acc)
                                mma_s.set(tcgen05.Field.ACCUMULATE, True)
                            ws_pipe.consumer_release(ws_cs)
                            ws_cs.advance()
                            h_pipe.consumer_release(h_cs)
                            h_cs.advance()

                    out_pipe.producer_commit(out_ps)
                    out_ps.advance()
                    xn_pipe.consumer_release(xn_cs)
                    xn_cs.advance()
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
                acc_pipe.producer_tail(acc_ps)
                out_pipe.producer_tail(out_ps)

        # ========================= Epilogue warps =========================
        if warp_idx < len(self.epilogue_warp_id):
            tmem.allocate(512)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            a_acc_b = cute.make_tensor(tmem_ptr, tAcc_e_fake.layout)
            off = const_expr(tcgen05.find_tmem_tensor_col_offset(a_acc_b))
            b_acc_b = cute.make_tensor(tmem_ptr + off, tAcc_e_fake.layout)
            if const_expr(self.defer):
                # deferred: h at [2*off, 2*off + NCHUNK*hcols); o_acc ALIASES a_acc at
                # offset 0 (a_acc is dead after Phase 1; a Phase1/2 barrier guards it).
                h_tmem = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + 2 * off, dtype=self.ab_dtype),
                    tH_fake.layout)
                o_acc_b = cute.make_tensor(tmem_ptr, tAcc_s_fake.layout)
            else:
                o_acc_b = cute.make_tensor(tmem_ptr + 2 * off, tAcc_s_fake.layout)
                off_s = const_expr(tcgen05.find_tmem_tensor_col_offset(o_acc_b))
                h_tmem = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + 2 * off + off_s, dtype=self.ab_dtype),
                    tH_fake.layout)

            # ---- silu t2r (expand a/b accumulators -> registers) ----
            idH = cute.make_identity_tensor((CTA_M, BN, 1))
            gH = cute.local_tile(idH, cute.slice_((CTA_M, BN, KT), (None, None, 0)), (None, None, None))
            tCgH = transform_partitioned_tensor_layout(thr_e.partition_C(gH))
            tAccA = transform_partitioned_tensor_layout(a_acc_b)
            tAccB = transform_partitioned_tensor_layout(b_acc_b)
            tc_t2r_h, tTR_tA, tTR_rA = epilogue_tmem_copy_and_partition(
                self, tidx, tAccA, tCgH, epi_tile_h, False)
            _, tTR_tB, tTR_rB = epilogue_tmem_copy_and_partition(
                self, tidx, tAccB, tCgH, epi_tile_h, False)

            # ---- silu r2t (registers -> TMEM h, A-operand layout) ----
            # a_sq_shape[0][0]==128 (M atom) -> St32x32b(Rep8), per reference
            # get_copy_atom_a_transform. make_tmem_copy gives correct TMEM
            # datapath addressing; the register source fragment is taken from the
            # r2t copy's own partition_S (atom-V==16 matches the St dst), so
            # cute.copy lines up without a cross-family retile (which aborts).
            # Linear reg index j maps to h K-column j == accumulator column j.
            copy_atom_r2t = cute.make_copy_atom(
                tcgen05.St32x32bOp(tcgen05.Repetition(8), tcgen05.Unpack.NONE), self.ab_dtype)
            tc_r2t = tcgen05.make_tmem_copy(copy_atom_r2t, h_tmem[(None, None, None, 0)])
            thr_r2t = tc_r2t.get_slice(tidx)
            # Partition the FULL staged h_tmem here (outside the loop) and only INDEX the
            # stage per-chunk inside. Calling partition_D inside the loop captures the
            # tiled_copy as a loop-carried SSA value, which fails to legalize.
            tRT_tH_all = thr_r2t.partition_D(h_tmem)
            # Source register fragment via the r2t copy's OWN partition_S (atom-V==16,
            # matching the St32x32b dst); filter_zeros drops the broadcast datapath.
            idHt = cute.make_identity_tensor(h_tmem[(None, None, None, 0)].shape)
            tRT_src = thr_r2t.partition_S(idHt)
            rC = cute.make_rmem_tensor(cute.filter_zeros(tRT_src.layout).shape, self.ab_dtype)
            # compute-view of rC in accumulator column order (compact phys pos == h K-col)
            rC_view = cute.make_tensor(rC.iterator, tTR_rA.layout)
            tTR_tA_s = cute.group_modes(tTR_tA[(None, None, None, None, None, 0)], 3, 5)
            tTR_tB_s = cute.group_modes(tTR_tB[(None, None, None, None, None, 0)], 3, 5)

            # ---- out t2r (helper) + simt register->global store ----
            tCgO = transform_partitioned_tensor_layout(thr_s.partition_C(gOut))
            tAccO = transform_partitioned_tensor_layout(o_acc_b)
            tc_t2r_o, tTR_tO, tTR_rO = epilogue_tmem_copy_and_partition(
                self, tidx, tAccO, tCgO, epi_tile_out, False)
            rOc = cute.make_rmem_tensor(tTR_rO.shape, self.c_dtype)
            tCgO_epi = cute.flat_divide(tCgO, epi_tile_out)
            tTR_gO = tc_t2r_o.get_slice(tidx).partition_D(tCgO_epi)
            tTR_tO_s = cute.group_modes(tTR_tO[(None, None, None, None, None, 0)], 3, 5)
            simt = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.c_dtype,
                num_bits_per_copy=self.c_dtype.width)

            acc_cs = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage)
            h_ps = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_h_stage)
            out_cs = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_out_stage)
            acc0_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            h0_ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)

            sub_h = cute.size(tTR_tA_s, mode=[3])
            sub_o = cute.size(tTR_tO_s, mode=[3])
            epi_bar = pipeline.NamedBarrier(
                barrier_id=self.epilog_sync_bar_id,
                num_threads=32 * len(self.epilogue_warp_id))
            phase_bar = pipeline.NamedBarrier(
                barrier_id=self.phase_sync_bar_id,
                num_threads=32 * (1 + len(self.epilogue_warp_id)))

            while work_tile.is_valid_tile:
                m = work_tile.tile_idx[0]
                if const_expr(self.pingpong):
                    for c in range(0, NCHUNK, 2):
                        acc_pipe0.consumer_wait(acc0_cs)
                        h_pipe0.producer_acquire(h0_ps)
                        tTR_tA_s = cute.group_modes(tTR_tA[(None, None, None, None, None, 0)], 3, 5)
                        tTR_tB_s = cute.group_modes(tTR_tB[(None, None, None, None, None, 0)], 3, 5)
                        tRT_tH = tRT_tH_all[(None,) * (const_expr(cute.rank(tRT_tH_all)) - 1) + (0,)]
                        for s in cutlass.range_constexpr(sub_h):
                            cute.copy(tc_t2r_h, tTR_tA_s[(None, None, None, s)], tTR_rA)
                            cute.copy(tc_t2r_h, tTR_tB_s[(None, None, None, s)], tTR_rB)
                            g = tTR_rA.load()
                            p = tTR_rB.load()
                            d = 1.0 + cmath.exp2(g * (-1.4426950408889634))
                            r = cmath.rsqrt(d)
                            rC_view.store((g * p * r * r).to(self.ab_dtype))
                            cute.copy(tc_r2t, rC, tRT_tH)
                        cute.arch.fence_view_async_tmem_load()
                        cute.arch.fence_view_async_tmem_store()
                        with cute.arch.elect_one():
                            acc_pipe0.consumer_release(acc0_cs)
                        acc0_cs.advance()
                        epi_bar.arrive_and_wait()
                        h_pipe0.producer_commit(h0_ps)
                        h0_ps.advance()
                else:
                    for c in range(NCHUNK):
                        acc_pipe.consumer_wait(acc_cs)
                        h_pipe.producer_acquire(h_ps)
                        ci = acc_cs.index
                        hpi = h_ps.index
                        tTR_tA_s = cute.group_modes(tTR_tA[(None, None, None, None, None, ci)], 3, 5)
                        tTR_tB_s = cute.group_modes(tTR_tB[(None, None, None, None, None, ci)], 3, 5)
                        tRT_tH = tRT_tH_all[(None,) * (const_expr(cute.rank(tRT_tH_all)) - 1) + (hpi,)]
                        for s in cutlass.range_constexpr(sub_h):
                            cute.copy(tc_t2r_h, tTR_tA_s[(None, None, None, s)], tTR_rA)
                            cute.copy(tc_t2r_h, tTR_tB_s[(None, None, None, s)], tTR_rB)
                            g = tTR_rA.load()
                            p = tTR_rB.load()
                            d = 1.0 + cmath.exp2(g * (-1.4426950408889634))
                            r = cmath.rsqrt(d)
                            rC_view.store((g * p * r * r).to(self.ab_dtype))
                            cute.copy(tc_r2t, rC, tRT_tH)
                        cute.arch.fence_view_async_tmem_load()
                        cute.arch.fence_view_async_tmem_store()
                        with cute.arch.elect_one():
                            acc_pipe.consumer_release(acc_cs)
                        acc_cs.advance()
                        epi_bar.arrive_and_wait()
                        h_pipe.producer_commit(h_ps)
                        h_ps.advance()

                if const_expr(self.defer):
                    # sync with MMA: all NCHUNK h committed & all a_acc read, so Phase 2
                    # may overwrite [0, 2*off) where o_acc aliases the now-dead a_acc.
                    phase_bar.arrive_and_wait()
                out_pipe.consumer_wait(out_cs)
                tTR_gO_m = cute.group_modes(tTR_gO[(None, None, None, None, None, m, 0, 0)], 3, 5)
                for s in cutlass.range_constexpr(sub_o):
                    cute.copy(tc_t2r_o, tTR_tO_s[(None, None, None, s)], tTR_rO)
                    rOc.store(tTR_rO.load().to(self.c_dtype))
                    cute.copy(simt, rOc, tTR_gO_m[(None, None, None, s)])
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    out_pipe.consumer_release(out_cs)
                out_cs.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            if const_expr(self.pingpong):
                h_pipe0.producer_tail(h0_ps)
            else:
                h_pipe.producer_tail(h_ps)
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        if const_expr(self.pingpong):
          if warp_idx >= self.silu1_warp_id[0] and warp_idx < self.mma_warp_id:
            # ===== silu group 1 (ping-pong): odd ND chunks, TMEM buffer 1 =====
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            a_acc_b = cute.make_tensor(tmem_ptr, tAcc_e_fake.layout)
            off = const_expr(tcgen05.find_tmem_tensor_col_offset(a_acc_b))
            b_acc_b = cute.make_tensor(tmem_ptr + off, tAcc_e_fake.layout)
            o_acc_b = cute.make_tensor(tmem_ptr + 2 * off, tAcc_s_fake.layout)
            off_s = const_expr(tcgen05.find_tmem_tensor_col_offset(o_acc_b))
            h_tmem = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + 2 * off + off_s, dtype=self.ab_dtype),
                tH_fake.layout)
            ltidx = tidx - 32 * len(self.epilogue_warp_id)
            idH = cute.make_identity_tensor((CTA_M, BN, 1))
            gH = cute.local_tile(idH, cute.slice_((CTA_M, BN, KT), (None, None, 0)), (None, None, None))
            tCgH = transform_partitioned_tensor_layout(thr_e.partition_C(gH))
            tAccA = transform_partitioned_tensor_layout(a_acc_b)
            tAccB = transform_partitioned_tensor_layout(b_acc_b)
            tc_t2r_h, tTR_tA, tTR_rA = epilogue_tmem_copy_and_partition(
                self, ltidx, tAccA, tCgH, epi_tile_h, False)
            _, tTR_tB, tTR_rB = epilogue_tmem_copy_and_partition(
                self, ltidx, tAccB, tCgH, epi_tile_h, False)
            copy_atom_r2t = cute.make_copy_atom(
                tcgen05.St32x32bOp(tcgen05.Repetition(8), tcgen05.Unpack.NONE), self.ab_dtype)
            tc_r2t = tcgen05.make_tmem_copy(copy_atom_r2t, h_tmem[(None, None, None, 0)])
            thr_r2t = tc_r2t.get_slice(ltidx)
            tRT_tH_all = thr_r2t.partition_D(h_tmem)
            idHt = cute.make_identity_tensor(h_tmem[(None, None, None, 0)].shape)
            tRT_src = thr_r2t.partition_S(idHt)
            rC = cute.make_rmem_tensor(cute.filter_zeros(tRT_src.layout).shape, self.ab_dtype)
            rC_view = cute.make_tensor(rC.iterator, tTR_rA.layout)
            tTR_tA_s = cute.group_modes(tTR_tA[(None, None, None, None, None, 1)], 3, 5)
            tTR_tB_s = cute.group_modes(tTR_tB[(None, None, None, None, None, 1)], 3, 5)
            tRT_tH = tRT_tH_all[(None,) * (const_expr(cute.rank(tRT_tH_all)) - 1) + (1,)]
            acc1_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            h1_ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            sub_h = cute.size(tTR_tA_s, mode=[3])
            silu1_bar = pipeline.NamedBarrier(
                barrier_id=self.silu1_sync_bar_id, num_threads=32 * len(self.silu1_warp_id))
            while work_tile.is_valid_tile:
                for c in range(1, NCHUNK, 2):
                    acc_pipe1.consumer_wait(acc1_cs)
                    h_pipe1.producer_acquire(h1_ps)
                    for s in cutlass.range_constexpr(sub_h):
                        cute.copy(tc_t2r_h, tTR_tA_s[(None, None, None, s)], tTR_rA)
                        cute.copy(tc_t2r_h, tTR_tB_s[(None, None, None, s)], tTR_rB)
                        g = tTR_rA.load()
                        p = tTR_rB.load()
                        d = 1.0 + cmath.exp2(g * (-1.4426950408889634))
                        r = cmath.rsqrt(d)
                        rC_view.store((g * p * r * r).to(self.ab_dtype))
                        cute.copy(tc_r2t, rC, tRT_tH)
                    cute.arch.fence_view_async_tmem_load()
                    cute.arch.fence_view_async_tmem_store()
                    with cute.arch.elect_one():
                        acc_pipe1.consumer_release(acc1_cs)
                    acc1_cs.advance()
                    silu1_bar.arrive_and_wait()
                    h_pipe1.producer_commit(h1_ps)
                    h1_ps.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            h_pipe1.producer_tail(h1_ps)


_CACHE = {}


def transition_b2b_fused_sm100(xn, wa, wb, ws):
    """out = (silu(xn@wa^T) * (xn@wb^T)) @ ws^T.
    xn:(M,K) pre-normalized bf16; wa/wb:(ND,K) bf16; ws:(D,ND) bf16. Returns out:(M,D) bf16."""
    M, K = xn.shape
    ND, _ = wa.shape
    D, _ = ws.shape

    def _mark(t3, ld):
        return from_dlpack(t3, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=ld)

    mXn = _mark(xn.detach().unsqueeze(0), 2)
    mWa = _mark(wa.detach().unsqueeze(0), 2)
    mWb = _mark(wb.detach().unsqueeze(0), 2)
    mWs = _mark(ws.detach().unsqueeze(0), 2)
    out = torch.empty(M, D, device=xn.device, dtype=torch.bfloat16)
    mOut = _mark(out.unsqueeze(0), 2)

    mac = get_max_active_clusters(1)  # memoized: avoid per-call CUTLASS-DSL probe recompile
    strm = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    key = (M, K, ND, D)
    if key not in _CACHE:
        kt = 128 if K % 128 == 0 else K
        BN = int(os.environ.get("MW_BN", "64"))
        _req_acc = int(os.environ.get("MW_ACC_STAGE", "0"))   # 0 = auto
        _req_h = int(os.environ.get("MW_H_STAGE", "0"))        # 0 = auto
        Di = int(D)

        _nchunk = int(ND) // BN

        def _fits(a, h):
            return 2 * a * BN + Di + h * (BN // 2) <= 512

        # Deferred squeeze: keep ALL NCHUNK h resident in TMEM (o_acc aliases the dead
        # a_acc, so it is not counted); Phase 1 = all expands+silu, Phase 2 = all
        # squeezes. Fits for d=128 (NCHUNK=8), overflows for d=256 (NCHUNK=16).
        def _fits_defer(a):
            return 2 * a * BN + _nchunk * (BN // 2) <= 512
        # Deferred squeeze is IMPLEMENTED and correct but benchmarks ~5% SLOWER than
        # the K-ahead interleaved path for d=128 (the epilogue TMEM t2r/r2t chain,
        # not expand/squeeze overlap, is the wall). Default OFF; set MW_DEFER=1 to try.
        _defer = int(os.environ.get("MW_DEFER", "0")) != 0 and _fits_defer(2)

        if _defer:
            acc, h = 2, _nchunk
        else:
            # auto default: 2-stage if it fits, else 1-stage.
            acc = _req_acc if _req_acc > 0 else (2 if _fits(2, 2) else 1)
            h = _req_h if _req_h > 0 else acc
            # TMEM-clamp: shrink acc first, then h, so overflowing configs (d=256)
            # fall back to a smaller stage count for that D only. Keeps h >= 1.
            while not _fits(acc, h) and acc > 1:
                acc -= 1
            while not _fits(acc, h) and h > 1:
                h -= 1

        op = TransitionB2BFusedKernel(cta_m=128, bn=BN, kt=kt)
        op.K, op.ND, op.D = int(K), int(ND), Di
        _pp = int(os.environ.get("MW_PINGPONG", "1")) != 0 and _fits(2, 2) and not _defer
        if _pp:
            acc, h = 2, 2
        op.num_acc_stage = acc
        op.num_h_stage = h
        op.num_out_stage = 1
        op.defer = _defer
        op.pingpong = _pp
        _CACHE[key] = cute.compile(op, mXn, mWa, mWb, mWs, mOut, mac, strm, options="--enable-tvm-ffi")
    _CACHE[key](mXn, mWa, mWb, mWs, mOut)
    return out
