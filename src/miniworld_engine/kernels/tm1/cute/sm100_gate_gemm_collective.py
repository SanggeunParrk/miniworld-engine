"""v16: gated dual-B GEMM built on CUTLASS's tuned Blackwell persistent collective.

out = sigmoid(A@Bg.T) * (A@Bp.T), stored M-major straight into [B,D,L,L] (zero-copy).
A:(M,K) bf16, Bp/Bg:(N,K) bf16 -> out:(M,N) bf16, fp32 accumulate.

Reuses `PersistentDenseGemmKernel` (vendored `_blackwell_dense_gemm.py`) for all the tuned
machinery: warp specialization (tma/mma/epi warps), multi-stage PipelineTmaUmma load<->MMA
overlap, acc double-buffering, StaticPersistentTileScheduler, optional 2-CTA cluster, TMA store.

Extensions over the plain collective:
  * Dual B: Bp and Bg share the single A load (A read once, gate never stored).
  * Dual TMEM accumulator: proj + gate in tensor memory, filled by two tcgen05 MMAs per tile.
  * Fused GLU TMA-store epilogue: sigmoid(gate)*proj -> bf16 -> TMA store (M-major).
  * mma_tiler K forced to full K so one AB buffer holds A+Bp+Bg (proj-then-gate mainloop).
"""

from __future__ import annotations
import os
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch
from cutlass import BFloat16, Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from miniworld_engine.kernels.tm1.cute._blackwell_dense_gemm import (
    PersistentDenseGemmKernel,
)
# Memoized occupancy query (== the wrapper every training kernel uses). The raw
# `utils.HardwareInfo().get_max_active_clusters(...)` JIT-compiles a probe kernel on
# EVERY call; @lru_cache makes it a one-time device-constant lookup. Same returned int
# -> identical kernel launch/numerics. Fixes the ~900ms/iter eager launch overhead.
from quack.cute_dsl_utils import get_max_active_clusters
from cutlass.utils.gemm.sm100 import (
    epilogue_tmem_copy_and_partition,
    epilogue_smem_copy_and_partition,
    transform_partitioned_tensor_layout,
)


class GatedPersistentGemmKernel(PersistentDenseGemmKernel):
    """Dual-B gated GEMM on the tuned Blackwell persistent collective."""

    proj_only = False  # side experiment: skip gate MMA + GLU (isolates dual-work cost)
    epi_gate = True     # side experiment: if False, run gate MMA but store proj only
    no_exp = False      # side experiment: skip sigmoid (ov=p*g) to isolate exp cost
    sig_mode = "rsqrt"  # sigmoid impl (default rsqrt: division-free, fast MUFU)
    epi_depth = 3       # epilogue t2r software-pipeline depth (v17: 3 = k+2 prefetch; 2 == v16)

    def _setup_attributes(self):
        # Inherit ALL of the collective's tuned choices (mma_tiler incl. K-tiling,
        # epi_tile, num_acc_stage, cluster layout, mcast, ...), then adjust only for
        # (a) the SECOND B operand in shared memory, and (b) the dual TMEM accumulator.
        super()._setup_attributes()
        tiled_mma = self._create_tiled_mma()

        # Recompute AB stage count with ab_bytes = A + 2*B (dual B shares the A load).
        a_one = utils.sm100.make_smem_layout_a(tiled_mma, self.mma_tiler, self.a_dtype, 1)
        b_one = utils.sm100.make_smem_layout_b(tiled_mma, self.mma_tiler, self.b_dtype, 1)
        ab_bytes = cute.size_in_bytes(self.a_dtype, a_one) + 2 * cute.size_in_bytes(
            self.b_dtype, b_one
        )
        c_smem_one = utils.sm100.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, 1
        )
        c_bytes_ps = cute.size_in_bytes(self.c_dtype, c_smem_one)
        mbar_bytes = 1024

        self.num_c_stage = 2
        num_ab_stage = (
            self.smem_capacity - (mbar_bytes + c_bytes_ps * self.num_c_stage)
        ) // ab_bytes
        if num_ab_stage < 1:
            num_ab_stage = 1
        self.num_ab_stage = num_ab_stage
        used = ab_bytes * self.num_ab_stage + mbar_bytes + c_bytes_ps * self.num_c_stage
        self.num_c_stage += (self.smem_capacity - used) // c_bytes_ps

        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )
        self.c_smem_layout_staged = utils.sm100.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage
        )

        # Dual accumulator in TMEM: proj + gate. Keep under the 512-col TMEM budget.
        single = self.num_tmem_alloc_cols  # parent value (uses num_acc_stage)
        if 2 * single > 512:
            self.num_acc_stage = 1
            single = self._compute_num_tmem_alloc_cols(
                tiled_mma, self.mma_tiler, self.num_acc_stage, self.arch
            )
        self.num_tmem_alloc_cols_single = single
        self.num_tmem_alloc_cols = 2 * single

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        bp: cute.Tensor,
        bg: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        # Inputs arrive as (L, X, Y); permute to CuTe MNKL convention (X, Y, L)
        # (same as ref `bmm`). Done here (trace time) because .layout is trace-only.
        a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))    # (M,K,L)
        bp = cute.make_tensor(bp.iterator, cute.select(bp.layout, mode=[1, 2, 0]))  # (N,K,L)
        bg = cute.make_tensor(bg.iterator, cute.select(bg.layout, mode=[1, 2, 0]))  # (N,K,L)
        c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))    # (M,N,L)

        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = bp.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(bp).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        # self.K is a static python int set at construction (mma_tiler needs constexpr K).

        tiled_mma = self._create_tiled_mma()
        self._setup_attributes()

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # TMA load A
        a_op = utils.sm100.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )

        # TMA load Bp and Bg (same layout; both share the A load)
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_bp, tma_tensor_bp = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, bp, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        tma_atom_bg, tma_tensor_bg = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, bg, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + 2 * b_copy_size) * atom_thr_size

        # TMA store C
        epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile
        )

        self.tile_sched_params, grid = self._compute_grid(
            c, self.cta_tile_shape_mnk, self.cluster_shape_mn, max_active_clusters
        )

        self.kernel(
            tiled_mma,
            tma_atom_a, tma_tensor_a,
            tma_atom_bp, tma_tensor_bp,
            tma_atom_bg, tma_tensor_bg,
            tma_atom_c, tma_tensor_c,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom, mA_mkl: cute.Tensor,
        tma_atom_bp: cute.CopyAtom, mBp_nkl: cute.Tensor,
        tma_atom_bg: cute.CopyAtom, mBg_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom, mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_bp)
            cpasync.prefetch_descriptor(tma_atom_bg)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: cute.struct.MemRange[cutlass.Int32, 1]

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.data_ptr(),
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.data_ptr(),
        )

        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sA = smem.allocate_tensor(
            element_type=self.a_dtype, layout=a_smem_layout_staged.outer,
            byte_alignment=128, swizzle=a_smem_layout_staged.inner,
        )
        sBp = smem.allocate_tensor(
            element_type=self.b_dtype, layout=b_smem_layout_staged.outer,
            byte_alignment=128, swizzle=b_smem_layout_staged.inner,
        )
        sBg = smem.allocate_tensor(
            element_type=self.b_dtype, layout=b_smem_layout_staged.outer,
            byte_alignment=128, swizzle=b_smem_layout_staged.inner,
        )

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gBp_nkl = cute.local_tile(
            mBp_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gBg_nkl = cute.local_tile(
            mBg_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgBp = thr_mma.partition_B(gBp_nkl)
        tCgBg = thr_mma.partition_B(gBg_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, block_in_cluster_coord_vmnk[2], a_cta_layout,
            cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsBp, tBgBp = cpasync.tma_partition(
            tma_atom_bp, block_in_cluster_coord_vmnk[1], b_cta_layout,
            cute.group_modes(sBp, 0, 3), cute.group_modes(tCgBp, 0, 3),
        )
        tBsBg, tBgBg = cpasync.tma_partition(
            tma_atom_bg, block_in_cluster_coord_vmnk[1], b_cta_layout,
            cute.group_modes(sBg, 0, 3), cute.group_modes(tCgBg, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrBp = tiled_mma.make_fragment_B(sBp)
        tCrBg = tiled_mma.make_fragment_B(sBg)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        # ---- TMA load warp: A, Bp, Bg into one AB buffer per tile ----
        if warp_idx == self.tma_warp_id:
            while work_tile.is_valid_tile:
                cur = work_tile.tile_idx
                mnl = (cur[0] // cute.size(tiled_mma.thr_id.shape), cur[1], cur[2])
                tAgA_slice = tAgA[(None, mnl[0], None, mnl[2])]
                tBgBp_slice = tBgBp[(None, mnl[1], None, mnl[2])]
                tBgBg_slice = tBgBg[(None, mnl[1], None, mnl[2])]

                ab_producer.reset()
                peek = ab_producer.try_acquire()
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    h = ab_producer.acquire_and_advance(peek)
                    cute.copy(tma_atom_a, tAgA_slice[(None, h.count)], tAsA[(None, h.index)],
                              tma_bar_ptr=h.barrier, mcast_mask=a_full_mcast_mask)
                    cute.copy(tma_atom_bp, tBgBp_slice[(None, h.count)], tBsBp[(None, h.index)],
                              tma_bar_ptr=h.barrier, mcast_mask=b_full_mcast_mask)
                    cute.copy(tma_atom_bg, tBgBg_slice[(None, h.count)], tBsBg[(None, h.index)],
                              tma_bar_ptr=h.barrier, mcast_mask=b_full_mcast_mask)
                    peek = cutlass.Boolean(1)
                    if h.count + 1 < k_tile_cnt:
                        peek = ab_producer.try_acquire()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            ab_producer.tail()

        # ---- MMA warp: two tcgen05 MMAs (proj, gate) into two TMEM accumulators ----
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            proj_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            col_off = const_expr(tcgen05.find_tmem_tensor_col_offset(proj_base))
            gate_base = cute.make_tensor(tmem_ptr + col_off, tCtAcc_fake.layout)

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            num_kblocks = cute.size(tCrA, mode=[2])

            while work_tile.is_valid_tile:
                proj = proj_base[(None, None, None, acc_producer_state.index)]
                gate = gate_base[(None, None, None, acc_producer_state.index)]

                ab_consumer.reset()
                peek = cutlass.Boolean(1)
                if is_leader_cta:
                    peek = ab_consumer.try_wait()
                    acc_pipeline.producer_acquire(acc_producer_state)

                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        h = ab_consumer.wait_and_advance(peek)
                        # proj accumulator (reset on first k-tile)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                        for kb in cutlass.range(num_kblocks, unroll_full=True):
                            crd = (None, None, kb, h.index)
                            cute.gemm(tiled_mma, proj, tCrA[crd], tCrBp[crd], proj)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        # gate accumulator (reset on first k-tile)
                        if const_expr(not self.proj_only):
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                            for kb in cutlass.range(num_kblocks, unroll_full=True):
                                crd = (None, None, kb, h.index)
                                cute.gemm(tiled_mma, gate, tCrA[crd], tCrBg[crd], gate)
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        h.release()
                        peek = cutlass.Boolean(1)
                        if h.count + 1 < k_tile_cnt:
                            peek = ab_consumer.try_wait()

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            acc_pipeline.producer_tail(acc_producer_state)

        # ---- Epilogue warps: fused GLU + M-major TMA store ----
        sC = smem.allocate_tensor(
            element_type=self.c_dtype, layout=c_smem_layout_staged.outer,
            byte_alignment=128, swizzle=c_smem_layout_staged.inner,
        )
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            proj_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            col_off = const_expr(tcgen05.find_tmem_tensor_col_offset(proj_base))
            gate_base = cute.make_tensor(tmem_ptr + col_off, tCtAcc_fake.layout)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 32 * len(self.epilogue_warp_id)
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage, producer_group=c_producer_group
            )

            while work_tile.is_valid_tile:
                cur = work_tile.tile_idx
                mnl = (cur[0] // cute.size(tiled_mma.thr_id.shape), cur[1], cur[2])
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                num_tiles_executed = tile_sched.num_tiles_executed
                acc_consumer_state = self._gated_epilogue(
                    tidx, warp_idx, tma_atom_c, proj_base, gate_base, sC, tCgC,
                    epi_tile, num_tiles_executed, mnl, acc_consumer_state,
                    acc_pipeline, c_pipeline,
                )

            c_pipeline.producer_tail()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

    @cute.jit
    def _gated_epilogue(
        self, epi_tidx, warp_idx, tma_atom_c, proj_base, gate_base, sC, tCgC_base,
        epi_tile, num_tiles_executed, mma_tile_coord_mnl, acc_consumer_state,
        acc_pipeline, c_pipeline,
    ):
        tCgC = transform_partitioned_tensor_layout(tCgC_base)
        tAccP = transform_partitioned_tensor_layout(proj_base)
        tAccG = transform_partitioned_tensor_layout(gate_base)

        tiled_copy_t2r, tTR_tAccP, tTR_rAccP = epilogue_tmem_copy_and_partition(
            self, epi_tidx, tAccP, tCgC, epi_tile, self.use_2cta_instrs
        )
        _, tTR_tAccG, tTR_rAccG = epilogue_tmem_copy_and_partition(
            self, epi_tidx, tAccG, tCgC, epi_tile, self.use_2cta_instrs
        )

        tTR_rC = cute.make_rmem_tensor(tTR_rAccP.shape, self.c_dtype)
        tiled_copy_r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(
            self, tiled_copy_t2r, tTR_rC, epi_tidx, sC
        )

        tCgC_epi = cute.flat_divide(tCgC, epi_tile)
        bSG_sC, bSG_gC_part = cpasync.tma_partition(
            tma_atom_c, 0, cute.make_layout(1),
            cute.group_modes(sC, 0, 2), cute.group_modes(tCgC_epi, 0, 2),
        )

        epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.epilog_sync_bar_id,
            num_threads=32 * len(self.epilogue_warp_id),
        )

        bSG_gC = bSG_gC_part[(None, None, None, *mma_tile_coord_mnl)]
        tTR_tAccP = tTR_tAccP[(None, None, None, None, None, acc_consumer_state.index)]
        tTR_tAccG = tTR_tAccG[(None, None, None, None, None, acc_consumer_state.index)]

        acc_pipeline.consumer_wait(acc_consumer_state)

        tTR_tAccP = cute.group_modes(tTR_tAccP, 3, cute.rank(tTR_tAccP))
        tTR_tAccG = cute.group_modes(tTR_tAccG, 3, cute.rank(tTR_tAccG))
        bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

        subtile_cnt = cute.size(tTR_tAccP.shape, mode=[3])
        num_prev_subtiles = num_tiles_executed * subtile_cnt

        # v16: software-pipelined epilogue. The dominant epilogue stall (ncu) is
        # long-scoreboard on the tcgen05 t2r (TMEM->reg) loads that feed the
        # sigmoid; issuing subtile k+1's t2r loads *before* computing subtile k
        # overlaps that latency with subtile k's sigmoid math + async TMA C-store.
        # Double-buffered register fragments (2x 32 fp32/thread). Occupancy is
        # smem/TMEM-capped (not register-capped) so the extra regs are free.
        # range_constexpr => python-unrolled loop so the double-buffer index is a
        # compile-time constant.
        # v17: generalize the v16 software-pipelined epilogue to DEPTH register
        # stages (v16 == DEPTH 2, i.e. prefetch k+1). DEPTH>2 issues subtile k+D-1's
        # t2r loads before computing subtile k, keeping more TMEM->reg loads in flight
        # to hide the residual long-scoreboard t2r latency at 1 CTA/SM (occupancy is
        # TMEM-capped, not register-capped, so extra fragment buffers cost no CTA).
        # env MW_EPI_DEPTH (default 2) => baseline preserved.
        depth = const_expr(self.epi_depth)
        gate_on = not (self.proj_only or not self.epi_gate)
        rAccP = [tTR_rAccP] + [cute.make_fragment_like(tTR_rAccP) for _ in range(depth - 1)]
        rAccG = [tTR_rAccG] + [cute.make_fragment_like(tTR_rAccG) for _ in range(depth - 1)]

        # prologue: prefetch the first (depth-1) subtiles' accumulators
        for j in cutlass.range_constexpr(depth - 1):
            if j < subtile_cnt:
                cute.copy(tiled_copy_t2r, tTR_tAccP[(None, None, None, j)], rAccP[j % depth])
                if const_expr(gate_on):
                    cute.copy(tiled_copy_t2r, tTR_tAccG[(None, None, None, j)], rAccG[j % depth])

        for subtile_idx in cutlass.range_constexpr(subtile_cnt):
            cur = subtile_idx % depth
            # issue subtile k+depth-1's t2r loads early so they overlap this subtile's
            # sigmoid compute + async TMA store (hides t2r long-scoreboard latency)
            pf = subtile_idx + depth - 1
            if pf < subtile_cnt:
                cute.copy(tiled_copy_t2r, tTR_tAccP[(None, None, None, pf)], rAccP[pf % depth])
                if const_expr(gate_on):
                    cute.copy(tiled_copy_t2r, tTR_tAccG[(None, None, None, pf)], rAccG[pf % depth])

            p = tiled_copy_r2s.retile(rAccP[cur]).load()
            if const_expr(not gate_on):
                ov = p
            else:
                g = tiled_copy_r2s.retile(rAccG[cur]).load()
                if const_expr(self.no_exp):
                    ov = p * g
                elif const_expr(self.sig_mode == "tanh"):
                    # sigmoid(g) = 0.5 + 0.5*tanh(0.5*g)  (exact identity, no division)
                    ov = p * (0.5 + 0.5 * cmath.tanh(0.5 * g))
                elif const_expr(self.sig_mode == "exp2nodiv"):
                    ov = p * cmath.exp2(g)
                elif const_expr(self.sig_mode == "rsqrt"):
                    # sigmoid(g)*p, division-free: 1/d == rsqrt(d)^2 (d=1+exp2(-g*log2e)>0),
                    # rsqrt -> MUFU (fast); IEEE division here costs ~0.32ms.
                    d = 1.0 + cmath.exp2(g * (-1.4426950408889634))
                    r = cmath.rsqrt(d)
                    ov = p * r * r
                else:
                    ov = p * (1.0 / (1.0 + cmath.exp2(g * (-1.4426950408889634))))
            tRS_rC.store(ov.to(self.c_dtype))

            c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
            cute.arch.fence_proxy("async.shared", space="cta")
            epilog_sync_barrier.arrive_and_wait()

            if warp_idx == self.epilogue_warp_id[0]:
                cute.copy(tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, subtile_idx)])
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()
            epilog_sync_barrier.arrive_and_wait()

        epilog_sync_barrier.arrive_and_wait()
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_consumer_state)
        acc_consumer_state.advance()
        return acc_consumer_state


_CACHE = {}


def gate_gemm(A, Bp, Bg, mmajor=False):
    """out = sigmoid(A@Bg.T) * (A@Bp.T). A:(M,K), Bp/Bg:(N,K), all bf16.

    mmajor=True: returns storage (N, M) contiguous (== M-major (M,N) view), zero-copy [B,D,L,L].
    mmajor=False: returns (M, N) row-major.
    """
    M, K = A.shape
    N, K2 = Bp.shape

    # Pad the GEMM's M (= flattened L*L) up to a multiple of the MMA tile (128). At inference
    # L is arbitrary, so M = L*L is often not tile-aligned; the persistent kernel's partial
    # last tile (tiny remainder, e.g. L=449 -> M%128==1) reads/writes out of bounds and, once
    # the caching allocator has served non-zero memory, leaks NaN. Padding removes the partial
    # tile entirely (the pad rows are zeros -> finite -> sliced off). Aligned M (training's
    # nice crops) is a no-op. Also shrinks the kernel cache (keys land on 128-multiples).
    _M_orig = M
    _TILE_M = 128
    if M % _TILE_M != 0:
        M = (M + _TILE_M - 1) // _TILE_M * _TILE_M
        A = torch.nn.functional.pad(A, (0, 0, 0, M - _M_orig))

    def _mark(t3, leading_dim):
        # (L,X,Y) torch tensor -> marked cute tensor; __call__ permutes to MNKL at trace time.
        return from_dlpack(t3, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=leading_dim)

    # A:(M,K) k-major, B:(N,K) k-major -> (1,X,K) with K contiguous (leading_dim=2)
    mA = _mark(A.detach().unsqueeze(0), 2)    # (L, M, K)
    mBp = _mark(Bp.detach().unsqueeze(0), 2)  # (L, N, K)
    mBg = _mark(Bg.detach().unsqueeze(0), 2)  # (L, N, K)

    if mmajor:
        Cnm = torch.empty(N, M, device=A.device, dtype=torch.bfloat16)  # (N, M) contiguous
        c3 = Cnm.t().unsqueeze(0)   # (1, M, N), M contiguous (leading_dim=1) -> c_major "m"
        mC = _mark(c3, 1)           # (L, M, N)
        ret = Cnm
    else:
        C = torch.empty(M, N, device=A.device, dtype=torch.bfloat16)
        c3 = C.unsqueeze(0)         # (1, M, N), N contiguous (leading_dim=2) -> c_major "n"
        mC = _mark(c3, 2)           # (L, M, N)
        ret = C

    mac = get_max_active_clusters(1)   # memoized: no per-call probe recompile
    strm = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    key = (M, N, K, mmajor)
    if key not in _CACHE:
        op = GatedPersistentGemmKernel(
            acc_dtype=Float32, use_2cta_instrs=False,
            mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 1), use_tma_store=True,
        )
        op.K = int(K)
        op.proj_only = os.environ.get("MW_PROJONLY") == "1"
        op.epi_gate = os.environ.get("MW_NOGATE_EPI") != "1"
        op.no_exp = os.environ.get("MW_NOEXP") == "1"
        op.sig_mode = os.environ.get("MW_SIG", "rsqrt")
        op.epi_depth = int(os.environ.get("MW_EPI_DEPTH", "3"))
        _CACHE[key] = cute.compile(op, mA, mBp, mBg, mC, mac, strm, options="--enable-tvm-ffi")
    _CACHE[key](mA, mBp, mBg, mC)
    if M != _M_orig:
        # slice off the padded rows; .contiguous() so the mmajor zero-copy [B,D,L,L]
        # reshape downstream still sees a dense (N, M_orig) / (M_orig, N) buffer.
        ret = (ret[:, :_M_orig] if mmajor else ret[:_M_orig]).contiguous()
    return ret
