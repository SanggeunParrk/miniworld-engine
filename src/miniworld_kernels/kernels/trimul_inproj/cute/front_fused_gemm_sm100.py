"""v14: FUSED front GEMM+GLU for the sm100 (B200) trimul TRAINING path.

Replaces the two-launch front (quack non-gated GEMM writing preact[4H,L,L] + a Triton
`_glu_bdll_kernel` that RE-READS preact[4H] (~1GB @ L=1024) to emit left/right[2H,L,L]) with
ONE persistent CUTLASS Blackwell GEMM whose epilogue emits BOTH outputs directly from the
in-TMEM accumulators — killing the preact re-read AND the second launch.

Built on the WORKING inference `GatedPersistentGemmKernel` (tm1.cute.sm100_gate_gemm_collective):
dual-B (share the single A load), two tcgen05 MMAs -> proj[2H] + gate[2H] TMEM accumulators,
fused-GLU TMA-store epilogue. This subclass ADDS two more TMA stores in the epilogue so the raw
pre-GLU logits are also written, in the EXACT interleaved [g,p]-per-plane (4H,M) M-major layout
that `triton.back_fused.front_bwd_dW` consumes (backward is UNCHANGED):
    gate[d] (d=0..2H-1, left|right) -> preact even planes (preact.view(4H,M)[0::2])  (strided TMA)
    proj[d]                          -> preact odd  planes (preact.view(4H,M)[1::2])  (strided TMA)
    sigmoid(gate[d])*proj[d]         -> lr[2H,M] contiguous (left = lr[:H], right = lr[H:])

Bp/Bg are the proj / gate columns of the interleaved b_lr operand:
    Bg = b_lr[:, 0::2].T   (2H, D)  = [WLg | WRg]
    Bp = b_lr[:, 1::2].T   (2H, D)  = [WL  | WR ]

Data / precision are byte-faithful to the two-launch path: bf16 in, fp32 acc, bf16 out; the
GLU sigmoid is computed in fp32 exactly as `tl.sigmoid` (exp-based, IEEE divide); preact stores
the raw bf16 logits (backward re-applies sigmoid). Algorithm unchanged — fusion only.
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

from miniworld_kernels.kernels.tm1.cute.sm100_gate_gemm_collective import (
    GatedPersistentGemmKernel,
)
from cutlass.utils.gemm.sm100 import (
    epilogue_tmem_copy_and_partition,
    epilogue_smem_copy_and_partition,
    transform_partitioned_tensor_layout,
)


class FusedPreactGemmKernel(GatedPersistentGemmKernel):
    """Gated dual-B GEMM that ALSO stores the raw pre-GLU logits (preact) interleaved."""

    # division-free sigmoid (1/d == rsqrt(d)^2, d=1+exp2(-g*log2e)>0): rsqrt -> fast MUFU.
    # The IEEE-divide "exact" path costs ~0.32ms of pure division over 2H*M elems (measured);
    # rsqrt is bit-accurate enough (left/right cos >=0.9999 vs fp32 ref, verified).
    sig_mode = "rsqrt"
    num_ab_stage_front = 2
    # When True the epilogue stores (lr, sg=σ(gate)) [2 stores] instead of (lr, gate, proj)
    # [3 stores]. The backward reconstructs the GLU grads from lr + sg alone:
    #   d_proj = d_out·σ(gate) ;  d_glogit = d_out·lr·(1-σ(gate))   (proj not needed).
    # -> forward drops the proj plane (−1/3 of stores) AND the backward reads sg[2H] instead of
    # preact[4H]. σ(gate) is already computed here (= r·r on the rsqrt path). `cpe` carries sg;
    # `cpo` is unused (still set up, never stored).
    store_sigmoid = False

    def _setup_attributes(self):
        # The front GEMM is K=D=128 -> a SINGLE k-tile, so a deep AB (load<->MMA) pipeline
        # buys nothing; but the epilogue writes THREE outputs (lr + 2x preact planes), so it
        # is store-bandwidth-bound. Reallocate smem: shallow AB (2 stages), MAX C stages ->
        # a deep TMA store pipeline that streams the 3 outputs at HBM bandwidth. (The parent
        # maxes AB, which starves the store pipeline to num_c_stage<3 and both serializes and
        # aliases the 3 per-subtile store buffers.)
        super()._setup_attributes()
        tiled_mma = self._create_tiled_mma()
        a_one = utils.sm100.make_smem_layout_a(tiled_mma, self.mma_tiler, self.a_dtype, 1)
        b_one = utils.sm100.make_smem_layout_b(tiled_mma, self.mma_tiler, self.b_dtype, 1)
        ab_bytes = cute.size_in_bytes(self.a_dtype, a_one) + 2 * cute.size_in_bytes(
            self.b_dtype, b_one)
        c_one = utils.sm100.make_smem_layout_epi(self.c_dtype, self.c_layout, self.epi_tile, 1)
        c_bytes = cute.size_in_bytes(self.c_dtype, c_one)
        mbar_bytes = 1024
        nab = self.num_ab_stage_front
        self.num_ab_stage = nab
        self.num_c_stage = (self.smem_capacity - mbar_bytes - ab_bytes * nab) // c_bytes
        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage)
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage)
        self.c_smem_layout_staged = utils.sm100.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage)

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        bp: cute.Tensor,
        bg: cute.Tensor,
        c: cute.Tensor,      # lr[2H,M] GLU output (M-major)
        cpe: cute.Tensor,    # preact even planes = gate logits (2H,M) strided M-major
        cpo: cute.Tensor,    # preact odd  planes = proj        (2H,M) strided M-major
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))     # (M,K,L)
        bp = cute.make_tensor(bp.iterator, cute.select(bp.layout, mode=[1, 2, 0]))   # (N,K,L)
        bg = cute.make_tensor(bg.iterator, cute.select(bg.layout, mode=[1, 2, 0]))   # (N,K,L)
        c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))     # (M,N,L)
        cpe = cute.make_tensor(cpe.iterator, cute.select(cpe.layout, mode=[1, 2, 0]))
        cpo = cute.make_tensor(cpo.iterator, cute.select(cpo.layout, mode=[1, 2, 0]))

        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = bp.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(bp).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        tiled_mma = self._create_tiled_mma()
        self._setup_attributes()

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = utils.sm100.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )

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

        # Three TMA store atoms (lr, preact_even, preact_odd) — same epi tile / smem layout,
        # different global descriptors (preact_* have N-stride 2M for the interleave).
        epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile
        )
        tma_atom_cpe, tma_tensor_cpe = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), cpe, epi_smem_layout, self.epi_tile
        )
        tma_atom_cpo, tma_tensor_cpo = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), cpo, epi_smem_layout, self.epi_tile
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
            tma_atom_cpe, tma_tensor_cpe,
            tma_atom_cpo, tma_tensor_cpo,
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
        tma_atom_cpe: cute.CopyAtom, mCpe_mnl: cute.Tensor,
        tma_atom_cpo: cute.CopyAtom, mCpo_mnl: cute.Tensor,
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
            cpasync.prefetch_descriptor(tma_atom_cpe)
            cpasync.prefetch_descriptor(tma_atom_cpo)

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
        gCpe_mnl = cute.local_tile(
            mCpe_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        gCpo_mnl = cute.local_tile(
            mCpo_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgBp = thr_mma.partition_B(gBp_nkl)
        tCgBg = thr_mma.partition_B(gBg_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)
        tCgCpe = thr_mma.partition_C(gCpe_mnl)
        tCgCpo = thr_mma.partition_C(gCpo_mnl)

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

        # ---- TMA load warp ----
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

        # ---- MMA warp: proj + gate MMAs into two TMEM accumulators ----
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
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                        for kb in cutlass.range(num_kblocks, unroll_full=True):
                            crd = (None, None, kb, h.index)
                            cute.gemm(tiled_mma, proj, tCrA[crd], tCrBp[crd], proj)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
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

        # ---- Epilogue warps: GLU store + two raw preact stores ----
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
                acc_consumer_state = self._preact_epilogue(
                    tidx, warp_idx, tma_atom_c, tma_atom_cpe, tma_atom_cpo,
                    proj_base, gate_base, sC, tCgC, tCgCpe, tCgCpo,
                    epi_tile, num_tiles_executed, mnl, acc_consumer_state,
                    acc_pipeline, c_pipeline,
                )

            c_pipeline.producer_tail()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

    @cute.jit
    def _preact_epilogue(
        self, epi_tidx, warp_idx, tma_atom_c, tma_atom_cpe, tma_atom_cpo,
        proj_base, gate_base, sC, tCgC_base, tCgCpe_base, tCgCpo_base,
        epi_tile, num_tiles_executed, mma_tile_coord_mnl, acc_consumer_state,
        acc_pipeline, c_pipeline,
    ):
        tCgC = transform_partitioned_tensor_layout(tCgC_base)
        tCgCpe = transform_partitioned_tensor_layout(tCgCpe_base)
        tCgCpo = transform_partitioned_tensor_layout(tCgCpo_base)
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

        # smem/global TMA partitions for the three outputs (share sC staging).
        tCgC_epi = cute.flat_divide(tCgC, epi_tile)
        bSG_sC, bSG_gC_part = cpasync.tma_partition(
            tma_atom_c, 0, cute.make_layout(1),
            cute.group_modes(sC, 0, 2), cute.group_modes(tCgC_epi, 0, 2),
        )
        tCgCpe_epi = cute.flat_divide(tCgCpe, epi_tile)
        bSG_sCpe, bSG_gCpe_part = cpasync.tma_partition(
            tma_atom_cpe, 0, cute.make_layout(1),
            cute.group_modes(sC, 0, 2), cute.group_modes(tCgCpe_epi, 0, 2),
        )
        tCgCpo_epi = cute.flat_divide(tCgCpo, epi_tile)
        bSG_sCpo, bSG_gCpo_part = cpasync.tma_partition(
            tma_atom_cpo, 0, cute.make_layout(1),
            cute.group_modes(sC, 0, 2), cute.group_modes(tCgCpo_epi, 0, 2),
        )

        epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.epilog_sync_bar_id,
            num_threads=32 * len(self.epilogue_warp_id),
        )

        bSG_gC = cute.group_modes(bSG_gC_part[(None, None, None, *mma_tile_coord_mnl)], 1,
                                  cute.rank(bSG_gC_part[(None, None, None, *mma_tile_coord_mnl)]))
        bSG_gCpe = cute.group_modes(bSG_gCpe_part[(None, None, None, *mma_tile_coord_mnl)], 1,
                                    cute.rank(bSG_gCpe_part[(None, None, None, *mma_tile_coord_mnl)]))
        bSG_gCpo = cute.group_modes(bSG_gCpo_part[(None, None, None, *mma_tile_coord_mnl)], 1,
                                    cute.rank(bSG_gCpo_part[(None, None, None, *mma_tile_coord_mnl)]))

        tTR_tAccP = tTR_tAccP[(None, None, None, None, None, acc_consumer_state.index)]
        tTR_tAccG = tTR_tAccG[(None, None, None, None, None, acc_consumer_state.index)]

        acc_pipeline.consumer_wait(acc_consumer_state)

        tTR_tAccP = cute.group_modes(tTR_tAccP, 3, cute.rank(tTR_tAccP))
        tTR_tAccG = cute.group_modes(tTR_tAccG, 3, cute.rank(tTR_tAccG))

        subtile_cnt = cute.size(tTR_tAccP.shape, mode=[3])
        num_prev_subtiles = num_tiles_executed * subtile_cnt

        rAccP = [tTR_rAccP, cute.make_fragment_like(tTR_rAccP)]
        rAccG = [tTR_rAccG, cute.make_fragment_like(tTR_rAccG)]

        cute.copy(tiled_copy_t2r, tTR_tAccP[(None, None, None, 0)], rAccP[0])
        cute.copy(tiled_copy_t2r, tTR_tAccG[(None, None, None, 0)], rAccG[0])

        # running smem-buffer counter; advances by #stores per subtile so the manual buffer
        # index stays lock-step with the c_pipeline commit/acquire count (2 = lr+sg for the
        # store_sigmoid path, else 3 = lr+gate+proj).
        sstride = 2 if const_expr(self.store_sigmoid) else 3
        store_ctr = num_prev_subtiles * sstride

        for subtile_idx in cutlass.range_constexpr(subtile_cnt):
            cur = subtile_idx % 2
            if subtile_idx + 1 < subtile_cnt:
                nxt = (subtile_idx + 1) % 2
                cute.copy(tiled_copy_t2r, tTR_tAccP[(None, None, None, subtile_idx + 1)], rAccP[nxt])
                cute.copy(tiled_copy_t2r, tTR_tAccG[(None, None, None, subtile_idx + 1)], rAccG[nxt])

            p = tiled_copy_r2s.retile(rAccP[cur]).load()
            g = tiled_copy_r2s.retile(rAccG[cur]).load()
            if const_expr(self.sig_mode == "rsqrt"):
                d = 1.0 + cmath.exp2(g * (-1.4426950408889634))
                r = cmath.rsqrt(d)
                sg = r * r                       # σ(gate)
                ov = p * sg
            else:  # "exact": exp-based sigmoid with IEEE divide (matches tl.sigmoid)
                sg = 1.0 / (1.0 + cmath.exp2(g * (-1.4426950408889634)))
                ov = p * sg

            # Fill the store buffers (lr[+sg | +gate,proj]) from reg fragments, ONE fence + ONE
            # barrier pair, then issue the overlapping TMA stores. Keeping the epilogue short lets
            # the persistent scheduler hide it under the next tile's mainloop (per-store barriers
            # serialized it -> 4x regression).
            b0 = (store_ctr + subtile_idx * sstride + 0) % self.num_c_stage
            b1 = (store_ctr + subtile_idx * sstride + 1) % self.num_c_stage
            tRS_rC.store(ov.to(self.c_dtype))
            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, b0)])
            if const_expr(self.store_sigmoid):
                tRS_rC.store(sg.to(self.c_dtype))
                cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, b1)])
            else:
                b2 = (store_ctr + subtile_idx * sstride + 2) % self.num_c_stage
                tRS_rC.store(g.to(self.c_dtype))
                cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, b1)])
                tRS_rC.store(p.to(self.c_dtype))
                cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, b2)])
            cute.arch.fence_proxy("async.shared", space="cta")
            epilog_sync_barrier.arrive_and_wait()
            if warp_idx == self.epilogue_warp_id[0]:
                cute.copy(tma_atom_c, bSG_sC[(None, b0)], bSG_gC[(None, subtile_idx)])
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()
                cute.copy(tma_atom_cpe, bSG_sCpe[(None, b1)], bSG_gCpe[(None, subtile_idx)])
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()
                if const_expr(not self.store_sigmoid):
                    cute.copy(tma_atom_cpo, bSG_sCpo[(None, b2)], bSG_gCpo[(None, subtile_idx)])
                    c_pipeline.producer_commit()
                    c_pipeline.producer_acquire()
            epilog_sync_barrier.arrive_and_wait()

        epilog_sync_barrier.arrive_and_wait()
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_consumer_state)
        acc_consumer_state.advance()
        return acc_consumer_state


_CACHE = {}


def fused_front_gemm(A, Bp, Bg, lr, preact):
    """lr[2H,M] = sigmoid(A@Bg.T)*(A@Bp.T); preact[4H,M] interleaved: even=A@Bg.T, odd=A@Bp.T.
    A:(M,K) bf16; Bp/Bg:(N=2H,K) bf16; lr:(2H,M) contiguous; preact:(4H,M) contiguous — all
    written M-major (M contiguous). No return; writes in place (caller owns the buffers)."""
    M, K = A.shape
    N, K2 = Bp.shape

    def _mark(t3, leading_dim):
        return from_dlpack(t3, assumed_align=16).mark_layout_dynamic(leading_dim=leading_dim)

    mA = _mark(A.detach().unsqueeze(0), 2)     # (L, M, K)
    mBp = _mark(Bp.detach().unsqueeze(0), 2)   # (L, N, K)
    mBg = _mark(Bg.detach().unsqueeze(0), 2)   # (L, N, K)

    # lr storage: (N, M) contiguous == M-major (M,N) view
    c3 = lr.t().unsqueeze(0)                    # (1, M, N), M contiguous (leading_dim=1)
    mC = _mark(c3, 1)
    # preact interleave views: even planes = gate, odd = proj — each (N, M) with row-stride 2M
    pe = preact[0::2]                           # (N, M) strides (2M, 1)
    po = preact[1::2]                           # (N, M) strides (2M, 1)
    mCpe = _mark(pe.t().unsqueeze(0), 1)        # (1, M, N), M contiguous, N-stride 2M
    mCpo = _mark(po.t().unsqueeze(0), 1)

    mac = utils.HardwareInfo().get_max_active_clusters(1)
    strm = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key = (M, N, K)
    if key not in _CACHE:
        op = FusedPreactGemmKernel(
            acc_dtype=Float32, use_2cta_instrs=False,
            mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 1), use_tma_store=True,
        )
        op.K = int(K)
        _CACHE[key] = cute.compile(op, mA, mBp, mBg, mC, mCpe, mCpo, mac, strm)
    _CACHE[key](mA, mBp, mBg, mC, mCpe, mCpo, mac, strm)


class FusedSigGemmKernel(FusedPreactGemmKernel):
    """store_sigmoid variant: the epilogue writes (lr[2H,M], sg=σ(gate)[2H,M]) — NO proj plane.
    The backward (`front_bwd_dW_sig`) reconstructs the GLU grads from lr + sg alone:
        d_proj = d_out·sg ;  d_glogit = d_out·lr·(1-sg).
    ~1/3 fewer forward store bytes than the preact[4H] path; sg is carried in the `cpe` slot and
    the `cpo` slot is unused (still set up, never stored)."""

    store_sigmoid = True


_CACHE_SIG = {}


def fused_front_gemm_sig(A, Bp, Bg, lr, sg):
    """lr[2H,M] = σ(A@Bg.T)·(A@Bp.T);  sg[2H,M] = σ(A@Bg.T). Both (2H,M) contiguous (M-major).
    A:(M,K) bf16; Bp/Bg:(2H,K) bf16. Writes in place. Replaces preact[4H] with sg[2H]."""
    M, K = A.shape
    N, _ = Bp.shape

    def _mark(t3, leading_dim):
        return from_dlpack(t3, assumed_align=16).mark_layout_dynamic(leading_dim=leading_dim)

    mA = _mark(A.detach().unsqueeze(0), 2)
    mBp = _mark(Bp.detach().unsqueeze(0), 2)
    mBg = _mark(Bg.detach().unsqueeze(0), 2)
    mC = _mark(lr.t().unsqueeze(0), 1)             # (1, M, 2H) M-major
    mSg = _mark(sg.t().unsqueeze(0), 1)            # (1, M, 2H) M-major  (carried as cpe; cpo unused)
    mac = utils.HardwareInfo().get_max_active_clusters(1)
    strm = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key = (M, N, K)
    if key not in _CACHE_SIG:
        op = FusedSigGemmKernel(
            acc_dtype=Float32, use_2cta_instrs=False,
            mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 1), use_tma_store=True,
        )
        op.K = int(K)
        _CACHE_SIG[key] = cute.compile(op, mA, mBp, mBg, mC, mSg, mSg, mac, strm)
    _CACHE_SIG[key](mA, mBp, mBg, mC, mSg, mSg, mac, strm)
