"""sm100 (B200) fused expand + SwiGLU-gate-BACKWARD GEMM.

Replaces the register-spilling Triton ``_transition_expand_gatebwd_kernel`` (50% of the
transition training step). Reuses the tuned dual-B gated GEMM collective
(``GatedPersistentGemmKernel``): the SAME dual-accumulator mainloop recomputes

    a = xn @ wa^T   (gate accumulator)     b = xn @ wb^T   (proj accumulator)

and the epilogue is swapped to the gate-backward math. Given the per-pair upstream grad
``ge = grad_expand`` (M, ND), the epilogue emits THREE bf16 outputs:

    sig  = sigmoid(a) ;  silu = a*sig
    h    = silu * b                         (for dWs = grad_out^T @ h)
    dA   = ge * b * (sig + silu*(1-sig))    (silu'(a) = sig + silu*(1-sig))
    dB   = ge * silu

Structural additions vs the forward collective (b2b_fwd_sm100.SwiGLUExpandKernel):
  * Extra INPUT ``grad_expand`` (M, ND): register-direct global load in the epilogue.
    The grad gmem tensor is partitioned with the SAME t2r thread-copy that lands the
    accumulator in registers (``thr_copy_t2r.partition_D``), so each thread's grad
    elements arrive already aligned to its a/b register fragment — no smem needed.
  * THREE OUTPUTS h/dA/dB: three TMA-store atoms + three smem C-stage buffers, driven
    in lockstep by a single TMA-store pipeline (all share the c_buffer stage index).

xn:(M,K), wa/wb:(ND,K), grad_expand:(M,ND) all bf16, row-major. M and ND are multiples of
128 for the transition shapes (L=384 -> M=147456; ND in {512,1024}) so no OOB predication.
"""

from __future__ import annotations
from miniworld_engine.autotune.configs import configs_for
import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch

from miniworld_engine.kernels._compile import opaque
from quack.cute_dsl_utils import get_max_active_clusters
from cutlass import BFloat16, Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.gemm.sm100 import (
    epilogue_tmem_copy_and_partition,
    epilogue_smem_copy_and_partition,
    transform_partitioned_tensor_layout,
)

from miniworld_engine.kernels.tm1.cute.sm100_gate_gemm_collective import (
    GatedPersistentGemmKernel,
)


class SwiGLUGateBwdKernel(GatedPersistentGemmKernel):
    """Dual-B gated GEMM whose epilogue computes the SwiGLU gate-backward (h, dA, dB)."""

    epi_depth = 3
    sig_mode = "rsqrt"
    no_grad = False   # MW_NOGRAD / MW_SPLIT: skip in-kernel grad load (ge=1); split fallback
    one_out = False   # MW_ONEOUT: store only h to isolate extra-store cost

    def _setup_attributes(self):
        # Inherit the collective's dual-B / dual-TMEM setup, then re-solve the smem stage
        # split for THREE C output buffers (h, dA, dB) instead of one.
        super()._setup_attributes()
        tiled_mma = self._create_tiled_mma()

        a_one = utils.sm100.make_smem_layout_a(tiled_mma, self.mma_tiler, self.a_dtype, 1)
        b_one = utils.sm100.make_smem_layout_b(tiled_mma, self.mma_tiler, self.b_dtype, 1)
        ab_bytes = cute.size_in_bytes(self.a_dtype, a_one) + 2 * cute.size_in_bytes(
            self.b_dtype, b_one
        )
        c_smem_one = utils.sm100.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, 1
        )
        c_bytes_ps = cute.size_in_bytes(self.c_dtype, c_smem_one)
        n_out = 3
        mbar_bytes = 1024

        # Grad input is TMA-loaded (coalesced) into a full-CTA-tile smem buffer, then
        # s2r-read aligned to the accumulator registers. One buffer of the full CTA tile.
        # subtile count from tile GEOMETRY (not byte division) and tx bytes = per-subtile
        # swizzled smem bytes * count, so expected-tx exactly matches TMA-deposited bytes.
        epi_m = cute.size(self.epi_tile[0])
        epi_n = cute.size(self.epi_tile[1])
        self.epi_subtile_cnt = (
            (self.cta_tile_shape_mnk[0] // epi_m) * (self.cta_tile_shape_mnk[1] // epi_n)
        )
        # Grad is TMA-loaded (coalesced) by the TMA warp into a staged smem buffer via a
        # producer(TMA-warp)->consumer(epilogue-warps) PipelineTmaAsync, then s2r-read in the
        # epilogue (canonical alpha-beta-gemm C-load pattern). One PIPELINE STAGE = one epi
        # subtile (c_bytes_ps swizzled bytes). num_grad_stage == epi_subtile_cnt so the TMA
        # warp can issue a whole tile's grad without blocking mid-tile.
        self.num_grad_stage = self.epi_subtile_cnt
        self.grad_tx_per_stage = c_bytes_ps       # per-subtile TMA tx (== ref tma_c_load_bytes)
        self.grad_tx_total = self.epi_subtile_cnt * c_bytes_ps
        self.ge_smem_layout_staged = utils.sm100.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.num_grad_stage
        )
        grad_bytes = self.num_grad_stage * c_bytes_ps

        self.num_c_stage = 2
        num_ab_stage = (
            self.smem_capacity
            - (mbar_bytes + grad_bytes + n_out * c_bytes_ps * self.num_c_stage)
        ) // ab_bytes
        if num_ab_stage < 1:
            num_ab_stage = 1
        self.num_ab_stage = num_ab_stage
        used = (
            ab_bytes * self.num_ab_stage
            + mbar_bytes + grad_bytes + n_out * c_bytes_ps * self.num_c_stage
        )
        self.num_c_stage += (self.smem_capacity - used) // (n_out * c_bytes_ps)

        if settings.current().sm100_setup_debug:
            print(f"[SETUP] epi_tile={self.epi_tile} cta={self.cta_tile_shape_mnk} "
                  f"epi_m={epi_m} epi_n={epi_n} logical_per_sub={epi_m*epi_n*2} "
                  f"c_bytes_ps={c_bytes_ps} epi_subtile_cnt={self.epi_subtile_cnt} "
                  f"grad_tx_total={self.grad_tx_total} expect_logical={self.epi_subtile_cnt*epi_m*epi_n*2} "
                  f"num_ab={self.num_ab_stage} num_c={self.num_c_stage} num_acc={self.num_acc_stage} "
                  f"ge_smem_bytes={cute.size_in_bytes(self.c_dtype, self.ge_smem_layout_staged)}",
                  flush=True)
        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )
        self.c_smem_layout_staged = utils.sm100.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        bp: cute.Tensor,   # wb -> proj = b (the "up")
        bg: cute.Tensor,   # wa -> gate = a (the silu operand)
        ge: cute.Tensor,   # grad_expand (M, ND)
        c_h: cute.Tensor,   # output h  (M, ND)
        c_da: cute.Tensor,  # output dA (M, ND)
        c_db: cute.Tensor,  # output dB (M, ND)
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        # (L, X, Y) -> MNKL (X, Y, L)
        a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))    # (M,K,L)
        bp = cute.make_tensor(bp.iterator, cute.select(bp.layout, mode=[1, 2, 0]))  # (N,K,L)
        bg = cute.make_tensor(bg.iterator, cute.select(bg.layout, mode=[1, 2, 0]))  # (N,K,L)
        ge = cute.make_tensor(ge.iterator, cute.select(ge.layout, mode=[1, 2, 0]))  # (M,N,L)
        c_h = cute.make_tensor(c_h.iterator, cute.select(c_h.layout, mode=[1, 2, 0]))
        c_da = cute.make_tensor(c_da.iterator, cute.select(c_da.layout, mode=[1, 2, 0]))
        c_db = cute.make_tensor(c_db.iterator, cute.select(c_db.layout, mode=[1, 2, 0]))

        self.a_dtype = a.element_type
        self.b_dtype = bp.element_type
        self.c_dtype = c_h.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(bp).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c_h)

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

        # TMA store atoms for the three outputs (identical layout, different descriptors).
        epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
        tma_atom_h, tma_tensor_h = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c_h, epi_smem_layout, self.epi_tile
        )
        tma_atom_da, tma_tensor_da = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c_da, epi_smem_layout, self.epi_tile
        )
        tma_atom_db, tma_tensor_db = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c_db, epi_smem_layout, self.epi_tile
        )

        # TMA LOAD atom for grad_expand (coalesced gmem->smem), epi-tiled like the outputs.
        tma_atom_ge, tma_tensor_ge = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), ge, epi_smem_layout, self.epi_tile
        )

        self.tile_sched_params, grid = self._compute_grid(
            c_h, self.cta_tile_shape_mnk, self.cluster_shape_mn, max_active_clusters
        )

        self.kernel(
            tiled_mma,
            tma_atom_a, tma_tensor_a,
            tma_atom_bp, tma_tensor_bp,
            tma_atom_bg, tma_tensor_bg,
            tma_atom_ge, tma_tensor_ge,
            tma_atom_h, tma_tensor_h,
            tma_atom_da, tma_tensor_da,
            tma_atom_db, tma_tensor_db,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.ge_smem_layout_staged,
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
        tma_atom_ge: cute.CopyAtom, mGe_mnl: cute.Tensor,
        tma_atom_h: cute.CopyAtom, mH_mnl: cute.Tensor,
        tma_atom_da: cute.CopyAtom, mDA_mnl: cute.Tensor,
        tma_atom_db: cute.CopyAtom, mDB_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
        ge_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_bp)
            cpasync.prefetch_descriptor(tma_atom_bg)
            cpasync.prefetch_descriptor(tma_atom_ge)
            cpasync.prefetch_descriptor(tma_atom_h)
            cpasync.prefetch_descriptor(tma_atom_da)
            cpasync.prefetch_descriptor(tma_atom_db)

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
            grad_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_grad_stage * 2]
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

        # Grad G2S pipeline: TMA-warp producer (1 thread) -> epilogue-warps consumer
        # (len(epilogue_warp_id) signalling lanes). Canonical PipelineTmaAsync (Hopper/
        # alpha-beta-gemm C-load). Created here (before pipeline_init_arrive) so its
        # mbarriers are fenced+cluster-published with the ab/acc pipelines. The G2S is
        # issued by the TMA warp (the PROVEN tx-delivering warp) — issuing it from the
        # epilogue warp silently delivered 0 tx and hung (root cause).
        grad_pipeline = None
        if const_expr(not self.no_grad):
            grad_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            grad_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.epilogue_warp_id)
            )
            grad_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.grad_full_mbar_ptr.data_ptr(),
                num_stages=self.num_grad_stage,
                producer_group=grad_producer_group,
                consumer_group=grad_consumer_group,
                tx_count=self.grad_tx_per_stage,
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
        # Grad staged smem — allocated in the COMMON section (all warps, same offset) so the
        # TMA warp (producer) can address it; the epilogue warps (consumer) s2r-read it.
        sGe = smem.allocate_tensor(
            element_type=self.c_dtype, layout=ge_smem_layout_staged.outer,
            byte_alignment=128, swizzle=ge_smem_layout_staged.inner,
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
        gH_mnl = cute.local_tile(
            mH_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        gGe_mnl = cute.local_tile(
            mGe_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgBp = thr_mma.partition_B(gBp_nkl)
        tCgBg = thr_mma.partition_B(gBg_nkl)
        tCgC = thr_mma.partition_C(gH_mnl)     # output coords (shared by h/dA/dB)
        tCgGe = thr_mma.partition_C(gGe_mnl)   # grad input coords

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

        # ---- TMA load warp: A, Bp, Bg into one AB buffer per tile + grad G2S producer ----
        if warp_idx == self.tma_warp_id:
            # Grad producer partition (ONCE, outside the persistent loop — tma_partition in a
            # loop fails to legalize). One TMA subtile per grad pipeline stage.
            if const_expr(not self.no_grad):
                tCgGe_epi = cute.flat_divide(
                    transform_partitioned_tensor_layout(tCgGe), epi_tile)
                bSG_sGe, bSG_gGe_part = cpasync.tma_partition(
                    tma_atom_ge, 0, cute.make_layout(1),
                    cute.group_modes(sGe, 0, 2), cute.group_modes(tCgGe_epi, 0, 2),
                )
                grad_prod_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_grad_stage
                )
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

                # grad G2S: one coalesced TMA per epi subtile into its own pipeline stage.
                if const_expr(not self.no_grad):
                    bSG_gGe_t = bSG_gGe_part[(None, None, None, *mnl)]
                    bSG_gGe = cute.group_modes(bSG_gGe_t, 1, cute.rank(bSG_gGe_t))
                    for st in cutlass.range_constexpr(self.epi_subtile_cnt):
                        grad_pipeline.producer_acquire(grad_prod_state)
                        cute.copy(
                            tma_atom_ge, bSG_gGe[(None, st)],
                            bSG_sGe[(None, grad_prod_state.index)],
                            tma_bar_ptr=grad_pipeline.producer_get_barrier(grad_prod_state),
                        )
                        grad_prod_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            ab_producer.tail()
            if const_expr(not self.no_grad):
                grad_pipeline.producer_tail(grad_prod_state)

        # ---- MMA warp: two tcgen05 MMAs (proj=b, gate=a) into two TMEM accumulators ----
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

        # ---- Epilogue warps: gate-backward math + three M-major TMA stores ----
        sH = smem.allocate_tensor(
            element_type=self.c_dtype, layout=c_smem_layout_staged.outer,
            byte_alignment=128, swizzle=c_smem_layout_staged.inner,
        )
        sDA = smem.allocate_tensor(
            element_type=self.c_dtype, layout=c_smem_layout_staged.outer,
            byte_alignment=128, swizzle=c_smem_layout_staged.inner,
        )
        sDB = smem.allocate_tensor(
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

            # Grad consumer state (persists across tiles; advances one per epi subtile in
            # lockstep with the TMA-warp producer).
            grad_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_grad_stage
            )

            while work_tile.is_valid_tile:
                cur = work_tile.tile_idx
                mnl = (cur[0] // cute.size(tiled_mma.thr_id.shape), cur[1], cur[2])
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                num_tiles_executed = tile_sched.num_tiles_executed

                acc_consumer_state, grad_consumer_state = self._gatebwd_epilogue(
                    tidx, warp_idx,
                    tma_atom_h, tma_atom_da, tma_atom_db,
                    proj_base, gate_base,
                    sH, sDA, sDB, sGe, tCgC,
                    epi_tile, num_tiles_executed, mnl, acc_consumer_state,
                    acc_pipeline, c_pipeline, grad_pipeline, grad_consumer_state,
                )

            c_pipeline.producer_tail()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

    @cute.jit
    def _gatebwd_epilogue(
        self, epi_tidx, warp_idx,
        tma_atom_h, tma_atom_da, tma_atom_db,
        proj_base, gate_base,
        sH, sDA, sDB, sGe, tCgC_base,
        epi_tile, num_tiles_executed, mma_tile_coord_mnl, acc_consumer_state,
        acc_pipeline, c_pipeline, grad_pipeline, grad_consumer_state,
    ):
        tCgC = transform_partitioned_tensor_layout(tCgC_base)
        tAccP = transform_partitioned_tensor_layout(proj_base)   # b (up)
        tAccG = transform_partitioned_tensor_layout(gate_base)   # a (gate)

        tiled_copy_t2r, tTR_tAccP, tTR_rAccP = epilogue_tmem_copy_and_partition(
            self, epi_tidx, tAccP, tCgC, epi_tile, self.use_2cta_instrs
        )
        _, tTR_tAccG, tTR_rAccG = epilogue_tmem_copy_and_partition(
            self, epi_tidx, tAccG, tCgC, epi_tile, self.use_2cta_instrs
        )

        # Three register C fragments + smem partitions (one tiled r2s copy, reused).
        tTR_rH = cute.make_rmem_tensor(tTR_rAccP.shape, self.c_dtype)
        tiled_copy_r2s, tRS_rH, tRS_sH = epilogue_smem_copy_and_partition(
            self, tiled_copy_t2r, tTR_rH, epi_tidx, sH
        )
        tTR_rDA = cute.make_rmem_tensor(tTR_rAccP.shape, self.c_dtype)
        _, tRS_rDA, tRS_sDA = epilogue_smem_copy_and_partition(
            self, tiled_copy_t2r, tTR_rDA, epi_tidx, sDA
        )
        tTR_rDB = cute.make_rmem_tensor(tTR_rAccP.shape, self.c_dtype)
        _, tRS_rDB, tRS_sDB = epilogue_smem_copy_and_partition(
            self, tiled_copy_t2r, tTR_rDB, epi_tidx, sDB
        )
        # grad_expand read: the TMA warp COALESCED-loaded each epi subtile into a grad
        # pipeline stage; s2r-read it per subtile in the SAME r2s register layout as the
        # h/dA/dB outputs (proven correct by the split path stores). partition_D ONCE here.
        tRS_sGe = tiled_copy_r2s.get_slice(epi_tidx).partition_D(sGe)

        tCgC_epi = cute.flat_divide(tCgC, epi_tile)
        bSG_sH, bSG_gH_part = cpasync.tma_partition(
            tma_atom_h, 0, cute.make_layout(1),
            cute.group_modes(sH, 0, 2), cute.group_modes(tCgC_epi, 0, 2),
        )
        bSG_sDA, bSG_gDA_part = cpasync.tma_partition(
            tma_atom_da, 0, cute.make_layout(1),
            cute.group_modes(sDA, 0, 2), cute.group_modes(tCgC_epi, 0, 2),
        )
        bSG_sDB, bSG_gDB_part = cpasync.tma_partition(
            tma_atom_db, 0, cute.make_layout(1),
            cute.group_modes(sDB, 0, 2), cute.group_modes(tCgC_epi, 0, 2),
        )

        epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.epilog_sync_bar_id,
            num_threads=32 * len(self.epilogue_warp_id),
        )

        bSG_gH = cute.group_modes(bSG_gH_part[(None, None, None, *mma_tile_coord_mnl)], 1, cute.rank(bSG_gH_part[(None, None, None, *mma_tile_coord_mnl)]))
        bSG_gDA = cute.group_modes(bSG_gDA_part[(None, None, None, *mma_tile_coord_mnl)], 1, cute.rank(bSG_gDA_part[(None, None, None, *mma_tile_coord_mnl)]))
        bSG_gDB = cute.group_modes(bSG_gDB_part[(None, None, None, *mma_tile_coord_mnl)], 1, cute.rank(bSG_gDB_part[(None, None, None, *mma_tile_coord_mnl)]))
        tTR_tAccP = tTR_tAccP[(None, None, None, None, None, acc_consumer_state.index)]
        tTR_tAccG = tTR_tAccG[(None, None, None, None, None, acc_consumer_state.index)]

        acc_pipeline.consumer_wait(acc_consumer_state)

        tTR_tAccP = cute.group_modes(tTR_tAccP, 3, cute.rank(tTR_tAccP))
        tTR_tAccG = cute.group_modes(tTR_tAccG, 3, cute.rank(tTR_tAccG))

        subtile_cnt = cute.size(tTR_tAccP.shape, mode=[3])
        num_prev_subtiles = num_tiles_executed * subtile_cnt

        depth = const_expr(self.epi_depth)
        rAccP = [tTR_rAccP] + [cute.make_fragment_like(tTR_rAccP) for _ in range(depth - 1)]
        rAccG = [tTR_rAccG] + [cute.make_fragment_like(tTR_rAccG) for _ in range(depth - 1)]
        rGe = cute.make_fragment_like(tRS_rH)  # one subtile's grad (smem->reg, r2s layout, bf16)

        for j in cutlass.range_constexpr(depth - 1):
            if j < subtile_cnt:
                cute.copy(tiled_copy_t2r, tTR_tAccP[(None, None, None, j)], rAccP[j % depth])
                cute.copy(tiled_copy_t2r, tTR_tAccG[(None, None, None, j)], rAccG[j % depth])

        for subtile_idx in cutlass.range_constexpr(subtile_cnt):
            cur = subtile_idx % depth
            pf = subtile_idx + depth - 1
            if pf < subtile_cnt:
                cute.copy(tiled_copy_t2r, tTR_tAccP[(None, None, None, pf)], rAccP[pf % depth])
                cute.copy(tiled_copy_t2r, tTR_tAccG[(None, None, None, pf)], rAccG[pf % depth])

            b = tiled_copy_r2s.retile(rAccP[cur]).load()   # proj = xn@wb (up)
            a = tiled_copy_r2s.retile(rAccG[cur]).load()   # gate = xn@wa (silu operand)

            if const_expr(self.no_grad):
                ge = 1.0
            else:
                # wait for the TMA-warp G2S of this subtile's grad, s2r-read it, release the
                # stage back to the producer (mirror the alpha-beta-gemm C-load consumer).
                grad_pipeline.consumer_wait(grad_consumer_state)
                cute.autovec_copy(
                    tRS_sGe[(None, None, None, grad_consumer_state.index)], rGe)
                cute.arch.fence_proxy("async.shared", space="cta")
                grad_pipeline.consumer_release(grad_consumer_state)
                grad_consumer_state.advance()
                ge = rGe.load().to(Float32)

            # sigmoid(a): division-free via rsqrt (1/d == rsqrt(d)^2, d=1+exp2(-a*log2e)).
            if const_expr(self.sig_mode == "rsqrt"):
                d = 1.0 + cmath.exp2(a * (-1.4426950408889634))
                r = cmath.rsqrt(d)
                sig = r * r
            else:
                sig = 1.0 / (1.0 + cmath.exp2(a * (-1.4426950408889634)))
            silu = a * sig
            h = silu * b
            dsilu = sig + silu * (1.0 - sig)   # silu'(a)
            dA = ge * b * dsilu
            dB = ge * silu

            tRS_rH.store(h.to(self.c_dtype))
            if const_expr(not self.one_out):
                tRS_rDA.store(dA.to(self.c_dtype))
                tRS_rDB.store(dB.to(self.c_dtype))

            c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
            cute.copy(tiled_copy_r2s, tRS_rH, tRS_sH[(None, None, None, c_buffer)])
            if const_expr(not self.one_out):
                cute.copy(tiled_copy_r2s, tRS_rDA, tRS_sDA[(None, None, None, c_buffer)])
                cute.copy(tiled_copy_r2s, tRS_rDB, tRS_sDB[(None, None, None, c_buffer)])
            cute.arch.fence_proxy("async.shared", space="cta")
            epilog_sync_barrier.arrive_and_wait()

            if warp_idx == self.epilogue_warp_id[0]:
                cute.copy(tma_atom_h, bSG_sH[(None, c_buffer)], bSG_gH[(None, subtile_idx)])
                if const_expr(not self.one_out):
                    cute.copy(tma_atom_da, bSG_sDA[(None, c_buffer)], bSG_gDA[(None, subtile_idx)])
                    cute.copy(tma_atom_db, bSG_sDB[(None, c_buffer)], bSG_gDB[(None, subtile_idx)])
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()
            epilog_sync_barrier.arrive_and_wait()

        epilog_sync_barrier.arrive_and_wait()
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_consumer_state)
        acc_consumer_state.advance()
        return acc_consumer_state, grad_consumer_state


import triton
import triton.language as tl

from miniworld_engine import settings



from miniworld_engine.autotune.buckets import bucket_mixed as _bucket
from miniworld_engine.autotune.shape_key import both_key, rows_of


def get_seq_group(rows) -> int:
    """Delegates to canonical size-bucketing (autotune.buckets)."""
    return _bucket(rows)


# IN-PLACE (dA/dB are read and written), so the autotuner must restore them between benched
# configs — otherwise each candidate multiplies the gradients by `ge` again and the sweep itself
# corrupts them. See the same note on tm1/cute/launch::_gate_mul_kernel.
@triton.autotune(configs=configs_for("transition_bwd_epilogue_triton"),
                 key=['shape_key'],
                 restore_value=['dA_ptr', 'dB_ptr'])
@triton.jit
def _grad_mul_kernel(dA_ptr, dB_ptr, ge_ptr, N, BLOCK_E: tl.constexpr, shape_key):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    m = offs < N
    ge = tl.load(ge_ptr + offs, mask=m)          # read grad once
    tl.store(dA_ptr + offs, tl.load(dA_ptr + offs, mask=m) * ge, mask=m)
    tl.store(dB_ptr + offs, tl.load(dB_ptr + offs, mask=m) * ge, mask=m)


@opaque(fake=lambda dA, dB, ge, shape_key=None: None,
        name="transition_grad_mul_inplace", mutates_args=("dA", "dB"))
def _grad_mul_inplace(dA: torch.Tensor, dB: torch.Tensor, ge: torch.Tensor,
                      shape_key: int | None = None) -> None:
    """dA *= ge; dB *= ge  in one pass (grad read once). All (M,ND) bf16 contiguous."""
    N = dA.numel()
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_E"]),)  # noqa: E731
    # N here is a FLAT ELEMENT COUNT (dA.numel()), not even a row count -- there is nothing
    # in it to recover L from, so the key comes from the caller: both_key(rows_of(<pre-
    # flatten shape>)). None = drivers_trans / checks_trans (coordinator-owned).
    _grad_mul_kernel[grid](dA, dB, ge, N,
                           shape_key=both_key(N) if shape_key is None else shape_key)


_CACHE = {}


def transition_expand_gatebwd_sm100(xn, wa, wb, grad_expand, *, shape_key: int | None = None):
    """SwiGLU gate-backward. xn:(M,K), wa/wb:(ND,K), grad_expand:(M,ND) bf16.
    Returns (h, dA, dB) each (M,ND) bf16, where a=xn@wa^T, b=xn@wb^T,
    h=silu(a)*b, dA=grad_expand*b*silu'(a), dB=grad_expand*silu(a)."""
    M, K = xn.shape
    ND = wa.shape[0]

    def _mark(t3, leading_dim):
        return from_dlpack(t3, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=leading_dim)

    mA = _mark(xn.detach().unsqueeze(0), 2)          # (L, M, K)
    mBp = _mark(wb.detach().unsqueeze(0), 2)         # (L, N, K)  proj = up
    mBg = _mark(wa.detach().unsqueeze(0), 2)         # (L, N, K)  gate = silu operand
    mGe = _mark(grad_expand.detach().unsqueeze(0), 2)  # (L, M, N)

    h = torch.empty(M, ND, device=xn.device, dtype=torch.bfloat16)
    dA = torch.empty(M, ND, device=xn.device, dtype=torch.bfloat16)
    dB = torch.empty(M, ND, device=xn.device, dtype=torch.bfloat16)
    mH = _mark(h.unsqueeze(0), 2)
    mDA = _mark(dA.unsqueeze(0), 2)
    mDB = _mark(dB.unsqueeze(0), 2)

    mac = get_max_active_clusters(1)  # memoized: avoid per-call CUTLASS-DSL probe recompile
    strm = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    key = (M, ND, K)
    if key not in _CACHE:
        op = SwiGLUGateBwdKernel(
            acc_dtype=Float32, use_2cta_instrs=False,
            mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 1), use_tma_store=True,
        )
        op.K = int(K)
        op.sig_mode = settings.current().sm100_sig_mode
        op.epi_depth = settings.current().sm100_epi_depth or 1
        # FUSED path (DEFAULT): the grad is TMA-warp-loaded COALESCED into a staged smem
        # buffer (PipelineTmaAsync producer=TMA warp, consumer=epilogue warps) and s2r-read
        # in the epilogue, so dA/dB are emitted directly (no extra elementwise pass). This
        # beats the old SPLIT fallback (~1.45x -> ~2.1x standalone). Set MW_SPLIT=1 to fall
        # back to the split path (kernel no_grad + coalesced elementwise grad-multiply).
        _fused = not settings.current().sm100_split
        op.no_grad = (not _fused) or settings.current().sm100_no_grad
        op.one_out = settings.current().sm100_one_out
        _CACHE[key] = (cute.compile(op, mA, mBp, mBg, mGe, mH, mDA, mDB, mac, strm, options="--enable-tvm-ffi"), op.no_grad)
    compiled, _split = _CACHE[key]
    compiled(mA, mBp, mBg, mGe, mH, mDA, mDB)
    if _split:
        # dA = grad_expand * (b*silu'(a)); dB = grad_expand * silu(a) — one fused pass
        # (grad read once) instead of two torch muls.
        _grad_mul_inplace(dA, dB, grad_expand, shape_key=shape_key)
    return h, dA, dB
