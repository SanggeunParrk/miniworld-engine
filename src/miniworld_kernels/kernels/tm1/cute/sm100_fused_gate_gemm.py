"""v9: fused BOTH-sides SM100 gated GEMM. One launch, read A once, 4 TMEM
accumulators (proj_l, gate_l, proj_r, gate_r), write left & right [B,D,L,L] M-major.
left = sigmoid(A@WLg.T)*(A@WLp.T);  right = sigmoid(A@WRg.T)*(A@WRp.T).
"""
from __future__ import annotations
import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import BFloat16, Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum

_TILE_M = 128
_THREADS = 128


class FusedGateGemm:
    def __init__(self, N, K):
        self.N, self.K = N, K
        self.tile_m, self.tile_n, self.tile_k = _TILE_M, N, K
        self.acc_dtype = Float32
        self.ab_dtype = BFloat16
        self.cta_tile = (self.tile_m, self.tile_n, self.tile_k)
        self.cluster_shape_mnk = (1, 1, 1)
        self.num_ab_stage = 1
        self.num_acc_stage = 1
        self.threads_per_cta = _THREADS
        self.shared_storage = None
        self.num_tmem_cols = None

    @cute.kernel
    def kernel(self, tma_atom_a, tA, ab_wlp, tWLp, ab_wlg, tWLg, ab_wrp, tWRp, ab_wrg, tWRg, tCl, tCr,
               sA_layout, sB_layout, tiled_mma, epi_tile,
               a_cta_layout, b_cta_layout, cluster_layout_vmnk, tCtAcc_fake):
        tx_bytes = (self.tile_m * self.tile_k + 4 * self.tile_n * self.tile_k) * 2
        tidx, _, _ = cute.arch.thread_idx()
        m_block, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        if warp_idx == 0:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(ab_wlp)
                cpasync.prefetch_descriptor(ab_wlg)
                cpasync.prefetch_descriptor(ab_wrp)
                cpasync.prefetch_descriptor(ab_wrg)

        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar.data_ptr(), num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=tx_bytes, cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
        ).make_participants()
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(), num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_cta),
            cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
        )
        acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)
        acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)

        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=0, num_threads=self.threads_per_cta)
        tmem = utils.TmemAllocator(storage.tmem_holding.data_ptr(), barrier_for_retrieve=tmem_alloc_barrier)
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
        sWLp = storage.sWLp.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)
        sWLg = storage.sWLg.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)
        sWRp = storage.sWRp.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)
        sWRg = storage.sWRg.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)

        gA = cute.local_tile(tA, (self.tile_m, self.tile_k), (m_block, 0))
        gWLp = cute.local_tile(tWLp, (self.tile_n, self.tile_k), (0, 0))
        gWLg = cute.local_tile(tWLg, (self.tile_n, self.tile_k), (0, 0))
        gWRp = cute.local_tile(tWRp, (self.tile_n, self.tile_k), (0, 0))
        gWRg = cute.local_tile(tWRg, (self.tile_n, self.tile_k), (0, 0))
        gCl = cute.local_tile(tCl, (self.tile_m, self.tile_n), (m_block, 0))
        gCr = cute.local_tile(tCr, (self.tile_m, self.tile_n), (m_block, 0))
        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA)
        def bpart(atom, s, g):
            return cpasync.tma_partition(atom, 0, b_cta_layout, cute.group_modes(s, 0, 3),
                                         cute.group_modes(thr_mma.partition_B(g), 0, 3))
        tAsA_p, tAgA_p = cpasync.tma_partition(tma_atom_a, 0, a_cta_layout,
            cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))
        sWLp_p, gWLp_p = bpart(ab_wlp, sWLp, gWLp)
        sWLg_p, gWLg_p = bpart(ab_wlg, sWLg, gWLg)
        sWRp_p, gWRp_p = bpart(ab_wrp, sWRp, gWRp)
        sWRg_p, gWRg_p = bpart(ab_wrg, sWRg, gWRg)

        rA = tiled_mma.make_fragment_A(sA)
        rWLp = tiled_mma.make_fragment_B(sWLp)
        rWLg = tiled_mma.make_fragment_B(sWLg)
        rWRp = tiled_mma.make_fragment_B(sWRp)
        rWRg = tiled_mma.make_fragment_B(sWRg)

        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)
        tmem.allocate(self.num_tmem_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        proj_l = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        off = const_expr(tcgen05.find_tmem_tensor_col_offset(proj_l))
        gate_l = cute.make_tensor(tmem_ptr + off, tCtAcc_fake.layout)
        proj_r = cute.make_tensor(tmem_ptr + 2 * off, tCtAcc_fake.layout)
        gate_r = cute.make_tensor(tmem_ptr + 3 * off, tCtAcc_fake.layout)

        if warp_idx == 0:
            ph = ab_producer.acquire_and_advance()
            cute.copy(tma_atom_a, tAgA_p, tAsA_p[(None, ph.index)], tma_bar_ptr=ph.barrier)
            cute.copy(ab_wlp, gWLp_p, sWLp_p[(None, ph.index)], tma_bar_ptr=ph.barrier)
            cute.copy(ab_wlg, gWLg_p, sWLg_p[(None, ph.index)], tma_bar_ptr=ph.barrier)
            cute.copy(ab_wrp, gWRp_p, sWRp_p[(None, ph.index)], tma_bar_ptr=ph.barrier)
            cute.copy(ab_wrg, gWRg_p, sWRg_p[(None, ph.index)], tma_bar_ptr=ph.barrier)
            ch = ab_consumer.wait_and_advance()
            nk = cute.size(rA, mode=[2])
            for acc, rB in ((proj_l, rWLp), (gate_l, rWLg), (proj_r, rWRp), (gate_r, rWRg)):
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(nk):
                    crd = (None, None, kb, ch.index)
                    cute.gemm(tiled_mma, acc, rA[crd], rB[crd], acc)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            ch.release()
            acc_pipeline.producer_commit(acc_producer_state)

        tmem.relinquish_alloc_permit()
        acc_pipeline.consumer_wait(acc_consumer_state)

        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile, LayoutEnum.ROW_MAJOR, BFloat16, self.acc_dtype, epi_tile, False)
        def epi_src(acc):
            return thr_t2r.partition_S(cute.flat_divide(acc[((None, None), 0, 0)], epi_tile))
        tAcc0 = cute.flat_divide(proj_l[((None, None), 0, 0)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc0[(None, None, 0, 0)])
        thr_t2r = tiled_copy_t2r.get_slice(tidx)
        tPL = epi_src(proj_l); tGL = epi_src(gate_l); tPR = epi_src(proj_r); tGR = epi_src(gate_r)
        cC = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tTR_cC = thr_t2r.partition_D(cute.flat_divide(cC, epi_tile))
        rP = cute.make_rmem_tensor(tTR_cC[None, None, None, 0, 0].shape, self.acc_dtype)
        rG = cute.make_rmem_tensor(tTR_cC[None, None, None, 0, 0].shape, self.acc_dtype)
        def gpart(gC):
            return thr_t2r.partition_D(cute.flat_divide(thr_mma.partition_C(gC)[((None, None), 0, 0)], epi_tile))
        gL = gpart(gCl); gR = gpart(gCr)
        simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16)
        nem = cute.size(tAcc0, mode=[2]); nen = cute.size(tAcc0, mode=[3])
        for ei in cutlass.range_constexpr(nem):
            for ej in cutlass.range_constexpr(nen):
                cute.copy(tiled_copy_t2r, tPL[None, None, None, ei, ej], rP)
                cute.copy(tiled_copy_t2r, tGL[None, None, None, ei, ej], rG)
                ov = rP.load() * (1.0 / (1.0 + cmath.exp(-rG.load())))
                rD = cute.make_fragment_like(rP, BFloat16); rD.store(ov.to(BFloat16))
                cute.copy(simt_atom, rD, gL[None, None, None, ei, ej])
                cute.copy(tiled_copy_t2r, tPR[None, None, None, ei, ej], rP)
                cute.copy(tiled_copy_t2r, tGR[None, None, None, ei, ej], rG)
                ov2 = rP.load() * (1.0 / (1.0 + cmath.exp(-rG.load())))
                rD2 = cute.make_fragment_like(rP, BFloat16); rD2.store(ov2.to(BFloat16))
                cute.copy(simt_atom, rD2, gR[None, None, None, ei, ej])
        cute.arch.fence_view_async_tmem_load()
        pipeline.sync(barrier_id=1)
        tmem.free(tmem_ptr)

    @cute.jit
    def __call__(self, mA, mWLp, mWLg, mWRp, mWRg, mCl, mCr):
        M = mA.shape[0]
        m_blocks = M // self.tile_m
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.ab_dtype, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
            self.acc_dtype, tcgen05.CtaGroup.ONE, (self.tile_m, self.tile_n))
        cluster_layout_vmnk = cute.tiled_divide(cute.make_layout(self.cluster_shape_mnk), (tiled_mma.thr_id.shape,))
        mma_tiler = (self.tile_m, self.tile_n, self.tile_k)
        a_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler, self.ab_dtype, self.num_ab_stage)
        b_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler, self.ab_dtype, self.num_ab_stage)
        epi_tile = sm100_utils.compute_epilogue_tile_shape(self.cta_tile, False, LayoutEnum.ROW_MAJOR, BFloat16)
        a1 = cute.slice_(a_smem_layout, (None, None, None, 0))
        b1 = cute.slice_(b_smem_layout, (None, None, None, 0))
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mnk, tiled_mma.thr_id)
        tma_atom_a, tma_a = cute.nvgpu.make_tiled_tma_atom_A(a_op, mA, a1, mma_tiler, tiled_mma, cluster_layout_vmnk.shape)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mnk, tiled_mma.thr_id)
        ab_wlp, tma_wlp = cute.nvgpu.make_tiled_tma_atom_B(b_op, mWLp, b1, mma_tiler, tiled_mma, cluster_layout_vmnk.shape)
        ab_wlg, tma_wlg = cute.nvgpu.make_tiled_tma_atom_B(b_op, mWLg, b1, mma_tiler, tiled_mma, cluster_layout_vmnk.shape)
        ab_wrp, tma_wrp = cute.nvgpu.make_tiled_tma_atom_B(b_op, mWRp, b1, mma_tiler, tiled_mma, cluster_layout_vmnk.shape)
        ab_wrg, tma_wrg = cute.nvgpu.make_tiled_tma_atom_B(b_op, mWRg, b1, mma_tiler, tiled_mma, cluster_layout_vmnk.shape)
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        acc_shape = tiled_mma.partition_shape_C((self.tile_m, self.tile_n))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        self.num_tmem_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake) * 4

        sA_struct = cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(a_smem_layout)], 1024]
        sB_struct = cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(b_smem_layout)], 1024]

        @cute.struct
        class SharedStorage:
            ab_mbar: cute.struct.MemRange[cutlass.Int64, 2 * self.num_ab_stage]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, 2 * self.num_acc_stage]
            tmem_holding: cute.struct.MemRange[Int32, 1]
            sA: sA_struct
            sWLp: sB_struct
            sWLg: sB_struct
            sWRp: sB_struct
            sWRg: sB_struct
        self.shared_storage = SharedStorage
        self.kernel(tma_atom_a, tma_a, ab_wlp, tma_wlp, ab_wlg, tma_wlg, ab_wrp, tma_wrp, ab_wrg, tma_wrg, mCl, mCr,
                    a_smem_layout, b_smem_layout, tiled_mma, epi_tile,
                    a_cta_layout, b_cta_layout, cluster_layout_vmnk, tCtAcc_fake
        ).launch(grid=[m_blocks, 1, 1], block=[self.threads_per_cta, 1, 1])


_CACHE = {}
def fused_gate_gemm(A, WLp, WLg, WRp, WRg):
    M, K = A.shape; N, _ = WLp.shape
    Cl = torch.empty(N, M, device=A.device, dtype=torch.bfloat16)
    Cr = torch.empty(N, M, device=A.device, dtype=torch.bfloat16)
    d = lambda t: from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mCl = from_dlpack(Cl.t(), assumed_align=16).mark_layout_dynamic(leading_dim=0)
    mCr = from_dlpack(Cr.t(), assumed_align=16).mark_layout_dynamic(leading_dim=0)
    key = (M, N, K)
    if key not in _CACHE:
        _CACHE[key] = cute.compile(FusedGateGemm(N, K), d(A), d(WLp), d(WLg), d(WRp), d(WRg), mCl, mCr)
    _CACHE[key](d(A), d(WLp), d(WLg), d(WRp), d(WRg), mCl, mCr)
    return Cl, Cr
