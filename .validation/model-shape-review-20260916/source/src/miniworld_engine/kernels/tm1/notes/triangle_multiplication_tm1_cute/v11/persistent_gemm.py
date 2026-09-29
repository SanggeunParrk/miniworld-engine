"""v11: persistent, warp-specialized tcgen05 GEMM (2D, single output C=A@B.T).
Warps: epilogue 0-3 (128t), MMA 4, TMA 5 (192 threads). Persistent tile scheduler
over M-tiles; num_acc_stage=2 double-buffers so epilogue(tile i) overlaps MMA(tile i+1).
Built on proven 4.4.2 primitives (micro_mma.py) + CUTLASS persistent structure.
"""
from __future__ import annotations
import cutlass, cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cuda.bindings.driver as cuda
import torch
from cutlass import BFloat16, Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum

_TILE_M = 128


class PersistentGemm:
    def __init__(self, N, K):
        self.N, self.K = N, K
        self.tile_m, self.tile_n, self.tile_k = _TILE_M, N, K
        self.cta_tile = (self.tile_m, self.tile_n, self.tile_k)
        self.acc_dtype = Float32; self.ab_dtype = BFloat16
        self.cluster_shape_mn = (1, 1); self.cluster_shape_mnk = (1, 1, 1)
        self.num_ab_stage = 3; self.num_acc_stage = 2
        self.epi_warps = (0, 1, 2, 3); self.mma_warp = 4; self.tma_warp = 5
        self.threads_per_cta = 32 * 6
        self.shared_storage = None; self.num_tmem_cols = None

    @cute.kernel
    def kernel(self, tma_atom_a, tA, tma_atom_b, tB, tma_atom_c, tC,
               sA_layout, sB_layout, sC_layout, tiled_mma, epi_tile,
               a_cta_layout, b_cta_layout, cluster_layout_vmnk, tCtAcc_fake, sched_params):
        tx_bytes = (self.tile_m * self.tile_k + self.tile_n * self.tile_k) * 2
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        smem = cutlass.utils.SmemAllocator(); storage = smem.allocate(self.shared_storage)
        if warp_idx == self.tma_warp:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_a); cpasync.prefetch_descriptor(tma_atom_b)

        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar.data_ptr(), num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=tx_bytes, cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True).make_participants()
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(), num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * len(self.epi_warps)),
            cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True)

        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (1 + len(self.epi_warps)))
        tmem = utils.TmemAllocator(storage.tmem_holding.data_ptr(), barrier_for_retrieve=tmem_bar,
                                   allocator_warp_id=self.epi_warps[0])
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
        sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)

        # global tiling: A (M,K) -> per-m-tile; B (N,K) single tile
        gA_mk = cute.local_tile(tA, (self.tile_m, self.tile_k), (None, 0))   # (tile_m, tile_k, RestM)
        gB = cute.local_tile(tB, (self.tile_n, self.tile_k), (0, 0))
        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA_mk)     # (MMA,MMA_M,MMA_K,RestM)
        tCgB = thr_mma.partition_B(gB)
        tAsA_p, tAgA_p = cpasync.tma_partition(tma_atom_a, 0, a_cta_layout,
            cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))   # sA:((atom),STAGE); gA:((atom),RestM)
        tBsB_p, tBgB_p = cpasync.tma_partition(tma_atom_b, 0, b_cta_layout,
            cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3))
        rA = tiled_mma.make_fragment_A(sA); rB = tiled_mma.make_fragment_B(sB)

        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        tile_sched = utils.StaticPersistentTileScheduler.create(sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
        work = tile_sched.initial_work_tile_info()

        # ---- TMA warp ----
        if warp_idx == self.tma_warp:
            while work.is_valid_tile:
                mtile = work.tile_idx[0]
                ab_producer.reset()
                peek = ab_producer.try_acquire()
                h = ab_producer.acquire_and_advance(peek)
                cute.copy(tma_atom_a, tAgA_p[(None, mtile)], tAsA_p[(None, h.index)], tma_bar_ptr=h.barrier)
                cute.copy(tma_atom_b, tBgB_p, tBsB_p[(None, h.index)], tma_bar_ptr=h.barrier)
                tile_sched.advance_to_next_work(); work = tile_sched.get_current_work()
            ab_producer.tail()

        # ---- MMA warp ----
        if warp_idx == self.mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            acc_ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)
            nkb = cute.size(rA, mode=[2])
            while work.is_valid_tile:
                tCtAcc = tCtAcc_base[(None, None, None, acc_ps.index)]
                ab_consumer.reset()
                peek = ab_consumer.try_wait()
                acc_pipeline.producer_acquire(acc_ps)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                h = ab_consumer.wait_and_advance(peek)
                for kb in cutlass.range_constexpr(nkb):
                    cute.gemm(tiled_mma, tCtAcc, rA[(None,None,kb,h.index)], rB[(None,None,kb,h.index)], tCtAcc)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                h.release()
                acc_pipeline.producer_commit(acc_ps)
                acc_ps.advance()
                tile_sched.advance_to_next_work(); work = tile_sched.get_current_work()
            acc_pipeline.producer_tail(acc_ps)

        # ---- Epilogue warps 0-3 ----
        if warp_idx < self.mma_warp:
            tmem.allocate(self.num_tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            acc_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)
            copy_atom_t2r = sm100_utils.get_tmem_load_op(self.cta_tile, LayoutEnum.ROW_MAJOR, BFloat16, self.acc_dtype, epi_tile, False)
            simt = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16)
            gC_mn = cute.local_tile(tC, (self.tile_m, self.tile_n), (None, 0))   # (tile_m,tile_n,RestM)
            while work.is_valid_tile:
                mtile = work.tile_idx[0]
                acc_pipeline.consumer_wait(acc_cs)
                tAcc = cute.flat_divide(tCtAcc_base[((None,None),0,0,acc_cs.index)], epi_tile)
                tcopy = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc[(None,None,0,0)])
                thr_t2r = tcopy.get_slice(tidx)
                tTR_t = thr_t2r.partition_S(tAcc)
                cC = cute.make_identity_tensor((self.tile_m, self.tile_n))
                tTR_cC = thr_t2r.partition_D(cute.flat_divide(cC, epi_tile))
                rAcc = cute.make_rmem_tensor(tTR_cC[None,None,None,0,0].shape, self.acc_dtype)
                gC = gC_mn[(None, None, mtile)]
                gC_epi = cute.flat_divide(thr_mma.partition_C(gC)[((None,None),0,0)], epi_tile)
                tTR_g = thr_t2r.partition_D(gC_epi)
                nem = cute.size(tAcc, mode=[2]); nen = cute.size(tAcc, mode=[3])
                for ei in cutlass.range_constexpr(nem):
                    for ej in cutlass.range_constexpr(nen):
                        cute.copy(tcopy, tTR_t[None,None,None,ei,ej], rAcc)
                        rD = cute.make_fragment_like(rAcc, BFloat16)
                        rD.store(rAcc.load().to(BFloat16))
                        cute.copy(simt, rD, tTR_g[None,None,None,ei,ej])
                cute.arch.fence_view_async_tmem_load()
                acc_pipeline.consumer_release(acc_cs)
                acc_cs.advance()
                tile_sched.advance_to_next_work(); work = tile_sched.get_current_work()
            cute.arch.barrier(barrier_id=2, number_of_threads=32*len(self.epi_warps))
            tmem.free(tmem_ptr, self.num_tmem_cols)

    @cute.jit
    def __call__(self, mA, mB, mC, max_active_clusters, stream):
        M = mA.shape[0]
        tiled_mma = sm100_utils.make_trivial_tiled_mma(self.ab_dtype, tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K, self.acc_dtype, tcgen05.CtaGroup.ONE, (self.tile_m, self.tile_n))
        cluster_layout_vmnk = cute.tiled_divide(cute.make_layout(self.cluster_shape_mnk), (tiled_mma.thr_id.shape,))
        mma_tiler = (self.tile_m, self.tile_n, self.tile_k)
        a_smem = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler, self.ab_dtype, self.num_ab_stage)
        b_smem = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler, self.ab_dtype, self.num_ab_stage)
        epi_tile = sm100_utils.compute_epilogue_tile_shape(self.cta_tile, False, LayoutEnum.ROW_MAJOR, BFloat16)
        sC_layout = sm100_utils.make_smem_layout_epi(BFloat16, LayoutEnum.ROW_MAJOR, epi_tile, 1)
        a1 = cute.slice_(a_smem,(None,None,None,0)); b1 = cute.slice_(b_smem,(None,None,None,0))
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        tma_a, ta = cute.nvgpu.make_tiled_tma_atom_A(a_op, mA, a1, mma_tiler, tiled_mma, cluster_layout_vmnk.shape)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        tma_b, tb = cute.nvgpu.make_tiled_tma_atom_B(b_op, mB, b1, mma_tiler, tiled_mma, cluster_layout_vmnk.shape)
        tma_c, tc = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), mC, cute.slice_(sC_layout,(None,None,0)), epi_tile)
        a_cta = cute.make_layout(cute.slice_(cluster_layout_vmnk,(0,0,None,0)).shape)
        b_cta = cute.make_layout(cute.slice_(cluster_layout_vmnk,(0,None,0,0)).shape)
        acc_shape = tiled_mma.partition_shape_C((self.tile_m,self.tile_n))
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))
        self.num_tmem_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        num_m = M // self.tile_m
        sched_params = utils.PersistentTileSchedulerParams((num_m, 1, 1), (1, 1, 1))
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(sched_params, max_active_clusters)
        sA_s = cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(a_smem)],1024]
        sB_s = cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(b_smem)],1024]
        @cute.struct
        class SS:
            ab_mbar: cute.struct.MemRange[cutlass.Int64, 2*self.num_ab_stage]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, 2*self.num_acc_stage]
            tmem_holding: cute.struct.MemRange[Int32,1]
            sA: sA_s
            sB: sB_s
        self.shared_storage = SS
        self.kernel(tma_a, ta, tma_b, tb, tma_c, mC, a_smem, b_smem, sC_layout, tiled_mma, epi_tile,
                    a_cta, b_cta, cluster_layout_vmnk, tCtAcc_fake, sched_params
        ).launch(grid=grid, block=[self.threads_per_cta,1,1], cluster=(1,1,1), stream=stream)


_CACHE={}
def persistent_gemm(A, B):
    M,K=A.shape; N,_=B.shape
    C=torch.empty(M,N,device=A.device,dtype=torch.bfloat16)
    d=lambda t: from_dlpack(t.detach(),assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mac=utils.HardwareInfo().get_max_active_clusters(1)
    stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key=(M,N,K)
    if key not in _CACHE:
        _CACHE[key]=cute.compile(PersistentGemm(N,K), d(A),d(B),d(C),mac,stream)
    _CACHE[key](d(A),d(B),d(C),mac,stream)
    return C

if __name__=="__main__":
    import sys
    torch.manual_seed(0)
    M=int(sys.argv[1]) if len(sys.argv)>1 else 1048576
    N=128;K=128
    A=torch.randn(M,K,device="cuda",dtype=torch.bfloat16)*0.3
    B=torch.randn(N,K,device="cuda",dtype=torch.bfloat16)*0.3
    print(f"PRE M={M}",flush=True)
    ref=A.float()@B.float().T
    C=persistent_gemm(A,B).float()
    cos=torch.nn.functional.cosine_similarity(C.flatten(),ref.flatten(),dim=0).item()
    print(f"persistent GEMM M={M}: cos={cos:.6f} maxabs={(C-ref).abs().max().item():.3e}",flush=True)
