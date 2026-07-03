"""Minimal tcgen05 SM100 GEMM, faithful port of CUTLASS example
cute/blackwell/kernel/dense_gemm/dense_gemm.py for the single-CTA single-tile case.
C[m,n] = sum_k A[m,k]*B[n,k]  (A:(M,K), B:(N,K)).
"""
from __future__ import annotations
import cutlass
import cutlass.cute as cute
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


class MicroGemm:
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
    def kernel(self, tma_atom_a, tAgA, tma_atom_b, tBgB, tma_atom_c, tC,
               sA_layout, sB_layout, sC_layout, tiled_mma, epi_tile,
               a_cta_layout, b_cta_layout, cluster_layout_vmnk, tCtAcc_fake):
        tx_bytes = (self.tile_m * self.tile_k + self.tile_n * self.tile_k) * 2
        tidx, _, _ = cute.arch.thread_idx()
        m_block, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        if warp_idx == 0:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_b)
                cpasync.prefetch_descriptor(tma_atom_c)

        # ---- pipelines (handle API for AB; state API for acc) ----
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=tx_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_cta),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)
        acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)

        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=0, num_threads=self.threads_per_cta)
        tmem = utils.TmemAllocator(storage.tmem_holding.data_ptr(), barrier_for_retrieve=tmem_alloc_barrier)

        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
        sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)
        sC = storage.sC.get_tensor(sC_layout.outer, swizzle=sC_layout.inner)

        # TMA partition (single stage, single k-tile)
        gA = cute.local_tile(tAgA, (self.tile_m, self.tile_k), (m_block, 0))
        gB = cute.local_tile(tBgB, (self.tile_n, self.tile_k), (0, 0))
        gC = cute.local_tile(tC, (self.tile_m, self.tile_n), (m_block, 0))
        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tAsA_p, tAgA_p = cpasync.tma_partition(
            tma_atom_a, 0, a_cta_layout,
            cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))
        tBsB_p, tBgB_p = cpasync.tma_partition(
            tma_atom_b, 0, b_cta_layout,
            cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3))

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        tmem.allocate(self.num_tmem_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        # ---- mainloop (warp 0) ----
        if warp_idx == 0:
            ph = ab_producer.acquire_and_advance()
            cute.copy(tma_atom_a, tAgA_p, tAsA_p[(None, ph.index)], tma_bar_ptr=ph.barrier)
            cute.copy(tma_atom_b, tBgB_p, tBsB_p[(None, ph.index)], tma_bar_ptr=ph.barrier)
            ch = ab_consumer.wait_and_advance()
            num_kblks = cute.size(tCrA, mode=[2])
            for kblk in cutlass.range_constexpr(num_kblks):
                crd = (None, None, kblk, ch.index)
                cute.gemm(tiled_mma, tCtAcc, tCrA[crd], tCrB[crd], tCtAcc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            ch.release()
            acc_pipeline.producer_commit(acc_producer_state)

        tmem.relinquish_alloc_permit()
        acc_pipeline.consumer_wait(acc_consumer_state)

        # ---- epilogue (all threads): TMEM -> reg -> smem -> gmem per epi-subtile ----
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile, LayoutEnum.ROW_MAJOR, BFloat16, self.acc_dtype, epi_tile, False)
        acc_mn = tCtAcc[((None, None), 0, 0)]
        tAcc_epi = cute.flat_divide(acc_mn, epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0)])
        thr_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_t2r.partition_S(tAcc_epi)
        cC = cute.make_identity_tensor((self.tile_m, self.tile_n))
        cC_epi = cute.flat_divide(cC, epi_tile)
        tTR_cC = thr_t2r.partition_D(cC_epi)
        tTR_rAcc = cute.make_rmem_tensor(tTR_cC[None, None, None, 0, 0].shape, self.acc_dtype)
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            LayoutEnum.ROW_MAJOR, BFloat16, self.acc_dtype, tiled_copy_t2r)
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_r2s = tiled_copy_r2s.get_slice(tidx)
        sC0 = cute.slice_(sC, (None, None, 0))
        tRS_sC = thr_r2s.partition_D(sC0)
        gC_epi = cute.flat_divide(gC, epi_tile)
        num_epi_m = cute.size(tAcc_epi, mode=[2])
        num_epi_n = cute.size(tAcc_epi, mode=[3])
        for ei in cutlass.range_constexpr(num_epi_m):
            for ej in cutlass.range_constexpr(num_epi_n):
                cute.copy(tiled_copy_t2r, tTR_tAcc[None, None, None, ei, ej], tTR_rAcc)
                rD = cute.make_fragment_like(tTR_rAcc, BFloat16)
                rD.store(tTR_rAcc.load().to(BFloat16))
                cute.copy(tiled_copy_r2s, tiled_copy_r2s.retile(rD), tRS_sC)
                cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space="cta")
                cute.arch.barrier()
                if warp_idx == 0:
                    with cute.arch.elect_one():
                        sC_t, gC_t = cpasync.tma_partition(
                            tma_atom_c, 0, cute.make_layout(1),
                            cute.group_modes(sC0, 0, cute.rank(sC0)),
                            cute.group_modes(gC_epi[None, None, ei, ej], 0, 2))
                        cute.copy(tma_atom_c, sC_t, gC_t)
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.barrier()

        pipeline.sync(barrier_id=1)
        tmem.free(tmem_ptr)

    @cute.jit
    def __call__(self, mA, mB, mC):
        M = mA.shape[0]
        m_blocks = M // self.tile_m
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.ab_dtype, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
            self.acc_dtype, tcgen05.CtaGroup.ONE, (self.tile_m, self.tile_n))
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma.thr_id.shape,))
        mma_tiler = (self.tile_m, self.tile_n, self.tile_k)
        a_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler, self.ab_dtype, self.num_ab_stage)
        b_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler, self.ab_dtype, self.num_ab_stage)
        epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile, False, LayoutEnum.ROW_MAJOR, BFloat16)
        sC_layout = sm100_utils.make_smem_layout_epi(BFloat16, LayoutEnum.ROW_MAJOR, epi_tile, 1)
        a_smem_layout_1 = cute.slice_(a_smem_layout, (None, None, None, 0))
        b_smem_layout_1 = cute.slice_(b_smem_layout, (None, None, None, 0))
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mnk, tiled_mma.thr_id)
        tma_atom_a, tma_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, mA, a_smem_layout_1, mma_tiler, tiled_mma, cluster_layout_vmnk.shape)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mnk, tiled_mma.thr_id)
        tma_atom_b, tma_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, mB, b_smem_layout_1, mma_tiler, tiled_mma, cluster_layout_vmnk.shape)
        tma_atom_c, tma_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mC, cute.slice_(sC_layout, (None, None, 0)), epi_tile)
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        acc_shape = tiled_mma.partition_shape_C((self.tile_m, self.tile_n))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        self.num_tmem_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)

        sA_struct = cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(a_smem_layout)], 1024]
        sB_struct = cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(b_smem_layout)], 1024]
        sC_struct = cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(sC_layout)], 1024]

        @cute.struct
        class SharedStorage:
            ab_mbar: cute.struct.MemRange[cutlass.Int64, 2 * self.num_ab_stage]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, 2 * self.num_acc_stage]
            tmem_holding: cute.struct.MemRange[Int32, 1]
            sC: sC_struct
            sA: sA_struct
            sB: sB_struct
        self.shared_storage = SharedStorage
        self.kernel(tma_atom_a, tma_a, tma_atom_b, tma_b, tma_atom_c, tma_c,
                    a_smem_layout, b_smem_layout, sC_layout, tiled_mma, epi_tile,
                    a_cta_layout, b_cta_layout, cluster_layout_vmnk, tCtAcc_fake
        ).launch(grid=[m_blocks, 1, 1], block=[self.threads_per_cta, 1, 1])


_CACHE = {}
def micro_gemm(A, B):
    M, K = A.shape; N, K2 = B.shape; assert K == K2
    C = torch.empty(M, N, device=A.device, dtype=torch.bfloat16)
    mA = from_dlpack(A, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mB = from_dlpack(B, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mC = from_dlpack(C, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    key = (M, N, K)
    if key not in _CACHE:
        _CACHE[key] = cute.compile(MicroGemm(N, K), mA, mB, mC)
    _CACHE[key](mA, mB, mC)
    return C

if __name__ == "__main__":
    import sys
    torch.manual_seed(0)
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    N = 128
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.3
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.3
    print(f"PRE M={M} N={N} K={K}", flush=True)
    ref = (A.float() @ B.float().T)
    out = micro_gemm(A, B).float()
    err = (out - ref).abs()
    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    print(f"M={M} N={N} K={K}: maxabs={err.max().item():.3e} meanabs={err.mean().item():.3e} cos={cos:.6f}", flush=True)
