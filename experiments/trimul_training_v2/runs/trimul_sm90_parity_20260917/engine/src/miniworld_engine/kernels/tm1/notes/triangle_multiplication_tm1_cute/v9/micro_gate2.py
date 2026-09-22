"""v9: K-looped gated GEMM. tile_k<K -> smaller smem -> more CTA/SM (occupancy).
out = sigmoid(A@Bg.T)*(A@Bp.T). A:(M,K), Bp/Bg:(N,K) -> [N,M] M-major.
"""
from __future__ import annotations
import cutlass, cutlass.cute as cute, cutlass.cute.math as cmath
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
import os
_TILE_K = int(os.environ.get('GK_TILEK','32'))
_STAGES = int(os.environ.get('GK_STAGES','4'))


class GateGemmK:
    def __init__(self, N, K):
        self.N, self.K = N, K
        self.tile_m, self.tile_n, self.tile_k = _TILE_M, N, _TILE_K
        self.acc_dtype = Float32; self.ab_dtype = BFloat16
        self.cta_tile = (self.tile_m, self.tile_n, self.tile_k)
        self.cluster_shape_mnk = (1, 1, 1)
        self.num_ab_stage = _STAGES; self.num_acc_stage = 1
        self.threads_per_cta = _THREADS
        self.shared_storage = None; self.num_tmem_cols = None

    @cute.kernel
    def kernel(self, tma_atom_a, tA, tma_atom_bp, tBp, tma_atom_bg, tBg, tCl,
               sA_layout, sB_layout, tiled_mma, epi_tile,
               a_cta_layout, b_cta_layout, cluster_layout_vmnk, tCtAcc_fake):
        tx_bytes = (self.tile_m * self.tile_k + 2 * self.tile_n * self.tile_k) * 2
        tidx, _, _ = cute.arch.thread_idx()
        m_block, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        smem = cutlass.utils.SmemAllocator(); storage = smem.allocate(self.shared_storage)
        if warp_idx == 0:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_bp)
                cpasync.prefetch_descriptor(tma_atom_bg)
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
            cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True)
        acc_ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)
        acc_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)
        tmem_bar = pipeline.NamedBarrier(barrier_id=0, num_threads=self.threads_per_cta)
        tmem = utils.TmemAllocator(storage.tmem_holding.data_ptr(), barrier_for_retrieve=tmem_bar)
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
        sBp = storage.sBp.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)
        sBg = storage.sBg.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)

        # global tiles over K: (tile, tile_k, num_k)
        gA = cute.local_tile(tA, (self.tile_m, self.tile_k), (m_block, None))
        gBp = cute.local_tile(tBp, (self.tile_n, self.tile_k), (0, None))
        gBg = cute.local_tile(tBg, (self.tile_n, self.tile_k), (0, None))
        gCl = cute.local_tile(tCl, (self.tile_m, self.tile_n), (m_block, 0))
        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA)   # (MMA,MMA_M,MMA_K,num_k)
        tCgBp = thr_mma.partition_B(gBp)
        tCgBg = thr_mma.partition_B(gBg)
        tAsA_p, tAgA_p = cpasync.tma_partition(tma_atom_a, 0, a_cta_layout,
            cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))     # sA:((atom),STAGE) gA:((atom),num_k)
        tBsBp_p, tBgBp_p = cpasync.tma_partition(tma_atom_bp, 0, b_cta_layout,
            cute.group_modes(sBp, 0, 3), cute.group_modes(tCgBp, 0, 3))
        tBsBg_p, tBgBg_p = cpasync.tma_partition(tma_atom_bg, 0, b_cta_layout,
            cute.group_modes(sBg, 0, 3), cute.group_modes(tCgBg, 0, 3))
        rA = tiled_mma.make_fragment_A(sA)   # (MMA,MMA_M,MMA_K,STAGE)
        rBp = tiled_mma.make_fragment_B(sBp)
        rBg = tiled_mma.make_fragment_B(sBg)

        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)
        tmem.allocate(self.num_tmem_cols); tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        proj = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        off = const_expr(tcgen05.find_tmem_tensor_col_offset(proj))
        gate = cute.make_tensor(tmem_ptr + off, tCtAcc_fake.layout)

        k_tiles = const_expr(self.K // self.tile_k)
        nkb = const_expr(cute.size(rA, mode=[2]))
        prefetch = const_expr(min(self.num_ab_stage - 1, k_tiles))
        if warp_idx == 0:
            for kt in cutlass.range_constexpr(prefetch):
                ph = ab_producer.acquire_and_advance()
                cute.copy(tma_atom_a, tAgA_p[(None, kt)], tAsA_p[(None, ph.index)], tma_bar_ptr=ph.barrier)
                cute.copy(tma_atom_bp, tBgBp_p[(None, kt)], tBsBp_p[(None, ph.index)], tma_bar_ptr=ph.barrier)
                cute.copy(tma_atom_bg, tBgBg_p[(None, kt)], tBsBg_p[(None, ph.index)], tma_bar_ptr=ph.barrier)
            for kt in cutlass.range_constexpr(k_tiles):
                if const_expr(kt + prefetch < k_tiles):
                    kk = kt + prefetch
                    ph = ab_producer.acquire_and_advance()
                    cute.copy(tma_atom_a, tAgA_p[(None, kk)], tAsA_p[(None, ph.index)], tma_bar_ptr=ph.barrier)
                    cute.copy(tma_atom_bp, tBgBp_p[(None, kk)], tBsBp_p[(None, ph.index)], tma_bar_ptr=ph.barrier)
                    cute.copy(tma_atom_bg, tBgBg_p[(None, kk)], tBsBg_p[(None, ph.index)], tma_bar_ptr=ph.barrier)
                ch = ab_consumer.wait_and_advance()
                for kb in cutlass.range_constexpr(nkb):
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, const_expr(kt != 0 or kb != 0))
                    cute.gemm(tiled_mma, proj, rA[(None, None, kb, ch.index)], rBp[(None, None, kb, ch.index)], proj)
                for kb in cutlass.range_constexpr(nkb):
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, const_expr(kt != 0 or kb != 0))
                    cute.gemm(tiled_mma, gate, rA[(None, None, kb, ch.index)], rBg[(None, None, kb, ch.index)], gate)
                ch.release()
            acc_pipeline.producer_commit(acc_ps)
        tmem.relinquish_alloc_permit()
        acc_pipeline.consumer_wait(acc_cs)

        copy_atom_t2r = sm100_utils.get_tmem_load_op(self.cta_tile, LayoutEnum.ROW_MAJOR, BFloat16, self.acc_dtype, epi_tile, False)
        tAcc0 = cute.flat_divide(proj[((None, None), 0, 0)], epi_tile)
        tcopy = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc0[(None, None, 0, 0)])
        thr_t2r = tcopy.get_slice(tidx)
        tPL = thr_t2r.partition_S(tAcc0)
        tGL = thr_t2r.partition_S(cute.flat_divide(gate[((None, None), 0, 0)], epi_tile))
        cC = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tTR_cC = thr_t2r.partition_D(cute.flat_divide(cC, epi_tile))
        rP = cute.make_rmem_tensor(tTR_cC[None, None, None, 0, 0].shape, self.acc_dtype)
        rG = cute.make_rmem_tensor(tTR_cC[None, None, None, 0, 0].shape, self.acc_dtype)
        def gp(gC):
            return thr_t2r.partition_D(cute.flat_divide(thr_mma.partition_C(gC)[((None, None), 0, 0)], epi_tile))
        gL = gp(gCl)
        simt = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16)
        nem = cute.size(tAcc0, mode=[2]); nen = cute.size(tAcc0, mode=[3])
        for ei in cutlass.range_constexpr(nem):
            for ej in cutlass.range_constexpr(nen):
                cute.copy(tcopy, tPL[None, None, None, ei, ej], rP)
                cute.copy(tcopy, tGL[None, None, None, ei, ej], rG)
                ov = rP.load() * (1.0 / (1.0 + cmath.exp(-rG.load())))
                rD = cute.make_fragment_like(rP, BFloat16); rD.store(ov.to(BFloat16))
                cute.copy(simt, rD, gL[None, None, None, ei, ej])
        cute.arch.fence_view_async_tmem_load()
        pipeline.sync(barrier_id=1)
        tmem.free(tmem_ptr)

    @cute.jit
    def __call__(self, mA, mBp, mBg, mCl):
        M = mA.shape[0]; m_blocks = M // self.tile_m
        tiled_mma = sm100_utils.make_trivial_tiled_mma(self.ab_dtype, tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K, self.acc_dtype, tcgen05.CtaGroup.ONE, (self.tile_m, self.tile_n))
        cluster_layout_vmnk = cute.tiled_divide(cute.make_layout(self.cluster_shape_mnk), (tiled_mma.thr_id.shape,))
        mma_tiler = (self.tile_m, self.tile_n, self.tile_k)
        a_smem = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler, self.ab_dtype, self.num_ab_stage)
        b_smem = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler, self.ab_dtype, self.num_ab_stage)
        epi_tile = sm100_utils.compute_epilogue_tile_shape((self.tile_m, self.tile_n, self.K), False, LayoutEnum.ROW_MAJOR, BFloat16)
        a1 = cute.slice_(a_smem, (None, None, None, 0)); b1 = cute.slice_(b_smem, (None, None, None, 0))
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mnk, tiled_mma.thr_id)
        tma_a, ta = cute.nvgpu.make_tiled_tma_atom_A(a_op, mA, a1, mma_tiler, tiled_mma, cluster_layout_vmnk.shape)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mnk, tiled_mma.thr_id)
        tma_bp, tbp = cute.nvgpu.make_tiled_tma_atom_B(b_op, mBp, b1, mma_tiler, tiled_mma, cluster_layout_vmnk.shape)
        tma_bg, tbg = cute.nvgpu.make_tiled_tma_atom_B(b_op, mBg, b1, mma_tiler, tiled_mma, cluster_layout_vmnk.shape)
        a_cta = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        b_cta = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        acc_shape = tiled_mma.partition_shape_C((self.tile_m, self.tile_n))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        self.num_tmem_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake) * 2
        sA_s = cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(a_smem)], 1024]
        sB_s = cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(b_smem)], 1024]
        @cute.struct
        class SS:
            ab_mbar: cute.struct.MemRange[cutlass.Int64, 2 * self.num_ab_stage]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, 2 * self.num_acc_stage]
            tmem_holding: cute.struct.MemRange[Int32, 1]
            sA: sA_s
            sBp: sB_s
            sBg: sB_s
        self.shared_storage = SS
        self.kernel(tma_a, ta, tma_bp, tbp, tma_bg, tbg, mCl, a_smem, b_smem, tiled_mma, epi_tile,
                    a_cta, b_cta, cluster_layout_vmnk, tCtAcc_fake).launch(grid=[m_blocks,1,1], block=[self.threads_per_cta,1,1])


_CACHE = {}
def gate_gemm_k(A, Bp, Bg):
    M, K = A.shape; N, _ = Bp.shape
    Cl = torch.empty(N, M, device=A.device, dtype=torch.bfloat16)
    d = lambda t: from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mCl = from_dlpack(Cl.t(), assumed_align=16).mark_layout_dynamic(leading_dim=0)
    key = (M, N, K)
    if key not in _CACHE:
        _CACHE[key] = cute.compile(GateGemmK(N, K), d(A), d(Bp), d(Bg), mCl)
    _CACHE[key](d(A), d(Bp), d(Bg), mCl)
    return Cl

if __name__ == "__main__":
    import sys
    torch.manual_seed(0)
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    K = 128; N = 128
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)*0.3
    Bp = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)*0.3
    Bg = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)*0.3
    print(f"PRE M={M} N={N} K={K}", flush=True)
    import triton
    Bp2 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)*0.3
    Bg2 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)*0.3
    lref = (torch.sigmoid(A.float()@Bg.float().T) * (A.float()@Bp.float().T)).t().contiguous()
    Cl = gate_gemm_k(A, Bp, Bg)
    cos = torch.nn.functional.cosine_similarity(Cl.float().flatten(), lref.flatten(), dim=0).item()
    print(f"tile_k={_TILE_K} stages={_STAGES} left cos={cos:.6f}", flush=True)
    def both():
        gate_gemm_k(A, Bp, Bg); gate_gemm_k(A, Bp2, Bg2)
    for _ in range(5): both()
    ms = triton.testing.do_bench(both, warmup=20, rep=50)
    print(f"tile_k={_TILE_K} stages={_STAGES} 2xGEMM(both sides)={ms:.3f} ms", flush=True)
