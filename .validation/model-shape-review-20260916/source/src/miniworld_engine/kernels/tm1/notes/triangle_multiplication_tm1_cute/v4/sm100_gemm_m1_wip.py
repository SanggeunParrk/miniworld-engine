"""M1: minimal SM100 (Blackwell / tcgen05) dense GEMM, from scratch.

C[m,n] = sum_k A[m,k] * B[n,k]      (B is (N,K), nn.Linear-style; n-major C out)

Single CTA tile in M (TILE_M=128), full K in SMEM (no K-staging pipeline),
one MMA sweep into TMEM, TMEM->reg epilogue, SMEM store, TMA store.
Single-CTA cluster (1,1,1) -> uses the SM100 TMA atom builders
(cluster_shape_to_tma_atom_A/B + make_tiled_tma_atom_A/B), NOT the plain
cpasync 2D-box path (which does not match the tcgen05 MMA smem layout).

Goal: prove we can drive tcgen05 MMA + TMEM on B200. No gate, no M-major yet.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import BFloat16, Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum, TmemAllocator

import cutlass.pipeline as pipeline

_TILE_M = 128
_THREADS = 128  # one warpgroup for the epilogue; warp 0 issues TMA + MMA


class SimpleGemmSm100:
    def __init__(self, N: int, K: int):
        self.N = N
        self.K = K
        self.tile_m = _TILE_M
        self.tile_n = N
        self.tile_k = K
        self.acc_dtype = Float32
        self.ab_dtype = BFloat16
        self.cta_tile = (self.tile_m, self.tile_n, self.tile_k)
        self.cluster_shape_mnk = (1, 1, 1)
        self.shared_storage = None
        self.num_tmem_cols = None

    @cute.kernel
    def kernel(
        self,
        tma_atom_A: cute.CopyAtom,
        tA: cute.Tensor,
        tma_atom_B: cute.CopyAtom,
        tB: cute.Tensor,
        tma_atom_C: cute.CopyAtom,
        tC: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        epi_tile: cute.Tile,
        a_cta_layout: cute.Layout,
        b_cta_layout: cute.Layout,
        tx_bytes: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
        sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)
        sC = storage.sC.get_tensor(sC_layout.outer, swizzle=sC_layout.inner)

        mbar_ab = storage.mbar_ab.data_ptr()
        mbar_acc = storage.mbar_acc.data_ptr()

        if warp_idx == 0:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_A)
                cpasync.prefetch_descriptor(tma_atom_B)
                cpasync.prefetch_descriptor(tma_atom_C)
                cute.arch.mbarrier_init(mbar_ab, 1)
                cute.arch.mbarrier_init(mbar_acc, 1)
        cute.arch.mbarrier_init_fence()

        # ---- TMEM allocator ----
        tmem_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=_THREADS)
        tmem = TmemAllocator(storage.tmem_holding.data_ptr(), barrier_for_retrieve=tmem_barrier)
        tmem.allocate(self.num_tmem_cols)
        cute.arch.barrier()

        # ---- global tiles (single k-tile: tile_k == K) ----
        gA = cute.local_tile(tA, (self.tile_m, self.tile_k), (m_block, 0))
        gB = cute.local_tile(tB, (self.tile_n, self.tile_k), (0, 0))
        gC = cute.local_tile(tC, (self.tile_m, self.tile_n), (m_block, 0))

        thr_mma = tiled_mma.get_slice(0)
        # MMA-partition gmem for the TMA copy: (MMA, MMA_M, MMA_K)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)

        sA0 = cute.slice_(sA, (None, None, None, 0))
        sB0 = cute.slice_(sB, (None, None, None, 0))

        # ---- TMA loads A, B -> smem (single stage) ----
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_ab, tx_bytes)
                tAsA, tAgA = cpasync.tma_partition(
                    tma_atom_A, 0, a_cta_layout,
                    cute.group_modes(sA0, 0, cute.rank(sA0)),
                    cute.group_modes(tCgA, 0, cute.rank(tCgA)),
                )
                cute.copy(tma_atom_A, tAgA, tAsA, tma_bar_ptr=mbar_ab)
                tBsB, tBgB = cpasync.tma_partition(
                    tma_atom_B, 0, b_cta_layout,
                    cute.group_modes(sB0, 0, cute.rank(sB0)),
                    cute.group_modes(tCgB, 0, cute.rank(tCgB)),
                )
                cute.copy(tma_atom_B, tBgB, tBsB, tma_bar_ptr=mbar_ab)
        cute.arch.mbarrier_wait(mbar_ab, Int32(0))
        cute.arch.fence_view_async_shared()

        # ---- TMEM accumulator tensor ----
        tmem.wait_for_alloc()
        acc_ptr = tmem.retrieve_ptr(self.acc_dtype)
        acc_shape = tiled_mma.partition_shape_C((self.tile_m, self.tile_n))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        acc = cute.make_tensor(acc_ptr, tCtAcc_fake.layout)

        # ---- MMA (single leader thread issues all k-blocks) ----
        tCrA = tiled_mma.make_fragment_A(sA0)
        tCrB = tiled_mma.make_fragment_B(sB0)
        num_k_blocks = cute.size(tCrA, mode=[2])
        # tcgen05 MMA is issued at warp level (matches quack): NOT inside elect_one,
        # so mutating the ACCUMULATE flag stays in this region and dominates the yield.
        if warp_idx == 0:
            # tcgen05 UMMA must be issued by a SINGLE thread: gemm goes inside
            # elect_one. The ACCUMULATE set stays in this (if) region -- not the
            # nested elect region -- so its value dominates the region yield.
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k in cutlass.range_constexpr(num_k_blocks):
                with cute.arch.elect_one():
                    cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tcgen05.commit(mbar_acc)
        cute.arch.mbarrier_wait(mbar_acc, Int32(0))

        # ---- Epilogue: TMEM -> reg -> smem -> gmem, per epi-subtile ----
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile, LayoutEnum.ROW_MAJOR, BFloat16, self.acc_dtype, epi_tile, False,
        )
        acc_mn = acc[((None, None), 0, 0)]  # MMA-atom C -> plain (tile_m, tile_n)
        tAcc_epi = cute.flat_divide(acc_mn, epi_tile)  # (EPI_M, EPI_N, num_m, num_n)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0)])
        thr_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_t2r.partition_S(tAcc_epi)

        cC = cute.make_identity_tensor((self.tile_m, self.tile_n))
        cC_epi = cute.flat_divide(cC, epi_tile)
        tTR_cC = thr_t2r.partition_D(cC_epi)
        tTR_rAcc = cute.make_rmem_tensor(tTR_cC[None, None, None, 0, 0].shape, self.acc_dtype)

        copy_atom_r2s = sm100_utils.get_smem_store_op(
            LayoutEnum.ROW_MAJOR, BFloat16, self.acc_dtype, tiled_copy_t2r,
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_r2s = tiled_copy_r2s.get_slice(tidx)
        sC0 = cute.slice_(sC, (None, None, 0))          # one epi subtile of smem
        tRS_sC = thr_r2s.partition_D(sC0)

        gC = cute.local_tile(tC, (self.tile_m, self.tile_n), (m_block, 0))
        gC_epi = cute.flat_divide(gC, epi_tile)          # (EPI_M, EPI_N, num_m, num_n)

        num_epi_m = cute.size(tAcc_epi, mode=[2])
        num_epi_n = cute.size(tAcc_epi, mode=[3])
        for ei in cutlass.range_constexpr(num_epi_m):
            for ej in cutlass.range_constexpr(num_epi_n):
                cute.copy(tiled_copy_t2r, tTR_tAcc[None, None, None, ei, ej], tTR_rAcc)
                rD = cute.make_fragment_like(tTR_rAcc, BFloat16)
                rD.store(tTR_rAcc.load().to(BFloat16))
                cute.copy(tiled_copy_r2s, tiled_copy_r2s.retile(rD), tRS_sC)
                # make the r2s smem writes visible to the async (TMA) proxy
                cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space="cta")
                cute.arch.barrier()
                if warp_idx == 0:
                    with cute.arch.elect_one():
                        sC_t, gC_t = cpasync.tma_partition(
                            tma_atom_C, 0, cute.make_layout(1),
                            cute.group_modes(sC0, 0, cute.rank(sC0)),
                            cute.group_modes(gC_epi[None, None, ei, ej], 0, 2),
                        )
                        cute.copy(tma_atom_C, sC_t, gC_t)
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.barrier()   # store done before sC is reused next subtile

        cute.arch.barrier()
        tmem.relinquish_alloc_permit()
        tmem.free(acc_ptr, self.num_tmem_cols)

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
        M = mA.shape[0]
        m_blocks = M // self.tile_m

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.ab_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            (self.tile_m, self.tile_n),
        )

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma.thr_id.shape,)
        )

        mma_tiler = (self.tile_m, self.tile_n, self.tile_k)
        a_smem_layout_staged = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler, self.ab_dtype, 1)
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler, self.ab_dtype, 1)

        epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile, False, LayoutEnum.ROW_MAJOR, BFloat16,
        )
        sC_layout = sm100_utils.make_smem_layout_epi(BFloat16, LayoutEnum.ROW_MAJOR, epi_tile, 1)

        # 4-mode staged smem layouts -> slice the stage for the TMA atom
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mnk, tiled_mma.thr_id)
        tma_atom_A, tma_A = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, mA, a_smem_layout, mma_tiler, tiled_mma, cluster_layout_vmnk.shape,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mnk, tiled_mma.thr_id)
        tma_atom_B, tma_B = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, mB, b_smem_layout, mma_tiler, tiled_mma, cluster_layout_vmnk.shape,
        )

        tma_atom_C, tma_C = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mC, cute.slice_(sC_layout, (None, None, 0)),
            epi_tile,
        )

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)

        tx_bytes = (self.tile_m * self.tile_k + self.tile_n * self.tile_k) * 2

        acc_shape = tiled_mma.partition_shape_C((self.tile_m, self.tile_n))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        self.num_tmem_cols = sm100_utils.get_num_tmem_alloc_cols(tCtAcc_fake)

        sA_struct = cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(a_smem_layout_staged)], 1024]
        sB_struct = cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(b_smem_layout_staged)], 1024]
        sC_struct = cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(sC_layout)], 1024]

        @cute.struct
        class SharedStorage:
            mbar_ab: cute.struct.MemRange[cutlass.Int64, 1]
            mbar_acc: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding: cute.struct.MemRange[Int32, 1]
            sC: sC_struct
            sA: sA_struct
            sB: sB_struct

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_A, tma_A, tma_atom_B, tma_B, tma_atom_C, tma_C,
            a_smem_layout_staged, b_smem_layout_staged, sC_layout, tiled_mma, epi_tile,
            a_cta_layout, b_cta_layout, Int32(tx_bytes),
        ).launch(grid=[m_blocks, 1, 1], block=[_THREADS, 1, 1])


_CACHE: dict = {}


def simple_gemm_sm100(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """C = A @ B.T ; A:(M,K) B:(N,K) bf16 -> C:(M,N) bf16."""
    assert A.dtype == torch.bfloat16 and B.dtype == torch.bfloat16
    M, K = A.shape
    N, K2 = B.shape
    assert K == K2
    C = torch.empty(M, N, device=A.device, dtype=torch.bfloat16)
    mA = from_dlpack(A, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mB = from_dlpack(B, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mC = from_dlpack(C, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    key = (M, N, K)
    if key not in _CACHE:
        _CACHE[key] = cute.compile(SimpleGemmSm100(N, K), mA, mB, mC)
    _CACHE[key](mA, mB, mC)
    return C


if __name__ == "__main__":
    import sys
    torch.manual_seed(0)
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    N = K = 128
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.3
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.3
    ref = (A.float() @ B.float().T)
    out = simple_gemm_sm100(A, B).float()
    err = (out - ref).abs()
    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    print(f"M={M} N={N} K={K}: maxabs={err.max().item():.3e} meanabs={err.mean().item():.3e} "
          f"cos={cos:.6f} ref|max|={ref.abs().max().item():.3e} out|max|={out.abs().max().item():.3e}")
