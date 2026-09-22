"""Hopper F567: independent dual GEMM + saved projection/gate + dropout/residual.

Derived from the engine's from-scratch TM2 TMA/WGMMA implementation. No
cuequiv dependency. The candidate policy is external to this kernel.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90h
import torch
from cuda.bindings import driver as cuda
from cutlass import BFloat16, Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import LayoutEnum
from miniworld_engine.kernels._compile import opaque
from quack import copy_utils as quack_copy


@dsl_user_op
def _reciprocal_full(x, *, loc=None, ip=None):
    """Match Triton's full-range approximate FP32 division, including subnormals."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "div.full.f32 $0, 0f3f800000, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )
    )


class ParityF567Sm90:
    """Two independent K reductions, tiled M/N, TMA loads and WGMMA math.

    The independent reductions reuse one num_stages operand ring. TMA prefetch
    overlaps WGMMA within each reduction; a wait and CTA barrier protect the
    phase handoff. No LN affine weight folding.
    """

    def __init__(
        self,
        N,
        KP,
        KG,
        L,
        BLOCK_M1,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
        num_warps,
        num_stages,
        wg_k_major,
        wp_k_major,
    ):
        tile_m, tile_n, tile_k, group_m = BLOCK_M1, BLOCK_N, BLOCK_K, GROUP_M
        self.num_stages = num_stages
        self.wg_k_major, self.wp_k_major = wg_k_major, wp_k_major
        self.N, self.KP, self.KG, self.L = N, KP, KG, L
        self.tile_m, self.tile_n, self.tile_k = tile_m, tile_n, tile_k
        self.group_m = group_m
        self.g_loop = (KG + tile_k - 1) // tile_k
        self.p_loop = (KP + tile_k - 1) // tile_k
        self.num_threads = num_warps * 32
        self.mma_mgroups = min(num_warps // 4, tile_m // 64)
        self.mma_ngroups = num_warps // 4 // self.mma_mgroups
        self.shared_storage = None

    @cute.kernel
    def kernel(
        self,
        tma_atom_X1: cute.CopyAtom,
        tX1: cute.Tensor,
        tma_atom_X2: cute.CopyAtom,
        tX2: cute.Tensor,
        tma_atom_W1: cute.CopyAtom,
        tW1: cute.Tensor,
        tma_atom_W2: cute.CopyAtom,
        tW2: cute.Tensor,
        atomO: cute.CopyAtom,
        atomP: cute.CopyAtom,
        atomG: cute.CopyAtom,
        atomR: cute.CopyAtom,
        atomD: cute.CopyAtom,
        tD_tma: cute.Tensor,
        sO_layout: cute.ComposedLayout,
        tO: cute.Tensor,
        tP: cute.Tensor,
        tG: cute.Tensor,
        tR: cute.Tensor,
        tD: cute.Tensor,
        sX1_layout: cute.ComposedLayout,
        sX2_layout: cute.ComposedLayout,
        sW1_layout: cute.ComposedLayout,
        sW2_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        tx_bytes_total: Int32,
    ):
        TILE_M: cutlass.Constexpr[int] = self.tile_m
        TILE_N: cutlass.Constexpr[int] = self.tile_n
        TILE_K: cutlass.Constexpr[int] = self.tile_k
        STAGES: cutlass.Constexpr[int] = self.num_stages
        G_LOOP: cutlass.Constexpr[int] = self.g_loop
        P_LOOP: cutlass.Constexpr[int] = self.p_loop

        tidx, _, _ = cute.arch.thread_idx()
        pid, _, _ = cute.arch.block_idx()
        nm = cute.ceil_div(tO.shape[0], TILE_M)
        nn = cute.ceil_div(self.N, TILE_N)
        first_m = (pid // (self.group_m * nn)) * self.group_m
        actual_group = min(self.group_m, nm - first_m)
        local = pid % (self.group_m * nn)
        m_block = first_m + local % actual_group
        n_block = local // actual_group
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sX1 = storage.sX1.get_tensor(sX1_layout.outer, swizzle=sX1_layout.inner)
        sX2 = storage.sX1.get_tensor(sX2_layout.outer, swizzle=sX2_layout.inner)
        sW1 = storage.sW1.get_tensor(sW1_layout.outer, swizzle=sW1_layout.inner)
        sW2 = storage.sW1.get_tensor(sW2_layout.outer, swizzle=sW2_layout.inner)
        sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)

        mbar_full_ptr = storage.mbar_full.data_ptr()
        if warp_idx == 0:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_X1)
                cpasync.prefetch_descriptor(tma_atom_X2)
                cpasync.prefetch_descriptor(tma_atom_W1)
                cpasync.prefetch_descriptor(tma_atom_W2)
                cpasync.prefetch_descriptor(atomR)
                cpasync.prefetch_descriptor(atomD)
                for b in cutlass.range_constexpr(2 * STAGES + 1):
                    cute.arch.mbarrier_init(mbar_full_ptr + b, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        gX1 = cute.local_tile(tX1, (TILE_M, TILE_K), (m_block, None))
        gX2 = cute.local_tile(tX2, (TILE_M, TILE_K), (m_block, None))
        gW1 = cute.local_tile(tW1, (TILE_N, TILE_K), (n_block, None))
        gW2 = cute.local_tile(tW2, (TILE_N, TILE_K), (n_block, None))

        load_X1, _, _ = quack_copy.tma_get_copy_fn(
            tma_atom_X1,
            0,
            cute.make_layout(1),
            gX1,
            sX1,
        )
        load_X2, _, prefetch_X2 = quack_copy.tma_get_copy_fn(
            tma_atom_X2,
            0,
            cute.make_layout(1),
            gX2,
            sX2,
        )
        load_W1, _, _ = quack_copy.tma_get_copy_fn(
            tma_atom_W1,
            0,
            cute.make_layout(1),
            gW1,
            sW1,
        )
        load_W2, _, _ = quack_copy.tma_get_copy_fn(
            tma_atom_W2,
            0,
            cute.make_layout(1),
            gW2,
            sW2,
        )

        gR = cute.local_tile(tR, (TILE_M, TILE_N), (m_block, n_block))
        load_R, _, _ = quack_copy.tma_get_copy_fn(
            atomR, 0, cute.make_layout(1), gR, sO, single_stage=True
        )
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_full_ptr + 2 * STAGES, TILE_M * TILE_N * 2
                )
            load_R(tma_bar_ptr=mbar_full_ptr + 2 * STAGES)
            for stage in cutlass.range_constexpr(STAGES):
                if cutlass.const_expr(stage < G_LOOP):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar_full_ptr + stage, tx_bytes_total
                        )
                    load_X1(
                        src_idx=stage, dst_idx=stage, tma_bar_ptr=mbar_full_ptr + stage
                    )
                    load_W1(
                        src_idx=stage, dst_idx=stage, tma_bar_ptr=mbar_full_ptr + stage
                    )
            # Warm the first projection operand tiles while the independent gate
            # GEMM runs. This is an L2 hint; actual TMA stage barriers stay intact.
            for future in cutlass.range_constexpr(min(STAGES, P_LOOP)):
                cute.prefetch(tma_atom_X2, prefetch_X2[None, future])

        thr_mma = tiled_mma.get_slice(tidx)
        acc_shape = thr_mma.partition_shape_C((TILE_M, TILE_N))
        acc_G = cute.make_fragment(acc_shape, Float32)
        acc_V = cute.make_fragment(acc_shape, Float32)
        tCsX1 = thr_mma.make_fragment_A(thr_mma.partition_A(sX1))
        tCsX2 = thr_mma.make_fragment_A(thr_mma.partition_A(sX2))
        # Each GEMM has its own MMA B-major mode, matching weight strides.
        mma_G = cute.make_mma_atom(tiled_mma.op)
        mma_P_tiled = sm90h.make_trivial_tiled_mma(
            BFloat16,
            BFloat16,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K
            if cutlass.const_expr(self.wp_k_major)
            else warpgroup.OperandMajorMode.MN,
            Float32,
            (self.mma_mgroups, self.mma_ngroups, 1),
            (64, TILE_N // self.mma_ngroups),
        )
        thr_p = mma_P_tiled.get_slice(tidx)
        tCsW1 = thr_mma.make_fragment_B(thr_mma.partition_B(sW1))
        tCsW2 = thr_p.make_fragment_B(thr_p.partition_B(sW2))
        mma_V = cute.make_mma_atom(mma_P_tiled.op)
        mma_G.set(warpgroup.Field.ACCUMULATE, False)
        mma_V.set(warpgroup.Field.ACCUMULATE, False)
        for k in cutlass.range_constexpr(G_LOOP):
            stage = k % STAGES
            cute.arch.mbarrier_wait(mbar_full_ptr + stage, Int32((k // STAGES) % 2))
            warpgroup.fence()
            for ki in cutlass.range_constexpr(cute.size(tCsX1.shape[2])):
                cute.gemm(
                    mma_G,
                    acc_G,
                    tCsX1[None, None, ki, stage],
                    tCsW1[None, None, ki, stage],
                    acc_G,
                )
                mma_G.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()
            if cutlass.const_expr(k + STAGES < G_LOOP):
                warpgroup.wait_group(0)
            if cutlass.const_expr(k + STAGES < G_LOOP):
                cute.arch.barrier()
                if warp_idx == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar_full_ptr + stage, tx_bytes_total
                        )
                    load_X1(
                        src_idx=k + STAGES,
                        dst_idx=stage,
                        tma_bar_ptr=mbar_full_ptr + stage,
                    )
                    load_W1(
                        src_idx=k + STAGES,
                        dst_idx=stage,
                        tma_bar_ptr=mbar_full_ptr + stage,
                    )
        warpgroup.wait_group(0)
        cute.arch.barrier()
        if warp_idx == 0:
            for stage in cutlass.range_constexpr(STAGES):
                if cutlass.const_expr(stage < P_LOOP):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar_full_ptr + STAGES + stage, tx_bytes_total
                        )
                    load_X2(
                        src_idx=stage,
                        dst_idx=stage,
                        tma_bar_ptr=mbar_full_ptr + STAGES + stage,
                    )
                    load_W2(
                        src_idx=stage,
                        dst_idx=stage,
                        tma_bar_ptr=mbar_full_ptr + STAGES + stage,
                    )

        # Reuse the gate accumulator for its FP32 sigmoid while projection TMA
        # transfers are in flight. Preserve the BF16 logit boundary; the epilogue
        # consumes this FP32 gate and saves its independently rounded BF16 copy.
        for gi in cutlass.range(cute.size(acc_G), unroll_full=True):
            acc_G[gi] = _reciprocal_full(
                1.0 + cute.math.exp(-acc_G[gi].to(BFloat16).to(Float32), fastmath=True)
            )
        for k in cutlass.range_constexpr(P_LOOP):
            stage = k % STAGES
            cute.arch.mbarrier_wait(
                mbar_full_ptr + STAGES + stage, Int32((k // STAGES) % 2)
            )
            warpgroup.fence()
            for ki in cutlass.range_constexpr(cute.size(tCsX2.shape[2])):
                cute.gemm(
                    mma_V,
                    acc_V,
                    tCsX2[None, None, ki, stage],
                    tCsW2[None, None, ki, stage],
                    acc_V,
                )
                mma_V.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()
            if cutlass.const_expr(k + STAGES < P_LOOP or k == P_LOOP - 1):
                warpgroup.wait_group(0)
            if cutlass.const_expr(k + STAGES < P_LOOP):
                cute.arch.barrier()
                if warp_idx == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar_full_ptr + STAGES + stage, tx_bytes_total
                        )
                    load_X2(
                        src_idx=k + STAGES,
                        dst_idx=stage,
                        tma_bar_ptr=mbar_full_ptr + STAGES + stage,
                    )
                    load_W2(
                        src_idx=k + STAGES,
                        dst_idx=stage,
                        tma_bar_ptr=mbar_full_ptr + STAGES + stage,
                    )
        cute.arch.mbarrier_wait(mbar_full_ptr + 2 * STAGES, Int32(0))
        coords = thr_mma.partition_C(cute.make_identity_tensor((TILE_M, TILE_N)))
        load_atom = quack_copy.sm90_get_smem_load_op(LayoutEnum.ROW_MAJOR, BFloat16)
        load_c = cute.make_tiled_copy_C(load_atom, tiled_mma).get_slice(tidx)
        if cutlass.const_expr(
            max(cute.cosize(sX1_layout), cute.cosize(sX2_layout))
            >= cute.cosize(sO_layout)
            and max(cute.cosize(sW1_layout), cute.cosize(sW2_layout))
            >= cute.cosize(sO_layout)
            and max(
                cute.cosize(sX1_layout),
                cute.cosize(sX2_layout),
                cute.cosize(sW1_layout),
                cute.cosize(sW2_layout),
            )
            >= 2 * cute.cosize(sO_layout)
        ):
            # Save projection to its final shared tile first, retiring FP32 acc_V.
            sP = storage.sX1.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
            sG = storage.sW1.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
            if cutlass.const_expr(
                max(cute.cosize(sX1_layout), cute.cosize(sX2_layout))
                >= 2 * cute.cosize(sO_layout)
            ):
                sY = cute.make_tensor(
                    sP.iterator + cute.cosize(sO_layout), sO_layout.outer
                )
            else:
                sY = cute.make_tensor(
                    sG.iterator + cute.cosize(sO_layout), sO_layout.outer
                )
            # Retired operand rings hold P/G/Y. If a fourth tile is free, stage
            # the broadcast dropout scale there without increasing shared memory.
            # Unaligned row wrapping / N tails retain the scalar/vector fallback.
            a_tiles = max(
                cute.cosize(sX1_layout), cute.cosize(sX2_layout)
            ) // cute.cosize(sO_layout)
            b_tiles = max(
                cute.cosize(sW1_layout), cute.cosize(sW2_layout)
            ) // cute.cosize(sO_layout)
            use_tma_ds = cutlass.const_expr(
                self.L % TILE_M == 0 and self.N % TILE_N == 0 and a_tiles + b_tiles >= 4
            )
            if cutlass.const_expr(use_tma_ds):
                if cutlass.const_expr(a_tiles >= 3):
                    sD = cute.make_tensor(
                        sP.iterator + 2 * cute.cosize(sO_layout), sO_layout.outer
                    )
                elif cutlass.const_expr(a_tiles >= 2 and b_tiles >= 2):
                    sD = cute.make_tensor(
                        sG.iterator + cute.cosize(sO_layout), sO_layout.outer
                    )
                else:
                    sD = cute.make_tensor(
                        sG.iterator + 2 * cute.cosize(sO_layout), sO_layout.outer
                    )
                gD_tma = cute.local_tile(
                    tD_tma, (TILE_M, TILE_N), (m_block % (self.L // TILE_M), n_block)
                )
                load_D, _, _ = quack_copy.tma_get_copy_fn(
                    atomD, 0, cute.make_layout(1), gD_tma, sD, single_stage=True
                )
            store_op = sm90h.get_smem_store_op(LayoutEnum.ROW_MAJOR, BFloat16, Float32)
            copyC = cute.make_tiled_copy_C(store_op, tiled_mma).get_slice(tidx)
            pfrag = cute.make_fragment_like(acc_V, BFloat16)
            pfrag.store(acc_V.load().to(BFloat16))
            cute.arch.barrier()
            if cutlass.const_expr(use_tma_ds):  # noqa: SIM102 - constexpr guards the DSL value
                if warp_idx == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar_full_ptr, TILE_M * TILE_N * 2
                        )
                    load_D(tma_bar_ptr=mbar_full_ptr)
            cute.copy(store_op, copyC.retile(pfrag), copyC.partition_D(sP))
            cute.arch.fence_view_async_shared()
            cute.arch.barrier()
            if warp_idx == 0:
                dst = cute.local_tile(tP, (TILE_M, TILE_N), (m_block, n_block))
                so, go = cpasync.tma_partition(
                    atomP,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sP, 0, cute.rank(sP)),
                    cute.group_modes(dst, 0, cute.rank(dst)),
                )
                cute.copy(atomP, so, go)
                with cute.arch.elect_one():
                    cute.arch.cp_async_bulk_commit_group()
            if cutlass.const_expr(use_tma_ds):
                # Gate stage zero has completed ceil(G_LOOP / STAGES) arrivals;
                # reuse its next phase after all gate/projection MMA reads retire.
                cute.arch.mbarrier_wait(
                    mbar_full_ptr, Int32(cute.ceil_div(G_LOOP, STAGES) % 2)
                )
            cg0 = copyC.retile(acc_G)
            cr0 = load_c.partition_S(sO)
            cc0 = copyC.retile(coords)
            cc = cute.group_modes(cc0, 1, cute.rank(cc0))
            global_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=32
            )
            if cutlass.const_expr(use_tma_ds):
                cd0 = load_c.partition_S(sD)
                cd = cute.group_modes(cd0, 1, cute.rank(cd0))
            elif cutlass.const_expr(self.L % TILE_M == 0 and self.N % TILE_N == 0):
                gD = cute.local_tile(
                    tD, (TILE_M, TILE_N), (m_block % (self.L // TILE_M), n_block)
                )
                cd0 = copyC.partition_S(gD)
                cd = cute.group_modes(cd0, 1, cute.rank(cd0))
            cp0 = load_c.partition_S(sP)
            co0 = copyC.partition_D(sY)
            cs0 = copyC.partition_D(sG)
            cg = cute.group_modes(cg0, 1, cute.rank(cg0))
            cr = cute.group_modes(cr0, 1, cute.rank(cr0))
            cp = cute.group_modes(cp0, 1, cute.rank(cp0))
            co = cute.group_modes(co0, 1, cute.rank(co0))
            cs = cute.group_modes(cs0, 1, cute.rank(cs0))
            for epi_idx in cutlass.range_constexpr(cute.size(cg, mode=[1])):
                gc = cg[None, epi_idx]
                pc = cute.make_fragment_like(gc, BFloat16)
                # Same C geometry as residual LDSM; the source P tile is read-only.
                cute.copy(load_atom, cp[None, epi_idx], pc)
                rc = cute.make_fragment_like(gc, BFloat16)
                dc = cute.make_fragment_like(gc, BFloat16)
                cute.copy(load_atom, cr[None, epi_idx], rc)
                if cutlass.const_expr(use_tma_ds):
                    cute.copy(load_atom, cd[None, epi_idx], dc)
                elif cutlass.const_expr(self.L % TILE_M == 0 and self.N % TILE_N == 0):
                    cute.copy(global_atom, cd[None, epi_idx], dc)
                else:
                    dc.fill(0)
                    for di in cutlass.range(cute.size(dc), unroll_full=True):
                        row = cutlass.Int64(m_block) * TILE_M + cc[None, epi_idx][di][0]
                        col = n_block * TILE_N + cc[None, epi_idx][di][1]
                        if row < tO.shape[0] and col < self.N:
                            dc[di] = tD[row % self.L, col]
                denom = cute.make_fragment_like(gc, Float32)
                denom.store(gc.load())
                yc = cute.make_fragment_like(gc, BFloat16)
                saved_gc = cute.make_fragment_like(gc, BFloat16)
                yc.store(
                    (
                        pc.load().to(Float32) * denom.load() * dc.load().to(Float32)
                        + rc.load().to(Float32)
                    ).to(BFloat16)
                )
                saved_gc.store(denom.load().to(BFloat16))
                cute.copy(store_op, yc, co[None, epi_idx])
                cute.copy(store_op, saved_gc, cs[None, epi_idx])
            cute.arch.fence_view_async_shared()
            cute.arch.barrier()
            if warp_idx == 0:
                for which in cutlass.range_constexpr(2):
                    if cutlass.const_expr(which == 0):
                        target, atom, shared = tO, atomO, sY
                    else:
                        target, atom, shared = tG, atomG, sG
                    dst = cute.local_tile(target, (TILE_M, TILE_N), (m_block, n_block))
                    so, go = cpasync.tma_partition(
                        atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(shared, 0, cute.rank(shared)),
                        cute.group_modes(dst, 0, cute.rank(dst)),
                    )
                    cute.copy(atom, so, go)
                with cute.arch.elect_one():
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.barrier()
        else:
            # Matrix load follows the accumulator fragment layout; TMA already
            # zero-filled rows/columns outside the residual's logical extent.
            residual_bf16 = cute.make_fragment_like(acc_G, BFloat16)
            load_atom = quack_copy.sm90_get_smem_load_op(LayoutEnum.ROW_MAJOR, BFloat16)
            load_c = cute.make_tiled_copy_C(load_atom, tiled_mma).get_slice(tidx)
            cute.copy(load_atom, load_c.partition_S(sO), load_c.retile(residual_bf16))
            rR = cute.make_fragment_like(acc_G, Float32)
            rR.store(residual_bf16.load().to(Float32))
            rD = cute.make_fragment_like(acc_G, Float32)
            rD.fill(0)
            if cutlass.const_expr(self.L % TILE_M == 0 and self.N % TILE_N == 0):
                # Each aligned row tile is one contiguous slice of the broadcast
                # scale. A packed BF16 pair is the accumulator's contiguous unit.
                gD = cute.local_tile(
                    tD, (TILE_M, TILE_N), (m_block % (self.L // TILE_M), n_block)
                )
                scale_bf16 = cute.make_fragment_like(acc_G, BFloat16)
                global_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=32
                )
                copy_d = cute.make_tiled_copy_C(global_atom, tiled_mma).get_slice(tidx)
                cute.copy(
                    global_atom, copy_d.partition_S(gD), copy_d.retile(scale_bf16)
                )
                rD.store(scale_bf16.load().to(Float32))
            else:
                for i in cutlass.range(cute.size(acc_G), unroll_full=True):
                    row = cutlass.Int64(m_block) * TILE_M + coords[i][0]
                    col = n_block * TILE_N + coords[i][1]
                    if row < tO.shape[0] and col < self.N:
                        rD[i] = tD[row % self.L, col].to(Float32)
            pv = acc_V.load().to(BFloat16)
            denom = cute.make_fragment_like(acc_G, Float32)
            denom.store(acc_G.load())
            gv = denom.load()
            out_frag = cute.make_fragment_like(acc_G, BFloat16)
            out_frag.store((pv.to(Float32) * gv * rD.load() + rR.load()).to(BFloat16))
            cute.arch.barrier()
            store_op = sm90h.get_smem_store_op(LayoutEnum.ROW_MAJOR, BFloat16, Float32)
            copyC = cute.make_tiled_copy_C(store_op, tiled_mma).get_slice(tidx)
            for which in cutlass.range_constexpr(3):
                if cutlass.const_expr(which == 0):
                    target = tO
                    atom = atomO
                elif cutlass.const_expr(which == 1):
                    out_frag.store(pv)
                    target = tP
                    atom = atomP
                else:
                    out_frag.store(gv.to(BFloat16))
                    target = tG
                    atom = atomG
                cute.copy(store_op, copyC.retile(out_frag), copyC.partition_D(sO))
                cute.arch.fence_view_async_shared()
                cute.arch.barrier()
                if warp_idx == 0:
                    dst = cute.local_tile(target, (TILE_M, TILE_N), (m_block, n_block))
                    so, go = cpasync.tma_partition(
                        atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sO, 0, cute.rank(sO)),
                        cute.group_modes(dst, 0, cute.rank(dst)),
                    )
                    cute.copy(atom, so, go)
                    with cute.arch.elect_one():
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.barrier()

    @cute.jit
    def __call__(
        self,
        mX1: cute.Tensor,
        mX2: cute.Tensor,
        mW1: cute.Tensor,
        mW2: cute.Tensor,
        mO: cute.Tensor,
        mP: cute.Tensor,
        mG: cute.Tensor,
        mR: cute.Tensor,
        mD: cute.Tensor,
        stream: cuda.CUstream,
    ):
        TILE_M: cutlass.Constexpr[int] = self.tile_m
        TILE_N: cutlass.Constexpr[int] = self.tile_n
        TILE_K: cutlass.Constexpr[int] = self.tile_k
        STAGES: cutlass.Constexpr[int] = self.num_stages
        G_LOOP: cutlass.Constexpr[int] = self.g_loop
        P_LOOP: cutlass.Constexpr[int] = self.p_loop

        M = mX1.shape[0]
        m_blocks = cute.ceil_div(M, TILE_M) * cute.ceil_div(self.N, TILE_N)

        atom = warpgroup.make_smem_layout_atom(
            sm90h.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, BFloat16, TILE_K), BFloat16
        )
        sX1_layout = cute.tile_to_shape(
            atom, (TILE_M, TILE_K, min(STAGES, G_LOOP)), order=(0, 1, 2)
        )
        sX2_layout = cute.tile_to_shape(
            atom, (TILE_M, TILE_K, min(STAGES, P_LOOP)), order=(0, 1, 2)
        )
        wg_layout = (
            LayoutEnum.ROW_MAJOR
            if cutlass.const_expr(self.wg_k_major)
            else LayoutEnum.COL_MAJOR
        )
        wp_layout = (
            LayoutEnum.ROW_MAJOR
            if cutlass.const_expr(self.wp_k_major)
            else LayoutEnum.COL_MAJOR
        )
        wg_atom = warpgroup.make_smem_layout_atom(
            sm90h.get_smem_layout_atom(
                wg_layout,
                BFloat16,
                TILE_K if cutlass.const_expr(self.wg_k_major) else TILE_N,
            ),
            BFloat16,
        )
        wp_atom = warpgroup.make_smem_layout_atom(
            sm90h.get_smem_layout_atom(
                wp_layout,
                BFloat16,
                TILE_K if cutlass.const_expr(self.wp_k_major) else TILE_N,
            ),
            BFloat16,
        )
        sW1_layout = cute.tile_to_shape(
            wg_atom,
            (TILE_N, TILE_K, min(STAGES, G_LOOP)),
            order=(0, 1, 2) if cutlass.const_expr(self.wg_k_major) else (1, 0, 2),
        )
        sW2_layout = cute.tile_to_shape(
            wp_atom,
            (TILE_N, TILE_K, min(STAGES, P_LOOP)),
            order=(0, 1, 2) if cutlass.const_expr(self.wp_k_major) else (1, 0, 2),
        )

        tiled_mma = sm90h.make_trivial_tiled_mma(
            BFloat16,
            BFloat16,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K
            if cutlass.const_expr(self.wg_k_major)
            else warpgroup.OperandMajorMode.MN,
            Float32,
            (self.mma_mgroups, self.mma_ngroups, 1),
            (64, TILE_N // self.mma_ngroups),
        )

        sX_stage = cute.slice_(sX1_layout, (None, None, 0))
        sW_stage = cute.slice_(sW1_layout, (None, None, 0))
        tma_atom_X1, tma_X1 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mX1,
            sX_stage,
            (TILE_M, TILE_K),
        )
        tma_atom_X2, tma_X2 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mX2,
            sX_stage,
            (TILE_M, TILE_K),
        )
        tma_atom_W1, tma_W1 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mW1,
            sW_stage,
            (TILE_N, TILE_K),
        )
        tma_atom_W2, tma_W2 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mW2,
            cute.slice_(sW2_layout, (None, None, 0)),
            (TILE_N, TILE_K),
        )
        out_atom = warpgroup.make_smem_layout_atom(
            sm90h.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, BFloat16, TILE_N), BFloat16
        )
        sO_layout = cute.tile_to_shape(out_atom, (TILE_M, TILE_N), order=(0, 1))
        atomO, tO = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mO, sO_layout, (TILE_M, TILE_N)
        )
        atomP, tP = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mP, sO_layout, (TILE_M, TILE_N)
        )
        atomG, tG = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mG, sO_layout, (TILE_M, TILE_N)
        )
        so = cute.struct.Align[
            cute.struct.MemRange[BFloat16, cute.cosize(sO_layout)], 1024
        ]
        atomR, tR = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mR, sO_layout, (TILE_M, TILE_N)
        )
        atomD, tD_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mD, sO_layout, (TILE_M, TILE_N)
        )
        tx_bytes_total = (TILE_M + TILE_N) * TILE_K * 2
        sx1 = cute.struct.Align[
            cute.struct.MemRange[
                BFloat16, max(cute.cosize(sX1_layout), cute.cosize(sX2_layout))
            ],
            1024,
        ]

        sw1 = cute.struct.Align[
            cute.struct.MemRange[
                BFloat16, max(cute.cosize(sW1_layout), cute.cosize(sW2_layout))
            ],
            1024,
        ]

        @cute.struct
        class SharedStorage:
            mbar_full: cute.struct.MemRange[cutlass.Int64, 2 * STAGES + 1]
            sX1: sx1
            sW1: sw1
            sO: so

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_X1,
            tma_X1,
            tma_atom_X2,
            tma_X2,
            tma_atom_W1,
            tma_W1,
            tma_atom_W2,
            tma_W2,
            atomO,
            atomP,
            atomG,
            atomR,
            atomD,
            tD_tma,
            sO_layout,
            tO,
            tP,
            tG,
            tR,
            mD,
            sX1_layout,
            sX2_layout,
            sW1_layout,
            sW2_layout,
            tiled_mma,
            Int32(tx_bytes_total),
        ).launch(
            grid=[m_blocks, 1, 1],
            stream=stream,
            block=[self.num_threads, 1, 1],  # whole 128-thread SM90 warpgroups
        )


_COMPILE_CACHE = {}


def _output_f567_sm90_fake(norm, x, wp, wg, residual, dropscale, seq_len):
    """Return output metadata without executing GPU work."""
    return tuple(norm.new_empty((norm.shape[0], wp.shape[0])) for _ in range(3))


def feasibility(config, smem_limit=232448, kp=None, kg=None):
    """Return an explicit hardware/resource rejection reason, or None."""
    bm, bn, bk, gm, nw, ns = (
        config[k]
        for k in (
            "BLOCK_M1",
            "BLOCK_N",
            "BLOCK_K",
            "GROUP_M",
            "num_warps",
            "num_stages",
        )
    )
    if bm not in (64, 128):
        return "WGMMA instruction M is 64; this schedule needs whole M warpgroups"
    if nw not in (4, 8):
        return "WGMMA needs whole four-warp groups"
    if (
        bn not in (32, 64, 128, 256)
        or bk not in (16, 32, 64, 128)
        or gm not in (1, 2, 4, 8)
        or ns not in (2, 3, 4)
    ):
        return "Config is outside corresponding Triton domain"
    # One shared A/B operand ring, one epilogue tile, and alignment padding.
    ps = min(ns, (kp + bk - 1) // bk) if kp is not None else ns
    gs = min(ns, (kg + bk - 1) // bk) if kg is not None else ns
    if 2 * (bm + bn) * bk * max(ps, gs) + 2 * bm * bn + 1024 > smem_limit:
        return "TMA pipeline stage storage exceeds per-CTA shared memory"
    return None


def output_f567_impl(norm, x, wp, wg, residual, dropscale, seq_len, config=None):
    """Strict no-copy SM90 launch returning (y, projection, gate).

    Config keys match Triton. Hardware-infeasible schedules are rejected, not
    silently rewritten. TMA supports contiguous/transposed aligned weights.
    """
    m, kp = norm.shape
    kg, n = wg.shape
    tensors = (norm, x, wp, wg, residual, dropscale)
    if not norm.is_cuda or torch.cuda.get_device_capability(norm.device) != (9, 0):
        raise ValueError("F567 requires SM90")
    if any(t.dtype != torch.bfloat16 or t.device != norm.device for t in tensors):
        raise ValueError("F567 requires BF16 operands on the same device")
    if min(m, kp, kg, n, seq_len) <= 0 or any(v % 8 for v in (kp, kg, n)):
        raise ValueError("TMA requires positive dimensions and aligned widths")
    if (
        x.shape != (m, kg)
        or wp.shape != (n, kp)
        or residual.shape != (m, n)
        or dropscale.shape != (seq_len, n)
    ):
        raise ValueError("F567 shapes disagree")
    if any(not t.is_contiguous() for t in (norm, x, residual, dropscale)):
        raise ValueError("F567 activations must be contiguous")
    for t in tensors:
        if t.data_ptr() % 16 or (min(t.stride()) != 1) or (max(t.stride()) % 8):
            raise ValueError(
                "F567 TMA layout must have unit inner stride and aligned outer stride"
            )
    limit = torch.cuda.get_device_properties(norm.device).shared_memory_per_block_optin
    outputs = _output_f567_sm90_fake(norm, x, wp, wg, residual, dropscale, seq_len)
    # The output buffers, metadata-only transpose, and DLPack adapters are made
    # once per invocation. Native tuning replays only the prepared launch below.
    wg_nk = wg.t()
    args = [
        from_dlpack(t.detach(), assumed_align=16)
        for t in (x, norm, wg_nk, wp, *outputs, residual, dropscale)
    ]
    args.append(cuda.CUstream(torch.cuda.current_stream(norm.device).cuda_stream))
    from miniworld_engine.autotune.native import tensor_key

    key_prefix = (str(norm.device), tensor_key(*tensors, extra=(seq_len,)))

    def launch_output_candidate(candidate):
        reason = feasibility(candidate, limit, kp, kg)
        if reason:
            raise ValueError(reason)
        key = (*key_prefix, tuple(sorted(candidate.items())))
        if key not in _COMPILE_CACHE:
            kernel = ParityF567Sm90(
                n,
                kp,
                kg,
                seq_len,
                **candidate,
                wg_k_major=wg_nk.stride(1) == 1,
                wp_k_major=wp.stride(1) == 1,
            )
            _COMPILE_CACHE[key] = cute.compile(kernel, *args)
        _COMPILE_CACHE[key](*args)

    if config is None:
        from miniworld_engine.autotune.trimul_sm90_config import resolve

        config = resolve(
            "trimul_output_f567_train_sm90_cute",
            tensors,
            extra=(seq_len, limit),
            feasibility=lambda c: feasibility(c, limit, kp, kg),
            run=launch_output_candidate,
        )
    launch_output_candidate(config)
    return outputs


@opaque(fake=_output_f567_sm90_fake, name="trimul_parity_f567_sm90")
def output_f567_sm90(
    norm: torch.Tensor,
    x: torch.Tensor,
    wp: torch.Tensor,
    wg: torch.Tensor,
    residual: torch.Tensor,
    dropscale: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact Triton F567 fusion with TMA/WGMMA and native config/cache selection."""
    return output_f567_impl(norm, x, wp, wg, residual, dropscale, seq_len)
