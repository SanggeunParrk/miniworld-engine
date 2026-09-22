"""Four-warp F2 with explicit TMA/WGMMA and Triton's output contract.

One CTA processes one M tile and traverses the interleaved N chunks. When all K
chunks fit the requested stage ring, A stays resident across N chunks. Separate
A/B operand rings and epilogue storage let the next B transfer overlap gate math,
and let output TMA stores overlap the next GEMM. Stage counts remain real slots.
The row mask is loaded once per CTA. Packed BF16 shuffles preserve exact bits
while rearranging gated columns for STSM/TMA stores. No global intermediate or
fusion boundary is added; raw saved values and both BF16 rounding steps match
Triton, including tiny sigmoid values.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90h
from cutlass import BFloat16, Float32
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum
from quack import copy_utils
from miniworld_engine.kernels.trimul_inproj.cute.parity_f567 import _reciprocal_full
from miniworld_engine.kernels._quack_compat import is_compile_only
from cuda.bindings import driver as cuda
import torch


class FrontSingleWarpgroupSm90:
    def __init__(self, k, n, bm, bh, bk, stages, save, mask):
        self.k, self.n, self.bm, self.bn, self.bk, self.stages = (
            k,
            n,
            bm,
            2 * bh,
            bk,
            stages,
        )
        self.save, self.mask = save, mask

    @cute.kernel
    def kernel(self, aa, ta, ab, tb, ao, to, ap, tp, mask, la, lb, lo, lp, mma, mmao):
        tid, _, _ = cute.arch.thread_idx()
        pid, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        storage = cutlass.utils.SmemAllocator().allocate(self.storage)
        sa = storage.a.get_tensor(la.outer, swizzle=la.inner)
        sb = storage.b.get_tensor(lb.outer, swizzle=lb.inner)
        bar = storage.bar.data_ptr()
        sp = storage.e.get_tensor(lp.outer, swizzle=lp.inner)
        output_pointer = cute.recast_ptr(
            storage.e.data_ptr() + (cute.cosize(lp) if self.save else 0),
            swizzle_=lo.inner,
        )
        so = cute.make_tensor(output_pointer, lo.outer)
        if warp == 0:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(aa)
                cpasync.prefetch_descriptor(ab)
                for i in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(bar + i, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        thr = mma.get_slice(tid)
        coords = thr.partition_C(cute.make_identity_tensor((self.bm, self.bn)))
        mask_cache = cute.make_rmem_tensor((cute.size(coords) // 2,), Float32)
        if cutlass.const_expr(self.mask):
            for i in cutlass.range(cute.size(mask_cache), unroll_full=True):
                row = pid * self.bm + coords[2 * i][0]
                value = Float32(0)
                if row < mask.shape[0]:
                    value = mask[row].to(Float32)
                mask_cache[i] = value
        # Barrier phases persist across N chunks. A stage is used ceil((Ktiles-j)/stages)
        # times per chunk; this also handles K-trip counts not divisible by stages.
        for ni in cutlass.range(cute.ceil_div(self.n, self.bn)):
            mi = pid
            ga = cute.local_tile(ta, (self.bm, self.bk), (mi, None))
            gb = cute.local_tile(tb, (self.bn, self.bk), (ni, None))
            load_a, _, _ = copy_utils.tma_get_copy_fn(
                aa, 0, cute.make_layout(1), ga, sa
            )
            load_b, _, _ = copy_utils.tma_get_copy_fn(
                ab, 0, cute.make_layout(1), gb, sb
            )
            acc = cute.make_fragment(thr.partition_shape_C((self.bm, self.bn)), Float32)
            xa = thr.make_fragment_A(thr.partition_A(sa))
            xb = thr.make_fragment_B(thr.partition_B(sb))
            atom = cute.make_mma_atom(mma.op)
            atom.set(warpgroup.Field.ACCUMULATE, True)
            acc.fill(0)
            loops = cutlass.const_expr((self.k + self.bk - 1) // self.bk)
            if warp == 0 and ni == 0:
                for i in cutlass.range_constexpr(min(self.stages, loops)):
                    with cute.arch.elect_one():
                        tx = self.bn * self.bk * 2
                        if ni == 0 or cutlass.const_expr(loops > self.stages):
                            tx = tx + self.bm * self.bk * 2
                        cute.arch.mbarrier_arrive_and_expect_tx(bar + i, tx)
                    if ni == 0 or cutlass.const_expr(loops > self.stages):
                        load_a(src_idx=i, dst_idx=i, tma_bar_ptr=bar + i)
                    load_b(src_idx=i, dst_idx=i, tma_bar_ptr=bar + i)
            for step in cutlass.range(loops):
                stage = step % self.stages
                cute.arch.mbarrier_wait(
                    bar + stage,
                    (
                        ni * ((loops + self.stages - 1 - stage) // self.stages)
                        + step // self.stages
                    )
                    & 1,
                )
                cute.arch.fence_view_async_shared()
                warpgroup.fence()
                for ki in cutlass.range_constexpr(cute.size(xa.shape[2])):
                    cute.gemm(
                        atom,
                        acc,
                        xa[None, None, ki, stage],
                        xb[None, None, ki, stage],
                        acc,
                    )
                warpgroup.commit_group()
                if cutlass.const_expr(loops > self.stages):
                    warpgroup.wait_group(0)
                    cute.arch.barrier()
                elif cutlass.const_expr(loops > 8):
                    warpgroup.wait_group(7)
                if warp == 0 and step + self.stages < loops:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            bar + stage, (self.bm + self.bn) * self.bk * 2
                        )
                    load_a(
                        src_idx=step + self.stages,
                        dst_idx=stage,
                        tma_bar_ptr=bar + stage,
                    )
                    load_b(
                        src_idx=step + self.stages,
                        dst_idx=stage,
                        tma_bar_ptr=bar + stage,
                    )
            warpgroup.wait_group(0)
            cute.arch.barrier()
            # Prefetch the next channel chunk while this chunk's epilogue runs.
            # Its inputs and outputs occupy disjoint shared-memory regions.
            if warp == 0 and ni + 1 < cute.ceil_div(self.n, self.bn):
                next_b = cute.local_tile(tb, (self.bn, self.bk), (ni + 1, None))
                load_next_b, _, _ = copy_utils.tma_get_copy_fn(
                    ab, 0, cute.make_layout(1), next_b, sb
                )
                for i in cutlass.range_constexpr(min(self.stages, loops)):
                    with cute.arch.elect_one():
                        tx = self.bn * self.bk * 2
                        if cutlass.const_expr(loops > self.stages):
                            tx = tx + self.bm * self.bk * 2
                        cute.arch.mbarrier_arrive_and_expect_tx(bar + i, tx)
                    if cutlass.const_expr(loops > self.stages):
                        load_a(src_idx=i, dst_idx=i, tma_bar_ptr=bar + i)
                    load_next_b(src_idx=i, dst_idx=i, tma_bar_ptr=bar + i)
            # The prior output store overlapped the current GEMM. Finish its
            # shared-memory reads only when this epilogue is about to overwrite E.
            if warp == 0:
                with cute.arch.elect_one():
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.barrier()
            gate_out = cute.make_rmem_tensor((cute.size(acc) // 2,), BFloat16)
            for i in cutlass.range(cute.size(acc) // 2, unroll_full=True):
                row, _ = coords[2 * i]
                gate = acc[2 * i]
                proj = acc[2 * i + 1]
                # WGMMA C's innermost two values are the adjacent N pair.
                value = (
                    proj * _reciprocal_full(1.0 + cute.math.exp(-gate, fastmath=True))
                ).to(BFloat16)
                if cutlass.const_expr(self.mask):
                    value = (value.to(Float32) * mask_cache[i]).to(BFloat16)
                gate_out[i] = value
            store_op = sm90h.get_smem_store_op(LayoutEnum.COL_MAJOR, BFloat16, Float32)
            shuffled = cute.make_fragment(
                mmao.get_slice(tid).partition_shape_C((self.bm, self.bn // 2)), BFloat16
            )
            lane = tid % 32
            # A gate pair collapses two N columns into one per lane. The half-N
            # STSM layout expects adjacent pairs again. Pack both possible BF16
            # source values into one word, shuffle once within the four-lane
            # subgroup, and select the required half without FP32 conversion.
            for i in cutlass.range(cute.size(shuffled), unroll_full=True):
                si = (i // 4) * 4 + (i % 4) // 2
                source = (lane // 4) * 4 + (2 * (lane % 4) + i % 2) % 4
                pair = cute.make_rmem_tensor((2,), BFloat16)
                pair[0] = gate_out[si]
                pair[1] = gate_out[si + 2]
                word = cute.recast_tensor(pair, cutlass.Uint32)
                moved = cute.arch.shuffle_sync(word[0], source)
                word[0] = moved >> ((lane % 4 // 2) * 16)
                shuffled[i] = pair[0]
            copy_o = cute.make_tiled_copy_C(store_op, mmao).get_slice(tid)
            cute.copy(store_op, copy_o.retile(shuffled), copy_o.partition_D(so))
            if cutlass.const_expr(self.save):
                raw = cute.make_fragment_like(acc, BFloat16)
                raw.store(acc.load().to(BFloat16))
                copy_p = cute.make_tiled_copy_C(store_op, mma).get_slice(tid)
                cute.copy(store_op, copy_p.retile(raw), copy_p.partition_D(sp))
            cute.arch.fence_view_async_shared()
            cute.arch.barrier()
            if warp == 0:
                go = cute.local_tile(to, (self.bm, self.bn // 2), (mi, ni))
                ss, gg = cpasync.tma_partition(
                    ao,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(so, 0, cute.rank(so)),
                    cute.group_modes(go, 0, cute.rank(go)),
                )
                cute.copy(ao, ss, gg)
                if cutlass.const_expr(self.save):
                    gp = cute.local_tile(tp, (self.bm, self.bn), (mi, ni))
                    ss, gg = cpasync.tma_partition(
                        ap,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sp, 0, cute.rank(sp)),
                        cute.group_modes(gp, 0, cute.rank(gp)),
                    )
                    cute.copy(ap, ss, gg)
                with cute.arch.elect_one():
                    cute.arch.cp_async_bulk_commit_group()
            cute.arch.barrier()
        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.cp_async_bulk_wait_group(0, read=True)
        cute.arch.barrier()

    @cute.jit
    def __call__(self, a, b, o, p, mask, stream: cuda.CUstream):
        def layout(rows, cols, major, stages=None):
            atom = warpgroup.make_smem_layout_atom(
                sm90h.get_smem_layout_atom(
                    LayoutEnum.ROW_MAJOR if major else LayoutEnum.COL_MAJOR,
                    BFloat16,
                    cols if major else rows,
                ),
                BFloat16,
            )
            shape = (rows, cols) if stages is None else (rows, cols, stages)
            order = (
                ((0, 1) if major else (1, 0))
                if stages is None
                else ((0, 1, 2) if major else (1, 0, 2))
            )
            return cute.tile_to_shape(atom, shape, order=order)

        la = layout(self.bm, self.bk, True, self.stages)
        lb = layout(self.bn, self.bk, False, self.stages)
        lo = layout(self.bm, self.bn // 2, False)
        lp = layout(self.bm, self.bn, False)
        aa, ta = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            a,
            cute.slice_(la, (None, None, 0)),
            (self.bm, self.bk),
        )
        ab, tb = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            b,
            cute.slice_(lb, (None, None, 0)),
            (self.bn, self.bk),
        )
        ao, to = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), o, lo, (self.bm, self.bn // 2)
        )
        if cutlass.const_expr(self.save):
            ap, tp = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), p, lp, (self.bm, self.bn)
            )
        else:
            ap, tp = ao, to
        mma = sm90h.make_trivial_tiled_mma(
            BFloat16,
            BFloat16,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            (1, 1, 1),
            (64, self.bn),
        )
        mmao = sm90h.make_trivial_tiled_mma(
            BFloat16,
            BFloat16,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            (1, 1, 1),
            (64, self.bn // 2),
        )
        at = cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(la)], 1024]
        bt = cute.struct.Align[
            cute.struct.MemRange[
                BFloat16,
                cute.cosize(lb),
            ],
            1024,
        ]

        et = cute.struct.Align[
            cute.struct.MemRange[
                BFloat16, cute.cosize(lo) + (cute.cosize(lp) if self.save else 0)
            ],
            1024,
        ]

        @cute.struct
        class Storage:
            bar: cute.struct.MemRange[cutlass.Int64, self.stages]
            a: at
            b: bt
            e: et

        self.storage = Storage
        self.kernel(
            aa, ta, ab, tb, ao, to, ap, tp, mask, la, lb, lo, lp, mma, mmao
        ).launch(
            grid=[cute.ceil_div(a.shape[0], self.bm), 1, 1],
            block=[128, 1, 1],
            stream=stream,
        )


_COMPILE_CACHE = {}


def launch_four_warp(a, w, out, pre, mask, c):
    tensors = (
        a,
        w.T,
        out.T,
        pre.T if pre is not None else out.T,
        mask if mask is not None else a[:, 0],
    )
    if any(
        t.dtype != torch.bfloat16 or not t.is_cuda or t.device != a.device
        for t in tensors
    ):
        raise ValueError("Four-warp F2 requires BF16 CUDA operands on the same device")
    args = [from_dlpack(t.detach(), assumed_align=16) for t in tensors]
    args.append(cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    key = (
        str(a.device),
        tuple((tuple(t.shape), t.stride(), str(t.dtype)) for t in tensors),
        tuple(sorted(c.items())),
        pre is not None,
        mask is not None,
    )
    if key not in _COMPILE_CACHE:
        _COMPILE_CACHE[key] = cute.compile(
            FrontSingleWarpgroupSm90(
                a.shape[1],
                w.shape[1],
                c["BLOCK_M1"],
                c["BLOCK_K_H2"],
                c["BLOCK_K_D"],
                c["num_stages"],
                pre is not None,
                mask is not None,
            ),
            *args,
        )
    if not is_compile_only():
        _COMPILE_CACHE[key](*args)


def four_warp_shared_bytes(config, save_preact=True):
    """Aligned, disjoint barrier/A-ring/B-ring/epilogue storage size."""
    bm, bn, bk = config["BLOCK_M1"], 2 * config["BLOCK_K_H2"], config["BLOCK_K_D"]
    stages = config["num_stages"]
    align = lambda size: (size + 1023) // 1024 * 1024
    return (
        align(stages * 8)
        + align(bm * bk * stages * 2)
        + align(bn * bk * stages * 2)
        + align(bm * bn * (3 if save_preact else 1))
    )
