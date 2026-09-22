"""SM90 implementation of Triton B9+B10 with identical rounding and buffers.

Explicit TMA loads and WGMMA. Tiles smaller than the instruction's 64-row
minimum are explicitly rejected.
Stages are real independent shared-memory K slots in a circular TMA pipeline.
Short gate reductions prefetch front tiles into unused and then retired gate
stages before the front reduction begins. No operand transposes are materialized. The second GEMM accepts the production
column-major dconc.T and column-major W_stack.T views directly.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90h
import torch
from cuda.bindings import driver as cuda
from cutlass import BFloat16, Float32
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum
from quack import copy_utils

from miniworld_engine.kernels._compile import opaque


class DualBackwardSm90:
    def __init__(
        self, n, kg, kp, BLOCK_M1, BLOCK_N, BLOCK_K, GROUP_M, num_warps=4, num_stages=2
    ):
        self.n, self.kg, self.kp = n, kg, kp
        self.logical_m, self.bn, self.bk = BLOCK_M1, BLOCK_N, BLOCK_K
        self.group, self.warps, self.stages = GROUP_M, num_warps, num_stages
        self.pm = BLOCK_M1
        self.mma_mgroups = min(num_warps // 4, BLOCK_M1 // 64)
        self.mma_ngroups = num_warps // 4 // self.mma_mgroups
        self.gate_tiles = (kg + BLOCK_K - 1) // BLOCK_K
        self.front_offset = self.gate_tiles if self.gate_tiles < num_stages else 0
        self.front_tiles = (kp + BLOCK_K - 1) // BLOCK_K
        # If the gate fits in the ring, unused slots can hold early front
        # tiles. Each gate slot joins that ring after its last WGMMA read.
        self.short_gate = self.gate_tiles <= num_stages
        self.front_initial = min(max(0, num_stages - self.gate_tiles), self.front_tiles)
        self.front_ahead = min(num_stages, self.front_tiles) if self.short_gate else 0
        self.shared_storage = None

    @cute.kernel
    def kernel(
        self,
        ag,
        tg,
        af,
        tf,
        aw,
        tw,
        av,
        tv,
        y,
        ay,
        ty,
        lo: cute.ComposedLayout,
        lg: cute.ComposedLayout,
        lf: cute.ComposedLayout,
        lw: cute.ComposedLayout,
        lv: cute.ComposedLayout,
        mg: cute.TiledMma,
        mf: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        pid, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        nm = cute.ceil_div(y.shape[0], self.logical_m)
        nn = cute.ceil_div(self.n, self.bn)
        first = pid // (self.group * nn) * self.group
        actual = min(self.group, nm - first)
        local = pid % (self.group * nn)
        mi, ni = first + local % actual, local // actual
        # Keep the grouped tile origin in TMA coordinates; edge rows are masked.
        row_origin = mi * self.logical_m
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sg = storage.sg.get_tensor(lg.outer, swizzle=lg.inner)
        sf = storage.sg.get_tensor(lf.outer, swizzle=lf.inner)
        sw = storage.sw.get_tensor(lw.outer, swizzle=lw.inner)
        sv = storage.sw.get_tensor(lv.outer, swizzle=lv.inner)
        bar = storage.bar.data_ptr()
        if warp == 0:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(ag)
                cpasync.prefetch_descriptor(af)
                cpasync.prefetch_descriptor(aw)
                cpasync.prefetch_descriptor(av)
                for stage in cutlass.range_constexpr(2 * self.stages):
                    cute.arch.mbarrier_init(bar + stage, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        # domain_offset moves the TMA coordinate origin, not the base pointer.
        gg = cute.local_tile(
            cute.domain_offset((row_origin, 0), tg), (self.pm, self.bk), (0, None)
        )
        gf = cute.local_tile(
            cute.domain_offset((row_origin, 0), tf), (self.pm, self.bk), (0, None)
        )
        gw = cute.local_tile(tw, (self.bn, self.bk), (ni, None))
        gv = cute.local_tile(tv, (self.bn, self.bk), (ni, None))
        loadg, _, _ = copy_utils.tma_get_copy_fn(ag, 0, cute.make_layout(1), gg, sg)
        loadf, _, f_prefetch_src = copy_utils.tma_get_copy_fn(af, 0, cute.make_layout(1), gf, sf)
        loadw, _, _ = copy_utils.tma_get_copy_fn(aw, 0, cute.make_layout(1), gw, sw)
        loadv, _, _ = copy_utils.tma_get_copy_fn(av, 0, cute.make_layout(1), gv, sv)
        if warp == 0:
            for pre in cutlass.range_constexpr(self.stages, min(3 * self.stages, self.front_tiles)):
                cute.prefetch(af, f_prefetch_src[None, pre])
        thrg, thrf = mg.get_slice(tidx), mf.get_slice(tidx)
        accg = cute.make_fragment(thrg.partition_shape_C((self.pm, self.bn)), Float32)
        accf = cute.make_fragment(thrf.partition_shape_C((self.pm, self.bn)), Float32)
        gate_saved = cute.make_fragment_like(accg, BFloat16)
        xg = thrg.make_fragment_A(thrg.partition_A(sg))
        wg = thrg.make_fragment_B(thrg.partition_B(sw))
        xf = thrf.make_fragment_A(thrf.partition_A(sf))
        vf = thrf.make_fragment_B(thrf.partition_B(sv))
        mag, maf = cute.make_mma_atom(mg.op), cute.make_mma_atom(mf.op)
        mag.set(warpgroup.Field.ACCUMULATE, True)
        maf.set(warpgroup.Field.ACCUMULATE, True)
        # Each stage has its own barrier/phase. Refill the consumed slot after
        # WGMMA releases it; that transfer overlaps computation on the next slot.
        for kind in cutlass.range_constexpr(2):
            kbar = bar + kind * self.stages
            if cutlass.const_expr(kind == 0):
                accg.fill(0)
                loops = cutlass.const_expr((self.kg + self.bk - 1) // self.bk)
            else:
                cute.arch.barrier()
                accf.fill(0)
                loops = cutlass.const_expr((self.kp + self.bk - 1) // self.bk)
            if warp == 0:
                for pre in cutlass.range_constexpr(min(self.stages, loops)):
                    if cutlass.const_expr(kind == 0):
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                kbar + pre, (self.pm + self.bn) * self.bk * 2
                            )
                        loadg(src_idx=pre, dst_idx=pre, tma_bar_ptr=kbar + pre)
                        loadw(src_idx=pre, dst_idx=pre, tma_bar_ptr=kbar + pre)
                    else:
                        if cutlass.const_expr(pre >= self.front_ahead):
                            dst_stage = (pre + self.front_offset) % self.stages
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    kbar + dst_stage, (self.pm + self.bn) * self.bk * 2
                                )
                            loadf(
                                src_idx=pre,
                                dst_idx=dst_stage,
                                tma_bar_ptr=kbar + dst_stage,
                            )
                            loadv(
                                src_idx=pre,
                                dst_idx=dst_stage,
                                tma_bar_ptr=kbar + dst_stage,
                            )
                if cutlass.const_expr(kind == 0):
                    for pre in cutlass.range_constexpr(self.front_initial):
                        dst_stage = pre + self.front_offset
                        front_bar = bar + self.stages + dst_stage
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                front_bar, (self.pm + self.bn) * self.bk * 2
                            )
                        loadf(src_idx=pre, dst_idx=dst_stage, tma_bar_ptr=front_bar)
                        loadv(src_idx=pre, dst_idx=dst_stage, tma_bar_ptr=front_bar)
            for step in cutlass.range(loops):
                stage = (step + (self.front_offset if kind == 1 else 0)) % self.stages
                phase = (step // self.stages) & 1
                cute.arch.mbarrier_wait(kbar + stage, phase)
                cute.arch.fence_view_async_shared()
                warpgroup.fence()
                for k in cutlass.range_constexpr(cute.size(xg.shape[2])):
                    if cutlass.const_expr(kind == 0):
                        cute.gemm(
                            mag,
                            accg,
                            xg[None, None, k, stage],
                            wg[None, None, k, stage],
                            accg,
                        )
                    else:
                        cute.gemm(
                            maf,
                            accf,
                            xf[None, None, k, stage],
                            vf[None, None, k, stage],
                            accf,
                        )
                warpgroup.commit_group()
                if cutlass.const_expr(kind == 1):
                    future = step + 3 * self.stages
                    if warp == 0 and future < self.front_tiles:
                        cute.prefetch(af, f_prefetch_src[None, future])
                warpgroup.wait_group(0)
                cute.arch.barrier()
                if cutlass.const_expr(kind == 0 and self.short_gate):
                    # wait_group(0) and the CTA barrier above retire this slot
                    # for every consumer before TMA changes its operand layout.
                    if warp == 0 and self.front_initial + step < self.front_tiles:
                        front_bar = bar + self.stages + step
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                front_bar, (self.pm + self.bn) * self.bk * 2
                            )
                        loadf(
                            src_idx=self.front_initial + step,
                            dst_idx=step,
                            tma_bar_ptr=front_bar,
                        )
                        loadv(
                            src_idx=self.front_initial + step,
                            dst_idx=step,
                            tma_bar_ptr=front_bar,
                        )
                if warp == 0 and step + self.stages < loops:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            kbar + stage, (self.pm + self.bn) * self.bk * 2
                        )
                    if cutlass.const_expr(kind == 0):
                        loadg(
                            src_idx=step + self.stages,
                            dst_idx=stage,
                            tma_bar_ptr=kbar + stage,
                        )
                        loadw(
                            src_idx=step + self.stages,
                            dst_idx=stage,
                            tma_bar_ptr=kbar + stage,
                        )
                    else:
                        loadf(
                            src_idx=step + self.stages,
                            dst_idx=stage,
                            tma_bar_ptr=kbar + stage,
                        )
                        loadv(
                            src_idx=step + self.stages,
                            dst_idx=stage,
                            tma_bar_ptr=kbar + stage,
                        )
            if cutlass.const_expr(kind == 0):
                # The Triton algorithm rounds this reduction before the front
                # GEMM. Preserve that BF16 value instead of a live FP32 tile.
                gate_saved.store(accg.load().to(BFloat16))
        coords = thrg.partition_C(cute.make_identity_tensor((self.pm, self.bn)))
        result = (accf.load() + gate_saved.load().to(Float32)).to(BFloat16)
        out = cute.make_fragment_like(accg, BFloat16)
        out.store(result)
        if cutlass.const_expr(self.n % 8 == 0):
            # Reuse the now-retired A staging storage for a coalesced TMA store.
            so = storage.sg.get_tensor(lo.outer, swizzle=lo.inner)
            cute.arch.barrier()
            store_op = sm90h.get_smem_store_op(LayoutEnum.ROW_MAJOR, BFloat16, Float32)
            copy_c = cute.make_tiled_copy_C(store_op, mg).get_slice(tidx)
            cute.copy(store_op, copy_c.retile(out), copy_c.partition_D(so))
            cute.arch.fence_view_async_shared()
            cute.arch.barrier()
            if warp == 0:
                dst = cute.local_tile(ty, (self.pm, self.bn), (mi, ni))
                shared, global_ = cpasync.tma_partition(
                    ay,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(so, 0, cute.rank(so)),
                    cute.group_modes(dst, 0, cute.rank(dst)),
                )
                cute.copy(ay, shared, global_)
                with cute.arch.elect_one():
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.barrier()
        else:
            for i in cutlass.range(cute.size(out), unroll_full=True):
                row = row_origin + coords[i][0]
                col = ni * self.bn + coords[i][1]
                if row < y.shape[0] and col < self.n:
                    y[row, col] = out[i]

    @cute.jit
    def __call__(self, g, f, w, v, y, stream: cuda.CUstream):
        def layout(rows, major):
            orient = LayoutEnum.ROW_MAJOR if major else LayoutEnum.COL_MAJOR
            atom = warpgroup.make_smem_layout_atom(
                sm90h.get_smem_layout_atom(orient, BFloat16, self.bk if major else rows),
                BFloat16,
            )
            return cute.tile_to_shape(
                atom,
                (rows, self.bk, self.stages),
                order=(0, 1, 2) if major else (1, 0, 2),
            )

        lg, lf = layout(self.pm, True), layout(self.pm, False)
        lw, lv = layout(self.bn, True), layout(self.bn, False)
        ag, tg = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            g,
            cute.slice_(lg, (None, None, 0)),
            (self.pm, self.bk),
        )
        af, tf = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            f,
            cute.slice_(lf, (None, None, 0)),
            (self.pm, self.bk),
        )
        aw, tw = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            w,
            cute.slice_(lw, (None, None, 0)),
            (self.bn, self.bk),
        )
        av, tv = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            v,
            cute.slice_(lv, (None, None, 0)),
            (self.bn, self.bk),
        )
        mg = sm90h.make_trivial_tiled_mma(
            BFloat16,
            BFloat16,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            (self.mma_mgroups, self.mma_ngroups, 1),
            (64, self.bn // self.mma_ngroups),
        )
        mf = sm90h.make_trivial_tiled_mma(
            BFloat16,
            BFloat16,
            warpgroup.OperandMajorMode.MN,
            warpgroup.OperandMajorMode.MN,
            Float32,
            (self.mma_mgroups, self.mma_ngroups, 1),
            (64, self.bn // self.mma_ngroups),
        )
        output_atom = warpgroup.make_smem_layout_atom(
            sm90h.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, BFloat16, self.bn),
            BFloat16,
        )
        lo = cute.tile_to_shape(output_atom, (self.pm, self.bn), order=(0, 1))
        if cutlass.const_expr(self.n % 8 == 0):
            ay, ty = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), y, lo, (self.pm, self.bn)
            )
        else:
            ay, ty = ag, y
        sg_type = cute.struct.Align[
            cute.struct.MemRange[BFloat16, max(cute.cosize(lg), cute.cosize(lo))], 1024
        ]
        sw_type = cute.struct.Align[
            cute.struct.MemRange[BFloat16, cute.cosize(lw)], 1024
        ]

        @cute.struct
        class Storage:
            bar: cute.struct.MemRange[cutlass.Int64, 2 * self.stages]
            sg: sg_type
            sw: sw_type

        self.shared_storage = Storage
        self.kernel(
            ag, tg, af, tf, aw, tw, av, tv, y, ay, ty, lo, lg, lf, lw, lv, mg, mf
        ).launch(
            grid=[
                cute.ceil_div(y.shape[0], self.logical_m)
                * cute.ceil_div(self.n, self.bn),
                1,
                1,
            ],
            block=[32 * self.warps, 1, 1],
            stream=stream,
        )


_COMPILE_CACHE = {}


def feasibility(config, smem_limit=232448):
    """Reject unsupported physical launches explicitly; never change tile axes."""
    bm, bn, bk = (config[k] for k in ("BLOCK_M1", "BLOCK_N", "BLOCK_K"))
    if bm < 64:
        return "WGMMA has a minimum 64-row instruction tile"
    if config["num_warps"] not in (4, 8):
        return "implementation requires one or two four-warp groups"
    sizes = [
        max(bm * bk * config["num_stages"] * 2, bm * bn * 2),
        bn * bk * config["num_stages"] * 2,
    ]
    usage = 1024 + sum((s + 1023) // 1024 * 1024 for s in sizes)
    if usage > smem_limit:
        return f"shared memory {usage} exceeds device limit {smem_limit}"
    return None


def input_dual_bwd_sm90_impl(g, f, w, v, length, config=None):
    """Same API and matrix views as Triton input_dual_bwd; explicit config first."""
    if any(t.ndim != 2 for t in (g, f, w, v)):
        raise ValueError("dual dgrad expects matrices")
    m, kg = g.shape
    kp = f.shape[1]
    n = w.shape[1]
    if min(m, kg, kp, n) <= 0:
        raise ValueError("dual dgrad expects positive dimensions")
    if f.shape[0] != m or w.shape[0] != kg or v.shape != (kp, n):
        raise ValueError("dual dgrad shape mismatch")
    if any(
        t.dtype != torch.bfloat16 or t.device != g.device or not t.is_cuda
        for t in (g, f, w, v)
    ):
        raise ValueError("requires BF16 CUDA operands on same device")
    if g.stride(1) != 1 or f.stride(0) != 1 or w.stride(0) != 1 or v.stride(1) != 1:
        raise ValueError("requires production row/column-major operand views")
    if any(
        t.data_ptr() % 16 or any(s % 8 for s in t.stride() if s != 1)
        for t in (g, f, w, v)
    ):
        raise ValueError("TMA requires 16-byte alignment and aligned non-unit strides")
    if torch.cuda.get_device_capability(g.device) != (9, 0):
        raise ValueError("requires Hopper SM90")
    y = g.new_empty((m, n))
    tensors = (g, f, w.t(), v.t(), y)
    limit = torch.cuda.get_device_properties(g.device).shared_memory_per_block_optin

    def launch(c):
        reason = feasibility(c, limit)
        if reason is not None:
            raise ValueError(reason)
        args = [from_dlpack(t.detach(), assumed_align=16) for t in tensors]
        args.append(cuda.CUstream(torch.cuda.current_stream(g.device).cuda_stream))
        key = (
            str(g.device),
            tuple((tuple(t.shape), t.stride()) for t in tensors),
            tuple(sorted(c.items())),
        )
        if key not in _COMPILE_CACHE:
            _COMPILE_CACHE[key] = cute.compile(DualBackwardSm90(n, kg, kp, **c), *args)
        _COMPILE_CACHE[key](*args)

    if config is None:
        from miniworld_engine.autotune.trimul_sm90_config import resolve

        config = resolve(
            "trimul_input_dual_bwd_sm90_cute",
            tensors[:-1],
            extra=(length, limit),
            feasibility=lambda c: feasibility(c, limit),
            run=launch,
        )
    launch(config)
    return y


def _fake(g, f, w, v, length):
    return g.new_empty((g.shape[0], w.shape[1]))


@opaque(fake=_fake, name="trimul_round2_prefetch2_sm90")
def input_dual_bwd_sm90(
    g: torch.Tensor, f: torch.Tensor, w: torch.Tensor, v: torch.Tensor, length: int
) -> torch.Tensor:
    return input_dual_bwd_sm90_impl(g, f, w, v, length)
