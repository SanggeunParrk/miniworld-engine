"""SM90 TMA LayerNorm backward with Triton's persistent feature-tile algorithm.

Every (persistent row program, feature tile) owns disjoint dx/dw/db slices.
Split feature tiles gather c1/c2 across all features before reloading their own
slice, exactly as the Triton kernel. Covering tiles reuse the loaded values.
Stages are real TMA buffers, not a compiler hint. No matrix multiply is present.
"""

from __future__ import annotations

import operator
import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90h
from cutlass import BFloat16, Float32, Int32
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum
from cuda.bindings import driver as cuda
from quack import copy_utils

OP = "layernorm_bwd_split_sm90_cute"
_COMPILED = {}


def config_rejection(config, *, n, itemsize, m_major, smem_limit):
    bm, bk = config["BLOCK_M1"], config["BLOCK_K"]
    nw, ns = config["num_warps"], config["num_stages"]
    if nw not in (1, 2, 4, 8, 16, 32) or ns < 1:
        return "invalid thread block or stage count"
    if min(bm, bk) < 1 or bk % 32:
        return "feature tile must contain whole warps"
    if m_major and bm * itemsize < 16:
        return "TMA contiguous row box must span at least 16 bytes"
    sizes = [
        bm * bk * ns * itemsize,
        bm * bk * ns * itemsize,
        bm * bk * itemsize,
        2 * nw * bk * 4,
    ]
    usage = 1024 + sum((s + 1023) // 1024 * 1024 for s in sizes)
    if usage > smem_limit:
        return f"shared memory {usage} exceeds {smem_limit}"
    return None


def input_rejection(x, dy, w, mean, rstd, dx_strides):
    if x.ndim != 2 or dy.shape != x.shape or min(x.shape) < 1:
        return "requires nonempty matching matrices"
    if x.dtype not in (torch.bfloat16, torch.float32) or dy.dtype != x.dtype:
        return "requires BF16 or FP32 activations"
    if not x.is_cuda or any(t.device != x.device for t in (dy, w, mean, rstd)):
        return "requires one CUDA device"
    if torch.cuda.get_device_capability(x.device) != (9, 0):
        return "requires Hopper SM90"
    if (
        w.shape != (x.shape[1],)
        or mean.shape != (x.shape[0],)
        or rstd.shape != mean.shape
    ):
        return "parameter/statistic shape mismatch"
    if any(t.dtype != torch.float32 or t.stride() != (1,) for t in (w, mean, rstd)):
        return "requires contiguous FP32 parameters and statistics"
    for strides in (x.stride(), dy.stride(), tuple(dx_strides)):
        if strides != x.stride() or 1 not in strides:
            return "requires matching row-major or column-major strides"
        if any(s != 1 and s * x.element_size() % 16 for s in strides):
            return "TMA noncontiguous strides must be 16-byte aligned"
    if any(t.data_ptr() % 16 for t in (x, dy)):
        return "TMA base pointers must be 16-byte aligned"
    if x.stride(0) == 1 and x.stride(1) < x.shape[0]:
        return "overlapping column-major layout"
    if x.stride(1) == 1 and x.stride(0) < x.shape[1]:
        return "overlapping row-major layout"
    return None


class LayerNormBackwardTma:
    def __init__(
        self, n, grid, dtype, m_major, BLOCK_M1, BLOCK_K, num_warps, num_stages
    ):
        self.n, self.grid, self.dtype = n, grid, dtype
        self.bm, self.bk, self.warps, self.stages = (
            BLOCK_M1,
            BLOCK_K,
            num_warps,
            num_stages,
        )
        self.m_major = m_major
        self.kt = (n + BLOCK_K - 1) // BLOCK_K
        self.passes = 1 if self.kt == 1 else self.kt + 1
        self.rows_per_warp = (BLOCK_M1 + num_warps - 1) // num_warps

    @cute.jit
    def __call__(self, x, dy, w, mean, rstd, out, dw, db, stream: cuda.CUstream):
        major = LayoutEnum.COL_MAJOR if self.m_major else LayoutEnum.ROW_MAJOR
        contiguous = self.bm if self.m_major else self.bk
        order = (1, 0, 2) if self.m_major else (0, 1, 2)
        if cutlass.const_expr(self.bm < 8):
            # WGMMA's layout atoms have a minimum 8-row shape. LayerNorm has
            # no WGMMA; valid smaller TMA boxes use an unswizzled exact layout.
            stride = (1, self.bm) if self.m_major else (self.bk, 1)
            lo = cute.make_composed_layout(
                cute.make_swizzle(0, 4, 3),
                0,
                cute.make_layout((self.bm, self.bk), stride=stride),
            )
            lx = cute.make_composed_layout(
                cute.make_swizzle(0, 4, 3),
                0,
                cute.make_layout(
                    (self.bm, self.bk, self.stages), stride=(*stride, self.bm * self.bk)
                ),
            )
        else:
            atom = warpgroup.make_smem_layout_atom(
                sm90h.get_smem_layout_atom(major, self.dtype, contiguous), self.dtype
            )
            lx = cute.tile_to_shape(atom, (self.bm, self.bk, self.stages), order=order)
            lo = cute.tile_to_shape(atom, (self.bm, self.bk), order=order[:2])
        ax, tx = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            x,
            cute.slice_(lx, (None, None, 0)),
            (self.bm, self.bk),
        )
        ay, ty = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            dy,
            cute.slice_(lx, (None, None, 0)),
            (self.bm, self.bk),
        )
        ao, to = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), out, lo, (self.bm, self.bk)
        )
        st = cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(lx)], 1024]
        ot = cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(lo)], 1024]
        dt = cute.struct.MemRange[Float32, 2 * self.warps * self.bk]

        @cute.struct
        class Storage:
            barriers: cute.struct.MemRange[cutlass.Int64, self.stages]
            x: st
            y: st
            o: ot
            partial: dt

        self.storage = Storage
        self.kernel(ax, tx, ay, ty, ao, to, w, mean, rstd, dw, db, lx, lo).launch(
            grid=[self.grid, self.kt, 1], block=[self.warps * 32, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, ax, tx, ay, ty, ao, to, w, mean, rstd, dw, db, lx, lo):
        tid, _, _ = cute.arch.thread_idx()
        pid, pk, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tid % 32
        storage = cutlass.utils.SmemAllocator().allocate(self.storage)
        sx = storage.x.get_tensor(lx.outer, swizzle=lx.inner)
        sy = storage.y.get_tensor(lx.outer, swizzle=lx.inner)
        so = storage.o.get_tensor(lo.outer, swizzle=lo.inner)
        partial = storage.partial.get_tensor(
            cute.make_layout(
                (2, self.warps, self.bk), stride=(self.warps * self.bk, self.bk, 1)
            )
        )
        bar = storage.barriers.data_ptr()
        if warp == 0:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(ax)
                cpasync.prefetch_descriptor(ay)
                for j in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(bar + j, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        gx = cute.local_tile(tx, (self.bm, self.bk), (None, None))
        gy = cute.local_tile(ty, (self.bm, self.bk), (None, None))
        gx = cute.group_modes(gx, 2, 4)
        gy = cute.group_modes(gy, 2, 4)
        loadx, _, _ = copy_utils.tma_get_copy_fn(ax, 0, cute.make_layout(1), gx, sx)
        loady, _, _ = copy_utils.tma_get_copy_fn(ay, 0, cute.make_layout(1), gy, sy)
        values = cute.make_rmem_tensor((self.bk // 32,), Float32)
        other = cute.make_rmem_tensor((self.bk // 32,), Float32)
        gamma = cute.make_rmem_tensor((self.bk // 32,), Float32)
        aw = cute.make_rmem_tensor((self.bk // 32,), Float32)
        ab = cute.make_rmem_tensor((self.bk // 32,), Float32)
        c1s = cute.make_rmem_tensor((self.rows_per_warp,), Float32)
        c2s = cute.make_rmem_tensor((self.rows_per_warp,), Float32)
        aw.fill(0)
        ab.fill(0)
        gamma.fill(0)
        c1s.fill(0)
        c2s.fill(0)
        for c in cutlass.range_constexpr(self.bk // 32):
            col = pk * self.bk + lane + c * 32
            if col < self.n:
                gamma[c] = w[col]
        tiles = cute.ceil_div(tx.shape[0], self.bm)
        tasks = cute.ceil_div(tiles - pid, self.grid) * self.passes
        if warp == 0:
            for j in cutlass.range_constexpr(self.stages):
                if j < tasks:
                    k = j % self.passes
                    if k == self.kt:
                        k = pk
                    rowtile = pid + (j // self.passes) * self.grid
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            bar + j, self.bm * self.bk * 2 * (self.dtype.width // 8)
                        )
                    loadx(src_idx=(rowtile, k), dst_idx=j, tma_bar_ptr=bar + j)
                    loady(src_idx=(rowtile, k), dst_idx=j, tma_bar_ptr=bar + j)
        for it in cutlass.range(0, tasks):
            stage = it % self.stages
            k = it % self.passes
            emit = k == self.passes - 1
            if k == self.kt:
                k = pk
            tile = pid + (it // self.passes) * self.grid
            cute.arch.mbarrier_wait(bar + stage, (it // self.stages) & 1)
            if warp == 0:
                with cute.arch.elect_one():
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.barrier()
            for ri in cutlass.range_constexpr(self.rows_per_warp):
                rl = warp + ri * self.warps
                row = tile * self.bm + rl
                if rl < self.bm and row < tx.shape[0]:
                    mu = mean[row]
                    rs = rstd[row]
                    s1 = Float32(0)
                    s2 = Float32(0)
                    for c in cutlass.range_constexpr(self.bk // 32):
                        col = k * self.bk + lane + c * 32
                        values[c] = Float32(0)
                        other[c] = Float32(0)
                        if col < self.n:
                            values[c] = (
                                sx[rl, lane + c * 32, stage].to(Float32) - mu
                            ) * rs
                            other[c] = sy[rl, lane + c * 32, stage].to(Float32)
                            weight = gamma[c]
                            if cutlass.const_expr(self.kt > 1):
                                weight = w[col]
                            wd = other[c] * weight
                            s1 = s1 + wd * values[c]
                            s2 = s2 + wd
                    if cutlass.const_expr(self.kt == 1):
                        c1s[ri] = cute.arch.warp_reduction(s1, operator.add) / self.n
                        c2s[ri] = cute.arch.warp_reduction(s2, operator.add) / self.n
                    else:
                        if not emit:
                            c1s[ri] = c1s[ri] + cute.arch.warp_reduction(
                                s1, operator.add
                            )
                            c2s[ri] = c2s[ri] + cute.arch.warp_reduction(
                                s2, operator.add
                            )
                    if emit:
                        if cutlass.const_expr(self.kt > 1):
                            c1s[ri] = c1s[ri] / self.n
                            c2s[ri] = c2s[ri] / self.n
                        for c in cutlass.range_constexpr(self.bk // 32):
                            dx = (
                                other[c] * gamma[c] - (values[c] * c1s[ri] + c2s[ri])
                            ) * rs
                            so[rl, lane + c * 32] = dx.to(self.dtype)
                            aw[c] = aw[c] + other[c] * values[c]
                            ab[c] = ab[c] + other[c]
                        c1s[ri] = Float32(0)
                        c2s[ri] = Float32(0)
            cute.arch.barrier()
            if warp == 0:
                ahead = it + self.stages
                if ahead < tasks:
                    kk = ahead % self.passes
                    if kk == self.kt:
                        kk = pk
                    rowtile = pid + (ahead // self.passes) * self.grid
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            bar + stage, self.bm * self.bk * 2 * (self.dtype.width // 8)
                        )
                    loadx(src_idx=(rowtile, kk), dst_idx=stage, tma_bar_ptr=bar + stage)
                    loady(src_idx=(rowtile, kk), dst_idx=stage, tma_bar_ptr=bar + stage)
            if emit:
                cute.arch.fence_view_async_shared()
                cute.arch.barrier()
                if warp == 0:
                    go = cute.local_tile(to, (self.bm, self.bk), (tile, pk))
                    ss, gg = cpasync.tma_partition(
                        ao,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(so, 0, cute.rank(so)),
                        cute.group_modes(go, 0, cute.rank(go)),
                    )
                    cute.copy(ao, ss, gg)
                    with cute.arch.elect_one():
                        cute.arch.cp_async_bulk_commit_group()
        for c in cutlass.range_constexpr(self.bk // 32):
            partial[0, warp, lane + c * 32] = aw[c]
            partial[1, warp, lane + c * 32] = ab[c]
        cute.arch.barrier()
        for c in cutlass.range_constexpr(cute.ceil_div(self.bk, self.warps * 32)):
            local = tid + c * self.warps * 32
            col = pk * self.bk + local
            if local < self.bk and col < self.n:
                sw = Float32(0)
                sb = Float32(0)
                for wi in cutlass.range_constexpr(self.warps):
                    sw = sw + partial[0, wi, local]
                    sb = sb + partial[1, wi, local]
                dw[pid, col] = sw
                db[pid, col] = sb
        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.cp_async_bulk_wait_group(0, read=True)
        cute.arch.barrier()


def prepare(x, dy, w, mean, rstd, out, dw, db, config):
    reason = input_rejection(x, dy, w, mean, rstd, out.stride())
    if reason:
        raise ValueError(reason)
    limit = torch.cuda.get_device_properties(x.device).shared_memory_per_block_optin
    reason = config_rejection(
        config,
        n=x.shape[1],
        itemsize=x.element_size(),
        m_major=x.stride(0) == 1,
        smem_limit=limit,
    )
    if reason:
        raise ValueError(reason)
    tensors = (x, dy, w, mean, rstd, out, dw, db)
    args = [
        from_dlpack(
            t.detach(), assumed_align=16 if t.data_ptr() % 16 == 0 else t.element_size()
        )
        for t in tensors
    ]
    key = (
        str(x.device),
        tuple((tuple(t.shape), t.stride(), t.dtype) for t in tensors),
        tuple(sorted(config.items())),
    )
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    if key not in _COMPILED:
        dtype = BFloat16 if x.dtype == torch.bfloat16 else Float32
        _COMPILED[key] = cute.compile(
            LayerNormBackwardTma(
                x.shape[1], dw.shape[0], dtype, x.stride(0) == 1, **config
            ),
            *args,
            stream,
        )
    compiled = _COMPILED[key]
    return lambda: compiled(
        *args, cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    )


def backward_impl(dy, x, w, mean, rstd, dx_strides, config=None):
    reason = input_rejection(x, dy, w, mean, rstd, dx_strides)
    if reason:
        raise ValueError(reason)
    from miniworld_engine.kernels.layernorm.triton.persistent import _persistent_grid
    from miniworld_engine.autotune.trimul_sm90_config import resolve

    g = _persistent_grid(x.device)
    dx = torch.empty_strided(x.shape, tuple(dx_strides), dtype=x.dtype, device=x.device)
    dw = torch.empty((g, x.shape[1]), device=x.device, dtype=torch.float32)
    db = torch.empty_like(dw)
    limit = torch.cuda.get_device_properties(x.device).shared_memory_per_block_optin

    def run(c):
        prepare(x, dy, w, mean, rstd, dx, dw, db, c)()

    if config is None:
        config = resolve(
            OP,
            (x, dy, w, mean, rstd),
            extra=(limit,),
            feasibility=lambda c: config_rejection(
                c,
                n=x.shape[1],
                itemsize=x.element_size(),
                m_major=x.stride(0) == 1,
                smem_limit=limit,
            ),
            run=run,
        )
    run(config)
    return dx, dw.sum(0), db.sum(0)
