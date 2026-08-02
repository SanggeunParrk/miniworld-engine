"""sm100 (B200) transition forward — Round 1 (bring-up).

Stage 1 of the H100->B200 transition b2b port. The Hopper hand-CUDA b2b
(`transition/cuda/transition_b2b_kernel.cu`) fed the SwiGLU intermediate straight
from registers into a register-source (RS) WGMMA squeeze. Blackwell tcgen05 has no
register-operand MMA (A must come from smem/TMEM), so the register-chained RS trick
cannot be reproduced; the fused b2b becomes a warp-specialized persistent kernel with
a smem/TMEM round-trip for `h`. That full fusion is Round 2+.

Round 1 splits the op and reuses the WORKING trimul sm100 gated GEMM verbatim:
  * expand+SwiGLU: `h = silu(xn@wa^T) * (xn@wb^T)`  via `SwiGLUExpandKernel`, a one-line
    epilogue-math change on `GatedPersistentGemmKernel` (which already computes
    `sigmoid(A@Bg^T)*(A@Bp^T)`; SwiGLU just multiplies by the gate once more).
  * squeeze: `out = h @ ws^T` via cuBLAS (torch.mm) for now.
LN is applied upstream (xn precomputed); matches the H100 "stats as a separate pass"
lesson (fused stats loses at 1 CTA/SM).

Operand mapping (swish_gate(a,b)=silu(a)*b, a=expand_a=gate, b=expand_b=up):
    gate accumulator = A@Bg^T  with Bg = wa   -> gets silu
    proj accumulator = A@Bp^T  with Bp = wb   -> the "up" factor
    h = gate * sigmoid(gate) * proj = silu(a)*b
"""

from __future__ import annotations
import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch
from quack.cute_dsl_utils import get_max_active_clusters
from cutlass import Float32, const_expr
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.utils.gemm.sm100 import (
    epilogue_tmem_copy_and_partition,
    epilogue_smem_copy_and_partition,
    transform_partitioned_tensor_layout,
)

from miniworld_engine.kernels.tm1.cute.sm100_gate_gemm_collective import (
    GatedPersistentGemmKernel,
)


class SwiGLUExpandKernel(GatedPersistentGemmKernel):
    """Dual-B gated GEMM whose epilogue emits SwiGLU `silu(gate)*proj` instead of the
    inference `sigmoid(gate)*proj`. Only `_gated_epilogue` changes vs the parent — the
    dual-B mainloop, dual-TMEM accumulator, pipelines and TMA store are inherited."""

    @cute.jit
    def _gated_epilogue(
        self, epi_tidx, warp_idx, tma_atom_c, proj_base, gate_base, sC, tCgC_base,
        epi_tile, num_tiles_executed, mma_tile_coord_mnl, acc_consumer_state,
        acc_pipeline, c_pipeline,
    ):
        tCgC = transform_partitioned_tensor_layout(tCgC_base)
        tAccP = transform_partitioned_tensor_layout(proj_base)
        tAccG = transform_partitioned_tensor_layout(gate_base)

        tiled_copy_t2r, tTR_tAccP, tTR_rAccP = epilogue_tmem_copy_and_partition(
            self, epi_tidx, tAccP, tCgC, epi_tile, self.use_2cta_instrs
        )
        _, tTR_tAccG, tTR_rAccG = epilogue_tmem_copy_and_partition(
            self, epi_tidx, tAccG, tCgC, epi_tile, self.use_2cta_instrs
        )

        tTR_rC = cute.make_rmem_tensor(tTR_rAccP.shape, self.c_dtype)
        tiled_copy_r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(
            self, tiled_copy_t2r, tTR_rC, epi_tidx, sC
        )

        tCgC_epi = cute.flat_divide(tCgC, epi_tile)
        bSG_sC, bSG_gC_part = cpasync.tma_partition(
            tma_atom_c, 0, cute.make_layout(1),
            cute.group_modes(sC, 0, 2), cute.group_modes(tCgC_epi, 0, 2),
        )

        epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.epilog_sync_bar_id,
            num_threads=32 * len(self.epilogue_warp_id),
        )

        bSG_gC = bSG_gC_part[(None, None, None, *mma_tile_coord_mnl)]
        tTR_tAccP = tTR_tAccP[(None, None, None, None, None, acc_consumer_state.index)]
        tTR_tAccG = tTR_tAccG[(None, None, None, None, None, acc_consumer_state.index)]

        acc_pipeline.consumer_wait(acc_consumer_state)

        tTR_tAccP = cute.group_modes(tTR_tAccP, 3, cute.rank(tTR_tAccP))
        tTR_tAccG = cute.group_modes(tTR_tAccG, 3, cute.rank(tTR_tAccG))
        bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

        subtile_cnt = cute.size(tTR_tAccP.shape, mode=[3])
        num_prev_subtiles = num_tiles_executed * subtile_cnt

        # DEPTH-staged software-pipelined epilogue (inherited design): prefetch subtile
        # k+depth-1's TMEM->reg loads to hide t2r long-scoreboard latency at 1 CTA/SM.
        depth = const_expr(self.epi_depth)
        rAccP = [tTR_rAccP] + [cute.make_fragment_like(tTR_rAccP) for _ in range(depth - 1)]
        rAccG = [tTR_rAccG] + [cute.make_fragment_like(tTR_rAccG) for _ in range(depth - 1)]

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

            p = tiled_copy_r2s.retile(rAccP[cur]).load()   # proj = A@Bp^T (= b, the "up")
            g = tiled_copy_r2s.retile(rAccG[cur]).load()   # gate = A@Bg^T (= a, gets silu)
            # SwiGLU: silu(gate)*proj = gate*sigmoid(gate)*proj. rsqrt path: 1/d==rsqrt(d)^2.
            if const_expr(self.sig_mode == "rsqrt"):
                d = 1.0 + cmath.exp2(g * (-1.4426950408889634))
                r = cmath.rsqrt(d)
                ov = g * p * r * r
            else:
                ov = g * p * (1.0 / (1.0 + cmath.exp2(g * (-1.4426950408889634))))
            tRS_rC.store(ov.to(self.c_dtype))

            c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
            cute.arch.fence_proxy("async.shared", space="cta")
            epilog_sync_barrier.arrive_and_wait()

            if warp_idx == self.epilogue_warp_id[0]:
                cute.copy(tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, subtile_idx)])
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()
            epilog_sync_barrier.arrive_and_wait()

        epilog_sync_barrier.arrive_and_wait()
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_consumer_state)
        acc_consumer_state.advance()
        return acc_consumer_state


_CACHE = {}


def swiglu_expand_gemm(xn, wb, wa):
    """h = silu(xn @ wa^T) * (xn @ wb^T).  xn:(M,K), wa/wb:(ND,K) bf16 -> h:(M,ND) row-major.
    (wa is the gate/silu weight, wb the up weight — matching swish_gate(a,b)=silu(a)*b.)"""
    M, K = xn.shape
    N, K2 = wb.shape

    def _mark(t3, leading_dim):
        return from_dlpack(t3, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=leading_dim)

    mA = _mark(xn.detach().unsqueeze(0), 2)   # (L, M, K)
    mBp = _mark(wb.detach().unsqueeze(0), 2)  # (L, N, K)  proj = up
    mBg = _mark(wa.detach().unsqueeze(0), 2)  # (L, N, K)  gate = silu operand

    h = torch.empty(M, N, device=xn.device, dtype=torch.bfloat16)
    mC = _mark(h.unsqueeze(0), 2)             # (1, M, N), N contiguous -> row-major [M,N]

    mac = get_max_active_clusters(1)  # memoized: HardwareInfo().get_max_active_clusters
    # JIT-recompiles a probe kernel on every call (~35ms eager); the memoized helper is
    # device-constant so the compiled op cache (_CACHE) is the only per-shape cost.
    strm = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    key = (M, N, K)
    if key not in _CACHE:
        op = SwiGLUExpandKernel(
            acc_dtype=Float32, use_2cta_instrs=False,
            mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 1), use_tma_store=True,
        )
        op.K = int(K)
        op.sig_mode = os.environ.get("MW_SIG", "rsqrt")
        op.epi_depth = int(os.environ.get("MW_EPI_DEPTH", "3"))
        _CACHE[key] = cute.compile(op, mA, mBp, mBg, mC, mac, strm, options="--enable-tvm-ffi")
    _CACHE[key](mA, mBp, mBg, mC)
    return h


def transition_b2b_sm100(xn, wa, wb, ws):
    """Round 1 split forward: expand+SwiGLU (fused sm100 kernel) then squeeze (cuBLAS).
    xn:(M,K) pre-normalized bf16; wa/wb:(ND,K); ws:(D,ND). Returns out:(M,D) bf16."""
    h = swiglu_expand_gemm(xn, wb, wa)        # (M, ND)
    return torch.mm(h, ws.t())                # (M, ND) @ (ND, D) -> (M, D)


def transition_b2b_sm100_ln(x, ln_weight, ln_bias, wa, wb, ws, eps=1e-5):
    """Module-facing sm100 transition forward: optimized LayerNorm (repo's tuned ``layernorm_kernel``,
    HBM-bound) -> normalized xn -> sm100 expand+SwiGLU -> cuBLAS squeeze. x:(...,K) bf16,
    ln_weight/ln_bias:(K,); wa/wb:(ND,K); ws:(D,ND). Returns out:(...,D) bf16.

    LN is done by the tuned standalone kernel (not a torch affine): xn is a cheap extra HBM pass
    (~2 K-wide passes) rather than several fp32 elementwise passes. Fusing the affine into the b2b
    kernel A-load (H100-style) is a later round; this already realizes the kernel's advantage."""
    from miniworld_engine.kernels.layernorm.interface import layernorm_kernel
    K = x.shape[-1]
    x2 = x.reshape(-1, K)
    xn = layernorm_kernel(x2, ln_weight, ln_bias, eps)
    out = transition_b2b_sm100(xn, wa, wb, ws)
    return out.reshape(*x.shape[:-1], out.shape[-1])
