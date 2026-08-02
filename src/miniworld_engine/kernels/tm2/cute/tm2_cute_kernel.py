"""From-scratch CuTeDSL SM90 kernel for tm2: fused dual-A gated GEMM.

Math (per row m, output column n):
    G[m, n] = sum_k  X1[m, k] * W1[n, k]      # K-major weights (nn.Linear-style)
    V[m, n] = sum_k  X2[m, k] * W2[n, k]
    O[m, n] = sigmoid(G[m, n]) * V[m, n]

Tensor layout (bf16, contiguous, K-dim last):
    X1, X2 : (M, K)
    W1, W2 : (N, K)     # nn.Linear-style; same as cuequiv's dual-x kernel
    O      : (M, N)     # bf16

Implementation notes
====================
* Single warpgroup per CTA (128 threads). No producer/consumer split — same
  threads issue the TMA loads (one elected lane) and run the WGMMAs. This
  removes the warp-specialised pipeline machinery; we just need one mbarrier
  to cover the four bulk loads.
* SMEM is double-staged in K (TILE_K = 64) so the SMEM layout matches a
  standard ``K_SW128`` swizzle atom that WGMMA's descriptor knows how to
  read.  Two stages are loaded by TMA in sequence into the same mbarrier;
  expected_tx covers the full byte count.
* WGMMAs accumulate ``G`` and ``V`` interleaved per K stage so the compiler
  can issue both A loads back-to-back before the next ``warpgroup.fence``.

Targets shapes used by tm2 in TriMul: K = N = D = 128 (B=1, L∈384..1024).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90h
import torch
from cutlass import BFloat16, Float32, Int32
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum
from quack import copy_utils as quack_copy


_NUM_THREADS = 128  # one warpgroup
_TILE_K = 64        # K-stage size — matches K_SW128 atom for bf16


class TM2DualKernel:
    """Single-warpgroup, K-staged dual-A fused gated GEMM (SM90).

    The whole kernel is parametrised by ``N`` and ``K`` (set in __init__),
    not pulled from the GMEM tensor shapes — that way every shape is a
    plain Python int at trace time and the canonical helpers never see a
    dynamic ``%`` operand.
    """

    def __init__(self, N: int, K: int, tile_m: int = 64):
        assert tile_m == 64, "currently only TILE_M=64 (single m64 atom) is supported"
        assert K % _TILE_K == 0, f"K={K} must be divisible by TILE_K={_TILE_K}"
        self.N = N
        self.K = K
        self.tile_m = tile_m
        self.tile_n = N
        self.tile_k = _TILE_K
        self.k_loop = K // _TILE_K
        self.shared_storage = None

    # ----------------------------------------------------------------- kernel

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
        tma_atom_O: cute.CopyAtom,
        tO: cute.Tensor,
        sX_layout: cute.ComposedLayout,
        sW_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        tx_bytes_total: Int32,
    ):
        TILE_M: cutlass.Constexpr[int] = self.tile_m
        TILE_N: cutlass.Constexpr[int] = self.tile_n
        TILE_K: cutlass.Constexpr[int] = self.tile_k
        K_LOOP: cutlass.Constexpr[int] = self.k_loop

        tidx, _, _ = cute.arch.thread_idx()
        m_block, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # ---- SMEM allocation -------------------------------------------
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sX1 = storage.sX1.get_tensor(sX_layout.outer, swizzle=sX_layout.inner)
        sX2 = storage.sX2.get_tensor(sX_layout.outer, swizzle=sX_layout.inner)
        sW1 = storage.sW1.get_tensor(sW_layout.outer, swizzle=sW_layout.inner)
        sW2 = storage.sW2.get_tensor(sW_layout.outer, swizzle=sW_layout.inner)
        sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)

        # ---- TMA descriptor prefetch + mbarrier init -------------------
        mbar_full_ptr = storage.mbar_full.data_ptr()
        if warp_idx == 0:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_X1)
                cpasync.prefetch_descriptor(tma_atom_X2)
                cpasync.prefetch_descriptor(tma_atom_W1)
                cpasync.prefetch_descriptor(tma_atom_W2)
                cpasync.prefetch_descriptor(tma_atom_O)
                cute.arch.mbarrier_init(mbar_full_ptr, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        # ---- Per-tile global tensors (K-axis tiled into K_LOOP stages) -
        gX1 = cute.local_tile(tX1, (TILE_M, TILE_K), (m_block, None))
        gX2 = cute.local_tile(tX2, (TILE_M, TILE_K), (m_block, None))
        gW1 = cute.local_tile(tW1, (TILE_N, TILE_K), (0, None))
        gW2 = cute.local_tile(tW2, (TILE_N, TILE_K), (0, None))
        gO = cute.local_tile(tO, (TILE_M, TILE_N), (m_block, 0))

        # ---- Build per-tensor TMA copy functions -----------------------
        load_X1, _, _ = quack_copy.tma_get_copy_fn(
            tma_atom_X1, 0, cute.make_layout(1), gX1, sX1,
        )
        load_X2, _, _ = quack_copy.tma_get_copy_fn(
            tma_atom_X2, 0, cute.make_layout(1), gX2, sX2,
        )
        load_W1, _, _ = quack_copy.tma_get_copy_fn(
            tma_atom_W1, 0, cute.make_layout(1), gW1, sW1,
        )
        load_W2, _, _ = quack_copy.tma_get_copy_fn(
            tma_atom_W2, 0, cute.make_layout(1), gW2, sW2,
        )

        # ---- Issue all four TMA loads × K_LOOP from one thread ---------
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_full_ptr, tx_bytes_total)
                for k in cutlass.range_constexpr(K_LOOP):
                    load_X1(src_idx=k, dst_idx=k, tma_bar_ptr=mbar_full_ptr)
                    load_X2(src_idx=k, dst_idx=k, tma_bar_ptr=mbar_full_ptr)
                    load_W1(src_idx=k, dst_idx=k, tma_bar_ptr=mbar_full_ptr)
                    load_W2(src_idx=k, dst_idx=k, tma_bar_ptr=mbar_full_ptr)

        # ---- Wait for loads --------------------------------------------
        # Initial parity is 0; first completion flips it. The cutlass-dsl
        # ``mbarrier_wait`` wrapper is a spin loop that blocks until
        # ``mbarrier_try_wait_parity(phase=arg)`` returns true. PTX docs:
        # try_wait.parity returns true when the phase whose parity ==
        # phaseParity has *completed* — pass phase=0 to wait for the first
        # completion.
        cute.arch.mbarrier_wait(mbar_full_ptr, Int32(0))

        # ---- WGMMAs ----------------------------------------------------
        # Make sure cp.async.bulk writes to SMEM are visible to wgmma:
        # the TMA proxy writes need a generic-proxy fence before WGMMA
        # reads.  ``fence_view_async_shared`` emits exactly this fence.
        cute.arch.fence_view_async_shared()

        thr_mma = tiled_mma.get_slice(tidx)
        acc_shape = thr_mma.partition_shape_C((TILE_M, TILE_N))
        acc_G = cute.make_fragment(acc_shape, Float32)
        acc_V = cute.make_fragment(acc_shape, Float32)

        # Pre-partition the staged SMEM tensors once.
        tCsX1 = thr_mma.make_fragment_A(thr_mma.partition_A(sX1))
        tCsX2 = thr_mma.make_fragment_A(thr_mma.partition_A(sX2))
        tCsW1 = thr_mma.make_fragment_B(thr_mma.partition_B(sW1))
        tCsW2 = thr_mma.make_fragment_B(thr_mma.partition_B(sW2))

        warpgroup.fence()
        mma_G = cute.make_mma_atom(tiled_mma.op)
        mma_V = cute.make_mma_atom(tiled_mma.op)
        mma_G.set(warpgroup.Field.ACCUMULATE, False)
        mma_V.set(warpgroup.Field.ACCUMULATE, False)

        # tCsX*/tCsW* have shape (CPY, MMA_M, MMA_K_per_stage, K_LOOP).
        # Outer loop is over K stages; inner over per-stage K atoms.
        per_stage_k = cute.size(tCsX1.shape[2])
        for s in cutlass.range_constexpr(K_LOOP):
            for ki in cutlass.range_constexpr(per_stage_k):
                cute.gemm(
                    mma_G, acc_G,
                    tCsX1[None, None, ki, s], tCsW1[None, None, ki, s], acc_G,
                )
                mma_G.set(warpgroup.Field.ACCUMULATE, True)
                cute.gemm(
                    mma_V, acc_V,
                    tCsX2[None, None, ki, s], tCsW2[None, None, ki, s], acc_V,
                )
                mma_V.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()
        warpgroup.wait_group(0)

        # ---- Epilogue: out = sigmoid(G) * V  (fp32 -> bf16) ------------
        out_frag = cute.make_fragment_like(acc_G, BFloat16)
        g_vals = acc_G.load()
        v_vals = acc_V.load()
        # σ(x) = 0.5 + 0.5·tanh(x/2) — cheap on H100, no division.
        sig_g = 0.5 + 0.5 * cute.math.tanh(0.5 * g_vals, fastmath=True)
        out_frag.store((sig_g * v_vals).to(BFloat16))

        # ---- Reg -> SMEM via SM90 STSM ---------------------------------
        smem_store_op = sm90h.get_smem_store_op(
            LayoutEnum.ROW_MAJOR, BFloat16, Float32,
        )
        tiled_copy_C = cute.make_tiled_copy_C(smem_store_op, tiled_mma)
        thr_copy_C = tiled_copy_C.get_slice(tidx)
        cute.copy(
            smem_store_op,
            thr_copy_C.retile(out_frag),
            thr_copy_C.partition_D(sO),
        )

        # ---- SMEM -> GMEM via TMA --------------------------------------
        cute.arch.fence_view_async_shared()
        cute.arch.barrier()

        if warp_idx == 0:
            with cute.arch.elect_one():
                sO_t, gO_t = cpasync.tma_partition(
                    tma_atom_O, 0, cute.make_layout(1),
                    cute.group_modes(sO, 0, cute.rank(sO)),
                    cute.group_modes(gO, 0, cute.rank(gO)),
                )
                cute.copy(tma_atom_O, sO_t, gO_t)
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)

    # ----------------------------------------------------------------- host

    @cute.jit
    def __call__(
        self,
        mX1: cute.Tensor,
        mX2: cute.Tensor,
        mW1: cute.Tensor,
        mW2: cute.Tensor,
        mO: cute.Tensor,
    ):
        TILE_M: cutlass.Constexpr[int] = self.tile_m
        TILE_N: cutlass.Constexpr[int] = self.tile_n
        TILE_K: cutlass.Constexpr[int] = self.tile_k
        K_LOOP: cutlass.Constexpr[int] = self.k_loop

        M = mX1.shape[0]
        m_blocks = M // TILE_M

        # ---- SMEM layouts (manual; all sizes are Python ints) ----------
        sX_atom = warpgroup.make_smem_layout_atom(
            sm90h.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, BFloat16, TILE_K),
            BFloat16,
        )
        sX_layout = cute.tile_to_shape(
            sX_atom, (TILE_M, TILE_K, K_LOOP), order=(0, 1, 2),
        )

        sW_atom = warpgroup.make_smem_layout_atom(
            sm90h.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, BFloat16, TILE_K),
            BFloat16,
        )
        sW_layout = cute.tile_to_shape(
            sW_atom, (TILE_N, TILE_K, K_LOOP), order=(0, 1, 2),
        )

        sO_atom = warpgroup.make_smem_layout_atom(
            sm90h.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, BFloat16, TILE_N),
            BFloat16,
        )
        sO_layout = cute.tile_to_shape(sO_atom, (TILE_M, TILE_N), order=(0, 1))

        # ---- Tiled MMA: m64nNk16, 1 warpgroup --------------------------
        tiled_mma = sm90h.make_trivial_tiled_mma(
            BFloat16, BFloat16,
            warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            Float32,
            (TILE_M // 64, 1, 1),
            (64, TILE_N),
        )

        # ---- TMA atoms (per-stage SMEM layout) -------------------------
        sX_stage = cute.slice_(sX_layout, (None, None, 0))
        sW_stage = cute.slice_(sW_layout, (None, None, 0))
        tma_atom_X1, tma_X1 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mX1, sX_stage, (TILE_M, TILE_K),
        )
        tma_atom_X2, tma_X2 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mX2, sX_stage, (TILE_M, TILE_K),
        )
        tma_atom_W1, tma_W1 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mW1, sW_stage, (TILE_N, TILE_K),
        )
        tma_atom_W2, tma_W2 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mW2, sW_stage, (TILE_N, TILE_K),
        )
        tma_atom_O, tma_O = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mO, sO_layout, (TILE_M, TILE_N),
        )

        # ---- tx_bytes (expected bytes for the single mbarrier) ---------
        tx_X = cute.size_in_bytes(BFloat16, sX_stage)
        tx_W = cute.size_in_bytes(BFloat16, sW_stage)
        tx_bytes_total = (2 * tx_X + 2 * tx_W) * K_LOOP

        # ---- SharedStorage --------------------------------------------
        sX_struct = cute.struct.Align[
            cute.struct.MemRange[BFloat16, cute.cosize(sX_layout)], 1024
        ]
        sW_struct = cute.struct.Align[
            cute.struct.MemRange[BFloat16, cute.cosize(sW_layout)], 1024
        ]
        sO_struct = cute.struct.Align[
            cute.struct.MemRange[BFloat16, cute.cosize(sO_layout)], 1024
        ]

        @cute.struct
        class SharedStorage:
            mbar_full: cute.struct.MemRange[cutlass.Int64, 1]
            sO: sO_struct
            sX1: sX_struct
            sX2: sX_struct
            sW1: sW_struct
            sW2: sW_struct

        self.shared_storage = SharedStorage

        # ---- Launch ----------------------------------------------------
        self.kernel(
            tma_atom_X1, tma_X1,
            tma_atom_X2, tma_X2,
            tma_atom_W1, tma_W1,
            tma_atom_W2, tma_W2,
            tma_atom_O, tma_O,
            sX_layout, sW_layout, sO_layout,
            tiled_mma,
            Int32(tx_bytes_total),
        ).launch(
            grid=[m_blocks, 1, 1],
            block=[_NUM_THREADS, 1, 1],
        )


# --------------------------------------------------------------- public API

_COMPILE_CACHE: dict = {}


def tm2_dual_from_scratch(
    x1: torch.Tensor,
    x2: torch.Tensor,
    Wg_nk: torch.Tensor,
    Wp_nk: torch.Tensor,
) -> torch.Tensor:
    """From-scratch CuTeDSL fused dual-A gated GEMM (tm2 forward).

    Inputs (bf16, contiguous):
      x1, x2   : (..., D)
      Wg_nk    : (N, K)   — nn.Linear-style; N output cols, K=D reduction
      Wp_nk    : (N, K)
    Output    : (..., N), sigmoid(x1·Wg.T) * (x2·Wp.T)
    """
    assert x1.dtype == torch.bfloat16
    assert x1.dtype == x2.dtype == Wg_nk.dtype == Wp_nk.dtype
    assert x1.shape == x2.shape and Wg_nk.shape == Wp_nk.shape
    assert x1.is_contiguous() and x2.is_contiguous()
    assert Wg_nk.is_contiguous() and Wp_nk.is_contiguous()

    orig_shape = x1.shape
    K = int(orig_shape[-1])
    M = int(x1.numel() // K)
    N, K2 = int(Wg_nk.shape[0]), int(Wg_nk.shape[1])
    assert K == K2

    tile_m = 64
    assert M % tile_m == 0, f"M={M} must be divisible by tile_m={tile_m}"

    x1_flat = x1.reshape(M, K)
    x2_flat = x2.reshape(M, K)
    out_flat = torch.empty(M, N, device=x1.device, dtype=x1.dtype)

    # Pin the contiguous-dim stride; matches quack's tensor setup pattern.
    mX1 = from_dlpack(x1_flat, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mX2 = from_dlpack(x2_flat, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mW1 = from_dlpack(Wg_nk, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mW2 = from_dlpack(Wp_nk, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mO = from_dlpack(out_flat, assumed_align=16).mark_layout_dynamic(leading_dim=1)

    key = (M, N, K, x1.dtype, tile_m)
    if key not in _COMPILE_CACHE:
        kernel = TM2DualKernel(N=N, K=K, tile_m=tile_m)
        _COMPILE_CACHE[key] = cute.compile(kernel, mX1, mX2, mW1, mW2, mO)
    _COMPILE_CACHE[key](mX1, mX2, mW1, mW2, mO)

    return out_flat.view(*orig_shape[:-1], N)
