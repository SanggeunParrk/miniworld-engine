"""SM90 F2 with the Triton bidirectional front's saved-value/rounding contract.

Reuses Quack's explicit TMA load/store and WGMMA mainloop, not the old
MaskedGatedSm90 arithmetic. Config names retain their Triton meaning; unsupported
physical configurations raise instead of silently changing their tile/warp count.
The one packed output allocation is split into disjoint left/right views outside
the opaque boundary. No intermediate GEMM output, cast, mask or copy is launched.
"""
from __future__ import annotations

from functools import partial

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warpgroup
from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters, torch2cute_dtype_map
from quack.gemm_act import GemmActMixin, GemmGatedMixin
from quack.gemm_sm90 import GemmSm90
from quack.gemm_tvm_ffi_utils import (
    compile_gemm_kernel, get_major, perm3d_single, make_fake_gemm_tensors,
    make_fake_scheduler_args, make_fake_varlen_args, make_scheduler_args, make_varlen_args,
)
from miniworld_engine.kernels.trimul_inproj.cute.parity_f567 import _reciprocal_full
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._quack_compat import jit_cache, is_compile_only


@cute.jit
def _front_glu(gate, projection):
    # Match Triton div.full, including sigmoid values below the normal FP32 range.
    return projection * _reciprocal_full(1.0 + cute.math.exp(-gate, fastmath=True))


def front_config_rejection(config, *, m=None, k=None, h2=None):
    """Return an explicit hardware/layout reason, or None for a legal candidate.

    The shared CSV is the complete *requested* domain. This implementation has
    either one four-warp shared producer/consumer group or a separate producer
    and consumer group (8 total).
    M128 is implemented as two M64 WGMMA atoms per consumer warpgroup, rather
    than Quack's default two consumer groups (which would use12 total warps).
    """
    if config["BLOCK_M1"] not in (64, 128):
        return "This WGMMA mainloop supports M64/M128; M32 needs an alternate masked-atom implementation"
    if config["num_warps"] not in (4, 8):
        return "WGMMA needs a four-warp group; implementations use4 or8 physical CTA warps"
    if config["BLOCK_K_D"] not in (16, 32, 64):
        return "BF16 WGMMA K tile must match the declared16/32/64 domain"
    if config["BLOCK_K_H2"] not in (16, 32, 64):
        return "interleaved physical N must be32/64/128"
    if config["num_stages"] < 1:
        return "TMA pipeline needs at least one real stage"
    if m is not None and m % 8:
        return "M-major output/preactivation TMA requires M stride multiple of8 BF16 elements"
    if k is not None and k % 8:
        return "TMA input row stride requires K multiple of8 BF16 elements"
    if h2 is not None and h2 % 2:
        return "TMA interleaved weight row stride requires4*H2 multiple of8"
    if config["num_warps"] == 4:
        from .front_single_warpgroup import four_warp_shared_bytes
        used = four_warp_shared_bytes(config)
        limit = cutlass.utils.get_smem_capacity_in_bytes("sm_90")
        if used > limit:
            return f"Requested stage ring needs {used} shared bytes, exceeds SM90 limit {limit}"
    return None


def _front_n_tile_group(weight_columns, channel_tile):
    """Keep every gate/projection N tile for one M tile next to each other.

    Quack's static scheduler accepts arbitrary positive group sizes through
    FastDivmod (including non-powers of two) and clamps to the fast tile axis.
    Grouping the full N extent also makes its heuristic choose AlongN for tall
    and short matrices. This is derived traversal, not an additional tuning axis.
    """
    physical_n_tile = 2 * channel_tile  # interleaved gate and projection
    return max(1, (weight_columns + physical_n_tile - 1) // physical_n_tile)


class ParityFrontSm90(GemmGatedMixin, GemmSm90):
    """TMA/WGMMA projection with FP32 sigmoid and BF16-before-mask rounding."""

    def __init__(self, *args, requested_stages, requested_warps, **kwargs):
        # Keep one consumer warpgroup for both legal M tiles. Quack's generic
        # partitioning supports multiple M64 MMA atoms per warpgroup; its usual
        # constructor instead chooses two consumer groups when tile_M=128.
        # N<=128 here, so at most128 FP32 accumulators/thread fit the same232
        # register consumer budget. All layouts/stages use the restored real M.
        tile = args[2]
        init_args = (*args[:2], (64, *tile[1:]), *args[3:])
        super().__init__(*init_args, **kwargs)
        self.cta_tile_shape_mnk = tile
        if self.threads_per_cta != requested_warps * 32:
            raise ValueError("Requested num_warps differs from physical CTA size")
        self.requested_stages = requested_stages
        # SM90 has 65536 registers per SM. M64's accumulator and epilogue fit
        # the per-thread budget for two CTAs; M128 keeps Quack's larger budget.
        # This depends on the physical tile, never the model sequence length.
        self.num_regs_mma = (
            65536 // (2 * self.threads_per_cta)
            if tile[0] == 64 else self.num_regs_mma
        )
        self.resident_ctas = 1

    def _compute_stages(self, *args, **kwargs):
        max_ab, epi, epi_c = super()._compute_stages(*args, **kwargs)
        if self.requested_stages > max_ab:
            raise ValueError("Requested TMA stages exceed shared memory capacity")
        # Quack sizes its epilogue at maximum AB depth. Keep that epilogue and
        # subtract the removed AB slots from the full SM budget: the remainder
        # is a conservative storage bound including alignment and unused slack.
        bm, bn, bk = self.cta_tile_shape_mnk
        ab_bytes = (bm + bn) * bk * 2  # This entry point accepts BF16 only.
        used_bound = self.smem_capacity - (max_ab - self.requested_stages) * ab_bytes
        self.resident_ctas = max(1, min(
            self.smem_capacity // used_bound,
            65536 // (self.num_regs_mma * self.threads_per_cta),
            2048 // self.threads_per_cta,
        ))
        return self.requested_stages, epi, epi_c

    def get_scheduler_class(self, varlen_m=False):
        parent = super().get_scheduler_class(varlen_m)
        resident = self.resident_ctas

        class ResourceScheduler(parent):
            @staticmethod
            def get_grid_shape(params, max_active_clusters, *, loc=None, ip=None):
                return parent.get_grid_shape(
                    params, max_active_clusters * resident, loc=loc, ip=ip
                )

        return ResourceScheduler

    @cute.jit
    def mma(self, ab_pipeline, ab_read_state, mma_fn, acc, acc_slow,
            k_tile_cnt, warp_group_idx):
        if cutlass.const_expr(self.requested_stages == 1):
            # Quack's normal loop keeps one WGMMA group in flight and tries to
            # acquire tile k+1 before releasing tile k. That deadlocks with one
            # shared-memory stage. Complete and release each tile before reuse.
            zero_init = cutlass.Boolean(True)
            for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                ab_pipeline.consumer_wait(ab_read_state)
                mma_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=zero_init)
                zero_init = cutlass.Boolean(False)
                warpgroup.wait_group(0)
                ab_pipeline.consumer_release(ab_read_state)
                ab_read_state.advance()
        else:
            ab_read_state = GemmSm90.mma(
                self, ab_pipeline, ab_read_state, mma_fn, acc, acc_slow,
                k_tile_cnt, warp_group_idx,
            )
        return ab_read_state

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        params = GemmActMixin.epi_to_underlying_arguments(self, args, loc=loc, ip=ip)
        self.cta_tile_shape_aux_out_mn = (
            self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1] // 2,
        )
        return params

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        # tRS_rD remains the raw FP32 gate/projection accumulator: the normal D
        # store writes its BF16 preactivations for the existing Triton backward.
        layout = cute.recast_layout(2, 1, tRS_rD.layout)
        out = cute.make_rmem_tensor(layout.shape, self.acc_dtype)
        for i in cutlass.range(cute.size(out), unroll_full=True):
            value = params.act_fn(tRS_rD[2 * i], tRS_rD[2 * i + 1])
            if cutlass.const_expr(params.mColVecBroadcast is not None):
                value = value.to(self.aux_out_dtype).to(cutlass.Float32)
                value = value * epi_loop_tensors["mColVecBroadcast"][2 * i].to(cutlass.Float32)
            out[i] = value
        return out


@jit_cache
def _compile_front(dtype, a_major, b_major, save_preact, has_mask, bm, bh, bk, warps, stages, device):
    a, b, d, c, m, n, k, batch = make_fake_gemm_tensors(
        dtype, dtype, dtype if save_preact else None, None,
        a_major, b_major, "m" if save_preact else None, None,
    )
    out = make_fake_tensor(dtype, (m, cute.sym_int(), batch), leading_dim=0, divisibility=8)
    mask = (make_fake_tensor(dtype, (batch, m), leading_dim=1, divisibility=8)
            if has_mask else None)
    epi = ParityFrontSm90.EpilogueArguments(out, _front_glu, mColVecBroadcast=mask)
    cls = partial(ParityFrontSm90, requested_stages=stages, requested_warps=warps)
    return compile_gemm_kernel(
        cls, dtype, (bm, 2 * bh, bk), (1, 1, 1), False, True, False, False,
        device, a, b, d, c, epi, make_fake_scheduler_args(False, False, batch),
        make_fake_varlen_args(False, False, False, None),
    )


def launch_front(a, w, packed, preact, pair_mask, config):
    """Launch into caller-owned (2H2,M)/(4H2,M) buffers; used by tuning too."""
    if a.ndim != 2 or w.ndim != 2 or a.shape[1] != w.shape[0] or w.shape[1] % 4:
        raise ValueError("Front expects (M,K) input and (K,4H2) interleaved weights")
    if not a.is_contiguous() or not w.is_contiguous():
        raise ValueError("Front expects contiguous normalized input and interleaved weights")
    if a.dtype != w.dtype or packed.shape != (w.shape[1] // 2, a.shape[0]):
        raise ValueError("Front input/weight dtype or packed output shape mismatch")
    if not packed.is_contiguous() or packed.dtype != a.dtype:
        raise ValueError("Front packed output must be contiguous and match input dtype")
    if preact is not None and (preact.shape != (w.shape[1], a.shape[0])
                              or preact.dtype != a.dtype or not preact.is_contiguous()):
        raise ValueError("Front saved preactivations must be contiguous (4H2,M) in input dtype")
    if pair_mask is not None and (pair_mask.numel() != a.shape[0]
                                 or pair_mask.dtype != a.dtype or not pair_mask.is_contiguous()):
        raise ValueError("Front mask must be contiguous M-vector in input dtype")
    reason = front_config_rejection(config, m=a.shape[0], k=a.shape[1], h2=w.shape[1] // 4)
    if reason:
        raise ValueError(reason)
    device = get_device_capacity(a.device)
    if device[0] != 9 or a.dtype != torch.bfloat16:
        raise ValueError("Parity front currently requires SM90 BF16")
    if config["num_warps"] == 4:
        from .front_single_warpgroup import launch_four_warp
        return launch_four_warp(a, w, packed, preact, pair_mask, config)
    ap, bp = perm3d_single(a.unsqueeze(0)), perm3d_single(w.T.unsqueeze(0))
    fn = _compile_front(
        torch2cute_dtype_map[a.dtype], get_major(ap, "m", "k"), get_major(bp, "n", "k"),
        preact is not None, pair_mask is not None,
        config["BLOCK_M1"], config["BLOCK_K_H2"], config["BLOCK_K_D"],
        config["num_warps"], config["num_stages"], device,
    )
    if is_compile_only():
        return
    epi = ParityFrontSm90.EpilogueArguments(
        perm3d_single(packed.T.unsqueeze(0)), None,
        mColVecBroadcast=None if pair_mask is None else pair_mask.reshape(1, -1), rounding_mode=None,
    )
    # Group=1 sweeps all M tiles before advancing N, evicting the normalized
    # input between channel chunks. Keep all N chunks adjacent for input reuse,
    # like Triton's inner channel loop, without changing GEMM/fusion boundaries.
    n_group = _front_n_tile_group(w.shape[1], config["BLOCK_K_H2"])
    sched = make_scheduler_args(get_max_active_clusters(1, device_capacity=device), n_group, None)
    fn(ap, bp, perm3d_single(preact.T.unsqueeze(0)) if preact is not None else None,
       None, epi, sched, make_varlen_args(None, None, None), None)


def _front_fake(a, w, pair_mask, save_preact, bm, bh, bk, warps, stages):
    return (a.new_empty((w.shape[1] // 2, a.shape[0])),
            a.new_empty((w.shape[1], a.shape[0])) if save_preact else a.new_empty((0, 0)))


@opaque(fake=_front_fake, name="trimul_front_parity_sm90")
def front_sm90(a: torch.Tensor, w: torch.Tensor, pair_mask: torch.Tensor | None,
               save_preact: bool, bm: int, bh: int, bk: int, warps: int, stages: int
               ) -> tuple[torch.Tensor, torch.Tensor]:
    packed, preact = _front_fake(a, w, pair_mask, save_preact, bm, bh, bk, warps, stages)
    config = dict(BLOCK_M1=bm, BLOCK_K_H2=bh, BLOCK_K_D=bk,
                  num_warps=warps, num_stages=stages)
    if bm == 0:
        from miniworld_engine.autotune.trimul_sm90_config import resolve
        config = resolve(
            "trimul_inproj_gemm_gate_mmajor_sm90_cute", (a, w, pair_mask), extra=(save_preact,),
            feasibility=lambda c: front_config_rejection(
                c, m=a.shape[0], k=a.shape[1], h2=w.shape[1] // 4),
            run=lambda c: launch_front(a, w, packed, preact if save_preact else None, pair_mask, c),
        )
    launch_front(a, w, packed, preact if save_preact else None, pair_mask, config)
    return packed, preact


def bidir_front_sm90(x_n, WL, WLg, WR, WRg, *, config=None, save_preact=True, pair_mask=None):
    """Same logical front interface and weight interleave as bidir_front_triton."""
    b, l, l2, k = x_n.shape
    if b != 1 or l != l2:
        raise ValueError("Bidirectional front requires one square pair tensor")
    h2, m = WL.shape[1], l * l
    left_w = torch.stack([WLg, WL], dim=2).reshape(k, 2 * h2)
    right_w = torch.stack([WRg, WR], dim=2).reshape(k, 2 * h2)
    w = torch.cat([left_w, right_w], dim=1).contiguous()
    mask = None if pair_mask is None else pair_mask.to(x_n.dtype).reshape(m).contiguous()
    if config is None:
        config = dict(BLOCK_M1=0, BLOCK_K_H2=0, BLOCK_K_D=0, num_warps=0, num_stages=0)
    packed, preact = front_sm90(
        x_n.reshape(m, k), w, mask, save_preact, config["BLOCK_M1"],
        config["BLOCK_K_H2"], config["BLOCK_K_D"], config["num_warps"], config["num_stages"],
    )
    return (packed[:h2].view(1, h2, l, l), packed[h2:].view(1, h2, l, l),
            preact if save_preact else None)
