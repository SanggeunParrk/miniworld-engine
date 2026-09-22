"""Hopper gated projection with pair masking in the GEMM output epilogue.

Preactivations remain unmasked for backward. Only the GLU output is masked;
the normalized input used by the separate output gate is never modified.
"""
from __future__ import annotations

import torch
import cutlass
import cutlass.cute as cute
from quack.activation import gate_fn_map
from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters, torch2cute_dtype_map
from quack.gemm_act import GemmActMixin, GemmGatedMixin
from quack.gemm_sm90 import GemmSm90
from quack.gemm_tvm_ffi_utils import (
    compile_gemm_kernel, get_major, perm3d_single, make_fake_gemm_tensors,
    make_fake_scheduler_args, make_fake_varlen_args, make_scheduler_args, make_varlen_args,
)
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._quack_compat import jit_cache, is_compile_only


class MaskedGatedSm90(GemmGatedMixin, GemmSm90):
    """Reuse Quack's tiled GEMM and gated stores, with a post-GLU row mask."""

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        # GemmActMixin accepts the M-major preactivation and output layouts.
        params = GemmActMixin.epi_to_underlying_arguments(self, args, loc=loc, ip=ip)
        self.cta_tile_shape_aux_out_mn = (
            self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1] // 2,
        )
        return params

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        # mColVecBroadcast is a multiplicative output mask in THIS class only.
        # No call to the default additive-bias epilogue: preactivation stays intact.
        scale = epi_loop_tensors["mColVecBroadcast"]
        layout = cute.recast_layout(2, 1, tRS_rD.layout)
        out = cute.make_rmem_tensor(layout.shape, self.acc_dtype)
        for i in cutlass.range(cute.size(out), unroll_full=True):
            out[i] = params.act_fn(tRS_rD[2 * i], tRS_rD[2 * i + 1]) * scale[2 * i]
        return out


@jit_cache
def _compile_masked_front(dtype, a_major, b_major, save_preact, tile, cluster, pp, dyn, device):
    a, b, d, c, m, n, k, batch = make_fake_gemm_tensors(
        dtype, dtype, dtype if save_preact else None, None,
        a_major, b_major, "m" if save_preact else None, None,
    )
    out = make_fake_tensor(dtype, (m, cute.sym_int(), batch), leading_dim=0, divisibility=8)
    mask = make_fake_tensor(cutlass.Float32, (batch, m), leading_dim=1, divisibility=4)
    epi = MaskedGatedSm90.EpilogueArguments(
        out, gate_fn_map["glu"], mColVecBroadcast=mask,
    )
    return compile_gemm_kernel(
        MaskedGatedSm90, dtype, tile, cluster, pp, True, False, dyn,
        device, a, b, d, c, epi, make_fake_scheduler_args(dyn, False, batch),
        make_fake_varlen_args(False, False, False, None),
    )


def _launch(a, b, out, preact, mask, config):
    from miniworld_engine.autotune.cute_config import validate_hopper_config
    validate_hopper_config(config)
    ap, bp = perm3d_single(a.unsqueeze(0)), perm3d_single(b.T.unsqueeze(0))
    device = get_device_capacity(a.device)
    assert device[0] == 9, "masked gated GEMM requires Hopper"
    fn = _compile_masked_front(
        torch2cute_dtype_map[a.dtype], get_major(ap, "m", "k"), get_major(bp, "n", "k"),
        preact is not None, (config.tile_m, config.tile_n),
        (config.cluster_m, config.cluster_n, 1), config.pingpong,
        config.is_dynamic_persistent, device,
    )
    if is_compile_only():
        return
    semaphore = (torch.zeros(1, dtype=torch.int32, device=a.device)
                 if config.is_dynamic_persistent else None)
    epi = MaskedGatedSm90.EpilogueArguments(
        perm3d_single(out.unsqueeze(0)), None, mColVecBroadcast=mask, rounding_mode=None,
    )
    sched = make_scheduler_args(
        get_max_active_clusters(config.cluster_m * config.cluster_n, device_capacity=device),
        config.max_swizzle_size, semaphore,
    )
    fn(ap, bp, perm3d_single(preact.unsqueeze(0)) if preact is not None else None, None, epi, sched,
       make_varlen_args(None, None, None), None)


def _masked_front_fake(a, b, pair_mask, save_preact=False):
    """Allocate outputs with the same shape, dtype and strides as masked_front."""
    (m, n) = (a.shape[0], b.shape[1])
    out = a.new_empty((n // 2, m)).T
    preact = a.new_empty((n, m)).T if save_preact else a.new_empty((0,))
    return (out, preact)


@opaque(fake=_masked_front_fake, name='trimul_inproj_masked_sm90_cute')
def masked_front(a: torch.Tensor, b: torch.Tensor, pair_mask: torch.Tensor, save_preact: bool=False) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute masked front behind an opaque compiler boundary."""
    from miniworld_engine.autotune.cute_config import resolve_config, gated_sm90_candidates
    from miniworld_engine.autotune.native import tensor_key
    (out, preact) = _masked_front_fake(a, b, pair_mask, save_preact)
    mask = pair_mask.reshape(1, a.shape[0]).to(torch.float32).contiguous()
    config = resolve_config('trimul_inproj_masked_sm90_cute', gated_sm90_candidates(), dtype=str(a.dtype), bucket=tensor_key(a, b, mask, extra=(save_preact,)), device_index=a.device.index, run=lambda c: _launch(a, b, out, preact if save_preact else None, mask, c))
    _launch(a, b, out, preact if save_preact else None, mask, config)
    return (out, preact)
