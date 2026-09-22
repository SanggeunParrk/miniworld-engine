"""Hopper squeeze GEMM with a separate residual C operand and no staging copy."""
from __future__ import annotations

import torch
import cutlass
import cutlass.cute as cute
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._quack_compat import jit_cache, is_compile_only
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters, torch2cute_dtype_map
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.gemm_sm90 import GemmSm90
from quack.gemm_tvm_ffi_utils import (
    compile_gemm_kernel, perm3d_single, make_fake_gemm_tensors,
    make_fake_scheduler_args, make_fake_varlen_args, make_scheduler_args, make_varlen_args,
)


class RoundedResidualSm90(GemmDefaultEpiMixin, GemmSm90):
    """WGMMA/TMA GEMM with the same rounding boundary as bf16 matmul + residual."""

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        rounded = tRS_rD.load().to(self.d_dtype).to(cutlass.Float32)
        tRS_rD.store(rounded + tRS_rC.load().to(cutlass.Float32))
        return None


@jit_cache
def _compile_rounded(dtype, tile, cluster, pingpong, dynamic, device):
    a, b, d, c, m, n, k, batch = make_fake_gemm_tensors(
        dtype, dtype, dtype, dtype, "k", "k", "n", "n",
    )
    return compile_gemm_kernel(
        RoundedResidualSm90, dtype, tile, cluster, pingpong, True, False, dynamic,
        device, a, b, d, c, RoundedResidualSm90.EpilogueArguments(),
        make_fake_scheduler_args(dynamic, False, batch),
        make_fake_varlen_args(False, False, False, None),
    )


def _launch(expand, weight, residual, out, config):
    from miniworld_engine.autotune.cute_config import validate_hopper_config
    validate_hopper_config(config)
    device = get_device_capacity(expand.device)
    if device[0] != 9:
        raise ValueError("Transition CuTe squeeze/residual requires Hopper")
    fn = _compile_rounded(
        torch2cute_dtype_map[expand.dtype], (config.tile_m, config.tile_n),
        (config.cluster_m, config.cluster_n, 1), config.pingpong,
        config.is_dynamic_persistent, device,
    )
    if is_compile_only():
        return
    semaphore = (torch.zeros(1, device=expand.device, dtype=torch.int32)
                 if config.is_dynamic_persistent else None)
    scheduler = make_scheduler_args(
        get_max_active_clusters(config.cluster_m * config.cluster_n, device_capacity=device),
        config.max_swizzle_size, semaphore,
    )
    operands = [perm3d_single(t.unsqueeze(0)) for t in (expand, weight, out, residual)]
    fn(*operands, RoundedResidualSm90.EpilogueArguments(add_to_output=None, rounding_mode=None), scheduler,
       make_varlen_args(None, None, None), None)


def _squeeze_residual_fake(expand, weight, residual):
    """Allocate outputs with the same shape, dtype and strides as squeeze_residual."""
    return residual.new_empty(residual.shape)


@opaque(fake=_squeeze_residual_fake, name='transition_squeeze_residual_sm90_cute')
def squeeze_residual(expand: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Execute squeeze residual behind an opaque compiler boundary."""
    from quack.gemm_config import GemmConfig
    from miniworld_engine.autotune.cute_config import plain_sm90_candidates, resolve_config
    from miniworld_engine.autotune.native import tensor_key
    if (expand.ndim != 2 or weight.ndim != 2 or residual.ndim != 2
            or expand.shape[1] != weight.shape[1]
            or residual.shape != (expand.shape[0], weight.shape[0])):
        raise ValueError("expected H[M,K], W[N,K], residual[M,N]")
    if any(t.dtype != torch.bfloat16 or t.device != expand.device or not t.is_contiguous()
           for t in (expand, weight, residual)):
        raise ValueError("CuTe squeeze/residual requires contiguous BF16 tensors on one device")
    out = _squeeze_residual_fake(expand, weight, residual)

    def launch(c):
        _launch(expand, weight, residual, out, c)
    default = GemmConfig(tile_m=128, tile_n=256, cluster_m=1, cluster_n=1, pingpong=False, is_dynamic_persistent=False, device_capacity=9)
    candidates = plain_sm90_candidates()
    if default not in candidates:
        candidates = [default] + candidates
    config = resolve_config('transition_squeeze_residual_sm90_cute', candidates, default=default, dtype=str(expand.dtype), bucket=tensor_key(expand, weight, residual), device_index=expand.device.index, run=launch)
    launch(config)
    return out
