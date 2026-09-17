"""Hopper squeeze GEMM with a separate residual C operand and no staging copy."""
from __future__ import annotations

import torch
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels import _quack_compat  # register source identity before Quack import
from quack.gemm import gemm


def _squeeze_residual_fake(expand, weight, residual):
    """Allocate outputs with the same shape, dtype and strides as squeeze_residual."""
    return residual.new_empty(residual.shape)


@opaque(fake=_squeeze_residual_fake, name='transition_squeeze_residual_sm90_cute')
def squeeze_residual(expand: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Execute squeeze residual behind an opaque compiler boundary."""
    from quack.gemm_config import GemmConfig
    from miniworld_engine.autotune.cute_config import plain_sm90_candidates, resolve_config
    from miniworld_engine.autotune.native import tensor_key
    out = _squeeze_residual_fake(expand, weight, residual)

    def launch(c):
        semaphore = torch.zeros(1, device=expand.device, dtype=torch.int32) if c.is_dynamic_persistent else None
        gemm(expand.unsqueeze(0), weight.unsqueeze(0), out.unsqueeze(0), residual.unsqueeze(0), semaphore, c.tile_m, c.tile_n, c.cluster_m, c.cluster_n, pingpong=c.pingpong, is_dynamic_persistent=c.is_dynamic_persistent, max_swizzle_size=c.max_swizzle_size)
    default = GemmConfig(tile_m=128, tile_n=256, cluster_m=1, cluster_n=1, pingpong=False, is_dynamic_persistent=False, device_capacity=9)
    candidates = plain_sm90_candidates()
    if default not in candidates:
        candidates = [default] + candidates
    config = resolve_config('transition_squeeze_residual_sm90_cute', candidates, default=default, dtype=str(expand.dtype), bucket=tensor_key(expand, weight, residual), device_index=expand.device.index, run=launch)
    launch(config)
    return out
