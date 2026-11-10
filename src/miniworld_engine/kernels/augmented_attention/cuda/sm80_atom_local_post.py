"""A100 gate/residual CUDA postpass with original BF rounding."""

import functools
from pathlib import Path
import torch
from torch.autograd.function import once_differentiable
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def ext():
    ensure_cuda_home()
    return load_extension(
        name="af3_local_outer_gate_residual_sm80",
        sources=[str(Path(__file__).with_suffix(".cu"))],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "-gencode=arch=compute_80,code=sm_80",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=True,
    )


def _fwd_fake(single, gate, projection, save, cfg):
    """(current like ``single``, sigmoid like ``single`` when ``save`` else empty [0])."""
    return (
        torch.empty_like(single),
        torch.empty_like(single) if save else single.new_empty((0,)),
    )


@opaque(fake=_fwd_fake, name="augmented_attention_local_outer_gate_residual_sm80")
def _fwd(
    single: torch.Tensor,
    gate: torch.Tensor,
    projection: torch.Tensor,
    save: bool,
    cfg: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AF3 local atom attention post stage: ``single + sigmoid(gate) * projection`` -> (current, sigmoid); the sigmoid is returned only when ``save`` (for the backward)."""
    current, sigmoid = ext().gate_residual(single, gate, projection, save, cfg)
    return (current, sigmoid)


class _GateResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, single, gate, projection, save, cfg):
        current, sigmoid = _fwd(single, gate, projection, save, cfg)
        if save:
            ctx.save_for_backward(sigmoid, projection)
        return current

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        from miniworld_engine.kernels.integrations.local_dit_sm80 import _gate_bwd

        dgate, dprojection = _gate_bwd(dy, *ctx.saved_tensors)
        return (dy, dgate, dprojection, None, None)
