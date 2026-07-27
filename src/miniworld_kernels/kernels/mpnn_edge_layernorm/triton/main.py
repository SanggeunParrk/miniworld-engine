"""Native-forward LayerNorm with a compressed edge activation save.

PyTorch autocast promotes ProteinMPNN's encoder edge LayerNorm to FP32 and its
native autograd node retains one FP32 ``[B, L, K, 128]`` input per layer. This
boundary calls the same native forward, but stores that input in BF16. Backward
reads the compressed tensor directly through the repository's existing Triton
atomic LayerNorm kernel; it never materializes a restored FP32 copy.
"""

from __future__ import annotations

import torch

from miniworld_kernels.kernels.layernorm.compile_native import (
    _bwd_atomic_impl,
    _bwd_partial_impl,
)


_WIDTH = 128


@torch.library.custom_op(
    "miniworld_kernels::mpnn_edge_layernorm_memory_bwd_v1",
    mutates_args=(),
)
def _backward_op(
    grad_output: torch.Tensor,
    saved_input: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Forward dispatch normally keeps deterministic steps on native PyTorch.
    # Retain the contract even if the global flag is enabled between forward
    # and backward by switching this already-created autograd node to the
    # non-atomic partial reduction.
    backward_impl = (
        _bwd_partial_impl
        if torch.are_deterministic_algorithms_enabled()
        else _bwd_atomic_impl
    )
    grad_input, grad_weight, grad_bias = backward_impl(
        grad_output,
        saved_input.reshape(-1, _WIDTH),
        weight,
        mean.reshape(-1),
        rstd.reshape(-1),
    )
    # The generic kernel returns a two-dimensional dX. Keep this custom op's
    # real output contract identical to its fake registration.
    return grad_input.view_as(saved_input), grad_weight, grad_bias


@_backward_op.register_fake
def _(
    grad_output: torch.Tensor,
    saved_input: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del mean, rstd
    # dX follows grad_output's dtype in _bwd_atomic_impl, not saved_input's BF16
    # dtype. This distinction is required for AOTAutograd/fullgraph correctness.
    return (
        grad_output.new_empty(saved_input.shape),
        weight.new_empty(weight.shape),
        weight.new_empty(weight.shape),
    )


class _MemoryLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        values: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        # Calling native_layer_norm under the active autocast context is exactly
        # the operation used by nn.LayerNorm/F.layer_norm in this model.
        output, mean, rstd = torch.native_layer_norm(
            values,
            (_WIDTH,),
            weight,
            bias,
            eps,
        )
        ctx.save_for_backward(values.to(torch.bfloat16), weight, mean, rstd)
        ctx.input_dtype = values.dtype
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved_input, weight, mean, rstd = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = _backward_op(
            grad_output.contiguous(),
            saved_input,
            weight,
            mean,
            rstd,
        )
        return (
            grad_input.to(ctx.input_dtype),
            grad_weight,
            grad_bias,
            None,
        )


def edge_layer_norm_memory(
    values: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply the native-forward, compressed-save LayerNorm boundary."""
    return _MemoryLayerNorm.apply(values, weight, bias, eps)


__all__ = ["edge_layer_norm_memory"]
