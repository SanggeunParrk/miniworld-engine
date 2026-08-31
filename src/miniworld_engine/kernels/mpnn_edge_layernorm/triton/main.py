"""Native-forward LayerNorm with a compressed edge activation save.

PyTorch autocast promotes ProteinMPNN's encoder edge LayerNorm to FP32 and its
native autograd node retains one FP32 ``[B, L, K, 128]`` input per layer. This
boundary calls the same native forward, but stores that input in BF16. Backward
reads the compressed tensor directly through the repository's existing Triton
atomic LayerNorm kernel; it never materializes a restored FP32 copy.
"""

from __future__ import annotations

import torch

from miniworld_engine.autotune.shape_key import both_key
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.layernorm.compile_native import (
    _bwd_atomic_impl,
    _bwd_persistent_impl,
)


_WIDTH = 128


def _backward_op_fake(
    grad_output: torch.Tensor,
    saved_input: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dX, dW, dB). dX takes grad_output's shape from saved_input and its DTYPE from
    grad_output -- `_bwd_atomic_impl` writes it that way, and AOTAutograd needs the fake to agree
    or a fullgraph compile disagrees with the real op about a dtype."""
    del mean, rstd
    # dX follows grad_output's dtype in _bwd_atomic_impl, not saved_input's BF16
    # dtype. This distinction is required for AOTAutograd/fullgraph correctness.
    return (
        grad_output.new_empty(saved_input.shape),
        weight.new_empty(weight.shape),
        weight.new_empty(weight.shape),
    )


@opaque(fake=_backward_op_fake, name="mpnn_edge_layernorm_memory_bwd_v1")
def _backward_op(
    grad_output: torch.Tensor,
    saved_input: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LayerNorm backward from a BF16 copy of the input rather than the fp32 original.

    The memory backend keeps the native forward bit-for-bit and saves only that copy; this is
    what reads it back. Deterministic mode picks the non-atomic reduction below.
    """
    # Forward dispatch normally keeps deterministic steps on native PyTorch. Retain the contract
    # even if the global flag is enabled between forward and backward by switching this
    # already-created autograd node to a non-atomic reduction.
    #
    # `_bwd_persistent_impl`, not the `_bwd_partial_impl` this used to name: that path was deleted
    # with the rest of the `partial` backward, which won zero of the 49 measured (d, M) buckets and
    # cost 1.5-2.0x median. Persistent has the property that matters here for the same reason
    # partial did -- each program writes its own slot of a (programs, N) buffer and the reduction
    # is a plain sum over it, so the order is fixed and the result is reproducible. Atomics are
    # what determinism rules out, and neither of these uses one.
    backward_impl = (
        _bwd_persistent_impl
        if torch.are_deterministic_algorithms_enabled()
        else _bwd_atomic_impl
    )
    # The ROW BUCKET is computed here and passed down; the callee folds the width in, on both of
    # its paths. An edge activation is genuinely (rows, 128) -- there is no (B, L, D) behind it --
    # and `rows_of` refuses a flat shape on purpose, because a caller holding only (M, D) cannot
    # know whether its M is the whole launch or one slice. This caller does know: every edge row is
    # in this launch.
    rows = saved_input.numel() // _WIDTH
    grad_input, grad_weight, grad_bias = backward_impl(
        grad_output,
        saved_input.reshape(-1, _WIDTH),
        weight,
        mean.reshape(-1),
        rstd.reshape(-1),
        row_bucket=both_key(rows),
    )
    # The generic kernel returns a two-dimensional dX. Keep this custom op's
    # real output contract identical to its fake registration.
    return grad_input.view_as(saved_input), grad_weight, grad_bias


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
