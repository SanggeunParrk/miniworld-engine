"""Forward-only full fusion for the ProteinMPNN hidden-message reduction.

Unlike the training implementation, this path does not materialize the
``[..., 48, 128]`` projected activation needed by backward.  It is therefore
selected only while autograd is disabled.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .._policy import _inference_int32_elements_supported

_HIDDEN = 128
_NEIGHBORS = 48
_SMALL_GROUP_LIMIT = 2048


@triton.jit
def _gelu(x):
    return 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))


@triton.jit
def _message_inference_kernel(
    preactivation_ptr,
    weight_ptr,
    bias_ptr,
    mask_ptr,
    reduced_ptr,
    groups,
    neighbor_scale,
    HIDDEN: tl.constexpr,
    NEIGHBORS: tl.constexpr,
    GROUPS_PER_CTA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COALESCED_WEIGHT_LOAD: tl.constexpr,
):
    # All offsets stay int32 on this inference-only fast path.  The Python
    # dispatch checks the tensor size before selecting it.
    group_block = tl.program_id(0)
    local_rows = tl.arange(0, BLOCK_M)
    local_group = local_rows // NEIGHBORS
    neighbor = local_rows - local_group * NEIGHBORS
    group = group_block * GROUPS_PER_CTA + local_group
    row = group * NEIGHBORS + neighbor
    output_columns = tl.arange(0, BLOCK_N)
    row_valid = (local_group < GROUPS_PER_CTA) & (group < groups)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for hidden_start in range(0, HIDDEN, BLOCK_K):
        hidden_columns = hidden_start + tl.arange(0, BLOCK_K)
        preactivation = tl.load(
            preactivation_ptr + row[:, None] * HIDDEN + hidden_columns[None, :],
            mask=row_valid[:, None],
            other=0.0,
        ).to(tl.float32)
        activated = _gelu(preactivation).to(tl.bfloat16)
        if COALESCED_WEIGHT_LOAD:
            # W is nn.Linear's contiguous [output, input] layout.  Loading
            # [output, input] and transposing the tile gives coalesced reads.
            weight_output_input = tl.load(
                weight_ptr + output_columns[:, None] * HIDDEN + hidden_columns[None, :]
            ).to(tl.bfloat16)
            weight = tl.trans(weight_output_input)
        else:
            weight = tl.load(
                weight_ptr + output_columns[None, :] * HIDDEN + hidden_columns[:, None]
            ).to(tl.bfloat16)
        accumulator += tl.dot(activated, weight)

    bias = tl.load(bias_ptr + output_columns)
    projected = (accumulator + bias.to(tl.bfloat16)).to(tl.bfloat16)
    hidden = _gelu(projected.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    edge_weight = tl.load(mask_ptr + row, mask=row_valid, other=0.0).to(tl.float32)
    contribution = hidden * edge_weight[:, None]

    # A CTA may own two logical residue groups.  Keep their neighbor reductions
    # separate while sharing the GEMM tile and weight traffic.
    for segment in range(GROUPS_PER_CTA):
        selected = tl.where(local_group[:, None] == segment, contribution, 0.0)
        reduced = tl.sum(selected, axis=0) / neighbor_scale
        output_group = group_block * GROUPS_PER_CTA + segment
        tl.store(
            reduced_ptr + output_group * HIDDEN + output_columns,
            reduced,
            mask=output_group < groups,
        )


def _int32_offsets_supported(preactivation: torch.Tensor) -> bool:
    """Return whether valid and masked padded offsets fit signed int32."""
    return _inference_int32_elements_supported(preactivation.numel())


@torch.library.triton_op(
    "miniworld_kernels::mpnn_message_inference",
    mutates_args={},
)
def _inference_op(
    preactivation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> torch.Tensor:
    groups = preactivation.numel() // (_NEIGHBORS * _HIDDEN)
    reduced = torch.empty(
        groups,
        _HIDDEN,
        device=preactivation.device,
        dtype=torch.float32,
    )

    if groups <= _SMALL_GROUP_LIMIT:
        # A5000 B=1,L=2048 winner: one residue per CTA, coalesced W load.
        groups_per_cta = 1
        block_m, block_k = 64, 16
        warps, stages = 2, 3
        coalesced_weight_load = True
    else:
        # A5000 B=4,L=2048 winner.  Two residues share one CTA, halving the
        # output grid and amortizing weight traffic.
        groups_per_cta = 2
        block_m, block_k = 128, 32
        warps, stages = 4, 3
        coalesced_weight_load = False

    grid = lambda meta: (triton.cdiv(groups, groups_per_cta),)  # noqa: E731
    torch.library.wrap_triton(_message_inference_kernel)[grid](
        preactivation,
        weight,
        bias,
        edge_mask,
        reduced,
        groups,
        neighbor_scale,
        HIDDEN=_HIDDEN,
        NEIGHBORS=_NEIGHBORS,
        GROUPS_PER_CTA=groups_per_cta,
        BLOCK_M=block_m,
        BLOCK_N=_HIDDEN,
        BLOCK_K=block_k,
        COALESCED_WEIGHT_LOAD=coalesced_weight_load,
        num_warps=warps,
        num_stages=stages,
    )
    return reduced.reshape(*preactivation.shape[:-2], _HIDDEN)


def triton_message_hidden_reduce_inference(
    preactivation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> torch.Tensor:
    """Run the forward-only no-save full fusion."""
    return _inference_op(
        preactivation,
        weight,
        bias,
        edge_mask,
        neighbor_scale,
    )


__all__ = [
    "triton_message_hidden_reduce_inference",
]
