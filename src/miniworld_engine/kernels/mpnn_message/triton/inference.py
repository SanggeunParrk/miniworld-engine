"""Forward-only full fusion for the ProteinMPNN hidden-message reduction.

Unlike the training implementation, this path does not materialize the
``[..., 48, 128]`` projected activation needed by backward.  It is therefore
selected only while autograd is disabled.

Registered as ``mpnn_message_fwd_gemm_gelu_reduce_triton``. Its two launch policies (one residue per CTA up to 2048 residue groups, two
above: the A5000 winners) are the default config set, and the early prune picks the one the group count asks for outside a build, so a
card with no tuned cache runs exactly what it ran; the grid set is what a build searches. Every config must hold its groups' rows in
one tile (``BLOCK_M >= GROUPS_PER_CTA * 48``): the prune drops the rest always, a build included.
"""

from __future__ import annotations

import torch

from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import both_key
from miniworld_engine.kernels._compile import opaque
import triton
import triton.language as tl

from miniworld_engine.kernels.mpnn_message.triton._policy import _inference_int32_elements_supported

_HIDDEN = 128
_NEIGHBORS = 48
_SMALL_GROUP_LIMIT = 2048


@triton.jit
def _gelu(x):
    return 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))


def _policy_groups(groups: int) -> int:
    """Residue groups per CTA of the fixed policy: 1 up to ``_SMALL_GROUP_LIMIT`` groups, 2 above."""
    return 1 if groups <= _SMALL_GROUP_LIMIT else 2


def _prune(configs, nargs, **meta):
    """Configs whose row tile holds their groups (always); outside a build, the policy's groups per CTA for this group count."""
    from miniworld_engine import settings

    valid = [c for c in configs if c.kwargs["BLOCK_M"] >= c.kwargs["GROUPS_PER_CTA"] * _NEIGHBORS]
    if settings.current().run_autotune:
        return valid
    groups = int(nargs["groups"] if "groups" in nargs else meta["groups"])
    keep = [c for c in valid if c.kwargs["GROUPS_PER_CTA"] == _policy_groups(groups)]
    return keep or valid


@triton.autotune(configs=configs_for("mpnn_message_fwd_gemm_gelu_reduce_triton"), key=["shape_key"],
                 prune_configs_by={"early_config_prune": _prune})
@triton.jit
def _message_inference_kernel(
    preactivation_ptr,
    weight_ptr,
    bias_ptr,
    mask_ptr,
    reduced_ptr,
    groups,
    neighbor_scale,
    shape_key,
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


def _inference_op_fake(
    preactivation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> torch.Tensor:
    """One contiguous FP32 vector per group; the neighbor dimension is reduced."""
    return preactivation.new_empty(*preactivation.shape[:-2], _HIDDEN, dtype=torch.float32)


@opaque(fake=_inference_op_fake, name="mpnn_message_inference")
def _inference_op(
    preactivation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> torch.Tensor:
    """Launch the fixed-policy BF16 message fusion and return FP32 group sums."""
    groups = preactivation.numel() // (_NEIGHBORS * _HIDDEN)
    reduced = torch.empty(
        groups,
        _HIDDEN,
        device=preactivation.device,
        dtype=torch.float32,
    )

    # The launch policies (default config set, mpnn_message_fwd_gemm_gelu_reduce_triton.csv): up to 2048 groups the A5000 B=1, L=2048
    # winner (one residue per CTA, coalesced W load: BLOCK_M 64, BLOCK_K 16, 2 warps); above, the B=4 one (two residues share a CTA,
    # halving the output grid and amortizing weight traffic: BLOCK_M 128, BLOCK_K 32, 4 warps); both 3 stages.
    grid = lambda meta: (triton.cdiv(groups, meta["GROUPS_PER_CTA"]),)  # noqa: E731
    _message_inference_kernel[grid](
        preactivation,
        weight,
        bias,
        edge_mask,
        reduced,
        groups,
        neighbor_scale,
        shape_key=both_key(groups * _NEIGHBORS, G=_policy_groups(groups)),
        HIDDEN=_HIDDEN,
        NEIGHBORS=_NEIGHBORS,
        BLOCK_N=_HIDDEN,
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
