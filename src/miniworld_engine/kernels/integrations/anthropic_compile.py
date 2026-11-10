"""Compile boundary for unchanged Anthropic A100 native attention dispatch.

The upstream CUDA launcher performs mask byte reinterpretation and pybind calls;
keeping that launcher opaque avoids Inductor's bool-to-byte bitcast failure.
The provider, selection policy and all upstream kernels remain unchanged.
"""

import torch

from miniworld_engine.kernels._compile import opaque


def _native_fake(q, k, v, bias, mask, scale, row):
    return torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)


@opaque(fake=_native_fake, name="anthropic_a100_native_triattn")
def native_triangle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float,
    row: str,
) -> torch.Tensor:
    from miniworld_engine.integrations import anthropic

    face = anthropic.provider("triattn")
    from opt_core.attn import pair_fused

    form = "mask_bias" if mask is not None else "bias_only"
    selection = pair_fused.resolve_tier_core(
        row, *pair_fused._call_class(q, k), form=form, direction=face.FWD
    )
    if selection.row != "triattn_native":
        raise RuntimeError(
            "Native compile boundary received a different upstream selection"
        )
    with torch.no_grad():
        return face.triangle_attention(
            q, k, v, bias, mask, scale, word=row, selection=selection, form=form
        )
