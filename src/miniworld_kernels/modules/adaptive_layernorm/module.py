# vendored from team-gm psk/benchmark : src/team_gm/modules/layers/adaln.py
import torch
import torch.nn as nn
from jaxtyping import Float

from miniworld_kernels._typecheck import typecheck
from miniworld_kernels.kernels import triton_adaptive_layer_norm
from miniworld_kernels.modules.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)
from miniworld_kernels.modules.primitives import Linear


class AdaptiveLayerNorm(nn.Module):
    """Adaptive LayerNorm."""

    def __init__(
        self,
        d_hidden: int,
        d_cond: int,
        *,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()
        self.implementation = implementation
        self.ln_in = nn.LayerNorm(d_hidden, elementwise_affine=False)
        self.ln_cond = nn.LayerNorm(d_cond, bias=False)
        self.to_scale = Linear(d_cond, d_hidden, init="gating")
        self.to_bias = Linear(d_cond, d_hidden, bias=False, init="zero")

    @typecheck
    def forward(
        self,
        x: Float[torch.Tensor, "* d_hidden"],
        cond: Float[torch.Tensor, "* d_cond"],
    ) -> Float[torch.Tensor, "* d_hidden"]:
        """Forward pass."""
        if self.implementation == ImplementationType.PYTORCH:
            x_norm = self.ln_in(x)
            cond_norm = self.ln_cond(cond)
            scale = self.to_scale(cond_norm)
            bias = self.to_bias(cond_norm)
            return torch.sigmoid(scale) * x_norm + bias

        if self.implementation == ImplementationType.TRITON:
            return triton_adaptive_layer_norm(
                x,
                cond,
                self.ln_cond.weight,
                self.to_scale.weight,
                self.to_scale.bias,
                self.to_bias.weight,
                self.ln_in.eps,
                self.ln_cond.eps,
            )

        raise InvalidImplementationError(self.implementation)
