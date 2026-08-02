# vendored from team-gm psk/benchmark : src/team_gm/modules/layers/adaln.py
import torch
import torch.nn as nn
from jaxtyping import Float

from miniworld_engine._typecheck import typecheck
from miniworld_engine.kernels import adaln_inference, adaln_train
from miniworld_engine.modules.dispatch import KernelBackend, resolve_adaptive_layernorm
from miniworld_engine.modules.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)
from miniworld_engine.modules.primitives import Linear


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
        self.implementation = ImplementationType(implementation)
        # 'miniworld' (auto) -> the TRITON family (the only fused adaln kernel).
        self._backend = resolve_adaptive_layernorm(self.implementation)
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
        """Forward pass. Routes on the resolved internal backend (``_backend``)."""
        if self._backend == KernelBackend.PYTORCH:
            x_norm = self.ln_in(x)
            cond_norm = self.ln_cond(cond)
            scale = self.to_scale(cond_norm)
            bias = self.to_bias(cond_norm)
            return torch.sigmoid(scale) * x_norm + bias

        if self._backend == KernelBackend.TRITON:
            # training → save-for-backward autograd path; inference → d-aware fused/materialize.
            # (The legacy single fused `triton_adaptive_layer_norm` compile-fails at d≥384.)
            fn = adaln_train if (self.training or x.requires_grad) else adaln_inference
            return fn(
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
