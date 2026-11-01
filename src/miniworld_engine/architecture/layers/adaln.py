import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from miniworld_engine.architecture.exceptions import legacy_options_removed
from miniworld_engine.architecture.primitives import Linear
from miniworld_engine.architecture.typing import typecheck


class AdaptiveLayerNorm(nn.Module):
    """Adaptive LayerNorm: AdaLN scale + shift, ``sigmoid(scale)*x + bias``.

    The rotation-modulation (``use_rotation``) and fully magnitude-preserving (``mp_full``) variants were removed with the other legacy
    MP options (``architecture.exceptions.legacy_options_removed``); asking for one raises.
    """

    def __init__(
        self,
        d_hidden: int,
        d_cond: int,
        *,
        use_rotation: bool = False,
        mp_full: bool = False,
    ) -> None:
        super().__init__()
        legacy_options_removed("AdaptiveLayerNorm", use_rotation=use_rotation, mp_full=mp_full)
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
        x = self.ln_in(x)
        cond = self.ln_cond(cond)
        return F.sigmoid(self.to_scale(cond)) * x + self.to_bias(cond)
