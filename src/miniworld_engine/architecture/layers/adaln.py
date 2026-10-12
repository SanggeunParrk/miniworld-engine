import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from miniworld_engine.architecture.layers.ops import apply_pairwise_rotation
from miniworld_engine.architecture.primitives import Linear, MPLinear
from miniworld_engine.architecture.typing import typecheck


class AdaptiveLayerNorm(nn.Module):
    """Adaptive LayerNorm.

    Standard mode applies AdaLN scale + shift: ``sigmoid(scale)*x + bias``.

    With ``use_rotation=True`` the additive shift is replaced by *rotation
    modulation* (arXiv:2505.19122): the conditioning predicts ``d_hidden/2``
    rotation angles applied to consecutive feature pairs. The shift is the term
    that breaks magnitude preservation (and whose ``to_bias`` weight grows
    unbounded -> late-training sampling collapse); a rotation is orthogonal so
    it conditions without changing activation magnitude. ``to_angle`` is
    zero-init, so at initialization the rotation is the identity.
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
        if use_rotation and d_hidden % 2 != 0:
            msg = f"use_rotation requires even d_hidden, got {d_hidden}."
            raise ValueError(msg)
        self.use_rotation = use_rotation
        self.ln_in = nn.LayerNorm(d_hidden, elementwise_affine=False)
        self.ln_cond = nn.LayerNorm(d_cond, bias=False)
        if mp_full:
            # MP the modulation projections too. to_scale keeps a bias=1 so the
            # sigmoid gate still starts ~0.73 (gating-like) despite the weight
            # being unit-Gaussian. to_angle is MP (non-zero) -> the rotation is a
            # small random rotation at init, not identity (DiT-MP convention).
            self.to_scale = MPLinear(d_cond, d_hidden, bias=True, init="normal")
            with torch.no_grad():
                self.to_scale.bias.fill_(1.0)
            if use_rotation:
                self.to_angle = MPLinear(
                    d_cond, d_hidden // 2, bias=False, init="normal"
                )
            else:
                self.to_bias = MPLinear(d_cond, d_hidden, bias=False, init="normal")
        else:
            self.to_scale = Linear(d_cond, d_hidden, init="gating")
            if use_rotation:
                self.to_angle = Linear(d_cond, d_hidden // 2, bias=False, init="zero")
            else:
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
        scaled = F.sigmoid(self.to_scale(cond)) * x
        if self.use_rotation:
            return apply_pairwise_rotation(scaled, self.to_angle(cond))
        return scaled + self.to_bias(cond)
