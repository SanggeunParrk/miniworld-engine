import torch
import torch.nn as nn
from jaxtyping import Float

from miniworld_engine.architecture.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)
from miniworld_engine.architecture.layers.adaln import AdaptiveLayerNorm
from miniworld_engine.architecture.layers.ops import (
    mp_swish_gate,
    sigmoid_gate,
    swish_gate,
)
from miniworld_engine.architecture.primitives import Linear, MPLinear
from miniworld_engine.architecture.typing import typecheck


class ConditionedTransition(nn.Module):
    """Transition layer with SwiGLU activation and adaptive conditioning.

    Parameters
    ----------
    d_hidden : int
        Dimension of the input and output features.
    d_cond : int
        Dimension of the conditioning features.
    n : int
        Expansion factor.
    implementation : ImplementationType
        Implementation to use.

    """

    def __init__(
        self,
        d_hidden: int,
        d_cond: int,
        n: int = 2,
        *,
        magnitude_preserving: bool = False,
        use_rotation: bool = False,
        mp_full: bool = False,
        implementation: ImplementationType = ImplementationType.MINIWORLD_ENGINE,
    ) -> None:
        super().__init__()

        self.n = n
        magnitude_preserving = magnitude_preserving or mp_full
        self.magnitude_preserving = magnitude_preserving
        self.mp_full = mp_full
        # The miniworld_engine transition kernel consumes raw expand/squeeze weights, which
        # bypasses MPLinear's on-use normalization. So MP forces the PyTorch path.
        self.implementation = (
            ImplementationType.PYTORCH if magnitude_preserving else implementation
        )

        dense = MPLinear if magnitude_preserving else Linear
        self.ada_ln_in = AdaptiveLayerNorm(
            d_hidden=d_hidden, d_cond=d_cond, use_rotation=use_rotation, mp_full=mp_full
        )
        self.expand_a = dense(d_hidden, d_hidden * n, bias=False, init="relu")
        self.expand_b = dense(d_hidden, d_hidden * n, bias=False, init="relu")
        # Under mp_full the output projection + conditioning gate are MP too
        # (lose zero-init); the bias=-2 on to_scale keeps the gate weak at init.
        if mp_full:
            self.squeeze = MPLinear(d_hidden * n, d_hidden, bias=False, init="normal")
            self.to_scale = MPLinear(d_cond, d_hidden, bias=True, init="normal")
        else:
            self.squeeze = Linear(d_hidden * n, d_hidden, bias=False, init="zero")
            self.to_scale = Linear(d_cond, d_hidden, bias=True, init="zero")
        self.to_scale.bias.data.fill_(-2.0)

    @typecheck
    def forward(
        self,
        x: Float[torch.Tensor, "* d_hidden"],
        cond: Float[torch.Tensor, "* d_cond"],
    ) -> Float[torch.Tensor, "* d_hidden"]:
        """Forward pass."""
        x = self.ada_ln_in(x, cond)

        if self.implementation == ImplementationType.MINIWORLD_ENGINE:
            # Whole-op path: the conditioned SwiGLU tail (SwiGLU + squeeze +
            # sigmoid(to_scale(cond)) gate) lives in miniworld-engine; the layer only
            # holds the weights (AdaLN above is model math, applied here). Lazy import.
            from miniworld_engine import ops

            return ops.conditioned_transition(
                x,
                cond,
                expand_a_weight=self.expand_a.weight,
                expand_b_weight=self.expand_b.weight,
                squeeze_weight=self.squeeze.weight,
                to_scale_weight=self.to_scale.weight,
                to_scale_bias=self.to_scale.bias,
                n=self.n,
            )

        if self.implementation == ImplementationType.PYTORCH:
            a = self.expand_a(x)
            b = self.expand_b(x)
            # MP-SwiGLU (÷0.596) keeps unit variance when expand layers are MP.
            x = mp_swish_gate(a, b) if self.magnitude_preserving else swish_gate(a, b)
            out = self.squeeze(x)
            return sigmoid_gate(self.to_scale(cond), out)

        raise InvalidImplementationError(self.implementation)
