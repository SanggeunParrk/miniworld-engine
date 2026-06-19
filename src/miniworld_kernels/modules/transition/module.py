# vendored from team-gm psk/benchmark : src/team_gm/modules/layers/transition.py
from contextlib import contextmanager

import torch
import torch.nn as nn
from jaxtyping import Float

from miniworld_kernels._typecheck import typecheck
from miniworld_kernels import kernels
from miniworld_kernels.modules.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)
from miniworld_kernels.modules.primitives import LayerNorm, Linear

from ..ops import swish_gate


@contextmanager
def nvtx_range(name: str, enabled: bool):
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


class Transition(nn.Module):
    """Transition layer with SwiGLU activation.

    Parameters
    ----------
    d_hidden : int
        Dimension of the input and output features.
    n : int
        Expansion factor.
    implementation : ImplementationType
        Implementation to use.

    """

    def __init__(
        self,
        d_hidden: int = 128,
        n: int = 4,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.n = n
        self.implementation = implementation

        self.ln_in = LayerNorm(
            d_hidden, implementation=implementation, dtype=torch.bfloat16
        )
        self.expand_a = Linear(
            d_hidden, d_hidden * n, bias=False, init="glorot", dtype=torch.bfloat16
        )
        self.expand_b = Linear(
            d_hidden, d_hidden * n, bias=False, init="glorot", dtype=torch.bfloat16
        )
        self.squeeze = Linear(
            d_hidden * n, d_hidden, bias=False, init="zero", dtype=torch.bfloat16
        )

    @typecheck
    def forward(self, x: Float[torch.Tensor, "*"]) -> Float[torch.Tensor, "*"]:
        """Forward pass."""
        x = self.ln_in(x)
        if self.implementation == ImplementationType.PYTORCH:
            a = self.expand_a(x)
            b = self.expand_b(x)
            x = swish_gate(a, b)
            return self.squeeze(x)

        if self.implementation == ImplementationType.CUDA:
            return kernels.cuda_transition(
                x,
                self.expand_a.weight,
                self.expand_b.weight,
                self.squeeze.weight,
                self.n,
            )

        if self.implementation in {
            ImplementationType.TRITON,
            ImplementationType.CUEQUIVARIANCE,
        }:
            return kernels.triton_transition(
                x,
                self.expand_a.weight,
                self.expand_b.weight,
                self.squeeze.weight,
                self.n,
            )

        raise InvalidImplementationError(self.implementation)
