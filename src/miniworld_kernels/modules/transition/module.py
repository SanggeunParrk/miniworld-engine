# vendored from team-gm psk/benchmark : src/team_gm/modules/layers/transition.py
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        if self.implementation == ImplementationType.PYTORCH:
            return self._torch_forward(x)

        if self.implementation == ImplementationType.CUDA:
            x = self.ln_in(x)
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
            is_training = self.training and torch.is_grad_enabled()
            if is_training:
                return self._training_forward(x)
            return self._inference_forward(x)

        if self.implementation == ImplementationType.CUTE:
            # Force the cute (quack SM90 WGMMA) backend regardless of d (for benchmarking /
            # explicit selection). Same fused structure; LN folded into the cute expand.
            return kernels.cute_transition_fused(
                x,
                self.ln_in.weight,
                self.ln_in.bias,
                self.expand_a.weight,
                self.expand_b.weight,
                self.squeeze.weight,
                self.n,
                self.ln_in.eps,
            )

        raise InvalidImplementationError(self.implementation)

    def _inference_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-only dispatch: no tensors are saved for backward."""
        if self.d_hidden >= 256:
            return kernels.cute_transition_fused(
                x,
                self.ln_in.weight,
                self.ln_in.bias,
                self.expand_a.weight,
                self.expand_b.weight,
                self.squeeze.weight,
                self.n,
                self.ln_in.eps,
            )
        return kernels.triton_transition_fused(
            x,
            self.ln_in.weight,
            self.ln_in.bias,
            self.expand_a.weight,
            self.expand_b.weight,
            self.squeeze.weight,
            self.n,
            self.ln_in.eps,
            save_xn=False,
        )

    def _training_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Training dispatch: route to kernels with an explicit backward contract."""
        if (
            self.implementation == ImplementationType.CUEQUIVARIANCE
            and self.d_hidden >= 256
        ):
            # Current cute backward is still slower than compiled PyTorch for wide
            # training shapes; keep MiniWorld dispatch performance non-regressive.
            return self._torch_forward(x)
        if self.d_hidden >= 256:
            return kernels.cute_transition_fused(
                x,
                self.ln_in.weight,
                self.ln_in.bias,
                self.expand_a.weight,
                self.expand_b.weight,
                self.squeeze.weight,
                self.n,
                self.ln_in.eps,
            )
        return kernels.triton_transition_fused(
            x,
            self.ln_in.weight,
            self.ln_in.bias,
            self.expand_a.weight,
            self.expand_b.weight,
            self.squeeze.weight,
            self.n,
            self.ln_in.eps,
            save_xn=True,
        )

    def _torch_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.layer_norm(
            x,
            (self.d_hidden,),
            self.ln_in.weight,
            self.ln_in.bias,
            self.ln_in.eps,
        )
        a = self.expand_a(x)
        b = self.expand_b(x)
        x = swish_gate(a, b)
        return self.squeeze(x)
