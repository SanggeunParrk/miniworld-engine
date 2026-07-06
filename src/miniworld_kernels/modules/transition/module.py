# vendored from team-gm psk/benchmark : src/team_gm/modules/layers/transition.py
import os
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


_LARGE_D_TRAINING_ENV = "MINIWORLD_TRANSITION_LARGE_D_TRAINING"
_CUTE_BACKWARD_ENV = "MINIWORLD_TRANSITION_CUTE_BACKWARD_BACKEND"
_CUDA_B2B_ENV = "MINIWORLD_TRANSITION_CUDA_B2B"


def _cuda_b2b_inference_enabled() -> bool:
    """Whether to route d=128/n=4 inference through the hand-CUDA fused b2b kernel
    (beats the Triton b2b ~1.29x). Default on; set MINIWORLD_TRANSITION_CUDA_B2B=0 to
    A/B against the Triton path."""
    value = os.getenv(_CUDA_B2B_ENV, "1").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _large_d_training_backend_from_env() -> str | None:
    value = os.getenv(_LARGE_D_TRAINING_ENV, "fallback").strip().lower()
    fallback_values = {"", "0", "false", "off", "fallback", "torch", "pytorch"}
    triton_values = {"1", "true", "yes", "on", "cute_triton", "cute-triton", "hybrid"}
    cute_values = {"cute", "all_cute", "all-cute"}
    if value in fallback_values:
        return None
    if value in triton_values:
        return "triton"
    if value in cute_values:
        return "cute"
    msg = (
        f"{_LARGE_D_TRAINING_ENV} must be one of fallback, cute_triton, or cute; "
        f"got {value!r}"
    )
    raise ValueError(msg)


def _explicit_cute_backward_backend() -> str:
    value = os.getenv(_CUTE_BACKWARD_ENV, "triton").strip().lower()
    if value in {"triton", "hybrid", "cute_triton", "cute-triton"}:
        return "triton"
    if value in {"cute", "all_cute", "all-cute"}:
        return "cute"
    msg = f"{_CUTE_BACKWARD_ENV} must be one of triton or cute; got {value!r}"
    raise ValueError(msg)


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
            backward_backend = _explicit_cute_backward_backend()
            return kernels.cute_transition_fused(
                x,
                self.ln_in.weight,
                self.ln_in.bias,
                self.expand_a.weight,
                self.expand_b.weight,
                self.squeeze.weight,
                self.n,
                self.ln_in.eps,
                backward_backend=backward_backend,
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
        # Hand-CUDA fused b2b beats the Triton b2b ~1.29x at the fixed AF3 shape
        # (d_hidden=128, n=4 -> K=128, ND=512, D=128). Requires bf16 + M%128==0.
        if (
            _cuda_b2b_inference_enabled()
            and self.d_hidden == 128
            and self.n == 4
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and (x.numel() // self.d_hidden) % 128 == 0
        ):
            return kernels.cuda_transition_b2b(
                x,
                self.ln_in.weight,
                self.ln_in.bias,
                self.expand_a.weight,
                self.expand_b.weight,
                self.squeeze.weight,
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
            backward_backend = _large_d_training_backend_from_env()
            if backward_backend is not None:
                return kernels.cute_transition_fused(
                    x,
                    self.ln_in.weight,
                    self.ln_in.bias,
                    self.expand_a.weight,
                    self.expand_b.weight,
                    self.squeeze.weight,
                    self.n,
                    self.ln_in.eps,
                    backward_backend=backward_backend,
                )
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
