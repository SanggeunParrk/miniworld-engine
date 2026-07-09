# vendored from team-gm psk/benchmark : src/team_gm/modules/layers/transition.py
import os
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from miniworld_kernels._typecheck import typecheck
from miniworld_kernels import kernels
from miniworld_kernels.modules import dispatch as _dispatch
from miniworld_kernels.modules.dispatch import KernelBackend, resolve_transition
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
        # 'miniworld' (ours, auto) resolves to the TRITON family, which itself
        # dispatches the best concrete kernel per shape/arch (hand-CUDA b2b for
        # d in {128,256} & n==4, cute split for d>=512, else triton). Transition has
        # no cuequivariance kernel, so CUEQUIVARIANCE shares that same auto path.
        # Resolution lives in modules.dispatch; forward routes on self._backend.
        self.implementation = ImplementationType(implementation)
        self._backend = resolve_transition(self.implementation)

        self.ln_in = LayerNorm(
            d_hidden, implementation=self.implementation, dtype=torch.bfloat16
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
        """Forward pass. Routes on the resolved internal backend (``_backend``),
        degrading to the pytorch reference (with a warning) on a dtype the fused
        kernels can't run."""
        backend = _dispatch.guard_dtype(self._backend, x.dtype, op="Transition")
        if backend == KernelBackend.PYTORCH:
            return self._torch_forward(x)

        if backend == KernelBackend.CUDA:
            x = self.ln_in(x)
            return kernels.cuda_transition(
                x,
                self.expand_a.weight,
                self.expand_b.weight,
                self.squeeze.weight,
                self.n,
            )

        if backend in {
            KernelBackend.TRITON,
            KernelBackend.CUEQUIVARIANCE,
        }:
            is_training = self.training and torch.is_grad_enabled()
            if is_training:
                return self._training_forward(x)
            return self._inference_forward(x)

        if backend == KernelBackend.CUTE:
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
        if _dispatch.is_sm100(x.device) and self.d_hidden >= 512:  # noqa: PLR2004
            # No fused sm_100 kernel fits d>=512 (b2b smem-OOMs; cute is SM90-only).
            # Fall back to the correct torch reference rather than crash (non-AF3 shape).
            return self._torch_forward(x)
        # Hand-CUDA fused b2b beats cute at d_hidden=128 (~2.07x) and d_hidden=256 (~1.21x)
        # for the AF3 shape (n=4 -> K=ND/4=D). Requires bf16 + n==4 + M%128==0. d_hidden=512
        # is hardware-limited for full fusion (smem cannot co-hold xn+weights+accumulator at
        # K=D=512) -> cute's non-fused tiled GEMM wins there, so it falls through below.
        if (
            _cuda_b2b_inference_enabled()
            # Hand-CUDA b2b hardcodes Hopper (sm_90a) cutlass includes and does not
            # build on Blackwell (sm_100) -> skip and fall through to the cute/triton
            # path (correctness-guard: never route to a backend that can't run here).
            and not _dispatch.is_sm100(x.device)
            and self.d_hidden in (128, 256)
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
        # cute_transition_fused is the quack SM90 (H100) WGMMA path; it asserts
        # SM90-only, so on Blackwell (sm_100) route the wide-d case to the triton
        # family too (its split path handles any d) — correctness-guard fallback.
        if self.d_hidden >= 256 and not _dispatch.is_sm100(x.device):
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
        """Training dispatch: fastest kernel per d (transition has NO cuequivariance kernel).

        Every path carries a real backward; measured fwd+bwd on H100 (L=384, bf16):
          d=128  b2b(+VersionA) 0.96ms  <  cute+tritonbwd 1.10  <  torch 1.73
          d=256  b2b 2.40 ~= cute+tritonbwd 2.39  <  torch 3.13
          d=512  cute+tritonbwd 6.90  <  torch 7.52   (b2b fwd OOMs smem at d=512)
        Mirrors the inference dispatch: b2b for d<=256, cute split for d=512.
        """
        if _dispatch.is_sm100(x.device) and self.d_hidden >= 512:  # noqa: PLR2004
            # No fused sm_100 kernel fits d>=512; correct torch fallback (non-AF3 shape).
            return self._torch_forward(x)
        if self.d_hidden >= 512 and not _dispatch.is_sm100(x.device):
            # d=512: b2b fusion can't fit smem (xn+weights+accumulator) and h round-trip is not
            # the bottleneck (compute-bound) -> cute split (expand + cuBLAS squeeze). The triton
            # (Version A style) backward beats the cute backend AND torch; env can override.
            # SM90 (H100) only; on Blackwell the triton family (below) carries d=512 too.
            backward_backend = _large_d_training_backend_from_env() or "triton"
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
        # d<=256 (Version A / save_xn=False): forward uses the fast hand-CUDA b2b kernel
        # (d in {128,256}, n==4, via triton_transition_fused's internal dispatch) and saves NO
        # xn; the shape-general backward recomputes xn from saved stats while using less memory.
        # Falls back to the split path when b2b is ineligible.
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

    def _torch_forward(self, x: torch.Tensor) -> torch.Tensor:
        # This module's weights are pinned bf16; match the input to them so the
        # reference runs even when reached as a dtype-fallback from a non-bf16 input.
        x = x.to(self.ln_in.weight.dtype)
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
