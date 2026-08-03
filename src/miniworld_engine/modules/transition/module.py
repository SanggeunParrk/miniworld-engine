# vendored from team-gm psk/benchmark : src/team_gm/modules/layers/transition.py
import os
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from miniworld_engine._typecheck import typecheck
from miniworld_engine import kernels
from miniworld_engine.modules import dispatch as _dispatch
from miniworld_engine.modules.dispatch import KernelBackend, resolve_transition
from miniworld_engine.modules.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)
from miniworld_engine.modules.primitives import LayerNorm, Linear

from ..ops import swish_gate


_LARGE_D_TRAINING_ENV = "MINIWORLD_TRANSITION_LARGE_D_TRAINING"
_CUTE_BACKWARD_ENV = "MINIWORLD_TRANSITION_CUTE_BACKWARD_BACKEND"
_CUDA_B2B_ENV = "MINIWORLD_TRANSITION_CUDA_B2B"


def _force_split_enabled() -> bool:
    """Force the split path (ln_in + non-fused triton_transition) instead of the fused
    kernels. The fused large-d path uses the bounded-smem k-tiled b2b (no OOM at d>=256),
    but the split is the proven, shape-general fallback: set MINIWORLD_TRANSITION_FORCE_SPLIT=1
    to A/B against it or as an escape hatch on a GPU where the fused path misbehaves. Static
    (env, compile-safe) rather than a runtime try/except, which is fragile under torch.compile."""
    return os.getenv("MINIWORLD_TRANSITION_FORCE_SPLIT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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
        # ==========================================================================
        # THIS MODULE ALWAYS APPLIES THE POST-TRANSITION RESIDUAL: y = x + transition(x).
        # The residual connection is UNCONDITIONAL — it is baked into the op (folded into the
        # b2b/triton epilogue for free, reusing the already-loaded input tile). This is the AF3
        # pairformer default (``pair = pair + transition(pair)``) and residual connections are
        # ubiquitous/standard in this domain, so there is deliberately NO flag to turn it off.
        # The transition track has NO dropout (AF3), so this module takes no dropout either.
        #
        # WHY IT'S FUSED IN (SPEED): the residual add is done INSIDE the transition kernel's
        # output epilogue rather than as a separate ``out + x`` elementwise op. That removes a
        # whole extra kernel launch and its M×D read+write round-trip through HBM — the input
        # tile is already resident in registers/smem at store time, so the add is effectively
        # free. Keeping it unconditional is what lets the kernel own that fused epilogue; a
        # runtime toggle would force the slow separate-add path.
        # >>> To run WITHOUT the residual (rare — benchmarking the raw op in isolation), you must
        # >>> EDIT THE CODE: flip the ``_ADD_RESIDUAL`` local at the top of forward() to False.
        # ==========================================================================
        # 'miniworld' (ours, auto) resolves to the TRITON family, which itself
        # dispatches the best concrete kernel per shape/arch (hand-CUDA b2b for
        # d in {128,256} & n==4, cute split for d>=512, else triton). Transition has
        # NO cuequivariance kernel, so an explicit CUEQUIVARIANCE request falls back to
        # the PYTORCH reference (resolve() maps cueq->pytorch for non-trimul ops) — NOT
        # the Triton fused path (whose bf16 kernel OOMs shared memory at d>=256 on H100).
        # Resolution lives in modules.dispatch; forward routes on self._backend.
        self.implementation = ImplementationType(implementation)
        self._backend = resolve_transition(self.implementation)

        self.ln_in = LayerNorm(
            d_hidden, implementation=self.implementation, dtype=torch.bfloat16
        )
        self.expand_a = Linear(
            d_hidden, d_hidden * n, bias=False, init="relu", dtype=torch.bfloat16
        )
        self.expand_b = Linear(
            d_hidden, d_hidden * n, bias=False, init="relu", dtype=torch.bfloat16
        )
        self.squeeze = Linear(
            d_hidden * n, d_hidden, bias=False, init="zero", dtype=torch.bfloat16
        )

    @typecheck
    def forward(self, x: Float[torch.Tensor, "*"]) -> Float[torch.Tensor, "*"]:
        """Forward pass. ALWAYS returns the residual output ``y = x + transition(x)`` (the
        residual is this module's own input ``x``). Routes on the resolved internal backend
        (``_backend``), degrading to the pytorch reference (with a warning) on a dtype the fused
        kernels can't run.

        The residual is UNCONDITIONAL and fused into the kernel epilogue FOR SPEED (no separate
        ``out + x`` kernel / HBM round-trip; the input tile is already resident at store time) —
        see the constructor comment. There is intentionally no runtime flag to disable it, as
        residual connections are the standard in this domain (AF3 ``pair = pair + transition(pair)``).
        >>> To disable the residual (benchmarking the raw op), EDIT the ``_ADD_RESIDUAL`` line below."""
        _ADD_RESIDUAL = True  # UNCONDITIONAL residual (fused epilogue, for speed). Edit to False to disable.
        add_residual = _ADD_RESIDUAL
        backend = _dispatch.guard_dtype(self._backend, x.dtype, op="Transition")
        if backend == KernelBackend.PYTORCH:
            out = self._torch_forward(x)
            return out + x if add_residual else out

        if backend == KernelBackend.CUDA:
            xln = self.ln_in(x)
            out = kernels.cuda_transition(
                xln,
                self.expand_a.weight,
                self.expand_b.weight,
                self.squeeze.weight,
                self.n,
            )
            return out + x if add_residual else out

        if backend in {
            KernelBackend.TRITON,
            KernelBackend.CUEQUIVARIANCE,
        }:
            is_training = self.training and torch.is_grad_enabled()
            if is_training:
                return self._training_forward(x, add_residual)
            return self._inference_forward(x, add_residual)

        if backend == KernelBackend.CUTE:
            # Force the cute (quack SM90 WGMMA) backend regardless of d (for benchmarking /
            # explicit selection). Same fused structure; LN folded into the cute expand.
            backward_backend = _explicit_cute_backward_backend()
            out = kernels.cute_transition_fused(
                x,
                self.ln_in.weight.to(x.dtype),
                self.ln_in.bias.to(x.dtype),
                self.expand_a.weight,
                self.expand_b.weight,
                self.squeeze.weight,
                self.n,
                self.ln_in.eps,
                backward_backend=backward_backend,
            )
            return out + x if add_residual else out

        raise InvalidImplementationError(self.implementation)

    def _inference_forward(self, x: torch.Tensor, add_residual: bool = False) -> torch.Tensor:
        """Forward-only dispatch: no tensors are saved for backward."""
        def _r(out):  # explicit residual add for paths that don't fold it in-kernel
            return out + x if add_residual else out

        if _force_split_enabled():
            return _r(self._old_triton_forward(x))
        # Pre-Hopper (sm_80 / A100), large d (>=256): the fused triton path uses the
        # bounded-smem k-tiled b2b there (correct, no OOM) but it is SLOWER than the
        # shape-general split on A100 (measured cudagraph-manual: d=256 2.28 vs 1.40 ms,
        # d=512 10.9 vs 4.8 ms). So default large-d to the split; d=128 (the AF3 shape)
        # still takes the fused b2b below, where it wins. is_sm90plus keeps H100/B200 on
        # their fused/cute paths unchanged.
        # sm_86 (RTX A5000/A6000): the fused b2b/triton path loses to the shape-general split
        # even at d=128 (measured cudagraph-manual: ~0.88-0.95x vs old_triton split, both L and
        # d sweeps), so route ALL d to the split on sm_86. A100 (sm_80) keeps d=128 fused, where
        # it wins, via the plain `>= 256` gate.
        if not _dispatch.is_sm90plus(x.device) and (
            self.d_hidden >= 256 or _dispatch.is_sm86(x.device)
        ):
            return _r(self._old_triton_forward(x))
        # sm_100 (B200): ALL d (128/256/512) go through the fused path below -> the cute
        # b2b_fwd_sm100 forward (fits smem + correct + cudagraph-capturable at every d; see
        # fused.py's cuda_b2b_ok gate). The legacy split (_old_triton_forward) is retained
        # only as an explicit fallback and is no longer on the default sm100 path.
        # Hand-CUDA fused b2b beats cute at d_hidden=128 (~2.07x) and d_hidden=256 (~1.21x)
        # for the AF3 shape (n=4 -> K=ND/4=D). Requires bf16 + n==4 + M%128==0. d_hidden=512
        # is hardware-limited for full fusion (smem cannot co-hold xn+weights+accumulator at
        # K=D=512) -> cute's non-fused tiled GEMM wins there, so it falls through below.
        if (
            _cuda_b2b_inference_enabled()
            # Hand-CUDA b2b is Hopper (sm_90a) WGMMA/TMA cutlass code: it does not
            # build/launch on Blackwell (sm_100) NOR on pre-Hopper (sm_80 / A100).
            # Gate on Hopper *exactly* (is_sm90) -- `not is_sm100` also matched A100
            # and crashed it. Everything else falls through to the portable Triton
            # path (correctness-guard: never route to a backend that can't run here).
            and _dispatch.is_sm90(x.device)
            and self.d_hidden in (128, 256)
            and self.n == 4
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and (x.numel() // self.d_hidden) % 128 == 0
        ):
            return kernels.cuda_transition_b2b(
                x,
                self.ln_in.weight.to(x.dtype),
                self.ln_in.bias.to(x.dtype),
                self.expand_a.weight,
                self.expand_b.weight,
                self.squeeze.weight,
                self.ln_in.eps,
                add_residual=add_residual,
            )
        # cute_transition_fused is the quack SM90 (H100) WGMMA path; it asserts
        # SM90-only. Route the wide-d case here on Hopper *exactly*; on Blackwell
        # (sm_100) AND on pre-Hopper (sm_80 / A100) fall through to the triton
        # family (its split path handles any d) — correctness-guard fallback.
        if self.d_hidden >= 256 and _dispatch.is_sm90(x.device):
            return _r(kernels.cute_transition_fused(
                x,
                self.ln_in.weight.to(x.dtype),
                self.ln_in.bias.to(x.dtype),
                self.expand_a.weight,
                self.expand_b.weight,
                self.squeeze.weight,
                self.n,
                self.ln_in.eps,
            ))
        return kernels.triton_transition_fused(
            x,
            self.ln_in.weight.to(x.dtype),
            self.ln_in.bias.to(x.dtype),
            self.expand_a.weight,
            self.expand_b.weight,
            self.squeeze.weight,
            self.n,
            self.ln_in.eps,
            save_xn=False,
            add_residual=add_residual,
        )

    def _training_forward(self, x: torch.Tensor, add_residual: bool = False) -> torch.Tensor:
        """Training dispatch: fastest kernel per d (transition has NO cuequivariance kernel).

        Every path carries a real backward; measured fwd+bwd on H100 (L=384, bf16):
          d=128  b2b(+VersionA) 0.96ms  <  cute+tritonbwd 1.10  <  torch 1.73
          d=256  b2b 2.40 ~= cute+tritonbwd 2.39  <  torch 3.13
          d=512  cute+tritonbwd 6.90  <  torch 7.52   (b2b fwd OOMs smem at d=512)
        Mirrors the inference dispatch: b2b for d<=256, cute split for d=512.
        """
        def _r(out):  # explicit residual add for paths that don't fold it in-kernel
            return out + x if add_residual else out

        if _force_split_enabled():
            return _r(self._old_triton_forward(x))
        # Pre-Hopper (sm_80 / A100), large d (>=256): split beats the fused k-tiled path
        # in training too (d=256 6.3 vs 7.0 ms, d=512 20.2 vs 26.8 ms). d=128 stays fused
        # on A100. sm_86 (RTX A5000/A6000) also loses at d=128 (~0.91x vs split), so route
        # ALL d to the split there too.
        if not _dispatch.is_sm90plus(x.device) and (
            self.d_hidden >= 256 or _dispatch.is_sm86(x.device)
        ):
            return _r(self._old_triton_forward(x))
        # sm_100 (B200): ALL d (128/256/512) go through the fused path below -> cute
        # b2b_fwd_sm100 forward + the sm100 gatebwd (Version A) backward. Verified fwd+bwd
        # cos=1.0 and cudagraph-capturable at every d; ~1.49x faster fwd+bwd than the legacy
        # split at d=256 (1.45 vs 2.16ms cudagraph). d=512 now also keeps gatebwd instead of
        # the split's legacy backward. (_old_triton_forward retained only as a fallback.)
        if self.d_hidden >= 512 and _dispatch.is_sm90(x.device):
            # d=512: b2b fusion can't fit smem (xn+weights+accumulator) and h round-trip is not
            # the bottleneck (compute-bound) -> cute split (expand + cuBLAS squeeze). The triton
            # (Version A style) backward beats the cute backend AND torch; env can override.
            # SM90 (H100) only; on Blackwell AND pre-Hopper (A100) the triton family (below)
            # carries d=512 too.
            backward_backend = _large_d_training_backend_from_env() or "triton"
            return _r(kernels.cute_transition_fused(
                x,
                self.ln_in.weight.to(x.dtype),
                self.ln_in.bias.to(x.dtype),
                self.expand_a.weight,
                self.expand_b.weight,
                self.squeeze.weight,
                self.n,
                self.ln_in.eps,
                backward_backend=backward_backend,
            ))
        # d<=256 (Version A / save_xn=False): fused forward (b2b / sm100 cute fwd) + the
        # sm100 gatebwd backward (recomputes xn from saved stats, less memory). On sm_100
        # only d=128 reaches here (d>=256 took the split branch above); the AF3 shape.
        return kernels.triton_transition_fused(
            x,
            self.ln_in.weight.to(x.dtype),
            self.ln_in.bias.to(x.dtype),
            self.expand_a.weight,
            self.expand_b.weight,
            self.squeeze.weight,
            self.n,
            self.ln_in.eps,
            save_xn=False,
            add_residual=add_residual,
        )

    def _old_triton_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Legacy split triton transition (ln_in + triton_transition): the shape-general
        path used on sm_100 for d>=256, where the fused kernels don't fit (b2b/cute are
        Hopper-only, triton_transition_fused smem-OOMs). Carries a real backward, so it
        serves both inference and training."""
        x = self.ln_in(x)
        return kernels.triton_transition(
            x,
            self.expand_a.weight,
            self.expand_b.weight,
            self.squeeze.weight,
            self.n,
        )

    def _torch_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Norm affine params are fp32-pinned; cast them to the activation dtype for the
        # reference LN so the downstream bf16 expand/squeeze get a matching activation.
        x = F.layer_norm(
            x,
            (self.d_hidden,),
            self.ln_in.weight.to(x.dtype),
            self.ln_in.bias.to(x.dtype),
            self.ln_in.eps,
        )
        a = self.expand_a(x)
        b = self.expand_b(x)
        x = swish_gate(a, b)
        return self.squeeze(x)
