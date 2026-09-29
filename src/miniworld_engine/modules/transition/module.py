# vendored from team-gm psk/benchmark : src/team_gm/modules/layers/transition.py
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from miniworld_engine._typecheck import typecheck
from miniworld_engine.modules import dispatch as _dispatch
from miniworld_engine.modules.dispatch import KernelBackend, resolve_transition
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.functional import swish_gate
from miniworld_engine.modules.primitives import LayerNorm, Linear


def _fused_sm90a_enabled() -> bool:
    """Whether to route the d=128/n=4 bf16 residual path on sm_90 through the fused hand-CUDA
    forward and backward (two launches instead of five, ~1.9x the Triton residual path on both
    sides). Default on; set MINIWORLD_TRANSITION_FUSED_SM90A=0 to A/B against Triton."""
    from miniworld_engine import settings

    return settings.current().transition_fused_sm90a and settings.current().engine_backend != "triton"


def _fused_sm80_enabled() -> bool:
    """Whether to route the d=128/n=4 bf16 residual path on sm_80 through the fused hand-CUDA forward and two-kernel backward
    (training step ~1.44x the Triton residual path).  Default on; MINIWORLD_TRANSITION_FUSED_SM80=0 to A/B against Triton."""
    from miniworld_engine import settings

    return settings.current().engine_backend != "triton"


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
        # Every backend returns x + transition(x); the residual is folded into the squeeze
        # epilogue and the input-LN backward. No dropout in this module.
        # 'miniworld' (ours, auto) resolves to the TRITON family: the hand-CUDA sm_90 kernels
        # (fused_sm90a at d=128, fused_wide_sm90a at d=64/256/384/512, n=4, bf16) where they
        # apply, else the shape-general Triton residual path. Transition has NO cuequivariance
        # kernel, so an explicit CUEQUIVARIANCE request falls back to the PYTORCH reference
        # (resolve() maps cueq->pytorch for non-trimul ops).
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

        The raw op without the residual is available through ``ops.transition``.
        """
        backend = _dispatch.guard_dtype(self._backend, x.dtype, op="Transition")
        if backend == KernelBackend.PYTORCH:
            return self._torch_forward(x) + x
        # TRITON / CUDA / CUEQUIVARIANCE(->PYTORCH above): one residual-fused path, which
        # picks the hand-CUDA kernel where it applies and the Triton kernel everywhere else.
        return self._residual_forward(x)

    def _residual_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual-fused Transition: LN, expand-SwiGLU, squeeze with the residual folded into
        its epilogue, and the matching backward.

        On sm_90 at the AF3 pair width (d=128, n=4, bf16, whole 128-row tiles) this runs the
        fused hand-CUDA kernels -- one launch each way instead of three and two. d = 64, 256, 384
        and 512 (n = 4) have their own hand-CUDA builds (``fused_wide_sm90a``). Every other
        shape, dtype and architecture keeps the Triton path, which is shape-general. The gates are
        ``fused_sm90a.available`` / ``fused_wide_sm90a.available``; they are the kernels' own
        requirements, not a policy.
        """
        wa = self.expand_a.weight.to(x.dtype)
        wb = self.expand_b.weight.to(x.dtype)
        ws = self.squeeze.weight.to(x.dtype)
        if _fused_sm90a_enabled():
            from miniworld_engine.kernels.transition.cuda import fused_sm90a

            if fused_sm90a.available(x, wa, ws):
                return fused_sm90a.transition_fused_sm90a(
                    x, self.ln_in.weight, self.ln_in.bias, wa, wb, ws, self.ln_in.eps)
            # the other widths with an sm_90a build: D = 64, 256, 384, 512 (see fused_wide_sm90a)
            from miniworld_engine.kernels.transition.cuda import fused_wide_sm90a

            if fused_wide_sm90a.available(x, wa, ws):
                return fused_wide_sm90a.transition_wide_sm90a(
                    x, self.ln_in.weight, self.ln_in.bias, wa, wb, ws, self.ln_in.eps)

        if _fused_sm80_enabled():
            from miniworld_engine.kernels.transition.cuda import fused_sm80

            if fused_sm80.available(x, wa, ws):
                return fused_sm80.transition_fused_sm80(
                    x, self.ln_in.weight, self.ln_in.bias, wa, wb, ws, self.ln_in.eps)

        from miniworld_engine.kernels.transition.triton.residual import (
            transition_residual,
        )

        return transition_residual(
            x, self.ln_in.weight, self.ln_in.bias, wa, wb, ws, self.ln_in.eps,
        )

    def _torch_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Norm affine params are fp32-pinned and the projections bf16-pinned; cast both to the
        # activation dtype. This is the path `guard_dtype` sends every non-bf16 input to, so
        # applying the bf16 Linear modules directly would fail on exactly the dtypes it serves.
        x = F.layer_norm(
            x,
            (self.d_hidden,),
            self.ln_in.weight.to(x.dtype),
            self.ln_in.bias.to(x.dtype),
            self.ln_in.eps,
        )
        a = F.linear(x, self.expand_a.weight.to(x.dtype))
        b = F.linear(x, self.expand_b.weight.to(x.dtype))
        x = swish_gate(a, b)
        return F.linear(x, self.squeeze.weight.to(x.dtype))
