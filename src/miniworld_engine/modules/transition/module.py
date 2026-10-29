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
    sides). Off only with ``force_portable_kernels``."""
    from miniworld_engine import settings

    return not settings.current().force_portable_kernels


def _fused_sm100a_enabled() -> bool:
    """Whether to route the n=4 bf16 residual path on sm_100 (B200) through the hand-CUDA kernels: ``fused_sm100a`` at d=128
    (module training step 2.5x / 2.3x the Triton residual path at L384 / L768) and ``fused_wide_sm100a`` at d=64/256/384/512.
    Default on; settings.transition_fused_sm100a=0 to A/B against Triton."""
    from miniworld_engine import settings

    return (not settings.current().force_portable_kernels)


def _fused_sm80_enabled() -> bool:
    """Whether to route the d=128/n=4 bf16 residual path on sm_80 through the fused hand-CUDA forward and two-kernel backward
    (training step ~1.44x the Triton residual path).  Default on; settings.transition_fused_sm80=0 to A/B against Triton."""
    from miniworld_engine import settings

    return (not settings.current().force_portable_kernels)


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
        # 'miniworld' (ours, auto) resolves to the PORTABLE family: the hand-CUDA sm_90 kernels
        # (fused_sm90a at d=128, fused_wide_sm90a at d=64/256/384/512, n=4 and n=2, bf16, whole 128-row tiles; fused_generic_sm90a
        # for any other d (multiple of 64), hidden (multiple of 128), ragged rows, and fp32 on TF32) where they
        # apply, else the shape-general Triton residual path. Transition has NO cuequivariance
        # kernel, so an explicit CUEQUIVARIANCE request falls back to the PYTORCH reference
        # (resolve() maps cueq->pytorch for non-trimul ops).
        # Resolution lives in modules.dispatch; forward routes on self._backend.
        self.implementation = ImplementationType(implementation)
        native_cuda = self._native_cuda_serves()
        self._backend = resolve_transition(self.implementation, native_cuda=native_cuda)
        # a card recorded for the forward kernels but not the backward ones: training calls take the PyTorch reference
        self._train_gate = _dispatch.train_gate("transition", self.implementation, self._backend, native_cuda=native_cuda)

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

        The same op with the weights as arguments (residual included) is ``ops.transition``.
        """
        if self._requires_h100_cuda(x):
            if not _fused_sm90a_enabled():
                return self._torch_forward(x) + x   # the native CUDA kernels are switched off: PyTorch, never Triton
            if x.dtype is torch.float32 and self._generic_sm90a_serves(x):
                # fp32 on TF32 tensor cores: the generic sm_90a kernels (the other fast paths are bf16 only)
                return self._residual_forward(x)
            matrices = (self.expand_a.weight, self.expand_b.weight, self.squeeze.weight)
            if (
                x.dtype != torch.bfloat16
                or not x.is_contiguous()
                or x.numel() == 0
                or any(
                    p.dtype != torch.float32 or not p.is_contiguous()
                    for p in (self.ln_in.weight, self.ln_in.bias)
                )
                or matrices[0].dtype not in (torch.bfloat16, torch.float32)
                or any(p.dtype != matrices[0].dtype for p in matrices)
            ):
                # no CUDA path takes this activation / parameter layout: the PyTorch reference (no Triton fallback on H100 / B200)
                return self._torch_forward(x) + x
        if (
            x.dtype is torch.float32
            and self._backend != KernelBackend.PYTORCH
            and self._generic_sm90a_serves(x)
        ):
            # fp32 on TF32 tensor cores: the generic sm_90a kernels (the other fast paths are bf16 only)
            return self._residual_forward(x)
        backend = _dispatch.call_backend(self, self._backend, x)
        if x.dtype is torch.float32 and backend == KernelBackend.PORTABLE and x.is_cuda and self._portable_fp32_allowed():
            # fp32 on TF32 tensor cores (the portable residual path; the H100 CUDA kernels above take the same switch): the card needs
            # fp32 rows in its record for what this call launches, else one UnrecordedCardWarning and the PyTorch reference
            backend = _dispatch.dtype_gate("transition", x.device, "fp32", training=_dispatch.needs_backward(self, x))
        else:
            backend = _dispatch.guard_dtype(backend, x.dtype, op="Transition")
        if backend == KernelBackend.PYTORCH:
            return self._torch_forward(x) + x
        # PORTABLE / CUDA / CUEQUIVARIANCE(->PYTORCH above): one residual-fused path, which
        # picks the hand-CUDA kernel where it applies and the portable kernel everywhere else (bf16; fp32 on TF32, see above).
        return self._residual_forward(x)

    @staticmethod
    def _portable_fp32_allowed() -> bool:
        """The portable fp32 path runs on TF32 tensor cores, and only where torch allows TF32 (``torch.backends.cuda.matmul.allow_tf32`` or
        ``set_float32_matmul_precision`` not ``"highest"``): the same switch the H100 generic kernels obey. With full-precision fp32 the call
        keeps the PyTorch reference (cuBLAS fp32), as on H100."""
        from miniworld_engine.kernels._dot_precision import tf32_allowed

        return tf32_allowed()

    def _native_cuda_serves(self) -> bool:
        """A hand-written CUDA path serves this layer on the current card without a Triton record: the sm_80 (A100) fused kernels at
        d = 128, n = 4 (``kernels.transition.cuda.fused_sm80``; whole 256-row tiles, bf16 are checked per call). Resolved once, when
        the module is built, by ``dispatch.resolve`` (``native_cuda``)."""
        from miniworld_engine import settings

        card = _dispatch._card_of(None)
        return (
            card is not None
            and card[1] == (8, 0)
            and self.d_hidden == 128
            and self.n == 4
            and settings.current().transition_fused_sm80
            and not settings.current().force_portable_kernels
        )

    def _requires_h100_cuda(self, x: torch.Tensor) -> bool:
        # H100 / B200 have no Triton fallback: a MINIWORLD call no CUDA path serves runs the PyTorch reference. An explicit
        # force_portable_kernels=True (or settings.portable_on_hopper_blackwell) is a comparison run and keeps Triton (as for TriMul).
        return (
            self.implementation == ImplementationType.MINIWORLD
            and x.is_cuda
            and _dispatch.restricted_card(x.device)
        )

    def _generic_sm90a_serves(self, x: torch.Tensor) -> bool:
        """Whether the generic sm_90a kernels take this activation (fp32 / TF32 included)."""
        if not _fused_sm90a_enabled():
            return False
        from miniworld_engine.kernels.transition.cuda import fused_generic_sm90a, fused_generic_sm100a

        # (and the B200 generic kernels, ``fused_generic_sm100a``: the same contract on tcgen05, fp32 on kind::tf32)
        return fused_generic_sm90a.available(
            x, self.expand_a.weight, self.squeeze.weight
        ) or fused_generic_sm100a.available(x, self.expand_a.weight, self.squeeze.weight)

    def _residual_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual-fused Transition: LN, expand-SwiGLU, squeeze with the residual folded into
        its epilogue, and the matching backward.

        On sm_90 at the AF3 pair width (d=128, n=4, bf16, whole 128-row tiles) this runs the
        fused hand-CUDA kernels -- one launch each way instead of three and two. d = 64, 256, 384
        and 512 have their own hand-CUDA builds (``fused_wide_sm90a``), and n = 2 (hidden 2 d) is built for d = 64 / 128 / 256 / 384 / 512.
        Every other width (multiple of 64), hidden width (multiple of 128), any row count and fp32 (TF32 tensor cores) run the generic
        sm_90a kernels (``fused_generic_sm90a``). sm_100 (B200) and
        sm_80 (A100) have their own d=128/n=4 builds (``fused_sm100a``, ``fused_sm80``). Every other
        shape, dtype and architecture keeps the Triton path, which is shape-general. The gates are
        ``fused_sm90a.available`` / ``fused_wide_sm90a.available`` / ``fused_generic_sm90a.available``; they are the kernels' own
        requirements, not a policy. The sm_100 entries take the parameters themselves and cast them outside autograd, so an fp32
        master gets unrounded fp32 gradients; the other paths take the casts in autograd.
        """
        if _fused_sm100a_enabled():
            from miniworld_engine.kernels.transition.cuda import (
                fused_sm100a,
                fused_wide_sm100a,
            )

            pa, pb, ps = self.expand_a.weight, self.expand_b.weight, self.squeeze.weight
            # the B200 kernels take whole 128-row tiles; the rows are independent, so another row count (a pair track at a ragged L)
            # is padded with zero rows and cut back: their dy is zero, so they add nothing to any weight gradient
            d = x.shape[-1] if x.ndim else 0
            rows = x.numel() // d if d else 0
            xs = x
            if rows and rows % fused_sm100a.ROWS:
                xs = torch.nn.functional.pad(x.reshape(rows, d), (0, 0, 0, -rows % fused_sm100a.ROWS))
            for kern, entry in (
                (fused_sm100a, fused_sm100a.transition_fused_sm100a),
                (fused_wide_sm100a, fused_wide_sm100a.transition_wide_sm100a),
            ):
                if kern.available(xs, pa, ps):
                    out = entry(
                        xs,
                        self.ln_in.weight,
                        self.ln_in.bias,
                        pa,
                        pb,
                        ps,
                        self.ln_in.eps,
                    )
                    return out if xs is x else out[:rows].reshape(x.shape)
            from miniworld_engine.kernels.transition.cuda import fused_generic_sm100a

            # every other width / hidden width, fp32 (TF32) and any row count: the generic tcgen05 kernels (no padding)
            if fused_generic_sm100a.available(x, pa, ps):
                return fused_generic_sm100a.transition_generic_sm100a(x, self.ln_in.weight, self.ln_in.bias, pa, pb, ps, self.ln_in.eps)

        if _fused_sm90a_enabled():
            from miniworld_engine.kernels.transition.cuda import transition_sm90a

            # the parameters as they are (fp32 masters included), then the copies cast to the activation dtype below
            out = transition_sm90a(
                x,
                self.ln_in.weight,
                self.ln_in.bias,
                self.expand_a.weight,
                self.expand_b.weight,
                self.squeeze.weight,
                self.ln_in.eps,
                self.n,
            )
            if out is not None:
                return out
            if self._requires_h100_cuda(x):
                return self._torch_forward(x) + x   # no sm_90a build serves this width / dtype: PyTorch, never Triton

        wa = self.expand_a.weight.to(x.dtype)
        wb = self.expand_b.weight.to(x.dtype)
        ws = self.squeeze.weight.to(x.dtype)
        if _fused_sm90a_enabled():
            from miniworld_engine.kernels.transition.cuda import transition_sm90a

            out = transition_sm90a(
                x, self.ln_in.weight, self.ln_in.bias, wa, wb, ws, self.ln_in.eps, self.n
            )
            if out is not None:
                return out

        if _fused_sm80_enabled():
            from miniworld_engine.kernels.transition.cuda import fused_sm80

            if fused_sm80.available(x, wa, ws):
                return fused_sm80.transition_fused_sm80(
                    x, self.ln_in.weight, self.ln_in.bias, wa, wb, ws, self.ln_in.eps
                )

        if self._requires_h100_cuda(x):
            return self._torch_forward(x) + x   # B200 / H100 without a CUDA build for this call: no Triton fallback

        from miniworld_engine.kernels.transition.triton.residual import (
            transition_residual,
        )

        return transition_residual(
            x,
            self.ln_in.weight,
            self.ln_in.bias,
            wa,
            wb,
            ws,
            self.ln_in.eps,
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
