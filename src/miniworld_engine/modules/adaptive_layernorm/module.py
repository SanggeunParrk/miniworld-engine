
# vendored from team-gm psk/benchmark : src/team_gm/modules/layers/adaln.py
import torch
import torch.nn as nn
from jaxtyping import Float

from miniworld_engine._typecheck import typecheck
from miniworld_engine.integrations.h100_master import is_h100
from miniworld_engine.kernels import _capture, adaln_inference, adaln_train
from miniworld_engine.modules.dispatch import (
    KernelBackend,
    call_backend,
    fp32_gate,
    needs_backward,
    resolve_adaptive_layernorm,
    train_gate,
)
from miniworld_engine.modules.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)
from miniworld_engine.modules.primitives import LayerNorm, Linear


class AdaptiveLayerNorm(nn.Module):
    """Adaptive LayerNorm."""

    def __init__(
        self,
        d_hidden: int,
        d_cond: int,
        *,
        implementation: ImplementationType = ImplementationType.PYTORCH,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.implementation = ImplementationType(implementation)
        # 'miniworld' (auto) -> the PORTABLE family (the only fused adaln kernel).
        self._backend = resolve_adaptive_layernorm(self.implementation)
        self._train_gate = train_gate("adaptive_layernorm", self.implementation, self._backend)
        # Constructor-level, like ConditionedTransition: the norm and the scale/bias projections
        # are one numerical unit. The module already followed a post-hoc ``.to(dtype)``; taking it
        # here as well means a caller can build a bf16 module directly instead of building fp32
        # and converting, which is what a parent passing its own dtype down needs.
        self.dtype = dtype
        self.ln_in = nn.LayerNorm(d_hidden, elementwise_affine=False, dtype=dtype)
        # The only norm affine here stays fp32 whatever ``dtype`` the rest runs in, including
        # after a parent's bulk ``.to(torch.bfloat16)``: a bf16 gamma at 1.0 cannot move by one
        # Adam step (see primitives._Fp32ParamsMixin). Every fused path already reads it as fp32.
        self.ln_cond = LayerNorm(d_cond, bias=False)
        self.to_scale = Linear(d_cond, d_hidden, init="gating", dtype=dtype)
        self.to_bias = Linear(d_cond, d_hidden, bias=False, init="zero", dtype=dtype)

    @typecheck
    def forward(
        self,
        x: Float[torch.Tensor, "* d_hidden"],
        cond: Float[torch.Tensor, "* d_cond"],
    ) -> Float[torch.Tensor, "* d_hidden"]:
        """Forward pass. Routes on the resolved internal backend (``_backend``)."""
        return self._adaln_forward(x, cond, with_stream=False)

    def forward_with_stream(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(self(x, cond), x)`` for a caller that adds ``x`` to something later (a residual stream). On the portable training path the
        returned ``x`` is an output of the same autograd node, so its gradient joins the LayerNorm backward's ``dx`` inside the kernel that
        writes it instead of in a separate add of the two paths into ``x``; everywhere else it is ``x`` itself."""
        return self._adaln_forward(x, cond, with_stream=True)

    def forward_with_stream_cond(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(self(x, cond), x, cond)``: like :meth:`forward_with_stream`, and ``cond`` too is returned as an output of the same autograd node, for a
        caller that feeds it to a second consumer (the ConditionedTransition's gate projection). On the portable training path the second
        consumer's gradient into ``cond`` joins the cond LayerNorm backward's ``dcond`` inside the kernel that writes it, instead of in a separate add
        of the two paths; everywhere else the second and third outputs are ``x`` and ``cond`` themselves."""
        return self._adaln_forward(x, cond, with_stream=True, with_cond=True)

    def _cuda_forward(self, x: torch.Tensor, cond: torch.Tensor, streams: bool = False):
        """The H100 hand-CUDA AdaLN: the one entry ``kernels.adaln.cuda.adaln``, which picks the route by mode, dtype and
        shape. ``None`` when it does not serve the call (another dtype, a width it does not tile, ...): the caller runs the PyTorch reference."""
        from miniworld_engine.kernels.adaln import cuda as adaln_cuda

        if self.ln_in.elementwise_affine or self.to_bias.bias is not None:
            return None
        device_type = x.device.type
        grad = needs_backward(self, x, cond)
        if torch.is_autocast_enabled(device_type) and not grad:
            dtype = torch.get_autocast_dtype(device_type)   # inference under autocast: the kernels take the compute dtype
            x, cond = x.to(dtype), cond.to(dtype)
        if x.dtype != cond.dtype or not adaln_cuda.available(x, cond):
            return None
        with torch.autocast(device_type=device_type, enabled=False):
            return adaln_cuda.adaln(
                x, cond, self.ln_cond.weight, self.to_scale.weight, self.to_scale.bias, self.to_bias.weight, self.ln_in.eps, self.ln_cond.eps, streams=streams)

    def _adaln_forward(self, x: torch.Tensor, cond: torch.Tensor, *, with_stream: bool, with_cond: bool = False):
        backend = call_backend(self, self._backend, x, cond)   # a card recorded for inference only: PyTorch for a training call
        if backend in (KernelBackend.PORTABLE, KernelBackend.CUDA) and not x.is_cuda:
            backend = KernelBackend.PYTORCH   # a CPU tensor (a module built on a CPU host keeps the kernel label): the reference
        backend = fp32_gate("adaptive_layernorm", backend, self, x, cond)   # an fp32 call: the card's fp32 rows
        if backend == KernelBackend.CUDA:
            out = self._cuda_forward(x, cond)
            if out is not None:
                return ((out, x, cond) if with_cond else (out, x)) if with_stream else out
            backend = KernelBackend.PYTORCH   # a call the sm_90a kernels do not serve: the reference
        if backend == KernelBackend.PYTORCH:
            # H100 / B200 resolve MINIWORLD to this leg (no Triton fallback there); fp32 master weights under a bf16 activation keep
            # their unrounded fp32 gradients (``h100_master.master_linear``), as the Triton path's own casts did.
            from miniworld_engine.integrations.h100_master import master_linear
            from miniworld_engine.modules import dispatch as _dispatch

            engine_leg = self.implementation == ImplementationType.MINIWORLD and _dispatch.restricted_card(x.device)
            x_norm = self.ln_in(x)
            cond_norm = self.ln_cond(cond)
            scale = master_linear(self.to_scale, cond_norm)
            bias = master_linear(self.to_bias, cond_norm)
            out = torch.sigmoid(scale) * x_norm + bias
            # the engine's own path returned the compute dtype (bf16 under autocast); torch's LayerNorm would hand back fp32 there
            out = out.to(x.dtype) if engine_leg and x.dtype == torch.bfloat16 else out
            return ((out, x, cond) if with_cond else (out, x)) if with_stream else out

        if backend == KernelBackend.PORTABLE:
            # training → save-for-backward autograd path; inference → d-aware fused/materialize.
            #
            # `dispatch.needs_backward` owns the condition -- see its docstring. It was written
            # out here, and ConditionedTransition had its own shorter version, and the two
            # disagreed; neither asked whether gradients were being recorded at all.
            fn = adaln_train if needs_backward(self, x, cond) else adaln_inference
            # AMP does not cast custom-op arguments. Keep the native GEMMs and
            # their saved backward operands in one dtype, with FP32 master params
            # connected through differentiable casts. Norm affine stays FP32.
            device_type = x.device.type
            compute_dtype = (torch.get_autocast_dtype(device_type)
                             if torch.is_autocast_enabled(device_type) else x.dtype)
            master = fn is adaln_train and compute_dtype == torch.bfloat16 and is_h100(x.device)
            params = (self.to_scale.weight, self.to_scale.bias, self.to_bias.weight)
            if master:
                w_scale, b_scale, w_bias = params
            elif fn is adaln_inference and any(p.dtype != compute_dtype for p in params):
                # inference: the casts of the fp32 master weights are made once while the weights are unchanged (re-casting them at
                # every call cost as much as the kernels at a small L), scoped to the CUDA-graph capture (``kernels._capture``)
                w_scale, b_scale, w_bias = _capture.param_forms(
                    f"adaln_portable_{compute_dtype}", params, lambda: tuple(p.to(compute_dtype) for p in params))
            else:
                w_scale, b_scale, w_bias = (p.to(compute_dtype) for p in params)
            stream = with_stream and fn is adaln_train and x.dtype == compute_dtype
            cond_out = with_cond and stream
            with torch.autocast(device_type=device_type, enabled=False):
                out = fn(
                    x.to(compute_dtype),
                    cond.to(compute_dtype),
                    self.ln_cond.weight,
                    w_scale,
                    b_scale,
                    w_bias,
                    self.ln_in.eps,
                    self.ln_cond.eps,
                    *((True, cond_out) if stream else ()),
                )
            if stream:
                return (*out, cond) if with_cond and not cond_out else out
            return ((out, x, cond) if with_cond else (out, x)) if with_stream else out

        raise InvalidImplementationError(self.implementation)


#: The epsilon of :class:`AdaptiveRMSNorm` (and of the q / k RMSNorms beside it): float32's machine epsilon whatever the input dtype.
RMS_EPS = torch.finfo(torch.float32).eps


class AdaptiveRMSNorm(nn.Module):
    """ESMFold2's atom AdaLN: an affine-free RMSNorm of the activation, SiLU (not a LayerNorm) of the conditioning, and the
    identity-centred scale with no bias on either projection:

        out = (1 + W_s silu(c)) * x / sqrt(mean(x^2) + eps) + W_b silu(c)          eps = float32 epsilon

    Parameters ``to_scale.weight`` and ``to_bias.weight`` (both zero-initialised: the identity). The norm runs in float32 and the result
    comes back in ``x``'s dtype.
    """

    def __init__(self, d_hidden: int, d_cond: int, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.to_scale = Linear(d_cond, d_hidden, bias=False, init="zero", dtype=dtype)
        self.to_bias = Linear(d_cond, d_hidden, bias=False, init="zero", dtype=dtype)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        from miniworld_engine.integrations.h100_master import master_linear

        wide = torch.float64 if x.dtype == torch.float64 else torch.float32
        xw = x.to(wide)
        norm = xw * torch.rsqrt(xw.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        context = torch.nn.functional.silu(cond)
        scale = master_linear(self.to_scale, context)
        bias = master_linear(self.to_bias, context)
        return ((scale.to(wide) + 1.0) * norm + bias.to(wide)).to(x.dtype)
