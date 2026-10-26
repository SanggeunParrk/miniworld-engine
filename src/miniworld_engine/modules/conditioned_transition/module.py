"""ConditionedTransition: AdaLN -> SwiGLU expand/squeeze + sigmoid conditioning gate.

Mirrors team-gm's ConditionedTransition (AF3 Algorithm 25). The MODULE owns the input
AdaLN (``ada_ln_in``); the fused KERNEL is the post-AdaLN tail and receives the already-
normalized activation. ``forward`` returns the residual output ``x_in + y``; ``delta`` returns
the update ``y`` alone. Given the RAW activation ``x_in`` and conditioning ``cond``::

    x = ada_ln_in(x_in, cond)              # AdaLN (module; step 1)
    a = x @ Wa^T ; b = x @ Wb^T            # expand_a, expand_b (no bias)
    h = silu(a) * b                        # SwiGLU
    out = h @ Ws^T                         # squeeze (no bias)
    scale = cond @ Wsc^T + b_sc            # to_scale (with bias)
    y = sigmoid(scale) * out               # output gate (delta)
    return x_in + y                        # residual (forward)

fp32 weights/activations with TF32 tensor-core matmuls. Two implementations:

  - PYTORCH: the readable reference (also the autograd path the kernel is checked against).
  - PORTABLE:  inference -> d-aware fused/composed kernel; training -> autograd Function
             (forward saves the minimum, custom backward).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from miniworld_engine import kernels
from miniworld_engine.kernels import _capture
from miniworld_engine._typecheck import typecheck
from miniworld_engine.autotune.shape_key import length_of
from miniworld_engine.integrations.h100_master import is_h100
from miniworld_engine.modules.adaptive_layernorm.module import AdaptiveLayerNorm
from miniworld_engine.modules.dispatch import (
    KernelBackend,
    call_backend,
    needs_backward,
    resolve_conditioned_transition,
    train_gate,
)
from miniworld_engine.modules.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)
from miniworld_engine.modules.primitives import Linear


class ConditionedTransition(nn.Module):
    """Post-AdaLN ConditionedTransition tail (SwiGLU + sigmoid conditioning gate).

    Parameters
    ----------
    d_hidden : int
        Feature dimension of ``x`` (and of the output). K = D = d_hidden.
    d_cond : int
        Feature dimension of the conditioning input ``cond`` (DC).
    n : int
        SwiGLU expansion factor (ND = n * d_hidden).
    implementation : ImplementationType
        PYTORCH (reference) or PORTABLE (fused/composed inference, custom-bwd training).
    """

    def __init__(
        self,
        # The atom side's widths -- AlphaFold-3's c_atom, which the model's `atom_dit` passes as both.
        # 128/384 was neither of the model's two combinations (768/384 token, 128/128 atom); a
        # default that names no real configuration is a shape only the tests and the driver ever
        # build, and the driver copied it.
        d_hidden: int = 128,
        d_cond: int = 128,
        n: int = 2,
        implementation: ImplementationType = ImplementationType.PYTORCH,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.d_cond = d_cond
        self.n = n
        self.implementation = ImplementationType(implementation)
        # 'miniworld' (auto) -> the PORTABLE fused family (inference d_hidden
        # sub-dispatch lives in the kernel); CUEQUIVARIANCE shares that path.
        self._backend = resolve_conditioned_transition(self.implementation)
        self._train_gate = train_gate("conditioned_transition", self.implementation, self._backend)

        # AF3 Algorithm 25 step 1: the conditioned transition STARTS with an AdaLN of the
        # input (matches ESMFold2 + pre-migration team-gm). The AdaLN is applied by this
        # MODULE; the fused KERNEL below stays the post-AdaLN tail (it receives the already-
        # normalized activation). Checkpoint keys: transition.ada_ln_in.* + the flat tail.
        self.ada_ln_in = AdaptiveLayerNorm(
            d_hidden=d_hidden, d_cond=d_cond, implementation=implementation, dtype=dtype
        )
        # dtype is a CONSTRUCTOR argument here, not a per-call one: unlike the attention core in
        # AugmentedAttentionPairBias, there is no sub-part of this module a caller would want at a
        # different precision from the rest -- the AdaLN, the SwiGLU expand/squeeze and the gate
        # are one numerical unit, and the fused kernel allocates its intermediates as ``x.dtype``
        # (train_fused.py), so a per-call split would only mean a cast at the door.
        #
        # These were pinned to fp32, which is why a bf16 input died with "expected mat1 and mat2 to
        # have the same dtype" no matter what the caller did with .to(). fp32 stays the DEFAULT so
        # existing callers and checkpoints are unaffected; bf16 is now reachable by asking for it.
        self.dtype = dtype
        self.expand_a = Linear(d_hidden, d_hidden * n, bias=False, init="relu", dtype=dtype)
        self.expand_b = Linear(d_hidden, d_hidden * n, bias=False, init="relu", dtype=dtype)
        self.squeeze = Linear(d_hidden * n, d_hidden, bias=False, init="zero", dtype=dtype)
        self.to_scale = Linear(d_cond, d_hidden, bias=True, init="default", dtype=dtype)

    def _reference(
        self,
        x: Float[torch.Tensor, "*"],
        cond: Float[torch.Tensor, "*"],
    ) -> Float[torch.Tensor, "*"]:
        from miniworld_engine.integrations.h100_master import master_linear

        a = master_linear(self.expand_a, x)
        b = master_linear(self.expand_b, x)
        h = F.silu(a) * b
        out = master_linear(self.squeeze, h)
        scale = master_linear(self.to_scale, cond)
        return torch.sigmoid(scale) * out

    @typecheck
    def forward(
        self,
        x: Float[torch.Tensor, "*"],
        cond: Float[torch.Tensor, "*"],
    ) -> Float[torch.Tensor, "*"]:
        """Forward pass. ALWAYS returns the residual output ``x + delta(x, cond)`` (the residual is
        this module's own input). A caller that composes the update some other way (a
        magnitude-preserving sum, a released checkpoint's own residual) takes :meth:`delta`."""
        return self._update(x, cond, residual=True)

    @typecheck
    def delta(
        self,
        x: Float[torch.Tensor, "*"],
        cond: Float[torch.Tensor, "*"],
    ) -> Float[torch.Tensor, "*"]:
        """The update alone (no residual). ``x`` is the RAW residual-stream activation, ``cond`` the
        conditioning signal.

        The module applies AdaLN (``ada_ln_in``) first (AF3 Alg. 25 step 1), then the
        SwiGLU + sigmoid-gate tail. Leading dims are flattened to (M, d_hidden) for the
        kernels, which operate on the already-normalized activation (the kernel is the
        post-AdaLN tail; the AdaLN lives in this module).
        """
        return self._update(x, cond, residual=False)

    def _update(self, x: torch.Tensor, cond: torch.Tensor, *, residual: bool) -> torch.Tensor:
        """``delta`` (``residual=False``) or ``x + delta`` (``residual=True``). The portable inference path folds the residual add into the
        squeeze+gate epilogue of its kernel (one launch fewer); every other path adds it here."""
        x_in = x
        if residual:
            # x_in: the stream, whose gradient the LayerNorm backward adds in; cond: handed on to the gate projection, whose gradient joins dcond there
            x, x_in, cond = self.ada_ln_in.forward_with_stream_cond(x_in, cond)
        else:
            x = self.ada_ln_in(x, cond)
        backend = call_backend(self, self._backend, x, cond)   # a card recorded for inference only: PyTorch for a training call
        if backend == KernelBackend.PORTABLE and not x.is_cuda:
            backend = KernelBackend.PYTORCH   # a CPU tensor (a module built on a CPU host keeps the Triton label): the reference
        if backend == KernelBackend.PYTORCH:
            y = self._reference(x, cond)
            return x_in + y if residual else y

        if backend in {KernelBackend.PORTABLE, KernelBackend.CUEQUIVARIANCE}:
            d = x.shape[-1]
            # L for the autotune shape key, read HERE because this is the last place that has it:
            # `x` is still (..., A, d), so `length_of` gives the ATOM count A. After the reshape
            # below the kernels only ever see M = B*A, which equals A only when B == 1 -- so the
            # kernels cannot recover it and it is threaded down as an explicit argument instead.
            length = length_of(x.shape)
            # AMP can produce BF16 activations while master weights remain FP32.
            # Native Triton dot operands must agree, and backward must save tensors
            # in that same compute dtype. Cast through autograd, never mutate params.
            device_type = x.device.type
            compute_dtype = (torch.get_autocast_dtype(device_type)
                             if torch.is_autocast_enabled(device_type)
                             else self.expand_a.weight.dtype)
            x2 = x.reshape(-1, d).to(compute_dtype)
            cond2 = cond.reshape(-1, cond.shape[-1]).to(compute_dtype)
            training = needs_backward(self, x2, cond2)
            master = training and compute_dtype == torch.bfloat16 and is_h100(x.device)
            params = (self.expand_a.weight, self.expand_b.weight, self.squeeze.weight, self.to_scale.weight,
                      self.to_scale.bias)
            if master:
                wa, wb, ws, wsc, bsc = params
            elif not training and any(p.dtype != compute_dtype for p in params):
                # inference: the casts of the fp32 master weights are made once while the weights are unchanged (about a sixth
                # of a small block's time went to re-casting them at every call), scoped to the CUDA-graph capture
                wa, wb, ws, wsc, bsc = _capture.param_forms(
                    f"ct_portable_{compute_dtype}", params, lambda: tuple(p.to(compute_dtype) for p in params))
            else:
                wa, wb, ws, wsc, bsc = (p.to(compute_dtype) for p in params)
            # `x2.requires_grad` alone missed the conditioning tensor AND the projection
            # weights, which are parameters -- so a detached x with live weights took the
            # inference path and the gradient vanished silently (the kernels are `@opaque`, so
            # there is no grad_fn and no error). `dispatch.needs_backward` is the one condition
            # both this and AdaptiveLayerNorm now ask.
            with torch.autocast(device_type=device_type, enabled=False):
                fold = residual and x_in.dtype == compute_dtype and x_in.shape == x.shape
                if training:
                    y = kernels.cond_transition_train(
                        x2, cond2,
                        wa, wb, ws, wsc, bsc,
                        length,
                        x_in.reshape(-1, d) if fold else None,   # the residual add rides in the gate pass of the forward
                    )
                    if fold:
                        return y.reshape(*x.shape[:-1], self.d_hidden)
                else:
                    y = kernels.cond_transition_inference_dispatch(
                        x2, cond2,
                        wa, wb, ws, wsc, bsc,
                        length,
                        residual=x_in.reshape(-1, d) if fold else None,
                    )
                    if fold:
                        return y.reshape(*x.shape[:-1], self.d_hidden)
            y = y.reshape(*x.shape[:-1], self.d_hidden)
            return x_in + y if residual else y

        raise InvalidImplementationError(self.implementation)
