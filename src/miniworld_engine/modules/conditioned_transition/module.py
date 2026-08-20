"""ConditionedTransition: AdaLN -> SwiGLU expand/squeeze + sigmoid conditioning gate.

Mirrors team-gm's ConditionedTransition (AF3 Algorithm 25). The MODULE owns the input
AdaLN (``ada_ln_in``); the fused KERNEL is the post-AdaLN tail and receives the already-
normalized activation. Given the RAW activation ``x`` and conditioning ``cond``::

    x = ada_ln_in(x, cond)                 # AdaLN (module; step 1)
    a = x @ Wa^T ; b = x @ Wb^T            # expand_a, expand_b (no bias)
    h = silu(a) * b                        # SwiGLU
    out = h @ Ws^T                         # squeeze (no bias)
    scale = cond @ Wsc^T + b_sc            # to_scale (with bias)
    y = sigmoid(scale) * out               # output gate

fp32 weights/activations with TF32 tensor-core matmuls. Two implementations:

  - PYTORCH: the readable reference (also the autograd path the kernel is checked against).
  - TRITON:  inference -> d-aware fused/composed kernel; training -> autograd Function
             (forward saves the minimum, custom backward).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from miniworld_engine import kernels
from miniworld_engine._typecheck import typecheck
from miniworld_engine.modules.dispatch import (
    KernelBackend,
    resolve_conditioned_transition,
)
from miniworld_engine.modules.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)
from miniworld_engine.modules.primitives import Linear
from ..adaptive_layernorm.module import AdaptiveLayerNorm


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
        PYTORCH (reference) or TRITON (fused/composed inference, custom-bwd training).
    """

    def __init__(
        self,
        d_hidden: int = 128,
        d_cond: int = 384,
        n: int = 2,
        implementation: ImplementationType = ImplementationType.PYTORCH,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.d_cond = d_cond
        self.n = n
        self.implementation = ImplementationType(implementation)
        # 'miniworld' (auto) -> the TRITON fused family (inference d_hidden
        # sub-dispatch lives in the kernel); CUEQUIVARIANCE shares that path.
        self._backend = resolve_conditioned_transition(self.implementation)

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
        a = self.expand_a(x)
        b = self.expand_b(x)
        h = F.silu(a) * b
        out = self.squeeze(h)
        scale = self.to_scale(cond)
        return torch.sigmoid(scale) * out

    @typecheck
    def forward(
        self,
        x: Float[torch.Tensor, "*"],
        cond: Float[torch.Tensor, "*"],
    ) -> Float[torch.Tensor, "*"]:
        """Forward pass. ``x`` is the RAW residual-stream activation, ``cond`` the
        conditioning signal.

        The module applies AdaLN (``ada_ln_in``) first (AF3 Alg. 25 step 1), then the
        SwiGLU + sigmoid-gate tail. Leading dims are flattened to (M, d_hidden) for the
        kernels, which operate on the already-normalized activation (the kernel is the
        post-AdaLN tail; the AdaLN lives in this module).
        """
        x = self.ada_ln_in(x, cond)
        if self._backend == KernelBackend.PYTORCH:
            return self._reference(x, cond)

        if self._backend in {KernelBackend.TRITON, KernelBackend.CUEQUIVARIANCE}:
            d = x.shape[-1]
            x2 = x.reshape(-1, d)
            cond2 = cond.reshape(-1, cond.shape[-1])
            if self.training or x2.requires_grad:
                y = kernels.cond_transition_train(
                    x2, cond2,
                    self.expand_a.weight, self.expand_b.weight, self.squeeze.weight,
                    self.to_scale.weight, self.to_scale.bias,
                )
            else:
                y = kernels.cond_transition_inference_dispatch(
                    x2, cond2,
                    self.expand_a.weight, self.expand_b.weight, self.squeeze.weight,
                    self.to_scale.weight, self.to_scale.bias,
                )
            return y.reshape(*x.shape[:-1], self.d_hidden)

        raise InvalidImplementationError(self.implementation)
