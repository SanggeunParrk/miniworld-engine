"""ConditionedTransition tail (post-AdaLN): SwiGLU expand/squeeze + sigmoid conditioning gate.

Mirrors team-gm's ConditionedTransition, but only the part AFTER the AdaLN: the AdaLN is a
separately-optimized op and is OUT OF SCOPE here. This module takes the *already-AdaLN'd*
activation ``x`` and the conditioning ``cond`` and computes::

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

from miniworld_kernels import kernels
from miniworld_kernels._typecheck import typecheck
from miniworld_kernels.modules.dispatch import (
    KernelBackend,
    resolve_conditioned_transition,
)
from miniworld_kernels.modules.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)
from miniworld_kernels.modules.primitives import Linear


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
    ) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.d_cond = d_cond
        self.n = n
        self.implementation = ImplementationType(implementation)
        # 'miniworld' (auto) -> the TRITON fused family (inference d_hidden
        # sub-dispatch lives in the kernel); CUEQUIVARIANCE shares that path.
        self._backend = resolve_conditioned_transition(self.implementation)

        self.expand_a = Linear(d_hidden, d_hidden * n, bias=False, init="glorot", dtype=torch.float32)
        self.expand_b = Linear(d_hidden, d_hidden * n, bias=False, init="glorot", dtype=torch.float32)
        self.squeeze = Linear(d_hidden * n, d_hidden, bias=False, init="zero", dtype=torch.float32)
        self.to_scale = Linear(d_cond, d_hidden, bias=True, init="default", dtype=torch.float32)

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
        """Forward pass. ``x`` is the AdaLN output, ``cond`` the conditioning signal.

        Flattens leading dims to (M, d_hidden) for the kernels; AdaLN is out of scope.
        """
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
