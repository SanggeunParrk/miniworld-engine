"""PyTorch reference for the ConditionedTransition tail kernel.

Mirrors ``kernels.cond_transition_train`` (``ConditionedTransitionTailFunction``, and
its forward-only twin ``cond_transition_inference_dispatch``), i.e. the post-AdaLN tail
of AF3 Algorithm 25. Same weights-as-args order, same layout, same single output::

    a     = x @ expand_a_weight.T                  # (M, n*D)
    b     = x @ expand_b_weight.T                  # (M, n*D)
    h     = silu(a) * b = a * sigmoid(a) * b       # SwiGLU, (M, n*D)
    out   = h @ squeeze_weight.T                   # (M, D)
    scale = cond @ to_scale_weight.T + to_scale_bias   # (M, D)
    y     = sigmoid(scale) * out                   # conditioning gate, (M, D)

The conditioning gate is an ELEMENTWISE per-feature multiplier: ``cond`` is projected
d_cond -> d_hidden by the biased ``to_scale`` linear, squashed by a sigmoid, and applied
to the squeeze output — it never touches ``a``/``b``.

Where the AdaLN sits: OUTSIDE this op. ``x`` arrives ALREADY adaptively normalised —
``modules.ConditionedTransition`` runs ``x = ada_ln_in(x, cond)`` (Alg. 25 step 1) before
calling the kernel, and the same ``cond`` is then reused here for the output gate. This
reference deliberately stops at the same boundary as the kernel, so it owns no AdaLN
parameters.

Weight convention mirrors ``nn.Linear`` (``weight`` is ``(out, in)``). Parameters default
to fp32 because that is the precision the kernel and its bench run at (fp32 storage,
TF32 tensor cores); pass ``dtype=torch.bfloat16`` to compare against a bf16 kernel run.

Provided as an ``nn.Module`` (:class:`ConditionedTransitionReference`, owns every weight
and the gate bias as parameters) so a kernel can be checked on both forward output and
backward gradients::

    ref = ConditionedTransitionReference(d_hidden=128, d_cond=384, n=2).cuda()
    y = ref(x_normed, cond)                                    # reference forward
    yk = cond_transition_train(                                # kernel forward
        x_normed, cond, ref.expand_a_weight, ref.expand_b_weight,
        ref.squeeze_weight, ref.to_scale_weight, ref.to_scale_bias,
    )
    y.sum().backward()                                         # -> ref.*.grad, x/cond grads

A plain functional form (:func:`conditioned_transition_pytorch`) is kept for callers that
already hold the weight tensors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conditioned_transition_pytorch(
    x: torch.Tensor,  # (..., D) — already AdaLN-normalised
    cond: torch.Tensor,  # (..., DC)
    expand_a_weight: torch.Tensor,  # (n*D, D)
    expand_b_weight: torch.Tensor,  # (n*D, D)
    squeeze_weight: torch.Tensor,  # (D, n*D)
    to_scale_weight: torch.Tensor,  # (D, DC)
    to_scale_bias: torch.Tensor,  # (D,)
    n: int | None = None,
) -> torch.Tensor:
    """Compute ``sigmoid(to_scale(cond)) * squeeze(silu(a) * b)``. Returns ``(..., D)``.

    ``n`` is accepted for parity with the whole-op signature and is unused: the expansion
    factor is implied by the weight shapes.
    """
    a = F.linear(x, expand_a_weight)
    b = F.linear(x, expand_b_weight)
    h = F.silu(a) * b
    out = F.linear(h, squeeze_weight)
    scale = F.linear(cond, to_scale_weight, to_scale_bias)
    return torch.sigmoid(scale) * out


class ConditionedTransitionReference(nn.Module):
    """nn.Module reference for the ConditionedTransition tail (forward + backward truth).

    ``forward`` takes the ALREADY AdaLN-normalised ``x`` plus the raw ``cond`` — the AdaLN
    is the caller's (module's) job, exactly as for the kernel.
    """

    def __init__(
        self,
        d_hidden: int = 128,
        d_cond: int = 384,
        n: int = 2,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.d_cond = d_cond
        self.n = n
        nd = n * d_hidden

        def w(out_features: int, in_features: int) -> nn.Parameter:
            scale = in_features**-0.5
            t = torch.randn(out_features, in_features, device=device, dtype=dtype)
            return nn.Parameter(t * scale)

        self.expand_a_weight = w(nd, d_hidden)
        self.expand_b_weight = w(nd, d_hidden)
        self.squeeze_weight = w(d_hidden, nd)
        self.to_scale_weight = w(d_hidden, d_cond)
        self.to_scale_bias = nn.Parameter(
            torch.zeros(d_hidden, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass; differentiable in ``x``, ``cond`` and every weight/bias."""
        return conditioned_transition_pytorch(
            x,
            cond,
            self.expand_a_weight,
            self.expand_b_weight,
            self.squeeze_weight,
            self.to_scale_weight,
            self.to_scale_bias,
            self.n,
        )
