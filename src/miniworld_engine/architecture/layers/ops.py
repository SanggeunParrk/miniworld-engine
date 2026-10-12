"""Additional operations."""

import math

import torch
from jaxtyping import Float

from miniworld_engine.architecture.typing import typecheck


@typecheck
def mp_sum(
    a: Float[torch.Tensor, "*"],
    b: Float[torch.Tensor, "*"],
    t: float = 0.3,
) -> Float[torch.Tensor, "*"]:
    """Magnitude-preserving weighted sum (EDM2 ``mp_sum``, arXiv:2312.02696 §2.5).

    A plain residual ``a + b`` of two unit-variance, uncorrelated signals has
    variance 2 -> magnitude grows ~sqrt(2) per block. This returns
    ``lerp(a, b, t) / sqrt((1-t)^2 + t^2)``, which keeps unit variance. ``t`` is
    the residual-branch weight (EDM2 used 0.3 in enc/dec blocks).
    """
    return torch.lerp(a, b, t) / math.sqrt((1.0 - t) ** 2 + t**2)


@typecheck
def apply_pairwise_rotation(
    x: Float[torch.Tensor, "* d"],
    theta: Float[torch.Tensor, "* d_half"],
) -> Float[torch.Tensor, "* d"]:
    """Rotation modulation: rotate consecutive 2D pairs by learned angles.

    Partitions the last (``d``) axis into ``d/2`` pairs ``(x_2i, x_2i+1)`` and
    applies a 2x2 Givens rotation of angle ``theta_i`` to each (RoPE-style, but
    the angles are predicted from the conditioning vector). Orthogonal ->
    magnitude-preserving (``||Rx|| = ||x||``). Ref: arXiv:2505.19122, Eq. 6.
    """
    cos, sin = torch.cos(theta), torch.sin(theta)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out = torch.stack((cos * x_even - sin * x_odd, sin * x_even + cos * x_odd), dim=-1)
    return out.flatten(-2)


@typecheck
def sigmoid_gate(
    gate: Float[torch.Tensor, "*"],
    x: Float[torch.Tensor, "*"],
) -> Float[torch.Tensor, "*"]:
    """Sigmoid gating mechanism."""
    return torch.sigmoid(gate) * x


@typecheck
def swish_gate(
    gate: Float[torch.Tensor, "*"],
    x: Float[torch.Tensor, "*"],
) -> Float[torch.Tensor, "*"]:
    """Swish gating mechanism."""
    return gate * torch.sigmoid(gate) * x


# E[silu(z)^2] for z~N(0,1) => std ~= 0.596. swish_gate(a,b)=silu(a)*b with a,b
# i.i.d. unit-variance has the same std (b is unit-variance, zero-mean,
# independent), so the magnitude-preserving constant is the same as EDM2's SiLU.
_MP_SWISH_SCALE = 0.596


@typecheck
def mp_swish_gate(
    gate: Float[torch.Tensor, "*"],
    x: Float[torch.Tensor, "*"],
) -> Float[torch.Tensor, "*"]:
    """Magnitude-preserving SwiGLU (EDM2 §2.5 MP-SiLU applied to swish_gate)."""
    return swish_gate(gate, x) / _MP_SWISH_SCALE
