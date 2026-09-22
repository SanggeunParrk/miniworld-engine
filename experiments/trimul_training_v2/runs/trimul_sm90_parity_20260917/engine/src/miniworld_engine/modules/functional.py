# vendored from team-gm psk/benchmark : src/team_gm/modules/layers/ops.py
"""Additional operations."""

import torch
from jaxtyping import Float

from miniworld_engine._typecheck import typecheck


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
