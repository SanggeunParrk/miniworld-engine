import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float

from miniworld_engine._typecheck import typecheck
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.functional import sigmoid_gate
from miniworld_engine.modules.primitives import Dropout, LayerNorm, Linear


class MSAPairWeightedAveraging(nn.Module):
    """MSA pair-weighted averaging.

    Parameters
    ----------
    d_msa : int
        Dimension of MSA representation.
    d_pair : int
        Dimension of pair representation.
    n_head : int
        Number of heads.
    d_hidden : int, optional
        Per-head hidden dimension. Defaults to ``d_msa // n_head``.

    """

    def __init__(
        self,
        d_msa: int,
        d_pair: int,
        n_head: int = 8,
        d_hidden: int | None = None,
        *,
        implementation: ImplementationType = ImplementationType.PYTORCH,
        p_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        # This layer ALWAYS applies the residual: msa + drop_msa(pwa(msa, pair)). The residual is
        # UNCONDITIONAL (domain standard); the row-broadcast dropout (drop_msa, broadcast_dim=1) is
        # OPTIONAL via p_drop and active only in training. Applied EXPLICITLY (team-gm layer, not
        # kernel-fused). To disable the residual, edit _ADD_RESIDUAL in forward().
        self.drop_msa = Dropout(broadcast_dim=1, p_drop=p_drop)

        if d_hidden is None:
            d_hidden = d_msa // n_head

        # Fused miniworld_engine LN (bf16) under MINIWORLD_ENGINE — raw nn.LayerNorm is fp32
        # native under autocast and dominates the MSA module at depth 2048.
        self.ln_msa = LayerNorm(d_msa, implementation=implementation)
        self.to_value = Linear(d_msa, d_hidden * n_head, bias=False, init="glorot")

        self.ln_pair = LayerNorm(d_pair, implementation=implementation)
        self.to_bias = Linear(d_pair, n_head, bias=False, init="default")

        self.to_gate = Linear(d_msa, d_hidden * n_head, bias=False, init="gating")
        self.to_out = Linear(d_hidden * n_head, d_msa, bias=False, init="zero")

    @typecheck
    def forward(
        self,
        msa: Float[torch.Tensor, "B M L d_msa"],
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B M L d_msa"]:
        """Forward pass. ALWAYS returns the residual output msa + drop_msa(pwa(msa, pair)) — the
        residual is UNCONDITIONAL (domain standard, explicit add) and drop_msa is optional (p_drop,
        training only). Edit _ADD_RESIDUAL to disable the residual."""
        _ADD_RESIDUAL = True  # unconditional residual (explicit add). Edit to False to disable.
        msa_res = msa  # residual == the ORIGINAL input (before ln_msa rebinds `msa`)
        msa = self.ln_msa(msa)
        value = self.to_value(msa)
        value = rearrange(value, "B M L (H D) -> B M L H D", H=self.n_head)

        pair = self.ln_pair(pair)
        bias = self.to_bias(pair)
        bias = rearrange(bias, "B L1 L2 H -> B H L1 L2")

        if mask is not None:
            bias = bias.masked_fill(
                ~mask[:, None, None, :], torch.finfo(bias.dtype).min
            )

        attention = F.softmax(bias, dim=-1)
        out = torch.einsum("bhij,bmjhd->bmihd", attention, value)

        gate = self.to_gate(msa)
        out = rearrange(out, "B M L H D -> B M L (H D)")
        out = sigmoid_gate(gate, out)
        out = self.drop_msa(self.to_out(out))
        return msa_res + out if _ADD_RESIDUAL else out
