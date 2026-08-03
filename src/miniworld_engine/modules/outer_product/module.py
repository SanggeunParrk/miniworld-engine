"""OuterProduct / OuterProductMean — MSA/single -> pair building blocks (ported from team-gm)."""
import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Bool, Float, Int

from miniworld_engine._typecheck import typecheck
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.primitives import LayerNorm, Linear


class OuterProduct(nn.Module):
    """Outer product single represetation to pair representation.

    Parameters
    ----------
    d_single : int
        Dimension of single representation.
    d_pair : int
        Dimension of pair representation.
    d_hidden : int
        Dimension of hidden layer.

    """

    def __init__(
        self,
        d_single: int,
        d_pair: int,
        d_hidden: int = 32,
    ) -> None:
        super().__init__()

        self.ln_single = nn.LayerNorm(d_single)
        self.to_left = Linear(d_single, d_hidden, bias=False, init="default")
        self.to_right = Linear(d_single, d_hidden, bias=False, init="default")
        self.to_out = Linear(d_hidden * d_hidden, d_pair, bias=True, init="zero")

    @typecheck
    def forward(
        self,
        single: Float[torch.Tensor, "B L d_single"],
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass."""
        single = self.ln_single(single)
        left = self.to_left(single)
        right = self.to_right(single)

        out = torch.einsum("bid,bje->bijde", left, right)
        out = rearrange(out, "B L1 L2 d1 d2 -> B L1 L2 (d1 d2)")
        return self.to_out(out)


class OuterProductMean(nn.Module):
    """Outer product mean of MSA representation to pair representation.

    Normalization uses ``clamp(min=1)`` on the mask count instead of adding a
    small epsilon, following the Boltz approach.  AlphaFold3 and Protenix add
    ``eps`` (1e-3) to the denominator.

    Parameters
    ----------
    d_msa : int
        Dimension of MSA representation.
    d_pair : int
        Dimension of pair representation.
    d_hidden : int
        Dimension of hidden layer.

    """

    def __init__(
        self,
        d_msa: int,
        d_pair: int,
        d_hidden: int = 32,
        *,
        mask_interchain: bool = False,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()

        self.mask_interchain = mask_interchain
        # LN over the MSA feature dim — route to the fused miniworld_engine LN (bf16) under
        # MINIWORLD_KERNELS; a raw nn.LayerNorm runs fp32-native under autocast and,
        # at MSA depth 2048, is the single biggest kernel in the MSA module.
        self.ln_msa = LayerNorm(d_msa, implementation=implementation)
        self.to_left = Linear(d_msa, d_hidden, bias=False)
        self.to_right = Linear(d_msa, d_hidden, bias=False)
        self.to_out = Linear(d_hidden * d_hidden, d_pair, bias=True, init="zero")

    @typecheck
    def forward(
        self,
        msa: Float[torch.Tensor, "B M L d_msa"],
        mask: Bool[torch.Tensor, "B M L"] | None = None,
        token_asym_id: Int[torch.Tensor, "B L"] | None = None,
        residual: Float[torch.Tensor, "B L L d_pair"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass. When ``residual`` (the running pair) is given, ALWAYS returns
        ``residual + OPM(msa)``. This is a CROSS-TENSOR residual — the residual (pair) is a
        DIFFERENT tensor than the input (msa), so unlike the self-residual layers it is passed in
        explicitly by the block (``pair = opm(msa, ..., residual=pair)``) rather than being the
        module's own input; the add is unconditional when ``residual`` is provided (no dropout on
        the OPM branch). ``residual=None`` returns the raw OPM output (standalone / benchmarking)."""
        msa = self.ln_msa(msa)
        left = self.to_left(msa)
        right = self.to_right(msa)

        if mask is None:
            mask = torch.ones(msa.shape[:3], dtype=torch.bool, device=msa.device)

        left = left * mask[..., None]
        right = right * mask[..., None]
        out = torch.einsum("bmid,bmje->bijde", left, right)
        out = rearrange(out, "B L1 L2 D1 D2 -> B L1 L2 (D1 D2)")

        norm = torch.einsum("bmi,bmj->bij", mask.float(), mask.float())
        # norm stays fp32 for count precision (>256 counts aren't bf16-exact), but cast the
        # normalized result back to the projection dtype so to_out runs in native bf16
        # (no autocast in the distogram trunk; a bf16/fp32 mismatch would otherwise crash).
        out = (out / norm.clamp(min=1)[..., None]).to(left.dtype)

        pair = self.to_out(out)
        if self.mask_interchain and token_asym_id is not None:
            same_chain = token_asym_id[:, :, None] == token_asym_id[:, None, :]
            pair = pair * same_chain[..., None].to(pair.dtype)
        return residual + pair if residual is not None else pair
