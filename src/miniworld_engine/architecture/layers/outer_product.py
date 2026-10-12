import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Bool, Float, Int

from miniworld_engine.architecture.exceptions import ImplementationType
from miniworld_engine.architecture.primitives import LayerNorm, Linear
from miniworld_engine.architecture.typing import typecheck


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
    normalize_before_proj : bool
        Where the ``1 / n_valid`` divide sits relative to ``to_out``. ``True``
        (default) computes ``to_out(outer / n)``, leaving ``to_out``'s bias
        unscaled. ``False`` computes ``to_out(outer) / n``, which also scales
        the bias by ``1 / n`` — required by ESMFold2 checkpoints, which were
        trained that way.
    row_chunk : int
        Rows of the first token axis to process at a time at inference. The
        outer product materialises ``[B, L, L, d_hidden**2]`` before ``to_out``
        knocks it down to ``d_pair``: at ``d_hidden=32`` that is 1024 floats per
        pair cell, so ``L=2048`` costs ~16 GiB in a single allocation, the
        largest tensor in the trunk. Output row ``i`` depends only on
        ``left[..., i]`` and all of ``right``, so streaming over ``i`` and
        projecting inside the loop is exact. Set ``0`` to disable.

    """

    def __init__(
        self,
        d_msa: int,
        d_pair: int,
        d_hidden: int = 32,
        *,
        mask_interchain: bool = False,
        normalize_before_proj: bool = True,
        row_chunk: int = 256,
        implementation: ImplementationType = ImplementationType.MINIWORLD_ENGINE,
    ) -> None:
        super().__init__()

        self.mask_interchain = mask_interchain
        self.normalize_before_proj = normalize_before_proj
        self.row_chunk = row_chunk
        # LN over the MSA feature dim — route to the fused miniworld_engine LN
        # (bf16) under MINIWORLD_ENGINE; a raw nn.LayerNorm runs fp32-native under
        # autocast and, at MSA depth 2048, is the single biggest kernel in the MSA
        # module.
        self.ln_msa = LayerNorm(d_msa, implementation=implementation)
        self.to_left = Linear(d_msa, d_hidden, bias=False)
        self.to_right = Linear(d_msa, d_hidden, bias=False)
        self.to_out = Linear(d_hidden * d_hidden, d_pair, bias=True, init="zero")

    def _project(
        self, out: torch.Tensor, norm: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """Normalise and project one block of outer products."""
        if self.normalize_before_proj:
            # norm stays fp32 for count precision (>256 counts aren't bf16-exact),
            # but cast the normalized result back to the projection dtype so to_out
            # runs in native bf16 (no autocast in the distogram trunk; a dtype
            # mismatch would otherwise crash).
            return self.to_out((out / norm).to(dtype))
        # Divide after the projection, so to_out's bias is scaled by 1/n too.
        return (self.to_out(out.to(dtype)) / norm).to(dtype)

    @typecheck
    def forward(
        self,
        msa: Float[torch.Tensor, "B M L d_msa"],
        mask: Bool[torch.Tensor, "B M L"] | None = None,
        token_asym_id: Int[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass."""
        msa = self.ln_msa(msa)
        left = self.to_left(msa)
        right = self.to_right(msa)

        if mask is None:
            mask = torch.ones(msa.shape[:3], dtype=torch.bool, device=msa.device)

        left = left * mask[..., None]
        right = right * mask[..., None]

        mask_f = mask.float()
        norm = torch.einsum("bmi,bmj->bij", mask_f, mask_f).clamp(min=1)[..., None]

        length = left.shape[2]
        # Chunk only at inference: under autograd every chunk's inputs are saved
        # anyway, so the peak barely moves while the graph gets L/chunk times
        # more nodes.
        chunk = self.row_chunk if self.row_chunk and not self.training else length
        if chunk >= length:
            out = rearrange(
                torch.einsum("bmid,bmje->bijde", left, right),
                "B L1 L2 D1 D2 -> B L1 L2 (D1 D2)",
            )
            pair = self._project(out, norm, left.dtype)
        else:
            blocks = []
            for start in range(0, length, chunk):
                stop = min(start + chunk, length)
                block = rearrange(
                    torch.einsum("bmid,bmje->bijde", left[:, :, start:stop], right),
                    "B L1 L2 D1 D2 -> B L1 L2 (D1 D2)",
                )
                blocks.append(self._project(block, norm[:, start:stop], left.dtype))
            pair = torch.cat(blocks, dim=1)

        if self.mask_interchain and token_asym_id is not None:
            same_chain = token_asym_id[:, :, None] == token_asym_id[:, None, :]
            pair = pair * same_chain[..., None].to(pair.dtype)
        return pair
