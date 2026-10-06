"""The AF3 atom transformer block (Alg. 23 with the block-local attention of Alg. 24): 32 queries x 128 keys per window.

    a = a + AttentionPairBias(a, s, z)     a = a + ConditionedTransition(a, s)

Atom ``i`` belongs to query window ``w = i // 32`` and attends to the 128 atoms ``[32 w - 48, 32 w + 80)`` (the window clipped to
``[0, N)``), with a per-window pair bias. The pair tensor is the *trunked* atom pair ``[B, nwin, 32, 128, d_pair]``,
``pair[b, w, i, j] = z[b, 32 w + i, 32 w - 48 + j]`` with ``nwin = ceil(N / 32)`` (:func:`to_windows` makes it from a dense pair;
entries outside ``[0, N)`` are never read). The block has the parameters of :class:`~miniworld_engine.modules.dit.DiTBlock` at atom
widths (d_single = d_cond = 128, d_pair = 16, 4 heads, transition n = 2), so the dense and the local block share a checkpoint layout;
only the attention differs. With ``cross_attention=True`` (AF3's / Protenix's atom blocks) the keys and values are projected from a second
AdaLN of the already normalised atoms, ``attention.ada_ln_kv`` (four more parameters). ``single`` and ``cond`` carry the augmentation axis ``[A, B, N, d]``, the pair does not.

The reference path below is plain PyTorch (the window gather by ``unfold``); on a B200 ``integrations/local_dit`` serves bf16 calls
with the sm_100a kernels of ``kernels/augmented_attention/cuda/sm100_atom_local``, and fp32 calls (fp32 parameters) with their TF32 twins.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from miniworld_engine.integrations import local_dit as _local
from miniworld_engine.modules.adaptive_layernorm import AdaptiveLayerNorm
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

QUERIES, KEYS, KEY_OFFSET = 32, 128, 48


def windows(n: int) -> int:
    return (n + QUERIES - 1) // QUERIES


def _key_windows(t: torch.Tensor, n: int, value: float = 0.0) -> torch.Tensor:
    """[..., N, C] -> [..., nwin, KEYS, C]: the key window of every query window (out-of-range positions are ``value``)."""
    nwin = windows(n)
    padded = F.pad(t, (0, 0, KEY_OFFSET, nwin * QUERIES + KEYS - QUERIES - KEY_OFFSET - n), value=value)
    return padded.unfold(-2, KEYS, QUERIES).transpose(-1, -2)


def to_windows(pair: torch.Tensor) -> torch.Tensor:
    """Dense pair [B, N, N, d] -> trunked pair [B, nwin, 32, 128, d] (positions outside [0, N) are zero)."""
    n = pair.shape[1]
    nwin = windows(n)
    query = torch.arange(nwin * QUERIES, device=pair.device).view(nwin, QUERIES)
    key = torch.arange(nwin, device=pair.device)[:, None] * QUERIES - KEY_OFFSET + torch.arange(KEYS, device=pair.device)[None]
    valid = (query < n)[:, :, None] & ((key >= 0) & (key < n))[:, None, :]
    out = pair[:, query.clamp(max=n - 1)[:, :, None], key.clamp(0, n - 1)[:, None, :]]
    return out * valid[None, :, :, :, None].to(pair.dtype)


class LocalDiTBlock(DiTBlock):
    """AF3 atom transformer block: block-local pair-bias attention, then the conditioned transition."""

    def __init__(
        self,
        d_single: int = 128,
        d_cond: int = 128,
        d_pair: int = 16,
        n_head: int = 4,
        n: int = 2,
        *,
        cross_attention: bool = False,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__(d_single, d_cond, d_pair, n_head, n, implementation=implementation)
        self.cross_attention = cross_attention
        if cross_attention:
            self.attention.ada_ln_kv = AdaptiveLayerNorm(d_single, d_cond, implementation=implementation)

    def attention_delta(self, single, cond, pair, mask=None) -> torch.Tensor:
        """The attention's update (no residual), PyTorch: AdaLN, q / k / v / gate, softmax over each 32 x 128 window, gates."""
        att = self.attention
        x = att.ada_ln_in(single, cond)
        a, b, n, _ = x.shape
        h, d = att.n_head, x.shape[-1] // att.n_head
        nwin = windows(n)
        if tuple(pair.shape[:4]) != (b, nwin, QUERIES, KEYS):
            raise ValueError(f"pair must be the trunked atom pair [B, {nwin}, {QUERIES}, {KEYS}, d_pair], got {tuple(pair.shape)}")
        z = F.layer_norm(pair, (pair.shape[-1],), att.ln_pair.weight.to(pair.dtype), None, att.ln_pair.eps)
        bias = F.linear(z, att.to_bias.weight.to(z.dtype)).permute(0, 4, 1, 2, 3)           # [B, H, nwin, 32, 128]
        q = att.to_query(x)
        xkv = att.ada_ln_kv(x, cond) if self.cross_attention else x          # AF3 cross-attention mode: a second AdaLN for K / V
        k, v = att.to_key(xkv), att.to_value(xkv)
        gate = att.to_gate(x)
        qw = F.pad(q, (0, 0, 0, nwin * QUERIES - n)).reshape(a, b, nwin, QUERIES, h, d)
        kw = _key_windows(k, n).reshape(a, b, nwin, KEYS, h, d)
        vw = _key_windows(v, n).reshape(a, b, nwin, KEYS, h, d)
        wide = torch.float64 if x.dtype == torch.float64 else torch.float32
        scores = torch.einsum("abwqhd,abwkhd->abhwqk", qw.to(wide), kw.to(wide)) * d**-0.5 + bias.to(wide)[None]
        keep = torch.ones(b, n, device=x.device) if mask is None else mask.float()
        valid = _key_windows(keep[..., None], n)[..., 0] > 0                    # [B, nwin, 128]
        scores = scores.masked_fill(~valid[None, :, None, :, None, :], float("-inf"))
        probs = torch.nan_to_num(torch.softmax(scores, dim=-1))
        out = torch.einsum("abhwqk,abwkhd->abwqhd", probs, vw.to(wide)).to(x.dtype)
        out = out.reshape(a, b, nwin * QUERIES, h * d)[:, :, :n]
        out = att.to_out(torch.sigmoid(gate) * out)
        return torch.sigmoid(att.to_scale(cond)) * out

    def forward(self, single, cond, pair, mask=None):
        """``single`` [A, B, N, d_single], ``cond`` [A, B, N, d_cond], trunked ``pair`` [B, nwin, 32, 128, d_pair], key ``mask`` [B, N]
        -> the stream after the attention and the transition (both residuals included)."""
        if _local.serves(self, single, cond, pair, mask):
            return _local.block(self, single, cond, pair, mask)
        single = single + self.attention_delta(single, cond, pair, mask)
        return self.transition(single, cond)
