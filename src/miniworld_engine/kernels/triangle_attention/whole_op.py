"""Whole-op wrapper for triangular gated self-attention.

Exposes :func:`triangle_attention` — the FULL layer op (``LN → q/k/v/bias/gate
projections → [optional qk-norm] → pair-biased attention → sigmoid-gate → out-proj``),
weights-as-args and autograd-transparent, with the fused attention kernel inside. A
model layer holds the weights as ``nn.Parameter`` and makes one call; it never touches
the primitive attention kernel or composes the projections itself.

Weight convention mirrors ``nn.Linear`` (``weight`` is ``(out, in)``, applied as
``F.linear`` = ``x @ weight.T``). ``use_self_attention`` is inferred from whether
``to_query_weight``/``to_key_weight`` are given; ``use_qk_norm`` from whether
``norm_query_weight``/``norm_key_weight`` are given.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange


def triangle_attention(
    pair: torch.Tensor,                       # (B, L, L, d_pair)
    mask: torch.Tensor | None = None,         # (B, L) residue mask
    *,
    n_head: int,
    ln_pair_weight: torch.Tensor,
    ln_pair_bias: torch.Tensor,
    to_value_weight: torch.Tensor,            # (d_hidden, d_pair)
    to_bias_weight: torch.Tensor,             # (n_head, d_pair)
    to_gate_weight: torch.Tensor,             # (d_hidden, d_pair)
    to_out_weight: torch.Tensor,              # (d_pair, d_hidden)
    to_query_weight: torch.Tensor | None = None,   # (d_hidden, d_pair) — enables self-attn
    to_key_weight: torch.Tensor | None = None,     # (d_hidden, d_pair)
    norm_query_weight: torch.Tensor | None = None,  # (d_head,) — enables qk-norm
    norm_key_weight: torch.Tensor | None = None,    # (d_head,)
    starting: bool = True,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fused triangle (gated) attention — whole-op call. Returns ``(B, L, L, d_pair)``.

    Autograd-transparent: back-prop produces gradients for ``pair`` and every weight.
    Matches the AF3 reference (query is scaled by ``d_head**-0.5`` inside the attention
    kernel; masked keys get ``finfo.min`` bias).
    """
    from .triton.main import triton_triangle_attention_pair_bias

    # Both weights, not just the query: the branch below uses to_key_weight too, so deriving the
    # flag from to_query_weight alone let `to_query_weight=W, to_key_weight=None` reach
    # F.linear(x, None). Testing the pair also narrows both to Tensor for a checker.
    use_self_attention = to_query_weight is not None and to_key_weight is not None
    if (to_query_weight is None) != (to_key_weight is None):
        msg = "to_query_weight and to_key_weight must be given together (self-attention) or both omitted"
        raise TypeError(msg)
    use_qk_norm = norm_query_weight is not None

    if not starting:
        pair = rearrange(pair, "B I J D -> B J I D").contiguous()

    x = F.layer_norm(pair, (pair.shape[-1],), ln_pair_weight, ln_pair_bias, eps)

    value = rearrange(F.linear(x, to_value_weight), "B L L2 (H D) -> B H L L2 D", H=n_head)
    bias = rearrange(F.linear(x, to_bias_weight), "B L L2 H -> B H L L2")
    if mask is not None:
        bias = bias.masked_fill(~mask[:, None, None, :], torch.finfo(bias.dtype).min)

    if use_self_attention:
        query = rearrange(F.linear(x, to_query_weight), "B L L2 (H D) -> B H L L2 D", H=n_head)
        key = rearrange(F.linear(x, to_key_weight), "B L L2 (H D) -> B H L L2 D", H=n_head)
        if use_qk_norm:
            d_head = query.shape[-1]
            query = F.rms_norm(query, (d_head,), norm_query_weight, eps)
            key = F.rms_norm(key, (d_head,), norm_key_weight, eps)
        out = triton_triangle_attention_pair_bias(query, key, value, bias)
    else:
        attention = F.softmax(bias, dim=-1)
        out = torch.einsum("bhjk,bhikd->bhijd", attention, value)

    out = rearrange(out, "B H L L2 D -> B L L2 (H D)")
    out = torch.sigmoid(F.linear(x, to_gate_weight)) * out
    out = F.linear(out, to_out_weight)

    if not starting:
        out = rearrange(out, "B J I D -> B I J D")
    return out
