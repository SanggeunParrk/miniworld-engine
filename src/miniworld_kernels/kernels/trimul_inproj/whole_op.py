"""Whole-op ``cuequivariance``-form wrapper for the fused triangle multiplicative update.

Exposes :func:`triangle_multiplicative_update` with the **exact signature** of
``cuequivariance_torch.triangle_multiplicative_update`` so a model can consume the
*entire* op — ``LN_in → gated in-proj → triangle contraction → LN_out → out-proj+gate`` —
including its backward, as a single autograd-transparent call. It is a drop-in
replacement for the cuequiv baseline.

This is the resolution of the "where does the kernel end and the model begin?"
ambiguity: for a composite op like trimul, *how the pieces combine* (the algorithm)
and *which backend runs it* both live **inside this package**, wrapped as one
autograd Function. The consumer only supplies tensors (pair + weights) and receives
the output; gradients flow back to every weight argument. See
``modules/triangle_multiplication`` for the nn.Module that owns the weights and
calls this.

Weight-packing convention mirrors cuequiv exactly (so a caller can literally swap
the import):

    p_in_weight : (2*d_hidden, d_pair)   stacked [to_left.weight ; to_right.weight]
    g_in_weight : (2*d_hidden, d_pair)   stacked [to_left_gate.weight ; to_right_gate.weight]
    p_out_weight: (d_pair, d_hidden)     to_out.weight   (nn.Linear form)
    g_out_weight: (d_pair, d_pair)       to_gate.weight  (nn.Linear form)

Backend: the TRITON pipeline (``trimul_triton``), which is a pure weights-as-args
autograd Function on both sm90 and sm100 (grads to all weight args), with a
forward-only no-grad inference path. ``d_hidden == d_pair`` is required (the
standard AF3 configuration).
"""

from __future__ import annotations

import torch


def triangle_multiplicative_update(
    x: torch.Tensor,                    # (B, L, L, d_pair) pair representation
    direction: str,                     # "outgoing" | "incoming"
    mask: torch.Tensor | None = None,   # (B, L, L) pair mask OR (B, L) residue mask
    norm_in_weight: torch.Tensor | None = None,   # (d_pair,)
    norm_in_bias: torch.Tensor | None = None,     # (d_pair,)
    p_in_weight: torch.Tensor | None = None,      # (2*d_hidden, d_pair)
    g_in_weight: torch.Tensor | None = None,      # (2*d_hidden, d_pair)
    norm_out_weight: torch.Tensor | None = None,  # (d_hidden,)
    norm_out_bias: torch.Tensor | None = None,    # (d_hidden,)
    p_out_weight: torch.Tensor | None = None,     # (d_pair, d_hidden)
    g_out_weight: torch.Tensor | None = None,     # (d_pair, d_pair)
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fused triangle multiplicative update — cuequiv-compatible whole-op call.

    Returns the updated pair ``(B, L, L, d_pair)``. Autograd-transparent: back-prop
    produces gradients for ``x`` and every weight/bias argument, so the caller can
    hold them as ``nn.Parameter`` and train normally — exactly like the cuequiv
    baseline it replaces.
    """
    from .triton.unidirectional import trimul_triton

    if direction not in ("outgoing", "incoming"):
        msg = f"direction must be 'outgoing' or 'incoming', got {direction!r}"
        raise ValueError(msg)
    outgoing = direction == "outgoing"

    # Unstack the packed cuequiv in-projection weights into the four (d_hidden, d_pair)
    # matrices trimul_triton expects. These are views; grads accumulate back into the
    # packed tensors through the slice + the transpose inside trimul_triton.
    two_h = p_in_weight.shape[0]
    if two_h % 2 != 0:
        msg = f"p_in_weight leading dim must be 2*d_hidden (even), got {two_h}"
        raise ValueError(msg)
    d_hidden = two_h // 2
    w_left, w_right = p_in_weight[:d_hidden], p_in_weight[d_hidden:]
    w_left_gate, w_right_gate = g_in_weight[:d_hidden], g_in_weight[d_hidden:]

    return trimul_triton(
        x,
        w_left,
        w_left_gate,
        w_right,
        w_right_gate,
        g_out_weight,        # Wg  (d_pair, d_pair)
        p_out_weight,        # Wout (d_pair, d_hidden)
        norm_in_weight,
        norm_in_bias,
        norm_out_weight,
        norm_out_bias,
        eps,                 # eps_in
        eps,                 # eps_out
        d_hidden,
        outgoing,
        mask=mask,
    )
