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
    """Fused triangle multiplicative update — whole-op call, cuequiv argument-compatible.

    Returns the RESIDUAL form ``x + triangle_multiplicative_update(x)``, shape
    ``(B, L, L, d_pair)``. ``cuequivariance_torch``'s function of the same name returns the
    bare update instead, so the two are argument-compatible but not return-compatible.

    To compare them, add the residual to THEIRS in plain torch rather than trying to strip it
    from this one::

        ours  = ops.triangle_multiplicative_update(x, ...)
        theirs = x + cuequivariance_torch.triangle_multiplicative_update(x, ...)

    That is exactly what ``modules.TriangleMultiplication`` does: its CUEQUIVARIANCE backend
    calls the bare cuequiv op and then applies the drop scale and the residual as ordinary torch
    ops, so every backend of that module computes the same function. Stripping the residual here
    would instead cost a whole zeroed ``[B,L,L,d_pair]`` operand or a lossy ``- x``, to undo an
    add that is worth 1.27x when it stays fused (see the measurement block in
    ``trimul_inproj/triton/gate_elem.py``).

    Autograd-transparent: back-prop produces gradients for ``x`` and every weight/bias
    argument, so the caller can hold them as ``nn.Parameter`` and train normally.
    """
    from miniworld_engine.kernels.trimul_inproj.triton.unidirectional import (
        trimul_triton,
    )

    if direction not in ("outgoing", "incoming"):
        msg = f"direction must be 'outgoing' or 'incoming', got {direction!r}"
        raise ValueError(msg)
    outgoing = direction == "outgoing"

    # Every weight below is annotated `Tensor | None` with a None default, and the body uses all
    # of them unguarded -- calling this without one raised `AttributeError: 'NoneType' object has
    # no attribute 'shape'` from inside the unpacking. They are not optional; the defaults exist
    # so the argument order can stay keyword-friendly. Say which one is missing instead.
    # Written as an explicit `or` chain rather than a comprehension over a dict so a type checker
    # narrows all six to Tensor past this point -- a comprehension proves nothing about the names.
    if (p_in_weight is None or g_in_weight is None or norm_out_weight is None
            or norm_out_bias is None or p_out_weight is None or g_out_weight is None):
        absent = [k for k, v in (("p_in_weight", p_in_weight), ("g_in_weight", g_in_weight),
                                 ("norm_out_weight", norm_out_weight),
                                 ("norm_out_bias", norm_out_bias),
                                 ("p_out_weight", p_out_weight), ("g_out_weight", g_out_weight))
                  if v is None]
        msg = (f"triangle_multiplicative_update requires {', '.join(absent)}; they default to "
               f"None only to keep the argument order keyword-friendly.")
        raise TypeError(msg)

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

    # Grads flow to every weight arg because the unpacked weights above are differentiable
    # slice+transpose VIEWS of the packed inputs.
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


def bidirectional_triangle_multiplicative_update(
    x: torch.Tensor,                        # (B, L, L, d_pair)
    mask: torch.Tensor | None = None,       # (B, L) residue mask
    *,
    norm_in_weight: torch.Tensor,           # (d_pair,)
    norm_in_bias: torch.Tensor,             # (d_pair,)
    to_left_weight: torch.Tensor,           # (2*d_hidden, d_pair)
    to_left_gate_weight: torch.Tensor,      # (2*d_hidden, d_pair)
    to_right_weight: torch.Tensor,          # (2*d_hidden, d_pair)
    to_right_gate_weight: torch.Tensor,     # (2*d_hidden, d_pair)
    norm_out_weight: torch.Tensor,          # (2*d_hidden,)
    norm_out_bias: torch.Tensor,            # (2*d_hidden,)
    to_out_weight: torch.Tensor,            # (d_pair, 2*d_hidden)
    to_gate_weight: torch.Tensor,           # (d_pair, d_pair)
    eps: float = 1e-5,
) -> torch.Tensor:
    """Bidirectional (outgoing+incoming, one fused block) triangle multiplicative
    update — whole-op call. Returns the RESIDUAL form ``x + bidir_trimul(x)``,
    ``(B, L, L, d_pair)``: the residual is fused into the gate store epilogue and is part of
    what this op is, not a flag on it (see ``trimul_inproj/triton/gate_elem.py`` for the
    measurement). cuequivariance has no bidirectional equivalent to compare against; for the
    single-direction one, add the residual to cuequiv's output in plain torch (see
    ``triangle_multiplicative_update`` above).
    Autograd-transparent: grads flow to ``x`` and every weight. Backed by the TRITON bidir pipeline
    (weights-as-args autograd Function + no-grad inference path). ``d_hidden == d_pair``
    required. Unlike single-direction trimul there is no cuequiv equivalent, so this
    takes the four (2·d_hidden, d_pair) projections directly.
    """
    from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import (
        bidirectional_trimul_triton,
    )

    # Grads flow to every weight arg (differentiable .t() views) and x (autograd-transparent LN_in).
    d_hidden = to_left_weight.shape[0] // 2  # to_left: (2*d_hidden, d_pair)
    return bidirectional_trimul_triton(
        x,
        to_left_weight,
        to_left_gate_weight,
        to_right_weight,
        to_right_gate_weight,
        to_gate_weight,       # Wg  (d_pair, d_pair)
        to_out_weight,        # Wout (d_pair, 2*d_hidden)
        norm_in_weight,
        norm_in_bias,
        norm_out_weight,
        norm_out_bias,
        eps,                  # eps_in
        eps,                  # eps_out
        d_hidden,
        mask=mask,
    )
