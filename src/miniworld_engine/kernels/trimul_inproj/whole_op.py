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


def _cute_eligible(x: torch.Tensor) -> bool:
    """True when the CuTeDSL trimul path should run for ``x``: Hopper+ (sm90+, where the
    module dispatch resolves CUTE), bf16, B == 1. Any other input falls back to triton."""
    if x.dtype != torch.bfloat16 or x.dim() != 4 or x.shape[0] != 1:
        return False
    try:
        from miniworld_engine.modules.dispatch import (
            KernelBackend,
            resolve_triangle_multiplication,
        )
        from miniworld_engine.modules.exceptions import ImplementationType

        return (
            resolve_triangle_multiplication(ImplementationType.MINIWORLD, x.device)
            == KernelBackend.CUTE
        )
    except Exception:  # noqa: BLE001 - dispatch/import hiccup -> just use triton
        return False


def _mask_2d(mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Residue mask (B, L) -> pair mask (B, L, L); a (B, L, L) mask passes through."""
    return mask if mask.dim() == 3 else (mask.unsqueeze(-1) & mask.unsqueeze(-2))


def _trimul_cute(
    x, outgoing, mask,
    w_left, w_left_gate, w_right, w_right_gate, g_out_weight, p_out_weight,
    ln_in_w, ln_in_b, ln_out_w, ln_out_b, eps,
):
    """CuTeDSL trimul with weights as args (cuequiv layout bridged to the kernel layout).

    Returns the updated pair, or None if the cute path can't run this shape (caller then
    uses triton). Mirrors ``modules/triangle_multiplication`` exactly: the dedicated
    ``bdll_sm100`` inference path under no-grad, the v6 merged autograd kernel under grad.
    Grads reach every weight arg — WL/… are differentiable ``.t()`` views of the packed
    inputs and the inference path's einsum/back-split are autograd-transparent too."""
    d_hidden, d_pair = w_left.shape[-2], w_left.shape[-1]
    if d_hidden != d_pair:  # the cute kernels require d_hidden == d_pair (AF3 config)
        return None
    # cuequiv nn.Linear weights -> the kernels' x@W ("...weight.T") form. Cheap D×D
    # transposes; .contiguous() so the kernels get dense operands; both differentiable.
    WL = w_left.t().contiguous()
    WLg = w_left_gate.t().contiguous()
    WR = w_right.t().contiguous()
    WRg = w_right_gate.t().contiguous()
    Wg = g_out_weight.t().contiguous()   # to_gate.weight.T
    Wp = p_out_weight                    # to_out.weight (nn.Linear form; back-split wants this)
    direction = "out" if outgoing else "in"

    grad = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad
        for t in (x, WL, WLg, WR, WRg, Wg, Wp, ln_in_w, ln_out_w)
    )
    if grad:
        # Capability dispatch: the merged v6 training whole-op has two arch variants.
        # Blackwell (sm100, cap major>=10) uses the tcgen05 sig-front kernel; Hopper
        # (sm90 / H100) uses the quack bdll front. Previously this path hardcoded the
        # sm100 variant, so on H100 it hit `sig front requires the fused sm100 kernel`.
        if torch.cuda.get_device_capability(x.device)[0] >= 10:
            from .cute.v6_training_merged_sm100 import (
                prepack_lr_operand_sm100 as _prepack_lr,
                v6_forward_merged_sm100 as _v6_forward_merged,
            )
        else:
            from .cute.v6_training_merged import (
                v6_forward_merged as _v6_forward_merged,
            )
            from .cute.launch import prepack_lr_operand as _prepack_lr

        b_lr = _prepack_lr(WL, WLg, WR, WRg)
        row_scale = _mask_2d(mask, x).reshape(-1).to(x.dtype) if mask is not None else None
        return _v6_forward_merged(
            x, WL, WLg, WR, WRg, Wg, Wp, ln_in_w, ln_in_b, ln_out_w, ln_out_b,
            eps, b_lr, direction, row_scale,
        )

    # inference (no-grad): the fast bdll_sm100 front + tcgen05 back-split
    from miniworld_engine.kernels.fused_ln_mask.cute.fused_ln_mask import fused_ln_mask
    from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm
    from miniworld_engine.kernels.tm1.cute.launch import tm1_cute_forward
    from miniworld_engine.kernels.trimul_inproj.cute.back_split_sm100 import (
        trimul_back_split_sm100,
    )
    from miniworld_engine.modules.triangle_multiplication.dispatch import resolve_out_layout

    b, l1, l2, d = x.shape
    if mask is not None:
        x_n = fused_ln_mask(x, ln_in_w, ln_in_b, _mask_2d(mask, x))
    else:
        x_n = triton_layernorm(
            x.reshape(b * l1 * l2, d), ln_in_w, ln_in_b, eps
        ).view(b, l1, l2, d)
    left, right = tm1_cute_forward(
        x_n, WL, WLg, WR, WRg, out_layout=resolve_out_layout(x.device)
    )
    tri = (
        torch.einsum("bdik,bdjk->bdij", left, right)
        if outgoing
        else torch.einsum("bdki,bdkj->bdij", left, right)
    )
    return trimul_back_split_sm100(tri, x_n, Wp, Wg, ln_out_w, ln_out_b, eps)


def _bidir_cute(
    x, mask, ln_in_w, ln_in_b,
    to_left_w, to_left_gate_w, to_right_w, to_right_gate_w,
    ln_out_w, ln_out_b, to_out_w, to_gate_w, eps,
):
    """CuTeDSL bidirectional trimul with weights as args. Returns the updated pair, or None
    if the shape isn't cute-supported (caller uses triton). Uses the same fused-bidir kernel
    the module resolves (``bidir_forward_sm100``); it is autograd-capable (fwd+bwd), so grads
    reach every weight arg (the .t() views are differentiable) AND ``x`` (the LN_in is a plain
    autograd-transparent triton layernorm). Under no-grad it is forward-only."""
    two_h, d_pair = to_left_w.shape[-2], to_left_w.shape[-1]
    h = two_h // 2
    if two_h % 2 != 0 or h != d_pair:  # per-direction hidden must equal d_pair (AF3 config)
        return None
    WL = to_left_w.t().contiguous()
    WLg = to_left_gate_w.t().contiguous()
    WR = to_right_w.t().contiguous()
    WRg = to_right_gate_w.t().contiguous()
    Wg = to_gate_w.t().contiguous()   # to_gate.weight.T (d, d)
    Wp = to_out_w                     # to_out.weight (d, 2h) nn.Linear form
    from .cute.bidir_training_sm100 import bidir_forward_sm100, prepack_lr_operand_sm100

    b_lr = prepack_lr_operand_sm100(WL, WLg, WR, WRg)
    row_scale = _mask_2d(mask, x).reshape(-1).to(x.dtype) if mask is not None else None
    return bidir_forward_sm100(
        x, WL, WLg, WR, WRg, Wg, Wp, ln_in_w, ln_in_b, ln_out_w, ln_out_b,
        eps, b_lr, h, row_scale,
    )


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

    # Backend dispatch: on Hopper+ (sm90+) with bf16 B=1, use the CuTeDSL (tcgen05) path —
    # the same one modules/triangle_multiplication resolves — which is ~1.5x faster than
    # triton on B200. Falls back to triton for any input the cute kernels don't cover
    # (non-bf16, B>1, pre-Hopper). Grads flow to every weight arg because the unpacked
    # weights below are differentiable slice+transpose VIEWS of the packed inputs.
    if _cute_eligible(x):
        out = _trimul_cute(
            x, outgoing, mask,
            w_left, w_left_gate, w_right, w_right_gate,
            g_out_weight, p_out_weight,
            norm_in_weight, norm_in_bias, norm_out_weight, norm_out_bias, eps,
        )
        if out is not None:
            return out

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
    update — whole-op call. Returns ``(B, L, L, d_pair)``. Autograd-transparent:
    grads flow to ``x`` and every weight. Backed by the TRITON bidir pipeline
    (weights-as-args autograd Function + no-grad inference path). ``d_hidden == d_pair``
    required. Unlike single-direction trimul there is no cuequiv equivalent, so this
    takes the four (2·d_hidden, d_pair) projections directly.
    """
    from .triton.bidirectional import bidirectional_trimul_triton

    # Backend dispatch: cute (tcgen05) on Hopper+ bf16 B=1 — the fused-bidir kernel the
    # module resolves (~1.8x faster than triton on B200) — else triton. Grads flow to
    # every weight arg (differentiable .t() views) and x (autograd-transparent LN_in).
    if _cute_eligible(x):
        out = _bidir_cute(
            x, mask, norm_in_weight, norm_in_bias,
            to_left_weight, to_left_gate_weight, to_right_weight, to_right_gate_weight,
            norm_out_weight, norm_out_bias, to_out_weight, to_gate_weight, eps,
        )
        if out is not None:
            return out

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
