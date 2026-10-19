"""Automatic dispatch to the H100 (sm_90a) TriangleAttention, bf16, inference and training (with the module's broadcast dropout):
d_pair = d_hidden = 128 with 4 heads x 32 on the fused wgmma kernels (``kernels/triangle_attention/cuda/h100_sm90.py``), every other
(d_pair <= 512, heads of 16 / 32 channels) on the width-generic path (``h100_sm90_generic.py``), L a multiple of 16 (padded to 128).

The kernel launches are ``opaque`` custom ops (``kernels._compile``): the module compiles to one graph, the weight packs and the launches
run eagerly inside the ops (also under a CUDA-graph capture), and the autograd function below calls one op forward and one backward."""

from __future__ import annotations

from typing import Any

import weakref

import torch
import torch.nn.functional as F

from miniworld_engine import settings
from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.modules.dispatch import KernelBackend

FUSED, GENERIC = 0, 1


def _fused_shape(module, pair) -> bool:
    """d_pair = d_hidden = 128, 4 heads x 32: the fused kernels (front / tail / backward on wgmma)."""
    return (
        module.n_head == 4
        and pair.shape[-1] == 128
        and tuple(module.to_query.weight.shape) == (128, 128)
    )


def _kind(module, pair: torch.Tensor, mask: torch.Tensor | None) -> int | None:
    """FUSED, GENERIC or None (not served): the mathematical and resource contract, before touching the CUDA builder."""
    # `_h100_cuda = False` keeps the existing path (the benchmark baseline), as `_b200_cuda` does on B200.
    if (
        getattr(module, "_backend", None) != KernelBackend.TRITON
        or not getattr(module, "_h100_cuda", True)
        or settings.current().engine_backend == "triton"
    ):
        return None
    if not module.use_self_attention or module.use_qk_norm:
        return None
    if module._bias_untransposed and (
        torch.is_grad_enabled() or (module.training and module.p_drop > 0.0)
    ):
        return None  # AF3's ending-node bias: inference only (the backward reads the front's orientation)
    if not pair.is_cuda or pair.dtype != torch.bfloat16:
        return None
    if pair.ndim != 4 or pair.shape[1] != pair.shape[2]:
        return None
    if mask is not None and (mask.dtype != torch.bool or mask.shape != pair.shape[:2]):
        return None
    if torch.cuda.get_device_capability(pair.device) != (9, 0):
        return None
    d_hidden, d_pair = module.to_query.weight.shape
    if pair.shape[-1] != d_pair:
        return None
    from miniworld_engine.kernels.triangle_attention.cuda import (
        h100_sm90_generic as gen,
    )

    if not gen.supports(d_pair, d_hidden, module.n_head, pair.shape[1]):
        return None
    return FUSED if _fused_shape(module, pair) else GENERIC


def serves(module, pair: torch.Tensor, mask: torch.Tensor | None = None) -> bool:
    return _kind(module, pair, mask) is not None


# ---------------------------------------------------------------- weight packs (cached while the parameters are unchanged)
#: Packs keyed by the weights' tensor objects (held weakly), valid while every version is unchanged; scoped to a CUDA-graph capture by
#: ``kernels._capture`` (a capture packs once and records the packing kernels, so each replay repacks the current weights).
_PACKS: dict = {}


def _packed(kind: int, weights: tuple[torch.Tensor, ...]):
    def build():
        with torch.no_grad():
            if kind == FUSED:
                from miniworld_engine.kernels.triangle_attention.cuda.h100_sm90 import (
                    _pack_params,
                )

                return _pack_params(*weights)  # one launch
            from miniworld_engine.kernels.triangle_attention.cuda.h100_sm90_generic import (
                pack_weights,
            )

            return pack_weights(*weights)

    try:
        versions = tuple(w._version for w in weights)
    except (
        RuntimeError
    ):  # inference tensors carry no version counter: nothing could invalidate an entry
        return build()
    key = _capture.scoped((kind, tuple(id(w) for w in weights)))
    if key is None:
        return build()
    hit = _PACKS.get(key)
    if hit is not None:
        refs, seen, pack = hit
        if seen == versions and all(
            r() is w for r, w in zip(refs, weights, strict=True)
        ):
            return pack
    _capture.prune(_PACKS)
    if len(_PACKS) >= 64:  # a model's worth of blocks; drop the oldest
        _PACKS.pop(next(iter(_PACKS)))
    pack = build()
    _PACKS[key] = (tuple(weakref.ref(w) for w in weights), versions, pack)
    return pack


# ---------------------------------------------------------------- custom ops
# args = [pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, drop]; mask [B, L] bool and drop [B, L, C] bf16 are empty when absent.
def _infer_fake(kind, args, eps, n_head, head_dim, trans, flip):
    return torch.empty_like(args[0])


@opaque(fake=_infer_fake, name="triattn_h100_infer")
def _infer_op(
    kind: int,
    args: list[torch.Tensor],
    eps: float,
    n_head: int,
    head_dim: int,
    trans: bool,
    flip: bool,
) -> torch.Tensor:
    pair, lnw, lnb, *weights, mask, _ = args
    mask = mask if mask.numel() else None
    if kind == FUSED:
        from miniworld_engine.kernels.triangle_attention.cuda.h100_sm90 import inference

        w4, wb16, wo16 = _packed(kind, tuple(weights))
        return inference(
            pair, lnw.float(), lnb.float(), w4, wb16, wo16, mask, eps, trans, flip
        )
    from miniworld_engine.kernels.triangle_attention.cuda import (
        h100_sm90_generic as gen,
    )

    wcat, wo_p = _packed(kind, tuple(weights))
    return gen.inference(
        pair, lnw, lnb, wcat, wo_p, n_head, head_dim, mask, eps, trans, flip
    )


def _fwd_fake(kind, args, eps, n_head, head_dim, trans):
    pair = args[0]
    B, L, _, C = pair.shape
    M = B * L * L
    kw: dict[str, Any] = {"device": pair.device, "dtype": torch.bfloat16}
    f32: dict[str, Any] = {"device": pair.device, "dtype": torch.float32}
    if kind == FUSED:
        return [
            torch.empty_like(pair),
            *(torch.empty(M, 128, **kw) for _ in range(4)),
            torch.empty(B, 4, L, L, **kw),
            torch.empty(B, L, L, 4, 32, **kw),
            torch.empty(B, L, 4, L, **f32),
        ]
    HD = n_head * 32
    return [
        torch.empty_like(pair),
        torch.empty(M, C, **kw),
        torch.empty(M, **f32),
        torch.empty(M, **f32),
        torch.empty(M, 4 * HD, **kw),
        torch.empty(B, n_head, L, L, **kw),
        torch.empty(B, L, L, n_head, 32, **kw),
        torch.empty(B, L, n_head, L, **f32),
        torch.empty(B, L, L, n_head, 32, **kw),
    ]


@opaque(fake=_fwd_fake, name="triattn_h100_fwd")
def _fwd_op(
    kind: int,
    args: list[torch.Tensor],
    eps: float,
    n_head: int,
    head_dim: int,
    trans: bool,
) -> list[torch.Tensor]:
    pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, drop = args
    mask = mask if mask.numel() else None
    drop = drop if drop.numel() else None
    if kind == FUSED:
        from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as fused

        out, saved = fused.train_forward(
            pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop, True, trans
        )
        return [out, *saved]
    from miniworld_engine.kernels.triangle_attention.cuda import (
        h100_sm90_generic as gen,
    )

    out, saved = gen.train_forward(
        pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop, n_head, head_dim, trans
    )
    return [out, *saved]


def _bwd_fake(kind, args, saved, eps, n_head, head_dim, trans):
    return [
        torch.empty_like(t) for t in args[1:10]
    ]  # pair, lnw, lnb, wq, wk, wv, wg, wb, wo


@opaque(fake=_bwd_fake, name="triattn_h100_bwd")
def _bwd_op(
    kind: int,
    args: list[torch.Tensor],
    saved: list[torch.Tensor],
    eps: float,
    n_head: int,
    head_dim: int,
    trans: bool,
) -> list[torch.Tensor]:
    dy, pair, lnw, lnb, wq, wk, wv, wg, wb, wo, drop = args
    drop = drop if drop.numel() else None
    if kind == FUSED:
        from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as fused

        return list(
            fused.train_backward(
                dy,
                pair,
                lnw,
                lnb,
                wq,
                wk,
                wv,
                wg,
                wb,
                wo,
                eps,
                drop,
                tuple(saved),
                trans,
            )
        )
    from miniworld_engine.kernels.triangle_attention.cuda import (
        h100_sm90_generic as gen,
    )

    return list(
        gen.train_backward(
            dy,
            pair,
            lnw,
            lnb,
            wq,
            wk,
            wv,
            wg,
            wb,
            wo,
            eps,
            drop,
            n_head,
            head_dim,
            tuple(saved),
            trans,
        )
    )


class _TriAttnFn(torch.autograd.Function):
    """pair + drop o TriangleAttention(pair) for the starting node: one op forward (the tensors the backward reads come back as outputs),
    one op backward.  Gradients come back in the parameters' dtype."""

    @staticmethod
    def forward(
        ctx,
        kind,
        eps,
        n_head,
        head_dim,
        trans,
        pair,
        lnw,
        lnb,
        wq,
        wk,
        wv,
        wg,
        wb,
        wo,
        mask,
        drop,
    ):
        out, *saved = _fwd_op(
            kind,
            [pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, drop],
            eps,
            n_head,
            head_dim,
            trans,
        )
        ctx.save_for_backward(pair, lnw, lnb, wq, wk, wv, wg, wb, wo, drop, *saved)
        ctx.cfg = (kind, eps, n_head, head_dim, trans)
        return out

    @staticmethod
    def backward(ctx, dy):
        pair, lnw, lnb, wq, wk, wv, wg, wb, wo, drop, *saved = ctx.saved_tensors
        kind, eps, n_head, head_dim, trans = ctx.cfg
        grads = _bwd_op(
            kind,
            [dy, pair, lnw, lnb, wq, wk, wv, wg, wb, wo, drop],
            saved,
            eps,
            n_head,
            head_dim,
            trans,
        )
        return (None, None, None, None, None, *grads, None, None)


def _padded(length: int) -> int:
    return -(-length // 128) * 128


def forward(module, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """pair + drop o TriangleAttention(pair).  A length that is not a multiple of 128 is zero-padded (the padded keys are masked, the
    padded rows and columns are dropped from the result).  Without grad or dropout: the forward kernels on cached bf16 weight packs;
    otherwise the autograd function (the parameters go in as they are and get gradients in their dtype)."""
    kind = _kind(module, pair, mask)
    B, L = pair.shape[:2]
    Lp = _padded(L)
    training_drop = module.training and module.p_drop > 0.0
    drop = None
    if training_drop:
        # the module's own draw on the original layout ([B, 1, L, C] starting / [B, L, 1, C] ending): in the kernels' starting
        # layout both index the row's last position, so either is [B, L, C] as it is
        drop = (
            module._make_drop_scale(pair, module.p_drop)
            .reshape(B, L, pair.shape[-1])
            .to(torch.bfloat16)
        )
    # the ending node reads and writes its pair through the transposed row map (4-D tensor maps / a row index), without copies
    trans = not module.starting
    # AF3's ending-node bias: transposed between the front and the core (inference; see ``_kind``)
    flip = trans and not module.transposed_bias
    if Lp != L:
        pair = F.pad(pair, (0, 0, 0, Lp - L, 0, Lp - L))
        keep = torch.zeros(B, Lp, dtype=torch.bool, device=pair.device)
        keep[:, :L] = True if mask is None else mask
        mask = keep
        if drop is not None:
            drop = F.pad(drop, (0, 0, 0, Lp - L))
    weights = (
        module.to_query.weight,
        module.to_key.weight,
        module.to_value.weight,
        module.to_gate.weight,
        module.to_bias.weight,
        module.to_out.weight,
    )
    needs_grad = torch.is_grad_enabled() and (
        pair.requires_grad or any(p.requires_grad for p in module.parameters())
    )
    pair = pair.contiguous()
    ln = module.ln_pair
    head_dim = module.to_query.weight.shape[0] // module.n_head
    mask_t = mask if mask is not None else pair.new_empty(0, dtype=torch.bool)
    drop_t = drop if drop is not None else pair.new_empty(0)
    if drop is None and not needs_grad:
        out = _infer_op(
            kind,
            [pair, ln.weight, ln.bias, *weights, mask_t, drop_t],
            ln.eps,
            module.n_head,
            head_dim,
            trans,
            flip,
        )
    else:
        out = _TriAttnFn.apply(
            kind,
            ln.eps,
            module.n_head,
            head_dim,
            trans,
            pair,
            ln.weight,
            ln.bias,
            *weights,
            mask_t,
            drop_t,
        )
    if Lp != L:
        out = out[:, :L, :L]
    return out.contiguous()
