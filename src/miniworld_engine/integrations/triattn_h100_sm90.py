"""Automatic dispatch to the H100 (sm_90a) TriangleAttention, bf16, inference and training (with the module's broadcast dropout):
d_pair = d_hidden = 128 with 4 heads x 32 on the fused wgmma kernels (``kernels/triangle_attention/cuda/h100_sm90.py``), every other
(d_pair <= 512, heads of 16 / 32 channels) on the width-generic path (``h100_sm90_generic.py``), L a multiple of 16 (padded to 128).
Any number of samples B (templates, structures or augments folded into the batch): up to ``max_batch`` per launch, more (or more than
fit in memory) in launches run one after another (``plan``).

The kernel launches are ``opaque`` custom ops (``kernels._compile``): the module compiles to one graph, the weight packs and the launches
run eagerly inside the ops (also under a CUDA-graph capture), and the autograd function below calls one op forward and one backward."""

from __future__ import annotations

import weakref
from typing import Any

import torch
import torch.nn.functional as F

from miniworld_engine import settings
from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import device_constant, opaque
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
def _infer_fake(kind, args, eps, n_head, head_dim, trans):
    return torch.empty_like(args[0])


@opaque(fake=_infer_fake, name="triattn_h100_infer")
def _infer_op(
    kind: int,
    args: list[torch.Tensor],
    eps: float,
    n_head: int,
    head_dim: int,
    trans: bool,
) -> torch.Tensor:
    return _infer(kind, args, eps, n_head, head_dim, trans)


def _infer(kind, args, eps, n_head, head_dim, trans, out=None):
    """The inference kernels; `out`: a contiguous tensor like the pair to write the result into (a chunk's rows of a batch's output,
    eager only: an op output may not alias an input)."""
    pair, lnw, lnb, *weights, mask, _ = args
    mask = mask if mask.numel() else None
    if kind == FUSED:
        from miniworld_engine.kernels.triangle_attention.cuda.h100_sm90 import inference

        w4, wb16, wo16 = _packed(kind, tuple(weights))
        return inference(
            pair, lnw.float(), lnb.float(), w4, wb16, wo16, mask, eps, trans, out
        )
    from miniworld_engine.kernels.triangle_attention.cuda import (
        h100_sm90_generic as gen,
    )

    wcat, wo_p = _packed(kind, tuple(weights))
    return gen.inference(
        pair, lnw, lnb, wcat, wo_p, n_head, head_dim, mask, eps, trans, out
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


# ---------------------------------------------------------------- batched samples and chunking
#: Samples one launch takes, per path (``FUSED`` / ``GENERIC``, see ``kind_of``).  Every kernel of both paths takes B samples in one
#: launch: the front, tail, LayerNorm, gate and weight-gradient kernels walk the B L^2 rows (b, i, j), the attention core and its
#: backward walk (sample, head, tile) tasks, the key mask [B, L] and the dropout scale [B, L, C] are read per sample, and the parameter
#: gradients are reduced over every sample of the launch.  The bound is what was validated (B up to 48 against B one-sample calls and
#: the fp32 reference, d_pair 64-512, L 128-768); the kernels' only index bound is the int32 row coordinate of their TMA maps,
#: B Lp^2 < 2^31 (``max_batch`` applies it).  A call with more samples runs in several launches (``plan``).  (The one exception to one
#: launch per kernel: the width-generic ending node's inference out-projection, a cublasLt GEMM whose transposed output rows are a
#: strided batch over i, runs one GEMM per sample.)
MAX_BATCH = {FUSED: 48, GENERIC: 48}


def max_batch(kind: int, length: int | None = None) -> int:
    """Largest number of samples one launch of the path ``kind`` (``FUSED`` / ``GENERIC``) takes, at pair length ``length`` when
    given; a call with more samples is split into launches run one after another (``plan``)."""
    limit = MAX_BATCH[kind]
    if length is not None:
        lp = _padded(length)
        limit = min(limit, (2**31 - 1) // (lp * lp))
    return max(1, limit)


def kind_of(module, pair: torch.Tensor, mask: torch.Tensor | None = None) -> int | None:
    """``FUSED``, ``GENERIC`` or None (not served): the path a call takes, the key of ``max_batch``."""
    return _kind(module, pair, mask)


#: Chunking.  A call is split when it has more samples than one launch takes (``max_batch``) or when its working set would not fit
#: CHUNK_MEMORY_FRACTION of the device: chunks of at most ``plan(...)[0]`` samples (balanced sizes) run one after another through the
#: same ops into one output (inference: written in place; training: each chunk's y and dx copied into the batch's), so a call never
#: runs out of memory because of its batch and never falls back to the PyTorch statements because of its size.  The chunk is a
#: deterministic function of the call's path, shapes, widths and mode and of the device's TOTAL memory (never of live free memory), so
#: torch.compile and CUDA-graph capture see the same split every time.  The working set is counted in bytes per pair token (Lp^2 per
#: sample) from the tensors each path allocates (``_bytes_per_token``; against the measured peaks of one launch, H100 80GB 2026-10-19: fused +0 / +5 %,
#: generic +0 / +25 %).  Training that does not fit in one launch is chunked in one of two ways (``_Chunked``): when every chunk's
#: saved tensors (held from the forward to the backward) fit next to one chunk's backward workspace, each chunk keeps them; otherwise
#: the backward reruns each chunk's forward first (checkpoint), so only the chunk being differentiated holds saved tensors.  No split
#: sheds the caller's own tensors (pair, the output, dy, dx): a call whose inputs and outputs alone exceed the device still runs out
#: of memory.
CHUNK_MEMORY_FRACTION = 0.85
#: cuBLAS / cublasLt workspaces, the query-side dbias slabs of one-sample launches (<= 132 x 128 x L x 4 bytes), the allocator's rounding
_SLACK = 1 << 30


def _bytes_per_token(
    kind: int, d_pair: int, n_head: int, padded: bool
) -> tuple[int, int, int]:
    """(inference working set, training saved tensors, training backward working set) in bytes per pair token of one sample, on top
    of the caller's tensors (pair and output; in training also dy and dx)."""
    c, h = d_pair, n_head
    # a length padded to 128: the padded copy of the pair (saved in training) and of dy, the padded output / dx before the cut
    pad = 4 * c if padded else 0
    if kind == FUSED:
        # inference: q | k | v | g, o, the output; sbias (bf16 per head)
        # saved: q | k | v | g, o; sbias; lse (fp32 per head)   backward: do, dg, dq, dk, dv; delta, dbias, its slab sum (fp32 per head)
        return 12 * c + 2 * h + pad, 10 * c + 6 * h + pad // 2, 10 * c + 12 * h + pad
    # heads padded to 32 channels, the bias columns to 8
    hd, hp = 32 * h, (h + 7) // 8 * 8
    # inference: y, the LN statistics, q | k | v | g | bias, sbias, u = sigmoid(g) o O, the output and cublasLt's tuning copy of it
    # saved: y, statistics, q | k | v | g, sbias, o, lse, u
    # backward: ddrop, dyln; du, do; dall = [dq | dk | dv | dg | dbias]; delta, dbias, its slab sum (fp32)
    return (
        6 * c + 10 * hd + 2 * hp + 2 * h + 8 + pad,
        2 * c + 12 * hd + 6 * h + 8 + pad // 2,
        4 * c + 12 * hd + 2 * hp + 12 * h + pad,
    )


def plan(
    kind: int,
    batch: int,
    length: int,
    d_pair: int,
    n_head: int,
    training: bool,
    total_memory: int,
) -> tuple[int, bool]:
    """(chunk, checkpoint): the samples one launch takes and whether training chunks rerun their forward in the backward (instead of
    keeping their saved tensors); chunk >= batch is one call.
    A pure function of its arguments (``total_memory`` = the device's total memory in bytes)."""
    limit = max_batch(kind, length)
    lp = _padded(length)
    tokens = lp * lp
    work_inf, held, work_bwd = _bytes_per_token(kind, d_pair, n_head, lp != length)
    budget = CHUNK_MEMORY_FRACTION * total_memory - _SLACK
    c = d_pair

    def fit(free: float, per_sample: int) -> int:
        return max(1, min(limit, int(free // (per_sample * tokens))))

    if not training:
        # the caller's pair and the output (the chunks write into it)
        resident = 4 * c
        if batch * (resident + work_inf) * tokens <= budget:
            return min(batch, limit), False
        return fit(budget - batch * resident * tokens, work_inf), False
    # training: the caller's pair, output, dy and dx (the chunks' results are copied into the output and dx)
    resident = 8 * c
    if batch * (resident + held + work_bwd) * tokens <= budget:
        return min(batch, limit), False
    # a chunk's own output (rerun or before its copy) and dx before its copy
    work = work_bwd + 4 * c
    free = budget - batch * (resident + held) * tokens
    # every chunk's saved tensors fit next to one chunk's backward
    if free >= work * tokens:
        return fit(free, work), False
    return fit(budget - batch * resident * tokens, held + work), True


def _split(batch: int, chunk: int) -> list[int]:
    """Balanced chunk sizes (they differ by at most one sample)."""
    count = -(-batch // chunk)
    return [batch // count + (i < batch % count) for i in range(count)]


@device_constant
def _total_memory(index: int) -> int:
    return torch.cuda.get_device_properties(index).total_memory


def forward(module, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """pair + drop o TriangleAttention(pair) for B samples: one launch per kernel, or launches of ``plan``'s chunk run one after
    another when the call has more samples than one launch takes or would not fit in memory."""
    kind = _kind(module, pair, mask)
    B, L, _, C = pair.shape
    drop = None
    if module.training and module.p_drop > 0.0:
        # the module's own draw on the original layout ([B, 1, L, C] starting / [B, L, 1, C] ending): in the kernels' starting
        # layout both index the row's last position, so either is [B, L, C] as it is
        drop = (
            module._make_drop_scale(pair, module.p_drop)
            .reshape(B, L, C)
            .to(torch.bfloat16)
        )
    needs_grad = torch.is_grad_enabled() and (
        pair.requires_grad or any(p.requires_grad for p in module.parameters())
    )
    index = (
        pair.device.index
        if pair.device.index is not None
        else torch.cuda.current_device()
    )
    chunk, checkpoint = plan(
        kind,
        B,
        L,
        C,
        module.n_head,
        needs_grad or drop is not None,
        _total_memory(index),
    )
    params = _params(module)
    if chunk >= B:
        return _forward_one(module, kind, params, pair, mask, drop, needs_grad)
    sizes = _split(B, chunk)
    if not needs_grad:
        # inference chunks write their rows of one output (eager, L a multiple of 128 and no dropout: the tail / out-projection
        # take the rows as their output); otherwise each chunk's result is copied into it
        out = torch.empty(pair.shape, dtype=pair.dtype, device=pair.device)
        into = drop is None and _padded(L) == L and not torch.compiler.is_compiling()
        head_dim = module.to_query.weight.shape[0] // module.n_head
        start = 0
        for n in sizes:
            sl = slice(start, start + n)
            m = None if mask is None else mask[sl]
            if into:
                x = pair[sl].contiguous()
                args = [
                    x,
                    *params,
                    m if m is not None else x.new_empty(0, dtype=torch.bool),
                    x.new_empty(0),
                ]
                _infer(
                    kind,
                    args,
                    module.ln_pair.eps,
                    module.n_head,
                    head_dim,
                    not module.starting,
                    out[sl],
                )
            else:
                out[sl] = _forward_one(
                    module,
                    kind,
                    params,
                    pair[sl],
                    m,
                    None if drop is None else drop[sl],
                    False,
                )
            start += n
        return out
    return _Chunked.apply(
        kind,
        module.ln_pair.eps,
        module.n_head,
        module.to_query.weight.shape[0] // module.n_head,
        not module.starting,
        tuple(sizes),
        checkpoint,
        pair,
        *params,
        mask if mask is not None else pair.new_empty(0, dtype=torch.bool),
        drop if drop is not None else pair.new_empty(0),
    )


def _chunk_inputs(pair, mask, drop, start, n):
    """The chunk's pair, mask and dropout scale, padded to a multiple of 128 (the pad keys masked) as `_forward_one` pads."""
    x = pair[start : start + n]
    m = mask[start : start + n] if mask.numel() else mask
    d = drop[start : start + n] if drop.numel() else drop
    L = x.shape[1]
    Lp = _padded(L)
    if Lp != L:
        x = F.pad(x, (0, 0, 0, Lp - L, 0, Lp - L))
        keep = torch.zeros(n, Lp, dtype=torch.bool, device=x.device)
        keep[:, :L] = True if not m.numel() else m
        m = keep
        if d.numel():
            d = F.pad(d, (0, 0, 0, Lp - L))
    return x.contiguous(), m, d


class _Chunked(torch.autograd.Function):
    """Training on B samples in launches of ``sizes`` run one after another: one forward op per chunk (its result copied into one
    output, its saved tensors kept), one backward op per chunk (its dx copied into one gradient).  ``checkpoint``: no chunk keeps its
    saved tensors; the backward reruns the chunk's forward op first (the dropout scale is an input, so it is the same draw).  The
    parameters go to the ops in fp32 (an exact cast: the same packed bf16 weights), so the chunks' parameter gradients come back in
    fp32, add up in fp32 and are rounded to the parameters' dtype once, as inside one launch."""

    @staticmethod
    def forward(
        ctx, kind, eps, n_head, head_dim, trans, sizes, checkpoint, pair, *rest
    ):
        *params, mask, drop = rest
        p32 = [p.float() for p in params]
        L = pair.shape[1]
        out = torch.empty(pair.shape, dtype=pair.dtype, device=pair.device)
        saved: list[torch.Tensor] = []
        start = 0
        for n in sizes:
            x, m, d = _chunk_inputs(pair, mask, drop, start, n)
            o, *s = _fwd_op(kind, [x, *p32, m, d], eps, n_head, head_dim, trans)
            out[start : start + n] = o[:, :L, :L]
            del o
            if not checkpoint:
                saved.extend(s)
            del s
            start += n
        ctx.save_for_backward(pair, *params, mask, drop, *saved)
        ctx.cfg = (kind, eps, n_head, head_dim, trans, sizes, checkpoint)
        return out

    @staticmethod
    def backward(ctx, dy):
        pair, *rest = ctx.saved_tensors
        params, (mask, drop), saved = rest[:8], rest[8:10], rest[10:]
        kind, eps, n_head, head_dim, trans, sizes, checkpoint = ctx.cfg
        p32 = [p.float() for p in params]
        L = pair.shape[1]
        Lp = _padded(L)
        per = len(saved) // len(sizes)
        dx = torch.empty(pair.shape, dtype=pair.dtype, device=pair.device)
        acc = [
            torch.zeros(p.shape, dtype=torch.float32, device=p.device) for p in params
        ]
        start = 0
        for c, n in enumerate(sizes):
            x, m, d = _chunk_inputs(pair, mask, drop, start, n)
            if checkpoint:
                _, *s = _fwd_op(kind, [x, *p32, m, d], eps, n_head, head_dim, trans)
            else:
                s = list(saved[c * per : (c + 1) * per])
            g = dy[start : start + n]
            if Lp != L:
                g = F.pad(g, (0, 0, 0, Lp - L, 0, Lp - L))
            grads = _bwd_op(
                kind, [g.contiguous(), x, *p32, d], s, eps, n_head, head_dim, trans
            )
            del s, g
            dx[start : start + n] = grads[0][:, :L, :L]
            for a, gp in zip(acc, grads[1:], strict=True):
                a += gp
            del grads
            start += n
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            dx,
            *(a.to(p.dtype) for a, p in zip(acc, params, strict=True)),
            None,
            None,
        )


def _params(module) -> tuple[torch.Tensor, ...]:
    """(ln weight, ln bias, Wq, Wk, Wv, Wg, Wb, Wo): the module's parameters in the ops' argument order."""
    return (
        module.ln_pair.weight,
        module.ln_pair.bias,
        module.to_query.weight,
        module.to_key.weight,
        module.to_value.weight,
        module.to_gate.weight,
        module.to_bias.weight,
        module.to_out.weight,
    )


def _forward_one(
    module,
    kind: int,
    params: tuple[torch.Tensor, ...],
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    drop: torch.Tensor | None,
    needs_grad: bool,
) -> torch.Tensor:
    """One launch of every kernel for the samples of ``pair``.  A length that is not a multiple of 128 is zero-padded (the padded
    keys are masked, the padded rows and columns are dropped from the result).  Without grad or dropout: the forward kernels on cached
    bf16 weight packs; otherwise the autograd function (the parameters go in as they are and get gradients in their dtype)."""
    B, L = pair.shape[:2]
    Lp = _padded(L)
    # the ending node reads and writes its pair through the transposed row map (4-D tensor maps / a row index), without copies
    trans = not module.starting
    if Lp != L:
        pair = F.pad(pair, (0, 0, 0, Lp - L, 0, Lp - L))
        keep = torch.zeros(B, Lp, dtype=torch.bool, device=pair.device)
        keep[:, :L] = True if mask is None else mask
        mask = keep
        if drop is not None:
            drop = F.pad(drop, (0, 0, 0, Lp - L))
    pair = pair.contiguous()
    eps = module.ln_pair.eps
    head_dim = module.to_query.weight.shape[0] // module.n_head
    mask_t = mask if mask is not None else pair.new_empty(0, dtype=torch.bool)
    drop_t = drop if drop is not None else pair.new_empty(0)
    if drop is None and not needs_grad:
        out = _infer_op(
            kind,
            [pair, *params, mask_t, drop_t],
            eps,
            module.n_head,
            head_dim,
            trans,
        )
    else:
        out = _TriAttnFn.apply(
            kind,
            eps,
            module.n_head,
            head_dim,
            trans,
            pair,
            *params,
            mask_t,
            drop_t,
        )
    if Lp != L:
        out = out[:, :L, :L]
    return out.contiguous()
