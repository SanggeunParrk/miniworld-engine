"""Automatic dispatch to the B200 (sm_100a) TriangleAttention, bf16, inference and training (with the module's broadcast dropout):
d_pair = d_hidden = 128 with 4 heads x 32 on the fused kernels (``FUSED``: tri_front / tri_tail / tri_gate_bwd / tri_head_bwd /
tri_wgrad around the attention core), d_pair 64-512 with heads of 16 / 32 channels on the wide kernels (``GENERIC``: tri_wfront /
tri_wtail / tri_wgbwd / tri_whbwd / cuBLAS / tri_wfinish), both in ``kernels/triangle_attention/cuda/b200_triattn.py``.
Any length L >= 16: the kernels take whole 128-token tiles, a length that is not a multiple of 128 is zero-padded (the pad keys
masked, the pad rows and columns cut from the result). Any number of samples B (templates, structures or augments folded into the
batch): up to ``max_batch`` per launch, more (or more than fit in memory) in launches run one after another (``plan``).

The kernel launches are ``opaque`` custom ops (``kernels._compile``): the module compiles to one graph, the weight packs and the
launches run eagerly inside the ops (also under a CUDA-graph capture), and the autograd function below calls one op forward and one
backward. The ending node runs the same kernels on the caller's [B, I, J, C] pair: the kernels that read or write the pair-shaped
tensors (x, y, dy, dx) address them through 4-D tensor maps (``trans``), the intermediates stay in the starting-node layout, and no
transposed copy is made."""

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
#: Shortest length served: shorter pairs run the module's PyTorch statements (a 16-token pair padded to one 128 tile is already 64x
#: the work; below it the reference is as fast)
MIN_LENGTH = 16
TILE = 128


def _padded(length: int) -> int:
    return -(-length // TILE) * TILE


def _kind(module, pair: torch.Tensor, mask: torch.Tensor | None) -> int | None:
    """FUSED, GENERIC or None (not served): the mathematical and resource contract, before touching the CUDA builder."""
    # The module resolves `miniworld` to its PORTABLE backend, which also hosts the B200 CUDA paths;
    # `_b200_cuda = False` keeps the Triton kernels (the benchmark baseline), as the `_fuse_*` switches do on H100.
    if (
        getattr(module, "_backend", None) != KernelBackend.PORTABLE
        or not getattr(module, "_b200_cuda", True)
        or settings.current().force_portable_kernels
    ):
        return None
    if not module.use_self_attention or module.use_qk_norm:
        return None
    if not pair.is_cuda or pair.dtype != torch.bfloat16 or pair.ndim != 4 or pair.shape[1] != pair.shape[2]:
        return None
    if pair.shape[1] < MIN_LENGTH:
        return None
    if mask is not None and (mask.dtype != torch.bool or mask.shape != pair.shape[:2]):
        return None
    d_hidden, d_pair = module.to_query.weight.shape
    if pair.shape[-1] != d_pair:
        return None
    if torch.cuda.get_device_capability(pair.device) != (10, 0):
        return None
    if module.n_head == 4 and d_pair == 128 and d_hidden == 128:
        return FUSED
    from miniworld_engine.kernels.triangle_attention.cuda.b200_triattn import (
        wide_supports,
    )

    if wide_supports(d_pair, d_hidden, module.n_head, _padded(pair.shape[1])):
        return GENERIC
    return None


def serves(module, pair: torch.Tensor, mask: torch.Tensor | None = None) -> bool:
    """The fused d_pair 128 / 4 x 32 path serves this call."""
    return _kind(module, pair, mask) == FUSED


def serves_wide(module, pair: torch.Tensor, mask: torch.Tensor | None = None) -> bool:
    """The other widths (d_pair 64 .. 512, heads of 16 / 32 channels) serve this call."""
    return _kind(module, pair, mask) == GENERIC


def kind_of(module, pair: torch.Tensor, mask: torch.Tensor | None = None) -> int | None:
    """``FUSED``, ``GENERIC`` or None (not served): the path a call takes, the key of ``max_batch``."""
    return _kind(module, pair, mask)


# ---------------------------------------------------------------- weight packs of the wide path (cached while the parameters are unchanged)
#: Packs keyed by the weights' tensor objects (held weakly), valid while every version is unchanged; scoped to a CUDA-graph capture by
#: ``kernels._capture`` (a capture packs once and records the packing kernels, so each replay repacks the current weights unless the
#: caller declared ``static_weights``). The fused path packs inside its forward (one launch).
_PACKS: dict = {}


def _packed(weights: tuple[torch.Tensor, ...]):
    from miniworld_engine.kernels.triangle_attention.cuda.b200_triattn import (
        pack_wide_weights,
    )

    def build():
        with torch.no_grad():
            return pack_wide_weights(*weights)

    try:
        versions = tuple(w._version for w in weights)
    except RuntimeError:  # inference tensors carry no version counter: nothing could invalidate an entry
        return build()
    key = _capture.scoped(tuple(id(w) for w in weights))
    if key is None:
        return build()
    hit = _PACKS.get(key)
    if hit is not None:
        refs, seen, pack = hit
        if seen == versions and all(r() is w for r, w in zip(refs, weights, strict=True)):
            return pack
    _capture.prune(_PACKS)
    if len(_PACKS) >= 64:  # a model's worth of blocks; drop the oldest
        _PACKS.pop(next(iter(_PACKS)))
    pack = build()
    _PACKS[key] = (tuple(weakref.ref(w) for w in weights), versions, pack)
    return pack


# ---------------------------------------------------------------- custom ops
# args = [pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, drop]; mask [B, L] bool and drop [B, L, C] bf16 are empty when absent; pair is the
# starting-node layout with L a multiple of 128.
def _infer_fake(kind, args, eps, n_head, head_dim, trans):
    return torch.empty_like(args[0])


@opaque(fake=_infer_fake, name="triattn_b200_infer")
def _infer_op(kind: int, args: list[torch.Tensor], eps: float, n_head: int, head_dim: int, trans: bool) -> torch.Tensor:
    return _infer(kind, args, eps, n_head, head_dim, trans)


def _infer(kind, args, eps, n_head, head_dim, trans, out=None):
    """The inference kernels; `out`: a contiguous tensor like the pair to write the result into (a chunk's rows of a batch's output,
    eager only: an op output may not alias an input)."""
    from miniworld_engine.kernels.triangle_attention.cuda import b200_triattn as k

    pair, lnw, lnb, *weights, mask, _ = args
    mask = mask if mask.numel() else None
    if kind == FUSED:
        y, _ = k.fused_forward(pair, lnw, lnb, *weights, mask, eps, None, False, out, trans)
        return y
    return k.wide_inference(pair, lnw, lnb, _packed(tuple(weights)), n_head, head_dim, mask, eps, out, trans)


def _fwd_fake(kind, args, eps, n_head, head_dim, trans):
    pair = args[0]
    B, L, _, C = pair.shape
    R = B * L * L
    kw: dict[str, Any] = {"device": pair.device, "dtype": torch.bfloat16}
    f32: dict[str, Any] = {"device": pair.device, "dtype": torch.float32}
    if kind == FUSED:
        return [
            torch.empty_like(pair),
            torch.empty(6, 128, **f32),                             # prm: lnw | lnb | Wb
            torch.empty(512, 128, **kw),                            # w4
            torch.empty(128, 128, **kw),                            # wo16
            *(torch.empty(R, 128, **kw) for _ in range(4)),         # q, k, v, g
            torch.empty(B, 4, L, L, **kw),                          # bias
            torch.empty(B, L, L, 4, 32, **kw),                      # o
            torch.empty(B, L, 4, L, **f32),                         # lse
        ]
    HDp = n_head * 32
    return [
        torch.empty_like(pair),
        *(torch.empty(R, HDp, **kw) for _ in range(4)),             # q, k, v, g
        torch.empty(B, n_head, L, L, **kw),                         # bias
        torch.empty(B, L, L, n_head, 32, **kw),                     # o
        torch.empty(B, L, n_head, L, **f32),                        # lse
        torch.empty(R, 2, **f32),                                   # LN statistics
    ]


@opaque(fake=_fwd_fake, name="triattn_b200_fwd")
def _fwd_op(kind: int, args: list[torch.Tensor], eps: float, n_head: int, head_dim: int, trans: bool) -> list[torch.Tensor]:
    from miniworld_engine.kernels.triangle_attention.cuda import b200_triattn as k

    pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, drop = args
    mask = mask if mask.numel() else None
    drop = drop if drop.numel() else None
    if kind == FUSED:
        out, saved = k.fused_forward(pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop, True, None, trans)
    else:
        out, saved = k.wide_forward(pair, lnw, lnb, _packed((wq, wk, wv, wg, wb, wo)), n_head, head_dim, mask, eps, drop, True, None,
                                    trans)
    return [out, *saved]


def _bwd_fake(kind, args, saved, eps, n_head, head_dim, trans):
    """The backward's stacked gradients (``b200_triattn.split_grads`` takes them apart): fused (dpair, [dgamma; dbeta; dWb],
    [dWq; dWk; dWv; dWg], dWo), wide (dpair, [dgamma; dbeta] fp32, [dWq; dWk; dWv; dWg], dWb, dWo); weight gradients fp32 when Wq is."""
    dy, pair, lnw, lnb, wq, wk, wv, wg, wb, wo, _ = args
    dt = torch.float32 if wq.dtype == torch.float32 else torch.bfloat16
    kw: dict[str, Any] = {"device": pair.device, "dtype": dt}
    C = pair.shape[-1]
    if kind == FUSED:
        return [torch.empty_like(pair), torch.empty(6, C, **kw), torch.empty(4, C, C, **kw), torch.empty(C, C, **kw)]
    hd = wq.shape[0]
    return [torch.empty_like(pair), torch.empty(2, C, device=pair.device, dtype=torch.float32), torch.empty(4, hd, C, **kw),
            torch.empty(n_head, C, **kw), torch.empty(C, hd, **kw)]


@opaque(fake=_bwd_fake, name="triattn_b200_bwd")
def _bwd_op(kind: int, args: list[torch.Tensor], saved: list[torch.Tensor], eps: float, n_head: int, head_dim: int,
            trans: bool) -> list[torch.Tensor]:
    from miniworld_engine.kernels.triangle_attention.cuda import b200_triattn as k

    dy, pair, lnw, lnb, wq, wk, wv, wg, wb, wo, drop = args
    drop = drop if drop.numel() else None
    dtypes = tuple(t.dtype for t in (lnw, lnb, wq, wk, wv, wg, wb, wo))
    if kind == FUSED:
        return list(k.fused_backward(dy, pair, eps, drop, list(saved), dtypes, trans))
    return list(k.wide_backward(dy, pair, lnw, lnb, _packed((wq, wk, wv, wg, wb, wo)), n_head, head_dim, eps, drop, list(saved), dtypes,
                                trans))


def _grads(kind, raw, params):
    from miniworld_engine.kernels.triangle_attention.cuda.b200_triattn import (
        split_grads,
    )

    return split_grads(kind == FUSED, raw, tuple(p.dtype for p in params))


class _TriAttnFn(torch.autograd.Function):
    """pair + drop o TriangleAttention(pair) for the starting node: one op forward (the tensors the backward reads come back as outputs),
    one op backward.  Gradients come back in the parameters' dtype."""

    @staticmethod
    def forward(ctx, kind, eps, n_head, head_dim, trans, pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, drop):
        out, *saved = _fwd_op(kind, [pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, drop], eps, n_head, head_dim, trans)
        ctx.save_for_backward(pair, lnw, lnb, wq, wk, wv, wg, wb, wo, drop, *saved)
        ctx.cfg = (kind, eps, n_head, head_dim, trans)
        return out

    @staticmethod
    def backward(ctx, dy):
        pair, lnw, lnb, wq, wk, wv, wg, wb, wo, drop, *saved = ctx.saved_tensors
        kind, eps, n_head, head_dim, trans = ctx.cfg
        raw = _bwd_op(kind, [dy.contiguous(), pair, lnw, lnb, wq, wk, wv, wg, wb, wo, drop], saved, eps, n_head, head_dim, trans)
        return (None, None, None, None, None, *_grads(kind, raw, (lnw, lnb, wq, wk, wv, wg, wb, wo)), None, None)


# ---------------------------------------------------------------- batched samples and chunking
#: Samples one launch takes, per path (``FUSED`` / ``GENERIC``, see ``kind_of``).  Every kernel of both paths takes B samples in one
#: launch: the front, tail, gate, head and weight-gradient kernels walk the B L^2 rows (b, i, j), the attention core and its backward
#: walk (sample, head, tile) tasks, the key mask [B, L] and the dropout scale [B, L, C] are read per sample, and the parameter gradients
#: are reduced over every sample of the launch.  The bound is what was validated (B up to 48 against B one-sample calls and the fp32
#: reference, d_pair 64-512, L 128-768); the kernels' only index bound is the int32 row coordinate of their TMA maps (and the int32 row
#: count of the front / tail launches), B Lp^2 < 2^31 (``max_batch`` applies it).  A call with more samples runs in several launches
#: (``plan``).
MAX_BATCH = {FUSED: 48, GENERIC: 48}


def max_batch(kind: int, length: int | None = None) -> int:
    """Largest number of samples one launch of the path ``kind`` (``FUSED`` / ``GENERIC``) takes, at pair length ``length`` when
    given; a call with more samples is split into launches run one after another (``plan``)."""
    limit = MAX_BATCH[kind]
    if length is not None:
        lp = _padded(length)
        limit = min(limit, (2**31 - 1) // (lp * lp))
    return max(1, limit)


#: Chunking.  A call is split when it has more samples than one launch takes (``max_batch``) or when its working set would not fit
#: CHUNK_MEMORY_FRACTION of the device: chunks of at most ``plan(...)[0]`` samples (balanced sizes) run one after another through the
#: same ops into one output (inference: written in place; training: each chunk's y and dx copied into the batch's), so a call never
#: runs out of memory because of its batch and never falls back to the PyTorch statements because of its size.  The chunk is a
#: deterministic function of the call's path, shapes, widths, node and mode and of the device's TOTAL memory (never of live free
#: memory), so torch.compile and CUDA-graph capture see the same split every time.  The working set is counted in bytes per pair token
#: (Lp^2 per sample) from the tensors each path allocates (``_bytes_per_token``; against the measured peaks of one launch, B200
#: 2026-10-20, seven widths x both nodes x padded / whole-tile L x B 1-4: -1.0 / +2.3 %, the -1 % fixed allocations at B = 1).  Training that does not fit in one launch is chunked in one of two ways (``_Chunked``): when
#: every chunk's saved tensors (held from the forward to the backward) fit next to one chunk's backward workspace, each chunk keeps
#: them; otherwise the backward reruns each chunk's forward first (checkpoint), so only the chunk being differentiated holds saved
#: tensors.  No split sheds the caller's own tensors (pair, the output, dy, dx): a call whose inputs and outputs alone exceed the
#: device still runs out of memory.
CHUNK_MEMORY_FRACTION = 0.85
#: cuBLAS workspaces, the weight packs and gradient workspaces, the allocator's rounding
_SLACK = 1 << 30


def _bytes_per_token(kind: int, d_pair: int, n_head: int, padded: bool, ending: bool) -> tuple[int, int, int]:
    """(inference working set, training saved tensors, training backward working set) in bytes per pair token of one sample, on top
    of the caller's tensors (pair and output; in training also dy and dx)."""
    c, h = d_pair, n_head
    # a padded length: the padded copy of the pair (saved in training), and of dy, and the padded output / dx before the cut; the copies
    # of the pair and of the output are not alive at the same peak (measured: +2c per token in inference and in each training phase).
    # The ending node reads and writes the caller's layout directly (no copies; `ending` kept for the callers)
    extra = 2 * c if padded else 0
    del ending
    if kind == FUSED:
        # inference: q | k | v | g, o; bias (bf16 per head)
        # saved: q | k | v | g, o; bias; lse (fp32 per head)   backward: do, dg, dq, dk, dv; delta, dbias, bias^T (fp32 per head)
        return 10 * c + 2 * h + extra, 10 * c + 6 * h + extra, 10 * c + 12 * h + extra
    hd = 32 * h  # heads padded to 32 channels
    # inference: q | k | v | g, o; bias
    # saved: q | k | v | g, o; bias; lse; the LN statistics (8 B)
    # backward: dy o drop; D = [dq | dk | dv | dg | 64 dbias columns]; do, u; delta, dbias, bias^T; [xhat | 1] (C + 64)
    return (
        10 * hd + 2 * h + extra,
        10 * hd + 6 * h + 8 + extra,
        4 * c + 12 * hd + 12 * h + 256 + extra,
    )


def plan(
    kind: int,
    batch: int,
    length: int,
    d_pair: int,
    n_head: int,
    training: bool,
    total_memory: int,
    ending: bool = False,
) -> tuple[int, bool]:
    """(chunk, checkpoint): the samples one launch takes and whether training chunks rerun their forward in the backward (instead of
    keeping their saved tensors); chunk >= batch is one call.
    A pure function of its arguments (``total_memory`` = the device's total memory in bytes)."""
    limit = max_batch(kind, length)
    lp = _padded(length)
    tokens = lp * lp
    work_inf, held, work_bwd = _bytes_per_token(kind, d_pair, n_head, lp != length, ending)
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


def forward(module, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """pair + drop o TriangleAttention(pair) for B samples: one launch per kernel, or launches of ``plan``'s chunk run one after
    another when the call has more samples than one launch takes or would not fit in memory."""
    kind = _kind(module, pair, mask)
    B, L, _, C = pair.shape
    drop = None
    if module.training and module.p_drop > 0.0:
        # the module's own draw on the original layout ([B, 1, L, C] starting / [B, L, 1, C] ending): in the kernels' starting
        # layout both index the row's last position, so either is [B, L, C] as it is
        drop = module._make_drop_scale(pair, module.p_drop).reshape(B, L, C).to(torch.bfloat16)
    needs_grad = torch.is_grad_enabled() and (pair.requires_grad or any(p.requires_grad for p in module.parameters()))
    index = pair.device.index if pair.device.index is not None else torch.cuda.current_device()
    ending = not module.starting
    chunk, checkpoint = plan(kind, B, L, C, module.n_head, needs_grad or drop is not None, _total_memory(index), ending)
    params = _params(module)
    head_dim = module.to_query.weight.shape[0] // module.n_head
    eps = module.ln_pair.eps
    # the ending node: the kernels read and write the caller's layout (trans)
    x, trans = pair, ending
    if chunk >= B:
        out = _forward_one(kind, params, x, mask, drop, needs_grad, eps, module.n_head, head_dim, trans)
    elif not needs_grad:
        out = _infer_chunks(kind, params, x, mask, drop, _split(B, chunk), eps, module.n_head, head_dim, trans)
    else:
        out = _Chunked.apply(
            kind, eps, module.n_head, head_dim, trans, tuple(_split(B, chunk)), checkpoint, x, *params,
            mask if mask is not None else pair.new_empty(0, dtype=torch.bool),
            drop if drop is not None else pair.new_empty(0),
        )
    return out.contiguous()


def _pad(x, mask, drop):
    """(x, mask, drop) padded to whole 128-token tiles: zero rows and columns, the pad keys masked (mask None -> a mask then)."""
    B, L = x.shape[:2]
    Lp = _padded(L)
    if Lp == L:
        return x, mask, drop
    x = F.pad(x, (0, 0, 0, Lp - L, 0, Lp - L))
    keep = torch.zeros(B, Lp, dtype=torch.bool, device=x.device)
    keep[:, :L] = True if mask is None else mask
    if drop is not None:
        drop = F.pad(drop, (0, 0, 0, Lp - L))
    return x, keep, drop


def _forward_one(kind, params, x, mask, drop, needs_grad, eps, n_head, head_dim, trans) -> torch.Tensor:
    """One launch of every kernel for the samples of ``x`` (the caller's layout, ``trans`` for the ending node; any strides).  Without grad or dropout: the inference
    op (bf16 weight packs); otherwise the autograd function (the parameters go in as they are and get gradients in their dtype)."""
    L = x.shape[1]
    xp, m, d = _pad(x, mask, drop)
    xp = xp.contiguous()
    mask_t = m if m is not None else xp.new_empty(0, dtype=torch.bool)
    drop_t = d if d is not None else xp.new_empty(0)
    if d is None and not needs_grad:
        out = _infer_op(kind, [xp, *params, mask_t, drop_t], eps, n_head, head_dim, trans)
    else:
        out = _TriAttnFn.apply(kind, eps, n_head, head_dim, trans, xp, *params, mask_t, drop_t)
    if xp.shape[1] != L:
        out = out[:, :L, :L]
    return out


def _infer_chunks(kind, params, x, mask, drop, sizes, eps, n_head, head_dim, trans) -> torch.Tensor:
    """Inference in chunks of ``sizes`` samples into one output (the caller's layout): a whole-tile length writes each chunk's rows in
    place (eager; the tail kernels take them as their output), otherwise (padding, a traced call) each chunk's result is copied in."""
    B, L = x.shape[:2]
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    into = drop is None and _padded(L) == L and not torch.compiler.is_compiling()
    start = 0
    for n in sizes:
        sl = slice(start, start + n)
        m = None if mask is None else mask[sl]
        if into:
            xc = x[sl].contiguous()
            args = [xc, *params, m if m is not None else xc.new_empty(0, dtype=torch.bool), xc.new_empty(0)]
            _infer(kind, args, eps, n_head, head_dim, trans, out[sl])
        else:
            out[sl] = _forward_one(kind, params, x[sl], m, None if drop is None else drop[sl], False, eps, n_head, head_dim, trans)
        start += n
    return out


def _chunk_inputs(x, mask, drop, start, n):
    """The chunk's pair, mask and dropout scale (the caller's layout), padded to whole tiles as `_forward_one` pads."""
    xc, m, d = _pad(x[start:start + n], mask[start:start + n] if mask.numel() else None, drop[start:start + n] if drop.numel() else None)
    return (xc.contiguous(), m if m is not None else xc.new_empty(0, dtype=torch.bool), d if d is not None else xc.new_empty(0))


class _Chunked(torch.autograd.Function):
    """Training on B samples in launches of ``sizes`` run one after another: one forward op per chunk (its result copied into one
    output, its saved tensors kept), one backward op per chunk (its dx copied into one gradient).  ``checkpoint``: no chunk keeps its
    saved tensors; the backward reruns the chunk's forward op first (the dropout scale is an input, so it is the same draw).  The
    parameters go to the ops in fp32 (an exact cast: the same packed bf16 weights), so the chunks' parameter gradients come back in
    fp32, add up in fp32 and are rounded to the parameters' dtype once, as inside one launch."""

    @staticmethod
    def forward(ctx, kind, eps, n_head, head_dim, trans, sizes, checkpoint, x, *rest):
        *params, mask, drop = rest
        p32 = [p.float() for p in params]
        L = x.shape[1]
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        saved: list[torch.Tensor] = []
        start = 0
        for n in sizes:
            xc, m, d = _chunk_inputs(x, mask, drop, start, n)
            o, *s = _fwd_op(kind, [xc, *p32, m, d], eps, n_head, head_dim, trans)
            out[start:start + n] = o[:, :L, :L]
            del o
            if not checkpoint:
                saved.extend(s)
            del s
            start += n
        ctx.save_for_backward(x, *params, mask, drop, *saved)
        ctx.cfg = (kind, eps, n_head, head_dim, trans, sizes, checkpoint)
        return out

    @staticmethod
    def backward(ctx, dy):
        x, *rest = ctx.saved_tensors
        params, (mask, drop), saved = rest[:8], rest[8:10], rest[10:]
        kind, eps, n_head, head_dim, trans, sizes, checkpoint = ctx.cfg
        p32 = [p.float() for p in params]
        L = x.shape[1]
        Lp = _padded(L)
        per = len(saved) // len(sizes)
        dx = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        acc = [torch.zeros(p.shape, dtype=torch.float32, device=p.device) for p in params]
        start = 0
        for c, n in enumerate(sizes):
            xc, m, d = _chunk_inputs(x, mask, drop, start, n)
            if checkpoint:
                _, *s = _fwd_op(kind, [xc, *p32, m, d], eps, n_head, head_dim, trans)
            else:
                s = list(saved[c * per:(c + 1) * per])
            g = dy[start:start + n]
            if Lp != L:
                g = F.pad(g, (0, 0, 0, Lp - L, 0, Lp - L))
            grads = _grads(kind, _bwd_op(kind, [g.contiguous(), xc, *p32, d], s, eps, n_head, head_dim, trans), p32)
            del s, g
            dx[start:start + n] = grads[0][:, :L, :L]
            for a, gp in zip(acc, grads[1:], strict=True):
                a += gp
            del grads
            start += n
        return (None, None, None, None, None, None, None, dx, *(a.to(p.dtype) for a, p in zip(acc, params, strict=True)), None, None)
