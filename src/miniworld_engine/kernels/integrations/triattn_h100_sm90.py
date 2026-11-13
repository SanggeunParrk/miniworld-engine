"""Automatic dispatch to the H100 (sm_90a) TriangleAttention, bf16, inference and training (with the module's broadcast dropout):
d_pair = d_hidden = 128 with 4 heads x 32 on the fused wgmma kernels (``kernels/triangle_attention/cuda/h100_sm90.py``), every other
(d_pair <= 512, heads of 16 / 32 channels) on the width-generic path (``h100_sm90_generic.py``), L a multiple of 16 (padded to 128).
Training serves every TriangleAttention variant (``_train_kind``): the standard module at d128 x 4 heads fused, the rest (AF3 / RF3 /
Chai-1 bias flip, projection biases, untransposed / dual output, heads of 64 channels on the h100_dh core) on the width-generic path.
Any number of samples B (templates, structures or augments folded into the batch): up to ``max_batch`` per launch, more (or more than
fit in memory) in launches run one after another (``plan``).

The kernel launches are ``opaque`` custom ops (``kernels._compile``): the module compiles to one graph, the weight packs and the launches
run eagerly inside the ops (also under a CUDA-graph capture), and the autograd function below calls one op forward and one backward."""

from __future__ import annotations

import itertools
from typing import Any

import torch
import torch.nn.functional as F

from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import device_constant, opaque

FUSED, GENERIC = 0, 1
#: variant bits of the training ops (``_variant``)
FLIP, UNTRANSPOSED, DUAL = 1, 2, 4


def _fused_shape(module, pair) -> bool:
    """d_pair = d_hidden = 128, 4 heads x 32: the fused kernels (front / tail / backward on wgmma).  d_pair = d_hidden = 256, 8 heads
    x 32 (``h100_sm90.WIDE``): the wide fused front, inference only (with grad or training dropout: the width-generic kernels)."""
    if module.n_head == 4 and pair.shape[-1] == 128 and tuple(module.to_query.weight.shape) == (128, 128):
        return True
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as fused

    d_hidden, d_pair = module.to_query.weight.shape
    return pair.shape[-1] == d_pair and fused.wide(d_pair, d_hidden, module.n_head) and not _trains(module, pair)


def _trains(module, pair) -> bool:
    """The call takes the autograd function (``forward``'s ``needs_grad``) or applies the module's training dropout."""
    if module.training and module.p_drop > 0.0:
        return True
    return torch.is_grad_enabled() and (pair.requires_grad or any(p.requires_grad for p in module.parameters()))


def _kind(module, pair: torch.Tensor, mask: torch.Tensor | None) -> int | None:
    """FUSED, GENERIC or None (not served): the mathematical and resource contract, before touching the CUDA builder."""
    # The caller (the module's "cuda" backend, modules/triangle.py) chose this path; `_h100_cuda = False` still opts a module out.
    if not getattr(module, "_h100_cuda", True):
        return None
    if not module.use_self_attention or module.use_qk_norm:
        return None
    if _trains(module, pair):  # grad or live dropout: the training kernels
        return _train_kind(module, pair, mask)
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


def _variant(module) -> int:
    """The module's variant bits for the training ops: FLIP (the ending node's bias from the pair as it is: AF3 / RF3 / Chai-1),
    UNTRANSPOSED (Chai-1's ending node returns its update in the transposed layout), DUAL (Chai-1's confidence head)."""
    v = FLIP if module._bias_untransposed else 0
    if getattr(module, "dual_output", False):
        v |= DUAL
    elif not module.starting and getattr(module, "untransposed_output", False):
        v |= UNTRANSPOSED
    return v


def _out_mode(variant: int) -> int:
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90_generic as gen

    return gen.DUAL if variant & DUAL else gen.UNTRANSPOSED if variant & UNTRANSPOSED else gen.STANDARD


def _train_kind(module, pair: torch.Tensor, mask: torch.Tensor | None) -> int | None:
    """Training (grad or live dropout): FUSED for the standard module at d_pair = d_hidden = 128 x 4 heads (no projection biases, the
    standard bias orientation and output); every other served shape / variant on the width-generic kernels (heads of 16 / 32 channels on
    the wgmma core, 64 on the h100_dh core; the variants' bias flip, projection biases and output layouts in the glue)."""
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
    if getattr(module, "dual_output", False) and module.to_out_t.bias is not None:
        return None  # no class has a biased second out-projection
    from miniworld_engine.kernels.triangle_attention.cuda import (
        h100_sm90_generic as gen,
    )

    if not gen.supports(d_pair, d_hidden, module.n_head, pair.shape[1], train=True):
        return None
    plain = _variant(module) == 0 and all(b is None for b in _biases(module))
    if plain and module.n_head == 4 and d_pair == 128 and d_hidden == 128:
        return FUSED
    return GENERIC


def serves(module, pair: torch.Tensor, mask: torch.Tensor | None = None) -> bool:
    return _kind(module, pair, mask) is not None


# ---------------------------------------------------------------- weight packs (``kernels._capture.param_forms``: cached while the parameters are unchanged)
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

    # inference-only packs (the training ops take the parameters as they are): a capture is served the pack its eager warm-up made
    # (``capture_reads_eager``, the portable kernels' contract), so a replay runs no packing kernel
    return _capture.param_forms(("triattn_h100_sm90", kind), weights, build, capture_reads_eager=True)


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
    return _infer(kind, args, eps, n_head, head_dim, trans, None, flip)


def _infer(kind, args, eps, n_head, head_dim, trans, out=None, flip=False):
    """The inference kernels; `out`: a contiguous tensor like the pair to write the result into (a chunk's rows of a batch's output,
    eager only: an op output may not alias an input)."""
    pair, lnw, lnb, *weights = args[:9]
    mask = args[9]
    gate_bias, out_bias = (t if t.numel() else None for t in args[11:13]) if len(args) > 11 else (None, None)
    mask = mask if mask.numel() else None
    blocks = _rows_blocks(kind, pair, n_head)
    if blocks > 1:
        wcat, wo_p = _packed(GENERIC, tuple(weights))
        return _generic_rows(pair, lnw, lnb, wcat, wo_p, n_head, head_dim, mask, eps, trans, out, flip, gate_bias, out_bias, blocks)
    if kind == FUSED:
        from miniworld_engine.kernels.triangle_attention.cuda.h100_sm90 import inference

        # the wide pack (d_pair 256) folds the LayerNorm in: its parameters are part of the pack's key
        wide = pair.shape[-1] != 128
        w4, wb16, wo16, *fold = _packed(kind, (*weights, lnw, lnb) if wide else tuple(weights))
        return inference(pair, lnw.float(), lnb.float(), w4, wb16, wo16, mask, eps, trans, out, flip, gate_bias, out_bias, *fold)
    from miniworld_engine.kernels.triangle_attention.cuda import (
        h100_sm90_generic as gen,
    )

    wcat, wo_p = _packed(kind, tuple(weights))
    return gen.inference(pair, lnw, lnb, wcat, wo_p, n_head, head_dim, mask, eps, trans, out, flip, gate_bias, out_bias)


def _rows_blocks(kind: int, pair: torch.Tensor, n_head: int) -> int:
    """Blocks of pair rows one sample's inference runs in: 1 while a sample's working set on the call's path fits the inference cap
    (``INFERENCE_WORKSET_FRACTION``), else the fewest equal blocks whose width-generic working set fits it.  A pure function of the
    shapes and the device's total memory."""
    L, C = pair.shape[1], pair.shape[3]
    index = pair.device.index if pair.device.index is not None else torch.cuda.current_device()
    cap = INFERENCE_WORKSET_FRACTION * _total_memory(index)
    if _bytes_per_token(kind, C, n_head, False)[0] * L * L <= cap:
        return 1
    blocks = -(-_bytes_per_token(GENERIC, C, n_head, False)[0] * L * L // int(cap))
    if blocks >= 3:
        # far past the cap (FoldForge OpenDDE's refiner: d_pair 384 at 2976 / 3968 tokens; the requirement sizes up to L2048 take 1 - 2
        # blocks): blocks of at most ROWS_BLOCK_BYTES of q | k | v | g, so an allocator holding a fragmented model can place them (a 9 GiB
        # block ran out of memory with 26 GiB reserved but free in the n = 16 cell's second fold)
        blocks = max(blocks, -(-L * L * 4 * 32 * n_head * 2 // ROWS_BLOCK_BYTES))
    return min(L, blocks)


#: The largest q | k | v | g block of a row-blocked inference far past the working-set cap (``_rows_blocks``).
ROWS_BLOCK_BYTES = 2 << 30


@torch.autocast("cuda", enabled=False)  # the fp16 core takes the fp16 projections as made: an enclosing bf16 autocast turned the
# q | k | v | g GEMM bf16 and the core raised (FoldForge's 5I28 run, OpenDDE at 2048 tokens: the first call split into row blocks)
def _generic_rows(pair, lnw, lnb, wcat, wo_p, n_head, head_dim, mask, eps, trans, out, flip, gate_bias, out_bias, blocks):
    """Inference of one sample at a time in ``blocks`` blocks of its pair rows, on the width-generic kernels (``_rows_blocks``).

    Rows of triangle attention are independent but share the bias, so the bias comes first: LayerNorm and the bias projection block by
    block of the pair's rows into one [L^2, Hp] tensor, then the scaled / masked head-major bias of the whole sample.  Then per block: LayerNorm again,
    q | k | v | g, the core on the block's rows against every key (it takes N rows != S keys), the gate and the out-projection with the
    residual, written into the block's rows of the output.  The ending node's rows are the pair's columns: its blocks are copied into
    row order and their results copied back (only in this path).  The kernels and the arithmetic per row are the one-launch path's; the
    GEMMs of fewer rows may pick other cuBLAS algorithms."""
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as core_mod
    from miniworld_engine.kernels.triangle_attention.cuda import (
        h100_sm90_generic as gen,
    )

    B, L, _, C = pair.shape
    H, D = n_head, gen.D
    HD = H * D
    g = gen._glue()
    h16 = core_mod.INFERENCE_FP16  # as the one-launch path at L > 256: fp16 projections, bias and core
    ww = gen._half(wcat) if h16 else wcat
    w4, wb = ww[: 4 * HD], ww[4 * HD :]
    lw, lb = lnw.float().contiguous(), lnb.float().contiguous()
    scale = head_dim**-0.5
    pair = pair.contiguous()
    out = torch.empty_like(pair) if out is None else out
    bo = None if out_bias is None else out_bias.to(torch.bfloat16).contiguous()
    gb = None
    if gate_bias is not None:
        hd = gate_bias.numel() // H
        gb = F.pad(gate_bias.view(H, hd), (0, D - hd))
    core = core_mod._core_h16().triattn_fwd_h16 if h16 else core_mod._core().triattn_fwd
    lt = gen._lt()
    bounds = [L * i // blocks for i in range(blocks + 1)]
    blocks_of = list(itertools.pairwise(bounds))
    for b in range(B):
        rows = pair[b].transpose(0, 1) if trans else pair[b]  # the sample's logical rows [i, j, C]

        def block(i0, i1, rows=rows):
            return rows[i0:i1].contiguous().view(-1, C)  # a copy only for the ending node

        # the bias of every token, block by block over the pair's own rows (no copies), then into the logical order (ending node:
        # the small [L, L, Hp] transposed)
        bias = torch.empty(L * L, wb.shape[0], device=pair.device, dtype=ww.dtype)
        for i0, i1 in blocks_of:
            y = g.ln_fwd(pair[b, i0:i1].reshape(-1, C), lw, lb, eps, L, False, h16)[0]
            torch.matmul(y, wb.t(), out=bias[i0 * L : i1 * L])
        if trans:
            bias = bias.view(L, L, -1).transpose(0, 1).reshape(L * L, -1)
        m = None if mask is None else mask[b : b + 1]
        front_mask = m if m is not None and not flip else pair.new_empty(0, dtype=torch.bool)
        sbias = g.front_bias(bias, front_mask, 1, L, H, 0, scale)
        del bias
        if flip:
            sbias = core_mod.flip_bias(sbias, m)
        for i0, i1 in blocks_of:
            R = i1 - i0
            x = block(i0, i1)
            y = g.ln_fwd(x, lw, lb, eps, L, False, h16)[0]
            qkvg = torch.matmul(y, w4.t())  # [R L, 4 HD]
            del y
            ld = qkvg.shape[1]
            q, k, v, gt = (qkvg.view(-1).as_strided((1, R, L, H, D), (R * L * ld, L * ld, ld, D, 1), i * HD) for i in range(4))
            if gb is not None:
                gt.add_(gb.to(gt.dtype))
            kw = {"key_tiles": core_mod.key_tiles(m, L)} if h16 else {}  # the dead trailing key tiles skipped
            u = core(q, k, v, sbias, scale, False, gt, False, **kw)[3]
            del qkvg, q, k, v, gt
            if trans:
                res = lt.lt_mm(u.view(R * L, HD), wo_p, x, L, False, None, bo)
                out[b, :, i0:i1] = res.view(R, L, C).transpose(0, 1)
            else:
                lt.lt_mm(u.view(R * L, HD), wo_p, x, L, False, out[b, i0:i1], bo)
    return out


# training ops: args = [pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, drop, gb, bo, wt]; mask, drop and the variants' gate bias, out
# bias and second out-projection (gb [H hd], bo [C], wt [C, H hd]) are empty when absent.  ``variant``: the bits of ``_variant``.
def _fwd_fake(kind, args, eps, n_head, head_dim, trans, variant):
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
    from miniworld_engine.kernels.triangle_attention.cuda.h100_sm90_generic import core_width

    dc = core_width(head_dim)
    HD = n_head * dc
    return [
        torch.empty_like(pair),
        torch.empty(M, C, **kw),
        torch.empty(M, **f32),
        torch.empty(M, **f32),
        torch.empty(M, 4 * HD, **kw),
        torch.empty(B, n_head, L, L, **kw),
        torch.empty(B, L, L, n_head, dc, **kw),
        torch.empty(B, L, n_head, L, **f32),
        torch.empty(B, L, L, n_head, dc, **kw),
    ]


def _opt(t: torch.Tensor) -> torch.Tensor | None:
    return t if t.numel() else None


@opaque(fake=_fwd_fake, name="triattn_h100_fwd")
def _fwd_op(
    kind: int,
    args: list[torch.Tensor],
    eps: float,
    n_head: int,
    head_dim: int,
    trans: bool,
    variant: int,
) -> list[torch.Tensor]:
    pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, drop, gb, bo, wt = args
    mask, drop = _opt(mask), _opt(drop)
    if kind == FUSED:
        from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as fused

        out, saved = fused.train_forward(pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop, True, trans)
        return [out, *saved]
    from miniworld_engine.kernels.triangle_attention.cuda import (
        h100_sm90_generic as gen,
    )

    out, saved = gen.train_forward(
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
        eps,
        drop,
        n_head,
        head_dim,
        trans,
        bool(variant & FLIP),
        _opt(gb),
        _opt(bo),
        _opt(wt),
        _out_mode(variant),
    )
    return [out, *saved]


def _bwd_fake(kind, args, saved, eps, n_head, head_dim, trans, variant):
    # pair, lnw, lnb, wq, wk, wv, wg, wb, wo, gb, bo, wt
    return [torch.empty_like(t) for t in (*args[1:10], *args[11:14])]


@opaque(fake=_bwd_fake, name="triattn_h100_bwd")
def _bwd_op(
    kind: int,
    args: list[torch.Tensor],
    saved: list[torch.Tensor],
    eps: float,
    n_head: int,
    head_dim: int,
    trans: bool,
    variant: int,
) -> list[torch.Tensor]:
    dy, pair, lnw, lnb, wq, wk, wv, wg, wb, wo, drop, gb, bo, wt = args
    drop = _opt(drop)
    if kind == FUSED:
        from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as fused

        grads = fused.train_backward(dy, pair, lnw, lnb, wq, wk, wv, wg, wb, wo, eps, drop, tuple(saved), trans)
        return [*grads, *(t.new_empty(0) for t in (gb, bo, wt))]
    from miniworld_engine.kernels.triangle_attention.cuda import (
        h100_sm90_generic as gen,
    )

    grads = gen.train_backward(
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
        bool(variant & FLIP),
        _opt(gb),
        _opt(bo),
        _opt(wt),
        _out_mode(variant),
    )
    return [
        g if g is not None else t.new_empty(0) for g, t in zip(grads, (pair, lnw, lnb, wq, wk, wv, wg, wb, wo, gb, bo, wt), strict=True)
    ]


#: parameters of the training ops, in their argument order (``_train_params``)
N_TRAIN_PARAMS = 11


class _TriAttnFn(torch.autograd.Function):
    """pair + drop o TriangleAttention(pair): one op forward (the tensors the backward reads come back as outputs), one op backward.
    Gradients come back in the parameters' dtype.  ``params`` = ``_train_params`` (absent variant parameters are empty tensors)."""

    @staticmethod
    def forward(ctx, kind, eps, n_head, head_dim, trans, variant, pair, *rest):
        params, (mask, drop) = rest[:N_TRAIN_PARAMS], rest[N_TRAIN_PARAMS:]
        args = [pair, *params[:8], mask, drop, *params[8:]]
        out, *saved = _fwd_op(kind, args, eps, n_head, head_dim, trans, variant)
        ctx.save_for_backward(pair, *params, drop, *saved)
        ctx.cfg = (kind, eps, n_head, head_dim, trans, variant)
        return out

    @staticmethod
    def backward(ctx, dy):
        pair, *rest = ctx.saved_tensors
        params, drop, saved = rest[:N_TRAIN_PARAMS], rest[N_TRAIN_PARAMS], rest[N_TRAIN_PARAMS + 1 :]
        kind, eps, n_head, head_dim, trans, variant = ctx.cfg
        grads = _bwd_op(kind, [dy, pair, *params[:8], drop, *params[8:]], saved, eps, n_head, head_dim, trans, variant)
        grads = [g if g.numel() else None for g in grads]  # empty: an absent variant parameter
        return (None, None, None, None, None, None, *grads, None, None)


def _padded(length: int) -> int:
    return -(-length // 128) * 128


# ---------------------------------------------------------------- batched samples and chunking
#: Samples one launch takes, per path (``FUSED`` / ``GENERIC``).  Every kernel of both paths takes B samples in one
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
#: Inference working-set cap: an inference launch's own tensors (on top of the caller's pair and output) take at most this share of
#: the device's total memory.  The budget above treats the call as the device's only tenant, but an inference call runs inside a
#: model that holds its activations, caches and the allocator's free blocks: FoldBench's 2432-token targets (Protenix's 4 templates on
#: the d_pair-64 stack, one 22.6 GiB q | k | v | g block) ran out of memory with 34 GiB in use and 44 GiB reserved but fragmented
#: (2026-10-30).  More samples than the cap takes run in sample chunks (``plan``); one sample that alone exceeds it runs in blocks of
#: pair rows (``_rows_blocks``, ``_generic_rows``).  Like the budget, a function of the device's total memory only.
INFERENCE_WORKSET_FRACTION = 0.25


def _bytes_per_token(kind: int, d_pair: int, n_head: int, padded: bool, dc: int = 32) -> tuple[int, int, int]:
    """(inference working set, training saved tensors, training backward working set) in bytes per pair token of one sample, on top
    of the caller's tensors (pair and output; in training also dy and dx).  ``dc``: channels per head of the core (64: the h100_dh
    core of 64-channel heads, training; its query-side backward also holds up to ``h100_dh.MAX_SLABS`` fp32 dbias slabs)."""
    c, h = d_pair, n_head
    # a length padded to 128: the padded copy of the pair (saved in training) and of dy, the padded output / dx before the cut
    pad = 4 * c if padded else 0
    if kind == FUSED:
        # inference: q | k | v | g, o, the output; sbias (bf16 per head); the wide path (d_pair 256: u instead of o) also cublasLt's
        # tuning copy of the output
        # saved: q | k | v | g, o; sbias; lse (fp32 per head)   backward: do, dg, dq, dk, dv; delta, dbias, its slab sum (fp32 per head)
        wide = 2 * c if c != 128 else 0
        return 12 * c + 2 * h + pad + wide, 10 * c + 6 * h + pad // 2, 10 * c + 12 * h + pad
    # heads padded to 32 channels, the bias columns to 8
    hd, hp = dc * h, (h + 7) // 8 * 8
    slabs = 4 * h * 8 if dc != 32 else 0
    # inference: y, the LN statistics, q | k | v | g | bias, sbias, u = sigmoid(g) o O, the output and cublasLt's tuning copy of it
    # saved: y, statistics, q | k | v | g, sbias, o, lse, u
    # backward: ddrop, dyln; du, do; dall = [dq | dk | dv | dg | dbias]; delta, dbias, its slab sum (fp32)
    return (
        6 * c + 10 * hd + 2 * hp + 2 * h + 8 + pad,
        2 * c + 12 * hd + 6 * h + 8 + pad // 2,
        4 * c + 12 * hd + 2 * hp + 12 * h + pad + slabs,
    )


def plan(
    kind: int,
    batch: int,
    length: int,
    d_pair: int,
    n_head: int,
    training: bool,
    total_memory: int,
    dc: int = 32,
) -> tuple[int, bool]:
    """(chunk, checkpoint): the samples one launch takes and whether training chunks rerun their forward in the backward (instead of
    keeping their saved tensors); chunk >= batch is one call.
    A pure function of its arguments (``total_memory`` = the device's total memory in bytes; ``dc`` the core's head width)."""
    limit = max_batch(kind, length)
    lp = _padded(length)
    tokens = lp * lp
    work_inf, held, work_bwd = _bytes_per_token(kind, d_pair, n_head, lp != length, dc)
    budget = CHUNK_MEMORY_FRACTION * total_memory - _SLACK
    c = d_pair

    def fit(free: float, per_sample: int) -> int:
        return max(1, min(limit, int(free // (per_sample * tokens))))

    if not training:
        # the caller's pair and the output (the chunks write into it); the launch's own tensors within the inference cap
        resident = 4 * c
        cap = INFERENCE_WORKSET_FRACTION * total_memory
        if batch * (resident + work_inf) * tokens <= budget and batch * work_inf * tokens <= cap:
            return min(batch, limit), False
        return fit(min(budget - batch * resident * tokens, cap), work_inf), False
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
    assert kind is not None  # the caller (serves) checked it
    B, L, _, C = pair.shape
    drop = None
    if module.training and module.p_drop > 0.0:
        # the module's own draw on the original layout ([B, 1, L, C] starting / [B, L, 1, C] ending): in the kernels' starting
        # layout both index the row's last position, so either is [B, L, C] as it is. One H100 Philox launch (h100_triattn.drop_scale:
        # the same tensor, bit for bit, as the module's torch draw for the same generator state)
        from miniworld_engine.kernels.triangle_attention.cuda import h100_triattn

        if h100_triattn.drop_scale_serves(pair, module.p_drop):
            drop = h100_triattn.drop_scale(pair, module.p_drop, module._drop_dim).reshape(B, L, C)
        else:
            drop = module._make_drop_scale(pair, module.p_drop).reshape(B, L, C).to(torch.bfloat16)
    needs_grad = torch.is_grad_enabled() and (pair.requires_grad or any(p.requires_grad for p in module.parameters()))
    index = pair.device.index if pair.device.index is not None else torch.cuda.current_device()
    head_dim = module.to_query.weight.shape[0] // module.n_head
    chunk, checkpoint = plan(
        kind,
        B,
        L,
        C,
        module.n_head,
        needs_grad or drop is not None,
        _total_memory(index),
        64 if head_dim > 32 else 32,
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
                    *_bias_args(module, x),
                ]
                _infer(
                    kind,
                    args,
                    module.ln_pair.eps,
                    module.n_head,
                    head_dim,
                    not module.starting,
                    out[sl],
                    (not module.starting) and not module.transposed_bias,
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
        head_dim,
        not module.starting,
        _variant(module),
        tuple(sizes),
        checkpoint,
        pair,
        *_train_params(module, pair),
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
    def forward(ctx, kind, eps, n_head, head_dim, trans, variant, sizes, checkpoint, pair, *rest):
        *params, mask, drop = rest
        p32 = [p.float() for p in params]
        L = pair.shape[1]
        out = torch.empty(pair.shape, dtype=pair.dtype, device=pair.device)
        saved: list[torch.Tensor] = []
        start = 0
        for n in sizes:
            x, m, d = _chunk_inputs(pair, mask, drop, start, n)
            o, *s = _fwd_op(kind, [x, *p32[:8], m, d, *p32[8:]], eps, n_head, head_dim, trans, variant)
            out[start : start + n] = o[:, :L, :L]
            del o
            if not checkpoint:
                saved.extend(s)
            del s
            start += n
        ctx.save_for_backward(pair, *params, mask, drop, *saved)
        ctx.cfg = (kind, eps, n_head, head_dim, trans, variant, sizes, checkpoint)
        return out

    @staticmethod
    def backward(ctx, dy):
        pair, *rest = ctx.saved_tensors
        n_p = N_TRAIN_PARAMS
        params, (mask, drop), saved = rest[:n_p], rest[n_p : n_p + 2], rest[n_p + 2 :]
        kind, eps, n_head, head_dim, trans, variant, sizes, checkpoint = ctx.cfg
        p32 = [p.float() for p in params]
        L = pair.shape[1]
        Lp = _padded(L)
        per = len(saved) // len(sizes)
        dx = torch.empty(pair.shape, dtype=pair.dtype, device=pair.device)
        acc = [torch.zeros(p.shape, dtype=torch.float32, device=p.device) for p in params]
        start = 0
        for c, n in enumerate(sizes):
            x, m, d = _chunk_inputs(pair, mask, drop, start, n)
            if checkpoint:
                _, *s = _fwd_op(kind, [x, *p32[:8], m, d, *p32[8:]], eps, n_head, head_dim, trans, variant)
            else:
                s = list(saved[c * per : (c + 1) * per])
            g = dy[start : start + n]
            if Lp != L:
                g = F.pad(g, (0, 0, 0, Lp - L, 0, Lp - L))
            grads = _bwd_op(kind, [g.contiguous(), x, *p32[:8], d, *p32[8:]], s, eps, n_head, head_dim, trans, variant)
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
            None,
            dx,
            *(a.to(p.dtype) if p.numel() else None for a, p in zip(acc, params, strict=True)),
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


def _train_params(module, like: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """``_params`` + (gate bias, out bias, second out-projection): the training ops' parameters, an empty tensor for an absent one."""
    wt = module.to_out_t.weight if getattr(module, "dual_output", False) else None
    extra = (*_biases(module), wt)
    return (*_params(module), *(t if t is not None else like.new_empty(0) for t in extra))


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
    # AF3's ending-node bias: transposed between the front and the core (inference; see ``_kind``)
    flip = trans and not module.transposed_bias
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
            [pair, *params, mask_t, drop_t, *_bias_args(module, pair)],
            eps,
            module.n_head,
            head_dim,
            trans,
            flip,
        )
    else:
        out = _TriAttnFn.apply(
            kind,
            eps,
            module.n_head,
            head_dim,
            trans,
            _variant(module),
            pair,
            *_train_params(module, pair),
            mask_t,
            drop_t,
        )
    if Lp != L:
        out = out[:, :L, :L]
    return out.contiguous()


def _biases(module) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """(gate bias, out-projection bias): RoseTTAFold3's TriangleAttention has both, every other model neither."""
    return module.to_gate.bias, module.to_out.bias


def _bias_args(module, like: torch.Tensor) -> list[torch.Tensor]:
    """The inference op's trailing arguments: the two biases, an empty tensor for an absent one."""
    return [b.detach() if b is not None else like.new_empty(0) for b in _biases(module)]
