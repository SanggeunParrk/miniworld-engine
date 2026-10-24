"""The portable path of the DiT block variants (Boltz-2, RoseTTAFold3, Chai-1): the structure of ``reference.py`` with its attention
cores on the engine's Triton kernels, for the cards that have no hand-written CUDA path (``interface.py`` is the H100 one).

    token stack   ``kernels/augmented_attention/triton/main.py`` (pair-bias attention over the whole sequence, differentiable)
    atom stack    ``kernels/augmented_attention/triton/window.py`` (32 queries x 128 keys per window, per-window bias, the slid / clipped
                  key window, Chai-1's same-conformer mask, differentiable)

What it changes against the reference, besides the cores:

  * the pair's LayerNorm statistics are computed ONCE for the stack (the reference normalises the pair again in every block, 24 times
    for a token stack); each block's LayerNorm weight is folded into its projection (``W diag(gamma) zhat + W beta``), and the bias of
    all the blocks comes out of one GEMM;
  * the logits and probabilities of the cores never reach HBM (the reference holds ``[A, H, L, L]`` fp32 scores and probabilities per
    block, or ``[A, H, nwin, 32, 128]`` for the atoms).

The row operations (AdaLN, residuals, gates, the SwiGLU) stay torch ops in the parameters' dtype with fp32 norms, exactly the
reference's, so the whole path is differentiable by autograd and gives the same function (bf16 / fp16 within the cores' tolerance, fp32
on TF32 tensor cores).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from miniworld_engine.kernels.dit_variants import reference as _ref

_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
MAX_HEAD_DIM = 64


def _leg_allowed(stack, single, op: str) -> bool:
    """The Triton leg of the ``miniworld`` backend for ``op`` (``dit_variant`` / ``dit_variant_atom`` of ``autotune.records``): not on
    H100 / B200 (PyTorch is the fallback there by design) and, off them, only on a card the engine holds a record for -- every kernel
    the call launches has an ``ok`` manifest row, the backward ones only when the call needs a gradient; otherwise one
    ``UnrecordedCardWarning`` and the reference. ``settings.force_portable_kernels`` allows it everywhere."""
    from miniworld_engine.modules import dispatch
    from miniworld_engine.modules.exceptions import ImplementationType

    if stack.implementation != ImplementationType.MINIWORLD:
        return False
    return dispatch.portable_leg_allowed(single.device, op, training=dispatch.needs_backward(stack, single))


def serves(stack, single, cond, pair, mask) -> bool:
    """Token stack: ``single`` [A, L, D] on a CUDA device in bf16 / fp16 / fp32, head width <= 64, ``cond`` [L, c] or [A, L, c], ``pair``
    [L, L, C], ``mask`` [L] or None; and the card is one the ``miniworld`` rule sends to the Triton kernels (:func:`_leg_allowed`)."""
    if not single.is_cuda or single.ndim != 3 or single.dtype not in _DTYPES:
        return False
    _, length, d = single.shape
    blk = stack.blocks[0]
    return (
        d % blk.n_head == 0
        and d // blk.n_head <= MAX_HEAD_DIM
        and cond.shape[-2] == length
        and tuple(pair.shape[:2]) == (length, length)
        and (mask is None or tuple(mask.shape) == (length,))
        and blk.expand.weight.shape[1] == d
        and _leg_allowed(stack, single, "dit_variant")
    )


def atom_serves(stack, single, cond, pair, mask, uid) -> bool:
    """Atom stack: ``single`` [A, N, 128] with N a multiple of 32, ``cond`` [N, 128] or [A, N, 128], trunked ``pair`` [N / 32, 32, 128, c],
    ``mask`` / ``uid`` [N] or None; and the card is one the ``miniworld`` rule sends to the Triton kernels (:func:`_leg_allowed`)."""
    if not single.is_cuda or single.ndim != 3 or single.dtype not in _DTYPES:
        return False
    _, n, d = single.shape
    return (
        n % _ref.ATOM_QUERIES == 0
        and d % _ref.ATOM_HEADS == 0
        and d // _ref.ATOM_HEADS <= MAX_HEAD_DIM
        and cond.shape[-2:] == (n, d)
        and tuple(pair.shape[:3]) == (n // _ref.ATOM_QUERIES, _ref.ATOM_QUERIES, _ref.ATOM_KEYS)
        and (mask is None or tuple(mask.shape) == (n,))
        and (uid is None or tuple(uid.shape) == (n,))
        and _leg_allowed(stack, single, "dit_variant_atom")
    )


# ---------------------------------------------------------------------------------------------------------------------- pair bias
def _fold(norms, projections, dtype):
    """(W' [sum of rows, C], const [sum of rows] or None): ``W diag(gamma)`` per block, and ``W beta`` when a norm has a bias (None when
    none has one: decided by the parameters' presence, never by their values, so a CUDA-graph capture or a compile sees no host sync)."""
    ws, consts = [], []
    for norm, proj in zip(norms, projections, strict=True):
        w = proj.float() * norm.weight.float()[None, :]
        ws.append(w.to(dtype))
        consts.append(proj.float() @ norm.bias.float() if norm.bias is not None else torch.zeros(proj.shape[0], device=proj.device))
    has_bias = any(norm.bias is not None for norm in norms)
    return torch.cat(ws), (torch.cat(consts) if has_bias else None)


def _normalised(pair, eps):
    """LayerNorm of the pair rows without the affine (fp32 statistics, result in the pair's dtype)."""
    return F.layer_norm(pair, pair.shape[-1:], eps=eps)


def _needs_grad(*tensors) -> bool:
    return torch.is_grad_enabled() and any(t.requires_grad for t in tensors)


def _fused_biases(pair, weight, const, eps):
    """``[rows of weight, pair rows]`` = ``weight`` (the blocks' ``W diag(gamma)`` stacked) applied to the pair's LayerNorm, in one pass over
    the pair (``layernorm_linear_triton_fwd_t``: stored [heads, rows], the layout the attention kernels read), plus ``const``."""
    from miniworld_engine.kernels.layernorm_linear.triton.fused import (
        layernorm_linear_triton_fwd_t,
    )

    ones = torch.ones(pair.shape[-1], device=pair.device, dtype=torch.float32)
    out = layernorm_linear_triton_fwd_t(pair, ones, torch.zeros_like(ones), weight, eps)
    return out if const is None else (out.float() + const[:, None]).to(out.dtype)


#: Blocks whose biases one pass over the pair makes at once (their heads' rows are held together): 8 x 16 heads = 128 rows of the pair.
_BLOCKS_PER_PASS = 8


def token_pair_biases(stack, pair, first: int = 0, count: int | None = None):
    """The biases [H, L, L] (natural units, the pair's dtype) of blocks ``first .. first + count``. A call that needs no gradient makes
    them in one fused pass over the pair per ``_BLOCKS_PER_PASS`` blocks; one that does normalises the pair once (torch) and projects."""
    blocks = list(stack.blocks)[first:first + (count or len(stack.blocks))]
    length, c = pair.shape[0], pair.shape[-1]
    h = blocks[0].n_head
    eps = blocks[0].pair_norm.eps
    w, const = _fold([b.pair_norm for b in blocks], [b.to_pair_bias.weight for b in blocks], pair.dtype)
    weights = [b.to_pair_bias.weight for b in blocks] + [p for b in blocks for p in (b.pair_norm.weight, b.pair_norm.bias) if p is not None]
    if _needs_grad(pair, *weights):
        zhat = _normalised(pair, eps).reshape(length * length, c)
        out = torch.mm(w.to(zhat.dtype), zhat.t())                                # [nb H, L L], block i at rows [H i, H (i + 1))
        if const is not None:
            out = (out.float() + const[:, None]).to(pair.dtype)
    else:
        out = _fused_biases(pair, w, const, eps)
    return [out[h * i:h * (i + 1)].view(h, length, length) for i in range(len(blocks))]


def atom_pair_biases(stack, z):
    """Per block, the bias [4, nwin, 32, 128] contiguous in the pair's dtype (natural units)."""
    blocks = list(stack.blocks)
    nb, heads = len(blocks), _ref.ATOM_HEADS
    norm = blocks[0].pair_norm if stack.per_block_pair_norm else stack.pair_norm
    if stack.per_block_pair_norm:
        w, const = _fold([b.pair_norm for b in blocks], [b.to_pair_bias.weight for b in blocks], z.dtype)
        weights = [b.to_pair_bias.weight for b in blocks] + [p for b in blocks for p in (b.pair_norm.weight, b.pair_norm.bias) if p is not None]
    else:
        w, const = _fold([stack.pair_norm] * nb, [stack.to_pair_bias.weight[heads * i:heads * (i + 1)] for i in range(nb)], z.dtype)
        weights = [p for p in (stack.to_pair_bias.weight, stack.pair_norm.weight, stack.pair_norm.bias) if p is not None]
    nwin = z.shape[0]
    if _needs_grad(z, *weights):
        zhat = _normalised(z, norm.eps)
        out = torch.mm(w, zhat.reshape(-1, z.shape[-1]).t())                      # [4 nb, nwin 32 128]
        if const is not None:
            out = (out.float() + const[:, None]).to(z.dtype)
    else:
        out = _fused_biases(z, w, const, norm.eps)
    return [out[heads * i:heads * (i + 1)].view(heads, nwin, _ref.ATOM_QUERIES, _ref.ATOM_KEYS) for i in range(nb)]


# ---------------------------------------------------------------------------------------------------------------------- token stack
def attention_delta(block, x, s, bias, mask):
    from miniworld_engine.kernels.augmented_attention.triton.main import (
        TritonAugmentedAttentionFunction,
    )

    a = _ref.adaln(block.attn_adaln, x, s, block.adaln_eps)
    q, k, v = block.to_q(a), block.to_k(a), block.to_v(a)
    if block.kq_norm:
        q, k = (F.layer_norm(t.float(), t.shape[-1:], n.weight.float(), n.bias.float(), n.eps).to(t.dtype)
                for t, n in ((q, block.q_norm), (k, block.k_norm)))
    n_aug, length, d_model = x.shape
    heads = block.n_head
    q, k, v = (t.reshape(n_aug, 1, length, heads, d_model // heads) for t in (q, k, v))
    key_mask = None if mask is None else mask.bool()[None, None].expand(n_aug, 1, length)
    bias = bias.to(q.dtype).permute(1, 2, 0)[None]                                # [1, L, L, H] (a view of the contiguous [H, L, L])
    o = TritonAugmentedAttentionFunction.apply(q, k, v, bias, key_mask).reshape(n_aug, length, d_model)
    if block.gating:
        o = torch.sigmoid(block.to_g(a)) * o
    return torch.sigmoid(block.attn_zero(s)) * block.to_out(o)


def stack_forward(stack, x, s, pair, mask):
    blocks = list(stack.blocks)
    for first in range(0, len(blocks), _BLOCKS_PER_PASS):
        chunk = blocks[first:first + _BLOCKS_PER_PASS]
        for block, bias in zip(chunk, token_pair_biases(stack, pair, first, len(chunk)), strict=True):
            da = attention_delta(block, x, s, bias, mask)
            if stack.parallel:
                x = x + da + _ref.transition_delta(block, x, s)
            else:
                x = x + da
                x = x + _ref.transition_delta(block, x, s)
    return x


# ---------------------------------------------------------------------------------------------------------------------- atom stack
def atom_block_forward(stack, i, x, c, bias, mask, uid, starts, keep):
    from miniworld_engine.kernels.augmented_attention.triton.window import (
        window_attention,
    )

    blk = stack.blocks[i]
    a, n, d_model = x.shape
    heads = _ref.ATOM_HEADS
    xq, xk = _ref.adaln(blk.q_adaln, x, c, blk.adaln_eps), _ref.adaln(blk.k_adaln, x, c, blk.adaln_eps)
    q, k, v = blk.to_q(xq), blk.to_k(xk), blk.to_v(xk)
    if blk.kq_norm:
        q, k = (F.layer_norm(t.float(), t.shape[-1:], m.weight.float(), m.bias.float(), m.eps).to(t.dtype)
                for t, m in ((q, blk.q_norm), (k, blk.k_norm)))
    q, k, v = (t.reshape(a, 1, n, heads, d_model // heads) for t in (q, k, v))
    o = window_attention(q, k, v, bias[None], starts, keep, uid=uid, keys=_ref.ATOM_KEYS).reshape(a, n, d_model)
    if blk.gating:
        o = torch.sigmoid(blk.to_g(xq)) * o
    if blk.project_output:
        o = blk.to_out(o)
    da = torch.sigmoid(blk.attn_zero(c)) * o
    if stack.parallel:
        return x + da + _ref.transition_delta(blk, x, c)
    x = x + da
    return x + _ref.transition_delta(blk, x, c)


def atom_stack_forward(stack, x, c, z, mask=None, uid=None):
    from miniworld_engine.kernels.augmented_attention.triton.window import plan

    a, n, _ = x.shape
    nwin = n // _ref.ATOM_QUERIES
    keep = torch.ones(1, n, dtype=torch.bool, device=x.device) if mask is None else mask.bool().reshape(1, n)
    starts = plan(stack.key_window, n, nwin, _ref.ATOM_KEYS, keep, x.device)
    uid_b = None if uid is None else uid.reshape(1, n)
    for i, bias in enumerate(atom_pair_biases(stack, z)):
        x = atom_block_forward(stack, i, x, c, bias, mask, uid_b, starts, keep)
    return x
