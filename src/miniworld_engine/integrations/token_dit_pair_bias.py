"""The attention biases of a stack of token DiT blocks, from one per-sample pair, in one pass.

Every block's bias is ``to_bias(LayerNorm(pair))`` with the block's own LayerNorm affine and projection, and only the affine and
the projection differ from block to block: the LayerNorm statistics are the same. So the pair is normalised ONCE (a warp per
row, ``pair_ln``), the blocks' ``to_bias.weight diag(ln_pair.weight) log2 e`` are stacked into one [blocks H, 128] matrix, and one
GEMM leaves [blocks H, A L L] -- block ``b``'s bias is the b-th group of H rows, which is exactly the head-major per-sample
layout ``token_dit_train.block_psb`` and the ``PSB`` attention cores read. Reading the pair once instead of once per block is the
point: the pair is [A, L, L, 128] (A = 48 augmentations), 1.8 GB at L = 384 in bf16, and a per-block LayerNorm + projection read
it twelve times each way.

The backward concatenates the blocks' bias gradients, so ``dpair`` and the blocks' ``ln_pair`` / ``to_bias`` weight gradients are
two GEMMs (dLN(pair) = dB^T Wf, dWf = dB LN(pair)) and the LayerNorm backward.
"""
from __future__ import annotations

import math

import torch

from miniworld_engine.integrations.token_dit_train import BF, EPS, _mm32

LOG2E = 1.0 / math.log(2.0)
D, DH = 768, 48
DP = 128


def _rows():
    from miniworld_engine.kernels.conditioned_transition.cuda.train import ext

    return ext(D, DH)


def _stacked(ln_weights, to_bias_weights):
    """[blocks H, 128] fp32: row (b, h) = to_bias_b[h] ln_pair_b log2 e."""
    return torch.cat([w.float() * l.float()[None, :] for l, w in zip(ln_weights, to_bias_weights, strict=True)]) * LOG2E


class _PairBiases(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pair, nblocks, *weights):  # noqa: D102
        ln_weights, to_bias_weights = weights[:nblocks], weights[nblocks:]
        A, L = pair.shape[0], pair.shape[1]
        R, T = A * L * L, _rows()
        heads = to_bias_weights[0].shape[0]
        p2 = pair.reshape(R, DP)
        ph = torch.empty(R, DP, device=pair.device, dtype=BF)
        pst = torch.empty(R, 2, device=pair.device, dtype=torch.float32)
        T.pair_ln(p2, ph, pst, EPS)
        wf = _stacked(ln_weights, to_bias_weights).to(BF)
        bias = torch.mm(wf, ph.t())                                        # [blocks H, R]
        ctx.save_for_backward(pair, ph, pst, *weights)
        ctx.meta = (nblocks, heads, A, L)
        return tuple(bias[b * heads : (b + 1) * heads].view(heads, A, L, L) for b in range(nblocks))

    @staticmethod
    def backward(ctx, *grads):  # noqa: D102
        pair, ph, pst, *weights = ctx.saved_tensors
        nblocks, heads, A, L = ctx.meta
        ln_weights, to_bias_weights = weights[:nblocks], weights[nblocks:]
        R, T = A * L * L, _rows()
        db = torch.empty(nblocks * heads, R, device=pair.device, dtype=BF)
        for b, g in enumerate(grads):
            rows = db[b * heads : (b + 1) * heads]
            if g is None:
                rows.zero_()
            else:
                rows.copy_(g.reshape(heads, R))
        wf = _stacked(ln_weights, to_bias_weights).to(BF)
        dph = _mm32(db.t(), wf)                                            # [R, 128] fp32
        dwf = _mm32(db, ph)                                                # [blocks H, 128] fp32
        dpair = torch.empty(R, DP, device=pair.device, dtype=pair.dtype)
        T.pair_ln_bwd(dph, pair.reshape(R, DP), pst, dpair)
        d_ln, d_w = [], []
        for b in range(nblocks):
            g = dwf[b * heads : (b + 1) * heads]
            ln, w = ln_weights[b].float(), to_bias_weights[b].float()
            d_w.append((g * ln[None, :] * LOG2E).to(to_bias_weights[b].dtype))
            d_ln.append(((g * w).sum(0) * LOG2E).to(ln_weights[b].dtype))
        return (dpair.view(pair.shape), None, *d_ln, *d_w)


def pair_biases(pair, ln_weights, to_bias_weights):
    """Every block's attention bias from the per-sample pair [A, L, L, 128]: a tuple of [H, A, L, L] bf16 tensors in log2 units
    (``bias_b = (to_bias_b LN(pair) ln_pair_b) log2 e``, no key mask), differentiable in the pair and in each block's
    ``ln_pair.weight`` / ``to_bias.weight``. The tuple's tensors are views of one [blocks H, A L L] buffer."""
    return _PairBiases.apply(pair, len(ln_weights), *ln_weights, *to_bias_weights)


__all__ = ["pair_biases"]
