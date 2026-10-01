"""Automatic dispatch to the B200 (sm_100a) AttentionPairBias: inference and training, all hand CUDA but the plain GEMMs
(cuBLAS).

Contract: bf16 activations, B = 1, d_single 384 as 8 heads x 48, 12 x 32, 24 x 16 or 16 x 24 (AF3's Pairformer single attention),
or d_single 512 as 16 x 32, d_pair
128, no QK-norm, ``implementation="miniworld"`` (engine backend not forced to Triton), sm_100. Inference: L a multiple of 16;
training: L a multiple of 128.

16 x 24 runs 32 wide per head: the packed q | k | v | g weights carry 8 zero rows after each head's 24 (and Wo 8 zero columns),
so q, k, v, g, O and every gradient on that side are W = 512 wide with exact zeros in the pads; the attention kernels see heads
of 32 (-DDHP=32) and the softmax scale of 24. ``finalize`` gathers the real rows / columns of the weight gradients.

Inference (one opaque op):
  ln_rows (LN(single), bf16) -> addmm q | k | v | g (q pre-scaled by log2(e) / sqrt(head dim): logits in exp2 units)
  -> pair_bias (LN(pair) . Wf, Wf = to_bias.weight * ln_pair.weight * log2(e), masked keys -1e30)
  -> attn_inf -DQPAIR (sigmoid(g) o over the q columns) -> to_out accumulated in place (cuBLAS beta = 1) onto the residual
  seed ln_rows wrote (out-of-place addmm would add a copy of x into the output first).
ln_pair's bias adds Wb . b to every (i, j) of a head: the softmax cancels it, so the forward drops it (its gradient is kept).

Training (autograd Function, opaque forward / backward): the same forward in natural units with attn_fwd2 -DQPAIR (O, LSE),
gate_rows and addmm; backward: dO GEMMs, gate_bwd (dO, D, dg), attn_dkv / attn_dqb -DNHEAD, qkv_bwd, the dxa / dW GEMMs,
ln_bwd (+ residual), pair_bias_bwd. A key masked for every query (the whole sample) gets the uniform softmax's gradient on
its pair bias where the PyTorch module's masked_fill gives zero."""

from __future__ import annotations

import math

import torch
from torch.autograd.function import once_differentiable

from miniworld_engine import settings
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.modules.exceptions import ImplementationType

DP = 128
SHAPES = ((8, 384), (12, 384), (16, 384), (24, 384), (16, 512))   # (heads, d_single): 8 x 48, 12 x 32, 16 x 24 (padded to 32), 24 x 16, 16 x 32
LOG2E = math.log2(math.e)
NEG = -1e30                     # masked keys: finite, so a fully masked sample stays NaN-free
_packs: dict = {}
_cores: dict = {}


def _eligible(module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None) -> bool:
    if module.implementation != ImplementationType.MINIWORLD or settings.current().engine_backend == "triton":
        return False
    H, D = module.n_head, single.shape[-1]
    if module.use_qk_norm or (H, D) not in SHAPES or tuple(module.to_query.weight.shape) != (D, D):
        return False
    if tuple(module.to_bias.weight.shape) != (H, DP) or module.ln_single.weight is None or module.ln_pair.weight is None:
        return False
    if not single.is_cuda or single.dtype != torch.bfloat16 or pair.dtype != torch.bfloat16:
        return False
    if single.ndim != 3 or single.shape[0] != 1 or single.shape[2] != D or not single.is_contiguous():
        return False
    L = single.shape[1]
    if tuple(pair.shape) != (1, L, L, DP) or not pair.is_contiguous():
        return False
    if mask is not None and tuple(mask.shape) != (1, L):
        return False
    return torch.cuda.get_device_capability(single.device) == (10, 0)


def serves_inference(module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None = None) -> bool:
    return not torch.is_grad_enabled() and single.shape[1] % 16 == 0 and _eligible(module, single, pair, mask)


def serves_train(module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None = None) -> bool:
    return torch.is_grad_enabled() and single.shape[1] % 128 == 0 and _eligible(module, single, pair, mask)


def _leaves(module) -> list[torch.Tensor]:
    """ln_single w / b (fp32), Wq, bq, Wk, Wv, Wg, Wo (bf16), ln_pair w / b (fp32), Wb (bf16): the casts stay in autograd so
    the module's own parameters receive the gradients."""
    bf = torch.bfloat16
    return [module.ln_single.weight.float(), module.ln_single.bias.float(), module.to_query.weight.to(bf),
            module.to_query.bias.to(bf), module.to_key.weight.to(bf), module.to_value.weight.to(bf), module.to_gate.weight.to(bf),
            module.to_out.weight.to(bf), module.ln_pair.weight.float(), module.ln_pair.bias.float(), module.to_bias.weight.to(bf)]


def _rows():
    from miniworld_engine.kernels.augmented_attention.cuda import apb as rows
    return rows


def _sm100():
    from miniworld_engine.kernels.augmented_attention.cuda import sm100
    return sm100


def _mask(mask: torch.Tensor | None, L: int) -> torch.Tensor | None:
    return None if mask is None else mask.reshape(L).to(torch.bool).contiguous()


def _geometry(leaves):
    """(heads, real head dim, row width W, d_single) from to_bias.weight [heads, 128] and ln_single.weight [d_single]."""
    H, D = leaves[10].shape[0], leaves[0].numel()
    return H, D // H, _rows().width(H, D), D


# ------------------------------------------------------------------------------------------------------------ inference
def _inference_packs(leaves):
    """``_prep`` with q scaled into exp2 units and Wf in exp2 units, cached per parameter version (the step reads them every
    call; repacking costs a launch)."""
    key = tuple((t.data_ptr(), t._version) for t in leaves)
    hit = _packs.get(key)
    if hit is None:
        if len(_packs) >= 8:
            _packs.clear()
        hit = _packs[key] = (*_prep(leaves, LOG2E / math.sqrt(_geometry(leaves)[1]), LOG2E), leaves)
    return hit[:4]


def _prep(leaves, qs: float, ws: float):
    """(W q | k | v | g [4 W, D] (padded rows, q x qs), its bias [4 W], Wf = to_bias.weight * ln_pair.weight * ws [H, 128], Wo
    [D, W] with padded columns -- or None when there is no padding (8 x 48: to_out.weight as is)), bf16, one launch."""
    _, _, wq, bq, wk, wv, wg, wo, lnpw, _, wb = leaves
    H, _, W, D = _geometry(leaves)
    wpack = wq.new_empty((4 * W, D))
    bvec = wq.new_empty((4 * W,))
    wf = wq.new_empty((H, DP))
    wop = wq.new_empty((D, W)) if W != D else None
    _rows().ext().prep(wq, bq, wk, wv, wg, wo if wop is not None else None, wb, lnpw, wpack, bvec, wop, wf, qs, ws)
    return wpack, bvec, wf, wop


def _inference_fake(single, pair, mask, leaves, eps_s, eps_p):
    return torch.empty_like(single)


@opaque(fake=_inference_fake, name="apb_b200_inference")
def inference(single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, leaves: list[torch.Tensor], eps_s: float,
              eps_p: float) -> torch.Tensor:
    """single [1, L, 384], pair [1, L, L, 128] bf16, mask [L] bool or None, leaves as ``_leaves`` -> single + APB(single)."""
    L = single.shape[1]
    rows, dev = _rows(), single.device
    lnw, lnb = leaves[0], leaves[1]
    H, _, W, D = _geometry(leaves)
    with torch.cuda.device(dev):
        x = single.view(L, D)
        wpack, bvec, wf, wop = _inference_packs(leaves)
        wo = leaves[7] if wop is None else wop
        xa = torch.empty_like(x)
        y = torch.empty_like(x)                   # residual seed, written by ln_rows; to_out accumulates onto it
        rows.ext().ln_rows(x, lnw, lnb, xa, None, eps_s, y)
        qkvg = torch.addmm(bvec, xa, wpack.t())
        bias = rows.pair_bias(pair.view(L * L, DP), wf, mask, L, eps_p, NEG)
        core = _cores.get((dev.index, H, D))
        if core is None:
            core = _cores[(dev.index, H, D)] = _sm100().ApbInferenceCore(dev.index, H, D)
        core(qkvg, bias)
        return y.addmm_(qkvg[:, :W], wo.t()).view(1, L, D)


def update_inference(module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    return inference(single, pair, _mask(mask, single.shape[1]), _leaves(module), module.ln_single.eps, module.ln_pair.eps)


# ------------------------------------------------------------------------------------------------------------- training
def _forward_fake(single, pair, mask, leaves, eps_s, eps_p):
    """y and the saved set: xa, (mean, rstd), qkvg, bias, O, LSE, og, Wf, W q | k | v | g (and the padded Wo for 16 x 24)."""
    L = single.shape[1]
    H, _, W, D = _geometry(leaves)
    e, f32 = single.new_empty, torch.float32
    out = [torch.empty_like(single), e((L, D)), e((L, 2), dtype=f32), e((L, 4 * W)), e((H, L, L)), e((L, W), dtype=f32),
           e((H, L), dtype=f32), e((L, W)), e((H, DP)), e((4 * W, D))]
    return out + ([e((D, W))] if W != D else [])


@opaque(fake=_forward_fake, name="apb_b200_train_fwd")
def forward(single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, leaves: list[torch.Tensor], eps_s: float,
            eps_p: float) -> list[torch.Tensor]:
    L = single.shape[1]
    rows = _rows()
    lnw, lnb = leaves[0], leaves[1]
    H, _, W, D = _geometry(leaves)
    with torch.cuda.device(single.device):
        x = single.view(L, D)
        wpack, bvec, wf, wop = _prep(leaves, 1.0, 1.0)
        wo = leaves[7] if wop is None else wop
        xa = torch.empty_like(x)
        st = x.new_empty((L, 2), dtype=torch.float32)
        y = torch.empty_like(x)
        rows.ext().ln_rows(x, lnw, lnb, xa, st, eps_s, y)
        qkvg = torch.addmm(bvec, xa, wpack.t())
        bias = rows.pair_bias(pair.view(L * L, DP), wf, mask, L, eps_p, NEG)
        O, LSE = _sm100().apb_forward(qkvg[:, :W], qkvg[:, W:2 * W], qkvg[:, 2 * W:3 * W], bias, L, H, D)
        og = x.new_empty((L, W))
        rows.ext().gate_rows(O, qkvg, og)
        y = y.addmm_(og, wo.t()).view(1, L, D)   # in place onto the residual seed: no copy of x into the output
    return [y, xa, st, qkvg, bias, O, LSE, og, wf, wpack] + ([wop] if wop is not None else [])


def _backward_fake(single, pair, mask, leaves, eps_s, eps_p, saved, dy):
    """One gradient per input and leaf, contiguous, in its dtype."""
    return [torch.empty(t.shape, dtype=t.dtype, device=t.device) for t in (single, pair, *leaves)]


@opaque(fake=_backward_fake, name="apb_b200_train_bwd")
def backward(single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, leaves: list[torch.Tensor], eps_s: float,
             eps_p: float, saved: list[torch.Tensor], dy: torch.Tensor) -> list[torch.Tensor]:
    L = single.shape[1]
    rows = _rows()
    E = rows.ext()
    lnw, lnpw, wb = leaves[0], leaves[8], leaves[10]
    H, _, W, D = _geometry(leaves)
    xa, st, qkvg, bias, O, LSE, og, wf, wpack = saved[:9]
    wo = saved[9] if W != D else leaves[7]                 # [D, W]
    f32 = torch.float32
    with torch.cuda.device(single.device):
        x = single.view(L, D)
        dy2 = dy.contiguous().view(L, D)
        acc = torch.zeros(rows.acc_n(H, D), device=x.device)    # dlnw | dlnb | dbq | dWf | head sums, added by the kernels
        dog = dy2 @ wo                                       # [L, W]: zero in the pads (Wo's pad columns are zero)
        dwo = torch.mm(dy2.t(), og, out_dtype=f32)           # [D, W]; finalize keeps the real columns
        dob = x.new_empty((L, W))
        dd = x.new_empty((H, L), dtype=f32)
        dqkvg = torch.empty_like(qkvg)
        E.gate_bwd(dog, O, qkvg, dob, dd, dqkvg)
        DQ, DK, DV, DB = _sm100().apb_backward(qkvg[:, :W], qkvg[:, W:2 * W], qkvg[:, 2 * W:3 * W], dob, bias, LSE, dd, L, H, D)
        E.qkv_bwd(DQ, DK, DV, dqkvg, acc, D)
        del DQ, DK, DV
        dxa = dqkvg @ wpack
        dwp = torch.mm(dqkvg.t(), xa, out_dtype=f32)
        dx = torch.empty_like(x)
        E.ln_bwd(dxa, x, st, lnw, dy2, dx, acc)
        dz, _, _ = rows.pair_bias_bwd(pair.view(L * L, DP), DB, wf, L, eps_p, acc, D)
        # custom-op outputs may not alias: every parameter gradient leaves as its own tensor, written by one kernel. ln_pair's
        # bias moves a head's logits by one constant: sum_j dbias[h, i, j] = 0 for every query, so its gradient and its share
        # of dWb are exactly 0 (the accumulated head sums are that 0 plus rounding noise; not used)
        outs = [torch.empty(t.shape, dtype=t.dtype, device=t.device) for t in leaves]
        E.finalize(dwp, dwo, acc, wb, lnpw, outs)
    return [dx.view(single.shape), dz.view(pair.shape), *outs]


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eps_s, eps_p, mask, single, pair, *leaves):
        y, *saved = forward(single, pair, mask, list(leaves), eps_s, eps_p)
        ctx.eps = (eps_s, eps_p)
        ctx.save_for_backward(mask, single, pair, *leaves, *saved)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        v = ctx.saved_tensors
        mask, single, pair, leaves, saved = v[0], v[1], v[2], list(v[3:14]), list(v[14:])          # saved: 10 or 11 tensors
        grads = backward(single, pair, mask, leaves, *ctx.eps, saved, dy)
        return (None, None, None, *grads)


def update_train(module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    return _Training.apply(module.ln_single.eps, module.ln_pair.eps, _mask(mask, single.shape[1]), single, pair, *_leaves(module))
