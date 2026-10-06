"""Automatic dispatch to the B200 (sm_100a) AttentionPairBias: inference and training, all hand CUDA but the plain GEMMs
(cuBLAS).

H100 inference and training use native CUDA by default for the required buckets.
The original B200 exact-zero LN beta gradient fails the FP32 oracle cosine, so
H100 training reconstructs that FP32 rounding with native CUDA and cuBLAS. This
accuracy-first shadow materializes extra FP32 workspaces: the B200 tensor
lifetimes and former fast-path latency do not describe this training default.
MINIWORLD_APB_H100_TRAIN=0 retains the explicit legacy comparison path; disabling
MINIWORLD_APB_H100_BETA_SHADOW selects the known-inaccurate experimental gradient.

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
ln_pair has no bias: an offset would add Wb . b to every (i, j) of a head, which the softmax cancels.

Training (autograd Function, opaque forward / backward): the same forward in natural units with attn_fwd2 -DQPAIR (O, LSE),
gate_rows and addmm; backward: dO GEMMs, gate_bwd (dO, D, dg), attn_dkv / attn_dqb -DNHEAD, qkv_bwd, the dxa / dW GEMMs,
ln_bwd (+ residual), pair_bias_bwd. B200 retains its finite masked logits. H100 uses
-inf in the native attention core, matching the FP32 efficient-SDPA reference's
zero attention update for an all-false key mask; its beta shadow also applies
masked_fill's zero derivative at every masked key."""

from __future__ import annotations

import math
import os

import torch
from torch.autograd.function import once_differentiable

from miniworld_engine import settings
from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.modules.exceptions import ImplementationType

DP = 128
SHAPES = ((8, 384), (12, 384), (16, 384), (24, 384), (16, 512))   # (heads, d_single): 8 x 48, 12 x 32, 16 x 24 (padded to 32), 24 x 16, 16 x 32
LOG2E = math.log2(math.e)
NEG = -1e30                     # masked keys: finite, so a fully masked sample stays NaN-free
_packs: dict = {}
_cores: dict = {}


def native_training_enabled() -> bool:
    """Validated H100 CUDA training; explicit zero retains the legacy baseline."""
    return os.environ.get("MINIWORLD_APB_H100_TRAIN", "1") == "1"


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
    capability = torch.cuda.get_device_capability(single.device)
    return capability == (10, 0) or (capability == (9, 0) and L % 128 == 0
                                    and (not torch.is_grad_enabled() or native_training_enabled()))


def serves_inference(module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None = None) -> bool:
    return not torch.is_grad_enabled() and single.shape[1] % 16 == 0 and _eligible(module, single, pair, mask)


def serves_train(module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None = None) -> bool:
    return torch.is_grad_enabled() and single.shape[1] % 128 == 0 and _eligible(module, single, pair, mask)


def _leaves(module) -> list[torch.Tensor]:
    """The module's parameters as they are (any float dtype, e.g. an fp32 master): ln_single w / b, Wq, bq, Wk, Wv, Wg, Wo,
    ln_pair w, Wb. The ops cast them (``_kernel_leaves``) outside autograd, so fp32 parameters get unrounded fp32 gradients."""
    return [module.ln_single.weight, module.ln_single.bias, module.to_query.weight, module.to_query.bias, module.to_key.weight,
            module.to_value.weight, module.to_gate.weight, module.to_out.weight, module.ln_pair.weight,
            module.to_bias.weight]


def _kernel_leaves(leaves) -> list[torch.Tensor]:
    """The kernels' dtypes: the LayerNorm vectors fp32, the projections bf16 (no copies when the parameters already are)."""
    if torch.cuda.get_device_capability(leaves[0].device) == (9, 0):
        return _rows().ext().cast_leaves(leaves)
    bf = torch.bfloat16
    lnw, lnb, wq, bq, wk, wv, wg, wo, lnpw, wb = leaves
    return [lnw.float(), lnb.float(), wq.to(bf), bq.to(bf), wk.to(bf), wv.to(bf), wg.to(bf), wo.to(bf), lnpw.float(),
            wb.to(bf)]


def _rows():
    from miniworld_engine.kernels.augmented_attention.cuda import apb as rows
    return rows


def _sm100():
    from miniworld_engine.kernels.augmented_attention.cuda import sm100
    return sm100


def _training_core(device):
    if torch.cuda.get_device_capability(device) == (9, 0):
        from miniworld_engine.kernels.augmented_attention import cuda as sm90
        return sm90
    return _sm100()


def _inference_core(device, heads: int, d: int):
    if torch.cuda.get_device_capability(device) == (9, 0):
        from miniworld_engine.kernels.augmented_attention import cuda as sm90
        return sm90.ApbInferenceCore(device.index, heads, d)
    return _sm100().ApbInferenceCore(device.index, heads, d)


def _mask(mask: torch.Tensor | None, L: int) -> torch.Tensor | None:
    return None if mask is None else mask.reshape(L).to(torch.bool).contiguous()


def _geometry(leaves):
    """(heads, real head dim, row width W, d_single) from to_bias.weight [heads, 128] and ln_single.weight [d_single]."""
    H, D = leaves[9].shape[0], leaves[0].numel()
    return H, D // H, _rows().width(H, D), D


# ------------------------------------------------------------------------------------------------------------ inference
def _inference_packs(params, leaves):
    """``_prep`` with q scaled into exp2 units and Wf in exp2 units, cached per parameter version (the step reads them every
    call; repacking costs a launch). Keyed on the parameters themselves (``params``), not on ``leaves``, their kernel-dtype casts,
    which are fresh tensors per call for fp32 parameters. Scoped to the CUDA-graph capture (``kernels._capture``): a capture packs
    once, recorded, so each replay packs the weights as they are then."""
    build = lambda: (*_prep(leaves, LOG2E / math.sqrt(_geometry(leaves)[1]), LOG2E), params)
    return _capture.lookup(_packs, tuple((t.data_ptr(), t._version) for t in params), build, limit=8)[:4]


def _prep(leaves, qs: float, ws: float):
    """(W q | k | v | g [4 W, D] (padded rows, q x qs), its bias [4 W], Wf = to_bias.weight * ln_pair.weight * ws [H, 128], Wo
    [D, W] with padded columns -- or None when there is no padding (8 x 48: to_out.weight as is)), bf16, one launch."""
    _, _, wq, bq, wk, wv, wg, wo, lnpw, wb = leaves
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
    params = leaves
    leaves = _capture.param_forms("apb_leaves", params, lambda: tuple(_kernel_leaves(params)))
    lnw, lnb = leaves[0], leaves[1]
    H, _, W, D = _geometry(leaves)
    with torch.cuda.device(dev):
        x = single.view(L, D)
        wpack, bvec, wf, wop = _inference_packs(params, leaves)
        wo = leaves[7] if wop is None else wop
        xa = torch.empty_like(x)
        y = torch.empty_like(x)                   # residual seed, written by ln_rows; to_out accumulates onto it
        rows.ext().ln_rows(x, lnw, lnb, xa, None, eps_s, y)
        qkvg = torch.addmm(bvec, xa, wpack.t())
        neg = -math.inf if torch.cuda.get_device_capability(dev) == (9, 0) else NEG
        bias = rows.pair_bias(pair.view(L * L, DP), wf, mask, L, eps_p, neg)
        core = _cores.get((dev.index, H, D))
        if core is None:
            core = _cores[(dev.index, H, D)] = _inference_core(dev, H, D)
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
        neg = -math.inf if torch.cuda.get_device_capability(single.device) == (9, 0) else NEG
        bias = rows.pair_bias(pair.view(L * L, DP), wf, mask, L, eps_p, neg)
        O, LSE = _training_core(single.device).apb_forward(qkvg[:, :W], qkvg[:, W:2 * W], qkvg[:, 2 * W:3 * W], bias, L, H, D)
        og = x.new_empty((L, W))
        rows.ext().gate_rows(O, qkvg, og)
        y = y.addmm_(og, wo.t()).view(1, L, D)   # in place onto the residual seed: no copy of x into the output
    return [y, xa, st, qkvg, bias, O, LSE, og, wf, wpack] + ([wop] if wop is not None else [])


def _backward_fake(single, pair, mask, leaves, eps_s, eps_p, saved, dy):
    """One gradient per input (its dtype) and per leaf (fp32), contiguous."""
    return [torch.empty_like(single), torch.empty_like(pair), *(torch.empty(t.shape, dtype=torch.float32, device=t.device) for t in leaves)]


@opaque(fake=_backward_fake, name="apb_b200_train_bwd")
def backward(single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, leaves: list[torch.Tensor], eps_s: float,
             eps_p: float, saved: list[torch.Tensor], dy: torch.Tensor) -> list[torch.Tensor]:
    L = single.shape[1]
    rows = _rows()
    E = rows.ext()
    lnw, lnpw, wb = leaves[0], leaves[8], leaves[9]
    H, _, W, D = _geometry(leaves)
    xa, st, qkvg, bias, O, LSE, og, wf, wpack = saved[:9]
    wo = saved[9] if W != D else leaves[7]                 # [D, W]
    f32 = torch.float32
    with torch.cuda.device(single.device):
        x = single.view(L, D)
        dy2 = dy.contiguous().view(L, D)
        if torch.cuda.get_device_capability(x.device) == (9, 0):
            # Every producer owns a block partial; finalize reduces in block order.
            nblocks = max(2 * torch.cuda.get_device_properties(x.device).multi_processor_count, (L + rows.RW - 1) // rows.RW)
            acc = torch.empty(nblocks, rows.acc_n(H, D), device=x.device)
            E.zero_acc(acc)
        else:
            acc = torch.zeros(rows.acc_n(H, D), device=x.device)
        dog = dy2 @ wo                                       # [L, W]: zero in the pads (Wo's pad columns are zero)
        dwo = torch.mm(dy2.t(), og, out_dtype=f32)           # [D, W]; finalize keeps the real columns
        dob = x.new_empty((L, W))
        dd = x.new_empty((H, L), dtype=f32)
        dqkvg = torch.empty_like(qkvg)
        E.gate_bwd(dog, O, qkvg, dob, dd, dqkvg)
        DQ, DK, DV, DB = _training_core(single.device).apb_backward(qkvg[:, :W], qkvg[:, W:2 * W], qkvg[:, 2 * W:3 * W], dob, bias, LSE, dd, L, H, D)
        E.qkv_bwd(DQ, DK, DV, dqkvg, acc, D)
        del DQ, DK, DV
        dxa = dqkvg @ wpack
        dwp = torch.mm(dqkvg.t(), xa, out_dtype=f32)
        dx = torch.empty_like(x)
        E.ln_bwd(dxa, x, st, lnw, dy2, dx, acc)
        dz, _, _ = rows.pair_bias_bwd(pair.view(L * L, DP), DB, wf, L, eps_p, acc, D)
        # custom-op outputs may not alias: every parameter gradient leaves as its own tensor, written by one kernel
        outs = [torch.empty(t.shape, dtype=f32, device=t.device) for t in leaves]
        E.finalize(dwp, dwo, acc, wb, lnpw, outs)
    return [dx.view(single.shape), dz.view(pair.shape), *outs]



def _beta_shadow_fake(single, pair, mask, leaves, eps_s, eps_p, dy):
    return torch.empty((DP,), device=single.device, dtype=torch.float32)


@opaque(fake=_beta_shadow_fake, name="apb_h100_beta_fp32_shadow")
def _beta_shadow(single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None,
                 leaves: list[torch.Tensor], eps_s: float, eps_p: float, dy: torch.Tensor) -> torch.Tensor:
    """Accuracy-first beta residual with native CUDA arithmetic and cuBLAS.

    This intentionally recreates the FP32 numerical path: beta's mathematical
    gradient is zero, but the required oracle contains reduction roundoff.
    The full FP32 pair normalization and attention probabilities are temporary
    workspaces. This is slower and does not retain the B200 tensor lifetimes.
    """
    E = _rows().ext()
    with torch.cuda.device(single.device), torch.autocast("cuda", enabled=False):
        values = E.exact_fp32_inputs([single, pair, dy, *leaves])
        x, z, dy32 = values[:3]
        lnw, lnb, wq, bq, wk, wv, wg, wo, lnpw, lnpb, wb = values[3:]
        L, D = single.shape[1:]
        H, dh = wb.shape[0], D // wb.shape[0]
        xa = E.exact_ln(x, lnw, lnb, eps_s).view(L, D)
        za = E.exact_ln(z, lnpw, lnpb, eps_p).view(L * L, DP)
        q = E.exact_mm(xa, wq.t(), bq)
        k = E.exact_mm(xa, wk.t())
        v = E.exact_mm(xa, wv.t())
        gate = E.exact_mm(xa, wg.t())
        bias = E.exact_mm(za, wb.t())
        dog = E.exact_mm(dy32.view(L, D), wo)
        do_rows = E.exact_point(dog, gate, 1.0, 1).view(L, H, dh)
        if mask is not None:
            # masked_fill makes the oracle's bias contiguous and selects its
            # efficient CUTLASS FP32 attention. Reproduce that arithmetic with
            # project CUDA launches of the installed C++ templates, including
            # the FP32 rounded product/reduction used for its delta tensor.
            qh, kh, vh = (t.view(L, H, dh) for t in (q, k, v))
            bm = E.exact_mask_bias(bias, mask, L)
            out32, lse32 = E.exact_attention_fwd(qh, kh, vh, bm)
            delta32 = E.exact_delta(do_rows, out32)
            ds = E.exact_attention_db(qh, kh, vh, bm, out32, lse32, do_rows, delta32)
            db = E.exact_db_transpose(ds, mask)
            return E.exact_beta(E.exact_mm(db, wb))
        scale = math.sqrt(1.0 / math.sqrt(dh))
        qs = E.exact_point(q, q, scale, 0).view(1, L, H, dh).transpose(1, 2)
        ks = E.exact_point(k, k, scale, 0).view(1, L, H, dh).transpose(1, 2)
        scores = E.exact_mm(qs, ks.transpose(-2, -1))
        E.exact_bias_add(scores, bias, mask)
        probs = E.exact_soft(scores, scores, False)
        do = do_rows.view(1, L, H, dh).transpose(1, 2)
        vh = v.view(1, L, H, dh).transpose(1, 2)
        dp = E.exact_mm(do, vh.transpose(-2, -1))
        ds = E.exact_soft(dp, probs, True)
        db = E.exact_db_transpose(ds, mask)
        dln = E.exact_mm(db, wb)
        return E.exact_beta(dln)


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eps_s, eps_p, mask, single, pair, *leaves):
        ctx.param_dtypes = [t.dtype for t in leaves]
        params = leaves
        ctx.beta_shadow = (torch.cuda.get_device_capability(single.device) == (9, 0)
                           and os.environ.get("MINIWORLD_APB_H100_BETA_SHADOW", "1") == "1")
        leaves = _kernel_leaves(leaves)
        y, *saved = forward(single, pair, mask, leaves, eps_s, eps_p)
        ctx.eps = (eps_s, eps_p)
        ctx.saved_count = len(saved)
        ctx.save_for_backward(mask, single, pair, *leaves, *saved, *(params if ctx.beta_shadow else ()))
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        v = ctx.saved_tensors
        mask, single, pair, leaves, saved = v[0], v[1], v[2], list(v[3:13]), list(v[13:])          # saved: 10 or 11 tensors
        dx, dz, *grads = backward(single, pair, mask, leaves, *ctx.eps, saved, dy)
        if ctx.beta_shadow:
            params = list(v[14 + ctx.saved_count:])
            grads[9] = _beta_shadow(single, pair, mask, params, *ctx.eps, dy)
        if torch.cuda.get_device_capability(single.device) == (9, 0):
            grads = _rows().ext().cast_grads(grads, [dt == torch.float32 for dt in ctx.param_dtypes])
            from .token_dit_train import _activate_native_caller
            _activate_native_caller(force=True)
            return (None, None, None, dx, dz, *grads)
        return (None, None, None, dx, dz, *(g.to(dt) for g, dt in zip(grads, ctx.param_dtypes, strict=True)))


def update_train(module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    out = _Training.apply(module.ln_single.eps, module.ln_pair.eps, _mask(mask, single.shape[1]), single, pair, *_leaves(module))
    if torch.cuda.get_device_capability(single.device) == (9, 0):
        from .token_dit_train import _native_caller_hooks
        # The result graph now owns its AccumulateGrad nodes. Installing on
        # fresh leaves before apply can hook a temporary weak-held node.
        _native_caller_hooks(single, pair, *module.parameters(), force=True)
    return out
