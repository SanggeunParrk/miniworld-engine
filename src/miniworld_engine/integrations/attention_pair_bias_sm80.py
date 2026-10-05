"""Automatic dispatch to the A100 (sm_80) AttentionPairBias: hand CUDA but the plain GEMMs (cuBLAS).

Contract: bf16 activations, B = 1, d_single 384 as 8 heads x 48, 12 x 32 or 16 x 24 (AF3's Pairformer single attention; run 32 wide per head, as on
B200), or d_single 512 as 16 x 32; d_pair 128, no QK-norm, any L, ``implementation="miniworld"`` (engine backend not forced to Triton), sm_80.

Inference (one opaque op), the B200 step with this family's A100 pieces:
  ln_rows (LN(single) -> bf16, and a copy of the single as the residual seed) -> addmm q | k | v | g (q pre-scaled by log2(e) / sqrt(head dim): logits in
  exp2 units) -> pair_bias (LN(pair) . Wf, Wf = to_bias.weight * ln_pair.weight * log2(e); one read of the pair; masked keys, keys past L and padded query
  rows at -1e30) -> the gated core (``augmented_attention/cuda/sm80``: sigmoid(g) o written over the q columns) -> to_out accumulated in place (cuBLAS
  beta = 1) onto the residual seed.
The attention core wants L in tiles of 128: the single is padded with zero rows (their k / v are finite and every padded key is masked) and the bias is
written [H, Lp, Lp]; the pair is read in place. ln_pair's bias adds Wb . b to every (i, j) of a head: the softmax cancels it, so the forward drops it.

``serves_*`` is the whole gate; ``MINIWORLD_APB_SM80=0`` turns the path off. Numerics and timings: ``docs/gpus/a100/attention_pair_bias/``.
"""

from __future__ import annotations

import math
import os
import warnings

import torch

from miniworld_engine import settings
from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.modules.exceptions import ImplementationType

DP = 128
SHAPES = ((8, 384), (12, 384), (16, 384), (16, 512))     # (heads, d_single): 8 x 48, 12 x 32, 16 x 24 (padded to 32), 16 x 32
LOG2E = math.log2(math.e)
NEG = -1e30                     # masked keys: finite, so a fully masked sample stays NaN-free
_packs: dict = {}
_cores: dict = {}
_FAILED = False


@torch.compiler.assume_constant_result
def _loads() -> bool:
    """Builds (first call) or loads the two sm_80 extensions; False, with one warning, when the toolchain fails (the module path then serves).

    A process-level constant, so ``torch.compile`` evaluates it once at trace time instead of tracing the nvcc lookup / JIT build into the graph."""
    global _FAILED
    if _FAILED:
        return False
    try:
        _rows().ext()
        _sm80()._ext()
    except Exception as exc:  # a toolchain problem keeps the module path
        _FAILED = True
        warnings.warn(f"sm_80 AttentionPairBias kernels unavailable, keeping the module path: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


def _pad(length: int) -> int:
    return -(-length // 128) * 128


def _rows():
    from miniworld_engine.kernels.augmented_attention.cuda import apb as rows
    return rows


def _sm80():
    from miniworld_engine.kernels.augmented_attention.cuda import sm80
    return sm80


def _eligible(module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None) -> bool:
    if os.environ.get("MINIWORLD_APB_SM80", "1") == "0":
        return False
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
    if L < 1 or tuple(pair.shape) != (1, L, L, DP) or not pair.is_contiguous():
        return False
    if mask is not None and tuple(mask.shape) != (1, L):
        return False
    if torch.cuda.get_device_capability(single.device) != (8, 0):
        return False
    heads_dim = _rows().GEOMETRY[(H, D)][1]
    idx = single.device.index if single.device.index is not None else torch.cuda.current_device()
    return _sm80().supported_gate(torch.bfloat16, _pad(L), H * heads_dim, H, idx) and _loads()


def serves_inference(module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None = None) -> bool:
    return not torch.is_grad_enabled() and _eligible(module, single, pair, mask)


def _leaves(module) -> list[torch.Tensor]:
    """The module's parameters as they are (any float dtype, e.g. an fp32 master): ln_single w / b, Wq, bq, Wk, Wv, Wg, Wo, ln_pair w / b, Wb. The ops cast them
    (``_kernel_leaves``) outside autograd, so fp32 parameters get unrounded fp32 gradients."""
    return [module.ln_single.weight, module.ln_single.bias, module.to_query.weight, module.to_query.bias, module.to_key.weight, module.to_value.weight,
            module.to_gate.weight, module.to_out.weight, module.ln_pair.weight, module.ln_pair.bias, module.to_bias.weight]


def _kernel_leaves(leaves) -> list[torch.Tensor]:
    """The kernels' dtypes: the LayerNorm vectors fp32, the projections bf16 (no copies when the parameters already are)."""
    bf = torch.bfloat16
    lnw, lnb, wq, bq, wk, wv, wg, wo, lnpw, lnpb, wb = leaves
    return [lnw.float(), lnb.float(), wq.to(bf), bq.to(bf), wk.to(bf), wv.to(bf), wg.to(bf), wo.to(bf), lnpw.float(), lnpb.float(), wb.to(bf)]


def _mask(mask: torch.Tensor | None, L: int) -> torch.Tensor | None:
    return None if mask is None else mask.reshape(L).to(torch.bool).contiguous()


def _geometry(leaves):
    """(heads, real head dim, row width W, d_single) from to_bias.weight [heads, 128] and ln_single.weight [d_single]."""
    H, D = leaves[10].shape[0], leaves[0].numel()
    return H, D // H, _rows().width(H, D), D


# ------------------------------------------------------------------------------------------------------------ inference
def _inference_packs(params, leaves):
    """``_prep`` with q scaled into exp2 units and Wf in exp2 units, cached per parameter version (the step reads them every call; repacking costs a launch).
    Keyed on the parameters, not on ``leaves`` (their kernel-dtype casts: fresh tensors per call for fp32), and scoped to the CUDA-graph capture."""
    build = lambda: (*_prep(leaves, LOG2E / math.sqrt(_geometry(leaves)[1]), LOG2E), params)
    return _capture.lookup(_packs, tuple((t.data_ptr(), t._version) for t in params), build, limit=8)[:4]


def _prep(leaves, qs: float, ws: float):
    """(W q | k | v | g [4 W, D] (padded rows, q x qs), its bias [4 W], Wf = to_bias.weight * ln_pair.weight * ws [H, 128], Wo [D, W] with padded columns
    -- or None when there is no padding (8 x 48, 12 x 32, 16 x 32: to_out.weight as is)), bf16, one launch."""
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


@opaque(fake=_inference_fake, name="apb_sm80_inference")
def inference(single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, leaves: list[torch.Tensor], eps_s: float,
              eps_p: float) -> torch.Tensor:
    """single [1, L, d], pair [1, L, L, 128] bf16, mask [L] bool or None, leaves as ``_leaves`` -> single + APB(single)."""
    L = single.shape[1]
    Lp = _pad(L)
    rows, dev = _rows(), single.device
    params, leaves = leaves, _kernel_leaves(leaves)
    lnw, lnb = leaves[0], leaves[1]
    H, _, W, D = _geometry(leaves)
    with torch.cuda.device(dev):
        x = single.view(L, D)
        wpack, bvec, wf, wop = _inference_packs(params, leaves)
        wo = leaves[7] if wop is None else wop
        # padded rows are zero: their k / v rows must be finite (a masked key still multiplies its v row by probability 0)
        xa = torch.empty_like(x) if Lp == L else x.new_zeros((Lp, D))
        y = torch.empty_like(x)                   # residual seed, written by ln_rows; to_out accumulates onto it
        rows.ext().ln_rows(x, lnw, lnb, xa, None, eps_s, y)
        qkvg = torch.addmm(bvec, xa, wpack.t())
        bias = rows.pair_bias(pair.view(L * L, DP), wf, mask, L, eps_p, NEG, padded=Lp)
        key = (dev.index, H, W // H)
        core = _cores.get(key)
        if core is None:
            core = _cores[key] = _sm80().GatedInferenceCore(dev.index, torch.bfloat16, H, W // H)
        core(qkvg, bias, 0, 1)
        return y.addmm_(qkvg[:L, :W], wo.t()).view(1, L, D)


def update_inference(module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    return inference(single, pair, _mask(mask, single.shape[1]), _leaves(module), module.ln_single.eps, module.ln_pair.eps)


# ------------------------------------------------------------------------------------------------------------- training
# The same forward in natural units for the plain core (bf16 o, the log-sum-exp; the pair bias in the core's raw units, natural x sqrt(head dim), masked and
# padded keys at -1e4 natural, as ``augattn_sm80``), gate_rows and addmm; backward: dO GEMMs, gate_bwd (dO, dg), the core's backward (dq, dk, dv, and the bias
# gradient in fp32), qkv_bwd, the dxa / dW GEMMs, ln_bwd (+ residual), pair_bias_bwd. The padded 16 x 24 heads (32 wide, zero columns) keep the softmax scale of 24.
MASKED = -1e4


def serves_train(module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None = None) -> bool:
    return torch.is_grad_enabled() and _eligible(module, single, pair, mask)


def _forward_fake(single, pair, mask, leaves, eps_s, eps_p):
    """y and the saved set: xa, (mean, rstd), qkvg, bias, O, LSE, og, W q | k | v | g (and the padded Wo for 16 x 24)."""
    L = single.shape[1]
    Lp = _pad(L)
    H, _, W, D = _geometry(leaves)
    e, f32 = single.new_empty, torch.float32
    out = [torch.empty_like(single), e((Lp, D)), e((L, 2), dtype=f32), e((Lp, 4 * W)), e((H, Lp, Lp)), e((Lp, W)), e((1, H, Lp), dtype=f32), e((L, W)),
           e((4 * W, D))]
    return out + ([e((D, W))] if W != D else [])


@opaque(fake=_forward_fake, name="apb_sm80_train_fwd")
def forward(single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, leaves: list[torch.Tensor], eps_s: float,
            eps_p: float) -> list[torch.Tensor]:
    L = single.shape[1]
    Lp = _pad(L)
    rows = _rows()
    lnw, lnb = leaves[0], leaves[1]
    H, hd, W, D = _geometry(leaves)
    scale = math.sqrt(hd)
    with torch.cuda.device(single.device):
        x = single.view(L, D)
        wpack, bvec, wf, wop = _prep(leaves, 1.0, scale)
        wo = leaves[7] if wop is None else wop
        xa = torch.empty_like(x) if Lp == L else x.new_zeros((Lp, D))
        st = x.new_empty((L, 2), dtype=torch.float32)
        y = torch.empty_like(x)
        rows.ext().ln_rows(x, lnw, lnb, xa, st, eps_s, y)
        qkvg = torch.addmm(bvec, xa, wpack.t())
        bias = rows.pair_bias(pair.view(L * L, DP), wf, mask, L, eps_p, MASKED * scale, padded=Lp)
        o, lse = _sm80().plain_forward(qkvg[:, :W], qkvg[:, W:2 * W], qkvg[:, 2 * W:3 * W], bias, 1, Lp, H, W // H, save_lse=True, sm_scale=1.0 / scale)
        og = x.new_empty((L, W))
        rows.ext().gate_rows_bf(o, qkvg, og, L)
        y = y.addmm_(og, wo.t()).view(1, L, D)           # in place onto the residual seed: no copy of x into the output
    return [y, xa, st, qkvg, bias, o, lse, og, wpack] + ([wop] if wop is not None else [])


def _backward_fake(single, pair, mask, leaves, eps_s, eps_p, saved, dy):
    """One gradient per input (its dtype) and per leaf (fp32), contiguous."""
    return [torch.empty_like(single), torch.empty_like(pair), *(torch.empty(t.shape, dtype=torch.float32, device=t.device) for t in leaves)]


@opaque(fake=_backward_fake, name="apb_sm80_train_bwd")
def backward(single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, leaves: list[torch.Tensor], eps_s: float, eps_p: float,
             saved: list[torch.Tensor], dy: torch.Tensor) -> list[torch.Tensor]:
    L = single.shape[1]
    Lp = _pad(L)
    rows = _rows()
    E = rows.ext()
    lnw, lnpw, wb = leaves[0], leaves[8], leaves[10]
    H, hd, W, D = _geometry(leaves)
    xa, st, qkvg, bias, o, lse, og, wpack = saved[:8]
    wo = saved[8] if W != D else leaves[7]                 # [D, W]
    f32 = torch.float32
    with torch.cuda.device(single.device):
        x = single.view(L, D)
        dy2 = dy.contiguous().view(L, D)
        acc = torch.zeros(rows.acc_n(H, D), device=x.device)    # dlnw | dlnb | dbq | dWf | head sums, added by the kernels
        dog = dy2 @ wo                                       # [L, W]: zero in the pads (Wo's pad columns are zero)
        dwo = torch.mm(dy2.t(), og, out_dtype=f32)           # [D, W]; finalize keeps the real columns
        dob = x.new_empty((Lp, W)) if Lp == L else x.new_zeros((Lp, W))     # the padded queries' gradient is 0
        dqkvg = x.new_empty((L, 4 * W))
        E.gate_bwd_bf(dog, o, qkvg, dob, dqkvg, L)
        dq, dk, dv, db = _sm80().plain_backward(qkvg[:, :W], qkvg[:, W:2 * W], qkvg[:, 2 * W:3 * W], dob, bias, o, lse, 1, Lp, H, W // H, db_natural=True,
                                                sm_scale=1.0 / math.sqrt(hd))
        E.qkv_bwd_bf(dq, dk, dv, dqkvg, acc, D, L)
        del dq, dk, dv
        dxa = dqkvg @ wpack
        dwp = torch.mm(dqkvg.t(), xa[:L], out_dtype=f32)
        dx = torch.empty_like(x)
        E.ln_bwd(dxa, x, st, lnw, dy2, dx, acc)
        wf_nat = (wb.float() * lnpw.float()[None]).to(torch.bfloat16)      # natural units: db is the gradient of the natural-unit bias
        dz, _, _ = rows.pair_bias_bwd(pair.view(L * L, DP), db, wf_nat, L, eps_p, acc, D)
        # custom-op outputs may not alias: every parameter gradient leaves as its own tensor, written by one kernel. ln_pair's bias moves a head's logits by one
        # constant: sum_j dbias[h, i, j] = 0 for every query, so its gradient and its share of dWb are exactly 0
        outs = [torch.empty(t.shape, dtype=f32, device=t.device) for t in leaves]
        E.finalize(dwp, dwo, acc, wb, lnpw, outs)
    return [dx.view(single.shape), dz.view(pair.shape), *outs]


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eps_s, eps_p, mask, single, pair, *leaves):
        ctx.param_dtypes = [t.dtype for t in leaves]
        leaves = _kernel_leaves(leaves)
        y, *saved = forward(single, pair, mask, leaves, eps_s, eps_p)
        ctx.eps = (eps_s, eps_p)
        ctx.save_for_backward(mask, single, pair, *leaves, *saved)
        return y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        v = ctx.saved_tensors
        mask, single, pair, leaves, saved = v[0], v[1], v[2], list(v[3:14]), list(v[14:])          # saved: 9 or 10 tensors
        dx, dz, *grads = backward(single, pair, mask, leaves, *ctx.eps, saved, dy)
        return (None, None, None, dx, dz, *(g.to(dt) for g, dt in zip(grads, ctx.param_dtypes, strict=True)))


def update_train(module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    return _Training.apply(module.ln_single.eps, module.ln_pair.eps, _mask(mask, single.shape[1]), single, pair, *_leaves(module))
