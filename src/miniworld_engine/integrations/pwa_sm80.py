"""Automatic dispatch to the A100 (sm_80) MSAPairWeightedAveraging: hand CUDA around cuBLAS, inference and training.

Contract: bf16 MSA ``[1, S, L, d_msa]`` (d_msa 64 or 128, any S) and bf16 pair ``[1, L, L, d_pair]`` (d_pair 128 / 256 / 384; L a multiple of 16, at most 1024), 8 heads of 8 / 16 / 32
channels (``d_hidden``; d_msa 128 with 32-wide heads is not built), an optional bool key mask ``[1, L]``, the module's residual and row dropout (``drop_msa``, training), ``implementation=
"miniworld"`` with the engine backend not forced to Triton, capability 8.0. Everything else keeps the module's previous path (Triton, or the statements for ``d_hidden = 8``).

The step (``kernels/pair_weighted_averaging/cuda/sm80``; ``.../cuda/sm80.py`` documents the kernels):

  forward   pair_fwd (LN(pair) . Wb, key mask, softmax -> w [8, L, L])  ->  ln_v (LN(msa) . Wv -> v [8, L, S C], head-major)  ->  o = w v (cuBLAS bmm)
            ->  gate_out (sigmoid gate, output projection, dropout, residual -> out)
  backward  see ``backward`` below.

Inference is one opaque op; training is one autograd Function whose forward and backward are each one opaque op. ``MINIWORLD_PWA_SM80=0`` turns the path off.
Numerics and timings: ``docs/gpus/a100/msa_pair_weighted_averaging/``.
"""

from __future__ import annotations

import os
import warnings

import torch

from miniworld_engine import settings
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.modules.exceptions import ImplementationType

ENV = "MINIWORLD_PWA_SM80"
_FAILED = False


@torch.compiler.assume_constant_result
def _loads() -> bool:
    """Builds (first call) or loads the sm_80 extension; False, with one warning, when the toolchain fails (the module path then serves).

    A process-level constant, so ``torch.compile`` evaluates it once at trace time instead of tracing the nvcc lookup / JIT build into the graph."""
    global _FAILED
    if _FAILED:
        return False
    try:
        _sm80().ext()
    except Exception as exc:  # a toolchain problem keeps the module path
        _FAILED = True
        warnings.warn(f"sm_80 MSAPairWeightedAveraging kernels unavailable, keeping the module path: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


def _sm80():
    from miniworld_engine.kernels.pair_weighted_averaging.cuda import sm80
    return sm80


def _eligible(module, msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None) -> bool:
    if os.environ.get(ENV, "1") == "0":
        return False
    if module.implementation != ImplementationType.MINIWORLD or settings.current().engine_backend == "triton":
        return False
    d_msa = msa.shape[-1] if msa.ndim == 4 else 0
    d_hidden = module.to_value.weight.shape[0] // module.n_head
    d_pair = pair.shape[-1] if pair.ndim == 4 else 0
    sm = _sm80()
    if not sm.supported(d_msa, d_pair, module.n_head, d_hidden) or module.to_value.weight.shape != (module.n_head * d_hidden, d_msa):
        return False
    if module.ln_msa.weight is None or module.ln_pair.weight is None or module.to_bias.weight.shape != (module.n_head, d_pair):
        return False
    if not msa.is_cuda or msa.dtype != torch.bfloat16 or pair.dtype != torch.bfloat16 or msa.shape[0] != 1 or msa.shape[1] < 1:
        return False
    length = msa.shape[2]
    if length % 16 or length > sm.MAX_LEN or tuple(pair.shape) != (1, length, length, d_pair):
        return False
    if mask is not None and (tuple(mask.shape) != (1, length) or mask.dtype != torch.bool):
        return False
    if torch.cuda.get_device_capability(msa.device) != (8, 0):
        return False
    return _loads()


def serves_inference(module, msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None = None) -> bool:
    """A grad-free call that is not a live dropout (``MSAPairWeightedAveraging.forward`` routes those, like the grad-enabled ones, through the training step)."""
    return not torch.is_grad_enabled() and not (module.training and module.drop_msa.p_drop > 0) and _eligible(module, msa, pair, mask)


def serves_train(module, msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None = None) -> bool:
    return (torch.is_grad_enabled() or (module.training and module.drop_msa.p_drop > 0)) and _eligible(module, msa, pair, mask)


def _leaves(module) -> list[torch.Tensor]:
    """The module's own parameters (LayerNorm weight / bias of the MSA, Wv, Wg, LayerNorm weight / bias of the pair, Wb, Wo; bf16 or fp32 masters). The opaque ops cast them to the
    kernels' dtypes (``_kernel_leaves``) outside autograd: no cast copy in the graph, and the weight gradients leave the backward unrounded in each parameter's dtype."""
    return [module.ln_msa.weight, module.ln_msa.bias, module.to_value.weight, module.to_gate.weight, module.ln_pair.weight, module.ln_pair.bias, module.to_bias.weight,
            module.to_out.weight]


def _kernel_leaves(leaves: list[torch.Tensor]) -> list[torch.Tensor]:
    """LayerNorm weights / biases (fp32), the projections (bf16), contiguous."""
    lnm_w, lnm_b, wv, wg, lnz_w, lnz_b, wb, wo = leaves
    bf = torch.bfloat16
    return [lnm_w.float().contiguous(), lnm_b.float().contiguous(), wv.to(bf).contiguous(), wg.to(bf).contiguous(), lnz_w.float().contiguous(), lnz_b.float().contiguous(),
            wb.to(bf).contiguous(), wo.to(bf).contiguous()]


def _key_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    return None if mask is None else mask[0].contiguous().view(torch.uint8)


# ------------------------------------------------------------------------------------------------------------ inference
def _inference_fake(msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, leaves: list[torch.Tensor], eps_m: float, eps_z: float) -> torch.Tensor:
    """msa [1, S, L, d_msa], pair [1, L, L, d_pair] -> msa + PWA(msa, pair), bf16 like msa."""
    return torch.empty_like(msa)


@opaque(fake=_inference_fake, name="pwa_sm80_inference")
def inference(msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, leaves: list[torch.Tensor], eps_m: float, eps_z: float) -> torch.Tensor:
    """msa + PWA(msa, pair): pair_fwd -> ln_v -> cuBLAS w v -> gate_out (no autograd, nothing saved, no dropout)."""
    sm = _sm80()
    lnm_w, lnm_b, wv, wg, lnz_w, lnz_b, wb, wo = _kernel_leaves(leaves)
    with torch.cuda.device(msa.device):
        m = msa[0]
        w = sm.pair_fwd(pair[0], _key_mask(mask), lnz_w.contiguous(), lnz_b.contiguous(), wb.contiguous(), eps_z)
        ns = sm.pick_split(m.shape[0])
        v, _ = sm.ln_v(m, lnm_w.contiguous(), lnm_b.contiguous(), wv.contiguous(), eps_m, False, ns)
        o = torch.bmm(_expand(w, ns), v)
        del v, w
        out = sm.gate_out(m, o, lnm_w.contiguous(), lnm_b.contiguous(), wg.contiguous(), wo.contiguous(), None, eps_m, 1.0)
    return out.unsqueeze(0)


def update_inference(module, msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    return inference(msa.contiguous(), pair.contiguous(), mask, _leaves(module), float(module.ln_msa.eps), float(module.ln_pair.eps))


# ------------------------------------------------------------------------------------------------------------- training
# Forward: pair_fwd -> ln_v -> o = w v (cuBLAS) -> gate_out (the module's row-broadcast dropout fused: a [L, d_msa] keep-mask shared by the MSA rows), keeping w, v, o.
# Backward: ``glue`` (the output gradient -> do, dgp, dWo, dWg; o is read, the g o / drb / y tiles never reach memory) -> dv = w^T do and dw = do v^T (cuBLAS bmm; dw in
# fp32) -> ``dgv_bwd`` (dy = dgp Wg + dv Wv, the LayerNorm backward, the residual gradient, dWv) -> ``pair_bwd`` (softmax backward, bias projection, the pair LayerNorm backward).
# The training tensors v, o, do, dgp, dv are head-major with the MSA rows in ns chunks ([8 ns, L, S C / ns], ``sm80.pick_split``): dw contracts over S C, a K that cuBLAS runs
# at 125 TFLOP/s as 8 long batches and at 165 as 8 ns short ones (summed in ``pair_bwd``); w is repeated per chunk for the products.  Inference keeps the plain layout.
def _expand(w: torch.Tensor, ns: int) -> torch.Tensor:
    """w [8, L, L] -> [8 ns, L, L] with every head repeated for its ns chunks of MSA rows (a small contiguous copy: the batch operand of the chunked bmm)."""
    return w if ns == 1 else w.unsqueeze(1).expand(8, ns, *w.shape[1:]).reshape(8 * ns, *w.shape[1:])


def _forward_fake(msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, leaves: list[torch.Tensor], eps_m: float, eps_z: float, p_drop: float) -> list[torch.Tensor]:
    """The output and what the backward keeps: w [8, L, L], v and o [8 ns, L, S C / ns] (head-major, the MSA rows in ns chunks), the dropout keep-mask [L, d_msa] (empty without dropout)."""
    depth, length = msa.shape[1], msa.shape[2]
    c = leaves[2].shape[0] // 8
    ns = _sm80().pick_split(depth)
    e = msa.new_empty
    return [torch.empty_like(msa), e((8, length, length)), e((8 * ns, length, depth // ns * c)), e((8 * ns, length, depth // ns * c)), e((length, msa.shape[3]) if p_drop > 0 else (0,))]


@opaque(fake=_forward_fake, name="pwa_sm80_train_fwd")
def forward(msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, leaves: list[torch.Tensor], eps_m: float, eps_z: float, p_drop: float) -> list[torch.Tensor]:
    sm = _sm80()
    lnm_w, lnm_b, wv, wg, lnz_w, lnz_b, wb, wo = _kernel_leaves(leaves)
    with torch.cuda.device(msa.device):
        m = msa[0]
        w = sm.pair_fwd(pair[0], _key_mask(mask), lnz_w.contiguous(), lnz_b.contiguous(), wb.contiguous(), eps_z)
        ns = sm.pick_split(m.shape[0])
        v, _ = sm.ln_v(m, lnm_w.contiguous(), lnm_b.contiguous(), wv.contiguous(), eps_m, False, ns)
        o = torch.bmm(_expand(w, ns), v)
        keep, dscale = None, 1.0
        if p_drop > 0:                                       # Dropout(broadcast_dim=1): one keep decision per (token, channel), shared over the MSA rows
            keep = (torch.rand(m.shape[1], m.shape[2], device=m.device) > p_drop).to(torch.bfloat16)      # fp32 draw: a bf16 `rand` has 8 mantissa bits and would bias the keep probability
            dscale = 1.0 / (1.0 - p_drop)
        out = sm.gate_out(m, o, lnm_w.contiguous(), lnm_b.contiguous(), wg.contiguous(), wo.contiguous(), keep, eps_m, dscale)
    return [out.unsqueeze(0), w, v, o, msa.new_empty((0,)) if keep is None else keep]


def _backward_fake(msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, leaves: list[torch.Tensor], eps_m: float, eps_z: float, p_drop: float,
                   saved: list[torch.Tensor], dy: torch.Tensor) -> list[torch.Tensor]:
    """dmsa, dpair and the gradient of every leaf, each in its own dtype."""
    return [torch.empty_like(msa), torch.empty_like(pair)] + [torch.empty_like(t) for t in leaves]


@opaque(fake=_backward_fake, name="pwa_sm80_train_bwd")
def backward(msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, leaves: list[torch.Tensor], eps_m: float, eps_z: float, p_drop: float,
             saved: list[torch.Tensor], dy: torch.Tensor) -> list[torch.Tensor]:
    sm = _sm80()
    lnm_w, lnm_b, wv, wg, lnz_w, lnz_b, wb, wo = _kernel_leaves(leaves)
    pdt = [t.dtype for t in leaves]
    w, v, o, keep = saved
    keep_t = None if keep.numel() == 0 else keep
    dscale = 1.0 / (1.0 - p_drop) if p_drop > 0 else 1.0
    with torch.cuda.device(msa.device):
        m, z = msa[0], pair[0]
        dres = dy[0].contiguous()
        do, dgp, dwo, dwg = sm.glue(m, dres, o, lnm_w.contiguous(), lnm_b.contiguous(), wg.contiguous(), wo.contiguous(), keep_t, eps_m, dscale)
        wx = _expand(w, v.shape[0] // 8)
        dv = torch.bmm(wx.transpose(1, 2), do)                                          # [8 ns, L(j), S C / ns]
        dw = torch.bmm(do, v.transpose(1, 2), out_dtype=torch.float32)                  # [8 ns, L(i), L(j)]: ns partial products per head (pair_bwd sums them)
        del do, wx
        dm, dwv, dgm, dbm = sm.dgv_bwd(m, dres, dgp, dv, lnm_w.contiguous(), lnm_b.contiguous(), wg.contiguous(), wv.contiguous(), eps_m)
        del dgp, dv
        dz, dwb, dgz, dbz = sm.pair_bwd(z, w, dw, _key_mask(mask), lnz_w.contiguous(), lnz_b.contiguous(), wb.contiguous(), eps_z)
    # custom-op outputs may not alias each other: the small gradients are views of reduction buffers, so each leaves as its own copy in its leaf's dtype
    return [dm.unsqueeze(0), dz.unsqueeze(0), dgm.to(pdt[0], copy=True), dbm.to(pdt[1], copy=True), dwv.to(pdt[2], copy=True), dwg.to(pdt[3], copy=True),
            dgz.to(pdt[4], copy=True), dbz.to(pdt[5], copy=True), dwb.to(pdt[6], copy=True), dwo.to(pdt[7], copy=True)]


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eps_m, eps_z, p_drop, msa, pair, mask, *leaves):
        out, *saved = forward(msa, pair, mask, list(leaves), eps_m, eps_z, p_drop)
        ctx.eps, ctx.p_drop = (eps_m, eps_z), p_drop
        ctx.save_for_backward(msa, pair, mask, *leaves, *saved)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        v = ctx.saved_tensors
        msa, pair, mask, leaves, saved = v[0], v[1], v[2], list(v[3:11]), list(v[11:])
        grads = backward(msa, pair, mask, leaves, *ctx.eps, ctx.p_drop, saved, dy.contiguous())
        return (None, None, None, grads[0], grads[1], None, *grads[2:])


def update_train(module, msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    p_drop = float(module.drop_msa.p_drop) if module.training else 0.0
    return _Training.apply(float(module.ln_msa.eps), float(module.ln_pair.eps), p_drop, msa.contiguous(), pair.contiguous(), mask, *_leaves(module))
