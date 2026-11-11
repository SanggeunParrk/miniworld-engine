"""Automatic dispatch to the A100 (sm_80) MSAPairWeightedAveraging: hand CUDA around cuBLAS, inference and training.

Contract: bf16 MSA ``[1, S, L, d_msa]`` (d_msa 64 or 128, any S) and bf16 pair ``[1, L, L, d_pair]`` (d_pair 128 / 256 / 384; L a multiple of 16, at most 1024), 8 heads of 8 / 16 / 32
channels (``d_hidden``; d_msa 128 with 32-wide heads is not built; d_msa 256 / d_pair 512 / 8 x 32 is ``pwa_wide_sm80``), an optional bool key mask ``[1, L]``, the module's residual and row dropout (``drop_msa``, training), capability 8.0.
``refusal`` is the range; ``modules.MSAPairWeightedAveraging``'s ``cuda`` backend refuses everything else (no fallback).

The step (``kernels/pair_weighted_averaging/cuda/sm80``; ``.../cuda/sm80.py`` documents the kernels):

  forward   pair_fwd (LN(pair) . Wb, key mask, softmax -> w [8, L, L])  ->  ln_v (LN(msa) . Wv -> v [8, L, S C], head-major)  ->  o = w v (cuBLAS bmm)
            ->  gate_out (sigmoid gate, output projection, dropout, residual -> out)
  backward  see ``backward`` below.

Inference is one opaque op; training is one autograd Function whose forward and backward are each one opaque op.
Numerics and timings: ``docs/gpus/a100/msa_pair_weighted_averaging/``.
"""

from __future__ import annotations

from typing import cast

import torch

from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._nvcc import build_failed
from miniworld_engine.kernels.integrations import pwa_wide_sm80

#: Opt-in experiments of the port, off: the searched cuBLASLt algorithms (``sm80.lt_bcast`` / ``lt_dw``) and the early-dw kernel.
_LT = False
_DW_EARLY = False


@torch.compiler.assume_constant_result
def _loads() -> bool:
    """Builds (first call) or loads the sm_80 extension; a failed build / load raises ``_nvcc.KernelBuildError`` (never a fallback).

    A process-level constant, so ``torch.compile`` evaluates it once at trace time instead of tracing the nvcc lookup / JIT build into the graph."""
    try:
        _sm80().ext()
    except Exception as exc:  # a build / load failure is an error, never a fallback
        build_failed("sm_80 MSAPairWeightedAveraging", exc)
        return False  # reached only while planning (build_failed returns there)
    return True


def _sm80():
    from miniworld_engine.kernels.pair_weighted_averaging.cuda import sm80

    return sm80


def refusal(module, msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None) -> str | None:
    """Why the sm_80 kernels cannot run this call, or ``None`` (the contract above). Shapes, dtypes and the device only, plus the build."""
    if not msa.is_cuda or torch.cuda.get_device_capability(msa.device) != (8, 0):
        return "the sm_80 MSAPairWeightedAveraging kernels are the A100 (capability 8.0) builds"
    if msa.dtype != torch.bfloat16 or pair.dtype != torch.bfloat16:
        return "the sm_80 kernels take bfloat16 msa / pair"
    if msa.ndim != 4 or pair.ndim != 4:
        return "msa must be [1, S, L, d_msa] and pair [1, L, L, d_pair]"
    d_msa, d_pair = msa.shape[-1], pair.shape[-1]
    d_hidden = module.to_value.weight.shape[0] // module.n_head
    sm = _sm80()
    if pwa_wide_sm80.supported(d_msa, d_pair, module.n_head, d_hidden):  # the BIG width: its own composition (pwa_wide_sm80)
        return _wide_refusal(module, msa, pair, mask)
    if not sm.supported(d_msa, d_pair, module.n_head, d_hidden) or module.to_value.weight.shape != (module.n_head * d_hidden, d_msa):
        return (
            f"d_msa {d_msa}, d_pair {d_pair}, {module.n_head} heads x {d_hidden}: not an sm_80 build (8 heads of 8 / 16 / 32, d_msa {sm.D_MSA}, "
            f"d_pair {sm.D_PAIR}, d_msa x d_hidden <= 2048, not d_msa 128 with 32-wide heads)"
        )
    if module.ln_msa.weight is None or module.ln_pair.weight is None or module.to_bias.weight.shape != (module.n_head, d_pair):
        return "the kernels take the affine msa / pair LayerNorms and a bias-free [n_head, d_pair] logit projection"
    if msa.shape[0] != 1 or msa.shape[1] < 1:
        return f"batch {msa.shape[0]}: the sm_80 kernels take one sample (batch 1) with at least one MSA row"
    length = msa.shape[2]
    training = torch.is_grad_enabled() or (module.training and module.drop_msa.p_drop > 0)  # the training step (serves_train)
    longest = sm.max_len(d_pair, training)
    if length % 16 or length > longest or tuple(pair.shape) != (1, length, length, d_pair):
        return (
            f"L {length}: the sm_80 kernels take L a multiple of 16 up to {longest} at d_pair {d_pair} in {'training' if training else 'inference'} "
            f"(training {sm.MAX_LEN}; inference: the pair forward's shared-memory row of logits) and a [1, L, L, d_pair] pair"
        )
    if mask is not None and (tuple(mask.shape) != (1, length) or mask.dtype != torch.bool):
        return "mask must be a bool [1, L] key mask"
    if not _loads():
        return "the sm_80 MSAPairWeightedAveraging extension failed to build"
    return None


def _wide_refusal(module, msa, pair, mask) -> str | None:
    """The range of ``pwa_wide_sm80`` (d_msa 256, d_pair 512, 8 heads x 32) past the width check."""
    if module.ln_msa.weight is None or module.ln_pair.weight is None or module.to_bias.weight.shape != (module.n_head, pair.shape[-1]):
        return "the kernels take the affine msa / pair LayerNorms and a bias-free [n_head, d_pair] logit projection"
    if msa.shape[0] != 1 or msa.shape[1] < 1:
        return f"batch {msa.shape[0]}: the sm_80 kernels take one sample (batch 1) with at least one MSA row"
    length = msa.shape[2]
    if tuple(pair.shape) != (1, length, length, pair.shape[-1]):
        return "pair must be [1, L, L, d_pair]"
    if mask is not None and (tuple(mask.shape) != (1, length) or mask.dtype != torch.bool):
        return "mask must be a bool [1, L] key mask"
    return None


def _wide(module) -> bool:
    return pwa_wide_sm80.supported(
        module.to_value.weight.shape[1], module.to_bias.weight.shape[1], module.n_head, module.to_value.weight.shape[0] // module.n_head
    )


def _eligible(module, msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None) -> bool:
    return refusal(module, msa, pair, mask) is None


def serves_inference(module, msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None = None) -> bool:
    """A grad-free call that is not a live dropout (those, like the grad-enabled ones, take the training step)."""
    return not torch.is_grad_enabled() and not (module.training and module.drop_msa.p_drop > 0) and _eligible(module, msa, pair, mask)


def serves_train(module, msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None = None) -> bool:
    return (torch.is_grad_enabled() or (module.training and module.drop_msa.p_drop > 0)) and _eligible(module, msa, pair, mask)


def _ln_pair_bias(module) -> torch.Tensor:
    """``ln_pair`` has no offset (it feeds only the logits, where a constant per head cancels in the softmax); the kernels still take the vector, so they get zeros."""
    bias = module.ln_pair.bias
    return torch.zeros_like(module.ln_pair.weight) if bias is None else bias


def _leaves(module) -> list[torch.Tensor]:
    """The module's own parameters (LayerNorm weight / bias of the MSA, Wv, Wg, LayerNorm weight / bias of the pair, Wb, Wo; bf16 or fp32 masters). The opaque ops cast them to the
    kernels' dtypes (``_kernel_leaves``) outside autograd: no cast copy in the graph, and the weight gradients leave the backward unrounded in each parameter's dtype."""
    return [
        module.ln_msa.weight,
        module.ln_msa.bias,
        module.to_value.weight,
        module.to_gate.weight,
        module.ln_pair.weight,
        _ln_pair_bias(module),
        module.to_bias.weight,
        module.to_out.weight,
    ]


def _kernel_leaves(leaves: list[torch.Tensor]) -> list[torch.Tensor]:
    """LayerNorm weights / biases (fp32), the projections (bf16), contiguous. Cached while the parameters are unmodified (``_capture.param_forms``, see opm_sm80)."""
    lnm_w, lnm_b, wv, wg, lnz_w, lnz_b, wb, wo = leaves
    bf = torch.bfloat16
    return list(
        _capture.param_forms(
            "pwa_sm80",
            leaves,
            lambda: (
                lnm_w.float().contiguous(),
                lnm_b.float().contiguous(),
                wv.to(bf).contiguous(),
                wg.to(bf).contiguous(),
                lnz_w.float().contiguous(),
                lnz_b.float().contiguous(),
                wb.to(bf).contiguous(),
                wo.to(bf).contiguous(),
            ),
        )
    )


def _key_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    return None if mask is None else mask[0].contiguous().view(torch.uint8)


def _fused_contract_enabled(m: torch.Tensor, wg: torch.Tensor, training: bool) -> bool:
    fused = None
    if fused is None:
        fused = not training
    return (
        fused
        and m.shape[1] == 128
        and m.shape[2] == 64
        and wg.shape[0] == 256
        and m.shape[0] in ((1024,) if training else (1024, 2048, 4096))
    )


# ------------------------------------------------------------------------------------------------------------ inference
def _inference_fake(
    msa: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps_m: float,
    eps_z: float,
) -> torch.Tensor:
    """msa [1, S, L, d_msa], pair [1, L, L, d_pair] -> msa + PWA(msa, pair), bf16 like msa."""
    return torch.empty_like(msa)


@opaque(fake=_inference_fake, name="pwa_sm80_inference")
def inference(
    msa: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps_m: float,
    eps_z: float,
) -> torch.Tensor:
    """msa + PWA(msa, pair): pair_fwd -> ln_v -> cuBLAS w v -> gate_out (no autograd, nothing saved, no dropout)."""
    sm = _sm80()
    lnm_w, lnm_b, wv, wg, lnz_w, lnz_b, wb, wo = _kernel_leaves(leaves)
    with torch.cuda.device(msa.device):
        m = msa[0]
        w = sm.pair_fwd(
            pair[0],
            _key_mask(mask),
            lnz_w.contiguous(),
            lnz_b.contiguous(),
            wb.contiguous(),
            eps_z,
        )
        ns = sm.pick_split(m.shape[0])
        v, _ = sm.ln_v(m, lnm_w.contiguous(), lnm_b.contiguous(), wv.contiguous(), eps_m, False, ns)
        if _fused_contract_enabled(m, wg, False):
            from miniworld_engine.kernels.pair_weighted_averaging.cuda import (
                contract_gate_fused_sm80,
            )

            out, _ = contract_gate_fused_sm80.result(m, w, v, lnm_w, lnm_b, wg, wo, eps=eps_m, mode=1)
        else:
            o = sm.lt_bcast(w, v) if ns > 1 and _LT else None
            if o is None:
                o = sm.broadcast_bmm(w, v) if ns > 1 else torch.bmm(_expand(w, ns), v)
            del v, w
            out = sm.gate_out(
                m,
                o,
                lnm_w.contiguous(),
                lnm_b.contiguous(),
                wg.contiguous(),
                wo.contiguous(),
                None,
                eps_m,
                1.0,
            )
    return out.unsqueeze(0)


def update_inference(module, msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if _wide(module):
        return pwa_wide_sm80.update_inference(
            _leaves(module), msa.contiguous(), pair.contiguous(), mask, float(module.ln_msa.eps), float(module.ln_pair.eps)
        )
    return inference(
        msa.contiguous(),
        pair.contiguous(),
        mask,
        _leaves(module),
        float(module.ln_msa.eps),
        float(module.ln_pair.eps),
    )


# ------------------------------------------------------------------------------------------------------------- training
# Forward: pair_fwd -> ln_v -> o = w v (cuBLAS) -> gate_out (the module's row-broadcast dropout fused: a [L, d_msa] keep-mask shared by the MSA rows), keeping w, v, o.
# Backward: ``glue`` (the output gradient -> do, dgp, dWo, dWg; o is read, the g o / drb / y tiles never reach memory) -> dv = w^T do and dw = do v^T (cuBLAS bmm; dw in
# fp32) -> ``dgv_bwd`` (dy = dgp Wg + dv Wv, the LayerNorm backward, the residual gradient, dWv) -> ``pair_bwd`` (softmax backward, bias projection, the pair LayerNorm backward).
# The training tensors v, o, do, dgp, dv are head-major with the MSA rows in ns chunks ([8 ns, L, S C / ns], ``sm80.pick_split``): dw contracts over S C, a K that cuBLAS runs
# at 125 TFLOP/s as 8 long batches and at 165 as 8 ns short ones (summed in ``pair_bwd``); w is repeated per chunk for the products.  Inference keeps the plain layout.
def _expand(w: torch.Tensor, ns: int) -> torch.Tensor:
    """w [8, L, L] -> [8 ns, L, L] with every head repeated for its ns chunks of MSA rows (a small contiguous copy: the batch operand of the chunked bmm)."""
    return w if ns == 1 else w.unsqueeze(1).expand(8, ns, *w.shape[1:]).reshape(8 * ns, *w.shape[1:])


def _broadcast_contract(w: torch.Tensor, v: torch.Tensor, transpose: bool = False) -> torch.Tensor:
    """Native contractions for qualified L128 training rows; preserve the faster path elsewhere."""
    if (
        True
        and w.shape[1] == 128
        and v.shape[2] % 128 == 0
        and w.is_contiguous()
        and v.is_contiguous()
        and w.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
    ):
        short_mode = "0"
        if short_mode in ("1", "2", "3") and w.shape[0] == 8 and v.shape[0] == 32 and v.shape[2] == 8192:
            if short_mode in ("2", "3"):
                from miniworld_engine.kernels.pair_weighted_averaging.cuda import (
                    broadcast_capture_sm80,
                )

                return broadcast_capture_sm80.result(w, v, transpose, 1 if short_mode == "2" else 2)
            from miniworld_engine.kernels.pair_weighted_averaging.cuda import (
                broadcast_pipeline_sm80,
            )

            return broadcast_pipeline_sm80.result(w, v, transpose, 1)
        from miniworld_engine.kernels.pair_weighted_averaging.cuda import (
            broadcast_native_sm80,
        )

        return broadcast_native_sm80.result(w, v, transpose, 0)
    if _LT and w.dim() == 3 and v.shape[0] % 8 == 0 and v.shape[0] > 8:
        out = _sm80().lt_bcast(w, v, transpose)  # cuBLASLt, pointer-array batches (w shared by the chunks of a head): no expanded copy of w
        if out is not None:
            return out
    expanded = _expand(w, v.shape[0] // 8)
    return torch.bmm(expanded.transpose(1, 2) if transpose else expanded, v)


def _weight_contract(do: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Native dw for qualified L384/640 rows; explicit1 tests other candidates."""
    if (
        _DW_EARLY
        and do.shape == v.shape
        and do.shape == (32, 128, 8192)
        and do.is_contiguous()
        and v.is_contiguous()
        and do.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
    ):
        from miniworld_engine.kernels.pair_weighted_averaging.cuda import dw_early_sm80

        return dw_early_sm80.result(do, v, 2, 9)
    override = None
    native = override if override is not None else do.shape[1] in (384, 640)
    if (
        native
        and do.shape[0] == 32
        and do.shape[1] in (256, 384, 512, 640, 768)
        and do.shape[2] == 8192
        and do.shape == v.shape
        and do.is_contiguous()
        and v.is_contiguous()
        and do.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
    ):
        from miniworld_engine.kernels.pair_weighted_averaging.cuda import dw_native_sm80

        return dw_native_sm80.result(do, v, 4 if do.shape[1] == 256 else 1, 0)
    if _LT and do.shape == v.shape and do.dim() == 3:
        out = _sm80().lt_dw(do, v)  # cuBLASLt with the searched algorithm (1.2-1.4x the heuristic default)
        if out is not None:
            return out
    return torch.bmm(do, v.transpose(1, 2), out_dtype=torch.float32)


def _forward_fake(
    msa: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps_m: float,
    eps_z: float,
    p_drop: float,
) -> list[torch.Tensor]:
    """The output and what the backward keeps: w [8, L, L], v and o [8 ns, L, S C / ns] (head-major, the MSA rows in ns chunks), the dropout keep-mask [L, d_msa] (empty without dropout)."""
    depth, length = msa.shape[1], msa.shape[2]
    c = leaves[2].shape[0] // 8
    ns = _sm80().pick_split(depth)
    e = msa.new_empty
    return [
        torch.empty_like(msa),
        e((8, length, length)),
        e((8 * ns, length, depth // ns * c)),
        e((8 * ns, length, depth // ns * c)),
        e((length, msa.shape[3]) if p_drop > 0 else (0,)),
    ]


@opaque(fake=_forward_fake, name="pwa_sm80_train_fwd")
def forward(
    msa: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps_m: float,
    eps_z: float,
    p_drop: float,
) -> list[torch.Tensor]:
    sm = _sm80()
    lnm_w, lnm_b, wv, wg, lnz_w, lnz_b, wb, wo = _kernel_leaves(leaves)
    with torch.cuda.device(msa.device):
        m = msa[0]
        w = sm.pair_fwd(
            pair[0],
            _key_mask(mask),
            lnz_w.contiguous(),
            lnz_b.contiguous(),
            wb.contiguous(),
            eps_z,
        )
        ns = sm.pick_split(m.shape[0])
        v, _ = sm.ln_v(m, lnm_w.contiguous(), lnm_b.contiguous(), wv.contiguous(), eps_m, False, ns)
        keep, dscale = None, 1.0
        if p_drop > 0:  # Dropout(broadcast_dim=1): one keep decision per (token, channel), shared over the MSA rows
            keep = (torch.rand(m.shape[1], m.shape[2], device=m.device) > p_drop).to(
                torch.bfloat16
            )  # fp32 draw: a bf16 `rand` has 8 mantissa bits and would bias the keep probability
            dscale = 1.0 / (1.0 - p_drop)
        if _fused_contract_enabled(m, wg, True):
            from miniworld_engine.kernels.pair_weighted_averaging.cuda import (
                contract_gate_fused_sm80,
            )

            out, o = contract_gate_fused_sm80.result(m, w, v, lnm_w, lnm_b, wg, wo, keep, eps_m, dscale, save_o=True, mode=1)
        else:
            o = _broadcast_contract(w, v)
            out = sm.gate_out(
                m,
                o,
                lnm_w.contiguous(),
                lnm_b.contiguous(),
                wg.contiguous(),
                wo.contiguous(),
                keep,
                eps_m,
                dscale,
            )
    return [out.unsqueeze(0), w, v, cast(torch.Tensor, o), msa.new_empty((0,)) if keep is None else keep]  # (save_o: o is returned)


def _backward_fake(
    msa: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps_m: float,
    eps_z: float,
    p_drop: float,
    saved: list[torch.Tensor],
    dy: torch.Tensor,
) -> list[torch.Tensor]:
    """dmsa, dpair and the gradient of every leaf, each in its own dtype."""
    return [torch.empty_like(msa), torch.empty_like(pair)] + [torch.empty_like(t) for t in leaves]


@opaque(fake=_backward_fake, name="pwa_sm80_train_bwd")
def backward(
    msa: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps_m: float,
    eps_z: float,
    p_drop: float,
    saved: list[torch.Tensor],
    dy: torch.Tensor,
) -> list[torch.Tensor]:
    sm = _sm80()
    lnm_w, lnm_b, wv, wg, lnz_w, lnz_b, wb, wo = _kernel_leaves(leaves)
    pdt = [t.dtype for t in leaves]
    w, v, o, keep = saved
    keep_t = None if keep.numel() == 0 else keep
    dscale = 1.0 / (1.0 - p_drop) if p_drop > 0 else 1.0
    with torch.cuda.device(msa.device):
        m, z = msa[0], pair[0]
        dres = dy[0].contiguous()
        typed_override = None
        typed = typed_override if typed_override is not None else m.shape[1] in (128, 256, 384)
        glue = sm.glue
        glue_override = None
        tuned_glue = glue_override if glue_override is not None else m.shape[1] in (256, 384, 512, 640, 768)
        if tuned_glue and m.shape[0] == 1024 and m.shape[2] == 64 and wg.shape[0] == 256 and m.shape[1] in (128, 256, 384, 512, 640, 768):
            from miniworld_engine.kernels.pair_weighted_averaging.cuda import (
                glue_tuned_sm80,
            )

            glue_mode = 1
            glue = lambda *args, **kwargs: glue_tuned_sm80.glue(*args, **kwargs, mode=glue_mode)
        grad_reduce_forced = None
        if grad_reduce_forced is None:
            grad_reduce_forced = m.shape == (1024, 128, 64) and wg.shape == (256, 64) and z.shape == (128, 128, 128)
        grad_reduce = (
            grad_reduce_forced
            and m.shape[0] == 1024
            and m.shape[2] == 64
            and wg.shape == (256, 64)
            and z.shape[-1] == 128
            and m.shape[1] in (128, 256, 384, 512, 640, 768)
        )
        if grad_reduce:
            typed = True
            from miniworld_engine.kernels.pair_weighted_averaging.cuda import (
                grad_partials_sm80,
            )

            glued = grad_partials_sm80.glue(
                m,
                dres,
                o,
                lnm_w.contiguous(),
                lnm_b.contiguous(),
                wg.contiguous(),
                wo.contiguous(),
                keep_t,
                eps_m,
                dscale,
                tuned_glue,
            )
        else:
            glued = glue(
                m,
                dres,
                o,
                lnm_w.contiguous(),
                lnm_b.contiguous(),
                wg.contiguous(),
                wo.contiguous(),
                keep_t,
                eps_m,
                dscale,
                raw=typed,
            )
        do, dgp = glued[:2]
        dv = _broadcast_contract(w, do, transpose=True)  # [8 ns, L(j), S C / ns]
        partial_pair = (
            True
            and do.shape == v.shape == (32, 128, 8192)
            and do.dtype == v.dtype == torch.bfloat16
            and do.is_contiguous()
            and v.is_contiguous()
            and z.shape == (128, 128, 128)
        )
        if partial_pair:
            from miniworld_engine.kernels.pair_weighted_averaging.cuda import (
                pair_partial_sm80,
            )

            dw = pair_partial_sm80.contract(do, v)
        else:
            dw = _weight_contract(do, v)  # ns partial products per head (pair_bwd sums them)
        del do
        if grad_reduce:
            from miniworld_engine.kernels.pair_weighted_averaging.cuda import (
                grad_reduce_vector_sm80,
            )

            dm, dgv_partial = grad_partials_sm80.dgv(
                m,
                dres,
                dgp,
                dv,
                lnm_w.contiguous(),
                lnm_b.contiguous(),
                wg.contiguous(),
                wv.contiguous(),
                eps_m,
            )
            del dgp, dv
            dz, pm, ps = grad_partials_sm80.pair(
                z,
                w,
                dw,
                _key_mask(mask),
                lnz_w.contiguous(),
                lnz_b.contiguous(),
                wb.contiguous(),
                eps_z,
            )
            dtype_bits = sum(1 << i for i, dt in enumerate(pdt) if dt == torch.bfloat16)
            outputs = grad_reduce_vector_sm80.result(glued[2], dgv_partial, pm, ps, lnz_w, lnz_b, wb, dtype_bits)
            return [dm.unsqueeze(0), dz.unsqueeze(0), *outputs]
        dm, dwv, dgm, dbm = sm.dgv_bwd(
            m,
            dres,
            dgp,
            dv,
            lnm_w.contiguous(),
            lnm_b.contiguous(),
            wg.contiguous(),
            wv.contiguous(),
            eps_m,
        )
        del dgp, dv
        if partial_pair:
            key_mask = _key_mask(mask)
            dz, pm, ps = pair_partial_sm80.pair(
                z,
                w,
                dw,
                z.new_empty((0,), dtype=torch.uint8) if key_mask is None else key_mask,
                lnz_w.contiguous(),
                lnz_b.contiguous(),
                wb.contiguous(),
                eps_z,
            )
            big_m = sm.reduce_rows(pm, torch.float32).view(8, z.shape[-1])
            sums = sm.reduce_rows(ps, torch.float32)
            if typed:
                paired = (dz, big_m, sums)
            else:
                wbf = wb.float()
                paired = (
                    dz,
                    lnz_w[None, :] * big_m + lnz_b[None, :] * sums[:, None],
                    (wbf * big_m).sum(0),
                    (wbf * sums[:, None]).sum(0),
                )
        else:
            paired = sm.pair_bwd(
                z,
                w,
                dw,
                _key_mask(mask),
                lnz_w.contiguous(),
                lnz_b.contiguous(),
                wb.contiguous(),
                eps_z,
                raw=typed,
            )
        dz = paired[0]
        if typed:
            outputs = [torch.empty(t.shape, dtype=t.dtype, device=t.device) for t in leaves]
            sm.ext().finalize_typed(
                [dgm, dbm, dwv, glued[2], paired[1], paired[2], lnz_w, lnz_b],
                wb,
                outputs,
            )
            return [dm.unsqueeze(0), dz.unsqueeze(0), *outputs]
        dwo, dwg = cast("tuple[torch.Tensor, ...]", glued)[2:]  # (not typed: the glue returns four)
        dwb, dgz, dbz = cast("tuple[torch.Tensor, ...]", paired)[1:]  # (not typed: four)
    # custom-op outputs may not alias each other: the small gradients are views of reduction buffers, so each leaves as its own copy in its leaf's dtype
    return [
        dm.unsqueeze(0),
        dz.unsqueeze(0),
        dgm.to(pdt[0], copy=True),
        dbm.to(pdt[1], copy=True),
        dwv.to(pdt[2], copy=True),
        dwg.to(pdt[3], copy=True),
        dgz.to(pdt[4], copy=True),
        dbz.to(pdt[5], copy=True),
        dwb.to(pdt[6], copy=True),
        dwo.to(pdt[7], copy=True),
    ]


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
        eps_m, eps_z = ctx.eps
        grads = backward(
            msa,
            pair,
            mask,
            leaves,
            eps_m,
            eps_z,
            ctx.p_drop,
            saved,
            dy.contiguous(),
        )
        return (None, None, None, grads[0], grads[1], None, *grads[2:])


def update_train(module, msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    p_drop = float(module.drop_msa.p_drop) if module.training else 0.0
    if _wide(module):
        return pwa_wide_sm80.update_train(
            _leaves(module), msa.contiguous(), pair.contiguous(), mask, float(module.ln_msa.eps), float(module.ln_pair.eps), p_drop
        )
    return _Training.apply(
        float(module.ln_msa.eps),
        float(module.ln_pair.eps),
        p_drop,
        msa.contiguous(),
        pair.contiguous(),
        mask,
        *_leaves(module),
    )
