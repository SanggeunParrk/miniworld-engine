"""Automatic dispatch to the A100 (sm_80) OuterProductMean: hand CUDA around cuBLAS, inference and training.

Contract: bf16 MSA ``[1, S, L, d_msa]`` (d_msa 64 or 128, any S, any L), d_hidden 32, d_pair 128 / 256 / 384, an optional bool mask ``[1, S, L]``, an optional bf16
pair residual ``[1, L, L, d_pair]`` (fused into the epilogue; its gradient is the output's), either normalisation order (``normalize_before_proj``), no interchain
mask, no projection bias / groups / count_offset, capability 8.0. ``refusal`` is the range; ``modules.OuterProductMean``'s ``cuda`` backend refuses everything else
(no fallback).

The step (``kernels/outer_product_mean/cuda/sm80``; ``kernels/outer_product_mean/cuda/sm80.py`` documents the kernels):

  forward   prologue (LayerNorm + both projections + mask -> A, B [S, 32 L], s-major) -> O = A^T B (cuBLAS, the grouped outer product) -> mask counts n = mask^T mask
            (a bf16 GEMM with fp32 accumulation: exact) -> epilogue (O / n, the 1024 -> d_pair projection, bias, residual).
  backward  dgrad (dz / n -> dO in the grouped layout, dbo) -> dA = B dO^T, dB = A dO (own CUDA at S1024/L128/CM64/CZ128; cuBLAS elsewhere) -> dwo (split over rows of i, fixed-order sum off the kept O)
            -> prologue_bwd (mask, both projection gradients, the LayerNorm backward -> dmsa, dWl, dWr, dgamma, dbeta).

Inference is one opaque op; training is one autograd Function whose forward and backward are each one opaque op. The backward keeps O (1.2 GB at L768). Numerics and timings: ``docs/gpus/a100/outer_product/``.

The qualified short training pipeline uses C++ stage dispatch and two own-CUDA gradient GEMMs; its forward outer product still uses cuBLAS.
"""

from __future__ import annotations

import warnings

import torch

from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import opaque

_FAILED = False
CH = 32


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
        warnings.warn(
            f"sm_80 OuterProductMean kernels unavailable, keeping the module path: {exc!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


def _sm80():
    from miniworld_engine.kernels.outer_product_mean.cuda import sm80

    return sm80


def refusal(
    module,
    msa: torch.Tensor,
    mask: torch.Tensor | None,
    token_asym_id: torch.Tensor | None,
    residual: torch.Tensor | None,
) -> str | None:
    """Why the sm_80 kernels cannot run this call, or ``None`` (the contract above). Shapes, dtypes and the device only, plus the build."""
    if not msa.is_cuda or torch.cuda.get_device_capability(msa.device) != (8, 0):
        return "the sm_80 OuterProductMean kernels are the A100 (capability 8.0) builds"
    if getattr(module, "groups", 1) > 1:
        return "grouped OPM (Chai-1): no sm_80 build"
    if module.to_left.bias is not None:
        return "projection biases (RoseTTAFold3): no sm_80 build"
    if getattr(module, "count_offset", None) is not None:
        return "count_offset: the sm_80 epilogue divides by the clamped pair count only"
    if module.mask_interchain and token_asym_id is not None:
        return "the interchain mask: no sm_80 build"
    ch, d_msa = module.to_left.weight.shape
    cz = module.to_out.weight.shape[0]
    if module.ln_msa.weight is None or module.ln_msa.bias is None or not _sm80().supported(d_msa, ch, cz) or module.to_out.weight.shape[1] != ch * ch:
        return (f"d_msa {d_msa}, d_hidden {ch}, d_pair {cz}: not an sm_80 build (d_hidden {CH}, d_msa {_sm80().D_MSA}, d_pair {_sm80().D_PAIR}, "
                "affine LayerNorm)")
    if msa.dtype != torch.bfloat16 or msa.ndim != 4 or msa.shape[3] != d_msa or msa.shape[1] < 1:
        return "msa must be a bfloat16 [1, S, L, d_msa]"
    if msa.shape[0] != 1:
        return f"batch {msa.shape[0]}: the sm_80 kernels take one sample (batch 1)"
    length = msa.shape[2]
    if residual is not None and (residual.dtype != torch.bfloat16 or tuple(residual.shape) != (1, length, length, cz)):
        return "the pair residual must be a bfloat16 [1, L, L, d_pair]"
    if mask is not None and (tuple(mask.shape) != tuple(msa.shape[:3]) or mask.dtype != torch.bool):
        return "mask must be a bool [1, S, L] mask"
    if not _loads():
        return "the sm_80 OuterProductMean extension failed to build"
    return None


def _eligible(module, msa, mask, token_asym_id, residual) -> bool:
    return refusal(module, msa, mask, token_asym_id, residual) is None


def serves_inference(
    module,
    msa: torch.Tensor,
    mask: torch.Tensor | None = None,
    token_asym_id: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> bool:
    return not torch.is_grad_enabled() and _eligible(
        module, msa, mask, token_asym_id, residual
    )


def serves_train(
    module,
    msa: torch.Tensor,
    mask: torch.Tensor | None = None,
    token_asym_id: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> bool:
    return torch.is_grad_enabled() and _eligible(
        module, msa, mask, token_asym_id, residual
    )


def _leaves(module) -> list[torch.Tensor]:
    """The module's own parameters (LayerNorm weight / bias, Wl, Wr, Wo, the bias; bf16 or fp32 masters). The opaque ops cast them to the kernels' dtypes (``_kernel_leaves``)
    outside autograd, so no cast copy sits in the graph and the weight gradients leave the backward unrounded in each parameter's dtype."""
    return [
        module.ln_msa.weight,
        module.ln_msa.bias,
        module.to_left.weight,
        module.to_right.weight,
        module.to_out.weight,
        module.to_out.bias,
    ]


def _kernel_leaves(leaves: list[torch.Tensor]) -> list[torch.Tensor]:
    """LayerNorm weight / bias (fp32), Wl, Wr, Wo (bf16), the bias (its bf16 value in fp32), contiguous. Cached while the parameters are unmodified (``_capture.param_forms``:
    an inference graph under ``static_weights`` and a training step's forward and backward cast them once instead of ~10 small kernels per call)."""
    lnw, lnb, wl, wr, wo, bo = leaves
    bf = torch.bfloat16
    return list(
        _capture.param_forms(
            "opm_sm80",
            leaves,
            lambda: (
                lnw.float().contiguous(),
                lnb.float().contiguous(),
                wl.to(bf).contiguous(),
                wr.to(bf).contiguous(),
                wo.to(bf).contiguous(),
                bo.to(bf).float().contiguous(),
            ),
        )
    )


def _norm(
    mask: torch.Tensor | None, depth: int, clamp: bool = True, training: bool = False
) -> tuple[torch.Tensor | None, float]:
    """The mask counts n_ij = sum_s mask[s, i] mask[s, j] clamped to >= 1 (fp32 [L, L], exact: a bf16 0 / 1 product with fp32 accumulation), or (None, S) without a mask."""
    if mask is None:
        return None, float(depth)
    override = None
    native = (
        override
        if override is not None
        else not training
        and (depth, mask.shape[1])
        in (
            (1024, 128),
            (1024, 256),
            (2048, 128),
            (2048, 768),
            (4096, 128),
            (4096, 640),
        )
    )
    if native:
        from miniworld_engine.kernels.outer_product_mean.cuda import (
            mask_count_ballot_sm80,
        )

        cfg = 2 if depth <= 1024 or mask.shape[1] == 768 else 1
        return mask_count_ballot_sm80.result(mask.contiguous(), clamp, cfg), 1.0
    mf = mask.to(torch.bfloat16)
    norm = torch.mm(mf.t(), mf, out_dtype=torch.float32)
    return (norm.clamp_(min=1.0) if clamp else norm), 1.0


def _mask_u8(mask: torch.Tensor | None) -> torch.Tensor | None:
    return None if mask is None else mask[0].contiguous().view(torch.uint8)


def _mask_epilogue_enabled(
    m: torch.Tensor, wo: torch.Tensor, training: bool = False
) -> bool:
    override = None
    if override is not None:
        return override
    return (
        not training
        and m.shape[-1] == 64
        and wo.shape[0] == 128
        and (m.shape[0], m.shape[1])
        in (
            (1024, 128),
            (1024, 256),
            (1024, 512),
            (2048, 256),
            (2048, 768),
            (4096, 640),
        )
    )


def _pipeline_enabled(msa: torch.Tensor, leaves: list[torch.Tensor]) -> bool:
    # E053 measures host dispatch overhead. The grouped GEMMs still use cuBLAS.
    return (
        False
        and msa.shape[1] == 1024
        and msa.shape[-1] == 64
        and leaves[4].shape[0] == 128
    )


def _native128_pipeline_enabled(msa: torch.Tensor, leaves: list[torch.Tensor]) -> bool:
    return (
        True
        and msa.shape[1:3] == (1024, 128)
        and msa.shape[-1] == 64
        and leaves[4].shape[0] == 128
    )


# ------------------------------------------------------------------------------------------------------------ inference
def _inference_fake(
    msa: torch.Tensor,
    mask: torch.Tensor | None,
    residual: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps: float,
    norm_first: bool,
) -> torch.Tensor:
    """msa [1, S, L, d_msa] -> the pair update (+ residual) [1, L, L, d_pair], bf16."""
    length = msa.shape[2]
    return msa.new_empty((1, length, length, leaves[4].shape[0]))


@opaque(fake=_inference_fake, name="opm_sm80_inference")
def inference(
    msa: torch.Tensor,
    mask: torch.Tensor | None,
    residual: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps: float,
    norm_first: bool,
) -> torch.Tensor:
    """residual + OuterProductMean(msa): prologue -> cuBLAS grouped outer product -> epilogue (no autograd, nothing saved)."""
    sm = _sm80()
    lnw, lnb, wl, wr, wo, bo = _kernel_leaves(leaves)
    with torch.cuda.device(msa.device):
        m = msa[0]
        mk = _mask_u8(mask)
        a, b, _ = sm.prologue(m, mk, lnw, lnb, wl, wr, eps, False)
        o = sm.lt_mm(a, b, True, False)
        del a, b
        if mk is not None and _mask_epilogue_enabled(m, wo):
            from miniworld_engine.kernels.outer_product_mean.cuda import (
                mask_epilogue_sm80,
            )

            out, _ = mask_epilogue_sm80.result(
                o,
                mk,
                wo,
                bo,
                None if residual is None else residual[0].contiguous(),
                norm_first,
            )
        else:
            norm, nconst = _norm(
                None if mk is None else mk,
                m.shape[0],
                clamp=True,   # the reference divides by clamp(n, 1)
            )
            out = sm.epilogue(
                o,
                norm,
                nconst,
                wo,
                bo,
                None if residual is None else residual[0].contiguous(),
                norm_first,
            )
    return out.unsqueeze(0)


def update_inference(
    module, msa: torch.Tensor, mask: torch.Tensor | None, residual: torch.Tensor | None
) -> torch.Tensor:
    return inference(
        msa.contiguous(),
        mask,
        residual,
        _leaves(module),
        float(module.ln_msa.eps),
        bool(module.normalize_before_proj),
    )


# ------------------------------------------------------------------------------------------------------------- training
def _forward_fake(
    msa: torch.Tensor,
    mask: torch.Tensor | None,
    residual: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps: float,
    norm_first: bool,
    save_o: bool,
) -> list[torch.Tensor]:
    """The output and what the backward keeps: A, B [S, 32 L], the LayerNorm statistics [S, L, 2], O [32 L, 32 L] (empty when it is recomputed), the mask counts [L, L] (empty without a mask)."""
    depth, length = msa.shape[1], msa.shape[2]
    e = msa.new_empty
    return [
        e((1, length, length, leaves[4].shape[0])),
        e((depth, length * CH)),
        e((depth, length * CH)),
        e((depth, length, 2), dtype=torch.float32),
        e((length * CH, length * CH)) if save_o else e((0,)),
        e((length, length), dtype=torch.float32)
        if mask is not None
        else e((0,), dtype=torch.float32),
    ]


@opaque(fake=_forward_fake, name="opm_sm80_train_fwd")
def forward(
    msa: torch.Tensor,
    mask: torch.Tensor | None,
    residual: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps: float,
    norm_first: bool,
    save_o: bool,
) -> list[torch.Tensor]:
    if _native128_pipeline_enabled(msa, leaves):
        from miniworld_engine.kernels.outer_product_mean.cuda import (
            pipeline_native128_sm80,
        )

        values = pipeline_native128_sm80.forward(
            msa[0],
            _mask_u8(mask),
            None if residual is None else residual[0].contiguous(),
            leaves,
            eps,
            norm_first,
            True,
            save_o,
        )
        return [values[0].unsqueeze(0), *values[1:]]
    if _pipeline_enabled(msa, leaves):
        from miniworld_engine.kernels.outer_product_mean.cuda import pipeline_sm80

        values = pipeline_sm80.forward(
            msa[0],
            _mask_u8(mask),
            None if residual is None else residual[0].contiguous(),
            leaves,
            eps,
            norm_first,
            True,
            save_o,
        )
        return [values[0].unsqueeze(0), *values[1:]]
    sm = _sm80()
    lnw, lnb, wl, wr, wo, bo = _kernel_leaves(leaves)
    with torch.cuda.device(msa.device):
        m = msa[0]
        mk = _mask_u8(mask)
        a, b, stats = sm.prologue(m, mk, lnw, lnb, wl, wr, eps, True)
        o = sm.lt_mm(a, b, True, False)
        if mk is not None and _mask_epilogue_enabled(m, wo, training=True):
            from miniworld_engine.kernels.outer_product_mean.cuda import (
                mask_epilogue_sm80,
            )

            out, norm = mask_epilogue_sm80.result(
                o,
                mk,
                wo,
                bo,
                None if residual is None else residual[0].contiguous(),
                norm_first,
                True,
            )
        else:
            norm, nconst = _norm(mk, m.shape[0], training=True)
            out = sm.epilogue(
                o,
                norm,
                nconst,
                wo,
                bo,
                None if residual is None else residual[0].contiguous(),
                norm_first,
            )
        if not save_o:
            o = msa.new_empty((0,))
        if norm is None:
            norm = msa.new_empty((0,), dtype=torch.float32)
    return [out.unsqueeze(0), a, b, stats, o, norm]


def _backward_fake(
    msa: torch.Tensor,
    mask: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps: float,
    norm_first: bool,
    saved: list[torch.Tensor],
    dz: torch.Tensor,
) -> list[torch.Tensor]:
    """dmsa and the gradient of every leaf (LayerNorm weight / bias, Wl, Wr, Wo, bias), each in its leaf's dtype."""
    return [torch.empty_like(msa)] + [torch.empty_like(t) for t in leaves]


@opaque(fake=_backward_fake, name="opm_sm80_train_bwd")
def backward(
    msa: torch.Tensor,
    mask: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps: float,
    norm_first: bool,
    saved: list[torch.Tensor],
    dz: torch.Tensor,
) -> list[torch.Tensor]:
    if _native128_pipeline_enabled(msa, leaves):
        from miniworld_engine.kernels.outer_product_mean.cuda import (
            pipeline_native128_sm80,
        )

        values = pipeline_native128_sm80.backward(
            msa[0], _mask_u8(mask), leaves, saved, dz[0].contiguous(), norm_first
        )
        return [values[0].unsqueeze(0), *values[1:]]
    if _pipeline_enabled(msa, leaves):
        from miniworld_engine.kernels.outer_product_mean.cuda import pipeline_sm80

        values = pipeline_sm80.backward(
            msa[0], _mask_u8(mask), leaves, saved, dz[0].contiguous(), norm_first
        )
        return [values[0].unsqueeze(0), *values[1:]]
    sm = _sm80()
    lnw, lnb, wl, wr, wo, _ = _kernel_leaves(leaves)
    pdt = [t.dtype for t in leaves]
    a, b, stats, o, norm = saved
    depth, length = msa.shape[1], msa.shape[2]
    with torch.cuda.device(msa.device):
        mk = _mask_u8(mask)
        wo16 = wo
        nconst = float(depth)
        nrm = None if norm.numel() == 0 else norm
        if nrm is not None:
            nconst = 1.0
        dO, dzn, dbo = sm.dgrad(
            dz.contiguous().view(length, length, -1), nrm, nconst, wo16, norm_first
        )
        da = sm.lt_mm(b, dO, False, True)  # [S, 32 L]: dA[s, (i, c)]
        db = sm.lt_mm(a, dO, False, False)  # dB[s, (j, e)]
        del dO
        if o.numel() == 0:
            o = sm.lt_mm(a, b, True, False)
        dwo = sm.dwo(dzn, o, pdt[4])
        del o, dzn
        dm, dwl, dwr, dg, dbt = sm.prologue_bwd(
            da, db, msa[0], stats, mk, lnw, lnb, wl, wr
        )
    if False:
        from miniworld_engine.kernels.outer_product_mean.cuda import finalize_typed_sm80

        typed = finalize_typed_sm80.result(
            [dg, dbt, dwl, dwr, dbo], [leaves[i] for i in (0, 1, 2, 3, 5)]
        )
        return [dm.unsqueeze(0), *typed[:4], dwo, typed[4]]
    # custom-op outputs may not alias each other: the small gradients are views of one reduction buffer, so each leaves as its own copy
    return [
        dm.unsqueeze(0),
        dg.to(pdt[0], copy=True),
        dbt.to(pdt[1], copy=True),
        dwl.to(pdt[2], copy=True),
        dwr.to(pdt[3], copy=True),
        dwo,
        dbo.to(pdt[5], copy=True),
    ]


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eps, norm_first, save_o, msa, mask, residual, *leaves):
        out, *saved = forward(
            msa, mask, residual, list(leaves), eps, norm_first, save_o
        )
        ctx.eps, ctx.norm_first, ctx.has_residual = (
            eps,
            norm_first,
            residual is not None,
        )
        ctx.save_for_backward(msa, mask, *leaves, *saved)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dz):
        v = ctx.saved_tensors
        msa, mask, leaves, saved = v[0], v[1], list(v[2:8]), list(v[8:])
        grads = backward(
            msa, mask, leaves, ctx.eps, ctx.norm_first, saved, dz.contiguous()
        )
        return (
            None,
            None,
            None,
            grads[0],
            None,
            dz if ctx.has_residual else None,
            *grads[1:],
        )


def update_train(
    module, msa: torch.Tensor, mask: torch.Tensor | None, residual: torch.Tensor | None
) -> torch.Tensor:
    save_o = True
    return _Training.apply(
        float(module.ln_msa.eps),
        bool(module.normalize_before_proj),
        save_o,
        msa.contiguous(),
        mask,
        residual,
        *_leaves(module),
    )
