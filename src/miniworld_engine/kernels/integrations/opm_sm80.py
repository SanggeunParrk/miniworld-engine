"""Automatic dispatch to the A100 (sm_80) OuterProductMean: hand CUDA around cuBLAS, inference and training.

Contract: bf16 MSA ``[1, S, L, d_msa]`` (d_msa 64 or 128, any S, any L), d_hidden 32, d_pair 128 / 256 / 384, an optional bool mask ``[1, S, L]``, an optional bf16
pair residual ``[1, L, L, d_pair]`` (fused into the epilogue; its gradient is the output's), either normalisation order (``normalize_before_proj``), no interchain
mask, RoseTTAFold3's projection biases (added by the prologue), ``count_offset`` (AlphaFold 3's divisor n + offset: the epilogues and the dgrad take max(n, dmin) + doff, FoldForge R2), no groups, capability 8.0. ``refusal`` is the range; ``modules.OuterProductMean``'s ``cuda`` backend refuses everything else
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

import torch

from miniworld_engine.kernels import _capture
from miniworld_engine.kernels import _msa_card as _card
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._nvcc import build_failed

CH = 32
#: The port's E053 short training pipeline (``pipeline_sm80``), an opt-in experiment: off.
_PIPELINE = False


@torch.compiler.assume_constant_result
def _loads() -> bool:
    """Builds (first call) or loads the sm_80 extension; a failed build / load raises ``_nvcc.KernelBuildError`` (never a fallback).

    A process-level constant, so ``torch.compile`` evaluates it once at trace time instead of tracing the nvcc lookup / JIT build into the graph."""
    try:
        _sm80().ext()
    except Exception as exc:  # a build / load failure is an error, never a fallback
        build_failed("sm_80 OuterProductMean", exc)
        return False  # reached only while planning (build_failed returns there)
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
    if not _card.device_ok(msa):
        return "the sm_80 OuterProductMean kernels are the A100 (capability 8.0) and RTX 5090 (12.0, sm_120a) builds"
    if getattr(module, "groups", 1) > 1:
        return "grouped OPM (Chai-1): no sm_80 build"
    if module.mask_interchain and token_asym_id is not None:
        return "the interchain mask: no sm_80 build"
    ch, d_msa = module.to_left.weight.shape
    cz = module.to_out.weight.shape[0]
    if (
        module.ln_msa.weight is None
        or module.ln_msa.bias is None
        or not _sm80().supported(d_msa, ch, cz)
        or module.to_out.weight.shape[1] != ch * ch
    ):
        return (
            f"d_msa {d_msa}, d_hidden {ch}, d_pair {cz}: not an sm_80 build (d_hidden {CH}, d_msa {_sm80().D_MSA}, d_pair {_sm80().D_PAIR}, "
            "affine LayerNorm)"
        )
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
    return not torch.is_grad_enabled() and _eligible(module, msa, mask, token_asym_id, residual)


def serves_train(
    module,
    msa: torch.Tensor,
    mask: torch.Tensor | None = None,
    token_asym_id: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> bool:
    return torch.is_grad_enabled() and _eligible(module, msa, mask, token_asym_id, residual)


def _divisor(module) -> tuple[float, float]:
    """(dmin, doff) of the pair divisor max(n, dmin) + doff: (1, 0) the clamped count, (0, count_offset) AlphaFold 3's n + offset."""
    offset = getattr(module, "count_offset", None)
    return (1.0, 0.0) if offset is None else (0.0, float(offset))


def _leaves(module) -> list[torch.Tensor]:
    """The module's own parameters (LayerNorm weight / bias, Wl, Wr, Wo, the bias, and RoseTTAFold3's left / right projection biases when present; bf16 or fp32 masters). The opaque ops cast them to the kernels' dtypes (``_kernel_leaves``)
    outside autograd, so no cast copy sits in the graph and the weight gradients leave the backward unrounded in each parameter's dtype."""
    return [
        module.ln_msa.weight,
        module.ln_msa.bias,
        module.to_left.weight,
        module.to_right.weight,
        module.to_out.weight,
        module.to_out.bias,
    ] + ([] if module.to_left.bias is None else [module.to_left.bias, module.to_right.bias])


def _kernel_leaves(leaves: list[torch.Tensor]) -> list[torch.Tensor]:
    """LayerNorm weight / bias (fp32), Wl, Wr, Wo (bf16), the bias (its bf16 value in fp32), the projection biases when present (their bf16 values in fp32), contiguous. Cached while the parameters are unmodified (``_capture.param_forms``:
    an inference graph under ``static_weights`` and a training step's forward and backward cast them once instead of ~10 small kernels per call)."""
    lnw, lnb, wl, wr, wo, bo = leaves[:6]
    bf = torch.bfloat16

    def build():
        with torch.no_grad():
            return (
                lnw.float().contiguous(),
                lnb.float().contiguous(),
                wl.to(bf).contiguous(),
                wr.to(bf).contiguous(),
                wo.to(bf).contiguous(),
                bo.to(bf).float().contiguous(),
                *(t.to(bf).float().contiguous() for t in leaves[6:]),
            )

    return list(_capture.param_forms("opm_sm80", leaves, build))


def _norm(mask: torch.Tensor | None, depth: int, clamp: bool = True, training: bool = False) -> tuple[torch.Tensor | None, float]:
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


def _mask_epilogue_enabled(m: torch.Tensor, wo: torch.Tensor, training: bool = False) -> bool:
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
    return _PIPELINE and len(leaves) == 6 and msa.shape[1] == 1024 and msa.shape[-1] == 64 and leaves[4].shape[0] == 128


def _native128_pipeline_enabled(msa: torch.Tensor, leaves: list[torch.Tensor]) -> bool:
    return (
        not _card.sm120()  # the A100-tuned short pipeline (pipeline_native128_sm80) is not built for the RTX 5090
        and len(leaves) == 6  # (no projection biases)
        and msa.shape[1:3] == (1024, 128)
        and msa.shape[-1] == 64
        and leaves[4].shape[0] == 128
    )


def _outer(sm, a, b, mk):
    """O = A^T B: with a mask the CUTLASS product that skips the tokens masked in every MSA row (``ogemm_sm80``: their O blocks are exactly 0;
    padding is the common case), else cuBLAS (``sm.lt_mm``)."""
    if mk is None:
        return sm.lt_mm(a, b, True, False)
    from miniworld_engine.kernels.outer_product_mean.cuda import ogemm_sm80

    return ogemm_sm80.outer(a, b, ogemm_sm80.live_tokens(mk))


# ------------------------------------------------------------------------------------------------------------ inference
def _inference_fake(
    msa: torch.Tensor,
    mask: torch.Tensor | None,
    residual: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps: float,
    norm_first: bool,
    dmin: float = 1.0,
    doff: float = 0.0,
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
    dmin: float = 1.0,
    doff: float = 0.0,
) -> torch.Tensor:
    """residual + OuterProductMean(msa): prologue -> cuBLAS grouped outer product -> epilogue (no autograd, nothing saved). The pair divisor is
    max(n, dmin) + doff (``opm_train.divisor_params``)."""
    sm = _sm80()
    lnw, lnb, wl, wr, wo, bo, *pb = _kernel_leaves(leaves)
    with torch.cuda.device(msa.device):
        m = msa[0]
        mk = _mask_u8(mask)
        a, b, _ = sm.prologue(m, mk, lnw, lnb, wl, wr, eps, False, None, *pb)
        o = _outer(sm, a, b, mk)
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
                dmin=dmin,
                doff=doff,
            )
        else:
            norm, nconst = _norm(
                None if mk is None else mk,
                m.shape[0],
                clamp=dmin > 0,  # the reference divides by clamp(n, 1) or (count_offset) n + offset: the epilogue takes max(n, dmin) + doff
            )
            out = sm.epilogue(
                o,
                norm,
                nconst,
                wo,
                bo,
                None if residual is None else residual[0].contiguous(),
                norm_first,
                dmin=dmin,
                doff=doff,
            )
    return out.unsqueeze(0)


def update_inference(module, msa: torch.Tensor, mask: torch.Tensor | None, residual: torch.Tensor | None) -> torch.Tensor:
    return inference(
        msa.contiguous(),
        mask,
        residual,
        _leaves(module),
        float(module.ln_msa.eps),
        bool(module.normalize_before_proj),
        *_divisor(module),
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
    dmin: float = 1.0,
    doff: float = 0.0,
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
        e((length, length), dtype=torch.float32) if mask is not None else e((0,), dtype=torch.float32),
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
    dmin: float = 1.0,
    doff: float = 0.0,
) -> list[torch.Tensor]:
    """The training forward; the pair divisor max(n, dmin) + doff, the saved counts max(n, dmin) (the dgrad adds doff)."""
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
            dmin,
            doff,
        )
        return [values[0].unsqueeze(0), *values[1:]]
    if _pipeline_enabled(msa, leaves) and (dmin, doff) == (1.0, 0.0):
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
    lnw, lnb, wl, wr, wo, bo, *pb = _kernel_leaves(leaves)
    with torch.cuda.device(msa.device):
        m = msa[0]
        mk = _mask_u8(mask)
        a, b, stats = sm.prologue(m, mk, lnw, lnb, wl, wr, eps, True, None, *pb)
        o = _outer(sm, a, b, mk)
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
                dmin=dmin,
                doff=doff,
            )
        else:
            norm, nconst = _norm(mk, m.shape[0], clamp=dmin > 0, training=True)
            out = sm.epilogue(
                o,
                norm,
                nconst,
                wo,
                bo,
                None if residual is None else residual[0].contiguous(),
                norm_first,
                dmin=dmin,
                doff=doff,
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
    dmin: float = 1.0,
    doff: float = 0.0,
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
    dmin: float = 1.0,
    doff: float = 0.0,
) -> list[torch.Tensor]:
    if _native128_pipeline_enabled(msa, leaves):
        from miniworld_engine.kernels.outer_product_mean.cuda import (
            pipeline_native128_sm80,
        )

        values = pipeline_native128_sm80.backward(msa[0], _mask_u8(mask), leaves, saved, dz[0].contiguous(), norm_first, dmin, doff)
        return [values[0].unsqueeze(0), *values[1:]]
    if _pipeline_enabled(msa, leaves) and (dmin, doff) == (1.0, 0.0):
        from miniworld_engine.kernels.outer_product_mean.cuda import pipeline_sm80

        values = pipeline_sm80.backward(msa[0], _mask_u8(mask), leaves, saved, dz[0].contiguous(), norm_first)
        return [values[0].unsqueeze(0), *values[1:]]
    sm = _sm80()
    lnw, lnb, wl, wr, wo, _, *pb = _kernel_leaves(leaves)
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
        dO, dzn, dbo = sm.dgrad(dz.contiguous().view(length, length, -1), nrm, nconst, wo16, norm_first, dmin=dmin, doff=doff)
        da = sm.lt_mm(b, dO, False, True)  # [S, 32 L]: dA[s, (i, c)]
        db = sm.lt_mm(a, dO, False, False)  # dB[s, (j, e)]
        dpb = []
        if pb:  # the projection biases: sum over the valid (s, i) of dA | dB (a masked row lost its bias), one bf16 0 / 1 GEMM with fp32 accumulation
            rows = depth * length
            ones = msa.new_ones((1, rows)) if mk is None else mk.reshape(1, rows).to(torch.bfloat16)
            dpb = [torch.mm(ones, t[:depth].reshape(rows, CH), out_dtype=torch.float32).view(CH) for t in (da, db)]
        del dO
        if o.numel() == 0:
            o = sm.lt_mm(a, b, True, False)
        dwo = sm.dwo(dzn, o, pdt[4])
        del o, dzn
        dm, dwl, dwr, dg, dbt = sm.prologue_bwd(da, db, msa[0], stats, mk, lnw, lnb, wl, wr)
    if False:
        from miniworld_engine.kernels.outer_product_mean.cuda import finalize_typed_sm80

        typed = finalize_typed_sm80.result([dg, dbt, dwl, dwr, dbo], [leaves[i] for i in (0, 1, 2, 3, 5)])
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
        *(g.to(dt, copy=True) for g, dt in zip(dpb, pdt[6:], strict=True)),
    ]


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eps, norm_first, save_o, divisor, msa, mask, residual, *leaves):
        out, *saved = forward(msa, mask, residual, list(leaves), eps, norm_first, save_o, *divisor)
        ctx.divisor = divisor
        ctx.eps, ctx.norm_first, ctx.has_residual = (
            eps,
            norm_first,
            residual is not None,
        )
        ctx.n_leaves = len(leaves)
        ctx.save_for_backward(msa, mask, *leaves, *saved)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dz):
        v = ctx.saved_tensors
        n = 2 + ctx.n_leaves
        msa, mask, leaves, saved = v[0], v[1], list(v[2:n]), list(v[n:])
        grads = backward(msa, mask, leaves, ctx.eps, ctx.norm_first, saved, dz.contiguous(), *ctx.divisor)
        return (
            None,
            None,
            None,
            None,
            grads[0],
            None,
            dz if ctx.has_residual else None,
            *grads[1:],
        )


def update_train(module, msa: torch.Tensor, mask: torch.Tensor | None, residual: torch.Tensor | None) -> torch.Tensor:
    save_o = True
    return _Training.apply(
        float(module.ln_msa.eps),
        bool(module.normalize_before_proj),
        save_o,
        _divisor(module),
        msa.contiguous(),
        mask,
        residual,
        *_leaves(module),
    )
