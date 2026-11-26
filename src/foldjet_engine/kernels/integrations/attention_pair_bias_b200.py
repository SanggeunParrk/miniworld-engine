"""Automatic dispatch to the B200 (sm_100a) AttentionPairBias: inference and training, all hand CUDA but the plain GEMMs
(cuBLAS).

H100 inference and training use native CUDA by default for the required buckets.
H100 training keeps the fused pair-weight (Wb) gradient by default (bf16 operands, fp32
accumulation: the same error as the bf16 PyTorch module). ``EXACT_DWB = True`` (below) opts
into the fp32 "shadow": the Wb gradient recomputed from bf16-rounded inputs with native CUDA
and cuBLAS at L 384 / 768 for the five head layouts (ln_pair has no beta, so only dWb is
repaired), which materializes extra FP32 workspaces and costs about 2x the step.

Contract: bf16 activations (parameters bf16 or fp32 masters) or, H100, fp32 activations with fp32 parameters (the TF32 path at the
end of this file), B >= 1 (H100 bf16 inference: the batch rows in one launch per stage; else one step per batch row; the packs made
once: FoldJet 5I28 confidence head), d_single 384 as 8
heads x 48, 12 x 32, 24 x 16 or 16 x 24 (AF3's Pairformer single attention), or d_single 512 as 16 x 32; d_pair 128 (any
of those), 256 / 384 (8 or 16 heads) or 512 (16 heads) -- the wide pair kernels, inference and training --; gate_bias (a
constant on the gate rows of the packed bias); no QK-norm, sm_90 (H100: ``refusal`` is the whole gate, the ``cuda`` backend of
``modules.attention.AttentionPairBias`` states it) or sm_100. H100: L a multiple of 128.

16 x 24 runs 32 wide per head: the packed q | k | v | g weights carry 8 zero rows after each head's 24 (and Wo 8 zero columns),
so q, k, v, g, O and every gradient on that side are W = 512 wide with exact zeros in the pads; the attention kernels see heads
of 32 (-DDHP=32) and the softmax scale of 24. ``finalize`` gathers the real rows / columns of the weight gradients.

Inference (one opaque op):
  ln_rows (LN(single), bf16) -> addmm q | k | v | g (q pre-scaled by log2(e) / sqrt(head dim): logits in exp2 units)
  -> pair_bias (LN(pair) . Wf, Wf = to_bias.weight * ln_pair.weight * log2(e); masked keys -inf on H100, NEG = -1e30 on B200)
  -> attn_inf -DQPAIR (sigmoid(g) o over the q columns) -> to_out accumulated in place (cuBLAS beta = 1) onto the residual
  seed ln_rows wrote (out-of-place addmm would add a copy of x into the output first).
ln_pair has no bias: an offset would add Wb . b to every (i, j) of a head, which the softmax cancels.

Training (autograd Function, opaque forward / backward): the same forward in natural units with attn_fwd2 -DQPAIR (O, LSE),
gate_rows and addmm; backward: dO GEMMs, gate_bwd (dO, D, dg), attn_dkv / attn_dqb -DNHEAD, qkv_bwd, the dxa / dW GEMMs,
ln_bwd (+ residual), pair_bias_bwd (a wide pair: pair_bias_bwd_wide per 128-column chunk, reading the forward's row
stats and unmasked fp32 bias). B200 retains its finite masked logits. H100 uses
-inf in the native attention core, matching the FP32 efficient-SDPA reference's
zero attention update for an all-false key mask; its dWb shadow also applies
masked_fill's zero derivative at every masked key."""

from __future__ import annotations

import math
import os
from typing import Any

import torch
from torch.autograd.function import once_differentiable

import foldjet_engine._envcompat  # noqa: F401  (old MINIWORLD_* environment names)
from foldjet_engine.kernels import _capture, _cast_cuda
from foldjet_engine.kernels._compile import opaque
from foldjet_engine.kernels._key_len import kernel_mask

DP = 128
SHAPES = (
    (8, 384),
    (12, 384),
    (16, 384),
    (24, 384),
    (16, 512),
)  # (heads, d_single): 8 x 48, 12 x 32, 16 x 24 (padded to 32), 24 x 16, 16 x 32
LOG2E = math.log2(math.e)
NEG = -1e30  # masked keys of the B200 bf16 path: finite, so a fully masked sample stays NaN-free (H100: -inf, see the docstring)
_packs: dict = _capture.weight_cache("attention_pair_bias_b200.packs")
_cores: dict = {}
#: Opt-in fp32 pair-weight gradient ("shadow") of H100 training at L 384 / 768 for the SHAPES head layouts (2.4.x
#: ``settings.apb_h100_exact_dwb``). False (default): the fused kernel's dWb (bf16 operands, fp32 accumulation; the same error
#: as the bf16 PyTorch module and every other gradient of the block). True: recompute dWb in fp32 from bf16-rounded inputs
#: (about 2x lower error on that one gradient, about 2.4x the step). A plain switch: set it before the training forward.
EXACT_DWB = False


def refusal(module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, grad: bool, padded: bool = False) -> str | None:
    """Why the H100 (sm_90) native path cannot take this call (``grad``: the training Function, else the inference op), or ``None``.
    The ``cuda`` backend of ``AttentionPairBias`` states exactly this. ``padded``: the caller pads L to a multiple of 128 (inference)."""
    if not single.is_cuda:
        return "CUDA tensors only"
    cap = torch.cuda.get_device_capability(single.device)
    if cap not in ((9, 0), (10, 0)):
        return "the H100 (sm_90) / B200 (sm_100a) kernels only"
    if module.use_qk_norm:
        return "per-head q / k RMSNorm (the packed q | k | v | g projection has no norm between the GEMM and the core)"
    if single.dtype == torch.float32 and pair.dtype == torch.float32:
        if torch.is_autocast_enabled("cuda"):
            return "float32 activations under autocast (the TF32 path runs fp32 activations without autocast)"
        if any(p.dtype != torch.float32 for p in module.parameters()):
            return "float32 activations take float32 parameters (the TF32 path)"
    elif single.dtype != torch.bfloat16 or pair.dtype != torch.bfloat16:
        return (
            f"dtype {single.dtype} / {pair.dtype}: bfloat16 activations (bf16 wgmma cores; parameters bf16 or fp32 masters) or float32 "
            "activations with float32 parameters (TF32 cores)"
        )
    if single.ndim != 3 or single.shape[0] < 1 or not single.is_contiguous():
        return "single must be one contiguous [B, L, d_single] tensor (B >= 1)"
    B = single.shape[0]
    H, D = module.n_head, single.shape[2]
    if (H, D) not in SHAPES or tuple(module.to_query.weight.shape) != (D, D):
        return f"{H} heads x d_single {D}: the kernels are built for (heads, d_single) in {SHAPES} with n_head * d_hidden == d_single"
    dp = module.to_bias.weight.shape[1]
    widths = _rows().PAIR_WIDTHS
    if dp not in widths:
        return f"d_pair {dp}: the pair kernels take {tuple(widths)}"
    if H not in widths[dp]:
        return f"d_pair {dp} with {H} heads: the pair kernels take {widths[dp]} heads at that width"
    if module.ln_single.weight is None or module.ln_pair.weight is None:
        return "the kernels take the affine LayerNorms"
    L = single.shape[1]
    if L % 128 and not padded:
        return f"L {L}: a multiple of 128 only (inference pads it: AttentionPairBias._forward_cuda)"
    if tuple(pair.shape) != (B, L, L, dp) or not pair.is_contiguous():
        return f"pair must be one contiguous [{B}, {L}, {L}, {dp}] tensor"
    if mask is not None and (tuple(mask.shape) != (B, L) or mask.dtype != torch.bool):
        return f"mask must be a bool [{B}, {L}] key mask"
    return None


def _leaves(module) -> list[torch.Tensor]:
    """The module's parameters as they are (any float dtype, e.g. an fp32 master): ln_single w / b, Wq, bq, Wk, Wv, Wg, Wo,
    ln_pair w, Wb. The ops cast them (``_kernel_leaves``) outside autograd, so fp32 parameters get unrounded fp32 gradients."""
    return [
        module.ln_single.weight,
        module.ln_single.bias,
        module.to_query.weight,
        module.to_query.bias,
        module.to_key.weight,
        module.to_value.weight,
        module.to_gate.weight,
        module.to_out.weight,
        module.ln_pair.weight,
        module.to_bias.weight,
    ]


def _kernel_leaves(leaves) -> list[torch.Tensor]:
    """The kernels' dtypes: the LayerNorm vectors fp32, the projections bf16 (no copies when the parameters already are).
    Hopper: the projections stay in their own dtype (one dtype for the seven; ``prep`` rounds them to bf16 inside its pass, so an
    fp32 master costs no cast launch); only the LayerNorm vectors become fp32 (a no-op for fp32 parameters)."""
    lnw, lnb, wq, bq, wk, wv, wg, wo, lnpw, wb = leaves
    if torch.cuda.get_device_capability(lnw.device) in ((9, 0), (10, 0)):  # B200 (2026-11-17): as Hopper, prep_dp rounds
        dt = wq.dtype if wq.dtype in (torch.bfloat16, torch.float32) else torch.bfloat16
        w = [t if t.dtype == dt else t.to(dt) for t in (wq, bq, wk, wv, wg, wo, wb)]
        return [lnw.float(), lnb.float(), *w[:6], lnpw.float(), w[6]]
    bf = torch.bfloat16
    return [
        lnw.float(),
        lnb.float(),
        wq.to(bf),
        bq.to(bf),
        wk.to(bf),
        wv.to(bf),
        wg.to(bf),
        wo.to(bf),
        lnpw.float(),
        wb.to(bf),
    ]


def _rows():
    from foldjet_engine.kernels.augmented_attention.cuda import apb as rows

    return rows


def _sm100():
    from foldjet_engine.kernels.augmented_attention.cuda import sm100

    return sm100


def _training_core(device) -> Any:  # the sm_90 or the sm_100 kernel module (the same entry points, duck-typed)
    if torch.cuda.get_device_capability(device) == (9, 0):
        from foldjet_engine.kernels.augmented_attention import cuda as sm90

        return sm90
    return _sm100()


def _inference_core(device, heads: int, d: int) -> Any:  # the sm_90 or the sm_100 core (duck-typed; ``batch``: sm_90 only)
    if torch.cuda.get_device_capability(device) == (9, 0):
        from foldjet_engine.kernels.augmented_attention import cuda as sm90

        return sm90.ApbInferenceCore(device.index, heads, d)
    return _sm100().ApbInferenceCore(device.index, heads, d)


def _mask(mask: torch.Tensor | None, L: int) -> torch.Tensor | None:
    """The key mask as one contiguous bool [B, L] (the ops hand row ``b``, a contiguous [L], to the kernels)."""
    return None if mask is None else mask.reshape(-1, L).to(torch.bool).contiguous()


def _row(t: torch.Tensor | None, b: int) -> torch.Tensor | None:
    """Row ``b`` of a batched operand (``None`` stays ``None``)."""
    return None if t is None else t[b]


def _geometry(leaves):
    """(heads, real head dim, row width W, d_single) from to_bias.weight [heads, 128] and ln_single.weight [d_single]."""
    H, D = leaves[9].shape[0], leaves[0].numel()
    return H, D // H, _rows().width(H, D), D


# ------------------------------------------------------------------------------------------------------------ inference
def _inference_packs(params, leaves, gate_bias: float = 0.0, ln: list | None = None):
    """``_prep`` with q scaled into exp2 units and Wf in exp2 units, cached per parameter version (the step reads them every
    call; repacking costs a launch). Keyed on the parameters themselves (``params``), not on ``leaves``, their kernel-dtype casts,
    which are fresh tensors per call for fp32 parameters. Scoped to the CUDA-graph capture (``kernels._capture``): a capture packs
    once, recorded, so each replay packs the weights as they are then."""
    build = lambda: (
        *_prep(leaves, LOG2E / math.sqrt(_geometry(leaves)[1]), LOG2E, gate_bias, ln),
        params,
    )
    return _capture.lookup(
        _packs,
        (gate_bias, *((t.data_ptr(), t._version) for t in params)),
        build,
        limit=_capture.BLOCK_CACHE_LIMIT,
    )[:5]


def _prep(
    leaves, qs: float, ws: float, gate_bias: float = 0.0, ln: list | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """(W q | k | v | g [4 W, D] (padded rows, q x qs), its bias [4 W], Wf = to_bias.weight * ln_pair.weight * ws [H, 128], Wo
    [D, W] bf16 with padded columns -- or None when there is no padding (8 x 48: to_out.weight as is) --, Wb bf16 [H, 128] or
    None), bf16, one launch. Hopper always packs Wo and Wb (its weights may be an fp32 master that this pass rounds)."""
    _, _, wq, bq, wk, wv, wg, wo, lnpw, wb = leaves
    H, _, W, D = _geometry(leaves)
    if torch.cuda.get_device_capability(wq.device) == (10, 0):
        # B200: every form in ONE launch at any pair width (apb_rows_sm100.cu prep_dp: Wf / Wb over all dp columns, the gate constant),
        # fp32 masters read as they are (each value rounded to bf16 first, as a cast would): no cast or Wf launches per call
        dp = wb.shape[1]
        bf = torch.bfloat16
        wpack, bvec = wq.new_empty((4 * W, D), dtype=bf), wq.new_empty((4 * W,), dtype=bf)
        wf, wbo, wop = wq.new_empty((H, dp), dtype=bf), wq.new_empty((H, dp), dtype=bf), wq.new_empty((D, W), dtype=bf)
        E = _rows().ext()
        E.prep_mb(_PREP_MB if D == 512 else 0)  # v28: one wave at D 512
        if ln is not None and ln[0].dtype == bf and ln[0].is_contiguous():
            # the pack and LayerNorm(single) of the first batch row in one launch (prep_ln); ``ln`` = ln_rows' arguments, True appended
            E.prep_ln(wq, bq, wk, wv, wg, wo, wb, lnpw, wpack, bvec, wop, wf, qs, ws, wbo, float(gate_bias), *ln)
            ln.append(True)
        else:
            E.prep_dp(wq, bq, wk, wv, wg, wo, wb, lnpw, wpack, bvec, wop, wf, qs, ws, wbo, float(gate_bias))
        return wpack, bvec, wf, wop, wbo
    hopper = torch.cuda.get_device_capability(wq.device) == (9, 0)
    bf = torch.bfloat16
    wpack = wq.new_empty((4 * W, D), dtype=bf)
    bvec = wq.new_empty((4 * W,), dtype=bf)
    wf = wq.new_empty((H, DP), dtype=bf)
    wop = wq.new_empty((D, W), dtype=bf) if (W != D or hopper) else None
    wbo = wq.new_empty((H, DP), dtype=bf) if hopper else None
    wide = wb.shape[1] != DP
    if wide:  # the prep pass is 128 wide: it packs everything but Wf (and Wb), which are made below from the full d_pair columns
        wb_full, lnpw_full = wb, lnpw
        wb, lnpw = wb[:, :DP].contiguous(), lnpw[:DP].contiguous()
    _rows().ext().prep(
        wq,
        bq,
        wk,
        wv,
        wg,
        wo if wop is not None else None,
        wb,
        lnpw,
        wpack,
        bvec,
        wop,
        wf,
        qs,
        ws,
        wbo,
    )
    if wide:
        wf = (wb_full.float() * lnpw_full.float() * ws).to(bf)
        if wbo is not None:
            wbo = wb_full.to(bf, copy=True)  # an op output: never an alias of the parameter
    if gate_bias:  # sigmoid(g + c): the constant rides on the gate rows of the packed bias (q | k | v | g, W each)
        bvec[3 * W :] += gate_bias
    return wpack, bvec, wf, wop, wbo


#: the H100 batched inference's bias planes per launch, at most (bytes, bf16 [H, rows, L, L]): the batch runs in chunks of rows under it
#: (deterministic, the same values: each row's math is the per-row step's), so a long L does not hold B rows of planes at once
BATCH_BIAS_BYTES = 256 << 20


def _batch_chunk(B: int, H: int, L: int) -> int:
    """Batch rows per launch of the H100 batched inference (``BATCH_BIAS_BYTES``), at least 1."""
    return max(1, min(B, BATCH_BIAS_BYTES // (2 * H * L * L)))


def _inference_fake(single, pair, mask, leaves, eps_s, eps_p, gate_bias=0.0, extra=None, lp=0):
    return torch.empty_like(single)


@opaque(fake=_inference_fake, name="apb_b200_inference")
def inference(
    single: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps_s: float,
    eps_p: float,
    gate_bias: float = 0.0,
    extra: torch.Tensor | None = None,
    lp: int = 0,
) -> torch.Tensor:
    """single [B, L, 384], pair [B, L, L, 128] bf16, mask [B, L] bool or None, leaves as ``_leaves`` -> single + APB(single). ``extra``
    [B, H, L, L] (any strides): an additive logit term (``extra_pair_bias``), added to the bias planes in their exp2 units.

    B > 1 (FoldJet 5I28 confidence head: the samples' rows), the weight packs made once: H100 runs the batch rows together, one
    launch per stage (chunks of ``_batch_chunk`` rows; the per-row math of a B = 1 call, bit for bit), B200 the one-row step per batch
    row, each row written into its slice of the one output."""
    B, L = single.shape[:2]
    rows, dev = _rows(), single.device
    hopper = torch.cuda.get_device_capability(dev) == (9, 0)
    # H100, L % 128 != 0 (FoldJet OpenDDE n = 12: 2976 tokens): the core takes whole 128-token tiles, so it runs at ``lq`` (single rows zero, the
    # pad keys masked, the pad query rows dropped); the PAIR is read as it is -- the pair pass writes the real [L, L] block of the [lq, lq]
    # bias planes (``pair_bias_padded``), so the call costs the unpadded pair read, not a padded copy of the whole pair
    lq = -(-L // 128) * 128 if (L % 128 and hopper) else L
    params = leaves
    leaves = _capture.param_forms("apb_leaves", params, lambda: tuple(_kernel_leaves(params)))
    lnw, lnb = leaves[0], leaves[1]
    H, _, W, D = _geometry(leaves)
    if torch.cuda.get_device_capability(dev) == (10, 0):
        return _inference_b200(single, pair, mask, params, leaves, eps_s, eps_p, gate_bias, extra, lp)
    with torch.cuda.device(dev):
        wpack, bvec, wf, wop, _ = _inference_packs(params, leaves, gate_bias)
        wo = leaves[7] if wop is None else wop
        neg = -math.inf if hopper else NEG
        if lq != L:
            single = torch.nn.functional.pad(single, (0, 0, 0, lq - L))
            keys = torch.zeros((B, lq), dtype=torch.bool, device=dev)
            keys[:, :L] = True if mask is None else mask
            km = kernel_mask(keys, lq)
        else:
            # H100: each row's key blocks past its last attended key are skipped by the core (``kernels._key_len``; the same bits)
            km = kernel_mask(mask, L) if neg == -math.inf else None
        core = _cores.get((dev.index, H, D))
        if core is None:
            core = _cores[(dev.index, H, D)] = _inference_core(dev, H, D)
        out = torch.empty_like(single)  # each row: the residual seed ln_rows writes; to_out accumulates onto it
        if B > 1 and hopper:
            # H100, B > 1: the batch rows in one launch per stage (LN rows and the GEMMs over the chunk's rows, the pair pass and the
            # core with the batch in the grid) -- per-row results identical to the per-row step, 5 launches per chunk instead of 5 B
            cb = _batch_chunk(B, H, lq)
            for b0 in range(0, B, cb):
                b1 = min(B, b0 + cb)
                n = (b1 - b0) * lq
                x, y = single[b0:b1].view(n, D), out[b0:b1].view(n, D)
                xa = single.new_empty((n, D))
                rows.ext().ln_rows(x, lnw, lnb, xa, None, eps_s, y)
                qkvg = torch.addmm(bvec, xa, wpack.t())
                if lq != L:
                    bias = rows.pair_bias_padded(pair[b0:b1], wf, None if mask is None else mask[b0:b1], eps_p, neg, lq)
                else:
                    bias = rows.pair_bias_batch(pair[b0:b1], wf, None if mask is None else mask[b0:b1], eps_p, neg)  # [H, b1 - b0, L, L]
                if extra is not None:
                    bias[:, :, :L, :L].add_(extra[b0:b1].transpose(0, 1).to(bias.dtype), alpha=LOG2E)
                core.batch(qkvg, bias, None if km is None else km[b0:b1])
                y.addmm_(qkvg[:, :W], wo.t())
            return out if lq == L else out[:, :L].contiguous()
        xa = single.new_empty((lq, D))  # one row's LN(single), rewritten row after row (stream order)
        for b in range(B):
            x, y = single[b], out[b]
            rows.ext().ln_rows(x, lnw, lnb, xa, None, eps_s, y)
            qkvg = torch.addmm(bvec, xa, wpack.t())
            if lq != L:
                bias = rows.pair_bias_padded(pair[b : b + 1], wf, None if mask is None else mask[b : b + 1], eps_p, neg, lq)[:, 0]
            else:
                bias = rows.pair_bias(pair[b].view(L * L, wf.shape[1]), wf, _row(mask, b), L, eps_p, neg)
            if extra is not None:  # a masked key stays at -inf (H100) / NEG + extra (B200: finite, still far below every real logit)
                bias[:, :L, :L].add_(extra[b].to(bias.dtype), alpha=LOG2E)
            if km is None:  # B200 (its masked keys are NEG in the bias planes: no key-length skip) or no mask
                core(qkvg, bias)
            else:
                core(qkvg, bias, km[b : b + 1])
            y.addmm_(qkvg[:, :W], wo.t())
        return out if lq == L else out[:, :L].contiguous()


_SIDE: dict = {}
#: B200 step stages on side streams (FOLDJET_ENGINE_APB_B200_STREAMS=0: every stage on the caller's stream, in program order). On by default:
#: measured under the speed tokens' graph timing (2026-11-17, same build, streams on / off): master eval 16 x 128 L 128 24.5 / 26.5 us,
#: L 512 46.9 / 49.0 us; master train 16 x 512 L 128 91.9 / 100.3 us, L 512 307 / 326 us.
_STREAMS = os.environ.get("FOLDJET_ENGINE_APB_B200_STREAMS", "1") != "0"
#: B200: the bf16 pair -> bias pass on tcgen05 (sm100/pair_fwd.cu) instead of the shared mma.sync pass
_PAIR_TC = os.environ.get("FOLDJET_ENGINE_APB_B200_PAIR_TC")
#: B200 inference: the weight packs through ``_capture.param_forms(capture_reads_eager=True)`` (grad off): made once per parameter version,
#: read by a capture as its eager warm-up made them. FOLDJET_ENGINE_APB_B200_EAGER_PACKS=0: packed inside every capture (one launch per replay).
_EAGER_PACKS = os.environ.get("FOLDJET_ENGINE_APB_B200_EAGER_PACKS", "1") != "0"
#: B200 bf16 inference, d_pair 128 at L 384-640: SMs the pair pass leaves free (``sm_reserve``; FOLDJET_ENGINE_APB_B200_FWD_RESERVE, 0: none).
#: The pass fills every SM, so the q | k | v | g GEMM (ready when LayerNorm(single) ends, a few microseconds into the step) sat queued
#: until the pass ended and ran as its tail; with 8 SMs' worth of blocks left out it starts under the pass.
_FWD_RESERVE = int(os.environ.get("FOLDJET_ENGINE_APB_B200_FWD_RESERVE", "8"))
#: B200 bf16 inference: the pair pass starts when LayerNorm(single) has ended (FOLDJET_ENGINE_APB_B200_PAIR_AFTER_LN=0: with the step). The pass
#: takes every SM at once; LayerNorm launched beside it ran starved (2.0 -> 2.8-5.6 us) and the GEMM after it later still. With LayerNorm
#: alone first (~2 us), the GEMM and the pass start together. Measured (2026-11-17, master eval, graph timing, two A / B rounds, with the
#: step -> after LN): 16 x 128 L 128 / 256 / 768 27.1 / 34.7 / 67.3 -> 24.4 / 32.6 / 63.5-65.0 us, 12 x 128 L 512 47.0 -> 43.1, 8 x 384 L 128 /
#: 256 / 1024 30.5 / 42.8 / 170.4 -> 26.6 / 39.0 / 165.9, 16 x 512 L 128 / 384 32.6 / 66.0 -> 30.5 / 63.4; no case slower. (The fp32 FMA pass
#: loses the same way round: 8 x 384 L 512 147.9 -> 169.8 us, so fp32 keeps its order.)
_PAIR_AFTER_LN = os.environ.get("FOLDJET_ENGINE_APB_B200_PAIR_AFTER_LN", "1") != "0"
#: B200 bf16 inference (v20, 2026-11-17): the core after the q | k | v | g GEMM (and its key-split combine after it) launched as programmatic
#: dependents (attn_inf -DPDLW=1, ksplit_combine through pdl_go). FOLDJET_ENGINE_APB_B200_INF_PDL=0: plain launches.
_INF_PDL = os.environ.get("FOLDJET_ENGINE_APB_B200_INF_PDL", "1") != "0"


def _fwd_reserve(dp: int, L: int) -> int:
    """Measured (2026-11-17, master eval, graph timing, reserve 0 / 8 / 16 / 32): 16 x 128 L 384 / 512 / 640 40.5 / 46.9 / 57.2 -> 39.3 /
    44.9 / 55.3 us at 8 (16: 38.8 / 45.1 / 57.3; 32: 39.4 / 44.9 / 57.3), 12 x 128 38.7 / 44.9 / 55.3 -> 36.7 / 42.9 / 55.2; level at
    L 768 (65.4 -> 65.1), slower from L 1024 at 16 / 32 (89.9 -> 92.0 / 98.0); nothing at L 128 / 256 (24 / 48 / 74 SMs: the same medians) and
    nothing in the training forward (its GEMM is not the tail there). The fp32 pass (8 x 384) only loses: 145.8 -> 145.8 / 147.9 / 158.2."""
    return _FWD_RESERVE if _STREAMS and dp == DP and 384 <= L <= 640 else 0


#: B200 bf16 inference, d_pair 128 (v21, 2026-11-17): the pair pass as ``pair_bias_fwd2`` (apb_pair_fwd_rows_sm100.cuh: the shared pass with
#: each warp's 16 rows loaded by its own lane / mbarrier) -- "stages:blocks per SM" (0: the occupancy); None: the shared pass.
#: FOLDJET_ENGINE_APB_B200_PF2 forces (0: off).
_PF2 = os.environ.get("FOLDJET_ENGINE_APB_B200_PF2", "2:0")


def _pf2(dp: int, L: int, heads: int, train: bool = False) -> tuple[int, int] | None:
    """Measured (2026-11-17, master eval step, shared -> pair_bias_fwd2 2:0, the same bits): 24 x 128 L 384 / 512 / 768 41.0 / 51.2 / 73.7
    -> 40.9 / 49.2 / 71.6 us, 12 x 128 L 512 / 768 45.1 / 62.8 -> 43.0 / 61.4, 16 x 128 (d 512) L 640 / 768 57.2 / 65.5 -> 55.3 / 63.5, the
    rest level; training L 768 12 x 128 194.4 -> 190.5, 24 x 128 254.0 -> 251.9, L 128 +-2 us (training: from L 384)."""
    if _PF2 is None or _PF2 == "0" or dp != DP:
        return None
    s, b = _PF2.split(":")
    if s == "2" and L >= _PF2_EVF_MIN_L and not train:
        s = "12"  # v22: the same pass with the pair loads under an L2 evict_first policy (the same bits)
    return int(s), int(b)


#: v22 (2026-11-17): from this L the pair tiles of ``pair_bias_fwd2`` load with an L2 evict_first policy (stages code 12). After the measurer's
#: flush (a 256 MB zero fill, the L2 full of dirty lines) every line the pass brings in first writes a dirty one back; lines marked
#: evict_first are chosen as victims before them, so the read-once stream mostly evicts itself. Standalone (16 x 128 L 640 / 1024, after the
#: write flush): 30.8 / 61.5 -> 28.7 / 57.4 us (the loads alone 28.7 / 58.4 -> 24.6 / 50.1); after a read flush (clean L2) every form reads
#: 28.7 / 54-55 us; 24 heads level (the mma / stores bound it). Step A / B (master eval, 2:0 -> 12:0, trimmed means): r41 L 768 / 1024 63.6 /
#: 90.5 -> 61.4 / 86.7, r146 L 768 61.4 -> 57.3, r147 L 512 / 768 50.7 / 69.6 -> 49.2 / 67.7, L 384 level to +1 us (r41 / r147): from L 512.
#: Inference only: the training forward (A / B L 512 level, L 768 -1 to -9 us) read 3 more cases above 0.667 in the measurer (r146 L 640 / 768,
#: r222 L 768, +5 to +9 us), so it keeps the plain loads.
_PF2_EVF_MIN_L = int(os.environ.get("FOLDJET_ENGINE_APB_B200_PF2_EVF_MIN_L", "512"))


def _pair_tc(dp: int, L: int, heads: int, train: bool = False) -> bool:
    """Whether the bf16 pair -> bias pass runs on tcgen05 (FOLDJET_ENGINE_APB_B200_PAIR_TC=0 / 1 forces). Measured (2026-11-17, master eval step,
    graph timing, the shared pass -> tcgen05): d_pair 256 L 512 / 768 59.3 / 94.1 -> 59.9 / 94.7 us, L 1024 / 1536 / 2048 139.7 / 274.3 / 458.6
    -> 135.6 / 252.4 / 410.5; d_pair 384 (8 heads) L 384 / 512 / 640 / 768 55.2 / 72.1 / 98.2 / 126.8 -> 55.9 / 71.7 / 92.7 / 113.2, (16 heads)
    L 256 / 384 / 512 42.9 / 59.4 / 79.8 -> 43.5 / 57.8 / 72.2; d_pair 512 L 256 / 384 / 512 / 768 45.0 / 69.5 / 88.0 / 159.7 -> 45.6 / 67.6 /
    84.4 / 141.8. The training forward (``train``: the pass also writes the row statistics and the unmasked bias) keeps d_pair 256 from
    L 512 (master train step 205.7 -> 202.8 us).

    Re-measured (2026-11-17, round 3 part 4, Wf read before the first tiles' loads, shared -> tcgen05): eval d_pair 128 never wins
    (16 x 128 L 128 / 512 / 768 24.5 / 44.8 / 65.5 -> 26.4 / 46.9 / 68.1 us); d_pair 256 L 512 / 768 / 1024 59.4 / 95.4 / 141.3 -> 59.2 / 94.2 / 134.9;
    d_pair 384 x 16 heads L 128 / 256 / 384 28.6 / 42.8 / 61.2 -> 30.5 / 40.8 / 55.8, x 8 heads L 256 / 384 38.9 / 57.2 -> 40.7 / 57.0; d_pair
    512 L 128 / 256 / 384 30.8 / 46.9 / 67.5 -> 30.6 / 42.8 / 61.4. Training (wide rows): d_pair 256 L 128 / 256 / 384 79.8 / 108.4 / 151.3 ->
    81.8 / 106.3 / 147.5; 384 x 16 heads L 256 139.3 -> 137.0, x 8 heads L 256 / 384 108.3 / 147.8 -> 108.4 / 147.3; 512 L 128 / 256 96.1 / 143.4
    -> 96.1 / 139.1."""
    if _PAIR_TC is not None:
        return _PAIR_TC != "0"
    if dp == 512:
        return True
    if dp == 384:
        return (256 if heads >= 16 else 384) <= L
    return dp == 256 and (256 if train else 768) <= L


#: B200 training backward: SMs the pair backward leaves free (``apb_rows_sm100.cu`` sm_reserve). It fills every SM for most of the
#: backward, so the main stream's row passes / GEMMs launched beside it (qkv_bwd, dW, dxa, ln_bwd) sat queued until it ended.
_RESERVE = int(os.environ.get("FOLDJET_ENGINE_APB_B200_SM_RESERVE", "8"))
#: B200 training backward: dxa = dqkvg W (the head of the tail dxa -> ln_bwd -> finalize) is issued before dW q|k|v|g (a side stream).
#: Measured (2026-11-17, master train, graph timing, after / before): 12 x 128 L 128 / 512 71.5 / 137.0 -> 69.4 / 133.1 us, 16 x 128 L 128 /
#: 512 / 768 75.6 / 143.1 / 227.0 -> 71.6 / 139.1 / 218.9, 16 x 256 L 384 / 512 153.2 / 204.6 -> 149.4 / 200.7. FOLDJET_ENGINE_APB_B200_DXA_FIRST=0: after.
_DXA_FIRST = os.environ.get("FOLDJET_ENGINE_APB_B200_DXA_FIRST", "1") != "0"
#: B200 training backward, d_pair 128: the pair backward on tcgen05 (sm100 ``pair_bwd.cu``) instead of the mma.sync pass (L % 128 == 0).
_PAIR_BWD_TC = os.environ.get("FOLDJET_ENGINE_APB_B200_PAIR_BWD_TC", "1") != "0"
#: v23 (2026-11-18): at L 128 the backward dQ + dbias pass as ``dqb_small`` (apb_rows_sm100.cu: a CTA per head and 32 query rows, mma.sync, no
#: dQ combine: NH L / 32 CTAs instead of attn_dqb's NH work items). FOLDJET_ENGINE_APB_B200_DQB_SMALL=0: attn_dqb.
_DQB_SMALL = os.environ.get("FOLDJET_ENGINE_APB_B200_DQB_SMALL", "1") != "0"


def _dqb_small(L: int) -> bool:
    return _DQB_SMALL and L in (128, 256)


#: v27 (2026-11-18): the training attention backward at L 128 / 256 in ONE launch (``small_core_bwd``: gate_bwd + attn_dkv + dqb + qkv_bwd;
#: a CTA per head and SBWD query / key rows). 0: the separate stages. FOLDJET_ENGINE_APB_B200_SBWD overrides.
_SBWD = int(os.environ.get("FOLDJET_ENGINE_APB_B200_SBWD", "32"))


#: v27: finalize_red split into its three parts, each launched on the stream that makes its inputs (dWf part after the pair backward on
#: side 1, the copies after the dW GEMMs on side 0, the row columns after ln_bwd on main). FOLDJET_ENGINE_APB_B200_FSPLIT=0: one launch at the end.
#: Measured (master train, A / B, the same bits): 16 heads L 128 -0.5 .. -1.9 us, L 256 -0.6 .. -3.0, L 384 -0.2 .. -1.6, L 768 level;
#: 12 heads L 128 +1.5, L 768 +4.1: 16 heads up to L 512 only.
_FSPLIT = os.environ.get("FOLDJET_ENGINE_APB_B200_FSPLIT", "1") != "0"
#: v29: the parts form at 12 heads up to L _FS12 (0: never) and at 24 heads at the lengths _FS24_L (16 heads: L <= 512 as above);
#: _FR_MAXH: the most heads finalize_red serves. A / B (master train r147 24 x 128, one launch -> parts): L 256 79.9 -> 76.1 us (with the
#: pair backward reserve 16), L 384 114.7 -> 111.5, L 512 149.7 -> 148.5 (measurer: 149.5 -> 153.7, not taken), L 640 201.4 -> 197.6, L 768
#: 248.7 -> 247.0, L 128 53.3 -> 54.5 (not taken); r146 12 x 128 L 256 69.8 -> 70.9 (not at 12 heads).
_FS12 = 0
_FS24_L = (256, 384, 640, 768)
_FR_MAXH = 16


def _fsplit(H: int, L: int) -> bool:
    return _FSPLIT and _STREAMS and (L <= 512 if H == 16 else L <= _FS12 if H == 12 else H == 24 and L in _FS24_L)


def _sbwd(L: int, heads: int) -> int:
    # L 256: 24 heads (192 CTAs) +1.5 .. +2.1 us, 8 heads (head width 48) +4 us: 12 / 16 heads only
    return _SBWD if _SBWD > 0 and (L == 128 or (L == 256 and 12 <= heads <= 16)) else 0


def _rsqd(heads: int, d: int) -> float:
    """1 / sqrt(real head dim)."""
    return float(d // heads) ** -0.5


_PBTC_RESERVE = os.environ.get("FOLDJET_ENGINE_APB_B200_PBTC_RESERVE")
#: B200 training, d_pair 128 tail (2026-11-17, part 5; measured per head count, ``_apb/ab.py`` interleaved graph replays):
#: acc_reduce + finalize in one launch (``finalize_red``) at <= 16 heads (24 heads: its 24-head column sums per thread lost 2-10 us);
#: the gate backward with every lane busy (``gate_bwd2``) at 12 heads (24: no gain); bias^T made in the forward beside the core
#: (``bias_transpose64``, 64-block persistent grid) at <= 16 heads or L <= 256 (24 heads L 768: 10 us slower than the backward's own
#: transpose beside dWo). FOLDJET_ENGINE_APB_B200_P5=0: the part 4 forms everywhere.
_P5 = os.environ.get("FOLDJET_ENGINE_APB_B200_P5", "1") != "0"
_TR_BLOCKS = 64


def _fwd_transpose(heads: int, L: int) -> bool:
    """The training forward makes bias^T beside its core (the backward's attn_dkv reads it) instead of the backward."""
    return _P5 and L % 64 == 0 and (heads <= 16 or L <= 256) and not _BIASNT_ALL and not _sbwd(L, heads)


#: B200 training backward (2026-11-17, part 6): attn_dkv reads the bias as stored (``-DBIASNT=1``) where neither the forward nor the
#: backward should transpose it. FOLDJET_ENGINE_APB_B200_BIASNT=0: never (the backward transposes); =all: everywhere (no forward transpose).
_BIASNT = os.environ.get("FOLDJET_ENGINE_APB_B200_BIASNT", "1")
_BIASNT_ALL = _BIASNT == "all"


def _bias_nt(heads: int, L: int) -> bool:
    """attn_dkv reads the bias as stored (called where the forward made no bias^T)."""
    return _BIASNT != "0"


def _pbtc_reserve(heads: int, L: int) -> int:
    """SMs the tcgen05 pair backward leaves to the row passes / GEMMs beside it (qkv_bwd, dxa, dW q|k|v|g, ln_bwd): it is bound by the
    card's memory rate rather than by its SM count. Measured (2026-11-17, master train, graph timing, reserve 8 / 32 / 56): 16 x 128 L 384 /
    512 / 768 120.5 / 139.2 / 223.2 -> 110.4 / 135.5 / 227.1 / (56) 114.4 / 145.2 / 244.9 us; 12 x 128 100.3 / 130.8 / 198.4 -> 98.1 /
    131.0 / 206.8; 24 x 128 118.6 / 155.5 / 266.2 -> 118.7 / 163.6 / 281.9. FOLDJET_ENGINE_APB_B200_PBTC_RESERVE forces."""
    if _PBTC_RESERVE is not None:
        return int(_PBTC_RESERVE)
    # v20 (2026-11-17, master train L 128, reserve 32 -> 64 / 96, trimmed means): 12 x 128 61.5 -> 61.4 / 59.4 us, 16 x 128 (d 512) 71.7 ->
    # 69.6 / 69.6, 16 x 128 65.5 -> 63.5 / 63.4, 24 x 128 70.1 -> - / 67.6; at L 256 both lose 2-4 us
    if L <= 128:
        # v27 (with small_core_bwd, master train, 96 -> 80): r147 -2.1, r148 -1.8 us, r146 / r222 level
        return 80 if _sbwd(L, heads) else 96
    if L == 256 and _sbwd(L, heads):
        return 48  # v27 (with small_core_bwd, 32 -> 48): r146 -3.1, r147 -2.5, r148 -2.2, r222 -6.0 us; 56 / 64 / 72 less
    if L == 256 and heads == 24 and _fsplit(heads, L):
        return 16  # v29 (24 heads, finalize parts; r147 reserve 8 / 16 / 24 / 32 / 40: 79.9 -> 76.1 / 76.7 / 76.7 / 76.6 us)
    return 32 if heads <= 16 and L <= 512 else 8


#: B200 bf16 training (v20, 2026-11-17): the main stream's own kernels launched as programmatic dependents of the kernel before them (PDL:
#: attn_fwd2 / attn_dkv -DPDLW=1, the row passes gate_rows / gate_bwd / qkv_bwd / ln_bwd / finalize_red through the extension's pdl_go; each
#: opens with griddepcontrol.wait, so the values are the same). Measured (master train, graph replays interleaved, off -> on): L 128 12 x 128
#: 63.5 -> 61.4 us, 16 x 256 77.8 -> 73.7, 16 x 384 90.1 -> 86.0; L 256 24 x 128 88.1 -> 84.0, 16 x 256 104.4 -> 98.4; L 512 / 768 level to
#: 2-8 us slower (12 x 128: 125.0 -> 127.0, 190.5 -> 198.7): L <= 256 only. FOLDJET_ENGINE_APB_B200_PDL=0 / 1 forces off / on.
_PDL = os.environ.get("FOLDJET_ENGINE_APB_B200_PDL")
_PDL_MAX = 256


def _pdl_on(L: int) -> bool:
    if _PDL is not None:
        return _PDL != "0"
    return L <= _PDL_MAX


def _pdl_call(E, pdl: bool, fn, *args) -> None:
    """fn(*args) with the extension's PDL launches on when ``pdl``, then the previous setting."""
    if not pdl:
        fn(*args)
        return
    was = E.pdl(1)
    try:
        fn(*args)
    finally:
        E.pdl(was)


def _bwd_reserve(dp: int, L: int) -> int:
    """Measured (2026-11-17, master training, graph timing, reserve 0 / 8 / 20): d_pair 128 x 12 heads L 256 / 512 / 768 92.1 / 141.1 /
    212.9 -> 86.0 / 134.8 / 209.0 us at 8 (20: 85.7 / 134.9 / 215.0); a wide pair's backward is the whole step from L 512 on and loses what
    it gives up (d_pair 256 L 512 / 768 204.8 / 362.5 -> 207.2 / 370.5 us), so it reserves only up to L 256 (112.4 -> 110.5 us)."""
    return _RESERVE if _STREAMS and (dp == DP or L <= 256) else 0


def _side_stream(device: torch.device, k: int = 0) -> torch.cuda.Stream:
    """The B200 step's side streams (``k`` 0 / 1, per device): the weight pack and the pair pass beside the single-track row / GEMM stages;
    in the backward the stages that do not depend on each other (the bias transpose and dWo; attn_dqb and the pair backward).
    With ``_STREAMS`` off: the current stream (every stage in program order; ``_join`` / ``_rec`` / ``_after`` do nothing)."""
    if not _STREAMS:
        return torch.cuda.current_stream(device)
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if (idx, k) not in _SIDE:
        _SIDE[(idx, k)] = torch.cuda.Stream(device=idx)
    return _SIDE[(idx, k)]


def _join(dst: torch.cuda.Stream, src: torch.cuda.Stream) -> None:
    if _STREAMS:
        dst.wait_stream(src)


def _rec(event: torch.cuda.Event, stream: torch.cuda.Stream) -> None:
    if _STREAMS:
        event.record(stream)


def _after(stream: torch.cuda.Stream, event: torch.cuda.Event) -> None:
    if _STREAMS:
        stream.wait_event(event)


def _inference_b200(single, pair, mask, params, leaves, eps_s, eps_p, gate_bias, extra, lp=None):
    """``inference`` on B200 (2026-11-17): the same kernels and values, on two streams. The side stream makes the weight pack (a capture
    records it; eager calls hit the cache) and then each batch row's pair pass (the bias planes need only Wf and the pair); the main stream
    runs LN(single), the q | k | v | g GEMM once the pack is there, the core once the row's planes are there, and to_out. At small L
    every stage is a few microseconds of latency, so the pair pass (the largest) runs under LN + GEMM instead of after them. The side
    stream makes row b + 1's planes only after the core of row b - 1 has read its own (one row ahead: at most two rows of planes)."""
    B, L0 = single.shape[:2]
    L = lp or L0  # the core's length; the pair pass runs at L0
    if L != L0:
        single = torch.nn.functional.pad(single, (0, 0, 0, L - L0))
    E = _rows().ext()
    lnw, lnb = leaves[0], leaves[1]
    H, _, W, D = _geometry(leaves)
    dev = single.device
    if L == L0 and _EAGER_PACKS and not torch.is_grad_enabled() and _front(leaves[9].shape[1], L0, H, D):
        return _inference_front(single, pair, mask, params, leaves, eps_s, eps_p, gate_bias, extra)
    with torch.cuda.device(dev):
        main, side = torch.cuda.current_stream(dev), _side_stream(dev)
        _join(side, main)  # the inputs (and every buffer the side stream may reuse) are ready
        out = torch.empty_like(single)  # each row: the residual seed ln_rows writes; to_out accumulates onto it
        xa = single.new_empty((L, D))
        wf_dp = leaves[9].shape[1]
        ln0: list = [single[0], lnw, lnb, xa, None, eps_s, out[0]]
        # the tcgen05 pair pass forms Wf itself (it starts with the step, on the side stream); the pack + LN launch then leads the main stream
        raw = tc = _pair_tc(wf_dp, L0, H)
        pack_stream = main if raw else side
        with torch.cuda.stream(pack_stream):
            if _EAGER_PACKS and not torch.is_grad_enabled():
                # grad off: a CUDA-graph capture reads the pack its eager warm-up made (``capture_reads_eager``): no pack launch per replay,
                # and neither the q | k | v | g GEMM nor the pair pass waits for one (LayerNorm(single) is then its own launch)
                wpack, bvec, wf, wop, _ = _capture.param_forms(
                    ("apb_b200_inference", float(gate_bias)),
                    params,
                    lambda: _prep(leaves, LOG2E / math.sqrt(_geometry(leaves)[1]), LOG2E, gate_bias),
                    capture_reads_eager=True,
                )
                raw = False  # the tcgen05 pass reads the cached Wf
            else:
                wpack, bvec, wf, wop, _ = _inference_packs(params, leaves, gate_bias, ln0)
        assert wop is not None  # B200: prep_dp always packs Wo
        packed = torch.cuda.Event()
        _rec(packed, pack_stream)
        prev_core: torch.cuda.Event | None = None
        for b in range(B):
            if prev_core is not None:
                _after(side, prev_core)
            x, y = single[b], out[b]
            ln_here = b > 0 or len(ln0) == 7  # row 0: LayerNorm(single) went out with the pack (prep_ln), when the pack was made here
            if ln_here:
                E.ln_rows(x, lnw, lnb, xa, None, eps_s, y)
                if _PAIR_AFTER_LN and _STREAMS:  # the pair pass starts beside the GEMM, LayerNorm(single) alone before them
                    ln_done = torch.cuda.Event()
                    _rec(ln_done, main)
                    _after(side, ln_done)
            with torch.cuda.stream(side):
                p2 = pair[b].view(L0 * L0, wf.shape[1])
                if raw:  # Wf from to_bias.weight / ln_pair.weight inside the pass
                    bias: torch.Tensor = _sm100().apb_pair_fwd(p2, leaves[9], leaves[8], LOG2E, _row(mask, b), L0, eps_p, NEG)
                elif tc:
                    bias = _sm100().apb_pair_fwd(p2, wf, None, 1.0, _row(mask, b), L0, eps_p, NEG)
                else:
                    was = E.sm_reserve(_fwd_reserve(wf_dp, L0))
                    pf2 = _pf2(wf_dp, L0, H)
                    if pf2 is not None:  # v21: the B200 form of the d_pair 128 pass (per-warp tile loads)
                        made = torch.empty(H, L0, L0, device=dev, dtype=torch.bfloat16)
                        E.pair_bias_fwd2(p2, wf, _row(mask, b), made, L0, eps_p, NEG, *pf2)
                    else:
                        made = _rows().pair_bias(p2, wf, _row(mask, b), L0, eps_p, NEG)
                    E.sm_reserve(was)
                    assert isinstance(made, torch.Tensor)
                    bias = made
                if extra is not None or L != L0:  # masked keys: NEG + extra (finite, still far below every real logit)
                    bias = _planes_fix(E, bias, None if extra is None else extra[b], L0, L, False)
            planes = torch.cuda.Event()
            _rec(planes, side)
            if b == 0:
                _after(main, packed)
            qkvg = torch.addmm(bvec, xa, wpack.t())
            _after(main, planes)
            bias.record_stream(main)
            # v27: small_core on this path at L 128 / 256 (d_pair 256 / 384 / 512; an extra bias is folded into the planes already)
            nf = _small_nf(L, H) if L == L0 and (extra is None or _SMALL_NFX) else 0
            if nf == 2:  # core + to_out in one launch (y = seed + o Wo^T)
                E.small_core(qkvg, bias, wop, y, L, _INF_PDL, _SCQB)
            elif nf == 1 or (extra is None and L == L0 and _small_long(L, H)):  # v26: the mma.sync core (128-key chunks from L 384)
                E.small_core(qkvg, bias, wop, y, L, _INF_PDL, _sc_mode(L, H))
            else:
                _cores_b200(dev, H, D)(qkvg, bias, pdl=_INF_PDL)  # v20: the core (and its key-split combine) as programmatic dependents
            if B > 1:
                prev_core = torch.cuda.Event()
                _rec(prev_core, main)
            if nf != 2:
                y.addmm_(qkvg[:, :W], wop.t())
        for t in (wpack, bvec, wop):
            t.record_stream(main)
        for t in (pair, wf) + (() if mask is None else (mask,)) + (() if extra is None else (extra,)):
            t.record_stream(side)
        return out if L == L0 else out[:, :L0].contiguous()


#: B200 bf16 inference (2026-11-17, part 6): LayerNorm(single), the q | k | v | g GEMM and the pair pass in ONE launch (``inf_front``,
#: apb_rows_sm100.cu), then the core and to_out: 3 launches per batch row instead of 5. FOLDJET_ENGINE_APB_B200_FRONT=0 / 1 forces off / on (where
#: the kernel serves: d_pair 128 / 256, L % 128 == 0); FOLDJET_ENGINE_APB_B200_FRONT_PB: its pair blocks (0: the kernel's rule).
_FRONT = os.environ.get("FOLDJET_ENGINE_APB_B200_FRONT")
_FRONT_PB = os.environ.get("FOLDJET_ENGINE_APB_B200_FRONT_PB")


def _front_pb(L: int, D: int, dp: int, heads: int) -> int:
    """inf_front's ``pblocks``. v24 (2026-11-18): at L 256 (d_pair 128, D 384) the overlapped roles in one wave (``inf_fronto_k``, 6 pair
    tiles per pair-only block): standalone after an L2 flush (CUPTI) 12 x 128 12.8 -> 10.6 us, 16 x 128 14.0 -> 13.3, 24 x 128 14.4 -> 11.7;
    D 512 (16 x 128 d 512) 15.7 -> 16.5 and L 128 6.2 -> 7.7: the default split there. Step (master eval L 256, interleaved replays): 12 x 128
    28.5 -> 26.7, 24 x 128 30.6 -> 27.7, 16 x 128 28.7 -> 29.5 us (not at 16 heads). L 384 (12 / 24 heads, D 384): KA 6 / 8 level,
    10 / 13 / 16 1.5-7 us slower."""
    if _FRONT_PB is not None:
        return int(_FRONT_PB)
    return -106 if (L in (256, 384) and D == 384 and dp == DP and heads in (12, 24)) else 0


_FRONT_MAX = int(os.environ.get("FOLDJET_ENGINE_APB_B200_FRONT_MAX", "256"))
#: the core after the front launched as a programmatic dependent (attn_inf -DPDLW=1). FOLDJET_ENGINE_APB_B200_FRONT_PDL=0: a plain launch.
_FRONT_PDL = os.environ.get("FOLDJET_ENGINE_APB_B200_FRONT_PDL", "1") != "0"


def _front(dp: int, L: int, heads: int, D: int = 384) -> bool:
    """Whether the bf16 inference runs the fused front (``inf_front``)."""
    if L % 128 or dp not in (DP, 2 * DP) or (dp != DP and heads not in (8, 16)) or _pair_tc(dp, L, heads):
        return False
    if _FRONT is not None:
        return _FRONT != "0"
    # v20 (2026-11-17): d_pair 256 at L 128 too (master eval, 400 interleaved replays, trimmed means: 16 x 256 + gate constant 28.67 ->
    # 26.62 us, 16 x 256 level 26.63 -> 26.62; at L 256 it loses 2-4 us)
    # v24 (2026-11-18): with the overlapped roles (``_front_pb``) also at L 384 for 12 / 24 heads at D 384 (master eval, interleaved replays:
    # 12 x 128 36.9 -> 35.2, 24 x 128 40.2 -> 37.0 us; 16 heads level, D 512 +4, L 512 / 640 level to +6: not there)
    return (dp == DP and (L <= _FRONT_MAX or (L == 384 and D == 384 and heads in (12, 24)))) or (dp == 2 * DP and L <= 128)


def _inference_front(single, pair, mask, params, leaves, eps_s, eps_p, gate_bias, extra):
    """``_inference_b200`` with the fused front: per batch row ONE launch for LayerNorm(single) (+ the residual seed), q | k | v | g = LN(x)
    W^T + b and the bias planes, then the core (sigmoid(g) o over q) and to_out onto the seed, all on the caller's stream."""
    B, L = single.shape[:2]
    E = _rows().ext()
    lnw, lnb = leaves[0], leaves[1]
    H, _, W, D = _geometry(leaves)
    dev = single.device
    dp = leaves[9].shape[1]
    with torch.cuda.device(dev):
        wpack, bvec, wf, wop, _ = _capture.param_forms(
            ("apb_b200_inference", float(gate_bias)),
            params,
            lambda: _prep(leaves, LOG2E / math.sqrt(_geometry(leaves)[1]), LOG2E, gate_bias),
            capture_reads_eager=True,
        )
        assert wop is not None
        out = torch.empty_like(single)
        core = _cores_b200(dev, H, D)
        for b in range(B):
            qkvg = single.new_empty((L, 4 * W))
            bias = single.new_empty((H, L, L))
            E.inf_front(
                pair[b].view(L * L, dp),
                wf,
                _row(mask, b),
                bias,
                L,
                eps_p,
                NEG,
                single[b],
                lnw,
                lnb,
                wpack,
                bvec,
                out[b],
                qkvg,
                eps_s,
                _front_pb(L, D, dp, H),
            )
            if extra is None and _small_core(L, H, dp):
                fused = _small_fused(L, dp)
                E.small_core(qkvg, bias, wop, out[b], L, _FRONT_PDL, _SCQB if fused else _sc_mode(L, H))
                if not fused:
                    out[b].addmm_(qkvg[:, :W], wop.t())
                continue
            core(qkvg, bias, pdl=_FRONT_PDL and extra is None)  # the core's prologue under the front's tail
            out[b].addmm_(qkvg[:, :W], wop.t())
        return out


#: v23 (2026-11-18): at L 128 / 256 the core as ``small_core`` (apb_rows_sm100.cu: a CTA per head and query block, mma.sync, launched as
#: the front's programmatic dependent) instead of attn_inf (tcgen05, 16 work items). 1: the rule below; 2: to_out fused wherever small_core
#: runs; 0: attn_inf + cuBLAS to_out.
#: v26 (2026-11-18): the fused form (core + to_out in one launch: a cluster of the heads of a 16-query block exchanges the gated o by
#: st.async into each peer's shared memory; 24 heads: 2 heads per CTA, a cluster of 12) at d_pair 128, L 128 (``_small_fused``).
_SMALL = int(os.environ.get("FOLDJET_ENGINE_APB_B200_SMALL", "1"))
#: v26: queries per CTA of the fused form (16 / 32 / 64): the o exchange per CTA is (H - heads per CTA) slots of that many rows
_SCQB = int(os.environ.get("FOLDJET_ENGINE_APB_B200_SCQB", "16"))


def _small_core(L: int, heads: int, dp: int) -> bool:
    # measured (master eval, interleaved graph replays, attn_inf -> small_core then cuBLAS to_out): L 256 16 x 128 30.72 -> 29.49 us, 12 x 128
    # 28.69 -> 27.25, 24 x 128 30.60 -> 29.97, 16 x 128 (d 512) 32.37 -> 30.73; d_pair 256 L 128 26.63 -> 24.59 (r42 / r43); d_pair 128 L 128
    # core-only level to one timer step slower (12 x 128 20.47 -> 22.53); v26: d_pair 128 L 128 fused (``_small_fused``)
    return _SMALL > 0 and (L == 256 or (L == 128 and (dp != DP or _small_fused(L, dp))) or _small_long(L, heads))


def _small_fused(L: int, dp: int) -> bool:
    # v26 (master eval, interleaved graph replays, L2 flushed, trimmed means; attn_inf + cuBLAS to_out -> fused, 16 queries per CTA):
    # L 128 12 x 128 20.5 -> 18.4 / 18.4 -> 16.4, 16 x 128 (d 512) 22.5 -> 18.5 / 19.7, 24 x 128 20.5 -> 19.5, 16 x 128 level (18.5);
    # 32 queries per CTA: about half of that; 64 (v23's form, 64 KB of o per CTA): level to +1.2. L 256: +2 to +10 us (the core-only form
    # stays); d_pair 256 L 128 (r42 / r43): level.
    return _SMALL == 2 or (_SMALL == 1 and L == 128 and dp == DP)


#: v26 (2026-11-18): at L 384 / 512 / 640 the core as ``small_core`` in 128-key chunks (running maximum), no key split and no combine launch,
#: where its CTAs fit in one wave (``_small_long``). FOLDJET_ENGINE_APB_B200_SMALL_LONG: 1 the rule, 0 off (attn_inf + key split + combine), 32 / 64
#: forces that many queries per CTA wherever the kernel serves (head width <= 32). FOLDJET_ENGINE_APB_B200_SMALL_QB: queries per CTA of the
#: core-only form at L 128 / 256 (0: the rule, 32 at L 256 up to 16 heads, else 64).
_SMALL_LONG = int(os.environ.get("FOLDJET_ENGINE_APB_B200_SMALL_LONG", "1"))
_SMALL_QB = int(os.environ.get("FOLDJET_ENGINE_APB_B200_SMALL_QB", "0"))  # 0: the rule in _sc_mode


#: v27 (2026-11-18): the non-front inference path (d_pair 256 / 384 / 512, where inf_front does not serve) at L 128 / 256: 2 the fused
#: small_core (core + to_out) at L 128, 1 the core-only small_core at L 256, 0 attn_inf + cuBLAS to_out. FOLDJET_ENGINE_APB_B200_SMALL_NF overrides.
_SMALL_NF = os.environ.get("FOLDJET_ENGINE_APB_B200_SMALL_NF")
_SMALL_NFX = os.environ.get("FOLDJET_ENGINE_APB_B200_SMALL_NFX", "1") != "0"  # also with an extra bias (folded into the planes by planes_fix)


def _small_nf(L: int, heads: int) -> int:
    if _SMALL == 0 or L not in (128, 256):
        return 0
    if _SMALL_NF is not None:
        v = int(_SMALL_NF)
        return 1 if (v == 2 and L == 256) else v
    return 2 if L == 128 else 1


def _small_long(L: int, heads: int) -> bool:
    # measured (master eval, interleaved graph replays, L2 flushed, trimmed means, attn_inf + combine -> small_core 32 / 64 queries per CTA):
    # L 384 16 x 128 36.9 -> 34.8 / 35.1, 12 x 128 35.9 -> 32.8 / 33.0, 24 x 128 37.9 -> 36.3 / 35.7, 16 x 128 (d 512) 39.0 -> 34.8 / 35.5;
    # L 512 (64) 16 x 128 43.0 -> 41.0, 12 x 128 43.0 -> 40.5, 16 x 128 (d 512) 44.5 -> 41.0, 24 x 128 48.0 -> 49.5 (192 CTAs: two waves);
    # L 640 (64) 12 x 128 53.3 -> 49.1, 16 heads +0.8 to +2.6 (160 CTAs), 24 x 128 +1.9; 32 per CTA at L 512 / 640: +1 to +8 us
    if _SMALL_LONG == 0 or heads == 8:  # head width <= 32
        return False
    if _SMALL_LONG > 1:
        return L in (384, 512, 640)
    return L == 384 or (L == 512 and heads <= 16) or (L == 640 and heads <= 12)


def _sc_mode(L: int, heads: int) -> int:
    """small_core's core-only mode: -(queries per CTA); 0 = 64."""
    if L <= 256:
        if _SMALL_QB:
            return 0 if _SMALL_QB == 64 else -_SMALL_QB
        # L 256 (master eval, 64 -> 32 queries per CTA): 16 x 128 28.7 -> 28.0, 12 x 128 26.7 -> 25.0, 16 x 128 (d 512) 30.7 -> 29.0, 24 x 128
        # 27.8 -> 28.5 (not there); L 128 (d_pair 256 core-only): not measured, stays 64
        return -32 if (L == 256 and heads <= 16) else 0
    if _SMALL_LONG > 1:
        return -_SMALL_LONG
    return -32 if (L == 384 and heads <= 16) else -64


def _cores_b200(dev, H: int, D: int):
    core = _cores.get((dev.index, H, D))
    if core is None:
        core = _cores[(dev.index, H, D)] = _inference_core(dev, H, D)
    return core


def update_inference(
    module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, extra: torch.Tensor | None = None, lp: int = 0
) -> torch.Tensor:
    """``extra``: [B, H, L, L] logit planes (``extra_pair_bias``, natural units) or None. ``lp`` (B200, L % 128 != 0 with L % 32 == 0): the
    core runs at ``lp`` tokens -- the pair pass at L, its planes padded (``planes_fix``), no padded copy of the pair."""
    op = inference32 if single.dtype == torch.float32 else inference
    return op(
        single,
        pair,
        _mask(mask, single.shape[1]),
        _leaves(module),
        module.ln_single.eps,
        module.ln_pair.eps,
        float(module.gate_bias),
        extra,
        int(lp),
    )


# ------------------------------------------------------------------------------------------------------------- training
def _row_saved(dp: int, b200: bool = False) -> int:
    """Saved tensors per batch row of the bf16 training forward: xa, (mean, rstd), qkvg, bias, O, LSE, og (a wide pair: its row
    stats and unmasked bias too; the B200: bias^T last)."""
    return 7 + (2 if dp != DP else 0) + (1 if b200 else 0)


def _forward_fake(single, pair, mask, leaves, eps_s, eps_p, gate_bias=0.0):
    """y and the saved set: the packs Wf, W q | k | v | g (and the padded Wo for 16 x 24 / Hopper, Wb bf16 on Hopper), then per batch
    row xa, (mean, rstd), qkvg, bias, O, LSE, og (a wide pair: its row stats and unmasked bias)."""
    B, L = single.shape[:2]
    H, _, W, D = _geometry(leaves)
    dp = leaves[9].shape[1]
    e, f32 = single.new_empty, torch.float32
    packs_all = torch.cuda.get_device_capability(single.device) in ((9, 0), (10, 0))  # Hopper / B200 pack Wo and Wb (``_prep``)
    b200 = torch.cuda.get_device_capability(single.device) == (10, 0)  # the B200 saves bias^T (made beside the forward core)
    packs = [e((H, dp)), e((4 * W, D))] + ([e((D, W))] if W != D or packs_all else []) + ([e((H, dp))] if packs_all else [])
    row = (
        [
            e((L, D)),
            e((L, 2), dtype=f32),
            e((L, 4 * W)),
            e((H, L, L)),
            e((L, W), dtype=f32),
            e((H, L), dtype=f32),
            e((L, W)),
        ]
        + ([e((L * L, 2), dtype=f32), e((H, L, L), dtype=f32)] if dp != DP else [])
        + ([e((H, L, L) if _fwd_transpose(H, L) else (0,))] if b200 else [])
    )
    return [torch.empty_like(single), *packs, *(t.clone() for _ in range(B) for t in row)]


@opaque(fake=_forward_fake, name="apb_b200_train_fwd")
def forward(
    single: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps_s: float,
    eps_p: float,
    gate_bias: float = 0.0,
) -> list[torch.Tensor]:
    """single [B, L, D], pair [B, L, L, dp] -> [y, *packs, *row 0's saved, *row 1's saved, ...] (``_forward_fake``): the one-row
    forward per batch row, the packs made once."""
    if torch.cuda.get_device_capability(single.device) == (10, 0):
        return _forward_b200(single, pair, mask, leaves, eps_s, eps_p, gate_bias)
    B, L = single.shape[:2]
    rows = _rows()
    lnw, lnb = leaves[0], leaves[1]
    H, _, W, D = _geometry(leaves)
    hopper = torch.cuda.get_device_capability(single.device) == (9, 0)
    with torch.cuda.device(single.device):
        wpack, bvec, wf, wop, wbo = _prep(leaves, 1.0, 1.0, gate_bias)
        wo = leaves[7] if wop is None else wop
        neg = -math.inf if hopper else NEG
        dp = wf.shape[1]
        core = _training_core(single.device)
        out = torch.empty_like(single)
        saved = []
        for b in range(B):
            x, y, m = single[b], out[b], _row(mask, b)
            xa = torch.empty_like(x)
            st = x.new_empty((L, 2), dtype=torch.float32)
            rows.ext().ln_rows(x, lnw, lnb, xa, st, eps_s, y)
            qkvg = torch.addmm(bvec, xa, wpack.t())
            wide = []  # a wide pair: (row stats, unmasked bias) for its backward
            if dp == DP:
                bias = rows.pair_bias(pair[b].view(L * L, DP), wf, m, L, eps_p, neg)
            else:
                bias, *wide = rows.pair_bias(pair[b].view(L * L, dp), wf, m, L, eps_p, neg, saved=True)
            if hopper:
                # The gate is the attention epilogue: O (fp32, for the backward's D) and og = sigmoid(g) O in one pass.
                O, og, LSE = core.apb_forward_gated(qkvg, bias, L, H, D)
            else:
                O, LSE = core.apb_forward(qkvg[:, :W], qkvg[:, W : 2 * W], qkvg[:, 2 * W : 3 * W], bias, L, H, D)
                og = x.new_empty((L, W))
                rows.ext().gate_rows(O, qkvg, og)
            y.addmm_(og, wo.t())  # in place onto the residual seed: no copy of x into the output
            saved += [xa, st, qkvg, bias, O, LSE, og, *wide]
    packs = [wf, wpack] + ([wop] if wop is not None else []) + ([wbo] if wbo is not None else [])
    return [out, *packs, *saved]


def _backward_fake(single, pair, mask, leaves, eps_s, eps_p, saved, dy):
    """One gradient per input (its dtype) and per leaf (fp32), contiguous."""
    return [
        torch.empty_like(single),
        torch.empty_like(pair),
        *(torch.empty(t.shape, dtype=torch.float32, device=t.device) for t in leaves),
    ]


#: v25 (2026-11-18): B200 bf16 / master training of 16 x 32 heads at d_single 512 (W == d_single: no head padding): the dW q | k | v | g GEMM
#: rows and dWo ARE the parameter gradients -- ``backward_nc`` returns the two GEMM outputs, the autograd function hands out views of them, and
#: finalize_red_nc (no copy blocks; it reads neither GEMM output) runs on the main stream beside the GEMMs. The same values (bitwise).
#: Measured (master train r148, graph replays interleaved, trimmed means, off -> on): L 128 63.6 / 63.5 -> 61.3 / 61.0 us, L 256 84.0 / 81.5 ->
#: 83.4 / 83.5, L 384 110.4 / 108.6 -> 107.3 / 108.4, L 512 135.1 -> 133.0, L 768 218.4 -> 215.6; the other unpadded layouts (12 x 32, 24 x 16
#: at 384: 5 MB less to copy) level to +1.6 us -- not routed. FOLDJET_ENGINE_APB_B200_NOCOPY=0: the copying finalize everywhere.
_NOCOPY = os.environ.get("FOLDJET_ENGINE_APB_B200_NOCOPY", "1") != "0"
_NC_SMALL = (0, 1, 3, 8, 9)  # ln_single w / b, bq, ln_pair w, Wb: finalize_red_nc's outputs


def _nocopy(single, leaves) -> bool:
    if not _NOCOPY or single.dtype == torch.float32 or torch.cuda.get_device_capability(single.device) != (10, 0):
        return False
    H, _, W, D = _geometry(leaves)
    return H == 16 and W == D == 512 and leaves[9].shape[1] == DP


def _backward_nc_fake(single, pair, mask, leaves, eps_s, eps_p, saved, dy):
    _, _, W, D = _geometry(leaves)
    f32 = torch.float32
    return [
        torch.empty_like(single),
        torch.empty_like(pair),
        torch.empty((4 * W, D), dtype=f32, device=single.device),
        torch.empty((D, W), dtype=f32, device=single.device),
        *(torch.empty(leaves[i].shape, dtype=f32, device=single.device) for i in _NC_SMALL),
    ]


@opaque(fake=_backward_nc_fake, name="apb_b200_train_bwd_nc")
def backward_nc(
    single: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps_s: float,
    eps_p: float,
    saved: list[torch.Tensor],
    dy: torch.Tensor,
) -> list[torch.Tensor]:
    """``backward`` on B200 at W == d_single: [dx, dpair, dW q|k|v|g [4 W, D], dWo [D, W], the gradients of ``_NC_SMALL``]."""
    return _backward_b200(single, pair, mask, leaves, eps_p, saved, dy, nocopy=True)


def _nc_grads(dwp: torch.Tensor, dwo: torch.Tensor, small: list[torch.Tensor]) -> list[torch.Tensor]:
    """The ten parameter gradients in ``_leaves`` order from ``backward_nc``'s outputs (views of the two GEMM outputs)."""
    W = dwo.shape[1]
    g0, g1, g3, g8, g9 = small
    return [g0, g1, dwp[:W], g3, dwp[W : 2 * W], dwp[2 * W : 3 * W], dwp[3 * W :], dwo, g8, g9]


@opaque(fake=_backward_fake, name="apb_b200_train_bwd")
def backward(
    single: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps_s: float,
    eps_p: float,
    saved: list[torch.Tensor],
    dy: torch.Tensor,
) -> list[torch.Tensor]:
    """The one-row backward per batch row (``saved`` as ``forward`` returns it after y): dx / dpair written into their rows of one
    gradient each, the parameter gradients (fp32) summed over the rows in row order (what accumulating per-sample calls gives)."""
    if torch.cuda.get_device_capability(single.device) == (10, 0):
        return _backward_b200(single, pair, mask, leaves, eps_p, saved, dy)
    B, L = single.shape[:2]
    rows = _rows()
    E = rows.ext()
    lnw, lnpw = leaves[0], leaves[8]
    H, _, W, D = _geometry(leaves)
    dp = leaves[9].shape[1]
    r = _row_saved(dp)
    packs = saved[: len(saved) - B * r]
    wf, wpack = packs[:2]
    # [D, W] bf16 and Wb bf16: packed by ``prep`` on Hopper (its leaves may be an fp32 master), the leaf itself otherwise
    wo = packs[2] if len(packs) > 2 else leaves[7]
    wb = packs[3] if len(packs) > 3 else leaves[9]
    f32 = torch.float32
    hopper = torch.cuda.get_device_capability(single.device) == (9, 0)
    with torch.cuda.device(single.device):
        dyc = dy.contiguous()
        dx_all = torch.empty_like(single)
        dz_all = torch.empty_like(pair)
        outs: list[torch.Tensor] = []
        for b in range(B):
            xa, st, qkvg, bias, O, LSE, og, *rest = saved[len(packs) + b * r : len(packs) + (b + 1) * r]
            wide = (rest[0], rest[1]) if dp != DP else None  # a wide pair: its forward's row stats and unmasked bias
            x = single[b]
            dy2 = dyc[b]
            if hopper:
                # Every producer owns a block partial row; finalize reduces the rows each grid wrote, in block order (no zero fill).
                rows_a = (L + rows.RW - 1) // rows.RW
                rows_b = E.pair_bwd_blocks(x, H, L)
                acc = torch.empty(max(rows_a, rows_b), rows.acc_n(H, D, dp), device=x.device)
            else:
                rows_a = rows_b = 1
                acc = torch.zeros(rows.acc_n(H, D, dp), device=x.device)
            dog = dy2 @ wo  # [L, W]: zero in the pads (Wo's pad columns are zero)
            dwo = torch.mm(dy2.t(), og, out_dtype=f32)  # [D, W]; finalize keeps the real columns
            dob = x.new_empty((L, W))
            dd = x.new_empty((H, L), dtype=f32)
            dqkvg = torch.empty_like(qkvg)
            E.gate_bwd(dog, O, qkvg, dob, dd, dqkvg)
            DQ, DK, DV, DB = _training_core(single.device).apb_backward(
                qkvg[:, :W],
                qkvg[:, W : 2 * W],
                qkvg[:, 2 * W : 3 * W],
                dob,
                bias,
                LSE,
                dd,
                L,
                H,
                D,
            )
            E.qkv_bwd(DQ, DK, DV, dqkvg, acc, D)
            del DQ, DK, DV
            dxa = dqkvg @ wpack
            dwp = torch.mm(dqkvg.t(), xa, out_dtype=f32)
            E.ln_bwd(dxa, x, st, lnw, dy2, dx_all[b], acc)
            rows.pair_bias_bwd(pair[b].view(L * L, dp), DB, wf, L, eps_p, acc, D, saved=wide, out=dz_all[b].view(L * L, dp))
            # custom-op outputs may not alias: every parameter gradient leaves as its own tensor, written by one kernel
            row_outs = [torch.empty(t.shape, dtype=f32, device=t.device) for t in leaves]
            E.finalize(dwp, dwo, acc, wb, lnpw, row_outs, rows_a, rows_b)
            if b == 0:
                outs = row_outs
            else:
                torch._foreach_add_(outs, row_outs)  # pyright: ignore[reportPrivateImportUsage]  -- torch internal, on purpose
    return [dx_all, dz_all, *outs]


#: v26: the bf16 training forward at L 128 (d_pair 128) as ONE launch for the core, the gate and to_out (``small_core_train``,
#: apb_rows_sm100.cu: the fused small_core form with attn_fwd2's scaling, also writing O / LSE / og for the backward); the value is the
#: queries per CTA (16 / 32), 0: attn_fwd2 -> gate_rows -> cuBLAS to_out. FOLDJET_ENGINE_APB_B200_TRF overrides.
_TRF = int(os.environ.get("FOLDJET_ENGINE_APB_B200_TRF", "16"))
_TRF256 = int(os.environ.get("FOLDJET_ENGINE_APB_B200_TRF256", "0"))  # 1: also at L 256 (16 queries per CTA)


def _train_fused(L: int, dp: int, heads: int) -> bool:
    # v26 (master train L 128, interleaved replays, trimmed means, attn_fwd2 + gate_rows + to_out -> fused 16 / 32 queries per CTA): 12 x 128
    # 53.6 -> 51.3 / 51.3, 24 x 128 60.3 -> 59.4 / 59.4, 16 x 128 (d 512) 60.4 -> 57.4 / 57.4, 16 x 128 58.6 -> 54.2 / 54.8; gradients within
    # 6e-3 (relative to the max) of the attn_fwd2 path
    return _TRF > 0 and (L == 128 or (L == 256 and _TRF256 > 0)) and dp == DP and heads != 8


#: v28 (2026-11-18): the q | k | v | g GEMM captured before the side stream's pair pass (graph node order), so its CTAs take SMs first
#: instead of queueing behind the pair pass's persistent grid (timeline r148 L 256: the GEMM 14.8 -> 20.9 us behind the pass; first: 8.4 ->
#: 12.0, the pass 9.1 -> 11.7 us beside it). A / B (master train, interleaved replays, trimmed means, the same bits): L 256 r147 83.2 -> 80.4,
#: r222 75.2 -> 74.7, r148 / r146 level; L 384 / 512 r148 106.0 / 133.6 -> 104.5 / 132.5, r222 103.9 -> 102.6, r146 level; L 768 r146 198.6
#: -> 193.6, r148 223.5 -> 218.3, r222 216.2 -> 213.8, r147 +3.4; L 640 level; L 128 +1.3 to +2.7 (not routed). Wide pairs (d_pair 256-512,
#: the tcgen05 / wide pass) lose: r223 / r225 / r226 / r47 L 256 / 384 +3 to +14 us -- d_pair 128 only. FOLDJET_ENGINE_APB_B200_GEMM_FIRST = 0 / 1
#: forces.
_GEMM_FIRST = os.environ.get("FOLDJET_ENGINE_APB_B200_GEMM_FIRST")


def _gemm_first(L: int, heads: int, dp: int) -> bool:
    if _GEMM_FIRST is not None:
        return _GEMM_FIRST != "0"
    return dp == DP and 256 <= L <= 768 and (heads <= 16 or L == 256)


#: v28: prep_ln's minimum blocks per SM at D 512 (__launch_bounds__(256, 5)): without it the D 512 launch took 64 registers a thread, 4
#: blocks per SM, and the 16 x 32 pack + LayerNorm (689 blocks) ran in two waves after the flush -- 6.2 -> 4.4 us (r148, timeline). A / B
#: (master train r148, the same bits): L 128 / 256 / 384 / 512 55.4 / 78.5 / 108.8 / 135.5 -> 53.4 / 75.9 / 106.0 / 133.6 us. D 384 (48
#: registers, 5 blocks per SM already) keeps the plain build. FOLDJET_ENGINE_APB_B200_PREP_MB=0: off.
_PREP_MB = int(os.environ.get("FOLDJET_ENGINE_APB_B200_PREP_MB", "5"))
#: v28: the training forward runs pair_bias_fwd2 (v21's 8-lane-issue form, the same bits) from L 256 at 16 heads, else from L 384. A / B (with
#: the GEMM first): L 256 r148 75.9 -> 73.9, r222 74.4 -> 74.0; 12 / 24 heads +1.4 / +0.8 (kept from 384). FOLDJET_ENGINE_APB_B200_PF2_TMIN forces.
_PF2_TMIN = os.environ.get("FOLDJET_ENGINE_APB_B200_PF2_TMIN")


def _pf2_tmin(heads: int) -> int:
    return int(_PF2_TMIN) if _PF2_TMIN is not None else (256 if heads == 16 else 384)


def _ln_gemm(E, ln: bool, x, lnw, lnb, xa, st, eps_s, y, main, packed, bvec, wpack) -> torch.Tensor:
    """The training forward's main-stream head of a batch row: LayerNorm(single) unless prep_ln made it (``ln``), the wait for the pack
    (row 0: ``packed``), then qkvg = xa Wpack^T + bias."""
    if ln:
        E.ln_rows(x, lnw, lnb, xa, st, eps_s, y)
    if packed is not None:
        _after(main, packed)
    return torch.addmm(bvec, xa, wpack.t())


def _forward_b200(single, pair, mask, leaves, eps_s, eps_p, gate_bias):
    """``forward`` on B200 (2026-11-17): the same kernels and values on two streams, as ``_inference_b200``: the weight pack and each row's
    pair pass on the side stream, LN(single) / the q | k | v | g GEMM / the core / the gate / to_out on the main one."""
    B, L = single.shape[:2]
    rows = _rows()
    E = rows.ext()
    lnw, lnb = leaves[0], leaves[1]
    H, _, W, D = _geometry(leaves)
    dev = single.device
    sm = _sm100()
    pdl = _pdl_on(L)  # v20: the main stream's own kernels as programmatic dependents
    with torch.cuda.device(dev):
        main, side = torch.cuda.current_stream(dev), _side_stream(dev)
        _join(side, main)
        out = torch.empty_like(single)
        xa0, st0 = torch.empty_like(single[0]), single.new_empty((L, 2), dtype=torch.float32)
        ln0: list = [single[0], lnw, lnb, xa0, st0, eps_s, out[0]]
        with torch.cuda.stream(side):
            wpack, bvec, wf, wop, wbo = _prep(leaves, 1.0, 1.0, gate_bias, ln0)
        assert wop is not None and wbo is not None  # B200: prep_dp always packs Wo and Wb
        packed = torch.cuda.Event()
        _rec(packed, side)
        dp = wf.shape[1]
        saved: list[torch.Tensor] = []
        side_made = [wpack, bvec, wf, wop, wbo]
        for b in range(B):
            m = _row(mask, b)
            x, y = single[b], out[b]
            xa = xa0 if b == 0 else torch.empty_like(x)
            st = st0 if b == 0 else x.new_empty((L, 2), dtype=torch.float32)
            gemm_args = (E, b > 0 or len(ln0) == 7, x, lnw, lnb, xa, st, eps_s, y, main, packed if b == 0 else None, bvec, wpack)
            gemm_first = _gemm_first(L, H, dp)
            if gemm_first:  # v28: the q | k | v | g GEMM captured before the side pair pass (its CTAs take SMs first)
                qkvg = _ln_gemm(*gemm_args)
            with torch.cuda.stream(side):
                wide: list[torch.Tensor] = []  # a wide pair: (row stats, unmasked bias) for its backward
                pf2 = _pf2(dp, L, H, train=True) if _pf2_tmin(H) <= L else None  # L 128: +-2 us, one timer step either way
                if pf2 is not None:  # v21: the B200 form of the d_pair 128 pass (the same bits)
                    bias = pair.new_empty((H, L, L))
                    E.pair_bias_fwd2(pair[b].view(L * L, DP), wf, m, bias, L, eps_p, NEG, *pf2)
                elif dp == DP:
                    bias = rows.pair_bias(pair[b].view(L * L, DP), wf, m, L, eps_p, NEG)
                elif _pair_tc(dp, L, H, True):  # the tcgen05 pass, with the row stats and the unmasked bias its backward reads
                    wide = [pair.new_empty((L * L, 2), dtype=torch.float32), pair.new_empty((H, L, L), dtype=torch.float32)]
                    bias = sm.apb_pair_fwd(pair[b].view(L * L, dp), wf, None, 1.0, m, L, eps_p, NEG, stats=wide[0], unmasked=wide[1])
                else:
                    bias, *wide = rows.pair_bias(pair[b].view(L * L, dp), wf, m, L, eps_p, NEG, saved=True)
            planes = torch.cuda.Event()
            _rec(planes, side)
            with torch.cuda.stream(side):  # bias^T for the backward's attn_dkv, beside the core (which leaves most SMs free)
                bias_t = sm.bias_transpose64(bias, _TR_BLOCKS) if _fwd_transpose(H, L) else pair.new_empty(0)
            t_done = torch.cuda.Event()
            _rec(t_done, side)
            side_made += [bias, *wide, bias_t]
            if not gemm_first:
                qkvg = _ln_gemm(*gemm_args)
            _after(main, planes)
            if _train_fused(L, dp, H):  # v26: core + gate + to_out in one launch (O, LSE, og saved as attn_fwd2 / gate_rows write them)
                O, LSE, og = x.new_empty((L, W), dtype=torch.float32), x.new_empty((H, L), dtype=torch.float32), x.new_empty((L, W))
                E.small_core_train(qkvg, bias, wop, y, O, LSE, og, L, pdl, sm.APB_GEOMETRY[(H, D)][1] ** -0.5, _TRF)
            else:
                O, LSE = sm.apb_forward(qkvg[:, :W], qkvg[:, W : 2 * W], qkvg[:, 2 * W : 3 * W], bias, L, H, D, pdl=pdl)
                og = x.new_empty((L, W))
                _pdl_call(E, pdl, E.gate_rows_p, O, qkvg, og)
                y.addmm_(og, wop.t())  # in place onto the residual seed
            _after(main, t_done)  # the backward reads bias^T on the main stream
            saved += [xa, st, qkvg, bias, O, LSE, og, *wide, bias_t]
        for t in side_made:
            t.record_stream(main)
        for t in (pair,) + (() if mask is None else (mask,)):
            t.record_stream(side)
    return [out, wf, wpack, wop, wbo, *saved]


def _backward_b200(single, pair, mask, leaves, eps_p, saved, dy, nocopy: bool = False):
    """``backward`` on B200 (2026-11-17): the same kernels and values, the independent stages on three streams per batch row --
    main: dO = dy Wo, the gate backward, attn_dkv, qkv_bwd, dxa, the LayerNorm backward; side 0: dWo, the bias transpose, dW q|k|v|g;
    side 1: attn_dqb, then the pair backward (it needs only dbias). The parameter-gradient accumulator's regions are disjoint per writer."""
    B, L = single.shape[:2]
    rows = _rows()
    E = rows.ext()
    sm = _sm100()
    pdl = _pdl_on(L)  # v20: the main stream's own kernels as programmatic dependents
    lnw, lnpw = leaves[0], leaves[8]
    H, _, W, D = _geometry(leaves)
    dp = leaves[9].shape[1]
    r = _row_saved(dp, True)
    wf, wpack, wo, wb = saved[:4]
    f32 = torch.float32
    dev = single.device
    outs: list[torch.Tensor] = []
    with torch.cuda.device(dev):
        main, s1, s2 = torch.cuda.current_stream(dev), _side_stream(dev, 0), _side_stream(dev, 1)
        dyc = dy.contiguous()
        dx_all = torch.empty_like(single)
        dz_all = torch.empty_like(pair)
        for b in range(B):
            xa, st, qkvg, bias, O, LSE, og, *rest = saved[4 + b * r : 4 + (b + 1) * r]
            wide = (rest[0], rest[1]) if dp != DP else None
            fwd_t = rest[-1].numel() > 0  # the forward made bias^T (``_fwd_transpose``)
            x, dy2 = single[b], dyc[b]
            # block partial rows, reduced in a fixed order by finalize (as on H100): the parameter-gradient sums are bitwise repeatable
            rows_a = (L + rows.RW - 1) // rows.RW
            pbtc = _PAIR_BWD_TC and dp == DP and L % 128 == 0
            was = E.sm_reserve(_bwd_reserve(dp, L))  # the partial rows and the launch's grid under the same SM count
            if pbtc:
                rows_b = min(torch.cuda.get_device_properties(dev).multi_processor_count - _pbtc_reserve(H, L), L * (L // 128))
            else:
                rows_b = E.pair_bwd_blocks(x, H, L) if dp == DP else E.pair_bwd_wide1_blocks(x, H, L, dp)
            E.sm_reserve(was)
            acc = torch.empty(max(rows_a, rows_b), rows.acc_n(H, D, dp), device=dev)
            _join(s1, main)
            dqkvg = torch.empty_like(qkvg)
            sbq = _sbwd(L, H)
            if sbq:  # v27: gate_bwd + dkv + dqb + qkv_bwd in one launch; the pair backward and dxa / dW start together after it
                DB = torch.empty((H, L, L), device=dev, dtype=f32)
                with torch.cuda.stream(s1):
                    dwo = torch.mm(dy2.t(), og, out_dtype=f32)
                dog = dy2 @ wo
                E.small_core_bwd(dog, O, og, qkvg, bias, LSE, dqkvg, DB, acc, D, L, _rsqd(H, D), pdl, sbq)
                _join(s2, main)
                with torch.cuda.stream(s2):
                    if pbtc:
                        z2, dz2 = pair[b].view(L * L, dp), dz_all[b].view(L * L, dp)
                        sm.apb_pair_bwd(z2, DB, wf, L, eps_p, acc[:, 2 * D + W :], dz2, _pbtc_reserve(H, L))
                    else:
                        was = E.sm_reserve(_bwd_reserve(dp, L))
                        rows.pair_bias_bwd(pair[b].view(L * L, dp), DB, wf, L, eps_p, acc, D, saved=wide, out=dz_all[b].view(L * L, dp))
                        E.sm_reserve(was)
            else:
                dwo = _backward_attn(
                    E,
                    sm,
                    rows,
                    s1,
                    s2,
                    main,
                    pdl,
                    qkvg,
                    bias,
                    LSE,
                    O,
                    og,
                    dy2,
                    wo,
                    dqkvg,
                    acc,
                    rest,
                    fwd_t,
                    pbtc,
                    pair,
                    dz_all,
                    b,
                    wf,
                    wide,
                    L,
                    H,
                    D,
                    W,
                    dp,
                    eps_p,
                )
            if _DXA_FIRST:  # dxa leads the tail (-> ln_bwd -> finalize); dW q|k|v|g is issued after it, on the side stream
                qkv_done = torch.cuda.Event()
                _rec(qkv_done, main)
                dxa = dqkvg @ wpack
                _after(s1, qkv_done)
            else:
                _join(s1, main)
            with torch.cuda.stream(s1):
                dwp = torch.mm(dqkvg.t(), xa, out_dtype=f32)
            if not _DXA_FIRST:
                dxa = dqkvg @ wpack
            _pdl_call(E, pdl, E.ln_bwd_p, dxa, x, st, lnw, dy2, dx_all[b], acc)
            if _fsplit(H, L):  # v27: finalize_red's parts where their inputs are made (the same blocks: the same bits)
                if nocopy:
                    small = [torch.empty(leaves[i].shape, dtype=f32, device=dev) for i in _NC_SMALL]
                    ph = small[0]  # out[2], out[4..7]: not written without the copy part
                    row_outs = [small[0], small[1], ph, small[2], ph, ph, ph, ph, small[3], small[4]]
                else:
                    row_outs = [torch.empty(t.shape, dtype=f32, device=dev) for t in leaves]
                fin = (dwp, dwo, acc, wb, lnpw, row_outs, rows_a, rows_b, not nocopy)
                with torch.cuda.stream(s2):  # dWf | dWb | dln_pair.weight: right after the pair backward
                    _pdl_call(E, pdl, E.finalize_part, *fin, 4)
                if not nocopy:
                    with torch.cuda.stream(s1):  # the dW copies: right after the dW GEMMs
                        E.finalize_part(*fin, 1)
                _pdl_call(E, pdl, E.finalize_part, *fin, 2)  # dlnw | dlnb | dbq: right after ln_bwd
                _join(main, s1)
                _join(main, s2)
                dwo.record_stream(main)
                dwp.record_stream(main)
                got = [dwp, dwo, *small] if nocopy else row_outs
                if b == 0:
                    outs = got
                else:
                    torch._foreach_add_(outs, got)  # pyright: ignore[reportPrivateImportUsage]  -- torch internal, on purpose
                continue
            if nocopy:  # v25: finalize reads only acc: it runs beside dWo / dW q|k|v|g, whose outputs are the gradients themselves
                _join(main, s2)
                small = [torch.empty(leaves[i].shape, dtype=f32, device=dev) for i in _NC_SMALL]
                ph = small[0]  # out[2], out[4..7]: not written by finalize_red_nc
                row_outs = [small[0], small[1], ph, small[2], ph, ph, ph, ph, small[3], small[4]]
                _pdl_call(E, pdl, E.finalize_red_nc, dwp, dwo, acc, wb, lnpw, row_outs, rows_a, rows_b)
                _join(main, s1)
                dwo.record_stream(main)
                dwp.record_stream(main)
                got = [dwp, dwo, *small]
                if b == 0:
                    outs = got
                else:
                    torch._foreach_add_(outs, got)  # pyright: ignore[reportPrivateImportUsage]  -- torch internal, on purpose
                continue
            _join(main, s1)
            _join(main, s2)
            dwo.record_stream(main)
            dwp.record_stream(main)
            # custom-op outputs may not alias: every parameter gradient leaves as its own tensor, written by one kernel
            row_outs = [torch.empty(t.shape, dtype=f32, device=t.device) for t in leaves]
            if _P5 and H <= _FR_MAXH:
                _pdl_call(E, pdl, E.finalize_red, dwp, dwo, acc, wb, lnpw, row_outs, rows_a, rows_b)
            else:
                E.finalize(dwp, dwo, acc, wb, lnpw, row_outs, rows_a, rows_b)
            if b == 0:
                outs = row_outs
            else:
                torch._foreach_add_(outs, row_outs)  # pyright: ignore[reportPrivateImportUsage]  -- torch internal, on purpose
    return [dx_all, dz_all, *outs]


def _backward_attn(
    E,
    sm,
    rows,
    s1,
    s2,
    main,
    pdl,
    qkvg,
    bias,
    LSE,
    O,
    og,
    dy2,
    wo,
    dqkvg,
    acc,
    rest,
    fwd_t,
    pbtc,
    pair,
    dz_all,
    b,
    wf,
    wide,
    L,
    H,
    D,
    W,
    dp,
    eps_p,
):
    """The v26 attention backward up to dqkvg on the main stream (gate_bwd, attn_dkv, attn_dqb / dqb_small, qkv_bwd; dWo and the bias
    transpose on s1, dqb and the pair backward on s2); returns dWo."""
    f32 = torch.float32
    dob = qkvg.new_empty((L, W))
    dd = qkvg.new_empty((H, L), dtype=f32)
    (tr, dkv, dqb), (DQ, DK, DV, DB, _bias_t) = sm.apb_backward_stages(  # _bias_t: held until dkv has run
        qkvg[:, :W],
        qkvg[:, W : 2 * W],
        qkvg[:, 2 * W : 3 * W],
        dob,
        bias,
        LSE,
        dd,
        L,
        H,
        D,
        bias_t=rest[-1] if fwd_t else None,
        bias_nt=not fwd_t and _bias_nt(H, L),
        dkv_pdl=pdl,
    )
    transposed = torch.cuda.Event()
    with torch.cuda.stream(s1):
        dwo = torch.mm(dy2.t(), og, out_dtype=f32)  # [D, W]; finalize keeps the real columns
        if not fwd_t:
            tr()
            _rec(transposed, s1)
    dog = dy2 @ wo
    _pdl_call(E, pdl, E.gate_bwd2 if _P5 and H == 12 else E.gate_bwd_p, dog, O, qkvg, dob, dd, dqkvg)
    _join(s2, main)
    with torch.cuda.stream(s2):
        if _dqb_small(L):  # v23: the query rows split over NH L / 32 CTAs (the same quantities)
            E.dqb_small(qkvg[:, :W], qkvg[:, W : 2 * W], qkvg[:, 2 * W : 3 * W], dob, bias, LSE, dd, DQ, DB, L, W // H, _rsqd(H, D))
        else:
            dqb()
        dq_done = torch.cuda.Event()
        _rec(dq_done, s2)
        if pbtc:  # tcgen05: its block partial rows go into the dWf | dbias-sum columns of acc
            z2, dz2 = pair[b].view(L * L, dp), dz_all[b].view(L * L, dp)
            sm.apb_pair_bwd(z2, DB, wf, L, eps_p, acc[:, 2 * D + W :], dz2, _pbtc_reserve(H, L))
        else:
            was = E.sm_reserve(_bwd_reserve(dp, L))
            rows.pair_bias_bwd(pair[b].view(L * L, dp), DB, wf, L, eps_p, acc, D, saved=wide, out=dz_all[b].view(L * L, dp))
            E.sm_reserve(was)
    _after(main, transposed)
    dkv()
    _after(main, dq_done)
    _pdl_call(E, pdl, E.qkv_bwd_p, DQ, DK, DV, dqkvg, acc, D)
    return dwo


def _pair_weight_shadow_fake(single, pair, mask, leaves, eps_s, eps_p, dy):
    return torch.empty(leaves[-1].shape, device=single.device, dtype=torch.float32)


@opaque(fake=_pair_weight_shadow_fake, name="apb_h100_pair_weight_fp32_shadow")
def _pair_weight_shadow(
    single: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps_s: float,
    eps_p: float,
    dy: torch.Tensor,
) -> torch.Tensor:
    """Accuracy-first exact pair-weight (to_bias.weight) gradient with native CUDA and cuBLAS.

    This intentionally recreates the FP32 numerical path of the oracle: the
    exact weight gradient retains reduction roundoff that the fused kernel does
    not. Its head-major operand matches the reference Linear backward layout.
    This materializes pair normalization for dWb and exceeds B200 lifetimes.
    """
    E = _rows().ext(shadow=True)
    with torch.cuda.device(single.device), torch.autocast("cuda", enabled=False):
        shape = (leaves[-1].shape[0], single.shape[-1])
        qualified = single.shape[1] in (384, 768) and shape in (
            (8, 384),
            (12, 384),
            (16, 384),
            (24, 384),
            (16, 512),
        )
        # The exact pair-weight gradient needs graph-owned scratch on the
        # qualified cells; the other cells keep the plain exact GEMM.
        mm = E.exact_mm_owned if qualified else E.exact_mm
        shadow_io = qualified
        convert = E.exact_fp32_inputs_grouped if shadow_io else E.exact_fp32_inputs
        pair_ln = 2 if qualified else 0
        fused_pair_project = qualified and shape[0] in (8, 12, 16)
        if pair_ln:
            values = convert([single, dy, *leaves])
            x, dy32 = values[:2]
            affine = values[2:]
        else:
            values = convert([single, pair, dy, *leaves])
            x, z, dy32 = values[:3]
            affine = values[3:]
        lnw, lnb, wq, bq, wk, wv, wg, wo, lnpw, wb = affine
        L, D = single.shape[1:]
        H, dh = wb.shape[0], D // wb.shape[0]
        xa = E.exact_ln(x, lnw, lnb, eps_s).view(L, D)
        # Preserve the accurate D2 producer when explicitly selected or for
        # H24. The fused producer keeps its normalized pair tile on chip and
        # writes masked head-major bias directly for the other required shapes.
        za = (
            None
            if fused_pair_project
            else (E.exact_ln_bf16(pair, lnpw, eps_p, pair_ln == 2) if pair_ln else E.exact_ln(z, lnpw, torch.zeros_like(lnpw), eps_p)).view(
                L * L, DP
            )
        )

        def finish(db):
            norm = za
            if norm is None:
                norm = E.exact_ln_bf16(pair, lnpw, eps_p, pair_ln == 2).view(L * L, DP)
            db_head = E.exact_transpose_tiled(db, None, L, False).view(H, L * L)
            return E.exact_mm_owned(db_head, norm)

        q = mm(xa, wq.t(), bq)
        k = mm(xa, wk.t())
        v = mm(xa, wv.t())
        gate = mm(xa, wg.t())
        bias = E.exact_ln_pair_project(pair, lnpw, wb, mask, eps_p) if fused_pair_project else mm(za, wb.t())
        dog = mm(dy32.view(L, D), wo)
        do_rows = E.exact_point(dog, gate, 1.0, 1).view(L, H, dh)
        if mask is not None:
            # masked_fill makes the oracle's bias contiguous and selects its
            # efficient CUTLASS FP32 attention. Reproduce that arithmetic with
            # project CUDA launches of the installed C++ templates, including
            # the FP32 rounded product/reduction used for its delta tensor.
            qh, kh, vh = (t.view(L, H, dh) for t in (q, k, v))
            bm = (
                bias
                if fused_pair_project
                else (E.exact_transpose_tiled(bias, mask, L, False) if shadow_io else E.exact_mask_bias(bias, mask, L))
            )
            out32, lse32 = E.exact_attention_fwd(qh, kh, vh, bm)
            delta32 = E.exact_delta(do_rows, out32)
            ds = (
                E.exact_attention_db_only(qh, kh, vh, bm, lse32, do_rows, delta32)
                if qualified
                else E.exact_attention_db(qh, kh, vh, bm, out32, lse32, do_rows, delta32)
            )
            db = E.exact_transpose_tiled(ds, mask, L, True) if shadow_io and H >= 16 else E.exact_db_transpose(ds, mask)
            return finish(db)
        scale = math.sqrt(1.0 / math.sqrt(dh))
        qs = E.exact_point(q, q, scale, 0).view(1, L, H, dh).transpose(1, 2)
        ks = E.exact_point(k, k, scale, 0).view(1, L, H, dh).transpose(1, 2)
        scores = mm(qs, ks.transpose(-2, -1))
        E.exact_bias_add(scores, bias, mask)
        probs = E.exact_soft(scores, scores, False)
        do = do_rows.view(1, L, H, dh).transpose(1, 2)
        vh = v.view(1, L, H, dh).transpose(1, 2)
        dp = mm(do, vh.transpose(-2, -1))
        ds = E.exact_soft(dp, probs, True)
        db = E.exact_transpose_tiled(ds, mask, L, True) if shadow_io and H >= 16 else E.exact_db_transpose(ds, mask)
        return finish(db)


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eps_s, eps_p, gate_bias, mask, single, pair, *leaves):
        ctx.param_dtypes = [t.dtype for t in leaves]
        params = leaves
        ctx.beta_shadow = torch.cuda.get_device_capability(single.device) == (9, 0)
        # Resolve the repair during forward so that the captured gradient policy
        # is fixed before backward.
        qualified_exact_dwb = single.shape[1] in (384, 768) and (
            leaves[-1].shape[0],
            single.shape[-1],
        ) in ((8, 384), (12, 384), (16, 384), (24, 384), (16, 512))
        # ln_pair has no beta: the shadow only repairs the pair-weight gradient (of a 128-wide pair without a gate constant: its
        # recompute has neither).
        ctx.exact_dwb = EXACT_DWB and ctx.beta_shadow and qualified_exact_dwb and leaves[-1].shape[1] == DP and not gate_bias
        leaves = _kernel_leaves(leaves)
        y, *saved = forward(single, pair, mask, leaves, eps_s, eps_p, gate_bias)
        ctx.eps = (eps_s, eps_p)
        ctx.saved_count = len(saved)
        ctx.save_for_backward(mask, single, pair, *leaves, *saved, *(params if ctx.exact_dwb else ()))
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        v = ctx.saved_tensors
        mask, single, pair, leaves, saved = (
            v[0],
            v[1],
            v[2],
            list(v[3:13]),
            list(v[13 : 13 + ctx.saved_count]),
        )
        eps_s, eps_p = ctx.eps
        if _nocopy(single, leaves):  # v25: B200, 16 x 32 at 512 (no head padding)
            dx, dz, dwp, dwo, *small = backward_nc(single, pair, mask, leaves, eps_s, eps_p, saved, dy)
            grads = _nc_grads(dwp, dwo, small)
        else:
            dx, dz, *grads = backward(single, pair, mask, leaves, eps_s, eps_p, saved, dy)
        if ctx.exact_dwb:  # per batch row (the shadow is a one-row recompute), summed in row order
            params = list(v[13 + ctx.saved_count :])
            dyc = dy.contiguous()
            for b in range(single.shape[0]):
                g = _pair_weight_shadow(single[b : b + 1], pair[b : b + 1], _row(mask, b), params, eps_s, eps_p, dyc[b : b + 1])
                grads[9] = g if b == 0 else grads[9] + g
        if torch.cuda.get_device_capability(single.device) == (9, 0):
            if torch.compiler.is_compiling():
                grads = [g.to(dt) for g, dt in zip(grads, ctx.param_dtypes, strict=True)]
            else:
                grads = _cast_cuda.cast(grads, [torch.float32 if dt == torch.float32 else torch.bfloat16 for dt in ctx.param_dtypes])
            from foldjet_engine.kernels.integrations.token_dit_train import (
                _activate_native_caller,
            )

            _activate_native_caller(force=True)
            return (None, None, None, None, dx, dz, *grads)
        return (
            None,
            None,
            None,
            None,
            dx,
            dz,
            *(g.to(dt) for g, dt in zip(grads, ctx.param_dtypes, strict=True)),
        )


def update_train(module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if single.dtype == torch.float32:  # the TF32 path: plain autograd accumulation
        return _Training32.apply(
            module.ln_single.eps, module.ln_pair.eps, float(module.gate_bias), _mask(mask, single.shape[1]), single, pair, *_leaves(module)
        )
    out = _Training.apply(
        module.ln_single.eps,
        module.ln_pair.eps,
        float(module.gate_bias),
        _mask(mask, single.shape[1]),
        single,
        pair,
        *_leaves(module),
    )
    if torch.cuda.get_device_capability(single.device) == (9, 0):
        from foldjet_engine.kernels.integrations.token_dit_train import _native_caller_hooks

        # The result graph now owns its AccumulateGrad nodes. Installing on
        # fresh leaves before apply can hook a temporary weak-held node.
        _native_caller_hooks(single, pair, *module.parameters(), force=True)
    return out


# ---------------------------------------------------------------------------------------------- H100 float32 activations (TF32)
# fp32 activations and fp32 parameters: the same op on the engine's TF32 attention core (augmented_attention/cuda/attn_tf32.cu, the
# token DiT's fp32 core, built for the head layout NHP x DHP) and the fp32 row / pair kernels of apb_rows.cu; the projections are fp32
# cuBLAS GEMMs (TF32 products when the session allows them, as the reference's). Head widths go to the core as 32 or 48 (16 x 24 and
# 24 x 16 padded to 32 with zero weight rows / columns; the q scale and dq use the real width). The core's bias is key-permuted
# (``conditioned_transition.cuda.key_perm``) and in log2 units, the masked keys -inf.
#   inference: ln_rows (fp32) -> addmm q | k | g (q in exp2 units, + gate_bias) and v^T -> pair_bias32_fwd -> attn_tf32_inf (sigmoid(g) o
#              over q) -> to_out accumulated onto the residual seed
#   training:  ln_rows -> addmm q | k | v | g -> pair_bias32_fwd (+ row stats) -> tf32_forward (O, LSE) -> og = sigmoid(g) O -> to_out;
#              backward: dog, the gate, tf32_backward (dQ, dK, dV, dbias), the dxa / dW GEMMs, ln_bwd (fp32), pair_bias32_bwd and
#              dWf = (rstd dbias) pair - (rstd dbias . mean) (one GEMM on the pair as it is).
def _geometry32(leaves):
    """(heads, real head dim, core head width 32 / 48, row width W, d_single, d_pair) of the fp32 path."""
    H, D, dp = leaves[9].shape[0], leaves[0].numel(), leaves[9].shape[1]
    dh = D // H
    dhp = 48 if dh == 48 else 32
    return H, dh, dhp, H * dhp, D, dp


def _pad_rows(w, H, dh, dhp):
    """[H dh, ...] -> [H dhp, ...] with zero rows after each head's dh (or w itself without padding)."""
    if dh == dhp:
        return w
    return torch.nn.functional.pad(w.reshape(H, dh, -1), (0, 0, 0, dhp - dh)).reshape(H * dhp, *w.shape[1:])


def _unpad_rows(w, H, dh, dhp):
    return w if dh == dhp else w.reshape(H, dhp, -1)[:, :dh].reshape(H * dh, *w.shape[1:])


def _pack32(leaves, gate_bias):
    """fp32 packs: W q | k | v | g [4 W, D] (padded rows, natural units), its bias [4 W] (bq, then gate_bias on g), Wo [D, W]
    (padded columns), Wf = to_bias.weight * ln_pair.weight [H, dp]. A handful of launches (the training step makes them once per
    parameter version: ``_train32_packs``)."""
    _, _, wq, bq, wk, wv, wg, wo, lnpw, wb = leaves
    H, dh, dhp, W, D, _ = _geometry32(leaves)
    b4 = wq.new_zeros(4 * W)
    if gate_bias:
        b4[3 * W :] = gate_bias
    if dh == dhp:
        w4, wop = torch.cat([wq, wk, wv, wg]), wo.clone()
        b4[:W] = bq
    else:
        w4 = wq.new_zeros((4, H, dhp, D))
        w4[:, :, :dh] = torch.stack([wq, wk, wv, wg]).view(4, H, dh, D)
        w4 = w4.view(4 * W, D)
        b4.view(4, H, dhp)[0, :, :dh] = bq.view(H, dh)
        wop = wo.new_zeros((D, H, dhp))
        wop[:, :, :dh] = wo.view(D, H, dh)
        wop = wop.view(D, W)
    return w4, b4, wop, wb * lnpw


def _train32_packs(params, gate_bias):
    """``_pack32`` cached per parameter version (``_capture.lookup``, as ``_inference_packs``): the training forward makes them, its
    backward reuses them."""
    b200 = torch.cuda.get_device_capability(params[0].device) == (10, 0)
    return _capture.lookup(
        _packs,
        ("f32train", gate_bias, *((t.data_ptr(), t._version) for t in params)),
        lambda: (*(_train32_packs_b200 if b200 else _pack32)(params, gate_bias), params),
        limit=_capture.BLOCK_CACHE_LIMIT,
    )[:4]


def _train32_packs_b200(params, gate_bias):
    """``_pack32`` on B200 in ONE launch (``bias_only_dit/cuda/bo_pack_sm100.cu`` through ``pack_sm100.Pack``, the same values bit for bit):
    a capture re-runs the pack at every replay, and the torch form is ~10 small kernels."""
    from foldjet_engine.kernels.bias_only_dit.cuda.pack_sm100 import Pack

    _, wq, bq, wk, wv, wg, wo, lnpw, wb = params[1:]
    H, dh, dhp, W, D, dp = _geometry32(params)
    e = lambda *s: wq.new_empty(s, dtype=torch.float32)
    w4, b4, wop, wf = e(4 * W, D), e(4 * W), e(D, W), e(H, dp)
    P = Pack(torch.float32)
    for i, w in enumerate((wq, wk, wv, wg)):
        P.pad_rows(w, w4[i * W : (i + 1) * W], dh, dhp)
    P.pad_cols(bq.view(1, H * dh), b4[:W].view(1, W), dh, dhp)
    P.zero(b4[W : 3 * W])
    P.zero(b4[3 * W :], float(gate_bias))
    P.pad_cols(wo, wop, dh, dhp)
    P.copy(wb, wf, lnpw)
    P.launch(wq.device)
    return w4, b4, wop, wf


def _key_perm32(L, device):
    """(perm, inv) of the TF32 core's key order, cached like the packs."""
    from foldjet_engine.kernels.conditioned_transition.cuda import key_perm

    def build():
        perm = key_perm(L, device)
        return perm, torch.argsort(perm)

    return _capture.lookup(_packs, ("kperm", L, device.index), build, limit=_capture.BLOCK_CACHE_LIMIT)


def _inference32_packs_b200(params, gate_bias):
    """B200: ``_inference32_packs``'s forms in ONE launch (``bias_only_dit/cuda/bo_pack_sm100.cu`` through ``pack_sm100.Pack``: head-padded
    rows / columns, the q scale and the log2 e of Wf as constant factors, the gate bias as a fill), bit for bit the torch pack (~15 small
    kernels, which a captured step re-runs every replay)."""
    from foldjet_engine.kernels.bias_only_dit.cuda.pack_sm100 import Pack

    _, wq, bq, wk, wv, wg, wo, lnpw, wb = params[1:]
    H, dh, dhp, W, D, dp = _geometry32(params)
    qs = LOG2E / math.sqrt(dh)
    e = lambda *s: wq.new_empty(s, dtype=torch.float32)
    w3, b3, wvp, wop, wf = e(3 * W, D), e(3 * W), e(W, D), e(D, W), e(H, dp)
    P = Pack(torch.float32)
    P.pad_rows(wq, w3[:W], dh, dhp, cs=qs)
    P.pad_rows(wk, w3[W : 2 * W], dh, dhp)
    P.pad_rows(wg, w3[2 * W :], dh, dhp)
    P.pad_cols(bq.view(1, H * dh), b3[:W].view(1, W), dh, dhp, cs=qs)
    P.zero(b3[W : 2 * W])
    P.zero(b3[2 * W :], float(gate_bias))
    P.pad_rows(wv, wvp, dh, dhp)
    P.pad_cols(wo, wop, dh, dhp)
    P.copy(wb, wf, lnpw, cs=LOG2E)
    P.launch(wq.device)
    return w3, b3, wvp, wop, wf


def _inference32_packs(params, gate_bias):
    """The inference forms, cached per parameter version as ``_inference_packs``: W q | k | g [3 W, D] and its bias (q in exp2
    units), Wv [W, D], Wo [D, W], Wf in log2 units."""

    def build():
        if torch.cuda.get_device_capability(params[0].device) == (10, 0):
            return (*_inference32_packs_b200(params, gate_bias), params)
        w4, b4, wop, wf = _pack32(params, gate_bias)
        _, dh, _, W, _, _ = _geometry32(params)
        qs = LOG2E / math.sqrt(dh)
        w3 = torch.cat([w4[:W] * qs, w4[W : 2 * W], w4[3 * W :]])
        b3 = torch.cat([b4[:W] * qs, b4[W : 2 * W], b4[3 * W :]])
        return w3, b3, w4[2 * W : 3 * W].contiguous(), wop, wf * LOG2E, params

    return _capture.lookup(
        _packs, ("f32", gate_bias, *((t.data_ptr(), t._version) for t in params)), build, limit=_capture.BLOCK_CACHE_LIMIT
    )[:5]


def _sm90():
    from foldjet_engine.kernels.augmented_attention import cuda as sm90

    return sm90


@opaque(fake=_inference_fake, name="apb_h100_inference_fp32")
def inference32(
    single: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps_s: float,
    eps_p: float,
    gate_bias: float = 0.0,
    extra: torch.Tensor | None = None,
    lp: int = 0,
) -> torch.Tensor:
    """single [B, L, D], pair [B, L, L, dp] fp32, mask [B, L] bool or None, leaves fp32 as ``_leaves`` -> single + APB(single).
    ``extra`` [B, H, L, L]: an additive logit term, added to the core's bias in its log2 units and key order (``key_perm``). B > 1:
    the one-row step per batch row, the packs made once (as ``inference``)."""
    B, L = single.shape[:2]
    H, _, dhp, W, D, dp = _geometry32(leaves)
    E = _rows().ext()
    if torch.cuda.get_device_capability(single.device) == (10, 0):
        return _inference32_b200(single, pair, mask, leaves, eps_s, eps_p, gate_bias, extra, lp)
    with torch.cuda.device(single.device):
        w3, b3, wv, wo, wf = _inference32_packs(leaves, gate_bias)
        out = torch.empty_like(single)
        km = kernel_mask(mask, L)  # each row's key blocks past its last attended key skipped by the TF32 core (the same bits)
        xa = single.new_empty((L, D))
        for b in range(B):
            x, y = single[b], out[b]
            E.ln_rows(x, leaves[0], leaves[1], xa, None, eps_s, y)
            qkg = torch.addmm(b3, xa, w3.t())
            vt = torch.mm(wv, xa.t())
            bias = x.new_empty((H, L, L))
            E.pair_bias32_fwd(pair[b].view(L * L, dp), wf, _row(mask, b), bias, None, None, L, eps_p, -math.inf)
            if extra is not None:
                bias.add_(extra[b].to(bias.dtype)[:, :, _key_perm32(L, single.device)[0]], alpha=LOG2E)
            _sm90().tf32_gated_inference(qkg, vt, bias, 0, 1, 2 * W, nh=H, dh=dhp, kmask=None if km is None else km[b : b + 1])
            y.addmm_(qkg[:, :W], wo.t())
        return out


_B200_TF32_CORES: dict = {}


def _b200_tf32_core(device_index: int, heads: int, dhp: int):
    """B200 (sm_100a): the token DiT's TF32 gated inference core (``augmented_attention/cuda/sm100`` ``GatedInferenceCore``, attn_inf_tf32.cu:
    q | k | g + v^T, logits in exp2 units, sigmoid(g) o written over q) built for this module's head layout (``-DNHEAD -DDHP``: 8 x 48,
    12 x 32, 16 x 32 (16 x 24 padded), 24 x 32 (24 x 16 padded), 16 x 32 at d_single 512 -- the H100 TF32 path's head widths) and
    ``-DKPERM=1``: it reads the bias in ``pair_bias32_fwd``'s key order (``key_perm``, a permutation within each group of 8 keys) and adds
    each value to its own key's logit, so k and v stay in natural order."""
    key = (device_index, heads, dhp)
    core = _B200_TF32_CORES.get(key)
    if core is None:
        from foldjet_engine.kernels._device import num_sms

        sm = _sm100()
        core = sm.GatedInferenceCore.__new__(sm.GatedInferenceCore)  # the class asserts the token DiT layouts; the kernel takes these too
        core.fp32, core.heads, core.dh = True, heads, dhp
        defs = (f"NHEAD={heads}", f"DHP={dhp}", "KPERM=1")
        core.k = sm._sm100_kernel("attn_inf_tf32", "augattn_inf_tf32_sm100", device_index, defs=defs)
        core.nsm, core.runs = num_sms(device_index), sm.BoundedRuns()
        # (2026-11-17, part 6) the attributes the class now reads: its device, and no -DQPAIR TF32 build (that one has no -DKPERM key order)
        core.device_index, core.tf32_qpair = device_index, False  # pyright: ignore[reportAttributeAccessIssue]
        _B200_TF32_CORES[key] = core
    return core


def _tc32(heads: int, dp: int, L: int) -> bool:
    """fp32 inference on B200: the pair pass on the TF32 tensor cores (``pair_bias32_tc``) where it is the faster one (measured alone, L 128 /
    512, 2026-11-17: 16 x 128 86 -> 47 us, 16 x 512 244 -> 135 us, 24 x 128 148 -> 51 us at L 512; at L 128 its fixed part makes it level or
    slower, and 8 heads' fp32 pass is short: 8 x 384 106 -> 125 us at L 512)."""
    return L % 128 == 0 and L >= 256 and (heads > 8 or (dp < 384 and L >= 384))


#: B200 fp32 inference (2026-11-17, part 6): the pair pass on tcgen05 kind::tf32 (``sm100.apb_pair_fwd32``). FOLDJET_ENGINE_APB_B200_PF32=0 / 1
#: forces off / on (L % 128 == 0, d_pair % 64 == 0).
_PF32 = os.environ.get("FOLDJET_ENGINE_APB_B200_PF32")


def _pf32(heads: int, dp: int, L: int) -> bool:
    """Whether the fp32 inference pair pass runs on tcgen05 (``pair_fwd32``)."""
    if L % 128 or dp % 64:
        return False
    if _PF32 is not None:
        return _PF32 != "0"
    # measured (2026-11-17, A / B, fp32 eval step, FMA or mma.sync TF32 pass -> tcgen05): 8 x 384 L 256 / 384 / 512 / 768 64.5 / 98.3 / 145.4 /
    # 264.2 -> 55.3 / 83.9 / 120.8 / 206.8 us; at L 512 every layout: 16 x 128 84.0 -> 73.7, 12 x 128 79.9 -> 71.7, 24 x 128 92.2 -> 75.8,
    # 16 x 256 116.7 -> 96.3, 16 x 384 167.9 -> 118.9, 16 x 512 172.0 -> 143.4
    return (256 if heads == 8 and dp == 384 else 512) <= L


def _planes_fix(E, bias: torch.Tensor, extra_b, L0: int, L: int, perm: bool) -> torch.Tensor:
    """B200: the bias planes [H, L0, L0] -> [H, L, L] (in place when L == L0) with ``extra_b`` ([H, L0, L0], a stride-0 head axis welcome;
    natural units and key order) added in the planes' exp2 units and key order (``perm``: the TF32 core's), the pad keys ``NEG``, the pad
    query rows 0 -- one launch (``planes_fix``, apb_rows_sm100.cu)."""
    ex = None
    if extra_b is not None:
        ex = extra_b[:1] if extra_b.stride(0) == 0 else extra_b
        if ex.dtype not in (torch.float32, torch.bfloat16):
            ex = ex.float()
        ex = ex.contiguous()
    dst = bias if L == L0 else bias.new_empty((bias.shape[0], L, L))
    E.planes_fix(bias, dst, ex, L0, L, LOG2E, NEG, perm)
    return dst


def _inference32_b200(single, pair, mask, leaves, eps_s, eps_p, gate_bias, extra, lp=None):
    """``inference32`` on B200: the same packs, LN rows and fp32 pair pass (``pair_bias32_fwd``, keys in the H100 TF32 core's ``key_perm``
    order), masked keys ``NEG`` (finite), then the B200 TF32 gated core built ``-DKPERM=1`` (``_b200_tf32_core``: the bias read in that key
    order, k and v^T in natural order). The B rows share one launch per row-wise stage (LN rows, the q | k | g GEMM, the v^T GEMM batched
    over the rows, to_out onto the residual seeds); the pair pass and the core run per row (each row its own bias planes).
    B = 1 (2026-11-17): the packs and the pair pass (+ ``planes_fix``: extra_pair_bias, the pad to ``lp``) on the side stream beside LN and
    the GEMMs, as ``_inference_b200``."""
    B, L0 = single.shape[:2]
    L = lp or L0  # the core's length; the pair pass runs at L0
    if L != L0:
        single = torch.nn.functional.pad(single, (0, 0, 0, L - L0))
    H, _, dhp, W, D, dp = _geometry32(leaves)
    E = _rows().ext()
    dev = single.device
    with torch.cuda.device(dev):
        main = torch.cuda.current_stream(dev)
        split = B == 1 and _STREAMS
        side = _side_stream(dev) if split else main
        if split:
            _join(side, main)
        out = torch.empty_like(single)
        x2, y2 = single.view(B * L, D), out.view(B * L, D)
        xa = single.new_empty((B * L, D))
        bias_p = single.new_empty((H, L0, L0))
        with torch.cuda.stream(side):
            # as ``_inference_b200``: a capture reads its warm-up's pack, no launch per replay
            if _EAGER_PACKS and not torch.is_grad_enabled():
                w3, b3, wv, wo, wf = _capture.param_forms(
                    ("apb_b200_inference32", float(gate_bias)),
                    leaves,
                    lambda: _inference32_packs_b200(leaves, gate_bias),
                    capture_reads_eager=True,
                )
            else:
                w3, b3, wv, wo, wf = _inference32_packs(leaves, gate_bias)
        packed = torch.cuda.Event()
        if split:
            _rec(packed, side)
        core = _b200_tf32_core(dev.index if dev.index is not None else torch.cuda.current_device(), H, dhp)
        E.ln_rows(x2, leaves[0], leaves[1], xa, None, eps_s, y2)
        if split:
            _after(main, packed)
        if split:  # the v^T GEMM beside the q | k | g GEMM (a third stream)
            s2 = _side_stream(dev, 1)
            _join(s2, main)
            with torch.cuda.stream(s2):
                vt = torch.mm(wv, xa.t())[None]  # [1, W, L]
        qkg = torch.addmm(b3, xa, w3.t())  # [B L, 3 W]
        if split:
            _join(main, s2)
            vt.record_stream(main)
        elif B == 1:
            vt = torch.mm(wv, xa.t())[None]  # [1, W, L]
        else:
            vt = torch.matmul(wv, xa.view(B, L, D).transpose(1, 2))  # [B, W, L]: each row's v^T
        tc = _tc32(H, dp, L0)
        pf32 = _pf32(H, dp, L0)
        for b in range(B):
            with torch.cuda.stream(side):
                if pf32:  # tcgen05 kind::tf32 (pair_fwd32.cu), in the core's key_perm order
                    _sm100().apb_pair_fwd32(pair[b].view(L0 * L0, dp), wf, _row(mask, b), bias_p, L0, eps_p, NEG, True)
                elif tc:  # TF32 MMAs (pair_bias32_tc), else the fp32 FMA pass; both in the core's key_perm order
                    E.pair_bias32_tc(pair[b].view(L0 * L0, dp), wf, _row(mask, b), bias_p, None, L0, eps_p, NEG, True)
                else:
                    E.pair_bias32_fwd(pair[b].view(L0 * L0, dp), wf, _row(mask, b), bias_p, None, None, L0, eps_p, NEG)
                planes = bias_p
                if extra is not None or L != L0:
                    planes = _planes_fix(E, bias_p, None if extra is None else extra[b], L0, L, True)
            if split:
                _join(main, side)
                planes.record_stream(main)
            core(qkg[b * L : (b + 1) * L], planes, 0, 1, vt=vt[b])
        y2.addmm_(qkg[:, :W], wo.t())
        if split:
            for t in (w3, b3, wv, wo):
                t.record_stream(main)
            for t in (pair, wf, bias_p) + (() if mask is None else (mask,)) + (() if extra is None else (extra,)):
                t.record_stream(side)
        return out if L == L0 else out[:, :L0].contiguous()


def _forward32_fake(single, pair, mask, leaves, eps_s, eps_p, gate_bias=0.0):
    """y and the saved set, per batch row: xa, (mean, rstd), qkvg, qs, k, v, O, og, LSE, bias (key-permuted) and its query-permuted
    transpose, pair stats (the packs are remade by the backward)."""
    B, L = single.shape[:2]
    H, _, _, W, D, _ = _geometry32(leaves)
    e = single.new_empty
    row = [(L, D), (L, 2), (L, 4 * W), (L, W), (L, W), (L, W), (L, W), (L, W), (1, H, L), (H, L, L), (H, L, L), (L * L, 2)]
    return [torch.empty_like(single)] + [e(s) for _ in range(B) for s in row]


#: saved tensors per batch row of the fp32 training forward (``_forward32_fake``)
_ROW_SAVED32 = 12


@opaque(fake=_forward32_fake, name="apb_h100_train_fwd_fp32")
def forward32(
    single: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps_s: float,
    eps_p: float,
    gate_bias: float = 0.0,
) -> list[torch.Tensor]:
    """single [B, L, D], pair [B, L, L, dp] fp32 -> [y, *row 0's saved, *row 1's saved, ...]: the one-row forward per batch row."""
    B, L = single.shape[:2]
    H, dh, dhp, W, _, dp = _geometry32(leaves)
    E = _rows().ext()
    saved = []
    with torch.cuda.device(single.device):
        w4, b4, wop, wf = _train32_packs(leaves, gate_bias)
        out = torch.empty_like(single)
        for b in range(B):
            x, y = single[b], out[b]
            xa = torch.empty_like(x)
            st = x.new_empty((L, 2))
            E.ln_rows(x, leaves[0], leaves[1], xa, st, eps_s, y)
            qkvg = torch.addmm(b4, xa, w4.t())
            qs = qkvg[:, :W] * (LOG2E / math.sqrt(dh))  # exp2 units (contiguous)
            k = qkvg[:, W : 2 * W].contiguous()
            v = qkvg[:, 2 * W : 3 * W].contiguous()
            bias, bias_tp = x.new_empty((H, L, L)), x.new_empty((H, L, L))
            stp = x.new_empty((L * L, 2))
            E.pair_bias32_fwd(pair[b].view(L * L, dp), wf * LOG2E, _row(mask, b), bias, stp, bias_tp, L, eps_p, -math.inf)
            O, lse = _sm90().tf32_forward(qs, k, v.t().contiguous(), bias, None, 1, L, nh=H, dh=dhp)
            og = qkvg[:, 3 * W :].sigmoid_() * O  # the g columns keep s = sigmoid(g) for the backward (q | k | v are saved separately)
            y.addmm_(og, wop.t())
            saved += [xa, st, qkvg, qs, k, v, O, og, lse, bias, bias_tp, stp]
    return [out, *saved]


def _backward32_fake(single, pair, mask, leaves, eps_s, eps_p, saved, dy, gate_bias=0.0):
    return [torch.empty_like(single), torch.empty_like(pair), *(torch.empty_like(t) for t in leaves)]


@opaque(fake=_backward32_fake, name="apb_h100_train_bwd_fp32")
def backward32(
    single: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps_s: float,
    eps_p: float,
    saved: list[torch.Tensor],
    dy: torch.Tensor,
    gate_bias: float = 0.0,
) -> list[torch.Tensor]:
    """The one-row backward per batch row: dx / dpair into their rows, the parameter gradients summed over the rows in row order."""
    B, L = single.shape[:2]
    H, dh, dhp, W, D, dp = _geometry32(leaves)
    rows = _rows()
    E = rows.ext()
    lnw, lnpw, wb = leaves[0], leaves[8], leaves[9]
    total: list[torch.Tensor] = []
    with torch.cuda.device(single.device):
        w4, _, wop, wf = _train32_packs(leaves, gate_bias)
        dyc = dy.contiguous()
        dx_all, dz_all = torch.empty_like(single), torch.empty_like(pair)
        perm, inv = _key_perm32(L, single.device)
        for b in range(B):
            xa, st, qkvg, qs, k, v, O, og, lse, bias, bias_tp, stp = saved[b * _ROW_SAVED32 : (b + 1) * _ROW_SAVED32]
            x = single[b]
            z = pair[b].view(L * L, dp)
            dy2 = dyc[b]
            dog = dy2 @ wop
            dwo = dy2.t() @ og
            dcat =x.new_empty((L, 4 * W))  # dq | dk | dv (the core writes them in place) | dg
            # dO = dog s; dd[h] = sum_d (dog O) s; the gate gradient (dog O) s (1 - s): one kernel (was eight launches)
            dO = torch.empty_like(dog)
            dd = x.new_empty((1, H, L))
            E.gate32_bwd(dog, O, qkvg, dO, dd, dcat, dhp)
            dq, _, _, db_p = _sm90().tf32_backward(
                qs,
                k,
                v,
                dO,
                qs.t().contiguous(),
                k.t().contiguous(),
                dO.t().contiguous(),
                bias,
                bias_tp,
                lse,
                dd,
                lse[..., perm].contiguous(),
                dd[..., perm].contiguous(),
                None,
                None,
                1,
                L,
                nh=H,
                dh=dhp,
                dpack=dcat,
            )
            if dhp != dh:  # the core's dQ carries 1 / sqrt(dhp); the logits were scaled by 1 / sqrt(dh)
                dq.mul_(math.sqrt(dhp / dh))
            dxa = dcat @ w4
            dw4 = dcat.t() @ xa
            acc = x.new_empty(((L + rows.RW - 1) // rows.RW, 2 * D))
            E.ln_bwd(dxa, x, st, lnw, dy2, dx_all[b], acc)
            acc = acc.sum(0)
            db = db_p[..., inv].contiguous()
            E.pair_bias32_bwd(z, stp, db, wf, dz_all[b].view(L * L, dp), L)
            a = db.view(H, L * L) * stp[:, 1]
            dwf = a @ z - (a @ stp[:, 0])[:, None]
            grads = [
                acc[:D],
                acc[D:],
                _unpad_rows(dw4[:W], H, dh, dhp),
                _unpad_rows(dq.sum(0)[:, None], H, dh, dhp)[:, 0],
                _unpad_rows(dw4[W : 2 * W], H, dh, dhp),
                _unpad_rows(dw4[2 * W : 3 * W], H, dh, dhp),
                _unpad_rows(dw4[3 * W :], H, dh, dhp),
                _unpad_rows(dwo.t(), H, dh, dhp).t(),
                (dwf * wb).sum(0),
                dwf * lnpw,
            ]
            # custom-op outputs may not alias each other or the inputs: each gradient its own contiguous tensor
            grads = [g.contiguous().clone() if g._base is not None else g.contiguous() for g in grads]
            if b == 0:
                total = grads
            else:
                torch._foreach_add_(total, grads)  # pyright: ignore[reportPrivateImportUsage]  -- torch internal, on purpose
    return [dx_all, dz_all, *total]


# ---------------------------------------------------------------------------------------------- B200 float32 activations (TF32)
# The same op on B200: the fp32 rows / pair pass of apb_rows.cu (sm_100 build; the pair bias un-permuted to natural key order) and the
# sm_100a TF32 training cores of ``augmented_attention/cuda/sm100`` (attn_fwd_tf32 / attn_dkv_tf32 / attn_dqb_tf32) built for this
# module's head layout (``_b200_tf32_defs``: -DNHEAD, -DDHP 48 or 32, -DRSQDV of the real head width). Those cores pair the samples:
# a batch row runs as A = 2 with a zero second sample (zero q / k / v / dO: its dbias contribution is exactly 0, its rows dropped).
# The bias in natural units (masked keys NEG), q / k / v unscaled (the cores apply 1 / sqrt(dh)), dQ / dK / dV / dbias in natural units.
def _rna(t: torch.Tensor, cols: int) -> None:
    """Round the first ``cols`` columns of every row of the fp32 ``t`` to the TF32 grid, to nearest, in place (``conditioned_transition.cuda
    .rna_rows``): the sm_100a TF32 cores truncate their operands, which biases the sums (the gradients 4-5x the fp32 reference's error
    without it, measured on B200)."""
    from foldjet_engine.kernels.conditioned_transition.cuda import rna_rows

    rna_rows(t, cols)


def _b200_tf32_defs(H: int, dh: int, dhp: int) -> tuple[str, ...]:
    # DET=1 (attn_dkv_tf32): dK / dV accumulated in block order, bitwise repeatable (the other two cores do not read it)
    return (f"NHEAD={H}", f"DHP={dhp}", f"RSQDV={dh**-0.5!r}f", "DET=1")


#: saved tensors per batch row of the B200 fp32 training forward (``_forward32_b200_fake``)
_ROW_SAVED32_B200 = 9


def _forward32_b200_fake(single, pair, mask, leaves, eps_s, eps_p, gate_bias=0.0):
    """y and, per batch row: xa, (mean, rstd), qkvg (g columns: sigmoid(g)), q | k | v of both samples [2 L, 3 W], O [2 L, W], og,
    LSE [2, H, L], bias [H, L, L] (natural keys), pair stats."""
    B, L = single.shape[:2]
    H, _, _, W, D, _ = _geometry32(leaves)
    e = single.new_empty
    row = [(L, D), (L, 2), (L, 4 * W), (2 * L, 3 * W), (2 * L, W), (L, W), (2, H, L), (H, L, L), (L * L, 2)]
    return [torch.empty_like(single)] + [e(s) for _ in range(B) for s in row]


@opaque(fake=_forward32_b200_fake, name="apb_b200_train_fwd_fp32")
def forward32_b200(
    single: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps_s: float,
    eps_p: float,
    gate_bias: float = 0.0,
) -> list[torch.Tensor]:
    """``forward32`` on B200 (the section's comment): [y, *row 0's saved, *row 1's saved, ...]."""
    B, L = single.shape[:2]
    H, dh, dhp, W, _, dp = _geometry32(leaves)
    E = _rows().ext()
    sm = _sm100()
    defs = _b200_tf32_defs(H, dh, dhp)
    saved = []
    with torch.cuda.device(single.device):
        w4, b4, wop, wf = _train32_packs(leaves, gate_bias)
        out = torch.empty_like(single)
        main, side = torch.cuda.current_stream(single.device), _side_stream(single.device)
        made: list[torch.Tensor] = []
        for b in range(B):
            # the pair pass (TF32 tensor cores, natural key order, the row stats for the backward) on the side stream, beside LN / GEMM
            _join(side, main)
            with torch.cuda.stream(side):
                bias, stp = single.new_empty((H, L, L)), single.new_empty((L * L, 2))
                E.pair_bias32_tc(pair[b].view(L * L, dp), wf, _row(mask, b), bias, stp, L, eps_p, NEG, False)
            made += [bias, stp]
            x, y = single[b], out[b]
            xa = torch.empty_like(x)
            st = x.new_empty((L, 2))
            E.ln_rows(x, leaves[0], leaves[1], xa, st, eps_s, y)
            qkvg = torch.addmm(b4, xa, w4.t())
            qkv2 = x.new_empty((2 * L, 3 * W))  # sample 0: this row's q | k | v rounded to TF32 to nearest; sample 1: zeros (one launch)
            E.pack2_rna(qkvg, qkv2, L)
            _join(main, side)
            O, lse = sm.forward_tf32(qkv2[:, :W], qkv2[:, W : 2 * W], qkv2[:, 2 * W :], bias, 2, L, H, dhp, defs=defs)
            og = x.new_empty((L, W))
            E.gate32(qkvg, O, og, L)  # the g columns keep s = sigmoid(g) for the backward
            y.addmm_(og, wop.t())
            saved += [xa, st, qkvg, qkv2, O, og, lse, bias, stp]
        for t in made:
            t.record_stream(main)
        for t in (pair, wf) + (() if mask is None else (mask,)):
            t.record_stream(side)
    return [out, *saved]


def _backward32_b200_fake(single, pair, mask, leaves, eps_s, eps_p, saved, dy, gate_bias=0.0):
    return [torch.empty_like(single), torch.empty_like(pair), *(torch.empty_like(t) for t in leaves)]


@opaque(fake=_backward32_b200_fake, name="apb_b200_train_bwd_fp32")
def backward32_b200(
    single: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps_s: float,
    eps_p: float,
    saved: list[torch.Tensor],
    dy: torch.Tensor,
    gate_bias: float = 0.0,
) -> list[torch.Tensor]:
    """``backward32`` on B200: dx / dpair into their rows, the parameter gradients summed over the rows in row order."""
    B, L = single.shape[:2]
    H, dh, dhp, W, D, dp = _geometry32(leaves)
    rows = _rows()
    E = rows.ext()
    sm = _sm100()
    defs = _b200_tf32_defs(H, dh, dhp)
    lnw, lnpw, wb = leaves[0], leaves[8], leaves[9]
    total: list[torch.Tensor] = []
    with torch.cuda.device(single.device):
        w4, _, wop, wf = _train32_packs(leaves, gate_bias)
        dyc = dy.contiguous()
        dx_all, dz_all = torch.empty_like(single), torch.empty_like(pair)
        main, s1, s2 = torch.cuda.current_stream(single.device), _side_stream(single.device, 0), _side_stream(single.device, 1)
        for b in range(B):
            xa, st, qkvg, qkv2, O, og, lse, bias, stp = saved[b * _ROW_SAVED32_B200 : (b + 1) * _ROW_SAVED32_B200]
            x = single[b]
            z = pair[b].view(L * L, dp)
            dy2 = dyc[b]
            _join(s1, main)
            with torch.cuda.stream(s1):
                dwo = dy2.t() @ og
            dog = dy2 @ wop
            # dq | dk | dv | dg of both samples (the cores write dq, dk, dv into their column blocks; sample 1's rows are dropped)
            dcat = x.new_empty((2 * L, 4 * W))
            do2, dd2 = x.new_empty((2 * L, W)), x.new_empty((2, H, L))
            E.gate32_bwd(dog, O, qkvg, do2, dd2, dcat[:L, 3 * W :], L, H, dhp)  # dO (rounded) and D of sample 0, sample 1 zero; dg
            dqp, _, _, db = sm.backward_tf32(
                qkv2[:, :W],
                qkv2[:, W : 2 * W],
                qkv2[:, 2 * W :],
                do2,
                bias,
                lse,
                dd2,
                2,
                L,
                H,
                dhp,
                dq=dcat[:, :W],
                dk=dcat[:, W : 2 * W],
                dv=dcat[:, 2 * W : 3 * W],
                defs=defs,
                dq_parts=True,
            )
            # dQ: the 64-key chunks' slices added in chunk order (attn_dqb_tf32 -DDQPART=1 -DDET=1: bitwise repeatable; the reduce-adds
            # arrived in any order, ~2e-5 relative run to run from L 384 on)
            torch.sum(dqp[:, :L], dim=0, out=dcat[:L, :W])
            dc = dcat[:L]
            # the pair backward and dWf on the side streams (they need only dbias), beside dxa / the LayerNorm backward / dW q|k|v|g
            _join(s1, main)
            _join(s2, main)
            with torch.cuda.stream(s2):
                E.pair_bias32_bwd(z, stp, db, wf, dz_all[b].view(L * L, dp), L)
            with torch.cuda.stream(s1):
                a = db.view(H, L * L) * stp[:, 1]
                dwf = a @ z - (a @ stp[:, 0])[:, None]
                dw4 = dc.t() @ xa
            dxa = dc @ w4
            acc = x.new_empty(((L + rows.RW - 1) // rows.RW, 2 * D))
            E.ln_bwd(dxa, x, st, lnw, dy2, dx_all[b], acc)
            acc = acc.sum(0)
            _join(main, s1)
            _join(main, s2)
            for t in (dwo, a, dwf, dw4):
                t.record_stream(main)
            grads = [
                acc[:D],
                acc[D:],
                _unpad_rows(dw4[:W], H, dh, dhp),
                _unpad_rows(dc[:, :W].sum(0)[:, None], H, dh, dhp)[:, 0],
                _unpad_rows(dw4[W : 2 * W], H, dh, dhp),
                _unpad_rows(dw4[2 * W : 3 * W], H, dh, dhp),
                _unpad_rows(dw4[3 * W :], H, dh, dhp),
                _unpad_rows(dwo.t(), H, dh, dhp).t(),
                (dwf * wb).sum(0),
                dwf * lnpw,
            ]
            grads = [g.contiguous().clone() if g._base is not None else g.contiguous() for g in grads]
            if b == 0:
                total = grads
            else:
                torch._foreach_add_(total, grads)  # pyright: ignore[reportPrivateImportUsage]  -- torch internal, on purpose
    return [dx_all, dz_all, *total]


class _Training32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eps_s, eps_p, gate_bias, mask, single, pair, *leaves):
        ctx.b200 = torch.cuda.get_device_capability(single.device) == (10, 0)
        fwd = forward32_b200 if ctx.b200 else forward32
        y, *saved = fwd(single, pair, mask, list(leaves), eps_s, eps_p, gate_bias)
        ctx.eps, ctx.gate_bias = (eps_s, eps_p), gate_bias
        ctx.n = len(leaves)
        ctx.save_for_backward(mask, single, pair, *leaves, *saved)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        v = ctx.saved_tensors
        mask, single, pair = v[0], v[1], v[2]
        leaves, saved = list(v[3 : 3 + ctx.n]), list(v[3 + ctx.n :])
        eps_s, eps_p = ctx.eps
        bwd = backward32_b200 if ctx.b200 else backward32
        dx, dz, *grads = bwd(single, pair, mask, leaves, eps_s, eps_p, saved, dy, ctx.gate_bias)
        return (None, None, None, None, dx, dz, *grads)
