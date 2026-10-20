"""Automatic dispatch to the B200 (sm_100a) AttentionPairBias: inference and training, all hand CUDA but the plain GEMMs
(cuBLAS).

H100 inference and training use native CUDA by default for the required buckets.
The B200 pair-weight (Wb) gradient fails the FP32 oracle cosine at the required
buckets, so H100 training reconstructs that FP32 rounding with native CUDA and
cuBLAS (ln_pair has no beta, so only dWb is repaired). This accuracy-first shadow
materializes extra FP32 workspaces: the B200 tensor lifetimes and former fast-path
latency do not describe this training default.

Contract: bf16 activations, d_single 384 as 8 heads x 48, 12 x 32, 24 x 16 or 16 x 24 (AF3's Pairformer single attention),
or d_single 512 as 16 x 32, d_pair
128, no QK-norm, ``implementation="miniworld"`` (engine backend not forced to Triton), sm_100. Inference: L a multiple of 16;
training: L a multiple of 128. B200: any number of samples B (each its own pair and key mask; up to ``max_batch()`` per launch,
more in fixed chunks of that size); H100: B = 1.

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
zero attention update for an all-false key mask; its dWb shadow also applies
masked_fill's zero derivative at every masked key."""

from __future__ import annotations

import functools
import math

import torch
from torch.autograd.function import once_differentiable

from miniworld_engine import settings
from miniworld_engine.kernels import _capture, _cast_cuda
from miniworld_engine.kernels._device import num_sms
from miniworld_engine.kernels._compile import device_constant, opaque
from miniworld_engine.modules.exceptions import ImplementationType

DP = 128
WIDE_DP, WIDE_HEADS = 256, 16  # Chai-1: d_pair 256 with 16 x 24 heads (H100 inference)
SHAPES = (
    (8, 384),
    (12, 384),
    (16, 384),
    (24, 384),
    (16, 512),
)  # (heads, d_single): 8 x 48, 12 x 32, 16 x 24 (padded to 32), 24 x 16, 16 x 32
LOG2E = math.log2(math.e)
#: B200: samples per launch. Each sample is computed exactly as alone (every kernel's work items are per sample, the key split
#: depends on L only), so a call is split into chunks of this many samples, in order, without changing a sample's bits.
MAX_BATCH = 48


def max_batch() -> int:
    """B200 samples per launch (``MAX_BATCH``); a call with more samples runs as consecutive chunks of this size."""
    return MAX_BATCH

NEG = -1e30  # masked keys: finite, so a fully masked sample stays NaN-free
_packs: dict = {}
_cores: dict = {}


@functools.lru_cache(maxsize=16)
def _capability_of(index: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(index)


def _cap(device) -> tuple[int, int]:
    """The device's compute capability, cached per device in eager calls (``torch.cuda.get_device_capability`` costs ~8 us of
    host time, and a training step used to ask a dozen times); traced as the plain query under torch.compile."""
    device = torch.device(device)
    if torch.compiler.is_compiling() or device.type != "cuda":
        return torch.cuda.get_device_capability(device)
    return _capability_of(device.index if device.index is not None else torch.cuda.current_device())


def _nsm(device) -> int:
    device = torch.device(device)
    return num_sms(device.index if device.index is not None else torch.cuda.current_device())


def _host(device) -> bool:
    """B200: the ops' C++ launch sequences (``apb_b200::*``, ``settings.apb_b200_host``) instead of the Python one."""
    return _cap(device) == (10, 0) and settings.current().apb_b200_host


def _eligible(
    module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None
) -> bool:
    if (
        module.implementation != ImplementationType.MINIWORLD
        or settings.current().force_portable_kernels
    ):
        return False
    H, D = module.n_head, single.shape[-1]
    if (
        module.use_qk_norm
        or (H, D) not in SHAPES
        or tuple(module.to_query.weight.shape) != (D, D)
    ):
        return False
    # Chai-1's pairformer single attention (16 x 24, d_pair 256, gate bias 1.0) is served on H100 for inference only: the gate bias
    # and the 256-wide pair are the H100 inference op's (module docstring); every other call keeps d_pair 128 and no gate bias.
    dp = module.to_bias.weight.shape[1]
    hopper = torch.cuda.get_device_capability(single.device) == (9, 0)
    if dp == WIDE_DP:
        if not (hopper and H == WIDE_HEADS and not torch.is_grad_enabled()):
            return False
    elif dp != DP or (module.gate_bias and (not hopper or torch.is_grad_enabled())):
        return False
    if (
        tuple(module.to_bias.weight.shape) != (H, dp)
        or module.ln_single.weight is None
        or module.ln_pair.weight is None
    ):
        return False
    if (
        not single.is_cuda
        or single.dtype != torch.bfloat16
        or pair.dtype != torch.bfloat16
    ):
        return False
    if single.ndim != 3 or single.shape[2] != D or not single.is_contiguous():
        return False
    B, L = single.shape[0], single.shape[1]
    if B < 1 or tuple(pair.shape) != (B, L, L, dp) or not pair.is_contiguous():
        return False
    if mask is not None and tuple(mask.shape) != (B, L):
        return False
    capability = _cap(single.device)
    if B != 1 and capability != (10, 0):
        return False
    if capability == (10, 0):
        return not settings.current().apb_b200_host or _host_ready()
    return (
        capability == (9, 0)
        and L % 128 == 0
    )


@device_constant
def _host_ready() -> bool:
    """Loads the B200 host-path ops (``apb_b200::*``) once; a constant to torch.compile (evaluated at trace time), so the ops
    exist before the traced code names them."""
    return _rows().host_ops()


def serves_inference(
    module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None = None
) -> bool:
    return (
        not torch.is_grad_enabled()
        and single.shape[1] % 16 == 0
        and _eligible(module, single, pair, mask)
    )


def serves_train(
    module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None = None
) -> bool:
    return (
        torch.is_grad_enabled()
        and single.shape[1] % 128 == 0
        and _eligible(module, single, pair, mask)
    )


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
    if _cap(lnw.device) == (9, 0):
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
    from miniworld_engine.kernels.augmented_attention.cuda import apb as rows

    return rows


def _sm100():
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    return sm100


def _training_core(device):
    if _cap(device) == (9, 0):
        from miniworld_engine.kernels.augmented_attention import cuda as sm90

        return sm90
    return _sm100()


def _inference_core(device, heads: int, d: int):
    if _cap(device) == (9, 0):
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
def _inference_packs(params, leaves, gate_bias: float = 0.0):
    """``_prep`` with q scaled into exp2 units and Wf in exp2 units, cached per parameter version (the step reads them every
    call; repacking costs a launch). Keyed on the parameters themselves (``params``), not on ``leaves``, their kernel-dtype casts,
    which are fresh tensors per call for fp32 parameters. Scoped to the CUDA-graph capture (``kernels._capture``): a capture packs
    once, recorded, so each replay packs the weights as they are then."""
    build = lambda: (
        *_prep(leaves, LOG2E / math.sqrt(_geometry(leaves)[1]), LOG2E, gate_bias),
        params,
    )
    return _capture.lookup(
        _packs,
        (gate_bias, *((t.data_ptr(), t._version) for t in params)),
        build,
        limit=_capture.BLOCK_CACHE_LIMIT,
    )[:5]


def _prep(leaves, qs: float, ws: float, gate_bias: float = 0.0):
    """(W q | k | v | g [4 W, D] (padded rows, q x qs), its bias [4 W], Wf = to_bias.weight * ln_pair.weight * ws [H, 128], Wo
    [D, W] bf16 with padded columns -- or None when there is no padding (8 x 48: to_out.weight as is) --, Wb bf16 [H, 128] or
    None), bf16, one launch. Hopper always packs Wo and Wb (its weights may be an fp32 master that this pass rounds)."""
    _, _, wq, bq, wk, wv, wg, wo, lnpw, wb = leaves
    H, _, W, D = _geometry(leaves)
    hopper = _cap(wq.device) == (9, 0)
    bf = torch.bfloat16
    wpack = wq.new_empty((4 * W, D), dtype=bf)
    bvec = wq.new_empty((4 * W,), dtype=bf)
    wf = wq.new_empty((H, DP), dtype=bf)
    wop = wq.new_empty((D, W), dtype=bf) if (W != D or hopper) else None
    wbo = wq.new_empty((H, DP), dtype=bf) if hopper else None
    wide = wb.shape[1] == WIDE_DP
    if wide:  # the prep pass is 128 wide: it packs everything but Wf, which is made below from the full 256 columns
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
    if gate_bias:  # sigmoid(g + c): the constant rides on the gate rows of the packed bias (q | k | v | g, W each)
        bvec[3 * W :] += gate_bias
    return wpack, bvec, wf, wop, wbo


def _inference_fake(single, pair, mask, leaves, eps_s, eps_p, gate_bias=0.0):
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
) -> torch.Tensor:
    """single [S, L, d], pair [S, L, L, 128] bf16, mask [S L] bool or None, leaves as ``_leaves`` -> single + APB(single)."""
    NB, L = single.shape[0], single.shape[1]
    rows, dev = _rows(), single.device
    params = leaves
    leaves = _capture.param_forms(
        "apb_leaves", params, lambda: tuple(_kernel_leaves(params))
    )
    lnw, lnb = leaves[0], leaves[1]
    H, _, W, D = _geometry(leaves)
    with torch.cuda.device(dev):
        x = single.view(NB * L, D)
        wpack, bvec, wf, wop, _ = _inference_packs(params, leaves, gate_bias)
        wo = leaves[7] if wop is None else wop
        if _host(dev):  # one C++ call: the same launches without the Python sequence
            return torch.ops.apb_b200.infer(
                single, pair, mask, [lnw, lnb, wpack, bvec, wf, wo], eps_s, eps_p, _sm100().apb_ksplit(L, H, D)
            )
        xa = torch.empty_like(x)
        y = torch.empty_like(
            x
        )  # residual seed, written by ln_rows; to_out accumulates onto it
        rows.ext().ln_rows(x, lnw, lnb, xa, None, eps_s, y)
        qkvg = torch.addmm(bvec, xa, wpack.t())
        neg = -math.inf if _cap(dev) == (9, 0) else NEG
        bias = rows.pair_bias(pair.view(NB * L * L, wf.shape[1]), wf, mask, L, eps_p, neg)
        core = _cores.get((dev.index, H, D))
        if core is None:
            core = _cores[(dev.index, H, D)] = _inference_core(dev, H, D)
        if _cap(dev) == (9, 0):
            core(qkvg, bias)
        else:
            core(qkvg, bias, L)
        return y.addmm_(qkvg[:, :W], wo.t()).view(NB, L, D)


def _chunks(single, pair, mask):
    """B200 calls with more than ``max_batch()`` samples: consecutive chunks of that many (the last one shorter)."""
    n = max_batch()
    for b in range(0, single.shape[0], n):
        yield single[b : b + n], pair[b : b + n], None if mask is None else mask[b : b + n]


def update_inference(
    module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    if single.shape[0] > max_batch():
        return torch.cat([update_inference(module, *c) for c in _chunks(single, pair, mask)])
    return inference(
        single,
        pair,
        _mask(mask, single.shape[0] * single.shape[1]),
        _leaves(module),
        module.ln_single.eps,
        module.ln_pair.eps,
        float(module.gate_bias),
    )


# ------------------------------------------------------------------------------------------------------------- training
def _forward_fake(single, pair, mask, leaves, eps_s, eps_p):
    """y and the saved set: xa, (mean, rstd), qkvg, bias, O, LSE, og, Wf, W q | k | v | g (and the padded Wo for 16 x 24)."""
    NB, L = single.shape[0], single.shape[1]
    M = NB * L
    H, _, W, D = _geometry(leaves)
    e, f32 = single.new_empty, torch.float32
    out = [
        torch.empty_like(single),
        e((M, D)),
        e((M, 2), dtype=f32),
        e((M, 4 * W)),
        e((NB * H, L, L)),
        e((M, W), dtype=f32),
        e((NB * H, L), dtype=f32),
        e((M, W)),
        e((H, DP)),
        e((4 * W, D)),
    ]
    hopper = _cap(single.device) == (9, 0)
    return (
        out
        + ([e((D, W))] if W != D or hopper else [])
        + ([e((H, DP))] if hopper else [])
    )


@opaque(fake=_forward_fake, name="apb_b200_train_fwd")
def forward(
    single: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps_s: float,
    eps_p: float,
) -> list[torch.Tensor]:
    NB, L = single.shape[0], single.shape[1]
    rows = _rows()
    lnw, lnb = leaves[0], leaves[1]
    H, _, W, D = _geometry(leaves)
    with torch.cuda.device(single.device):
        x = single.view(NB * L, D)
        wpack, bvec, wf, wop, wbo = _prep(leaves, 1.0, 1.0)
        wo = leaves[7] if wop is None else wop
        xa = torch.empty_like(x)
        st = x.new_empty((NB * L, 2), dtype=torch.float32)
        y = torch.empty_like(x)
        rows.ext().ln_rows(x, lnw, lnb, xa, st, eps_s, y)
        qkvg = torch.addmm(bvec, xa, wpack.t())
        neg = (
            -math.inf
            if _cap(single.device) == (9, 0)
            else NEG
        )
        bias = rows.pair_bias(pair.view(NB * L * L, DP), wf, mask, L, eps_p, neg)
        core = _training_core(single.device)
        if _cap(single.device) == (9, 0):
            # The gate is the attention epilogue: O (fp32, for the backward's D) and og = sigmoid(g) O in one pass.
            O, og, LSE = core.apb_forward_gated(qkvg, bias, L, H, D)
        else:
            # attn_fwd2 (+ the key split's combine) and the gate: O, LSE, og = sigmoid(g) O
            O, LSE, og = core.apb_forward_og(qkvg, bias, L, H, D)
        y = y.addmm_(og, wo.t()).view(
            NB, L, D
        )  # in place onto the residual seed: no copy of x into the output
    return (
        [y, xa, st, qkvg, bias, O, LSE, og, wf, wpack]
        + ([wop] if wop is not None else [])
        + ([wbo] if wbo is not None else [])
    )


def _backward_fake(single, pair, mask, leaves, eps_s, eps_p, saved, dy):
    """One gradient per input (its dtype) and per leaf (fp32), contiguous."""
    return [
        torch.empty_like(single),
        torch.empty_like(pair),
        *(torch.empty(t.shape, dtype=torch.float32, device=t.device) for t in leaves),
    ]


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
    NB, L = single.shape[0], single.shape[1]
    M = NB * L
    rows = _rows()
    E = rows.ext()
    lnw, lnpw = leaves[0], leaves[8]
    H, _, W, D = _geometry(leaves)
    xa, st, qkvg, bias, O, LSE, og, wf, wpack = saved[:9]
    # [D, W] bf16 and Wb bf16: packed by ``prep`` on Hopper (its leaves may be an fp32 master), the leaf itself otherwise
    wo = saved[9] if len(saved) > 9 else leaves[7]
    wb = saved[10] if len(saved) > 10 else leaves[9]
    f32 = torch.float32
    with torch.cuda.device(single.device):
        x = single.view(M, D)
        dy2 = dy.contiguous().view(M, D)
        # Every producer owns a block partial row; finalize reduces the rows each grid wrote, in block order (no zero fill, no
        # atomics: bitwise run to run). B200 caps the row kernels' grid (their blocks then walk several rows) at two per SM.
        rows_a = (M + rows.RW - 1) // rows.RW
        if _cap(x.device) != (9, 0):
            rows_a = min(rows_a, 2 * _nsm(x.device))
        rows_b = E.pair_bwd_blocks(x, H, L, NB)
        acc = torch.empty(max(rows_a, rows_b), rows.acc_n(H, D), device=x.device)
        dog = dy2 @ wo  # [L, W]: zero in the pads (Wo's pad columns are zero)
        dwo = torch.mm(
            dy2.t(), og, out_dtype=f32
        )  # [D, W]; finalize keeps the real columns
        dob = x.new_empty((M, W))
        dd = x.new_empty((H, L) if NB == 1 else (NB, H, L), dtype=f32)
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
        E.qkv_bwd(DQ, DK, DV, dqkvg, acc, D, rows_a)
        del DQ, DK, DV
        dxa = dqkvg @ wpack
        dwp = torch.mm(dqkvg.t(), xa, out_dtype=f32)
        dx = torch.empty_like(x)
        E.ln_bwd(dxa, x, st, lnw, dy2, dx, acc, rows_a)
        dz, _, _ = rows.pair_bias_bwd(pair.view(M * L, DP), DB, wf, L, eps_p, acc, D)
        # custom-op outputs may not alias: every parameter gradient leaves as its own tensor, written by one kernel
        outs = [torch.empty(t.shape, dtype=f32, device=t.device) for t in leaves]
        E.finalize(dwp, dwo, acc, wb, lnpw, outs, rows_a, rows_b)
    return [dx.view(single.shape), dz.view(pair.shape), *outs]


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
            else (
                E.exact_ln_bf16(pair, lnpw, eps_p, pair_ln == 2)
                if pair_ln
                else E.exact_ln(z, lnpw, torch.zeros_like(lnpw), eps_p)
            ).view(L * L, DP)
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
        bias = (
            E.exact_ln_pair_project(pair, lnpw, wb, mask, eps_p)
            if fused_pair_project
            else mm(za, wb.t())
        )
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
                else (
                    E.exact_transpose_tiled(bias, mask, L, False)
                    if shadow_io
                    else E.exact_mask_bias(bias, mask, L)
                )
            )
            out32, lse32 = E.exact_attention_fwd(qh, kh, vh, bm)
            delta32 = E.exact_delta(do_rows, out32)
            ds = (
                E.exact_attention_db_only(qh, kh, vh, bm, lse32, do_rows, delta32)
                if qualified
                else E.exact_attention_db(
                    qh, kh, vh, bm, out32, lse32, do_rows, delta32
                )
            )
            db = (
                E.exact_transpose_tiled(ds, mask, L, True)
                if shadow_io and H >= 16
                else E.exact_db_transpose(ds, mask)
            )
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
        db = (
            E.exact_transpose_tiled(ds, mask, L, True)
            if shadow_io and H >= 16
            else E.exact_db_transpose(ds, mask)
        )
        return finish(db)


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eps_s, eps_p, mask, single, pair, *leaves):
        ctx.param_dtypes = [t.dtype for t in leaves]
        params = leaves
        ctx.beta_shadow = _cap(single.device) == (9, 0)
        # Resolve the repair during forward so that the captured gradient policy
        # is fixed before backward.
        qualified_exact_dwb = single.shape[1] in (384, 768) and (
            leaves[-1].shape[0],
            single.shape[-1],
        ) in ((8, 384), (12, 384), (16, 384), (24, 384), (16, 512))
        # ln_pair has no beta: the shadow only repairs the pair-weight gradient.
        ctx.exact_dwb = (
            ctx.beta_shadow
            and qualified_exact_dwb
            and settings.current().apb_h100_exact_dwb
        )
        leaves = _kernel_leaves(leaves)
        y, *saved = forward(single, pair, mask, leaves, eps_s, eps_p)
        ctx.eps = (eps_s, eps_p)
        ctx.saved_count = len(saved)
        ctx.save_for_backward(
            mask, single, pair, *leaves, *saved, *(params if ctx.exact_dwb else ())
        )
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
        dx, dz, *grads = backward(single, pair, mask, leaves, *ctx.eps, saved, dy)
        if ctx.exact_dwb:
            params = list(v[13 + ctx.saved_count :])
            grads[9] = _pair_weight_shadow(single, pair, mask, params, *ctx.eps, dy)
        if _cap(single.device) == (9, 0):
            if torch.compiler.is_compiling():
                grads = [
                    g.to(dt) for g, dt in zip(grads, ctx.param_dtypes, strict=True)
                ]
            else:
                grads = _cast_cuda.cast(
                    grads, [torch.float32 if dt == torch.float32 else torch.bfloat16 for dt in ctx.param_dtypes]
                )
            from miniworld_engine.integrations.token_dit_train import (
                _activate_native_caller,
            )

            _activate_native_caller(force=True)
            return (None, None, None, dx, dz, *grads)
        return (
            None,
            None,
            None,
            dx,
            dz,
            *(g.to(dt) for g, dt in zip(grads, ctx.param_dtypes, strict=True)),
        )


class _TrainingHost(torch.autograd.Function):
    """B200 training with the C++ launch sequences: ``apb_b200::train_fwd`` / ``train_bwd`` take the module's parameters as they
    are (bf16 or an fp32 master; ``prep`` rounds the weights to bf16 inside its pass) and return fp32 gradients, cast here to
    each parameter's dtype (a no-op for an fp32 master)."""

    @staticmethod
    def forward(ctx, eps_s, eps_p, ns, nq, mask, single, pair, *params):
        ctx.param_dtypes = [t.dtype for t in params]
        ctx.nq = nq
        y, *saved = torch.ops.apb_b200.train_fwd(single, pair, mask, list(params), eps_s, eps_p, ns)
        ctx.eps_p = eps_p
        ctx.save_for_backward(mask, single, pair, *params, *saved)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        v = ctx.saved_tensors
        mask, single, pair = v[0], v[1], v[2]
        dx, dz, *grads = torch.ops.apb_b200.train_bwd(single, pair, mask, list(v[3:13]), list(v[13:]), dy, ctx.eps_p, ctx.nq)
        return (
            None,
            None,
            None,
            None,
            None,
            dx,
            dz,
            *(g if g.dtype == dt else g.to(dt) for g, dt in zip(grads, ctx.param_dtypes, strict=True)),
        )


def update_train(
    module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    if single.shape[0] > max_batch():
        return torch.cat([update_train(module, *c) for c in _chunks(single, pair, mask)])
    if _host(single.device):
        return _TrainingHost.apply(
            module.ln_single.eps,
            module.ln_pair.eps,
            _sm100().apb_ksplit(single.shape[1], module.n_head, single.shape[2]),
            _sm100().apb_qsplit(single.shape[1], module.n_head),
            mask,
            single,
            pair,
            *_leaves(module),
        )
    out = _Training.apply(
        module.ln_single.eps,
        module.ln_pair.eps,
        _mask(mask, single.shape[0] * single.shape[1]),
        single,
        pair,
        *_leaves(module),
    )
    if _cap(single.device) == (9, 0):
        from miniworld_engine.integrations.token_dit_train import _native_caller_hooks

        # The result graph now owns its AccumulateGrad nodes. Installing on
        # fresh leaves before apply can hook a temporary weak-held node.
        _native_caller_hooks(single, pair, *module.parameters(), force=True)
    return out
