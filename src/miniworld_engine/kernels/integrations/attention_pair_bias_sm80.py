"""Automatic dispatch to the A100 (sm_80) AttentionPairBias: hand CUDA but the plain GEMMs (cuBLAS).

Contract: bf16 activations, B = 1, d_single 384 as 8 heads x 48, 12 x 32, 24 x 16 (inference only) or 16 x 24 (AF3's Pairformer single attention; run 32 wide per head, as on
B200), or d_single 512 as 16 x 32; d_pair 128, no QK-norm, any L, ``implementation="miniworld"`` (engine backend not forced to Triton), sm_80.

Inference (one opaque op), the B200 step with this family's A100 pieces:
  ln_rows (LN(single) -> bf16, and a copy of the single as the residual seed) -> addmm q | k | v | g (q pre-scaled by log2(e) / sqrt(head dim): logits in
  exp2 units) -> pair_bias (LN(pair) . Wf, Wf = to_bias.weight * ln_pair.weight * log2(e); one read of the pair; masked keys, keys past L and padded query
  rows at -1e30) -> the gated core (``augmented_attention/cuda/sm80``: sigmoid(g) o written over the q columns) -> to_out accumulated in place (cuBLAS
  beta = 1) onto the residual seed.
The attention core wants L in tiles of 128: the single is padded with zero rows (their k / v are finite and every padded key is masked) and the bias is
written [H, Lp, Lp]; the pair is read in place. ln_pair's bias adds Wb . b to every (i, j) of a head: the softmax cancels it, so the forward drops it.

``serves_*`` is the whole gate. Numerics and timings: ``docs/gpus/a100/attention_pair_bias/``.
"""

from __future__ import annotations

import functools
import math

import torch

from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._nvcc import build_failed

DP = 128
#: the inference core's ``bn`` (key tile) launch argument: the A100 port's verified default
_ATTN_BN = 32
SHAPES = ((8, 384), (12, 384), (16, 384), (24, 384), (16, 512))
LOG2E = math.log2(math.e)
_packs: dict = {}
_lib = torch.library.Library("miniworld_apb", "FRAGMENT")


def _native(schema: str, fake):
    """Registers the decorated launcher as the CUDA kernel of an operator of the ``miniworld_apb`` library (``schema``: ``name(args) -> returns``) and ``fake`` as its meta
    function: ``torch.compile`` keeps the call as one node, as with ``kernels._compile.opaque`` -- minus
    ``torch.library.custom_op``'s Python layers (a ~0.2 ms host cost per call that its schema checks and aliasing test add). The result is callable like the operator and
    carries the launcher as ``__wrapped__`` (``_launch``)."""
    name = schema.split("(")[0]

    def decorate(fn):
        _lib.define(schema)
        _lib.impl(name, fn, "CUDA")
        torch.library.register_fake(f"miniworld_apb::{name}", fake)
        op = getattr(torch.ops.miniworld_apb, name)

        def call(*args):
            return op(*args)

        call.__wrapped__ = fn
        return call

    return decorate


@functools.cache
def _capability(index: int) -> tuple[int, int]:
    major, minor = torch.cuda.get_device_capability(index)
    return major, minor


@torch.compiler.assume_constant_result
def _is_a100_index(index: int) -> bool:
    """Compute capability 8.0 (a per-device constant: cached, and a constant to ``torch.compile``)."""
    return _capability(index) == (8, 0)


def _is_a100(device: torch.device) -> bool:
    index = device.index
    return _is_a100_index(torch.cuda.current_device() if index is None else index)


def _overlap() -> bool:
    """Whether the step forks side streams: only while a CUDA graph is being captured. The streams put the long memory-bound pair pass next to the small kernels around
    it, a GPU-time gain -- but an eager or compiled step without a graph is bound by its host time (the GPU runs 0.2 - 0.5 ms of a 1.2 ms step), where every stream call
    only adds to it."""
    return torch.cuda.is_current_stream_capturing()


def _launch(op, *args):
    """The step's opaque op: through the dispatcher when ``torch.compile`` traces it (the node Dynamo keeps), otherwise the launcher itself -- eager and CUDA-graph
    capture run the same Python, minus the custom-op layer (its schema checks and the aliasing test cost ~0.2 ms of host time per call)."""
    if torch.compiler.is_compiling():
        return op(*args)
    return getattr(op, "__wrapped__", op)(*args)


@torch.compiler.assume_constant_result
def _loads() -> bool:
    """Builds (first call) or loads the two sm_80 extensions; a failed build / load raises ``_nvcc.KernelBuildError`` (never a fallback).

    A process-level constant, so ``torch.compile`` evaluates it once at trace time instead of tracing the nvcc lookup / JIT build into the graph."""
    try:
        _rows().ext()
        _sm80()._ext()
    except Exception as exc:  # a build / load failure is an error, never a fallback
        build_failed("sm_80 AttentionPairBias", exc)
        return False  # reached only while planning (build_failed returns there)
    return True


def _pad(length: int) -> int:
    return -(-length // 128) * 128


@functools.cache
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def _bwd_pair_grid(dev: torch.device, heads: int, beside: bool) -> int:
    """CTAs of the persistent pair-gradient pass. Running beside the key side's kernels (``beside``) it leaves them room: 8 heads (two CTAs of 75 KB per SM at full
    occupancy) take 4/3 CTAs per SM (measured at L128-768: the pass slows by a few percent, the kernels next to it run for free); 12 / 16 heads (one CTA per SM) keep all.
    0: the full occupancy."""
    return _sm_count(dev.index) * 4 // 3 if beside and heads == 8 else 0


def _core_warps(padded: int, heads: int) -> int:
    """Warps per CTA of the inference core (16 queries each): at one sample the 128-query CTAs are (L / 128) x heads, one per SM, each a chain of L / 32 key tiles
    whose time is set by its warps sharing the SM. CTAs of 2 warps (32 queries) when that still leaves one per SM, else of 4 while the grid stays below two per
    SM (measured on 8 x 48, 12 x 32, 16 x 24 and 16 x 32 at L128-768); 8: the usual kernel, for the long grids."""
    if (padded // 32) * heads <= 108:
        return 2
    return 4 if (padded // 64) * heads <= 216 else 8


def _rows():
    from miniworld_engine.kernels.augmented_attention.cuda import apb as rows

    return rows


def _sm80():
    from miniworld_engine.kernels.augmented_attention.cuda import sm80

    return sm80


def refusal(module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, grad: bool) -> str | None:
    """Why the A100 (sm_80) paths cannot take this call (``grad``: the training Function, else the inference op), or ``None``: the fused C++ step
    (``_step_refusal``), else the composed path (``_generic_refusal``: fp32 on the TF32 core, the wide pairs, ``gate_bias``). The ``cuda`` backend of
    ``AttentionPairBias`` states exactly this on an A100."""
    why = _step_refusal(module, single, pair, mask, grad)
    if why is None:
        return None
    generic = _generic_refusal(module, single, pair, mask, grad)
    return None if generic is None else f"{why}; composed path: {generic}"


def _step_refusal(module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, grad: bool) -> str | None:
    """Why the fused C++ step (``step_sm80.cuh``) cannot take this call, or ``None``."""
    if not single.is_cuda or not _is_a100(single.device):
        return "the A100 (sm_80) kernels only"
    H, D = module.n_head, single.shape[-1]
    if module.use_qk_norm:
        return "per-head q / k RMSNorm (the packed q | k | v | g projection has no norm between the GEMM and the core)"
    if getattr(module, "gate_bias", 0.0):
        return "gate_bias: the A100 kernels have no gate constant"
    if single.dtype != torch.bfloat16 or pair.dtype != torch.bfloat16:
        return f"dtype {single.dtype} / {pair.dtype}: bfloat16 activations only (parameters bf16 or fp32 masters)"
    if single.ndim != 3 or single.shape[0] != 1 or not single.is_contiguous():
        return "single must be one contiguous [1, L, d_single] row (B == 1)"
    if (H, D) not in SHAPES or tuple(module.to_query.weight.shape) != (D, D):
        return f"{H} heads x d_single {D}: the kernels are built for (heads, d_single) in {SHAPES} with n_head * d_hidden == d_single"
    if tuple(module.to_bias.weight.shape) != (H, DP):
        return f"d_pair {module.to_bias.weight.shape[1]}: the A100 pair kernels take d_pair {DP}"
    if module.ln_single.weight is None or module.ln_pair.weight is None:
        return "the kernels take the affine LayerNorms"
    L = single.shape[1]
    if L < 1 or tuple(pair.shape) != (1, L, L, DP) or not pair.is_contiguous():
        return f"pair must be one contiguous [1, {L}, {L}, {DP}] row"
    if mask is not None and (tuple(mask.shape) != (1, L) or mask.dtype != torch.bool):
        return f"mask must be a bool [1, {L}] key mask"
    heads_dim = _rows().GEOMETRY[(H, D)][1]
    idx = single.device.index if single.device.index is not None else torch.cuda.current_device()
    if not _sm80().supported_gate(torch.bfloat16, _pad(L), H * heads_dim, H, idx):
        return f"L {L}, {H} x {heads_dim}: outside the A100 attention core's gate (augmented_attention/cuda/sm80.supported_gate)"
    if not _loads():
        return "the A100 APB extensions failed to build / load"
    return None


def _eligible(module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None) -> bool:
    return _step_refusal(module, single, pair, mask, torch.is_grad_enabled()) is None


def serves_inference(module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None = None) -> bool:
    return not torch.is_grad_enabled() and _eligible(module, single, pair, mask)


_zeros: dict = {}


def _ln_pair_bias(module) -> torch.Tensor:
    """``ln_pair`` has no offset (it feeds only the logits, where a constant per head cancels in the softmax); the kernels still take the vector, so they get zeros
    (one constant tensor per shape / dtype / device, not one made per call)."""
    bias = module.ln_pair.bias
    if bias is not None:
        return bias
    w = module.ln_pair.weight
    if torch.compiler.is_compiling():  # traced: the constant is part of the graph
        return torch.zeros_like(w)
    key = (w.device, w.dtype, tuple(w.shape))
    z = _zeros.get(key)
    if z is None:
        z = _zeros[key] = torch.zeros_like(w).detach()
    return z


def _leaves(module, inference: bool = False) -> list[torch.Tensor]:
    """The module's parameters as they are (any float dtype, e.g. an fp32 master): ln_single w / b, Wq, bq, Wk, Wv, Wg, Wo, ln_pair w / b, Wb. The ops cast them
    (``_kernel_leaves``) outside autograd, so fp32 parameters get unrounded fp32 gradients. Inference never reads ln_pair's offset: its slot holds the weight
    (no zero tensor made per call, and nothing for the pack cache to key on)."""
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
        module.ln_pair.weight if inference else _ln_pair_bias(module),
        module.to_bias.weight,
    ]


def _kernel_leaves(leaves) -> list[torch.Tensor]:
    """The kernels' dtypes: the LayerNorm vectors fp32, the projections bf16 (no copies when the parameters already are)."""
    bf = torch.bfloat16
    lnw, lnb, wq, bq, wk, wv, wg, wo, lnpw, lnpb, wb = leaves
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
        lnpb.float(),
        wb.to(bf),
    ]


def _train_leaves(leaves) -> list[torch.Tensor]:
    """The training kernels' parameters: the projections as they are (the pack kernel rounds an fp32 master to bf16 itself), the LayerNorm vectors fp32."""
    lnw, lnb, wq, bq, wk, wv, wg, wo, lnpw, lnpb, wb = leaves
    return [
        lnw.float(),
        lnb.float(),
        wq,
        bq,
        wk,
        wv,
        wg,
        wo,
        lnpw.float(),
        lnpb.float(),
        wb,
    ]


def _mask(mask: torch.Tensor | None, L: int) -> torch.Tensor | None:
    return None if mask is None else mask.reshape(L).to(torch.bool).contiguous()


def _geometry(leaves):
    """(heads, real head dim, row width W, d_single) from to_bias.weight [heads, 128] and ln_single.weight [d_single]."""
    H, D = leaves[10].shape[0], leaves[0].numel()
    return H, D // H, _rows().width(H, D), D


# ------------------------------------------------------------------------------------------------------------ inference
def _inference_packs(params):
    """(W q | k | v | g, its bias, Wf, the padded Wo or None, ln_single weight / bias fp32, Wo in kernel dtype): ``_prep`` with q scaled into exp2 units and Wf
    in exp2 units, and the kernel-dtype casts of the parameters (fp32 masters: a cast kernel each), cached per parameter version (the step reads them every
    call; repacking costs launches). Keyed on the parameters (ln_pair's offset slot is never read), and scoped to the CUDA-graph capture."""

    def build():
        leaves = _kernel_leaves(params)
        return (
            *_prep(leaves, LOG2E / math.sqrt(_geometry(leaves)[1]), LOG2E),
            leaves[0],
            leaves[1],
            leaves[7],
            params,
        )

    if not torch.is_grad_enabled():
        # grad off: a CUDA-graph capture reads the pack its eager warm-up made (``capture_reads_eager``), so a replay runs no cast / pack kernel
        keyed = [t for i, t in enumerate(params) if i != 9]
        return _capture.param_forms("apb_sm80_inference", keyed, lambda: build()[:7], capture_reads_eager=True)
    key = tuple((t.data_ptr(), t._version) for i, t in enumerate(params) if i != 9)
    return _capture.lookup(_packs, key, build, limit=8)[:7]


def _prep(leaves, qs: float, ws: float):
    """(W q | k | v | g [4 W, D] (padded rows, q x qs), its bias [4 W], Wf = to_bias.weight * ln_pair.weight * ws [H, 128], Wo [D, W] with padded columns
    -- or None when there is no padding (8 x 48, 12 x 32, 16 x 32: to_out.weight as is)), bf16, one launch."""
    _, _, wq, bq, wk, wv, wg, wo, lnpw, _, wb = leaves
    H, _, W, D = _geometry(leaves)
    wpack = wq.new_empty((4 * W, D))
    bvec = wq.new_empty((4 * W,))
    wf = wq.new_empty((H, DP))
    wop = wq.new_empty((D, W)) if W != D else None
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
        None,  # wbo: no bf16 copy of Wb (only the backward reads it)
    )
    return wpack, bvec, wf, wop


def _inference_fake(single, pair, mask, leaves, eps_s, eps_p):
    return torch.empty_like(single)


def _inference_cpp(single, pair, mask, leaves, eps_s, eps_p):
    """The inference step as one C++ call (``step_sm80.cuh``): the cached packs in, the output out."""
    Lp = _pad(single.shape[1])
    H = _geometry(leaves)[0]
    wpack, bvec, wf, wop, lnw, lnb, wo_k = _inference_packs(leaves)
    return (
        _rows()
        .ext()
        .apb_inference(
            single,
            pair,
            mask,
            wpack,
            bvec,
            wf,
            wo_k if wop is None else wop,
            lnw,
            lnb,
            eps_s,
            eps_p,
            _overlap(),
            _core_warps(Lp, H),
            _ATTN_BN,
        )
    )


@_native(
    "inference(Tensor single, Tensor pair, Tensor? mask, Tensor[] leaves, float eps_s, float eps_p) -> Tensor",
    _inference_fake,
)
def inference(single, pair, mask, leaves, eps_s, eps_p):
    """single [1, L, d], pair [1, L, L, 128] bf16, mask [L] bool or None, leaves as ``_leaves`` -> single + APB(single)."""
    return _inference_cpp(single, pair, mask, leaves, eps_s, eps_p)


def update_inference(module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if _step_refusal(module, single, pair, mask, False) is not None:
        return update_generic(module, single, pair, mask, False)
    return _launch(
        inference,
        single,
        pair,
        _mask(mask, single.shape[1]),
        _leaves(module, inference=True),
        module.ln_single.eps,
        module.ln_pair.eps,
    )


# ------------------------------------------------------------------------------------------------------------- training
def _forward_cpp(single, pair, mask, leaves, eps_s, eps_p, save_natural=False):
    H = _geometry(leaves)[0]
    lnw, lnb, wq, bq, wk, wv, wg, wo, lnpw, _, wb = leaves
    return list(
        _rows()
        .ext()
        .apb_train_forward(
            single,
            pair,
            mask,
            lnw,
            lnb,
            wq,
            bq,
            wk,
            wv,
            wg,
            wo,
            lnpw,
            wb,
            eps_s,
            eps_p,
            save_natural,
            _overlap(),
            _core_warps(_pad(single.shape[1]), H),
        )
    )


def _backward_cpp(single, pair, mask, leaves, eps_s, eps_p, saved, dy, parameter_refs):
    H = _geometry(leaves)[0]
    overlap = _overlap()
    pair_stream = overlap
    return list(
        _rows()
        .ext()
        .apb_train_backward(
            single,
            pair,
            leaves[0],
            leaves[8],
            leaves[10],
            eps_s,
            eps_p,
            saved,
            dy,
            parameter_refs or leaves,
            bool(parameter_refs),
            overlap,
            pair_stream,
            _bwd_pair_grid(single.device, H, pair_stream),
            _core_warps(_pad(single.shape[1]), H),
        )
    )


def serves_train(module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None = None) -> bool:
    return torch.is_grad_enabled() and _eligible(module, single, pair, mask)


def _forward_fake(single, pair, mask, leaves, eps_s, eps_p, save_natural=False):
    """y and the saved set: xa, (mean, rstd), qkvg, bias, O, LSE, og, W q | k | v | g, Wo [D, W] (and the natural-unit Wf)."""
    L = single.shape[1]
    Lp = _pad(L)
    H, _, W, D = _geometry(leaves)
    e, f32 = single.new_empty, torch.float32
    out = [
        torch.empty_like(single),
        e((Lp, D)),
        e((L, 2), dtype=f32),
        e((Lp, 4 * W)),
        e((H, Lp, Lp)),
        e((Lp, W)),
        e((1, H, Lp), dtype=f32),
        e((L, W)),
        e((4 * W, D)),
    ]
    return out + [e((D, W))] + ([e((H, DP))] if save_natural else [])


def _backward_fake(single, pair, mask, leaves, eps_s, eps_p, saved, dy, parameter_refs):
    """One gradient per input (its dtype) and per leaf (fp32), contiguous."""
    gradients = (
        [torch.empty(t.shape, dtype=t.dtype, device=t.device) for t in parameter_refs]
        if parameter_refs
        else [torch.empty(t.shape, dtype=torch.float32, device=t.device) for t in leaves]
    )
    return [torch.empty_like(single), torch.empty_like(pair), *gradients]


@_native(
    "train_forward(Tensor single, Tensor pair, Tensor? mask, Tensor[] leaves, float eps_s, float eps_p, bool save_natural=False) -> Tensor[]",
    _forward_fake,
)
def forward(single, pair, mask, leaves, eps_s, eps_p, save_natural=False):
    return _forward_cpp(single, pair, mask, leaves, eps_s, eps_p, save_natural)


@_native(
    "train_backward(Tensor single, Tensor pair, Tensor? mask, Tensor[] leaves, float eps_s, float eps_p, Tensor[] saved, Tensor dy, Tensor[] parameter_refs) -> Tensor[]",
    _backward_fake,
)
def backward(single, pair, mask, leaves, eps_s, eps_p, saved, dy, parameter_refs):
    return _backward_cpp(single, pair, mask, leaves, eps_s, eps_p, saved, dy, parameter_refs)


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eps_s, eps_p, mask, single, pair, *leaves):
        parameter_refs = leaves  # typed gradients: each parameter's gradient in its own dtype
        leaves = _train_leaves(leaves)
        heads, width, length = leaves[10].shape[0], single.shape[2], single.shape[1]
        # Every selected shape improved the previous pack in paired module
        # measurements. Keep the unproven h8d384/L768 and unmeasured shapes.
        save_natural = (
            (heads, width) in ((8, 384), (12, 384), (16, 384), (16, 512))
            and length in (128, 256, 384, 512, 640, 768)
            and not (heads == 8 and width == 384 and length == 768)
        )
        y, *saved = _launch(forward, single, pair, mask, leaves, eps_s, eps_p, save_natural)
        ctx.eps = (eps_s, eps_p)
        ctx.save_for_backward(mask, single, pair, *leaves, *parameter_refs, *saved)
        return y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        v = ctx.saved_tensors
        mask, single, pair, leaves = v[0], v[1], v[2], list(v[3:14])
        parameter_refs, saved = list(v[14:25]), list(v[25:])
        dx, dz, *grads = _launch(
            backward,
            single,
            pair,
            mask,
            leaves,
            *ctx.eps,
            saved,
            dy,
            parameter_refs,
        )
        return (None, None, None, dx, dz, *grads)


def update_train(module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if _step_refusal(module, single, pair, mask, True) is not None:
        return update_generic(module, single, pair, mask, True)
    return _Training.apply(
        module.ln_single.eps,
        module.ln_pair.eps,
        _mask(mask, single.shape[1]),
        single,
        pair,
        *_leaves(module),
    )


# ------------------------------------------------------------------------------------------------------------- the composed path
# The calls the fused C++ step does not take -- fp32 activations (the TF32 core), a pair wider than 128 (256 / 384 / 512), a gate constant -- run composed
# from this card's other sm_80 kernels, the H100 APB recipe (``attention_pair_bias_b200.py``: LN rows, one packed q | k | v | g GEMM, the pair LayerNorm +
# projection pass, the gated core, to_out onto the residual) on A100 pieces, forward and backward each one opaque op:
#
#   s = LN(single)                                     ATen LayerNorm, fp32 statistics
#   [q | k | v | g] = s W^T + [bq | 0 | 0 | gate_bias]  cuBLAS, one GEMM; heads narrower than 32 (16 x 24, 24 x 16) are run 32 wide (zero weight rows)
#   bias = LN(pair) Wf^T, head-major [H, L, L]         ``augmented_attention/cuda/sm80.pair_bias_generic`` (any width; the key mask folded in)
#   o = attention core                                 ``sm80.pgate_forward`` / ``plain_forward`` (bf16) or ``tf32_forward`` (fp32, TF32 mma.sync)
#   out = single + (sigmoid(g) o) Wo^T                 cuBLAS (the residual is the GEMM's C)
#
# and the backward reversed (the core's backward, ``pair_bias_generic_backward``, cuBLAS weight gradients in fp32, the LayerNorm backward).
GENERIC_HEAD_DIMS = (16, 24, 32, 48, 64)


def _hdp(hd: int) -> int:
    """The head width the core runs: 32 for the narrow heads, else the head dim."""
    return max(hd, 32)


def _generic_refusal(module, single, pair, mask, grad: bool) -> str | None:
    sm80 = _sm80()
    if module.use_qk_norm:
        return "per-head q / k RMSNorm"
    dt = single.dtype
    if dt not in (torch.bfloat16, torch.float32) or pair.dtype is not dt:
        return f"dtype {single.dtype} / {pair.dtype} (bf16 or fp32, one dtype)"
    if torch.is_autocast_enabled("cuda") and (torch.get_autocast_dtype("cuda") is not torch.bfloat16 or dt is not torch.bfloat16):
        return "autocast: bf16 activations under bf16 autocast only"
    ok = (torch.bfloat16, torch.float32) if dt is torch.bfloat16 else (torch.float32,)
    if any(p.dtype not in ok for p in module.parameters()):
        return f"parameters must be {ok} for {dt} activations"
    if single.ndim != 3 or single.shape[0] != 1:
        return "single must be [1, L, d_single] (B == 1)"
    L, H = single.shape[1], module.n_head
    if L % 128:
        return f"L {L}: a multiple of 128"
    hd = module.to_query.weight.shape[0] // H
    if hd not in GENERIC_HEAD_DIMS or module.to_query.weight.shape[0] != H * hd:
        return f"head dim {hd} (16 / 24 / 32 / 48 / 64)"
    dp = module.to_bias.weight.shape[1]
    if tuple(pair.shape) != (1, L, L, dp):
        return f"pair must be [1, {L}, {L}, {dp}]"
    if not sm80.pair_bias_generic_supported(dp, H, dt) or (grad and not sm80.pair_bias_generic_bwd_supported(dp, H, dt)):
        return f"d_pair {dp} x {H} heads: outside the sm_80 pair-bias kernels"
    if mask is not None and (tuple(mask.shape) != (1, L) or mask.dtype is not torch.bool):
        return f"mask must be a bool [1, {L}] key mask"
    if module.ln_single.weight is None or module.ln_single.bias is None or module.ln_pair.weight is None:
        return "the kernels take the affine LayerNorms"
    idx = single.device.index if single.device.index is not None else torch.cuda.current_device()
    hdp = _hdp(hd)
    if not (
        sm80.supported_plain(torch.bfloat16, L, H * hdp, H, idx) if dt is torch.bfloat16 else sm80.supported_tf32(dt, L, H * hdp, H, idx)
    ):
        return f"{H} x {hdp}, L {L}: outside the sm_80 attention core"
    if not _generic_loads(dt):
        return "the sm_80 attention extensions failed to build / load"
    return None


@torch.compiler.assume_constant_result
def _generic_loads(dt: torch.dtype) -> bool:
    try:
        sm80 = _sm80()
        sm80._ext()
        sm80._glue_ext()
        if dt is torch.float32:
            sm80._tf32_rna_ext()
    except Exception as exc:  # a build / load failure is an error, never a fallback
        build_failed("sm_80 attention", exc)
        return False  # reached only while planning (build_failed returns there)
    return True


def _gen_rows(L: int) -> int:
    """Rows a thread of the one-pass pair-bias kernel takes: 1 at L128 (its grid is (L / 64) x (L / 4 rows) CTAs: two rows a thread leave 32 CTAs for 108
    SMs; A100, 16 heads, fp32: d_pair 384 69 -> 55 us, 512 107 -> 86 us), else the kernel's default 2 (L256 d_pair 512: 180 against 258 us at one row)."""
    return 1 if L <= 128 else 2


def _bias_split(z: torch.Tensor, wcm: torch.Tensor, mask, L: int, eps: float) -> torch.Tensor:
    """The fp32 one-pass pair bias with its heads split in two halves on two streams: at d_pair 512, L128 the kernel's 64 CTAs (8192 multiply-adds a thread)
    leave the card half idle; two concurrent launches of 8 heads each double the CTAs (A100, 16 heads, cold L2: 99 -> 84 us; four quarters: 101 us)."""
    sm80 = _sm80()
    heads, half = wcm.shape[1], wcm.shape[1] // 2
    out = torch.empty(heads, L, L, device=z.device, dtype=z.dtype)
    w0, w1 = wcm[:, :half].contiguous(), wcm[:, half:].contiguous()
    main = torch.cuda.current_stream(z.device)
    side = _SIDE.get(z.device)
    if side is None:
        side = _SIDE[z.device] = torch.cuda.Stream(device=z.device)
    side.wait_stream(main)
    for t in (z, w1, out) + (() if mask is None else (mask,)):
        t.record_stream(side)
    with torch.cuda.stream(side):
        sm80.pair_bias_generic(z, w1, mask, L, eps, oscale=1.0, rows=1, out=out[half:])
    sm80.pair_bias_generic(z, w0, mask, L, eps, oscale=1.0, rows=1, out=out[:half])
    main.wait_stream(side)
    return out


_SIDE: dict = {}


#: the longest L whose fp32 wide-pair backward runs as LayerNorm + TF32 GEMMs (``_bias_gemm_bwd``); beyond it the one-pass backward kernel is faster
GEMM_BWD_MAX_L = 384


def _gemm_pair(pair: torch.Tensor, f32: bool) -> bool:
    """The backward of an fp32 pair of 256 channels or more at L <= ``GEMM_BWD_MAX_L``: a LayerNorm recompute + TF32 cuBLAS GEMMs + the LayerNorm backward,
    faster there than the one-pass kernel (A100, 16 heads, L128 train: d_pair 512 0.58 -> 0.49 ms, 384 0.48 -> 0.44); at L768 the one-pass kernel is faster."""
    return f32 and pair.shape[-1] >= 256 and pair.shape[1] <= GEMM_BWD_MAX_L


def _bias_gemm_bwd(z: torch.Tensor, wcm: torch.Tensor, db: torch.Tensor, L: int, eps: float):
    """(dz [L, L, dp], dWf [H, dp] fp32) of ``_bias_gemm`` from db [H, L, L] fp32 (zero on the masked keys)."""
    dp, heads = wcm.shape
    z2 = z.reshape(L * L, dp)
    zn, mean, rstd = torch.native_layer_norm(z2, (dp,), None, None, eps)
    d2 = db.reshape(heads, L * L)
    dzn = torch.mm(d2.t(), wcm.t())
    dz = torch.ops.aten.native_layer_norm_backward(dzn, z2, (dp,), mean, rstd, None, None, [True, False, False])[0]
    return dz.view(L, L, dp), torch.mm(d2, zn)


def _gparams(module) -> list[torch.Tensor]:
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


def _pad_rows(w: torch.Tensor, heads: int, hd: int, hdp: int) -> torch.Tensor:
    """[H hd, ...] -> [H hdp, ...] with zero rows after each head's hd."""
    if hd == hdp:
        return w
    v = w.reshape(heads, hd, *w.shape[1:])
    return torch.cat([v, v.new_zeros(heads, hdp - hd, *w.shape[1:])], 1).reshape(heads * hdp, *w.shape[1:])


def _unpad_rows(w: torch.Tensor, heads: int, hd: int, hdp: int) -> torch.Tensor:
    return w if hd == hdp else w.reshape(heads, hdp, *w.shape[1:])[:, :hd].reshape(heads * hd, *w.shape[1:])


def _gpack(params, heads: int, gate_bias: float, dt: torch.dtype, save: bool = True):
    """(W q | k | v | g [4 H hdp, D], its bias [bq | 0 | 0 | gate_bias], Wo [D, H hdp], Wf channel-major fp32 [dp, H]) in ``dt``, per parameter version."""
    from miniworld_engine.kernels.integrations.adaln_sm80 import cached

    _, _, wq, bq, wk, wv, wg, wo, lnpw, wb = params
    hd = wq.shape[0] // heads
    hdp = _hdp(hd)

    def build():
        w = torch.cat([_pad_rows(t.to(dt), heads, hd, hdp) for t in (wq, wk, wv, wg)]).contiguous()
        n = heads * hdp
        b = torch.zeros(4 * n, device=wq.device, dtype=dt)
        b[:n] = _pad_rows(bq.to(dt), heads, hd, hdp)
        b[3 * n :] = gate_bias
        wop = _pad_rows(wo.to(dt).t(), heads, hd, hdp).t().contiguous()
        wcm = (wb.float() * lnpw.float()[None]).t().contiguous()
        return w, b, wop, wcm

    if not save and not torch.is_grad_enabled():
        # inference: a CUDA-graph capture reads the pack its eager warm-up made (``capture_reads_eager``), so a replay runs none of the pack's
        # ~20 casts / pads / fills; a training call keeps the capture-scoped cache (a captured step repacks after an optimizer step)
        return _capture.param_forms(("apb_sm80_generic", dt, gate_bias, heads), list(params), build, capture_reads_eager=True)
    return cached(f"apb_sm80_generic_{dt}_{gate_bias}", tuple(params), build)


def _gfwd_fake(single, pair, mask, params, heads, gate_bias, eps_s, eps_p, save):
    L, D = single.shape[1], single.shape[2]
    n = heads * _hdp(params[2].shape[0] // heads)
    e = single.new_empty
    outs = [torch.empty_like(single)]
    if save:
        outs += [
            e((L, D)),
            e((2, L, 1), dtype=torch.float32),
            e((L, 4 * n)),
            e((heads, L, L)),
            e((L, n)),
            e((heads, L), dtype=torch.float32),
            e((L, n)),
        ]
    return outs


@opaque(fake=_gfwd_fake, name="apb_sm80_generic_fwd")
def _gfwd(
    single: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    params: list[torch.Tensor],
    heads: int,
    gate_bias: float,
    eps_s: float,
    eps_p: float,
    save: bool,
) -> list[torch.Tensor]:
    """single [1, L, D], pair [1, L, L, dp] -> [single + APB(single, pair)], with ``save`` also s, the LN statistics, qkvg, bias, o, lse, og."""
    from miniworld_engine.kernels.integrations.adaln_sm80 import _tf32

    sm80 = _sm80()
    dt = single.dtype
    f32 = dt is torch.float32
    L, D = single.shape[1], single.shape[2]
    hd = params[2].shape[0] // heads
    hdp = _hdp(hd)
    n = heads * hdp
    w, b, wop, wcm = _gpack(params, heads, gate_bias, dt, save)
    x = single.reshape(L, D)
    with torch.cuda.device(single.device), _tf32(f32):
        s32, mu, rs = torch.native_layer_norm(x.float(), (D,), params[0].float(), params[1].float(), eps_s)
        s = s32.to(dt)
        qkvg = torch.addmm(b, s, w.t())
        q, k, v, g = (qkvg[:, i * n : (i + 1) * n] for i in range(4))
        m = None if mask is None else mask.reshape(L)
        bias = (
            _bias_split(pair[0], wcm, m, L, eps_p)
            if f32 and pair.shape[-1] >= 512 and L <= 128 and heads % 8 == 0
            else sm80.pair_bias_generic(pair[0], wcm, m, L, eps_p, oscale=1.0 if f32 else math.sqrt(hd), rows=_gen_rows(L))
        )
        scale = hd**-0.5
        if save:
            o = torch.empty(L, n, device=x.device, dtype=dt)
            lse = torch.empty(1, heads, L, device=x.device, dtype=torch.float32)
            if f32:
                sm80.tf32_forward(q, k, v, bias, 1, L, heads, hdp, sm_scale=scale, out=o, lse=lse, rna=True)
            else:
                sm80.plain_forward(q, k, v, bias, 1, L, heads, hdp, save_lse=True, sm_scale=scale, out=o, lse=lse)
            og = torch.empty_like(o)
            sm80.gate_rows(o, g, og)
        else:
            if f32:
                sm80.tf32_forward(q, k, v, bias, 1, L, heads, hdp, sm_scale=scale, gate=g, out=q, rna=True)
            else:
                sm80.pgate_forward(q, k, v, g, bias, 1, L, heads, hdp, sm_scale=scale, out=q)
            og = q
        out = torch.addmm(x, og, wop.t()).view(single.shape)
    if not save:
        return [out]
    return [out, s, torch.stack([mu.view(L, 1), rs.view(L, 1)]), qkvg, bias, o, lse.view(heads, L), og]


def _gbwd_fake(dout, single, pair, mask, params, saved, heads, gate_bias, eps_p):
    return [torch.empty_like(single), torch.empty_like(pair), *(torch.empty_like(p) for p in params)]


@opaque(fake=_gbwd_fake, name="apb_sm80_generic_bwd")
def _gbwd(
    dout: torch.Tensor,
    single: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    params: list[torch.Tensor],
    saved: list[torch.Tensor],
    heads: int,
    gate_bias: float,
    eps_p: float,
) -> list[torch.Tensor]:
    """The backward of ``_gfwd`` (``save=True``): [d single, d pair, *d params] (each in its input's dtype)."""
    from miniworld_engine.kernels.integrations.adaln_sm80 import _tf32, _wgrad

    sm80 = _sm80()
    s, stats, qkvg, bias, o, lse, og = saved
    dt = single.dtype
    f32 = dt is torch.float32
    L, D = single.shape[1], single.shape[2]
    hd = params[2].shape[0] // heads
    hdp = _hdp(hd)
    n = heads * hdp
    w, _, wop, wcm = _gpack(params, heads, gate_bias, dt)
    dy = dout.reshape(L, D)
    with torch.cuda.device(single.device), _tf32(f32):
        dog = torch.mm(dy, wop)  # [L, n]
        dwo = _wgrad(dy, og)  # [D, n]
        dqkvg = torch.empty_like(qkvg)
        dob = torch.empty_like(o)
        sm80.gate_bwd(dog, o, qkvg[:, 3 * n :], dob, dqkvg[:, 3 * n :])
        q, k, v = (qkvg[:, i * n : (i + 1) * n] for i in range(3))
        dq, dk, dv = (dqkvg[:, i * n : (i + 1) * n] for i in range(3))
        scale = hd**-0.5
        if f32:
            _, _, _, db = sm80.tf32_backward(
                q, k, v, dob, bias, o, lse.view(1, heads, L), 1, L, heads, hdp, sm_scale=scale, dq=dq, dk=dk, dv=dv, rna=True
            )
        else:
            _, _, _, db = sm80.plain_backward(
                q, k, v, dob, bias, o, lse.view(1, heads, L), 1, L, heads, hdp, db_natural=True, sm_scale=scale, dq=dq, dk=dk, dv=dv
            )
        m = None if mask is None else mask.reshape(L)
        dz, dwf = (
            _bias_gemm_bwd(pair[0], wcm, db, L, eps_p)
            if _gemm_pair(pair, f32)
            else sm80.pair_bias_generic_backward(pair[0], wcm, db, m, L, eps_p)
        )
        ds = torch.mm(dqkvg, w)  # [L, D]
        dw = _wgrad(dqkvg, s)  # [4 n, D]
        dbq = torch.sum(dqkvg[:, :n], dim=0, dtype=torch.float32)
        dx, dlnw, dlnb = torch.ops.aten.native_layer_norm_backward(
            ds.float(), single.reshape(L, D).float(), (D,), stats[0], stats[1], params[0].float(), params[1].float(), [True, True, True]
        )
        dx = dx + dy.float()
    lnpw, wb = params[8], params[9]
    grads = [
        dlnw,
        dlnb,
        _unpad_rows(dw[:n], heads, hd, hdp),
        _unpad_rows(dbq, heads, hd, hdp),
        _unpad_rows(dw[n : 2 * n], heads, hd, hdp),
        _unpad_rows(dw[2 * n : 3 * n], heads, hd, hdp),
        _unpad_rows(dw[3 * n :], heads, hd, hdp),
        _unpad_rows(dwo.t(), heads, hd, hdp).t(),
        (dwf * wb.float()).sum(0),
        dwf * lnpw.float()[None],
    ]
    outs = [torch.empty(p.shape, device=p.device, dtype=p.dtype) for p in params]
    torch._foreach_copy_(outs, [g_.reshape(p.shape) for g_, p in zip(grads, params, strict=True)])  # pyright: ignore[reportPrivateImportUsage]  -- torch internal, on purpose
    return [dx.to(dt).view(single.shape), dz.view(pair.shape), *outs]


class _Generic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, single, pair, mask, heads, gate_bias, eps_s, eps_p, *params):
        out, *saved = _gfwd(single, pair, mask, list(params), heads, gate_bias, eps_s, eps_p, True)
        ctx.save_for_backward(single, pair, *params, *saved)
        ctx.meta = (mask, heads, gate_bias, eps_p, len(params))
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dout):
        mask, heads, gate_bias, eps_p, npar = ctx.meta
        v = ctx.saved_tensors
        single, pair, params, saved = v[0], v[1], list(v[2 : 2 + npar]), list(v[2 + npar :])
        ds, dz, *pg = _gbwd(dout.contiguous(), single, pair, mask, params, saved, heads, gate_bias, eps_p)
        need = ctx.needs_input_grad
        return (
            ds if need[0] else None,
            dz if need[1] else None,
            None,
            None,
            None,
            None,
            None,
            *(g if need[7 + i] else None for i, g in enumerate(pg)),
        )


def update_generic(module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, grad: bool) -> torch.Tensor:
    """The composed path (``_generic_refusal`` must have returned None): single + APB(single, pair)."""
    args = (module.n_head, float(module.gate_bias), float(module.ln_single.eps), float(module.ln_pair.eps))
    m = None if mask is None else mask.contiguous()
    with torch.autocast("cuda", enabled=False):
        single, pair = single.contiguous(), pair.contiguous()
        if grad:
            return _Generic.apply(single, pair, m, *args, *_gparams(module))
        return _gfwd(single, pair, m, _gparams(module), *args, False)[0]
