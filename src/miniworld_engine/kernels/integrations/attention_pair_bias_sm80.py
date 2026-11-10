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
import warnings

import torch

from miniworld_engine.kernels import _capture

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


_FAILED = False


@functools.cache
def _capability(index: int) -> tuple[int, int]:
    return tuple(torch.cuda.get_device_capability(index))


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
        warnings.warn(
            f"sm_80 AttentionPairBias kernels unavailable, keeping the module path: {exc!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
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


def refusal(
    module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, grad: bool
) -> str | None:
    """Why the A100 (sm_80) path cannot take this call (``grad``: the training Function, else the inference op), or ``None``.
    The ``cuda`` backend of ``AttentionPairBias`` states exactly this on an A100."""
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


def _eligible(
    module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None
) -> bool:
    return refusal(module, single, pair, mask, torch.is_grad_enabled()) is None


def serves_inference(
    module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None = None
) -> bool:
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


def update_inference(
    module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
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


def serves_train(
    module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None = None
) -> bool:
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
        else [
            torch.empty(t.shape, dtype=torch.float32, device=t.device) for t in leaves
        ]
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
        y, *saved = _launch(
            forward, single, pair, mask, leaves, eps_s, eps_p, save_natural
        )
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


def update_train(
    module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    return _Training.apply(
        module.ln_single.eps,
        module.ln_pair.eps,
        _mask(mask, single.shape[1]),
        single,
        pair,
        *_leaves(module),
    )
