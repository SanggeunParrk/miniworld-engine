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

``serves_*`` is the whole gate; ``settings.apb_sm80 = False`` turns the path off. Numerics and timings: ``docs/gpus/a100/attention_pair_bias/``.
"""

from __future__ import annotations

import contextlib
import functools
import math
import warnings

import torch

from miniworld_engine import settings
from miniworld_engine.kernels import _capture

DP = 128
SHAPES = ((8, 384), (12, 384), (16, 384), (24, 384), (16, 512))
LOG2E = math.log2(math.e)
NEG = -1e30  # masked keys: finite, so a fully masked sample stays NaN-free
_packs: dict = {}
_cores: dict = {}
_sides: dict = {}
_lib = torch.library.Library("miniworld_apb", "FRAGMENT")


def _native(schema: str, fake):
    """Registers the decorated launcher as the CUDA kernel of an operator of the ``miniworld_apb`` library (``schema``: ``name(args) -> returns``) and ``fake`` as its meta
    function: ``torch.compile`` keeps the call as one node, as with ``kernels._compile.opaque`` (which this follows when ``compile_wrap`` is ``disable``) -- minus
    ``torch.library.custom_op``'s Python layers (a ~0.2 ms host cost per call that its schema checks and aliasing test add). The result is callable like the operator and
    carries the launcher as ``__wrapped__`` (``_launch``)."""
    name = schema.split("(")[0]

    def decorate(fn):
        if settings.current().compile_wrap == "disable":
            return torch.compiler.disable(fn)
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


def _on(device: torch.device):
    """A device guard for the ATen calls of a launcher -- none when the tensors' device already is the current one (the extension's own entry points guard themselves)."""
    if device.index is None or device.index == torch.cuda.current_device():
        return contextlib.nullcontext()
    return torch.cuda.device(device)


def _overlap(cfg) -> bool:
    """Whether the step forks side streams: only while a CUDA graph is being captured. The streams put the long memory-bound pair pass next to the small kernels around
    it, a GPU-time gain -- but an eager or compiled step without a graph is bound by its host time (the GPU runs 0.2 - 0.5 ms of a 1.2 ms step), where every stream call
    only adds to it."""
    return cfg.apb_sm80_overlap and torch.cuda.is_current_stream_capturing()


def _launch(op, *args):
    """The step's opaque op: through the dispatcher when ``torch.compile`` traces it (the node Dynamo keeps), otherwise the launcher itself -- eager and CUDA-graph
    capture run the same Python, minus the custom-op layer (its schema checks and the aliasing test cost ~0.2 ms of host time per call)."""
    if torch.compiler.is_compiling():
        return op(*args)
    return getattr(op, "__wrapped__", op)(*args)


def _side_stream(dev: torch.device, which: int = 0) -> torch.cuda.Stream:
    """The second (``which`` 0) or third stream of the overlapped steps (one each per device)."""
    s = _sides.get((dev.index, which))
    if s is None:
        s = _sides[(dev.index, which)] = torch.cuda.Stream(device=dev)
    return s


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


def _bwd_pair_grid(dev: torch.device, heads: int, beside: bool, setting: int) -> int:
    """CTAs of the persistent pair-gradient pass. Running beside the key side's kernels (``beside``) it leaves them room: 8 heads (two CTAs of 75 KB per SM at full
    occupancy) take 4/3 CTAs per SM (measured at L128-768: the pass slows by a few percent, the kernels next to it run for free); 12 / 16 heads (one CTA per SM) keep all.
    0: the full occupancy."""
    if setting > 0:
        return setting
    return _sm_count(dev.index) * 4 // 3 if beside and heads == 8 else 0


def _core_warps(padded: int, heads: int, setting: int) -> int:
    """Warps per CTA of the inference core (16 queries each): at one sample the 128-query CTAs are (L / 128) x heads, one per SM, each a chain of L / 32 key tiles
    whose time is set by its warps sharing the SM. CTAs of 2 warps (32 queries) when that still leaves one per SM, else of 4 while the grid stays below two per
    SM (measured on 8 x 48, 12 x 32, 16 x 24 and 16 x 32 at L128-768); 8: the usual kernel, for the long grids."""
    if setting in (2, 4, 8):
        return setting
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
    if not settings.current().apb_sm80:
        return "settings.apb_sm80 is off"
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
    if grad and H == 24 and not settings.current().apb_sm80_train16:
        return "24 x 16 heads in training: settings.apb_sm80_train16 is off"
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


def _prep_train(leaves, ws: float, natural: bool):
    """The training pack from the parameters as they are (bf16, or an fp32 master rounded inside the kernel): (W q | k | v | g, its bias, Wf, Wo [D, W] bf16 with
    padded columns, and with ``natural`` Wf in natural units), q unscaled."""
    _, _, wq, bq, wk, wv, wg, wo, lnpw, _, wb = leaves
    H, _, W, D = _geometry(leaves)
    bf = torch.bfloat16
    wpack = wq.new_empty((4 * W, D), dtype=bf)
    bvec = wq.new_empty((4 * W,), dtype=bf)
    wf = wq.new_empty((H, DP), dtype=bf)
    wop = wq.new_empty((D, W), dtype=bf)
    wf_nat = torch.empty_like(wf) if natural else None
    _rows().ext().prep_train(
        wq, bq, wk, wv, wg, wo, wb, lnpw, wpack, bvec, wop, wf, 1.0, ws, wf_nat
    )
    return wpack, bvec, wf, wop, wf_nat


def _inference_fake(single, pair, mask, leaves, eps_s, eps_p):
    return torch.empty_like(single)


def _inference_py(
    single: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps_s: float,
    eps_p: float,
) -> torch.Tensor:
    """single [1, L, d], pair [1, L, L, 128] bf16, mask [L] bool or None, leaves as ``_leaves`` -> single + APB(single)."""
    L = single.shape[1]
    Lp = _pad(L)
    rows, dev = _rows(), single.device
    H, _, W, D = _geometry(leaves)
    with _on(dev):
        x = single.view(L, D)
        wpack, bvec, wf, wop, lnw, lnb, wo_k = _inference_packs(leaves)
        wo = wo_k if wop is None else wop
        # padded rows are zero: their k / v rows must be finite (a masked key still multiplies its v row by probability 0)
        xa = torch.empty_like(x) if Lp == L else x.new_zeros((Lp, D))
        y = torch.empty_like(
            x
        )  # residual seed, written by ln_rows; to_out accumulates onto it
        key = (dev.index, H, W // H)
        core = _cores.get(key)
        if core is None:
            core = _cores[key] = _sm80().GatedInferenceCore(
                dev.index, torch.bfloat16, H, W // H
            )
        cfg = settings.current()
        if _overlap(cfg):
            # The pair pass (memory bound, the longest kernel) runs on a second stream next to ln_rows + the projection GEMM; the core waits for both.
            main, side = torch.cuda.current_stream(dev), _side_stream(dev)
            side.wait_stream(main)
            rows.ext().ln_rows(x, lnw, lnb, xa, None, eps_s, y)
            qkvg = torch.addmm(bvec, xa, wpack.t())
            with torch.cuda.stream(side):
                bias = rows.pair_bias(
                    pair.view(L * L, DP), wf, mask, L, eps_p, NEG, padded=Lp
                )
            main.wait_stream(side)
            bias.record_stream(main)
        else:
            rows.ext().ln_rows(x, lnw, lnb, xa, None, eps_s, y)
            qkvg = torch.addmm(bvec, xa, wpack.t())
            bias = rows.pair_bias(
                pair.view(L * L, DP), wf, mask, L, eps_p, NEG, padded=Lp
            )
        core(qkvg, bias, 0, 1, warps=_core_warps(Lp, H, cfg.apb_sm80_attn_warps))
        return y.addmm_(qkvg[:L, :W], wo.t()).view(1, L, D)


def _inference_cpp(single, pair, mask, leaves, eps_s, eps_p):
    """The inference step as one C++ call (``step_sm80.cuh``): the cached packs in, the output out."""
    Lp = _pad(single.shape[1])
    cfg = settings.current()
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
            _overlap(cfg),
            _core_warps(Lp, H, cfg.apb_sm80_attn_warps),
            cfg.apb_sm80_attn_bn,
        )
    )


@_native(
    "inference(Tensor single, Tensor pair, Tensor? mask, Tensor[] leaves, float eps_s, float eps_p) -> Tensor",
    _inference_fake,
)
def inference(single, pair, mask, leaves, eps_s, eps_p):
    """single [1, L, d], pair [1, L, L, 128] bf16, mask [L] bool or None, leaves as ``_leaves`` -> single + APB(single)."""
    if settings.current().apb_sm80_cpp_step:
        return _inference_cpp(single, pair, mask, leaves, eps_s, eps_p)
    return _inference_py(single, pair, mask, leaves, eps_s, eps_p)


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
    cfg = settings.current()
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
            _overlap(cfg),
            _core_warps(_pad(single.shape[1]), H, cfg.apb_sm80_attn_warps),
        )
    )


def _backward_cpp(single, pair, mask, leaves, eps_s, eps_p, saved, dy, parameter_refs):
    cfg = settings.current()
    H = _geometry(leaves)[0]
    overlap = _overlap(cfg)
    pair_stream = overlap and cfg.apb_sm80_bwd_mode == 1
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
            _bwd_pair_grid(single.device, H, pair_stream, cfg.apb_sm80_bwd_pair_grid),
            _core_warps(_pad(single.shape[1]), H, cfg.apb_sm80_attn_warps),
        )
    )


# The same forward in natural units for the plain core (bf16 o, the log-sum-exp; the pair bias in the core's raw units, natural x sqrt(head dim), masked and
# padded keys at -1e4 natural, as ``augattn_sm80``), gate_rows and addmm; backward: dO GEMMs, gate_bwd (dO, dg), the core's backward (dq, dk, dv, and the bias
# gradient in fp32), qkv_bwd, the dxa / dW GEMMs, ln_bwd (+ residual), pair_bias_bwd. The padded 16 x 24 heads (32 wide, zero columns) keep the softmax scale of 24.
MASKED = -1e4


def serves_train(
    module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None = None
) -> bool:
    # All twelve 24 x 16 rows improve the old path; zero preserves it for diagnosis.
    train16 = settings.current().apb_sm80_train16
    return (
        torch.is_grad_enabled()
        and (module.n_head != 24 or train16)
        and _eligible(module, single, pair, mask)
    )


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


def _forward_py(
    single: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps_s: float,
    eps_p: float,
    save_natural: bool = False,
) -> list[torch.Tensor]:
    L = single.shape[1]
    Lp = _pad(L)
    rows = _rows()
    lnw, lnb = leaves[0], leaves[1]
    H, hd, W, D = _geometry(leaves)
    scale = math.sqrt(hd)
    with _on(single.device):
        x = single.view(L, D)
        overlap = _overlap(settings.current())
        if overlap:  # the pair pass (memory bound) on a second stream next to the pack, LayerNorm and projection
            main, side = torch.cuda.current_stream(x.device), _side_stream(x.device)
            side.wait_stream(main)
        wpack, bvec, _wf, wo, wf_nat = _prep_train(leaves, scale, save_natural)
        xa = torch.empty_like(x) if Lp == L else x.new_zeros((Lp, D))
        st = x.new_empty((L, 2), dtype=torch.float32)
        y = torch.empty_like(x)
        rows.ext().ln_rows(x, lnw, lnb, xa, st, eps_s, y)
        qkvg = torch.addmm(bvec, xa, wpack.t())
        with torch.cuda.stream(side) if overlap else contextlib.nullcontext():
            bias = rows.pair_bias_raw(  # Wf from the parameters: no dependency on the pack kernel
                pair.view(L * L, DP),
                leaves[10],
                leaves[8],
                scale,
                mask,
                L,
                eps_p,
                MASKED * scale,
                Lp,
            )
        if overlap:
            main.wait_stream(side)
            bias.record_stream(main)
        o, lse = _sm80().plain_forward(
            qkvg[:, :W],
            qkvg[:, W : 2 * W],
            qkvg[:, 2 * W : 3 * W],
            bias,
            1,
            Lp,
            H,
            W // H,
            save_lse=True,
            sm_scale=1.0 / scale,
            warps=_core_warps(Lp, H, settings.current().apb_sm80_attn_warps),
        )
        og = x.new_empty((L, W))
        rows.ext().gate_rows_bf(o, qkvg, og, L)
        y = y.addmm_(og, wo.t()).view(
            1, L, D
        )  # in place onto the residual seed: no copy of x into the output
    return [y, xa, st, qkvg, bias, o, lse, og, wpack, wo] + (
        [wf_nat] if save_natural else []
    )


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


def _backward_py(
    single: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    leaves: list[torch.Tensor],
    eps_s: float,
    eps_p: float,
    saved: list[torch.Tensor],
    dy: torch.Tensor,
    parameter_refs: list[torch.Tensor],
) -> list[torch.Tensor]:
    L = single.shape[1]
    Lp = _pad(L)
    rows = _rows()
    E = rows.ext()
    lnw, lnpw, wb = leaves[0], leaves[8], leaves[10]
    H, hd, W, D = _geometry(leaves)
    xa, st, qkvg, bias, o, lse, og, wpack = saved[:8]
    wo = saved[8]  # [D, W] bf16, the padded columns zero
    f32 = torch.float32
    with _on(single.device):
        x = single.view(L, D)
        dy2 = dy.contiguous().view(L, D)
        acc = torch.empty(
            rows.acc_n(H, D), device=x.device
        ).zero_()  # dlnw | dlnb | dbq | dWf | head sums, added by the kernels
        # the two weight-gradient GEMMs are off the critical path (only finalize reads them): a second stream
        overlap = _overlap(settings.current())
        main, side = torch.cuda.current_stream(x.device), _side_stream(x.device, 1)

        def _on_side():
            return torch.cuda.stream(side) if overlap else contextlib.nullcontext()

        if overlap:
            side.wait_stream(main)
        dog = dy2 @ wo  # [L, W]: zero in the pads (Wo's pad columns are zero)
        with _on_side():
            dwo = torch.mm(
                dy2.t(), og, out_dtype=f32
            )  # [D, W]; finalize keeps the real columns
        dob = (
            x.new_empty((Lp, W)) if Lp == L else x.new_zeros((Lp, W))
        )  # the padded queries' gradient is 0
        dqkvg = x.new_empty((L, 4 * W))
        E.gate_bwd_bf(dog, o, qkvg, dob, dqkvg, L)
        # A saved natural-unit pack rounds the unscaled product directly. Dividing
        # the scaled BF16 forward pack would introduce a second rounding error.
        wf_nat = (
            saved[9]
            if len(saved) > 9
            else (wb.float() * lnpw.float()[None]).to(torch.bfloat16)
        )
        cfg = settings.current()
        pair_stream = overlap and cfg.apb_sm80_bwd_mode == 1
        dz_box: list = []

        def _pair_bwd(db):
            dz_box.append(
                rows.pair_bias_bwd(
                    pair.view(L * L, DP),
                    db,
                    wf_nat,
                    L,
                    eps_p,
                    acc,
                    D,
                    grid=_bwd_pair_grid(
                        x.device, H, pair_stream, cfg.apb_sm80_bwd_pair_grid
                    ),
                )[0]
            )

        def _on_db(
            db,
        ):  # the pair pass needs only db: its own stream, next to the key side's kernels
            ps = _side_stream(x.device, 2)
            ps.wait_stream(main)
            with torch.cuda.stream(ps):
                _pair_bwd(db)

        dq, dk, dv, db = _sm80().plain_backward(
            qkvg[:, :W],
            qkvg[:, W : 2 * W],
            qkvg[:, 2 * W : 3 * W],
            dob,
            bias,
            o,
            lse,
            1,
            Lp,
            H,
            W // H,
            db_natural=True,
            sm_scale=1.0 / math.sqrt(hd),
            warps=_core_warps(Lp, H, settings.current().apb_sm80_attn_warps),
            on_db=_on_db if pair_stream else None,
            db_bf16=True,
        )
        E.qkv_bwd_bf(dq, dk, dv, dqkvg, acc, D, L)
        del dq, dk, dv
        if overlap:
            side.wait_stream(main)
        with _on_side():
            dwp = torch.mm(dqkvg.t(), xa[:L], out_dtype=f32)
        dxa = dqkvg @ wpack
        dx = torch.empty_like(x)
        E.ln_bwd(dxa, x, st, lnw, dy2, dx, acc)
        if not pair_stream:
            _pair_bwd(db)
        dz = dz_box[0]
        if overlap:
            main.wait_stream(side)
            dwo.record_stream(main)
            dwp.record_stream(main)
        if pair_stream:
            main.wait_stream(_side_stream(x.device, 2))
            dz.record_stream(main)
        # custom-op outputs may not alias: every parameter gradient leaves as its own tensor, written by one kernel. ln_pair's bias moves a head's logits by one
        # constant: sum_j dbias[h, i, j] = 0 for every query, so its gradient and its share of dWb are exactly 0
        if parameter_refs:
            outs = [
                torch.empty(t.shape, dtype=t.dtype, device=t.device)
                for t in parameter_refs
            ]
            E.finalize_typed(dwp, dwo, acc, wb, lnpw, outs)
        else:
            outs = [torch.empty(t.shape, dtype=f32, device=t.device) for t in leaves]
            E.finalize_typed(dwp, dwo, acc, wb, lnpw, outs)
    return [dx.view(single.shape), dz.view(pair.shape), *outs]


@_native(
    "train_forward(Tensor single, Tensor pair, Tensor? mask, Tensor[] leaves, float eps_s, float eps_p, bool save_natural=False) -> Tensor[]",
    _forward_fake,
)
def forward(single, pair, mask, leaves, eps_s, eps_p, save_natural=False):
    if settings.current().apb_sm80_cpp_step:
        return _forward_cpp(single, pair, mask, leaves, eps_s, eps_p, save_natural)
    return _forward_py(single, pair, mask, leaves, eps_s, eps_p, save_natural)


@_native(
    "train_backward(Tensor single, Tensor pair, Tensor? mask, Tensor[] leaves, float eps_s, float eps_p, Tensor[] saved, Tensor dy, Tensor[] parameter_refs) -> Tensor[]",
    _backward_fake,
)
def backward(single, pair, mask, leaves, eps_s, eps_p, saved, dy, parameter_refs):
    if settings.current().apb_sm80_cpp_step:
        return _backward_cpp(
            single, pair, mask, leaves, eps_s, eps_p, saved, dy, parameter_refs
        )
    return _backward_py(
        single, pair, mask, leaves, eps_s, eps_p, saved, dy, parameter_refs
    )


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eps_s, eps_p, mask, single, pair, *leaves):
        ctx.param_dtypes = [t.dtype for t in leaves]
        parameter_refs = leaves
        ctx.typed_gradients = settings.current().apb_sm80_typed_grad
        leaves = _train_leaves(leaves)
        heads, width, length = leaves[10].shape[0], single.shape[2], single.shape[1]
        # Every selected shape improved the previous pack in paired module
        # measurements. Keep the unproven h8d384/L768 and unmeasured shapes.
        save_natural = (
            settings.current().apb_sm80_save_natural
            and (heads, width) in ((8, 384), (12, 384), (16, 384), (16, 512))
            and length in (128, 256, 384, 512, 640, 768)
            and not (heads == 8 and width == 384 and length == 768)
        )
        y, *saved = _launch(
            forward, single, pair, mask, leaves, eps_s, eps_p, save_natural
        )
        ctx.eps = (eps_s, eps_p)
        if ctx.typed_gradients:
            ctx.save_for_backward(mask, single, pair, *leaves, *parameter_refs, *saved)
        else:
            ctx.save_for_backward(mask, single, pair, *leaves, *saved)
        return y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        v = ctx.saved_tensors
        mask, single, pair, leaves = v[0], v[1], v[2], list(v[3:14])
        parameter_refs, saved = (
            (list(v[14:25]), list(v[25:]))
            if ctx.typed_gradients
            else ([], list(v[14:]))
        )
        dx, dz, *grads = _launch(
            backward,
            single,
            pair,
            mask,
            leaves,
            *ctx.eps,
            saved,
            dy,
            parameter_refs if ctx.typed_gradients else [],
        )
        if ctx.typed_gradients:
            return (None, None, None, dx, dz, *grads)
        return (
            None,
            None,
            None,
            dx,
            dz,
            *(g.to(dt) for g, dt in zip(grads, ctx.param_dtypes, strict=True)),
        )


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
