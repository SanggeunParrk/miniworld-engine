"""The AF3 atom transformer block with native block-local attention on B200 and H100.

H100 cross-attention training at A=1/5/48, N=4096/8192 and inference at
(A,N)=(1,4096),(5,2048),(5,4096),(48,4096) use the CUDA block by default.
These paths require contiguous bf16 activations and bf16 or fp32 weights;
unsupported contracts raise. The A=5, N=2048/4096 inference cells use a packed
bf16 post-attention epilogue and persistent conditioning schedule, qualified
against the previous CUDA path and the recorded compiled PyTorch baseline.

One ``LocalDiTBlock`` call runs, in place of the module path,

    forward   cond_fwd -> pre_fwd -> pair bias (trunked) -> local attn_fwd -> post_fwd
    backward  tr_bwd_gate -> post_bwd -> local attn_dq / attn_dkv (+ dbias) -> pair-bias backward -> pre_bwd -> cond_bwd,
              then the weight gradients as cuBLAS GEMMs on the activations those kernels write.

The row kernels (conditioning, input projections, post-attention, transition, and their backward) are ``sm100_atom``'s: they do not
depend on what the attention attends to. The attention and the pair bias are ``sm100_atom_local``'s: 32 queries x 128 keys per window,
the trunked pair ``[nwin, 32, 128, 16]`` (bias ``LN(z) Wb`` per window, shared by the samples), no N x N tensor anywhere.
Both residuals are inside post_fwd (the block returns the stream).

``serves()`` is the whole gate: B200, the engine's kernel backend (implementation MINIWORLD), bf16 single / cond / pair,
the atom widths (d_single = d_cond = 128, 4 heads x 32, d_pair = 16, transition n = 2, no QK-norm), B == 1, any N, a [B, N] bool key
mask or none, LayerNorm eps 1e-5. N is padded to a multiple of 128 for the row kernels (zero single / cond rows, the padded atoms
masked as keys, the pair padded with zero windows) and the first N rows come back. ``settings.local_dit_sm100=0`` turns it off. A
build or load failure warns once and keeps the module path.

**Hoisted per-item tables (inference).** The conditioning tables (``cond_fwd``'s modulation and, in the cross-attention mode, the key / value
modulation) depend on the conditioning tensor and the weights only, and the windowed pair bias on the pair and two weights only; a sampler calls
every block tens of times with one conditioning and one pair. An inference call takes them from the hoist: made once per tensor (pointer and
in-place version) and weights, so later calls launch neither ``cond_fwd``, ``cond_ln`` + its GEMM nor ``pair_bias_fwd``. Inside a CUDA-graph capture
the entries are scoped to the capture (a replay recomputes them, as it repacks the weights), unless the caller declares the inputs static
(``kernels._capture.static_inputs()`` / ``settings.static_conditioning=1``): then a replay runs none of those kernels. The training path
(``save``) computes them every call. ``settings.local_dit_hoist=0`` turns the hoist off.

The block is two opaque ops, ``local_dit_block_fwd`` and ``local_dit_block_bwd`` (``kernels._compile.opaque``), so a ``torch.compile``d
model keeps them in its graph. The ops take the 23 parameters as a list (``WEIGHTS``); packing them for the kernels is reused while
every weight's tensor object and version are unchanged, except while a CUDA graph is captured. Parameters may be bf16 or fp32; their
gradients come back in their own dtype.

**The fp32 path.** fp32 single / cond / pair with fp32 parameters (and no autocast) run the same block on fp32 kernels: every product on
tcgen05 kind::tf32 (``sm100_atom.gemm32`` -- projections and activation gradients, with the conditioning-sigmoid, gated-residual and
SwiGLU epilogues -- and the ``sm100_atom_local`` ``*_tf32`` attention kernels), the LayerNorms, AdaLNs, gates, softmax, residual stream,
saved activations and gradients fp32 (``sm100_atom`` ``rows_tf32.cu``, ``sm100_atom_local`` ``lbias_tf32.cu``); the weight gradients are
cuBLAS with TF32 forced on and restored. Two more opaque ops (``local_dit_block_fwd_tf32`` / ``_bwd_tf32``), the same hoist (fp32 tables
under their own keys), the same padding. The kernels load separately from the bf16 ones: a failure warns once and keeps the module path for
fp32 calls only. ``settings.local_dit_tf32=0`` keeps fp32 calls on the module path. The fp32 weight pack is one flat buffer (one cat and an
in-place tf32 rounding: three kernels when a graph replay repacks; the backward's transposes are packed separately, by training calls only),
and ``static_inputs()`` covers it (a static capture replays no packing kernel). An inference call's conditioning tables are one LayerNorm
launch and one grouped GEMM, and its block is three kernels: ``atom_pre_tf32`` (AdaLN 1, the K / V AdaLN, the four projections),
the attention, ``atom_post_tf32`` (gate, Wo, both gated residuals, AdaLN 2, SwiGLU, Ws) -- the unfused kernels' arithmetic with the
activations kept on chip (``settings.local_dit_tf32_fused=0``: the unfused kernels; training always runs them, they save the
activations). The fp32 kernels launch with programmatic dependent launch.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from miniworld_engine import settings
from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import device_constant, opaque
from miniworld_engine.kernels._tf32 import tf32_grid

# ------------------------------------------------------------------------------------------------------------ weights
# The block's parameters and the kernels' weight packs (stacked / transposed bf16 packs and the backward's transposed A operands).
DS_, DC_, DP_, NH_, DH_ = 128, 128, 16, 4, 32
EPS = 1e-5
BF = torch.bfloat16

#: The block's parameters, in the order the ops take and return them (every ``DiTBlock.named_parameters`` entry at atom widths).
WEIGHTS = (
    "attention.ada_ln_in.ln_cond.weight",
    "attention.ada_ln_in.to_scale.weight",
    "attention.ada_ln_in.to_scale.bias",
    "attention.ada_ln_in.to_bias.weight",
    "attention.to_query.weight",
    "attention.to_query.bias",
    "attention.to_key.weight",
    "attention.to_value.weight",
    "attention.ln_pair.weight",
    "attention.to_bias.weight",
    "attention.to_gate.weight",
    "attention.to_out.weight",
    "attention.to_scale.weight",
    "attention.to_scale.bias",
    "transition.ada_ln_in.ln_cond.weight",
    "transition.ada_ln_in.to_scale.weight",
    "transition.ada_ln_in.to_scale.bias",
    "transition.ada_ln_in.to_bias.weight",
    "transition.expand_a.weight",
    "transition.expand_b.weight",
    "transition.squeeze.weight",
    "transition.to_scale.weight",
    "transition.to_scale.bias",
)
_GAMMA, _WBIAS = (
    WEIGHTS.index("attention.ln_pair.weight"),
    WEIGHTS.index("attention.to_bias.weight"),
)


def _namespace(weights) -> SimpleNamespace:
    """The attribute tree ``sm100_atom.cond_params`` / ``pre_params`` / ``post_params`` read (``block.attention.to_query.weight``)."""
    block = SimpleNamespace()
    for path, weight in zip(WEIGHTS, weights, strict=True):
        node = block
        *parents, leaf = path.split(".")
        for name in parents:
            if not hasattr(node, name):
                setattr(node, name, SimpleNamespace())
            node = getattr(node, name)
        setattr(node, leaf, weight)
    return block


def _pack(weights, backward=False):
    """The kernels' stacked bf16 weights and the backward's transposed A operands (Hopper: the live CUDA pack, ``backward`` adds the
    transposes)."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom as sm100

    if sm100._hopper(
        weights[0].device.index
        if weights[0].device.index is not None
        else torch.cuda.current_device()
    ):
        return sm100.pack_hopper(weights, backward=backward)
    block = _namespace(weights)
    with torch.no_grad():
        wc, wp, wo = (
            sm100.cond_params(block),
            sm100.pre_params(block),
            sm100.post_params(block),
        )

        def transposed(t):
            return t.t().contiguous()

        wb = {
            "WsT": transposed(wo[2]),
            "WuT": transposed(wo[1]),
            "WoT": transposed(wo[0]),
            "WpT": transposed(wp[0]),
            "WmodT": transposed(wc[0]),
        }
    return wc, wp, wo, wb


# The packs are ``kernels._capture.param_forms``: the weights' tensor objects (held weakly) and versions decide, not an address a new
# tensor may reuse.
def _weights(weights, backward=False):
    """Scoped to the CUDA-graph capture (``kernels._capture``): a capture packs once and records it. ``backward`` only matters on Hopper
    (the pack then also holds the transposed A operands) and is part of the key."""
    return _capture.param_forms(("local_dit", bool(backward)), weights, lambda: _pack(weights, backward))


def _padded(t, n):
    """[A, 1, N, d] -> [A, 1, n, d], zero rows appended (the tensor itself when n == N)."""
    return (
        t if t.shape[2] == n else torch.nn.functional.pad(t, (0, 0, 0, n - t.shape[2]))
    )


def _unpadded(t, shape):
    """[A, 1, n, d] (a view of the kernels' rows) -> the caller's [A, 1, N, d]."""
    return t if t.shape == shape else t[:, :, : shape[2]].contiguous()


def _padded_length(n0: int) -> int:
    return (n0 + 127) // 128 * 128


QUERIES, KEYS = 32, 128
_LOADED: set[int] = set()
_HOIST_COND: dict = {}  # (conditioning tensor, weights) -> (mod, mkv): the per-item conditioning tables of an inference call
_HOIST_BIAS: dict = {}  # (pair tensor, LN / bias weights) -> the windowed pair bias
_KV: dict = {}  # weights -> the cross-attention operands (kv_pack)


def _hoisting() -> bool:
    return settings.current().local_dit_hoist


_FAILED = False
F32 = torch.float32
_LOADED32: set[int] = set()
_FAILED32 = False
_ROW_KERNELS = ("cond", "pre", "prex", "prexp", "post", "trg", "postb", "preb", "prebx", "prebxp", "condb")
#: The parameters of the cross-attention mode's second AdaLN (keys / values), appended to ``WEIGHTS``.
CROSS_WEIGHTS = (
    "attention.ada_ln_kv.ln_cond.weight",
    "attention.ada_ln_kv.to_scale.weight",
    "attention.ada_ln_kv.to_scale.bias",
    "attention.ada_ln_kv.to_bias.weight",
)


@device_constant
def _kernels_ready(index: int) -> bool:
    """Build and load the row kernels and the local attention kernels on ``index`` once; False (after one warning) when a toolchain or
    driver problem keeps the module path. A constant to ``torch.compile``."""
    global _FAILED
    if _FAILED:
        return False
    if index in _LOADED:
        return True
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom as rows
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm100_atom_local as local,
    )

    try:
        hopper = rows._hopper(index)
        for name in ("cond", "pre", "post", "pack") if hopper else _ROW_KERNELS:
            rows.atom_kernel(name, index)
        for name in (
            ("fwd", "pb_f", "cond_ln", "kv_f", "kv_pack") if hopper else local.KERNELS
        ):
            local._load(name, index)
    except Exception as exc:  # a toolchain or driver problem keeps the module path
        if rows._hopper(index):
            raise RuntimeError("Hopper local DiT kernel build failed") from exc
        _FAILED = True
        warnings.warn(
            f"sm_100a local atom DiT kernels unavailable, keeping the module path: {exc!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    _LOADED.add(index)
    return True


@device_constant
def _kernels32_ready(index: int) -> bool:
    """Build and load the fp32 path's kernels (``sm100_atom.KERNELS_TF32``, ``sm100_atom_local.KERNELS_TF32``) on ``index`` once; False
    (after one warning) when a toolchain or driver problem keeps the module path for fp32 calls. Independent of the bf16 kernels."""
    global _FAILED32
    if _FAILED32:
        return False
    if index in _LOADED32:
        return True
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom as rows
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm100_atom_local as local,
    )

    try:
        rows.load_all32(index)
        for name in local.KERNELS_TF32:
            local._load32(name, index)
    except Exception as exc:  # a toolchain or driver problem keeps the module path
        _FAILED32 = True
        warnings.warn(
            f"sm_100a local atom DiT TF32 kernels unavailable, keeping the module path for fp32: {exc!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    _LOADED32.add(index)
    return True


_FUSED32_FAILED = False


def _fused32_ready(device) -> bool:
    """The fused inference kernels of the fp32 path (``sm100_atom`` ``fused_tf32.cu``: AdaLNs + projections, and gate + Wo + AdaLN 2 +
    SwiGLU + Ws with both residuals): two launches in place of eight. ``settings.local_dit_tf32_fused=0`` keeps the unfused kernels; a
    build or load failure warns once and keeps them too (the rest of the fp32 path is unaffected)."""
    global _FUSED32_FAILED
    if _FUSED32_FAILED or (not settings.current().local_dit_tf32_fused):
        return False
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom as rows

    try:
        rows.load_fused32(
            device.index if device.index is not None else torch.cuda.current_device()
        )
    except Exception as exc:  # a toolchain or driver problem keeps the unfused kernels
        _FUSED32_FAILED = True
        warnings.warn(
            f"sm_100a fused fp32 atom kernels unavailable, keeping the unfused fp32 kernels: {exc!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


def _fp32_call(module, single, cond, pair) -> bool:
    """A call for the fp32 path: fp32 single / cond / pair and parameters, no autocast (under autocast the module path runs its products in
    the autocast dtype), ``settings.local_dit_tf32`` not 0. A bf16 model fed fp32 activations keeps the module path."""
    if (single.dtype, cond.dtype, pair.dtype) != (F32, F32, F32):
        return False
    if (not settings.current().local_dit_tf32) or torch.is_autocast_enabled("cuda"):
        return False
    return all(p.dtype == F32 for p in module.parameters())


def _af3_variant(module) -> bool:
    """AlphaFold 3's block: the K/V AdaLN beside the first one (``kv_norm="parallel"``) or the slid key window (``key_window="slide"``);
    or ESMFold2's options on top of it (192 keys, the rank window, RMS / SiLU conditioning, rotary)."""
    return (getattr(module, "kv_norm", "chained") != "chained" or getattr(module, "key_window", "pad") != "pad"
            or _esm_variant(module))


def _esm_variant(module) -> bool:
    """Any of ESMFold2's options (``modules.local_dit.ESMFOLD2``): only the H100 kernels know them."""
    return (getattr(module, "keys", KEYS) != KEYS or getattr(module, "window_half", None) is not None
            or getattr(module, "conditioning", "layernorm") != "layernorm" or getattr(module, "rope", False))


def _b200_af3_servable(module, single, cond, pair) -> bool:
    """AlphaFold 3's K/V norm (``kv_norm="parallel"``) and slid key window (``key_window="slide"``) on B200: the parallel norm in inference
    and training (``pre_fwd`` / ``pre_bwd`` ``-DPARALLEL=1``, the fused / unfused fp32 kernels likewise), the slid window in inference only
    (``sm100_atom_local.slide_edges`` recomputes the windows it moves; the backward kernels know the clipped window only). ESMFold2's
    options are not served here."""
    if _esm_variant(module):
        return False
    if not (single.is_cuda and torch.cuda.get_device_capability(single.device) == (10, 0)):
        return False
    if module.key_window == "slide":
        params = [module.get_parameter(n) for n in (*WEIGHTS, *CROSS_WEIGHTS)]
        if torch.is_grad_enabled() and any(t.requires_grad for t in (single, cond, pair, *params)):
            return False
    return True


def _af3_flags(module) -> tuple[bool, bool]:
    """(parallel K/V norm, slid key window) of a block."""
    return getattr(module, "kv_norm", "chained") == "parallel", getattr(module, "key_window", "pad") == "slide"


def serves(module, single, cond, pair, mask, rope=None) -> bool:
    from miniworld_engine.integrations import local_dit_h100

    if local_dit_h100.matches(module, single, cond, pair, mask, rope):
        return True
    if _af3_variant(module) and not _b200_af3_servable(module, single, cond, pair):
        return False
    qualified = _serves_hopper_training(module, single, cond, pair, mask)
    if qualified is not None:
        return qualified
    qualified = _serves_hopper_inference(module, single, cond, pair, mask)
    if qualified is not None:
        return qualified
    fp32 = _fp32_call(module, single, cond, pair)
    if (
        (_FAILED32 if fp32 else _FAILED)
        or (not settings.current().local_dit_sm100)
        or settings.current().force_portable_kernels
    ):
        return False
    from miniworld_engine.modules.dispatch import KernelBackend

    a, tr = module.attention, module.transition
    if a._backend != KernelBackend.PORTABLE or a.use_qk_norm:
        return False
    cap = torch.cuda.get_device_capability(single.device) if single.is_cuda else None
    if cap != (10, 0):
        return False
    if (single.dtype, cond.dtype, pair.dtype) != (BF, BF, BF) and not fp32:
        return False
    if (
        single.ndim != 4
        or single.shape[1] != 1
        or single.shape[-1] != DS_
        or single.shape[2] == 0
    ):
        return False
    n = single.shape[2]
    if mask is not None and (
        mask.dtype != torch.bool
        or tuple(mask.shape) != (1, n)
        or mask.device != single.device
    ):
        return False
    if tuple(cond.shape) != (*single.shape[:3], DC_) or tuple(pair.shape) != (
        1,
        (n + QUERIES - 1) // QUERIES,
        QUERIES,
        KEYS,
        DP_,
    ):
        return False
    if (
        a.n_head,
        a.to_query.weight.shape[0],
        a.to_bias.weight.shape[1],
        tr.expand_a.weight.shape[0],
    ) != (NH_, DS_, DP_, 2 * DS_):
        return False
    norms = (
        a.ada_ln_in.ln_in,
        a.ada_ln_in.ln_cond,
        a.ln_pair,
        tr.ada_ln_in.ln_in,
        tr.ada_ln_in.ln_cond,
    )
    if module.cross_attention:
        norms += (a.ada_ln_kv.ln_in, a.ada_ln_kv.ln_cond)
    if any(norm.eps != EPS for norm in norms):
        return False
    index = (
        single.device.index
        if single.device.index is not None
        else torch.cuda.current_device()
    )
    return _kernels32_ready(index) if fp32 else _kernels_ready(index)


class _Unserved(NotImplementedError):
    """A call outside the Hopper native contract: the composed path serves it (re-raised only when native is forced)."""


def _serves_hopper_training(module, single, cond, pair, mask):
    """Hopper training: True (native CUDA), None (not Hopper / not training: the other routes decide) or, for a call outside
    the native geometry, None as well -- the module's composed path serves it."""
    try:
        return _require_hopper_training(module, single, cond, pair, mask)
    except _Unserved:
        return None


def _serves_hopper_inference(module, single, cond, pair, mask):
    """Hopper inference: True (native CUDA) or None -- a call outside the native cells falls back to the composed path."""
    try:
        return _require_hopper_inference(module, single, cond, pair, mask)
    except _Unserved:
        return None


def _require_hopper_training(module, single, cond, pair, mask):
    """Required Hopper training geometry: native CUDA or an explicit error."""
    from miniworld_engine.modules.dispatch import KernelBackend

    a, tr = module.attention, module.transition
    if (
        a._backend != KernelBackend.PORTABLE
        or not torch.is_grad_enabled()
        or not single.is_cuda
        or torch.cuda.get_device_capability(single.device) != (9, 0)
    ):
        return None
    def require(ok, detail):
        if not ok:
            raise _Unserved(
                f"H100 local atom DiT training requires {detail}; no composed fallback"
            )

    require(not settings.current().force_portable_kernels, "the CUDA engine backend")
    require(
        module.cross_attention and not a.use_qk_norm, "cross-attention without QK norm"
    )
    require(
        single.ndim == 4
        and single.shape[0] in (1, 5, 48)
        and single.shape[1] == 1
        and single.shape[2] in (4096, 8192)
        and single.shape[3] == DS_,
        "[A,1,N,128] input with A=1/5/48 and N=4096/8192",
    )
    augment, _, n, _ = single.shape
    require(
        tuple(cond.shape) == (augment, 1, n, DC_)
        and tuple(pair.shape) == (1, n // QUERIES, QUERIES, KEYS, DP_),
        "canonical conditioning and trunked pair shapes",
    )
    require(
        all(
            t.dtype == BF and t.device == single.device and t.is_contiguous()
            for t in (single, cond, pair)
        ),
        "contiguous bf16 activations on one CUDA device",
    )
    require(
        (
            a.n_head,
            a.to_query.weight.shape[0],
            a.to_query.weight.shape[1],
            a.to_bias.weight.shape[1],
            a.ada_ln_in.to_scale.weight.shape[1],
            tr.expand_a.weight.shape[0],
        )
        == (NH_, DS_, DS_, DP_, DC_, 2 * DS_),
        "128/128/16 widths, four heads, transition n=2",
    )
    weights = tuple(module.get_parameter(name) for name in (*WEIGHTS, *CROSS_WEIGHTS))
    require(
        all(
            w.dtype in (BF, torch.float32)
            and w.device == single.device
            and w.is_contiguous()
            for w in weights
        ),
        "contiguous bf16 or fp32 parameters on the input device",
    )
    require(
        mask is None
        or (
            mask.dtype == torch.bool
            and tuple(mask.shape) == (1, n)
            and mask.device == single.device
            and mask.is_contiguous()
        ),
        "a contiguous [1,N] bool mask or None",
    )
    norms = (
        a.ada_ln_in.ln_in,
        a.ada_ln_in.ln_cond,
        a.ln_pair,
        tr.ada_ln_in.ln_in,
        tr.ada_ln_in.ln_cond,
        a.ada_ln_kv.ln_in,
        a.ada_ln_kv.ln_cond,
    )
    require(all(norm.eps == EPS for norm in norms), "LayerNorm eps=1e-5")
    index = (
        single.device.index
        if single.device.index is not None
        else torch.cuda.current_device()
    )
    if not _kernels_ready(index):
        raise RuntimeError(
            "H100 local atom DiT CUDA kernels unavailable; no composed fallback"
        )
    return True


def _require_hopper_inference(module, single, cond, pair, mask):
    """The required inference cells use native CUDA, including recorded losses."""
    from miniworld_engine.modules.dispatch import KernelBackend

    a, tr = module.attention, module.transition
    if (
        a._backend != KernelBackend.PORTABLE
        or torch.is_grad_enabled()
        or not single.is_cuda
    ):
        return None
    if torch.cuda.get_device_capability(single.device) != (9, 0):
        return None

    def require(ok, detail):
        if not ok:
            raise _Unserved(
                f"H100 local atom DiT inference requires {detail}; no composed fallback"
            )

    require(not settings.current().force_portable_kernels, "the CUDA engine backend")
    require(
        module.cross_attention and not a.use_qk_norm, "cross-attention without QK norm"
    )
    require(
        single.ndim == 4
        and (single.shape[0], single.shape[2])
        in ((1, 4096), (5, 2048), (5, 4096), (48, 4096)),
        "(A,N)=(1,4096),(5,2048),(5,4096),(48,4096)",
    )
    augment, _, n, _ = single.shape
    require(
        (
            a.n_head,
            a.to_query.weight.shape[0],
            a.to_query.weight.shape[1],
            a.to_bias.weight.shape[1],
            a.ada_ln_in.to_scale.weight.shape[1],
            tr.expand_a.weight.shape[0],
        )
        == (NH_, DS_, DS_, DP_, DC_, 2 * DS_),
        "128/128/16 widths, four heads, transition n=2",
    )
    require(
        tuple(single.shape) == (augment, 1, n, DS_)
        and tuple(cond.shape) == (augment, 1, n, DC_)
        and tuple(pair.shape) == (1, n // QUERIES, QUERIES, KEYS, DP_),
        "canonical single, conditioning and trunked pair shapes",
    )
    require(
        all(
            t.dtype == BF and t.device == single.device and t.is_contiguous()
            for t in (single, cond, pair)
        ),
        "contiguous bf16 activations on one CUDA device",
    )
    weights = tuple(module.get_parameter(name) for name in (*WEIGHTS, *CROSS_WEIGHTS))
    require(
        all(
            w.dtype in (BF, torch.float32)
            and w.device == single.device
            and w.is_contiguous()
            for w in weights
        ),
        "contiguous bf16 or fp32 parameters on the input device",
    )
    require(
        mask is None
        or (
            mask.dtype == torch.bool
            and tuple(mask.shape) == (1, n)
            and mask.device == single.device
            and mask.is_contiguous()
        ),
        "a contiguous [1,N] bool mask or None",
    )
    norms = (
        a.ada_ln_in.ln_in,
        a.ada_ln_in.ln_cond,
        a.ln_pair,
        tr.ada_ln_in.ln_in,
        tr.ada_ln_in.ln_cond,
        a.ada_ln_kv.ln_in,
        a.ada_ln_kv.ln_cond,
    )
    require(all(norm.eps == EPS for norm in norms), "LayerNorm eps=1e-5")
    index = (
        single.device.index
        if single.device.index is not None
        else torch.cuda.current_device()
    )
    if not _kernels_ready(index):
        raise RuntimeError(
            "H100 local atom DiT CUDA kernels unavailable; no composed fallback"
        )
    return True


def _keys(mask, n0, n):
    """The valid-key bytes [n] the attention kernels read (None: every key valid and no padding)."""
    if mask is not None:
        if n == n0:
            return mask[0].contiguous()
        return F.pad(mask[0], (0, n - n0)).contiguous()
    return None if n == n0 else (torch.arange(n, device=_device_of(mask)) < n0)


def _device_of(mask):
    return torch.cuda.current_device() if mask is None else mask.device


def _hoisted(store, tensor, key, build):
    """The table ``build()`` makes from ``tensor``, once per tensor (pointer, version, ``key``): ``_capture.hoisted_table``."""
    return _capture.hoisted_table(store, tensor, key, build, limit=64)


def _tkey(t):
    """A tensor's identity for a cache key: storage address, in-place version, shape, dtype, device."""
    return (t.data_ptr(), t._version, tuple(t.shape), t.dtype, t.device)


def _kv_pack(weights, packed_pre=None):
    """The cross-attention mode's operands: Wm [256, 128] = [to_scale; to_bias] bf16, the scale bias and the LN weight (fp32), Wk, Wv (bf16) and [Wk; Wv] [256, 128].
    Packed once per set of weights (pointer and version), scoped to the CUDA-graph capture like every weight pack.
    Hopper (``packed_pre`` = the pack's ``wp``): the live CUDA pack, Wk / Wv views of ``packed_pre`` and no [Wk; Wv] (None)."""
    hopper = packed_pre is not None and torch.cuda.get_device_capability(
        weights[len(WEIGHTS)].device
    ) == (9, 0)

    def build():
        if hopper:
            from miniworld_engine.kernels.augmented_attention.cuda import (
                sm100_atom_local as local,
            )

            return (*local.kv_pack_hopper(weights[len(WEIGHTS) :], packed_pre), None)
        gamma, w_scale, b_scale, w_bias = weights[len(WEIGHTS) :]
        wk, wv = (
            weights[WEIGHTS.index("attention.to_key.weight")],
            weights[WEIGHTS.index("attention.to_value.weight")],
        )
        wk, wv = wk.to(BF), wv.to(BF)
        return (
            torch.cat([w_scale, w_bias]).to(BF),
            b_scale.float(),
            gamma.float(),
            wk,
            wv,
            torch.cat([wk, wv]),
        )

    tail = [
        *weights[len(WEIGHTS) :],
        weights[WEIGHTS.index("attention.to_key.weight")],
        weights[WEIGHTS.index("attention.to_value.weight")],
    ]
    # the entry keeps the weights alive: a freed address cannot alias another tensor
    return _capture.lookup(
        _KV,
        (hopper, *((t.data_ptr(), t._version) for t in tail)),
        lambda: (build(), tuple(tail)),
        limit=16,
    )[0]


def _cond_tables(
    cond, c2_fn, n, weights, wc, cross, save, wp=None, packed_infer=False
):
    """The conditioning tables of a call: mod [rows, 768] (cond_fwd) and, in the cross-attention mode, cn (the LN of the conditioning) and the
    key / value modulation mkv [rows, 256]. They depend on the conditioning tensor and the weights only -- the same at every block call that
    reads one conditioning (a sampler's steps), so an inference call (``save`` False) takes them from the hoist: made once per conditioning
    tensor (pointer, version) and weights, scoped to the CUDA-graph capture (``kernels._capture.lookup_inputs``).
    Hopper: ``wp`` (the pack's input-projection operands) feeds the KV pack, ``packed_infer`` picks the persistent conditioning
    schedule."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom as rows
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm100_atom_local as local,
    )

    def build():
        c2 = c2_fn()
        mod = rows.cond_fwd(
            c2,
            *wc,
            **(
                {"hopper_waves": 0.5 if packed_infer else 2.0} if wp is not None else {}
            ),
        )
        if not cross:
            return mod, None, None
        wm, _, gkv, *_ = _kv_pack(weights, wp)
        cn = local.cond_ln(c2, gkv)
        return mod, cn, cn @ wm.t()

    if save or not _hoisting():
        return build()
    return _hoisted(
        _HOIST_COND,
        cond,
        (tuple((w.data_ptr(), w._version) for w in weights), n),
        build,
    )


def _pair_bias(pair, n, weights, save):
    """The windowed pair bias LN(z) Wb of the trunked pair, once per pair tensor in an inference call (it depends on the pair and two weights)."""
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm100_atom_local as local,
    )

    def build():
        return local.pair_bias_fwd(_trunk(pair, n), weights[_GAMMA], weights[_WBIAS])

    if save or not _hoisting():
        return build()
    g, wb = weights[_GAMMA], weights[_WBIAS]
    return _hoisted(
        _HOIST_BIAS,
        pair,
        ((g.data_ptr(), g._version, wb.data_ptr(), wb._version), n),
        build,
    )


def _trunk(pair, n):
    """[1, nwin, 32, 128, 16] -> [n / 32, 32, 128, 16]: zero windows appended for the padded atoms."""
    z = pair[0]
    return (
        z
        if z.shape[0] == n // QUERIES
        else F.pad(z, (0, 0, 0, 0, 0, 0, 0, n // QUERIES - z.shape[0])).contiguous()
    )


# ------------------------------------------------------------------------------------------------------------ the two ops
def _fwd_fake(single, cond, pair, mask, weights, save, parallel=False, slide=False):
    out = torch.empty_like(single)
    if not save:
        return [out]
    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    rows = a * n

    def tile(width=DS_):
        return single.new_empty((rows, width))

    saved = [
        out,
        tile(6 * DS_),
        tile(),
        tile(),
        tile(),
        tile(),
        tile(),
        single.new_empty((NH_, n // QUERIES, QUERIES, KEYS), dtype=torch.float32),
        tile(),
        single.new_empty((a, NH_, n), dtype=torch.float32),
        tile(),
        tile(),
        tile(),
        tile(),
    ]
    return (
        [*saved, tile(), tile(2 * DS_), tile()]
        if len(weights) > len(WEIGHTS)
        else saved
    )


@opaque(fake=_fwd_fake, name="local_dit_block_fwd")
def _block_fwd(
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    weights: list[torch.Tensor],
    save: bool,
    parallel: bool = False,
    slide: bool = False,
) -> list[torch.Tensor]:
    """The block output [A, 1, N, 128] bf16 and, when ``save``, the activations the backward reads (mod, q, k, v, sg, x1, bias, O, LSE,
    u, a2, x2, t: all fresh tensors). ``parallel``: AlphaFold 3's K/V AdaLN of the raw atoms; ``slide``: its slid key window (inference)."""
    assert not (slide and save), "the slid key window is served for inference only"
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom as rows
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm100_atom_local as local,
    )

    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    m = a * n
    with torch.cuda.device(single.device):
        cross = len(weights) > len(WEIGHTS)
        hopper = rows._hopper(
            single.device.index
            if single.device.index is not None
            else torch.cuda.current_device()
        )
        packed_infer = not save and cross and a == 5 and n0 in (2048, 4096) and hopper
        wc, wp, wo, _ = _weights(weights[: len(WEIGHTS)], backward=save)
        s2 = _padded(single, n).reshape(m, DS_).contiguous()
        mod, cn, mkv = _cond_tables(
            cond,
            lambda: _padded(cond, n).reshape(m, DC_).contiguous(),
            n,
            weights,
            wc,
            cross,
            save,
            wp=wp if hopper else None,
            packed_infer=packed_infer,
        )
        if (
            cross and hopper
        ):  # Hopper: pre_fwd's own k / v are not used; keys and values from a second AdaLN of x1 (kv_fwd) and cuBLAS
            _, bs, _, wk, wv, _ = _kv_pack(weights, wp)
            q, k, v, sg, x1 = rows.pre_fwd(s2, mod, *wp, save=True, cross=True)
            xkv = local.kv_fwd(x1, mkv, bs)
            k, v = xkv @ wk.t(), xkv @ wv.t()
        elif cross:  # keys and values from a second AdaLN of x1, inside pre_fwd
            bs = _kv_pack(weights)[1]
            w_pre, b_pre = wp
            q, k, v, sg, x1, xkv = rows.pre_fwd_kv(
                s2, mod, w_pre, b_pre, mkv, bs, save=save, parallel=parallel
            )
        else:
            q, k, v, sg, x1 = rows.pre_fwd(s2, mod, *wp, save=save)
        bias = _pair_bias(pair, n, weights, save)
        o, lse = local.attn_fwd(
            q.view(a, n, DS_),
            k.view(a, n, DS_),
            v.view(a, n, DS_),
            bias,
            _keys(mask, n0, n),
        )
        if slide:  # the windows the slid key window moves, over their slid keys
            local.slide_edges(q.view(a, n, DS_), k.view(a, n, DS_), v.view(a, n, DS_), bias, _keys(mask, n0, n), o, n0)
        out, u, a2, x2, t = rows.post_fwd(
            s2,
            o.view(m, DS_),
            sg,
            mod,
            *wo,
            save=save,
            **({"packed_hopper": packed_infer} if hopper else {}),
        )
    out = _unpadded(out.view(a, 1, n, DS_), single.shape)
    if not save:
        return [out]
    saved = [out, mod, q, k, v, sg, x1, bias, o.view(m, DS_), lse, u, a2, x2, t]
    return [*saved, cn, mkv, xkv] if cross else saved


def _fused_wgrads(spec, weights, names, f32, dev, rows_r):
    """The weight gradients ``spec`` names -- (A, B, [(weight name, 128-row block), ...]) per product D = A^T B -- in one split-K launch
    (``sm100_atom.AtomDw``), each written straight into a fresh buffer of its weight's shape and dtype; -> {name: gradient}. A weight made of
    several 128-row blocks (expand_a / expand_b: 256 rows) lists the block index of each destination."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom as rows

    wmap = dict(zip(names, weights, strict=True))
    out = {}
    problems = []
    for A, B, dests in spec:
        ds = []
        for name, blk in dests:
            if name not in out:
                out[name] = torch.empty(
                    wmap[name].shape, device=dev, dtype=wmap[name].dtype
                )
            ds.append(
                out[name].view(
                    -1, DS_ if out[name].shape[-1] == DS_ else out[name].shape[-1]
                )[blk * 128 : (blk + 1) * 128]
            )
        problems.append((A, B, ds))
    rows.atom_dw(
        dev.index if dev.index is not None else torch.cuda.current_device(), f32
    )(problems, rows_r)
    return out


def _wgrad(dout, x):
    """dout^T x, fp32: the weight gradients stay unrounded for fp32 parameters (an fp32 master); bf16 ones round once, as before."""
    return torch.mm(dout.t(), x, out_dtype=torch.float32)


def _bwd_fake(dy, single, cond, pair, mask, weights, saved, parallel=False):
    return [
        torch.empty_like(single),
        torch.empty_like(cond),
        torch.empty_like(pair),
        *(torch.empty_like(w) for w in weights),
    ]


@opaque(fake=_bwd_fake, name="local_dit_block_bwd")
def _block_bwd(
    dy: torch.Tensor,
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    weights: list[torch.Tensor],
    saved: list[torch.Tensor],
    parallel: bool = False,
) -> list[torch.Tensor]:
    """d single, d cond, d pair, then one gradient per weight in ``WEIGHTS`` order, in the weight's dtype (all fresh tensors).
    ``parallel``: AlphaFold 3's K/V AdaLN of the raw atoms."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom as rows
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm100_atom_local as local,
    )

    mod, q, k, v, sg, x1, bias, o, lse, u, a2, x2, t, *extra = saved
    cross = bool(extra)
    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    m = a * n
    dev = single.device
    with torch.cuda.device(dev):
        hopper = rows._hopper(
            dev.index if dev.index is not None else torch.cuda.current_device()
        )
        if hopper and (
            dy.dtype != BF
            or dy.device != dev
            or dy.shape != single.shape
            or not dy.is_contiguous()
        ):
            raise NotImplementedError(
                "H100 local DiT backward requires a contiguous bf16 upstream gradient matching single"
            )
        wc, wp, wo, wb = _weights(weights[: len(WEIGHTS)], backward=True)
        s2 = _padded(single, n).reshape(m, DS_).contiguous()
        c2 = _padded(cond, n).reshape(m, DC_).contiguous()
        dy = _padded(dy, n).reshape(m, DS_).to(BF).contiguous()
        dmod = torch.empty(
            m, 6 * DS_, device=dev, dtype=BF
        )  # [dsc1 | dbi1 | dsc2 | dbi2 | dos | dts]
        dp = torch.empty(m, 4 * DS_, device=dev, dtype=BF)  # [dq | dk | dv | dg]
        allocate = torch.empty if hopper else torch.zeros
        dbias_cols = allocate(6 * DS_, device=dev)
        dbq = allocate(DS_, device=dev)
        dgamma_cond = allocate(2 * DS_, device=dev)
        dgparts = (
            torch.empty((m + 63) // 64, 2 * DS_, device=dev) if hopper else dgamma_cond
        )
        dd = torch.empty(a, NH_, n, device=dev)
        dt, hh, dab = rows.tr_bwd_gate(
            dy, t, mod, x2, wo[1], wb["WsT"], dmod, dbias_cols[640:]
        )
        da, du, gated, do = rows.post_bwd(
            dab,
            dy,
            a2,
            mod,
            u,
            sg,
            o,
            wb["WuT"],
            wb["WoT"],
            dmod,
            dp,
            dd,
            dbias_cols,
            n,
        )
        db = local.attn_bwd(
            q.view(a, n, DS_),
            k.view(a, n, DS_),
            v.view(a, n, DS_),
            do.view(a, n, DS_),
            bias,
            lse,
            dd,
            dp,
            _keys(mask, n0, n),
        )
        dz, dgamma, dwb = local.pair_bias_bwd(
            _trunk(pair, n), weights[_GAMMA], weights[_WBIAS], db
        )
        dmkv = dbs = dg_kv = None
        if cross:  # the K / V branch: dxkv -> its AdaLN -> dx1 (through AdaLN 1 below) and the conditioning
            cn, mkv, xkv = extra
        if hopper:
            if cross:
                wm, bs, gkv, wk, wv, _ = _kv_pack(weights, wp)
                dk, dv = dp[:, DS_ : 2 * DS_], dp[:, 2 * DS_ : 3 * DS_]
                dxkv = torch.addmm(dk @ wk, dv, wv)
                dwk, dwv = _wgrad(dk, xkv), _wgrad(dv, xkv)
                dx1_kv, dmkv = local.kv_bwd(x1, mkv, bs, dxkv)
                dwm = _wgrad(dmkv, cn)
                dbs = local.kv_bias_hopper(dmkv)
                dg_kv = allocate(DS_, device=dev)
                dc_kv = local.cond_ln_bwd(c2, dmkv @ wm, gkv, dg_kv)
                local.zero_kv_hopper(
                    dp
                )  # pre_bwd's dgrad then covers q and the gate only
            ds = rows.pre_bwd(dp, s2, mod, da, wb["WpT"], dmod, dbias_cols, dbq)
            if cross:
                local.adaln1_extra(s2, mod, dx1_kv, ds, dmod, dbias_cols[0:DS_])
            dc, cn1, cn2 = rows.cond_bwd(dmod, c2, wb["WmodT"], wc[2], wc[3], dgparts)
            if cross:
                local.add_cond_hopper(dc, dc_kv)
            rows.finish_columns_hopper(dmod, dp, dgparts, dbias_cols, dbq, dgamma_cond)
            dws, dwu, dwo, dwp = (
                _wgrad(dt, hh),
                _wgrad(dab, x2),
                _wgrad(du, gated),
                _wgrad(dp, x1),
            )
            dw1, dw2, dw3 = (
                _wgrad(dmod[:, 0:256], cn1),
                _wgrad(dmod[:, 256:512], cn2),
                _wgrad(dmod[:, 512:768], c2),
            )
        else:
            if cross:
                wm, bs, gkv, *_ = _kv_pack(weights)
                dbs = torch.zeros(DS_, device=dev)
                dg_kv = torch.zeros(DS_, device=dev)
                # dk Wk + dv Wv through the K / V AdaLN into dx1, then AdaLN 1: one kernel; dmkv = [d pre-sigmoid scale | dxkv]
                ds, dmkv = rows.pre_bwd_kv(
                    dp, s2, mod, da, wb["WpT"], dmod, dbias_cols, dbq, x1, mkv, bs, dbs, parallel=parallel
                )
            else:
                ds = rows.pre_bwd(dp, s2, mod, da, wb["WpT"], dmod, dbias_cols, dbq)
            dc, cn1, cn2 = rows.cond_bwd(
                dmod, c2, wb["WmodT"], wc[2], wc[3], dgamma_cond
            )
            if cross:
                local.cond_ln_bwd(
                    c2, dmkv @ wm, gkv, dg_kv, acc=dc
                )  # dc += the K / V modulation's gradient through LN(c) * gamma
            A_ = "attention."
            spec = [
                (dt, hh, [("transition.squeeze.weight", 0)]),
                (
                    dab,
                    x2,
                    [
                        ("transition.expand_a.weight", 0),
                        ("transition.expand_a.weight", 1),
                        ("transition.expand_b.weight", 0),
                        ("transition.expand_b.weight", 1),
                    ],
                ),
                (du, gated, [(A_ + "to_out.weight", 0)]),
                (
                    dmod[:, 0:256],
                    cn1,
                    [
                        (A_ + "ada_ln_in.to_scale.weight", 0),
                        (A_ + "ada_ln_in.to_bias.weight", 0),
                    ],
                ),
                (
                    dmod[:, 256:512],
                    cn2,
                    [
                        ("transition.ada_ln_in.to_scale.weight", 0),
                        ("transition.ada_ln_in.to_bias.weight", 0),
                    ],
                ),
                (
                    dmod[:, 512:768],
                    c2,
                    [(A_ + "to_scale.weight", 0), ("transition.to_scale.weight", 0)],
                ),
            ]
            if cross:
                spec += [
                    (dp[:, 0:DS_], x1, [(A_ + "to_query.weight", 0)]),
                    (dp[:, 3 * DS_ : 4 * DS_], x1, [(A_ + "to_gate.weight", 0)]),
                    (
                        dp[:, DS_ : 3 * DS_],
                        xkv,
                        [(A_ + "to_key.weight", 0), (A_ + "to_value.weight", 0)],
                    ),
                    (
                        dmkv,
                        cn,
                        [
                            (A_ + "ada_ln_kv.to_scale.weight", 0),
                            (A_ + "ada_ln_kv.to_bias.weight", 0),
                        ],
                    ),
                ]
            else:
                spec += [
                    (
                        dp,
                        x1,
                        [
                            (A_ + "to_query.weight", 0),
                            (A_ + "to_key.weight", 0),
                            (A_ + "to_value.weight", 0),
                            (A_ + "to_gate.weight", 0),
                        ],
                    )
                ]
            names = WEIGHTS + CROSS_WEIGHTS[: len(weights) - len(WEIGHTS)]
            fused = _fused_wgrads(spec, weights, names, False, dev, m)
    grads: dict[str, torch.Tensor] = {
        "attention.ada_ln_in.ln_cond.weight": dgamma_cond[:128],
        "attention.ada_ln_in.to_scale.bias": dbias_cols[0:128],
        "attention.to_query.bias": dbq,
        "attention.ln_pair.weight": dgamma,
        "attention.to_bias.weight": dwb,
        "attention.to_scale.bias": dbias_cols[512:640],
        "transition.ada_ln_in.ln_cond.weight": dgamma_cond[128:],
        "transition.ada_ln_in.to_scale.bias": dbias_cols[256:384],
        "transition.to_scale.bias": dbias_cols[640:768],
    }
    names = WEIGHTS + CROSS_WEIGHTS[: len(weights) - len(WEIGHTS)]
    if hopper:
        grads.update(
            {
                "attention.ada_ln_in.to_scale.weight": dw1[:128],
                "attention.ada_ln_in.to_bias.weight": dw1[128:],
                "attention.to_query.weight": dwp[0:128],
                "attention.to_key.weight": dwp[128:256],
                "attention.to_value.weight": dwp[256:384],
                "attention.to_gate.weight": dwp[384:512],
                "attention.to_out.weight": dwo,
                "attention.to_scale.weight": dw3[:128],
                "transition.ada_ln_in.to_scale.weight": dw2[:128],
                "transition.ada_ln_in.to_bias.weight": dw2[128:],
                "transition.expand_a.weight": dwu[:256],
                "transition.expand_b.weight": dwu[256:],
                "transition.squeeze.weight": dws,
                "transition.to_scale.weight": dw3[128:],
            }
        )
        if cross:
            grads["attention.to_key.weight"], grads["attention.to_value.weight"] = (
                dwk,
                dwv,
            )
            assert (
                dbs is not None and dg_kv is not None
            )  # both are set by the cross branch above
            grads.update(
                {
                    "attention.ada_ln_kv.ln_cond.weight": dg_kv,
                    "attention.ada_ln_kv.to_scale.weight": dwm[:DS_],
                    "attention.ada_ln_kv.to_scale.bias": dbs,
                    "attention.ada_ln_kv.to_bias.weight": dwm[DS_:],
                }
            )
        per_weight = rows.grad_pack_hopper(
            [grads[name] for name in WEIGHTS], weights[: len(WEIGHTS)]
        )
        if cross:
            per_weight += local.cross_grad_pack_hopper(
                [grads[name] for name in CROSS_WEIGHTS], weights[len(WEIGHTS) :]
            )
        dpair = dz.view(pair.shape)
    else:
        if cross:
            assert (
                dbs is not None and dg_kv is not None
            )  # both are set by the cross branch above
            grads.update(
                {
                    "attention.ada_ln_kv.ln_cond.weight": dg_kv,
                    "attention.ada_ln_kv.to_scale.bias": dbs,
                }
            )
        per_weight = [
            fused[name]
            if name in fused
            else grads[name].to(w.dtype).reshape(w.shape).clone()
            for name, w in zip(names, weights, strict=True)
        ]
        dpair = dz[: pair.shape[1]].reshape(pair.shape).clone()
    return [
        _unpadded(ds.view(a, 1, n, DS_), single.shape),
        _unpadded(dc.view(a, 1, n, DC_), cond.shape),
        dpair,
        *per_weight,
    ]


# ------------------------------------------------------------------------------------------------------------ autograd
class _LocalBlock(torch.autograd.Function):
    """The whole block forward and backward on the sm_100a kernels; gradients for single, cond, pair and every weight. ``parallel``:
    AlphaFold 3's K/V AdaLN of the raw atoms."""

    @staticmethod
    def forward(ctx, parallel, mask, single, cond, pair, *weights):
        out, *saved = _block_fwd(single, cond, pair, mask, list(weights), True, parallel)
        ctx.save_for_backward(mask, single, cond, pair, *saved, *weights)
        ctx.n_saved = len(saved)
        ctx.parallel = parallel
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        mask, single, cond, pair, *rest = ctx.saved_tensors
        saved, weights = rest[: ctx.n_saved], rest[ctx.n_saved :]
        grads = _block_bwd(dy, single, cond, pair, mask, list(weights), list(saved), ctx.parallel)
        if single.is_cuda and torch.cuda.get_device_capability(single.device) == (9, 0):
            from miniworld_engine.integrations.token_dit_train import (
                _activate_native_caller,
            )

            _activate_native_caller(force=True)
        return (None, None, *grads)


def _infer(single, cond, pair, mask, weights, parallel=False, slide=False):
    return _block_fwd(single, cond, pair, mask, list(weights), False, parallel, slide)[0]


def block(module, single, cond, pair, mask=None, rope=None):
    """The block's output stream (both residuals included), on the sm_100a kernels; ``serves()`` must have accepted the call."""
    from miniworld_engine.integrations import local_dit_h100

    if local_dit_h100.matches(module, single, cond, pair, mask, rope):
        return local_dit_h100.run_block(module, single, cond, pair, mask, rope)
    weights = [
        module.get_parameter(name)
        for name in (*WEIGHTS, *(CROSS_WEIGHTS if module.cross_attention else ()))
    ]
    parallel, slide = _af3_flags(module)
    if single.dtype == F32:  # the fp32 path (TF32 tensor cores)
        if torch.is_grad_enabled() and any(
            t.requires_grad for t in (single, cond, pair, *weights)
        ):
            return _LocalBlock32.apply(parallel, mask, single, cond, pair, *weights)
        return _infer32(single, cond, pair, mask, weights, parallel, slide)
    if torch.is_grad_enabled() and any(
        t.requires_grad for t in (single, cond, pair, *weights)
    ):
        out = _LocalBlock.apply(parallel, mask, single, cond, pair, *weights)
        if single.is_cuda and torch.cuda.get_device_capability(single.device) == (9, 0):
            from miniworld_engine.integrations.token_dit_train import (
                _native_caller_hooks,
            )

            _native_caller_hooks(single, cond, pair, *weights, force=True)
        return out
    return _infer(single, cond, pair, mask, weights, parallel, slide)


# ============================================================================================================ the fp32 path (TF32)
# fp32 single / cond / pair and fp32 parameters: every stage on CUDA, products on tcgen05 kind::tf32 (``sm100_atom.gemm32``, the
# ``sm100_atom_local`` *_tf32 attention kernels), the residual stream, LayerNorms, gates, softmax and every saved activation fp32; the
# weight gradients are cuBLAS GEMMs with TF32 forced on (restored after), whatever the caller's allow_tf32. Every MMA operand is rounded
# to tf32 to nearest (RNA) -- kind::tf32 truncates, a one-sided error 5x the cuBLAS TF32 path's at the block output: the weight packs here,
# the GEMMs' A operands in shared memory, q / k / v in the projection's epilogue, dO in gate_bwd, P / dS in the attention kernels; every
# stored activation other than q / k / v (and dO) stays exact fp32.
#
# Layouts of the fp32 path (its own, not the bf16 kernels'):
#   mod32 [M, 768] = [s1 | bi1 | s2 | bi2 | so | st]: three GEMMs (cn1, cn2, c against [Wsc1; Wbi1], [Wsc2; Wbi2], [Wos; Wts]); the scales
#                    and gates (blocks 0, 2, 4, 5) leave as sigmoids
#   P32 [M, 512]   = [q | k | v | g] (one GEMM of x1), or [q | g | k | v] in the cross mode (x1 -> q | g, xkv -> k | v)
#   u [M, 512]     = the transition's [a | b] products in the interleaved order [a_0 | b_0 | a_1 | b_1] (the SwiGLU epilogue's tiles)
#   mkv [M, 256]   = [sigmoid(scale) | shift] of the cross mode's K / V AdaLN (the same convention as mod32)
_S1, _B1, _S2, _B2, _SO, _ST = (slice(DS_ * i, DS_ * (i + 1)) for i in range(6))
_SIG_MOD = 0b110101  # sigmoid on mod32 blocks 0, 2, 4, 5
_P_COLS = {
    False: (0, DS_, 2 * DS_, 3 * DS_),
    True: (0, 2 * DS_, 3 * DS_, DS_),
}  # (q, k, v, g) columns of P32
_P_RND = {
    False: 0b0111,
    True: 0b1101,
}  # P32 blocks stored tf32-rounded: q, k, v (only MMA operands), not g


_ZEROS32: dict = {}


def _zeros32(device):
    """A persistent fp32 zero vector [128] per device (the zero bias blocks of the packs). Made eagerly and kept; inside a CUDA-graph capture
    with none kept yet, a fresh one (its memset is recorded, and it is not kept: its contents exist only once the graph replays)."""
    z = _ZEROS32.get(device)
    if z is None:
        z = torch.zeros(DS_, device=device, dtype=F32)
        if not torch.cuda.is_current_stream_capturing():
            _ZEROS32[device] = z
    return z


def _pack32(weights):
    """The fp32 path's forward weight operands (K-major [n, k] for gemm32), in ONE flat fp32 buffer: one ``torch.cat`` of the parameters and
    an in-place tf32 rounding (RNA) of its matrix part -- a kind::tf32 MMA truncates its operands; the biases and LayerNorm weights stay
    fp32. Three kernels per pack (a CUDA-graph replay that repacks runs only these; the previous per-tensor packing recorded ~30 small
    kernels per block). Matrix rows, in order: wmod [768] (+ wm [256] right after it in the cross mode: ``wcond`` = [wmod; wm] is one
    operand), wp [512], wu [512], wo [128], ws [128 x 256]; then bmod [768], bm [256] (cross), bp [512], g1, g2, gkv (cross) [128 each].
    The backward's transposed operands are ``_pack32_bwd``'s (training only)."""
    cross = len(weights) > len(WEIGHTS)
    w = dict(
        zip(
            (*WEIGHTS, *CROSS_WEIGHTS[: len(weights) - len(WEIGHTS)]),
            weights,
            strict=True,
        )
    )

    def m(name):
        return w[name].detach().float().reshape(-1)

    with torch.no_grad():
        z = _zeros32(weights[0].device)
        wa, wb = (
            w["transition.expand_a.weight"].detach().float(),
            w["transition.expand_b.weight"].detach().float(),
        )
        proj = (
            ("query", "gate", "key", "value")
            if cross
            else ("query", "key", "value", "gate")
        )
        mats = [
            m("attention.ada_ln_in.to_scale.weight"),
            m("attention.ada_ln_in.to_bias.weight"),
            m("transition.ada_ln_in.to_scale.weight"),
            m("transition.ada_ln_in.to_bias.weight"),
            m("attention.to_scale.weight"),
            m("transition.to_scale.weight"),
        ]
        if cross:
            mats += [
                m("attention.ada_ln_kv.to_scale.weight"),
                m("attention.ada_ln_kv.to_bias.weight"),
            ]
        mats += [m(f"attention.to_{p}.weight") for p in proj]
        mats += [
            wa[:DS_].reshape(-1),
            wb[:DS_].reshape(-1),
            wa[DS_:].reshape(-1),
            wb[DS_:].reshape(-1),
        ]
        mats += [m("attention.to_out.weight"), m("transition.squeeze.weight")]
        vecs = [
            m("attention.ada_ln_in.to_scale.bias"),
            z,
            m("transition.ada_ln_in.to_scale.bias"),
            z,
            m("attention.to_scale.bias"),
            m("transition.to_scale.bias"),
        ]
        if cross:
            vecs += [m("attention.ada_ln_kv.to_scale.bias"), z]
        vecs += [
            m("attention.to_query.bias"),
            z,
            z,
            z,
            m("attention.ada_ln_in.ln_cond.weight"),
            m("transition.ada_ln_in.ln_cond.weight"),
        ]
        if cross:
            vecs.append(m("attention.ada_ln_kv.ln_cond.weight"))
        flat = torch.cat(mats + vecs)
        nmat = sum(t.numel() for t in mats)
        flat[:nmat].copy_(tf32_grid(flat[:nmat]))  # the matrices on the TF32 grid, to nearest

        off = [0]

        def take(rows, cols=DS_):
            t = flat[off[0] : off[0] + rows * cols].view(rows, cols)
            off[0] += rows * cols
            return t

        rc = 8 if cross else 6
        wcond = take(rc * DS_)
        wp, wu, wo, ws = take(4 * DS_), take(4 * DS_), take(DS_), take(DS_, 2 * DS_)
        bcond = flat[off[0] : off[0] + rc * DS_]
        off[0] += rc * DS_
        bp = flat[off[0] : off[0] + 4 * DS_]
        off[0] += 4 * DS_
        g1, g2 = flat[off[0] : off[0] + DS_], flat[off[0] + DS_ : off[0] + 2 * DS_]
        off[0] += 2 * DS_
        pk = SimpleNamespace(
            wmod=wcond[: 6 * DS_],
            bmod=bcond[: 6 * DS_],
            g1=g1,
            g2=g2,
            wp=wp,
            bp=bp,
            wo=wo,
            wu=wu,
            ws=ws,
            gkv=None,
            wcond=wcond,
            bcond=bcond,
            flat=flat,
        )
        if cross:
            pk.wm, pk.bm, pk.gkv = (
                wcond[6 * DS_ :],
                bcond[6 * DS_ :],
                flat[off[0] : off[0] + DS_],
            )
            off[0] += DS_
        assert off[0] == flat.numel(), (off[0], flat.numel())
    return pk


def _pack32_bwd(weights):
    """The backward's operands: the forward pack and its matrices transposed (wmodT [128, 768], wpT [128, 512], woT, wuT [128, 512],
    wsT [256, 128], wmT [128, 256]), all tf32-rounded (they are transposes of the rounded forward pack)."""
    pk = _weights32(weights)

    def tr(t):
        return t.t().contiguous()

    with torch.no_grad():
        bk = SimpleNamespace(**vars(pk))
        bk.wmodT, bk.wpT, bk.woT, bk.wuT, bk.wsT = (
            tr(pk.wmod),
            tr(pk.wp),
            tr(pk.wo),
            tr(pk.wu),
            tr(pk.ws),
        )
        bk.wmT = tr(pk.wm) if pk.gkv is not None else None
    return bk


def _cached32(build, weights):
    """``build(weights)``, reused while every weight's tensor object and version are unchanged; scoped to the CUDA-graph capture
    (``kernels._capture``) as the bf16 packs are -- except under ``kernels._capture.static_inputs()``, whose contract (the weights AND the
    conditioning / pair keep their contents between replays) covers the weights: a capture then serves from, and fills, the eager entry and a
    replay runs no packing kernel (``static_weights()`` has the same effect through ``scoped``)."""
    return _capture.param_forms(("local_dit_tf32", build.__name__), weights, lambda: build(weights), inputs_scope=True)


def _weights32(weights):
    """The forward pack (``_pack32``), cached (``_cached32``)."""
    return _cached32(_pack32, weights)


def _weights32_bwd(weights):
    """The backward pack (``_pack32_bwd``), cached (``_cached32``)."""
    return _cached32(_pack32_bwd, weights)


def _cond_tables32(cond, c2_fn, n, weights, pk, cross, save):
    """The fp32 conditioning tables of a call: mod32 [rows, 768] and, in the cross mode, mkv [rows, 256]; with ``save`` also the
    LayerNorms cn1, cn2 (and cnkv) the weight gradients read. An inference call takes them from the hoist (``_cond_tables``)."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom as rows

    def build():
        c2 = c2_fn()
        if not save:  # inference: two launches, the tables only
            # cat = [cn1 | cn2 | c | cnkv] (one LayerNorm launch), then ONE grouped GEMM: column group g (256 wide) of [mod | mkv] reads
            # cat's 128-column block g against wcond's rows 256 g .. -- mod and mkv are column views of one [M, 1024] (768) table
            rc = 8 if cross else 6
            cat = torch.empty(c2.shape[0], rc // 2 * DS_, device=c2.device, dtype=F32)
            rows.ln32(c2, pk.g1, pk.g2, pk.gkv if cross else None, out=cat)
            tab = rows.gemm32(
                cat,
                pk.wcond,
                k=DS_,
                a_grp=2 * DS_,
                bias=pk.bcond,
                sigmask=_SIG_MOD | (0b1000000 if cross else 0),
            )
            return (
                tab[:, : 6 * DS_],
                None,
                None,
                None,
                (tab[:, 6 * DS_ :] if cross else None),
            )
        cn1, cn2, cnkv = rows.ln32(c2, pk.g1, pk.g2, pk.gkv if cross else None)
        mod = torch.empty(c2.shape[0], 6 * DS_, device=c2.device, dtype=F32)
        for i, x in enumerate((cn1, cn2, c2)):
            rows.gemm32(
                x,
                pk.wmod,
                mod,
                n=2 * DS_,
                b_row0=2 * DS_ * i,
                ocol0=2 * DS_ * i,
                bias=pk.bmod,
                sigmask=_SIG_MOD,
            )
        mkv = rows.gemm32(cnkv, pk.wm, bias=pk.bm, sigmask=1) if cross else None
        return mod, cn1, cn2, cnkv, mkv

    if save or not _hoisting():
        return build()
    return _hoisted(
        _HOIST_COND,
        cond,
        ("tf32", tuple((w.data_ptr(), w._version) for w in weights), n),
        build,
    )


def _pair_bias32(pair, n, weights, save):
    """The windowed pair bias of an fp32 trunked pair, hoisted per pair tensor in an inference call (``_pair_bias``)."""
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm100_atom_local as local,
    )

    def build():
        return local.pair_bias_fwd32(_trunk(pair, n), weights[_GAMMA], weights[_WBIAS])

    if save or not _hoisting():
        return build()
    g, wb = weights[_GAMMA], weights[_WBIAS]
    return _hoisted(
        _HOIST_BIAS,
        pair,
        ("tf32", (g.data_ptr(), g._version, wb.data_ptr(), wb._version), n),
        build,
    )


def _fwd_fake32(single, cond, pair, mask, weights, save, parallel=False, slide=False):
    out = torch.empty_like(single)
    if not save:
        return [out]
    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    m = a * n

    def tile(width=DS_):
        return single.new_empty((m, width))

    saved = [
        out,
        tile(6 * DS_),
        tile(),
        tile(),
        tile(),
        tile(4 * DS_),
        single.new_empty((NH_, n // QUERIES, QUERIES, KEYS)),
        single.new_empty((a, n, DS_)),
        single.new_empty((a, NH_, n)),
        tile(),
        tile(),
        tile(),
        tile(),
        tile(4 * DS_),
        tile(),
    ]
    return (
        [*saved, tile(), tile(2 * DS_), tile()]
        if len(weights) > len(WEIGHTS)
        else saved
    )


@opaque(fake=_fwd_fake32, name="local_dit_block_fwd_tf32")
def _block_fwd32(
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    weights: list[torch.Tensor],
    save: bool,
    parallel: bool = False,
    slide: bool = False,
) -> list[torch.Tensor]:
    """The fp32 path's forward: the block output [A, 1, N, 128] fp32 and, when ``save``, the activations the backward reads (mod32, cn1,
    cn2, x1, P32, bias, O, LSE, gated, y, a1, x2, u, t; cross mode: cnkv, mkv, xkv -- all fresh fp32 tensors). ``parallel`` / ``slide``:
    AlphaFold 3's K/V AdaLN of the raw atoms / slid key window (inference)."""
    assert not (slide and save), "the slid key window is served for inference only"
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom as rows
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm100_atom_local as local,
    )

    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    m = a * n
    dev = single.device
    with torch.cuda.device(dev):
        cross = len(weights) > len(WEIGHTS)
        pk = _weights32(weights)
        qc, kc, vc, gc = _P_COLS[cross]
        s2 = _padded(single, n).reshape(m, DS_).contiguous()
        mod, cn1, cn2, cnkv, mkv = _cond_tables32(
            cond,
            lambda: _padded(cond, n).reshape(m, DC_).contiguous(),
            n,
            weights,
            pk,
            cross,
            save,
        )
        if not save and _fused32_ready(
            dev
        ):  # inference: pre (AdaLNs + projections), attention, post (the rest)
            p = rows.pre32(
                s2,
                mod[:, _S1],
                mod[:, _B1],
                pk.wp,
                pk.bp[:DS_],
                _P_RND[cross],
                mks=mkv[:, :DS_] if cross else None,
                mkb=mkv[:, DS_:] if cross else None,
                parallel=parallel,
            )
            bias = _pair_bias32(pair, n, weights, save)
            p3 = p.view(a, n, 4 * DS_)
            o, _ = local.attn_fwd32(
                p3[..., qc : qc + DS_],
                p3[..., kc : kc + DS_],
                p3[..., vc : vc + DS_],
                bias,
                _keys(mask, n0, n),
            )
            if slide:
                local.slide_edges(p3[..., qc : qc + DS_], p3[..., kc : kc + DS_], p3[..., vc : vc + DS_], bias, _keys(mask, n0, n), o, n0)
            out = rows.post32(
                o.view(m, DS_),
                p[:, gc : gc + DS_],
                s2,
                mod[:, _SO],
                mod[:, _S2],
                mod[:, _B2],
                mod[:, _ST],
                pk.wo,
                pk.wu,
                pk.ws,
            )
            return [_unpadded(out.view(a, 1, n, DS_), single.shape)]
        x1 = rows.adaln32(s2, mod[:, _S1], mod[:, _B1])  # AdaLN 1
        p = torch.empty(m, 4 * DS_, device=dev, dtype=F32)
        xkv = None
        if cross:  # q | g from x1, k | v from a second AdaLN of x1
            rows.gemm32(x1, pk.wp, p, n=2 * DS_, bias=pk.bp, rndmask=_P_RND[cross])
            xkv = rows.adaln32(s2 if parallel else x1, mkv[:, :DS_], mkv[:, DS_:])
            rows.gemm32(
                xkv,
                pk.wp,
                p,
                n=2 * DS_,
                b_row0=2 * DS_,
                ocol0=2 * DS_,
                rndmask=_P_RND[cross],
            )
        else:
            rows.gemm32(x1, pk.wp, p, bias=pk.bp, rndmask=_P_RND[cross])
        bias = _pair_bias32(pair, n, weights, save)
        p3 = p.view(a, n, 4 * DS_)
        o, lse = local.attn_fwd32(
            p3[..., qc : qc + DS_],
            p3[..., kc : kc + DS_],
            p3[..., vc : vc + DS_],
            bias,
            _keys(mask, n0, n),
        )
        if slide:
            local.slide_edges(p3[..., qc : qc + DS_], p3[..., kc : kc + DS_], p3[..., vc : vc + DS_], bias, _keys(mask, n0, n), o, n0)
        gated = rows.gate32(p[:, gc : gc + DS_], o.view(m, DS_))
        y = torch.empty(m, DS_, device=dev, dtype=F32) if save else None
        a1 = rows.gemm32(
            gated, pk.wo, mode=rows.RESID_GATE, res=s2, gate=mod[:, _SO], out2=y
        )  # a1 = s + so * (gated Wo^T)
        x2 = rows.adaln32(a1, mod[:, _S2], mod[:, _B2])  # AdaLN 2
        u = torch.empty(m, 4 * DS_, device=dev, dtype=F32) if save else None
        h = rows.gemm32(
            x2, pk.wu, mode=rows.SWIGLU, out2=u
        )  # h = silu(x2 Wa^T) * (x2 Wb^T)
        t = torch.empty(m, DS_, device=dev, dtype=F32) if save else None
        out = rows.gemm32(
            h, pk.ws, mode=rows.RESID_GATE, res=a1, gate=mod[:, _ST], out2=t
        )  # out = a1 + st * (h Ws^T)
    out = _unpadded(out.view(a, 1, n, DS_), single.shape)
    if not save:
        return [out]
    assert (
        y is not None and u is not None and t is not None
    )  # allocated above: `save` is true here
    saved = [out, mod, cn1, cn2, x1, p, bias, o, lse, gated, y, a1, x2, u, t]
    if not cross:
        return saved
    assert (
        cnkv is not None and mkv is not None and xkv is not None
    )  # the cross mode built all three
    return [*saved, cnkv, mkv, xkv]


def _mm32(dout, x):
    """dout^T x (a weight gradient), fp32 on TF32 tensor cores (the caller holds ``_tf32()``)."""
    return torch.mm(dout.t(), x)


@opaque(fake=_bwd_fake, name="local_dit_block_bwd_tf32")
def _block_bwd32(
    dy: torch.Tensor,
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    weights: list[torch.Tensor],
    saved: list[torch.Tensor],
    parallel: bool = False,
) -> list[torch.Tensor]:
    """The fp32 path's backward: d single, d cond, d pair, then one gradient per weight (``WEIGHTS`` order, the weight's dtype)."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom as rows
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm100_atom_local as local,
    )

    mod, cn1, cn2, x1, p, bias, o, lse, gated, y, a1, x2, u, t, *extra = saved
    cross = bool(extra)
    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    m = a * n
    dev = single.device
    with torch.cuda.device(dev):
        pk = _weights32_bwd(weights)
        qc, kc, vc, gc = _P_COLS[cross]
        s2 = _padded(single, n).reshape(m, DS_).contiguous()
        c2 = _padded(cond, n).reshape(m, DC_).contiguous()
        dy = _padded(dy, n).reshape(m, DS_).float().contiguous()
        dmod = torch.empty(
            m, 6 * DS_, device=dev, dtype=F32
        )  # d [s1 | bi1 | s2 | bi2 | so | st] (pre-sigmoid for the gates)
        dt = rows.tail_bwd32(dy, t, mod[:, _ST], dmod[:, _ST])  # out = a1 + st t
        dh = rows.gemm32(dt, pk.wsT)  # [m, 256]
        dab, hh = rows.swiglu_bwd32(dh, u)
        dx2 = rows.gemm32(dab, pk.wuT)  # K = 512
        da1, dyo = rows.adaln_bwd32(
            a1,
            mod[:, _S2],
            dx2,
            dy,
            dmod[:, _S2],
            dmod[:, _B2],
            y=y,
            so=mod[:, _SO],
            dso=dmod[:, _SO],
        )
        dgated = rows.gemm32(dyo, pk.woT)
        dp = torch.empty(
            m, 4 * DS_, device=dev, dtype=F32
        )  # d P32 (same column layout)
        do, dd = rows.gate_bwd32(
            dgated, p[:, gc : gc + DS_], o.view(m, DS_), dp[:, gc : gc + DS_], a, n
        )
        p3 = p.view(a, n, 4 * DS_)
        db = local.attn_bwd32(
            p3[..., qc : qc + DS_],
            p3[..., kc : kc + DS_],
            p3[..., vc : vc + DS_],
            do.view(a, n, DS_),
            bias,
            lse,
            dd,
            dp,
            (qc, kc, vc),
            _keys(mask, n0, n),
        )
        dz, dgamma, dwb = local.pair_bias_bwd32(
            _trunk(pair, n), weights[_GAMMA], weights[_WBIAS], db
        )
        dmkv = None
        if cross:  # x1 -> q | g directly, -> k | v through the K / V AdaLN
            cnkv, mkv, xkv = extra
            dx1q = rows.gemm32(dp, pk.wpT, k=2 * DS_)
            dxkv = rows.gemm32(dp, pk.wpT, k=2 * DS_, a_col0=2 * DS_, b_col0=2 * DS_)
            dmkv = torch.empty(m, 2 * DS_, device=dev, dtype=F32)
            if parallel:  # the K / V AdaLN of s: its LN backward joins the residual's gradient
                dres, _ = rows.adaln_bwd32(
                    s2, mkv[:, :DS_], dxkv, da1, dmkv[:, :DS_], dmkv[:, DS_:]
                )
                dx1 = dx1q
            else:
                dx1, _ = rows.adaln_bwd32(
                    x1, mkv[:, :DS_], dxkv, dx1q, dmkv[:, :DS_], dmkv[:, DS_:]
                )
                dres = da1
        else:
            dx1 = rows.gemm32(dp, pk.wpT)  # K = 512
            dres = da1
        ds, _ = rows.adaln_bwd32(
            s2, mod[:, _S1], dx1, dres, dmod[:, _S1], dmod[:, _B1]
        )  # AdaLN 1 + the residual
        dc = rows.gemm32(
            dmod, pk.wmodT, k=2 * DS_, a_col0=4 * DS_, b_col0=4 * DS_
        )  # so, st <- c
        dcn1 = rows.gemm32(dmod, pk.wmodT, k=2 * DS_)
        dcn2 = rows.gemm32(dmod, pk.wmodT, k=2 * DS_, a_col0=2 * DS_, b_col0=2 * DS_)
        dcn3 = rows.gemm32(dmkv, pk.wmT) if cross else None
        dgc = rows.ln_bwd32(
            c2, dc, dcn1, pk.g1, dcn2, pk.g2, dcn3, pk.gkv
        )  # dc += the LayerNorm backwards
        dcols = rows.colsum32(dmod)
        dbq = rows.colsum32(dp[:, qc : qc + DS_])
        dbkv = rows.colsum32(dmkv[:, :DS_]) if dmkv is not None else None
        names = WEIGHTS + CROSS_WEIGHTS[: len(weights) - len(WEIGHTS)]
        A_ = "attention."
        spec = [
            (dt, hh, [("transition.squeeze.weight", 0)]),
            (
                dab,
                x2,
                [
                    ("transition.expand_a.weight", 0),
                    ("transition.expand_b.weight", 0),
                    ("transition.expand_a.weight", 1),
                    ("transition.expand_b.weight", 1),
                ],
            ),  # u interleaves [a_0 | b_0 | a_1 | b_1]
            (dyo, gated, [(A_ + "to_out.weight", 0)]),
            (
                dmod[:, 0:256],
                cn1,
                [
                    (A_ + "ada_ln_in.to_scale.weight", 0),
                    (A_ + "ada_ln_in.to_bias.weight", 0),
                ],
            ),
            (
                dmod[:, 256:512],
                cn2,
                [
                    ("transition.ada_ln_in.to_scale.weight", 0),
                    ("transition.ada_ln_in.to_bias.weight", 0),
                ],
            ),
            (
                dmod[:, 512:768],
                c2,
                [(A_ + "to_scale.weight", 0), ("transition.to_scale.weight", 0)],
            ),
        ]
        if cross:
            spec += [
                (
                    dp[:, : 2 * DS_],
                    x1,
                    [(A_ + "to_query.weight", 0), (A_ + "to_gate.weight", 0)],
                ),
                (
                    dp[:, 2 * DS_ :],
                    xkv,
                    [(A_ + "to_key.weight", 0), (A_ + "to_value.weight", 0)],
                ),
                (
                    dmkv,
                    cnkv,
                    [
                        (A_ + "ada_ln_kv.to_scale.weight", 0),
                        (A_ + "ada_ln_kv.to_bias.weight", 0),
                    ],
                ),
            ]
        else:
            spec += [
                (
                    dp,
                    x1,
                    [
                        (A_ + "to_query.weight", 0),
                        (A_ + "to_key.weight", 0),
                        (A_ + "to_value.weight", 0),
                        (A_ + "to_gate.weight", 0),
                    ],
                )
            ]
        fused = _fused_wgrads(spec, weights, names, True, dev, m)
    grads: dict[str, torch.Tensor] = {
        "attention.ada_ln_in.ln_cond.weight": dgc[0:128],
        "attention.ada_ln_in.to_scale.bias": dcols[0:128],
        "attention.to_query.bias": dbq,
        "attention.ln_pair.weight": dgamma,
        "attention.to_bias.weight": dwb,
        "attention.to_scale.bias": dcols[512:640],
        "transition.ada_ln_in.ln_cond.weight": dgc[128:256],
        "transition.ada_ln_in.to_scale.bias": dcols[256:384],
        "transition.to_scale.bias": dcols[640:768],
    }
    if cross:
        assert (
            dbkv is not None
        )  # computed above from dmkv, which the cross mode allocated
        grads.update(
            {
                "attention.ada_ln_kv.ln_cond.weight": dgc[256:384],
                "attention.ada_ln_kv.to_scale.bias": dbkv,
            }
        )
    per_weight = [
        fused[name]
        if name in fused
        else grads[name].to(w.dtype).reshape(w.shape).clone()
        for name, w in zip(names, weights, strict=True)
    ]
    dpair = dz[: pair.shape[1]].reshape(pair.shape).clone()
    return [
        _unpadded(ds.view(a, 1, n, DS_), single.shape),
        _unpadded(dc.view(a, 1, n, DC_), cond.shape),
        dpair,
        *per_weight,
    ]


class _LocalBlock32(torch.autograd.Function):
    """The fp32 path's block forward and backward (TF32 kernels); gradients for single, cond, pair and every weight."""

    @staticmethod
    def forward(ctx, parallel, mask, single, cond, pair, *weights):
        out, *saved = _block_fwd32(single, cond, pair, mask, list(weights), True, parallel)
        ctx.save_for_backward(mask, single, cond, pair, *saved, *weights)
        ctx.n_saved = len(saved)
        ctx.parallel = parallel
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        mask, single, cond, pair, *rest = ctx.saved_tensors
        saved, weights = rest[: ctx.n_saved], rest[ctx.n_saved :]
        grads = _block_bwd32(dy, single, cond, pair, mask, list(weights), list(saved), ctx.parallel)
        return (None, None, *grads)


def _infer32(single, cond, pair, mask, weights, parallel=False, slide=False):
    return _block_fwd32(single, cond, pair, mask, list(weights), False, parallel, slide)[0]
