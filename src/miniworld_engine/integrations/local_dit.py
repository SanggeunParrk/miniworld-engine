"""The AF3 atom transformer block with block-local attention on B200 (sm_100a): inference and training.

One ``LocalDiTBlock`` call runs, in place of the module path,

    forward   cond_fwd -> pre_fwd -> pair bias (trunked) -> local attn_fwd -> post_fwd
    backward  tr_bwd_gate -> post_bwd -> local attn_dq / attn_dkv (+ dbias) -> pair-bias backward -> pre_bwd -> cond_bwd,
              then the weight gradients as cuBLAS GEMMs on the activations those kernels write.

The row kernels (conditioning, input projections, post-attention, transition, and their backward) are ``sm100_atom``'s: they do not
depend on what the attention attends to. The attention and the pair bias are ``sm100_atom_local``'s: 32 queries x 128 keys per window,
the trunked pair ``[nwin, 32, 128, 16]`` (bias ``LN(z) Wb`` per window, shared by the samples), no N x N tensor anywhere.
Both residuals are inside post_fwd (the block returns the stream).

``serves()`` is the whole gate: B200, the engine's kernel backend (implementation MINIWORLD or TRITON), bf16 single / cond / pair,
the atom widths (d_single = d_cond = 128, 4 heads x 32, d_pair = 16, transition n = 2, no QK-norm), B == 1, any N, a [B, N] bool key
mask or none, LayerNorm eps 1e-5. N is padded to a multiple of 128 for the row kernels (zero single / cond rows, the padded atoms
masked as keys, the pair padded with zero windows) and the first N rows come back. ``MINIWORLD_LOCAL_DIT_SM100=0`` turns it off. A
build or load failure warns once and keeps the module path.

**Hoisted per-item tables (inference).** The conditioning tables (``cond_fwd``'s modulation and, in the cross-attention mode, the key / value
modulation) depend on the conditioning tensor and the weights only, and the windowed pair bias on the pair and two weights only; a sampler calls
every block tens of times with one conditioning and one pair. An inference call takes them from the hoist: made once per tensor (pointer and
in-place version) and weights, so later calls launch neither ``cond_fwd``, ``cond_ln`` + its GEMM nor ``pair_bias_fwd``. Inside a CUDA-graph capture
the entries are scoped to the capture (a replay recomputes them, as it repacks the weights), unless the caller declares the inputs static
(``kernels._capture.static_inputs()`` / ``MINIWORLD_STATIC_CONDITIONING=1``): then a replay runs none of those kernels. The training path
(``save``) computes them every call. ``MINIWORLD_LOCAL_DIT_HOIST=0`` turns the hoist off.

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
fp32 calls only. ``MINIWORLD_LOCAL_DIT_TF32=0`` keeps fp32 calls on the module path. The fp32 weight pack is one flat buffer (one cat and an
in-place tf32 rounding: three kernels when a graph replay repacks; the backward's transposes are packed separately, by training calls only),
and ``static_inputs()`` covers it (a static capture replays no packing kernel). An inference call's conditioning tables are one LayerNorm
launch and one grouped GEMM, and its block is three kernels: ``atom_pre_tf32`` (AdaLN 1, the K / V AdaLN, the four projections),
the attention, ``atom_post_tf32`` (gate, Wo, both gated residuals, AdaLN 2, SwiGLU, Ws) -- the unfused kernels' arithmetic with the
activations kept on chip (``MINIWORLD_LOCAL_DIT_TF32_FUSED=0``: the unfused kernels; training always runs them, they save the
activations). The fp32 kernels launch with programmatic dependent launch (``sm100_atom.PDL32``, ``MINIWORLD_TF32_PDL=0``
turns it off).
"""

from __future__ import annotations

import contextlib
import os
import warnings
import weakref
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from miniworld_engine import settings
from miniworld_engine.kernels import _capture
from miniworld_engine.integrations.atom_dit import (  # the row kernels' weight packs are the dense atom block's
    _GAMMA,
    _WBIAS,
    BF,
    DC_,
    DP_,
    DS_,
    EPS,
    NH_,
    WEIGHTS,
    _padded,
    _padded_length,
    _unpadded,
    _weights,
)
from miniworld_engine.kernels._compile import device_constant, opaque

QUERIES, KEYS = 32, 128
_LOADED: set[int] = set()
_HOIST_COND: dict = {}      # (conditioning tensor, weights) -> (mod, mkv): the per-item conditioning tables of an inference call
_HOIST_BIAS: dict = {}      # (pair tensor, LN / bias weights) -> the windowed pair bias
_KV: dict = {}              # weights -> the cross-attention operands (kv_pack)


def _hoisting() -> bool:
    return os.environ.get("MINIWORLD_LOCAL_DIT_HOIST", "1") != "0"
_FAILED = False
F32 = torch.float32
_LOADED32: set[int] = set()
_FAILED32 = False
_ROW_KERNELS = ("cond", "pre", "post", "trg", "postb", "preb", "condb")
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
        for name in _ROW_KERNELS:
            rows.atom_kernel(name, index)
        for name in local.KERNELS:
            local._load(name, index)
    except Exception as exc:  # a toolchain or driver problem keeps the module path
        _FAILED = True
        warnings.warn(f"sm_100a local atom DiT kernels unavailable, keeping the module path: {exc!r}", RuntimeWarning, stacklevel=2)
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
        warnings.warn(f"sm_100a local atom DiT TF32 kernels unavailable, keeping the module path for fp32: {exc!r}", RuntimeWarning,
                      stacklevel=2)
        return False
    _LOADED32.add(index)
    return True


_FUSED32_FAILED = False


def _fused32_ready(device) -> bool:
    """The fused inference kernels of the fp32 path (``sm100_atom`` ``fused_tf32.cu``: AdaLNs + projections, and gate + Wo + AdaLN 2 +
    SwiGLU + Ws with both residuals): two launches in place of eight. ``MINIWORLD_LOCAL_DIT_TF32_FUSED=0`` keeps the unfused kernels; a
    build or load failure warns once and keeps them too (the rest of the fp32 path is unaffected)."""
    global _FUSED32_FAILED
    if _FUSED32_FAILED or os.environ.get("MINIWORLD_LOCAL_DIT_TF32_FUSED", "1") == "0":
        return False
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom as rows

    try:
        rows.load_fused32(device.index if device.index is not None else torch.cuda.current_device())
    except Exception as exc:  # a toolchain or driver problem keeps the unfused kernels
        _FUSED32_FAILED = True
        warnings.warn(f"sm_100a fused fp32 atom kernels unavailable, keeping the unfused fp32 kernels: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


def _fp32_call(module, single, cond, pair) -> bool:
    """A call for the fp32 path: fp32 single / cond / pair and parameters, no autocast (under autocast the module path runs its products in
    the autocast dtype), ``MINIWORLD_LOCAL_DIT_TF32`` not 0. A bf16 model fed fp32 activations keeps the module path."""
    if (single.dtype, cond.dtype, pair.dtype) != (F32, F32, F32):
        return False
    if os.environ.get("MINIWORLD_LOCAL_DIT_TF32", "1") == "0" or torch.is_autocast_enabled("cuda"):
        return False
    return all(p.dtype == F32 for p in module.parameters())


def serves(module, single, cond, pair, mask) -> bool:
    fp32 = _fp32_call(module, single, cond, pair)
    if (_FAILED32 if fp32 else _FAILED) or os.environ.get("MINIWORLD_LOCAL_DIT_SM100", "1") == "0" \
            or settings.current().engine_backend == "triton":
        return False
    from miniworld_engine.modules.dispatch import KernelBackend

    a, tr = module.attention, module.transition
    if a._backend != KernelBackend.TRITON or a.use_qk_norm:
        return False
    if not (single.is_cuda and torch.cuda.get_device_capability(single.device) == (10, 0)):
        return False
    if (single.dtype, cond.dtype, pair.dtype) != (BF, BF, BF) and not fp32:
        return False
    if single.ndim != 4 or single.shape[1] != 1 or single.shape[-1] != DS_ or single.shape[2] == 0:
        return False
    n = single.shape[2]
    if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != (1, n) or mask.device != single.device):
        return False
    if tuple(cond.shape) != (*single.shape[:3], DC_) or tuple(pair.shape) != (1, (n + QUERIES - 1) // QUERIES, QUERIES, KEYS, DP_):
        return False
    if (a.n_head, a.to_query.weight.shape[0], a.to_bias.weight.shape[1], tr.expand_a.weight.shape[0]) != (NH_, DS_, DP_, 2 * DS_):
        return False
    norms = (a.ada_ln_in.ln_in, a.ada_ln_in.ln_cond, a.ln_pair, tr.ada_ln_in.ln_in, tr.ada_ln_in.ln_cond)
    if module.cross_attention:
        norms += (a.ada_ln_kv.ln_in, a.ada_ln_kv.ln_cond)
    if any(norm.eps != EPS for norm in norms):
        return False
    index = single.device.index if single.device.index is not None else torch.cuda.current_device()
    return _kernels32_ready(index) if fp32 else _kernels_ready(index)


def _keys(mask, n0, n):
    """The valid-key bytes [n] the attention kernels read (None: every key valid and no padding)."""
    if mask is not None:
        return F.pad(mask[0], (0, n - n0)).contiguous()
    return None if n == n0 else (torch.arange(n, device=_device_of(mask)) < n0)


def _device_of(mask):
    return torch.cuda.current_device() if mask is None else mask.device


def _hoisted(store, tensor, key, build):
    """The table ``build()`` makes from ``tensor``, once per tensor (pointer, version, ``key``): the entry keeps a weak reference to the tensor, so
    a new tensor that reuses a freed address is not served the old one's table, and entries of freed tensors are dropped."""
    entry = _capture.lookup_inputs(store, (_tkey(tensor), key), lambda: (weakref.ref(tensor), build()), limit=64,
                                   valid=lambda e: e[0]() is tensor, alive=lambda e: e[0]() is not None)
    return entry[1]


def _tkey(t):
    """A tensor's identity for a cache key: storage address, in-place version, shape, dtype, device."""
    return (t.data_ptr(), t._version, tuple(t.shape), t.dtype, t.device)


def _kv_pack(weights):
    """The cross-attention mode's operands: Wm [256, 128] = [to_scale; to_bias] bf16, the scale bias and the LN weight (fp32), Wk, Wv (bf16).
    Packed once per set of weights (pointer and version), scoped to the CUDA-graph capture like every weight pack."""
    def build():
        gamma, w_scale, b_scale, w_bias = weights[len(WEIGHTS):]
        wk, wv = weights[WEIGHTS.index("attention.to_key.weight")], weights[WEIGHTS.index("attention.to_value.weight")]
        return torch.cat([w_scale, w_bias]).to(BF), b_scale.float(), gamma.float(), wk.to(BF), wv.to(BF)

    tail = weights[len(WEIGHTS):] + [weights[WEIGHTS.index("attention.to_key.weight")], weights[WEIGHTS.index("attention.to_value.weight")]]
    return _capture.lookup(_KV, tuple((t.data_ptr(), t._version) for t in tail), build, limit=16)


def _cond_tables(cond, c2_fn, n, weights, wc, cross, save):
    """The conditioning tables of a call: mod [rows, 768] (cond_fwd) and, in the cross-attention mode, cn (the LN of the conditioning) and the
    key / value modulation mkv [rows, 256]. They depend on the conditioning tensor and the weights only -- the same at every block call that
    reads one conditioning (a sampler's steps), so an inference call (``save`` False) takes them from the hoist: made once per conditioning
    tensor (pointer, version) and weights, scoped to the CUDA-graph capture (``kernels._capture.lookup_inputs``)."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom as rows
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm100_atom_local as local,
    )

    def build():
        c2 = c2_fn()
        mod = rows.cond_fwd(c2, *wc)
        if not cross:
            return mod, None, None
        wm, _, gkv, _, _ = _kv_pack(weights)
        cn = local.cond_ln(c2, gkv)
        return mod, cn, cn @ wm.t()

    if save or not _hoisting():
        return build()
    return _hoisted(_HOIST_COND, cond, (tuple((w.data_ptr(), w._version) for w in weights), n), build)


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
    return _hoisted(_HOIST_BIAS, pair, ((g.data_ptr(), g._version, wb.data_ptr(), wb._version), n), build)


def _trunk(pair, n):
    """[1, nwin, 32, 128, 16] -> [n / 32, 32, 128, 16]: zero windows appended for the padded atoms."""
    z = pair[0]
    return z if z.shape[0] == n // QUERIES else F.pad(z, (0, 0, 0, 0, 0, 0, 0, n // QUERIES - z.shape[0])).contiguous()


# ------------------------------------------------------------------------------------------------------------ the two ops
def _fwd_fake(single, cond, pair, mask, weights, save):
    out = torch.empty_like(single)
    if not save:
        return [out]
    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    rows = a * n

    def tile(width=DS_):
        return single.new_empty((rows, width))

    saved = [out, tile(6 * DS_), tile(), tile(), tile(), tile(), tile(), single.new_empty((NH_, n // QUERIES, QUERIES, KEYS), dtype=torch.float32),
             tile(), single.new_empty((a, NH_, n), dtype=torch.float32), tile(), tile(), tile(), tile()]
    return [*saved, tile(), tile(2 * DS_), tile()] if len(weights) > len(WEIGHTS) else saved


@opaque(fake=_fwd_fake, name="local_dit_block_fwd")
def _block_fwd(single: torch.Tensor, cond: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None,
               weights: list[torch.Tensor], save: bool) -> list[torch.Tensor]:
    """The block output [A, 1, N, 128] bf16 and, when ``save``, the activations the backward reads (mod, q, k, v, sg, x1, bias, O, LSE,
    u, a2, x2, t: all fresh tensors)."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom as rows
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm100_atom_local as local,
    )

    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    m = a * n
    with torch.cuda.device(single.device):
        cross = len(weights) > len(WEIGHTS)
        wc, wp, wo, _ = _weights(weights[: len(WEIGHTS)])
        s2 = _padded(single, n).reshape(m, DS_).contiguous()
        mod, cn, mkv = _cond_tables(cond, lambda: _padded(cond, n).reshape(m, DC_).contiguous(), n, weights, wc, cross, save)
        q, k, v, sg, x1 = rows.pre_fwd(s2, mod, *wp, save=save or cross)
        if cross:       # keys and values from a second AdaLN of x1 (pre_fwd's own k / v are not used)
            _, bs, _, wk, wv = _kv_pack(weights)
            xkv = local.kv_fwd(x1, mkv, bs)
            k, v = xkv @ wk.t(), xkv @ wv.t()
        bias = _pair_bias(pair, n, weights, save)
        o, lse = local.attn_fwd(q.view(a, n, DS_), k.view(a, n, DS_), v.view(a, n, DS_), bias, _keys(mask, n0, n))
        out, u, a2, x2, t = rows.post_fwd(s2, o.view(m, DS_), sg, mod, *wo, save=save)
    out = _unpadded(out.view(a, 1, n, DS_), single.shape)
    if not save:
        return [out]
    saved = [out, mod, q, k, v, sg, x1, bias, o.view(m, DS_), lse, u, a2, x2, t]
    return [*saved, cn, mkv, xkv] if cross else saved


def _wgrad(dout, x):
    """dout^T x, fp32: the weight gradients stay unrounded for fp32 parameters (an fp32 master); bf16 ones round once, as before."""
    return torch.mm(dout.t(), x, out_dtype=torch.float32)


def _bwd_fake(dy, single, cond, pair, mask, weights, saved):
    return [torch.empty_like(single), torch.empty_like(cond), torch.empty_like(pair), *(torch.empty_like(w) for w in weights)]


@opaque(fake=_bwd_fake, name="local_dit_block_bwd")
def _block_bwd(dy: torch.Tensor, single: torch.Tensor, cond: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None,
               weights: list[torch.Tensor], saved: list[torch.Tensor]) -> list[torch.Tensor]:
    """d single, d cond, d pair, then one gradient per weight in ``WEIGHTS`` order, in the weight's dtype (all fresh tensors)."""
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
        wc, _, wo, wb = _weights(weights[: len(WEIGHTS)])
        s2 = _padded(single, n).reshape(m, DS_).contiguous()
        c2 = _padded(cond, n).reshape(m, DC_).contiguous()
        dy = _padded(dy, n).reshape(m, DS_).to(BF).contiguous()
        dmod = torch.empty(m, 6 * DS_, device=dev, dtype=BF)         # [dsc1 | dbi1 | dsc2 | dbi2 | dos | dts]
        dp = torch.empty(m, 4 * DS_, device=dev, dtype=BF)           # [dq | dk | dv | dg]
        dbias_cols = torch.zeros(6 * DS_, device=dev)
        dbq = torch.zeros(DS_, device=dev)
        dgamma_cond = torch.zeros(2 * DS_, device=dev)
        dd = torch.empty(a, NH_, n, device=dev)
        dt, hh, dab = rows.tr_bwd_gate(dy, t, mod, x2, wo[1], wb["WsT"], dmod, dbias_cols[640:])
        da, du, gated, do = rows.post_bwd(dab, dy, a2, mod, u, sg, o, wb["WuT"], wb["WoT"], dmod, dp, dd, dbias_cols, n)
        db = local.attn_bwd(q.view(a, n, DS_), k.view(a, n, DS_), v.view(a, n, DS_), do.view(a, n, DS_), bias, lse, dd, dp,
                            _keys(mask, n0, n))
        dz, dgamma, dwb = local.pair_bias_bwd(_trunk(pair, n), weights[_GAMMA], weights[_WBIAS], db)
        if cross:   # the K / V branch: dxkv -> its AdaLN -> dx1 (through AdaLN 1 below) and the conditioning
            cn, mkv, xkv = extra
            wm, bs, gkv, wk, wv = _kv_pack(weights)
            dk, dv = dp[:, DS_:2 * DS_], dp[:, 2 * DS_:3 * DS_]
            dxkv = torch.addmm(dk @ wk, dv, wv)
            dwk, dwv = _wgrad(dk, xkv), _wgrad(dv, xkv)
            dx1_kv, dmkv = local.kv_bwd(x1, mkv, bs, dxkv)
            dwm, dbs = _wgrad(dmkv, cn), dmkv[:, :DS_].float().sum(0)
            dg_kv = torch.zeros(DS_, device=dev)
            dc_kv = local.cond_ln_bwd(c2, dmkv @ wm, gkv, dg_kv)
            dp[:, DS_:3 * DS_].zero_()      # pre_bwd's dgrad then covers q and the gate only
        ds = rows.pre_bwd(dp, s2, mod, da, wb["WpT"], dmod, dbias_cols, dbq)
        if cross:
            local.adaln1_extra(s2, mod, dx1_kv, ds, dmod, dbias_cols[0:DS_])
        dc, cn1, cn2 = rows.cond_bwd(dmod, c2, wb["WmodT"], wc[2], wc[3], dgamma_cond)
        if cross:
            dc = dc + dc_kv
        dws, dwu, dwo, dwp = _wgrad(dt, hh), _wgrad(dab, x2), _wgrad(du, gated), _wgrad(dp, x1)
        dw1, dw2, dw3 = _wgrad(dmod[:, 0:256], cn1), _wgrad(dmod[:, 256:512], cn2), _wgrad(dmod[:, 512:768], c2)
    grads = {
        "attention.ada_ln_in.ln_cond.weight": dgamma_cond[:128], "attention.ada_ln_in.to_scale.weight": dw1[:128],
        "attention.ada_ln_in.to_scale.bias": dbias_cols[0:128], "attention.ada_ln_in.to_bias.weight": dw1[128:],
        "attention.to_query.weight": dwp[0:128], "attention.to_query.bias": dbq, "attention.to_key.weight": dwp[128:256],
        "attention.to_value.weight": dwp[256:384], "attention.ln_pair.weight": dgamma, "attention.to_bias.weight": dwb,
        "attention.to_gate.weight": dwp[384:512], "attention.to_out.weight": dwo, "attention.to_scale.weight": dw3[:128],
        "attention.to_scale.bias": dbias_cols[512:640], "transition.ada_ln_in.ln_cond.weight": dgamma_cond[128:],
        "transition.ada_ln_in.to_scale.weight": dw2[:128], "transition.ada_ln_in.to_scale.bias": dbias_cols[256:384],
        "transition.ada_ln_in.to_bias.weight": dw2[128:], "transition.expand_a.weight": dwu[:256],
        "transition.expand_b.weight": dwu[256:], "transition.squeeze.weight": dws, "transition.to_scale.weight": dw3[128:],
        "transition.to_scale.bias": dbias_cols[640:768],
    }
    if cross:
        grads["attention.to_key.weight"], grads["attention.to_value.weight"] = dwk, dwv
        grads.update({"attention.ada_ln_kv.ln_cond.weight": dg_kv, "attention.ada_ln_kv.to_scale.weight": dwm[:DS_],
                      "attention.ada_ln_kv.to_scale.bias": dbs, "attention.ada_ln_kv.to_bias.weight": dwm[DS_:]})
    names = WEIGHTS + CROSS_WEIGHTS[: len(weights) - len(WEIGHTS)]
    per_weight = [grads[name].to(w.dtype).reshape(w.shape).clone() for name, w in zip(names, weights, strict=True)]
    dpair = dz[: pair.shape[1]].reshape(pair.shape).clone()
    return [_unpadded(ds.view(a, 1, n, DS_), single.shape), _unpadded(dc.view(a, 1, n, DC_), cond.shape), dpair, *per_weight]


# ------------------------------------------------------------------------------------------------------------ autograd
class _LocalBlock(torch.autograd.Function):
    """The whole block forward and backward on the sm_100a kernels; gradients for single, cond, pair and every weight."""

    @staticmethod
    def forward(ctx, mask, single, cond, pair, *weights):
        out, *saved = _block_fwd(single, cond, pair, mask, list(weights), True)
        ctx.save_for_backward(mask, single, cond, pair, *saved, *weights)
        ctx.n_saved = len(saved)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        mask, single, cond, pair, *rest = ctx.saved_tensors
        saved, weights = rest[: ctx.n_saved], rest[ctx.n_saved :]
        grads = _block_bwd(dy, single, cond, pair, mask, list(weights), list(saved))
        return (None, *grads)


def _infer(single, cond, pair, mask, weights):
    return _block_fwd(single, cond, pair, mask, list(weights), False)[0]


def block(module, single, cond, pair, mask=None):
    """The block's output stream (both residuals included), on the sm_100a kernels; ``serves()`` must have accepted the call."""
    weights = [module.get_parameter(name) for name in (*WEIGHTS, *(CROSS_WEIGHTS if module.cross_attention else ()))]
    if single.dtype == F32:                                          # the fp32 path (TF32 tensor cores)
        if torch.is_grad_enabled() and any(t.requires_grad for t in (single, cond, pair, *weights)):
            return _LocalBlock32.apply(mask, single, cond, pair, *weights)
        return _infer32(single, cond, pair, mask, weights)
    if torch.is_grad_enabled() and any(t.requires_grad for t in (single, cond, pair, *weights)):
        return _LocalBlock.apply(mask, single, cond, pair, *weights)
    return _infer(single, cond, pair, mask, weights)


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
_SIG_MOD = 0b110101                                                  # sigmoid on mod32 blocks 0, 2, 4, 5
_P_COLS = {False: (0, DS_, 2 * DS_, 3 * DS_), True: (0, 2 * DS_, 3 * DS_, DS_)}     # (q, k, v, g) columns of P32
_P_RND = {False: 0b0111, True: 0b1101}                               # P32 blocks stored tf32-rounded: q, k, v (only MMA operands), not g


@contextlib.contextmanager
def _tf32():
    """cuBLAS on TF32 tensor cores for the fp32 path's weight gradients (restored after): the path is the TF32 recipe."""
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


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
    w = dict(zip((*WEIGHTS, *CROSS_WEIGHTS[: len(weights) - len(WEIGHTS)]), weights, strict=True))

    def m(name):
        return w[name].detach().float().reshape(-1)

    with torch.no_grad():
        z = _zeros32(weights[0].device)
        wa, wb = w["transition.expand_a.weight"].detach().float(), w["transition.expand_b.weight"].detach().float()
        proj = ("query", "gate", "key", "value") if cross else ("query", "key", "value", "gate")
        mats = [m("attention.ada_ln_in.to_scale.weight"), m("attention.ada_ln_in.to_bias.weight"), m("transition.ada_ln_in.to_scale.weight"),
                m("transition.ada_ln_in.to_bias.weight"), m("attention.to_scale.weight"), m("transition.to_scale.weight")]
        if cross:
            mats += [m("attention.ada_ln_kv.to_scale.weight"), m("attention.ada_ln_kv.to_bias.weight")]
        mats += [m(f"attention.to_{p}.weight") for p in proj]
        mats += [wa[:DS_].reshape(-1), wb[:DS_].reshape(-1), wa[DS_:].reshape(-1), wb[DS_:].reshape(-1)]
        mats += [m("attention.to_out.weight"), m("transition.squeeze.weight")]
        vecs = [m("attention.ada_ln_in.to_scale.bias"), z, m("transition.ada_ln_in.to_scale.bias"), z, m("attention.to_scale.bias"),
                m("transition.to_scale.bias")]
        if cross:
            vecs += [m("attention.ada_ln_kv.to_scale.bias"), z]
        vecs += [m("attention.to_query.bias"), z, z, z, m("attention.ada_ln_in.ln_cond.weight"), m("transition.ada_ln_in.ln_cond.weight")]
        if cross:
            vecs.append(m("attention.ada_ln_kv.ln_cond.weight"))
        flat = torch.cat(mats + vecs)
        nmat = sum(t.numel() for t in mats)
        flat[:nmat].view(torch.int32).add_(0x1000).bitwise_and_(-0x2000)      # RNA to tf32: half an ulp into the magnitude, drop 13 bits

        off = [0]

        def take(rows, cols=DS_):
            t = flat[off[0]:off[0] + rows * cols].view(rows, cols)
            off[0] += rows * cols
            return t

        rc = 8 if cross else 6
        wcond = take(rc * DS_)
        wp, wu, wo, ws = take(4 * DS_), take(4 * DS_), take(DS_), take(DS_, 2 * DS_)
        bcond = flat[off[0]:off[0] + rc * DS_]
        off[0] += rc * DS_
        bp = flat[off[0]:off[0] + 4 * DS_]
        off[0] += 4 * DS_
        g1, g2 = flat[off[0]:off[0] + DS_], flat[off[0] + DS_:off[0] + 2 * DS_]
        off[0] += 2 * DS_
        pk = SimpleNamespace(wmod=wcond[:6 * DS_], bmod=bcond[:6 * DS_], g1=g1, g2=g2, wp=wp, bp=bp, wo=wo, wu=wu, ws=ws, gkv=None,
                             wcond=wcond, bcond=bcond, flat=flat)
        if cross:
            pk.wm, pk.bm, pk.gkv = wcond[6 * DS_:], bcond[6 * DS_:], flat[off[0]:off[0] + DS_]
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
        bk.wmodT, bk.wpT, bk.woT, bk.wuT, bk.wsT = tr(pk.wmod), tr(pk.wp), tr(pk.wo), tr(pk.wu), tr(pk.ws)
        bk.wmT = tr(pk.wm) if pk.gkv is not None else None
    return bk


_PACKS32: dict = {}
_PACKS32B: dict = {}


def _cached32(store, weights, build):
    """``build(weights)``, reused while every weight's tensor object and version are unchanged; scoped to the CUDA-graph capture
    (``kernels._capture``) as the bf16 packs are -- except under ``kernels._capture.static_inputs()``, whose contract (the weights AND the
    conditioning / pair keep their contents between replays) covers the weights: a capture then serves from, and fills, the eager entry and a
    replay runs no packing kernel (``static_weights()`` has the same effect through ``scoped``)."""
    try:
        versions = tuple(w._version for w in weights)
    except RuntimeError:  # inference tensors carry no version counter: nothing could invalidate an entry
        return build(weights)
    raw = ("tf32", *(id(w) for w in weights))
    key = (None, raw) if _capture._STATIC_INPUTS[0] else _capture.scoped(raw)
    if key is None:
        return build(weights)
    hit = store.get(key)
    if hit is not None:
        refs, seen, pack = hit
        if seen == versions and all(r() is w for r, w in zip(refs, weights, strict=True)):
            return pack
    if not _capture._STATIC_INPUTS[0]:
        _capture.prune(store)
    if len(store) >= 64:  # a model's worth of blocks; drop the oldest
        store.pop(next(iter(store)))
    pack = build(weights)
    store[key] = (tuple(weakref.ref(w) for w in weights), versions, pack)
    return pack


def _weights32(weights):
    """The forward pack (``_pack32``), cached (``_cached32``)."""
    return _cached32(_PACKS32, weights, _pack32)


def _weights32_bwd(weights):
    """The backward pack (``_pack32_bwd``), cached (``_cached32``)."""
    return _cached32(_PACKS32B, weights, _pack32_bwd)


def _cond_tables32(cond, c2_fn, n, weights, pk, cross, save):
    """The fp32 conditioning tables of a call: mod32 [rows, 768] and, in the cross mode, mkv [rows, 256]; with ``save`` also the
    LayerNorms cn1, cn2 (and cnkv) the weight gradients read. An inference call takes them from the hoist (``_cond_tables``)."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom as rows

    def build():
        c2 = c2_fn()
        if not save:                                                 # inference: two launches, the tables only
            # cat = [cn1 | cn2 | c | cnkv] (one LayerNorm launch), then ONE grouped GEMM: column group g (256 wide) of [mod | mkv] reads
            # cat's 128-column block g against wcond's rows 256 g .. -- mod and mkv are column views of one [M, 1024] (768) table
            rc = 8 if cross else 6
            cat = torch.empty(c2.shape[0], rc // 2 * DS_, device=c2.device, dtype=F32)
            rows.ln32(c2, pk.g1, pk.g2, pk.gkv if cross else None, out=cat)
            tab = rows.gemm32(cat, pk.wcond, k=DS_, a_grp=2 * DS_, bias=pk.bcond, sigmask=_SIG_MOD | (0b1000000 if cross else 0))
            return tab[:, :6 * DS_], None, None, None, (tab[:, 6 * DS_:] if cross else None)
        cn1, cn2, cnkv = rows.ln32(c2, pk.g1, pk.g2, pk.gkv if cross else None)
        mod = torch.empty(c2.shape[0], 6 * DS_, device=c2.device, dtype=F32)
        for i, x in enumerate((cn1, cn2, c2)):
            rows.gemm32(x, pk.wmod, mod, n=2 * DS_, b_row0=2 * DS_ * i, ocol0=2 * DS_ * i, bias=pk.bmod, sigmask=_SIG_MOD)
        mkv = rows.gemm32(cnkv, pk.wm, bias=pk.bm, sigmask=1) if cross else None
        return mod, cn1, cn2, cnkv, mkv

    if save or not _hoisting():
        return build()
    return _hoisted(_HOIST_COND, cond, ("tf32", tuple((w.data_ptr(), w._version) for w in weights), n), build)


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
    return _hoisted(_HOIST_BIAS, pair, ("tf32", (g.data_ptr(), g._version, wb.data_ptr(), wb._version), n), build)


def _fwd_fake32(single, cond, pair, mask, weights, save):
    out = torch.empty_like(single)
    if not save:
        return [out]
    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    m = a * n

    def tile(width=DS_):
        return single.new_empty((m, width))

    saved = [out, tile(6 * DS_), tile(), tile(), tile(), tile(4 * DS_), single.new_empty((NH_, n // QUERIES, QUERIES, KEYS)),
             single.new_empty((a, n, DS_)), single.new_empty((a, NH_, n)), tile(), tile(), tile(), tile(), tile(4 * DS_), tile()]
    return [*saved, tile(), tile(2 * DS_), tile()] if len(weights) > len(WEIGHTS) else saved


@opaque(fake=_fwd_fake32, name="local_dit_block_fwd_tf32")
def _block_fwd32(single: torch.Tensor, cond: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None,
                 weights: list[torch.Tensor], save: bool) -> list[torch.Tensor]:
    """The fp32 path's forward: the block output [A, 1, N, 128] fp32 and, when ``save``, the activations the backward reads (mod32, cn1,
    cn2, x1, P32, bias, O, LSE, gated, y, a1, x2, u, t; cross mode: cnkv, mkv, xkv -- all fresh fp32 tensors)."""
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
        mod, cn1, cn2, cnkv, mkv = _cond_tables32(cond, lambda: _padded(cond, n).reshape(m, DC_).contiguous(), n, weights, pk, cross,
                                                  save)
        if not save and _fused32_ready(dev):                         # inference: pre (AdaLNs + projections), attention, post (the rest)
            p = rows.pre32(s2, mod[:, _S1], mod[:, _B1], pk.wp, pk.bp[:DS_], _P_RND[cross], mks=mkv[:, :DS_] if cross else None,
                           mkb=mkv[:, DS_:] if cross else None)
            bias = _pair_bias32(pair, n, weights, save)
            p3 = p.view(a, n, 4 * DS_)
            o, _ = local.attn_fwd32(p3[..., qc:qc + DS_], p3[..., kc:kc + DS_], p3[..., vc:vc + DS_], bias, _keys(mask, n0, n))
            out = rows.post32(o.view(m, DS_), p[:, gc:gc + DS_], s2, mod[:, _SO], mod[:, _S2], mod[:, _B2], mod[:, _ST], pk.wo, pk.wu, pk.ws)
            return [_unpadded(out.view(a, 1, n, DS_), single.shape)]
        x1 = rows.adaln32(s2, mod[:, _S1], mod[:, _B1])                                      # AdaLN 1
        p = torch.empty(m, 4 * DS_, device=dev, dtype=F32)
        xkv = None
        if cross:                                                    # q | g from x1, k | v from a second AdaLN of x1
            rows.gemm32(x1, pk.wp, p, n=2 * DS_, bias=pk.bp, rndmask=_P_RND[cross])
            xkv = rows.adaln32(x1, mkv[:, :DS_], mkv[:, DS_:])
            rows.gemm32(xkv, pk.wp, p, n=2 * DS_, b_row0=2 * DS_, ocol0=2 * DS_, rndmask=_P_RND[cross])
        else:
            rows.gemm32(x1, pk.wp, p, bias=pk.bp, rndmask=_P_RND[cross])
        bias = _pair_bias32(pair, n, weights, save)
        p3 = p.view(a, n, 4 * DS_)
        o, lse = local.attn_fwd32(p3[..., qc:qc + DS_], p3[..., kc:kc + DS_], p3[..., vc:vc + DS_], bias, _keys(mask, n0, n))
        gated = rows.gate32(p[:, gc:gc + DS_], o.view(m, DS_))
        y = torch.empty(m, DS_, device=dev, dtype=F32) if save else None
        a1 = rows.gemm32(gated, pk.wo, mode=rows.RESID_GATE, res=s2, gate=mod[:, _SO], out2=y)       # a1 = s + so * (gated Wo^T)
        x2 = rows.adaln32(a1, mod[:, _S2], mod[:, _B2])                                      # AdaLN 2
        u = torch.empty(m, 4 * DS_, device=dev, dtype=F32) if save else None
        h = rows.gemm32(x2, pk.wu, mode=rows.SWIGLU, out2=u)                                  # h = silu(x2 Wa^T) * (x2 Wb^T)
        t = torch.empty(m, DS_, device=dev, dtype=F32) if save else None
        out = rows.gemm32(h, pk.ws, mode=rows.RESID_GATE, res=a1, gate=mod[:, _ST], out2=t)   # out = a1 + st * (h Ws^T)
    out = _unpadded(out.view(a, 1, n, DS_), single.shape)
    if not save:
        return [out]
    saved = [out, mod, cn1, cn2, x1, p, bias, o, lse, gated, y, a1, x2, u, t]
    return [*saved, cnkv, mkv, xkv] if cross else saved


def _mm32(dout, x):
    """dout^T x (a weight gradient), fp32 on TF32 tensor cores (the caller holds ``_tf32()``)."""
    return torch.mm(dout.t(), x)


@opaque(fake=_bwd_fake, name="local_dit_block_bwd_tf32")
def _block_bwd32(dy: torch.Tensor, single: torch.Tensor, cond: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None,
                 weights: list[torch.Tensor], saved: list[torch.Tensor]) -> list[torch.Tensor]:
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
        dmod = torch.empty(m, 6 * DS_, device=dev, dtype=F32)        # d [s1 | bi1 | s2 | bi2 | so | st] (pre-sigmoid for the gates)
        dt = rows.tail_bwd32(dy, t, mod[:, _ST], dmod[:, _ST])                                # out = a1 + st t
        dh = rows.gemm32(dt, pk.wsT)                                                          # [m, 256]
        dab, hh = rows.swiglu_bwd32(dh, u)
        dx2 = rows.gemm32(dab, pk.wuT)                                                        # K = 512
        da1, dyo = rows.adaln_bwd32(a1, mod[:, _S2], dx2, dy, dmod[:, _S2], dmod[:, _B2], y=y, so=mod[:, _SO], dso=dmod[:, _SO])
        dgated = rows.gemm32(dyo, pk.woT)
        dp = torch.empty(m, 4 * DS_, device=dev, dtype=F32)          # d P32 (same column layout)
        do, dd = rows.gate_bwd32(dgated, p[:, gc:gc + DS_], o.view(m, DS_), dp[:, gc:gc + DS_], a, n)
        p3 = p.view(a, n, 4 * DS_)
        db = local.attn_bwd32(p3[..., qc:qc + DS_], p3[..., kc:kc + DS_], p3[..., vc:vc + DS_], do.view(a, n, DS_), bias, lse, dd, dp,
                              (qc, kc, vc), _keys(mask, n0, n))
        dz, dgamma, dwb = local.pair_bias_bwd32(_trunk(pair, n), weights[_GAMMA], weights[_WBIAS], db)
        dmkv = None
        if cross:                                                    # x1 -> q | g directly, -> k | v through the K / V AdaLN
            cnkv, mkv, xkv = extra
            dx1q = rows.gemm32(dp, pk.wpT, k=2 * DS_)
            dxkv = rows.gemm32(dp, pk.wpT, k=2 * DS_, a_col0=2 * DS_, b_col0=2 * DS_)
            dmkv = torch.empty(m, 2 * DS_, device=dev, dtype=F32)
            dx1, _ = rows.adaln_bwd32(x1, mkv[:, :DS_], dxkv, dx1q, dmkv[:, :DS_], dmkv[:, DS_:])
        else:
            dx1 = rows.gemm32(dp, pk.wpT)                                                     # K = 512
        ds, _ = rows.adaln_bwd32(s2, mod[:, _S1], dx1, da1, dmod[:, _S1], dmod[:, _B1])      # AdaLN 1 + the residual
        dc = rows.gemm32(dmod, pk.wmodT, k=2 * DS_, a_col0=4 * DS_, b_col0=4 * DS_)           # so, st <- c
        dcn1 = rows.gemm32(dmod, pk.wmodT, k=2 * DS_)
        dcn2 = rows.gemm32(dmod, pk.wmodT, k=2 * DS_, a_col0=2 * DS_, b_col0=2 * DS_)
        dcn3 = rows.gemm32(dmkv, pk.wmT) if cross else None
        dgc = rows.ln_bwd32(c2, dc, dcn1, pk.g1, dcn2, pk.g2, dcn3, pk.gkv)                  # dc += the LayerNorm backwards
        with _tf32():
            dws, dwu, dwo = _mm32(dt, hh), _mm32(dab, x2), _mm32(dyo, gated)
            dw1, dw2, dw3 = _mm32(dmod[:, 0:256], cn1), _mm32(dmod[:, 256:512], cn2), _mm32(dmod[:, 512:768], c2)
            if cross:
                dwqg, dwkv, dwm = _mm32(dp[:, :2 * DS_], x1), _mm32(dp[:, 2 * DS_:], xkv), _mm32(dmkv, cnkv)
            else:
                dwp = _mm32(dp, x1)
        dcols = dmod.sum(0)
        dbq = dp[:, qc:qc + DS_].sum(0)
    grads = {
        "attention.ada_ln_in.ln_cond.weight": dgc[0:128], "attention.ada_ln_in.to_scale.weight": dw1[:128],
        "attention.ada_ln_in.to_scale.bias": dcols[0:128], "attention.ada_ln_in.to_bias.weight": dw1[128:],
        "attention.to_query.bias": dbq, "attention.ln_pair.weight": dgamma, "attention.to_bias.weight": dwb,
        "attention.to_out.weight": dwo, "attention.to_scale.weight": dw3[:128], "attention.to_scale.bias": dcols[512:640],
        "transition.ada_ln_in.ln_cond.weight": dgc[128:256], "transition.ada_ln_in.to_scale.weight": dw2[:128],
        "transition.ada_ln_in.to_scale.bias": dcols[256:384], "transition.ada_ln_in.to_bias.weight": dw2[128:],
        "transition.expand_a.weight": torch.cat([dwu[0:128], dwu[256:384]]), "transition.expand_b.weight": torch.cat([dwu[128:256], dwu[384:512]]),
        "transition.squeeze.weight": dws, "transition.to_scale.weight": dw3[128:], "transition.to_scale.bias": dcols[640:768],
    }
    if cross:
        grads.update({"attention.to_query.weight": dwqg[:128], "attention.to_gate.weight": dwqg[128:],
                      "attention.to_key.weight": dwkv[:128], "attention.to_value.weight": dwkv[128:],
                      "attention.ada_ln_kv.ln_cond.weight": dgc[256:384], "attention.ada_ln_kv.to_scale.weight": dwm[:DS_],
                      "attention.ada_ln_kv.to_scale.bias": dmkv[:, :DS_].sum(0), "attention.ada_ln_kv.to_bias.weight": dwm[DS_:]})
    else:
        grads.update({"attention.to_query.weight": dwp[0:128], "attention.to_key.weight": dwp[128:256],
                      "attention.to_value.weight": dwp[256:384], "attention.to_gate.weight": dwp[384:512]})
    names = WEIGHTS + CROSS_WEIGHTS[: len(weights) - len(WEIGHTS)]
    per_weight = [grads[name].to(w.dtype).reshape(w.shape).clone() for name, w in zip(names, weights, strict=True)]
    dpair = dz[: pair.shape[1]].reshape(pair.shape).clone()
    return [_unpadded(ds.view(a, 1, n, DS_), single.shape), _unpadded(dc.view(a, 1, n, DC_), cond.shape), dpair, *per_weight]


class _LocalBlock32(torch.autograd.Function):
    """The fp32 path's block forward and backward (TF32 kernels); gradients for single, cond, pair and every weight."""

    @staticmethod
    def forward(ctx, mask, single, cond, pair, *weights):
        out, *saved = _block_fwd32(single, cond, pair, mask, list(weights), True)
        ctx.save_for_backward(mask, single, cond, pair, *saved, *weights)
        ctx.n_saved = len(saved)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        mask, single, cond, pair, *rest = ctx.saved_tensors
        saved, weights = rest[: ctx.n_saved], rest[ctx.n_saved :]
        grads = _block_bwd32(dy, single, cond, pair, mask, list(weights), list(saved))
        return (None, *grads)


def _infer32(single, cond, pair, mask, weights):
    return _block_fwd32(single, cond, pair, mask, list(weights), False)[0]
