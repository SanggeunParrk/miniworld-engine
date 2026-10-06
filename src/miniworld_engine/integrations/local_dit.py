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
"""

from __future__ import annotations

import os
import warnings
import weakref

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


def serves(module, single, cond, pair, mask) -> bool:
    if _FAILED or os.environ.get("MINIWORLD_LOCAL_DIT_SM100", "1") == "0" or settings.current().engine_backend == "triton":
        return False
    from miniworld_engine.modules.dispatch import KernelBackend

    a, tr = module.attention, module.transition
    if a._backend != KernelBackend.TRITON or a.use_qk_norm:
        return False
    if not (single.is_cuda and torch.cuda.get_device_capability(single.device) == (10, 0)):
        return False
    if (single.dtype, cond.dtype, pair.dtype) != (BF, BF, BF):
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
    return _kernels_ready(index)


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
    if torch.is_grad_enabled() and any(t.requires_grad for t in (single, cond, pair, *weights)):
        return _LocalBlock.apply(mask, single, cond, pair, *weights)
    return _infer(single, cond, pair, mask, weights)
