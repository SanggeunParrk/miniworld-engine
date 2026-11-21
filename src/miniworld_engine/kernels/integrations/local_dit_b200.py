"""The block-local AF3 atom block on the B200 (sm_100a, card ``b200_sm100``): the tcgen05 / TMEM / TMA local DiT of
``origin/b200/ladder-final`` (dc425caa5, ``integrations/local_dit.py``), B200 only. ``modules/atom_dit._LocalAtomDiT`` calls
:func:`run_block` when :func:`reason` accepts the call (``AtomDiTProtenix``'s chained K / V AdaLN, ``AtomDiTOpenFold3``'s parallel one,
the clipped ``"pad"`` or AlphaFold 3's slid ``"slide"`` key window; bf16 activations with bf16 or fp32 master parameters, or fp32
activations and parameters on the TF32 kernels; inference and training).

One call runs, in place of the module path,

    forward   cond_fwd -> pre_fwd(_kv) -> pair bias (trunked) -> local attn_fwd -> post_fwd
    backward  tr_bwd_gate -> post_bwd -> local attn_dq / attn_dkv (+ dbias) -> pair-bias backward -> pre_bwd(_kv) -> cond_bwd, then every
              dense weight gradient in one grouped split-K launch (``b200_atom.atom_dw``, fixed order: deterministic).

The row kernels (conditioning, input projections, post-attention, transition, and their backward) are ``b200_atom``'s; the attention and
the pair bias are ``b200_atom_local``'s: 32 queries x 128 keys per window, the trunked pair ``[nwin, 32, 128, 16]`` (bias ``LN(z) Wb``
per window, shared by the samples), no N x N tensor anywhere. The slid window: the windows it moves run as edge CTAs of the same
attention launches (``b200_atom_local/slide.cuh``, ``lslide.cu``). N is padded to a multiple of 128 for the row kernels (zero rows, the
padded atoms masked as keys, the pair padded with zero windows) and the first N rows come back.

**Hoisted per-item tables (inference).** The conditioning tables and the windowed pair bias depend on the conditioning / pair and the
weights only; an inference call takes them from ``_capture.hoisted_table`` (made once per tensor and in-place version and weights,
scoped to the CUDA-graph capture unless the inputs are declared static). Training computes them every call.

**The fp32 path.** fp32 single / cond / pair with fp32 parameters (no autocast): every product on tcgen05 kind::tf32 (operands rounded
to nearest tf32), the LayerNorms, AdaLNs, gates, softmax, residual stream, saved activations and gradients fp32; the weight gradients
on the grouped split-K kernel in tf32. Inference runs the fused ``atom_pre_tf32`` / ``atom_post_tf32`` kernels around the attention.
"""

from __future__ import annotations

import functools
import os
from types import SimpleNamespace
from typing import cast

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import device_constant, opaque
from miniworld_engine.kernels._device import num_sms
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
    """The attribute tree ``b200_atom.cond_params`` / ``pre_params`` / ``post_params`` read (``block.attention.to_query.weight``)."""
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
    """The kernels' stacked bf16 weights and (``backward``) the backward's transposed A operands (inference: an empty dict)."""
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom as sm100

    block = _namespace(weights)
    with torch.no_grad():
        wc, wp, wo = (
            sm100.cond_params(block),
            sm100.pre_params(block),
            sm100.post_params(block),
        )

        def transposed(t):
            return t.t().contiguous()

        wb = (
            {
                "WsT": transposed(wo[2]),
                "WuT": transposed(wo[1]),
                "WoT": transposed(wo[0]),
                "WpT": transposed(wp[0]),
                "WmodT": transposed(wc[0]),
            }
            if backward
            else {}
        )
    return wc, wp, wo, wb


# The packs are ``kernels._capture.param_forms``: the weights' tensor objects (held weakly) and versions decide, not an address a new
# tensor may reuse.
def _weights(weights, backward=False):
    """Scoped to the CUDA-graph capture (``kernels._capture``): a training capture packs once and records it. ``backward``: the pack also
    holds the transposed A operands (part of the key). Inference (``backward`` False): a capture reads the pack its eager warm-up made
    (``capture_reads_eager``, the engine's inference-pack contract), so a replay runs no cast / cat kernel."""
    return _capture.param_forms(("local_dit", bool(backward)), weights, lambda: _pack(weights, backward), capture_reads_eager=not backward)


def _padded(t, n):
    """[A, 1, N, d] -> [A, 1, n, d], zero rows appended (the tensor itself when n == N)."""
    return t if t.shape[2] == n else torch.nn.functional.pad(t, (0, 0, 0, n - t.shape[2]))


def _unpadded(t, shape):
    """[A, 1, n, d] (a view of the kernels' rows) -> the caller's [A, 1, N, d]."""
    return t if t.shape == shape else t[:, :, : shape[2]].contiguous()


def _padded_length(n0: int) -> int:
    return (n0 + 127) // 128 * 128


QUERIES, KEYS = 32, 128
F32 = torch.float32
#: inference hoists (``_capture.weight_cache``: cleared with the other weight caches)
_HOIST_COND: dict = _capture.weight_cache("local_dit_b200.hoist_cond")  # (conditioning tensor, weights) -> the conditioning tables
_HOIST_BIAS: dict = _capture.weight_cache("local_dit_b200.hoist_bias")  # (pair tensor, LN / bias weights) -> the windowed pair bias
#: Inference: the attention as a programmatic dependent launch (``-DLATTN_PDL``: ``post_fwd`` stages its weights while it runs; 41.0 ->
#: 36.9 us at 1024 atoms, ~2 us at the larger lengths).
_PDL_ATTN = True
_ROW_KERNELS = ("cond", "cond_kv", "pre", "prex", "prexp", "post", "trg", "postb", "preb", "prebx", "prebxp", "condb")
#: The parameters of the cross-attention mode's second AdaLN (keys / values), appended to ``WEIGHTS``.
CROSS_WEIGHTS = (
    "attention.ada_ln_kv.ln_cond.weight",
    "attention.ada_ln_kv.to_scale.weight",
    "attention.ada_ln_kv.to_scale.bias",
    "attention.ada_ln_kv.to_bias.weight",
)


def _hoisting() -> bool:
    return True


@device_constant
def _kernels_ready(index: int) -> bool:
    """Build and load the bf16 row kernels and the local attention kernels on ``index`` once (raises on a toolchain / driver failure).
    A constant to ``torch.compile``."""
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom as rows
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom_local as local

    for name in _ROW_KERNELS:
        rows.atom_kernel(name, index)
    for name in local.KERNELS:
        local._load(name, index)
    return True


@device_constant
def _kernels32_ready(index: int) -> bool:
    """Build and load the fp32 path's kernels (``b200_atom.KERNELS_TF32`` and the fused inference pair, ``b200_atom_local.KERNELS_TF32``)
    on ``index`` once (raises on a failure)."""
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom as rows
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom_local as local

    rows.load_all32(index)
    rows.load_fused32(index)
    for name in local.KERNELS_TF32:
        local._load32(name, index)
    return True


def _fused32_ready(device) -> bool:
    """The fused inference kernels of the fp32 path (``b200_atom`` ``fused_tf32.cu``: AdaLNs + projections, and gate + Wo + AdaLN 2 +
    SwiGLU + Ws with both residuals): two launches in place of eight (loaded with the rest by ``_kernels32_ready``)."""
    return _kernels32_ready(device.index if device.index is not None else torch.cuda.current_device())


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


def _kv_pack(weights, infer=False):
    """The cross-attention mode's operands: Wm [256, 128] = [to_scale; to_bias] bf16, the scale bias and the LN weight (fp32), Wk, Wv
    (bf16) and [Wk; Wv] [256, 128]. Packed once per set of weights (pointer and version), scoped to the CUDA-graph capture like every
    weight pack."""

    def build():
        gamma, w_scale, b_scale, w_bias = weights[len(WEIGHTS) :]
        wk, wv = (
            weights[WEIGHTS.index("attention.to_key.weight")].to(BF),
            weights[WEIGHTS.index("attention.to_value.weight")].to(BF),
        )
        return torch.cat([w_scale, w_bias]).to(BF), b_scale.float(), gamma.float(), wk, wv, torch.cat([wk, wv])

    tail = [
        *weights[len(WEIGHTS) :],
        weights[WEIGHTS.index("attention.to_key.weight")],
        weights[WEIGHTS.index("attention.to_value.weight")],
    ]
    return _capture.param_forms(("local_dit_b200_kv",), tail, build, capture_reads_eager=infer)  # (inference: as ``_weights``)


def _cond_kv_pack(weights, wc):
    """Inference, cross mode: the operands of ``cond_fwd``'s ``-DKV=1`` build, which makes mod and the K / V AdaLN's tables in one launch:
    W [1152, 128] bf16 = [Wmod; Wkv_scale; Wkv_shift; 0], b [1152] fp32 = [bmod; 0; 0; 0], the K / V cond LayerNorm weight (fp32). A capture
    reads the eager warm-up's pack (``capture_reads_eager``)."""

    def build():
        wm, _, gkv, *_ = _kv_pack(weights, True)
        with torch.no_grad():
            w = torch.cat([wc[0], wm, torch.zeros(DS_, DS_, device=wm.device, dtype=BF)]).contiguous()
            b = torch.cat([wc[1], torch.zeros(3 * DS_, device=wm.device)]).contiguous()
        return w, b, gkv.contiguous()

    return _capture.param_forms(("local_dit_b200_cond_kv",), weights, build, capture_reads_eager=True)


def _cond_tables(cond, c2_fn, n, weights, wc, cross, save, ident=False, pf=None):
    """The conditioning tables of a call: mod [rows, 768] (cond_fwd) and, in the cross-attention mode, cn (the LN of the conditioning) and
    the key / value modulation mkv [rows, 256]. They depend on the conditioning tensor and the weights only -- the same at every block call
    that reads one conditioning (a sampler's steps), so an inference call (``save`` False) takes them from the hoist."""
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom as rows
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom_local as local

    def build():
        c2 = c2_fn()
        if cross and not save:  # inference: mod and the K / V tables in one launch (no LayerNorm launch, no cuBLAS product)
            w9, b9, gkv = _cond_kv_pack(weights, wc)
            mod, mkv = rows.cond_fwd(c2, w9, b9, wc[2], wc[3], g3=gkv)
            return mod, None, mkv
        mod = rows.cond_fwd(c2, *wc, ident=ident, pf=pf)
        if not cross:
            return mod, None, None
        wm, _, gkv, *_ = _kv_pack(weights, not save)
        cn = local.cond_ln(c2, gkv)
        return mod, cn, cn @ wm.t()

    if save or not _hoisting():
        return build()
    return _hoisted(_HOIST_COND, cond, (tuple((w.data_ptr(), w._version) for w in weights), n, ident), build)


def _pair_bias(pair, n, weights, save):
    """The windowed pair bias LN(z) Wb of the trunked pair, once per pair tensor in an inference call (it depends on the pair and two weights).
    Its own launch (``lbias.cu``): round 3 part 4 measured it against the pair-bias rows riding the conditioning launch (``cond_fwd -DPB``,
    removed) -- the conditioning CTAs hold a whole SM each, so those rows ran after the tables at one CTA an SM; the stand-alone kernel at
    eight CTAs an SM is level up to 8192 atoms and faster above (Protenix master 16384 atoms 116.7 -> 110.4 us, 12288: 92.1 -> 90.1)."""
    from miniworld_engine.kernels.augmented_attention.cuda import (
        b200_atom_local as local,
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


def _var_packs(weights, kqln, up, backward=False):
    """The variant block's weight packs in ONE launch (``b200_atom.var_pack``; the per-tensor packing of ``_pack`` recorded ~18 small kernels
    into every captured step): (wc, wp, wo) as ``b200_atom.cond_params`` / ``pre_params`` / ``post_params`` give them, kq fp32 [4, 128] =
    [wq | bq | wk | bk] (KQLN), Wu [256, 128] bf16 (UP) and, ``backward``, the backward row kernels' transposed bf16 A operands wb = {WsT,
    WuT, WoT, WpT, WmodT[, WupT]} as ``_pack`` gives them (else None). ``weights``: the 23 canonical leaves then the option leaves
    (``_variant_weights``). Packed once per set of weights (pointer and version; ``_capture.param_forms``, scoped to a CUDA-graph capture)."""
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom as rows

    def build():
        w = dict(zip(WEIGHTS, weights[: len(WEIGHTS)], strict=True))
        extra = list(weights[len(WEIGHTS) :])
        a1, a2, at, tr = "attention.ada_ln_in.", "transition.ada_ln_in.", "attention.", "transition."
        mats = [w[a1 + "to_scale.weight"], w[a1 + "to_bias.weight"], w[at + "to_scale.weight"], w[a2 + "to_scale.weight"], w[a2 + "to_bias.weight"],
                w[tr + "to_scale.weight"], w[at + "to_query.weight"], w[at + "to_key.weight"], w[at + "to_value.weight"], w[at + "to_gate.weight"],
                w[at + "to_out.weight"], w[tr + "expand_a.weight"], w[tr + "expand_b.weight"], w[tr + "squeeze.weight"]]
        if up:
            mats.append(extra[4 if kqln else 0])
        vecs = [w[a1 + "to_scale.bias"], None, w[at + "to_scale.bias"], w[a2 + "to_scale.bias"], None, w[tr + "to_scale.bias"],
                w[a1 + "ln_cond.weight"], w[a2 + "ln_cond.weight"], w[at + "to_query.bias"], *(extra[:4] if kqln else [])]
        with torch.no_grad():
            bf, vec, bt = rows.var_pack([t.detach() for t in mats], [None if t is None else t.detach() for t in vecs], backward)

        def mat(buf, row0, nrow, shape):
            return buf[row0 * DS_ : (row0 + nrow) * DS_].view(shape)

        wc = (mat(bf, 0, 768, (6 * DS_, DS_)), vec[0:6].reshape(-1), vec[6], vec[7])
        wp = (mat(bf, 768, 512, (4 * DS_, DS_)), vec[8])
        wo = (mat(bf, 1280, 128, (DS_, DS_)), mat(bf, 1408, 512, (4 * DS_, DS_)), mat(bf, 1920, 256, (DS_, 2 * DS_)))
        kq = vec[9:13] if kqln else None
        wup = mat(bf, 2176, 256, (2 * DS_, DS_)) if up else None
        wb = None
        if backward:
            wb = {
                "WsT": mat(bt, 0, 256, (2 * DS_, DS_)),
                "WuT": mat(bt, 256, 512, (DS_, 4 * DS_)),
                "WoT": mat(bt, 768, 128, (DS_, DS_)),
                "WpT": mat(bt, 896, 512, (DS_, 4 * DS_)),
                "WmodT": mat(bt, 1408, 768, (DS_, 6 * DS_)),
            }
            if up:
                wb["WupT"] = mat(bt, 2176, 256, (DS_, 2 * DS_))
        return wc, wp, wo, kq, wup, wb

    # inference (grad off): a capture reads the pack its eager warm-up made (``capture_reads_eager``, the contract of the other B200 / H100
    # inference packs: triattn_b200, swa_dit.dispatch, trimul_h100): a replay runs no packing kernel
    return _capture.param_forms(
        ("local_dit_b200_var", bool(kqln), bool(up), bool(backward)),
        list(weights),
        build,
        capture_reads_eager=not backward and not torch.is_grad_enabled(),
    )


def _var_pack(weights, kqln, up):
    """``_var_packs``' forward packs (wc, wp, wo, kq, wup)."""
    return _var_packs(weights, kqln, up)[:5]


_UP_NAME = "transition.up.weight"  # Boltz-2's up-gate weight (an option leaf: after the canonical ones)


def _dpair(dz, pair):
    """The pair gradient in the caller's shape: the kernel's own buffer when the pair was not padded (a view: nothing to copy), else a copy
    of its rows."""
    if dz.shape[0] == pair.shape[1] and dz.dtype == pair.dtype:
        return dz.view(pair.shape)
    return dz[: pair.shape[1]].reshape(pair.shape).to(pair.dtype).clone()


def _var_pack_bwd(weights, kqln, up):
    """``_var_packs`` with the backward's transposed operands: (wc, wp, wo, wb, wup)."""
    wc, wp, wo, _, wup, wb = _var_packs(weights, kqln, up, True)
    return wc, wp, wo, wb, wup


@functools.lru_cache(maxsize=8)
def _side_stream(index: int) -> torch.cuda.Stream:
    """A side stream per device (the variant block's pair bias runs on it beside the main chain)."""
    with torch.cuda.device(index):
        return torch.cuda.Stream()


def _fork_bias(dev, build):
    """``build()`` (the pair bias, + Chai-1's conformer mask) on the device's side stream, beside what the caller launches next on the current
    one (a fork / join in a captured graph): -> (bias, join) -- ``torch.cuda.current_stream().wait_event(join)`` before the bias is read."""
    cur = torch.cuda.current_stream()
    side = _side_stream(dev.index if dev.index is not None else torch.cuda.current_device())
    fork = torch.cuda.Event()
    fork.record(cur)
    side.wait_event(fork)
    with torch.cuda.stream(side):
        bias = build()
    join = torch.cuda.Event()
    join.record(side)
    if not torch.cuda.is_current_stream_capturing():
        bias.record_stream(cur)
    return bias, join


def _pair_bias_conf(pair, n, weights, save, uid, mask, n0, slide, f32=False):
    """``_pair_bias`` with Chai-1's conformer mask (``b200_atom.conf_mask_``, in place on the fresh bias): hoisted per pair tensor, mask,
    conformer ids and window in an inference call. ``uid``: [n] int64, padded with -1."""
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom as rows
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom_local as local

    def build():
        bias = (local.pair_bias_fwd32 if f32 else local.pair_bias_fwd)(_trunk(pair, n), weights[_GAMMA], weights[_WBIAS])
        return rows.conf_mask_(bias, uid, _keys(mask, n0, n), n, slide)

    if save or not _hoisting():
        return build()
    g, wb = weights[_GAMMA], weights[_WBIAS]
    key = ((g.data_ptr(), g._version, wb.data_ptr(), wb._version), n, "conf32" if f32 else "conf", _tkey(uid), None if mask is None else _tkey(mask), bool(slide))
    return _hoisted(_HOIST_BIAS, pair, key, build)


#: fp32 inference (round 3 part 8): the pair bias as a programmatic dependent of atom_pre_tf32, running beside it (it reads the pair only but
#: ran after it: 1024 atoms 4.4 us, 6144: 13.3 us of the replay's critical path)
_EARLY_BIAS = os.environ.get("MW_LD8_EB", "1") != "0"
#: ... up to this many rows, or _EARLY_ROWS_SHARED when the samples share the conditioning (its tables are small: the projections start
#: sooner). Above, atom_pre_tf32 holds every SM and the bias CTAs beside it ran 2-3 x slower (one sample, us, own launch -> beside:
#: 8192 atoms 98.3 -> 100.5, 16384: 161.8 -> 176.1; 6144: 92.2 -> 86.0; 5 shared samples of 3072 atoms 90.1 -> 86.0). The bf16 path keeps
#: its own launch (beside pre_fwd, which lets its dependents start before its own wait, it also ran 3 x slower: 2048 atoms 43.0 -> 55.3).
_EARLY_ROWS, _EARLY_ROWS_SHARED = 6144, 16384


def _early_bias(m, mrows):
    """Whether an fp32 inference call of m rows (tables of mrows rows) launches its pair bias beside atom_pre_tf32."""
    return _EARLY_BIAS and m <= (_EARLY_ROWS_SHARED if mrows < m else _EARLY_ROWS)


def _pair_bias_deferred(pair, n, weights):
    """-> (bias, launch or None): the fp32 inference pair bias (hoisted per pair tensor as ``_pair_bias32``) made ready here and launched
    by ``launch()`` right after ``pre32`` (``b200_atom_local.pair_bias_early32``). A hoisted table needs no launch; a
    later rebuild of the entry (a refresh) launches at once."""
    from miniworld_engine.kernels.augmented_attention.cuda import (
        b200_atom_local as local,
    )

    g, wb = weights[_GAMMA], weights[_WBIAS]
    st: dict = {"defer": True}

    def build():
        z = _trunk(pair, n)
        if st["defer"]:
            bias, st["launch"] = local.pair_bias_early32(z, g, wb)
            return bias
        return local.pair_bias_fwd32(z, g, wb)

    key = ("tf32", (g.data_ptr(), g._version, wb.data_ptr(), wb._version), n)
    bias = _hoisted(_HOIST_BIAS, pair, key, build) if _hoisting() else build()
    st["defer"] = False
    return bias, st.pop("launch", None)


#: bf16 inference (round 3 part 12): the pair bias made by the input-projection launch's compute warps before their first tile (``pre_*_eb``,
#: ``-DEB=1``) -- they wait there for the conditioning tables anyway; no bias launch on the chain (Protenix: it ran after pre_fwd, 6.9 us at
#: 2048 atoms; the variant path: forked beside the tables, where its 8-CTA-an-SM grid held the SMs the tables wanted: cond started at 7.6 us)
_EB16 = os.environ.get("MW_LD12_EB", "1") != "0"
#: ... up to this many rows m = A N (one conditioning per sample), or _EB16_ROWS_SHARED when the samples share it (small tables: the
#: projection CTAs start sooner). Above, the projections hold every SM and the bias rides their critical path (Protenix 16384 atoms
#: 108.5 -> 114.7 us; 8192: OpenFold3 69.6 -> 67.5)
_EB16_ROWS = int(os.environ.get("MW_LD12_EBR", "12288"))
_EB16_ROWS_SHARED = int(os.environ.get("MW_LD12_EBRS", "16384"))


def _pair_bias_pre(pair, n, weights):
    """-> (bias, eb or None): the bf16 inference pair bias (hoisted per pair tensor, the key of ``_pair_bias``: the same bits) made ready here;
    ``eb`` = the trailing arguments of the projection launch that computes it (``b200_atom_local.pair_bias_in_pre``), None when the hoist
    already holds it (no work). Call it before the chain's launches."""
    from miniworld_engine.kernels.augmented_attention.cuda import (
        b200_atom_local as local,
    )

    g, wb = weights[_GAMMA], weights[_WBIAS]
    st: dict = {"defer": True}

    def build():
        z = _trunk(pair, n)
        if st["defer"] and z.dtype == BF:
            bias, st["eb"] = local.pair_bias_in_pre(z, g, wb)
            return bias
        return local.pair_bias_fwd(z, g, wb)

    bias = _hoisted(_HOIST_BIAS, pair, ((g.data_ptr(), g._version, wb.data_ptr(), wb._version), n), build) if _hoisting() else build()
    st["defer"] = False
    return bias, st.pop("eb", None)


def _trunk(pair, n):
    """[F, nwin, 32, 128, 16] -> [F n / 32, 32, 128, 16] (the F folds' trunks end to end; F = 1: one trunk): zero windows appended for
    the padded atoms."""
    nw = n // QUERIES
    z = pair if pair.shape[1] == nw else F.pad(pair, (0, 0, 0, 0, 0, 0, 0, nw - pair.shape[1], 0, 0))
    return z.reshape(pair.shape[0] * nw, *pair.shape[2:]).contiguous()


def _fork(fn, which=0):
    """``fn()`` on side stream ``which`` (0 / 1) after the work queued so far -> (its result, the event :func:`_join` makes the current
    stream wait for). The launches of ``fn`` then run beside the current stream's next ones (CUDA-graph capturable: a fork and a join of the
    capture); the tensors they write must not be read on the current stream before the join."""
    from miniworld_engine.kernels.swa_dit.cuda import sm100 as swa

    cur = torch.cuda.current_stream()
    side = swa._side_streams(cur.device.index)[which]
    start = torch.cuda.Event()
    start.record(cur)
    side.wait_event(start)
    with torch.cuda.stream(side):
        result = fn()
    done = torch.cuda.Event()
    done.record(side)
    return result, done


def _join(*events):
    for ev in events:
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)


# ------------------------------------------------------------------------------------------------------------ the two ops
def _fwd_fake(single, cond, pair, mask, weights, save, parallel=False, slide=False, qblock=False, var=False, ident=False, pars=False, up=False, kqln=False, kq_eps=1e-5, eps=1e-5, uid=None):
    out = torch.empty_like(single)
    if not save:
        return [out]
    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    rows = a * n

    def tile(width=DS_, r=rows):
        return single.new_empty((r, width))

    rc = cond.shape[0] * n  # the conditioning tables' rows: per sample, or once when the samples share the conditioning

    saved = [
        out,
        tile(6 * DS_, rc),
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
    if var:
        return [*saved, tile(2 * DS_)] if kqln else saved  # (KQLN: the raw q | k)
    return [*saved, tile(DS_, rc), tile(2 * DS_, rc), tile()] if len(weights) > len(WEIGHTS) else saved


@opaque(fake=_fwd_fake, name="local_dit_b200_fwd")
def _block_fwd(
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    weights: list[torch.Tensor],
    save: bool,
    parallel: bool = False,
    slide: bool = False,
    qblock: bool = False,
    var: bool = False,
    ident: bool = False,
    pars: bool = False,
    up: bool = False,
    kqln: bool = False,
    kq_eps: float = 1e-5,
    eps: float = 1e-5,
    uid: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """The block output [A, F, N, 128] bf16 and, when ``save``, the activations the backward reads (mod, q, k, v, sg, x1, bias, O, LSE,
    u, a2, x2, t: all fresh tensors). ``parallel``: AlphaFold 3's K/V AdaLN of the raw atoms; ``slide``: its slid key window (``qblock``:
    IntelliFold2's slide_qblock rule). ``cond`` [1, F, N, 128] with A > 1 samples: the conditioning the samples share (``block``): its tables
    are made once ([F N, ..]) and the row kernels read them at row r % (F N). F > 1 folds (inference only, ``reason``): ``pair``
    [F, nwin, 32, 128, 16], one key mask; the rows are [A][F][N] and the attention takes fold f's pair bias for its rows (one launch).
    ``var`` (2026-11-17): the variant block (Boltz-2 / RoseTTAFold3 / Chai-1, ``variant_leaves``): one AdaLN for q and k / v, ``weights`` the
    23 canonical leaves then the option leaves (KQLN: wq bq wk bk, UP: Wu); ``ident`` / ``pars`` / ``up`` / ``kqln`` the kernel builds
    (``b200_atom.KERNELS_VAR``), ``eps`` the activations' AdaLN epsilon, ``uid`` [N] Chai-1's conformer ids (inference only so far)."""
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom as rows
    from miniworld_engine.kernels.augmented_attention.cuda import (
        b200_atom_local as local,
    )

    a, nf, n0, _ = single.shape
    n = _padded_length(n0)
    a = a * nf  # (sample, fold) row blocks
    m = a * n
    with torch.cuda.device(single.device):
        join = None
        eb = None
        cap = _EB16_ROWS_SHARED if cond.shape[0] * nf * n < m else _EB16_ROWS
        early = _EB16 and not save and uid is None and m <= cap and pair.dtype == BF and (var or len(weights) > len(WEIGHTS))
        if early:  # round 3 part 12: the bias inside the projection launch (ready now if hoisted: eb None)
            bias, eb = _pair_bias_pre(pair, n, weights)
        elif var:  # the pair bias (+ conformer mask) on a side stream, beside the pack, conditioning and input projections (graph fork / join)
            cur = torch.cuda.current_stream()
            side = _side_stream(single.device.index if single.device.index is not None else torch.cuda.current_device())
            fork = torch.cuda.Event()
            fork.record(cur)
            side.wait_event(fork)
            with torch.cuda.stream(side):
                if uid is not None:  # Chai-1: a real query sees only the keys of its own conformer (-inf in the bias at the keys the kernel reads)
                    bias = _pair_bias_conf(pair, n, weights, save, uid, mask, n0, slide)
                else:
                    bias = _pair_bias(pair, n, weights, save)
            join = torch.cuda.Event()
            join.record(side)
            if not torch.cuda.is_current_stream_capturing():
                bias.record_stream(cur)
        cross = len(weights) > len(WEIGHTS) and not var
        if var:
            wc, wp, wo, kq, wup = _var_packs(weights, kqln, up, save)[:5]  # (training: the backward's operands in the same launch)
        else:
            wc, wp, wo, _ = _weights(weights[: len(WEIGHTS)], backward=save)
            kq = wup = None
        s2 = _padded(single, n).reshape(m, DS_).contiguous()
        mod, cn, mkv = _cond_tables(
            cond, lambda: _padded(cond, n).reshape(cond.shape[0] * nf * n, DC_).contiguous(), n, weights, wc, cross, save, ident,
            True if var else None,
        )
        qkr = None
        xkv = None
        from miniworld_engine.kernels.augmented_attention.cuda.b200_atom import f8 as _f8

        f8 = _f8 if _f8.ENABLED else None  # FP8 (opt-in, ``b200_atom.f8``): inference, the plain and cross pre_fwd builds / the non-up post_fwd
        if cross and f8 is not None and f8.serves("pre", m, save):
            bs = _kv_pack(weights, not save)[1]
            w_pre, b_pre = wp
            q, k, v, sg = f8.pre_fwd(s2, mod, w_pre, b_pre, mkv=mkv, bs=bs, parallel=parallel, eb=eb)
        elif cross:  # keys and values from a second AdaLN of x1 (or of the raw atoms, ``parallel``), inside pre_fwd
            bs = _kv_pack(weights, not save)[1]
            w_pre, b_pre = wp
            q, k, v, sg, x1, xkv = cast(tuple, rows.pre_fwd_kv(s2, mod, w_pre, b_pre, mkv, bs, save=save, parallel=parallel, eb=eb))
        elif not kqln and f8 is not None and f8.serves("pre", m, save):
            assert eb is None or var, "eb: the variant (ADA1) or cross builds"
            q, k, v, sg = f8.pre_fwd(s2, mod, *wp, eps=eps, ada1=var, eb=eb)
        elif kqln and f8 is not None and f8.serves("pre", m, save):
            q, k, v, sg = f8.pre_fwd(s2, mod, wp[0], wp[1], eps=eps, ada1=True, eb=eb, kq=kq, kq_eps=kq_eps)
        elif kqln:
            q, k, v, sg, x1, qkr = rows.pre_fwd_kq(s2, mod, wp[0], wp[1], kq, kq_eps, save=save, eps=eps, ada1=True, eb=eb)
        else:
            assert eb is None or var, "eb: the variant (ADA1) or cross builds"
            q, k, v, sg, x1 = rows.pre_fwd(s2, mod, *wp, save=save, eps=eps, ada1=var, eb=eb)
        if join is not None:
            torch.cuda.current_stream().wait_event(join)
        elif not early:
            bias = _pair_bias(pair, n, weights, save)
        o, lse = local.attn_fwd(
            q.view(a, n, DS_),
            k.view(a, n, DS_),
            v.view(a, n, DS_),
            bias,
            _keys(mask, n0, n),
            slide=slide,  # the windows the slid key window moves: edge CTAs of the same launch
            qblock=qblock,
            folds=nf,
            # the variant path: the attention as a programmatic dependent launch (post_fwd stages its weights while it runs: 41 -> 35 us
            # at 1024 atoms; measured against the join before pre_fwd and against the bias on the main stream: level or slower)
            pdl=var or (not save and _PDL_ATTN),
        )
        if f8 is not None and f8.serves("post", m, save):
            out = f8.post_fwd(s2, o.view(m, DS_), sg, mod, *wo, eps=eps, pars=pars, ada1=var, wup=wup)
            u = a2 = x2 = t = None
        else:
            out, u, a2, x2, t = rows.post_fwd(s2, o.view(m, DS_), sg, mod, *wo, save=save, eps=eps, pars=pars, wup=wup, ada1=var)
    out = _unpadded(out.view(a // nf, nf, n, DS_), single.shape)
    if not save:
        return [out]
    saved = [out, mod, q, k, v, sg, x1, bias, o.view(m, DS_), lse, u, a2, x2, t]
    if var and kqln:
        assert qkr is not None
        return [*saved, qkr]
    if not cross:
        return saved
    assert cn is not None and mkv is not None and xkv is not None  # the cross mode made all three
    return [*saved, cn, mkv, xkv]


def _fused_wgrads(spec, weights, names, f32, dev, rows_r):
    """The weight gradients ``spec`` names -- (A, B, [(weight name, 128-row block), ...]) per product D = A^T B -- in one split-K launch
    (``b200_atom.AtomDw``), each written straight into a fresh buffer of its weight's shape and dtype; -> {name: gradient}. A weight made of
    several 128-row blocks (expand_a / expand_b: 256 rows) lists the block index of each destination. (One mechanism for autograd's
    AccumulateGrad on both the local and the variant blocks: the returned views below; the merge of 2026-11-17 dropped the variant path's
    ``release`` flag, which solved the same copy.)"""
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom as rows

    wmap = dict(zip(names, weights, strict=True))
    out = {}
    problems = []
    for A, B, dests in spec:
        ds = []
        for name, blk in dests:
            if name not in out:
                out[name] = torch.empty(wmap[name].shape, device=dev, dtype=wmap[name].dtype)
            ds.append(out[name].view(-1, DS_ if out[name].shape[-1] == DS_ else out[name].shape[-1])[blk * 128 : (blk + 1) * 128])
        problems.append((A, B, ds))
    rows.atom_dw(dev.index if dev.index is not None else torch.cuda.current_device(), f32)(problems, rows_r)
    # A view of each buffer, not the buffer: ``AtomDw`` holds the launch's destination views (hence the buffers) until its next call, and
    # autograd's AccumulateGrad copies a gradient another reference still holds instead of taking it -- 16 device copies a training step
    # (measured: every dense weight gradient was copied). A fresh view is referenced by nobody else and is taken as it is.
    return {name: g.view(g.shape) for name, g in out.items()}


def _wgrad(dout, x):
    """dout^T x, fp32: the weight gradients stay unrounded for fp32 parameters (an fp32 master); bf16 ones round once, as before."""
    return torch.mm(dout.t(), x, out_dtype=torch.float32)


def _bwd_fake(dy, single, cond, pair, mask, weights, saved, parallel=False, slide=False, qblock=False, var=False, ident=False, pars=False, up=False, kqln=False, kq_eps=1e-5, eps=1e-5):
    return [
        torch.empty_like(single),
        torch.empty_like(cond),
        torch.empty_like(pair),
        *(torch.empty_like(w) for w in weights),
    ]


@opaque(fake=_bwd_fake, name="local_dit_b200_bwd")
def _block_bwd(
    dy: torch.Tensor,
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    weights: list[torch.Tensor],
    saved: list[torch.Tensor],
    parallel: bool = False,
    slide: bool = False,
    qblock: bool = False,
    var: bool = False,
    ident: bool = False,
    pars: bool = False,
    up: bool = False,
    kqln: bool = False,
    kq_eps: float = 1e-5,
    eps: float = 1e-5,
) -> list[torch.Tensor]:
    """d single, d cond, d pair, then one gradient per weight in ``WEIGHTS`` order, in the weight's dtype (all fresh tensors).
    ``parallel`` / ``slide``: AlphaFold 3's K/V AdaLN of the raw atoms / slid key window. ``var`` (2026-11-17): the variant block
    (``_block_fwd``): ``weights`` the 23 canonical leaves then the option leaves, whose gradients follow in that order (KQLN: wq bq wk bk;
    UP: Wu); the flags are the forward's."""
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom as rows
    from miniworld_engine.kernels.augmented_attention.cuda import (
        b200_atom_local as local,
    )

    mod, q, k, v, sg, x1, bias, o, lse, u, a2, x2, t, *extra = saved
    cross = bool(extra) and not var
    opt_leaves = list(weights[len(WEIGHTS) :]) if var else []
    if var:
        weights = list(weights[: len(WEIGHTS)])
    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    m = a * n
    dev = single.device
    with torch.cuda.device(dev):
        wup = wupT = None
        if var:  # the forward's pack (one cat + one cast) and its transposed A operands (one cat)
            wc, _, wo, wb, wup = _var_pack_bwd([*weights, *opt_leaves], kqln, up)
            wupT = wb.get("WupT")
        else:
            wc, _, wo, wb = _weights(weights[: len(WEIGHTS)], backward=True)
        s2 = _padded(single, n).reshape(m, DS_).contiguous()
        shared = cond.shape[0] == 1 and a > 1  # one conditioning for the samples: its tables' gradient is summed over them
        c2 = _padded(cond, n).reshape(cond.shape[0] * n, DC_).contiguous()
        dy = _padded(dy, n).reshape(m, DS_).to(BF).contiguous()
        dtab = torch.empty(m, (8 if cross else 6) * DS_, device=dev, dtype=BF) if shared else None  # per sample row: [dmod | dmkv]
        tsums = local.TableSums(a, n, cross, dev) if shared else None  # (shared: the row kernels sum it over the samples instead)
        dmod = (
            dtab[:, : 6 * DS_] if dtab is not None else torch.empty(m, 6 * DS_, device=dev, dtype=BF)
        )  # [dsc1 | dbi1 | dsc2 | dbi2 | dos | dts]
        dp = torch.empty(m, 4 * DS_, device=dev, dtype=BF)  # [dq | dk | dv | dg]
        # the row kernels' column sums: one partial row per CTA (``COLS_STRIDE`` = 1024 floats; their grids are at most one CTA per SM), summed
        # in a fixed order below -- cols [0, 768): the bias columns, [768, 896): d bq, [896, 1024): d bs (K / V); qcols [0, 256): the cond LN weights
        nsm = num_sms(dev)
        pcols = torch.zeros(nsm, 1024, device=dev)
        qcols = torch.zeros(nsm, 1024, device=dev)
        dbias_cols, dbq, dbs_cols = pcols, pcols[:, 768:], pcols[:, 896:]
        dgamma_cond = qcols
        dd = torch.empty(a, NH_, n, device=dev)
        dt, hh, dab = rows.tr_bwd_gate(dy, t, mod, x2, wo[1], wb["WsT"], dmod, pcols[:, 640:], part=tsums, wup=wup)
        da, du, gated, do = rows.post_bwd(
            dab,
            dy,
            s2 if pars else a2,  # (the parallel schedule: the transition normalised the block input)
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
            part=tsums,
            eps=eps,
            pars=pars,
            ident=ident,
            wupT=wupT,
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
            slide=slide,  # the windows the slid key window moves: edge CTAs of the same launches
            qblock=qblock,
        )
        dz, dgamma, dwb = local.pair_bias_bwd(_trunk(pair, n), weights[_GAMMA], weights[_WBIAS], db)
        dmkv = dbs = dg_kv = cs_kq = None
        if cross:  # the K / V branch: dxkv -> its AdaLN -> dx1 (through AdaLN 1 below) and the conditioning
            cn, mkv, xkv = extra
        if cross:
            wm, bs, gkv, *_ = _kv_pack(weights)
            dbs = dbs_cols
            dg_kv = torch.zeros(DS_, device=dev)
            # dk Wk + dv Wv through the K / V AdaLN into dx1, then AdaLN 1: one kernel; dmkv = [d pre-sigmoid scale | dxkv]
            ds, dmkv = rows.pre_bwd_kv(
                dp,
                s2,
                mod,
                da,
                wb["WpT"],
                dmod,
                dbias_cols,
                dbq,
                x1,
                mkv,
                bs,
                dbs,
                parallel=parallel,
                dmkv=dtab[:, 6 * DS_ :] if dtab is not None else None,
                part=tsums,
            )
        else:
            if kqln:  # the flat q / k LayerNorm's backward on dp's dq, dk, in place (after the attention backward); kq from the pack
                cs_kq = rows.kqln_bwd(dp, extra[0], _var_packs([*weights, *opt_leaves], kqln, up, True)[3], kq_eps, partials=True)
            ds = rows.pre_bwd(dp, s2, mod, da, wb["WpT"], dmod, dbias_cols, dbq, eps=eps, part=tsums, ident=ident)
        if shared:  # the tables' gradient summed over the samples (in the row kernels, fixed order): the conditioning backward on N rows
            assert tsums is not None  # made above when shared
            dsum = tsums.finish()
            dmod = dsum[:, : 6 * DS_]
            if cross:
                dmkv = dsum[:, 6 * DS_ :]
        dc, cn1, cn2 = rows.cond_bwd(dmod, c2, wb["WmodT"], wc[2], wc[3], dgamma_cond, ident=ident)
        csum = None
        if not var:  # (var: the partial rows are summed into the gradients below, ``col_jobs``)
            csum = pcols.sum(0)  # the partial rows in order: deterministic
            dbias_cols, dbq, dgamma_cond = csum[:768], csum[768:896], qcols[:, :256].sum(0)
        if cross:
            assert csum is not None  # (cross is never var)
            dbs = csum[896:]
        if cross:
            assert dmkv is not None  # the cross mode's pre_bwd_kv (or the table sums) set it
            local.cond_ln_bwd(c2, dmkv @ wm, gkv, dg_kv, acc=dc)  # dc += the K / V modulation's gradient through LN(c) * gamma
        A_ = "attention."
        spec = [
            (dt, hh, [("transition.squeeze.weight", 0)]),
            (
                dab[:, : 4 * DS_] if up else dab,
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
        wg_weights = weights
        if up:  # Boltz-2's up-gate weight: du^T x2 (two 128-row blocks), in the same launch
            spec.append((dab[:, 4 * DS_ :], x2, [(_UP_NAME, 0), (_UP_NAME, 1)]))
            names = (*names, _UP_NAME)
            wg_weights = [*weights, opt_leaves[4 if kqln else 0]]
        if shared:  # the conditioning's products have N rows: their own launch
            fused = _fused_wgrads([p for p in spec if p[0].shape[0] == m], wg_weights, names, False, dev, m)
            fused.update(_fused_wgrads([p for p in spec if p[0].shape[0] != m], wg_weights, names, False, dev, c2.shape[0]))
        else:
            fused = _fused_wgrads(spec, wg_weights, names, False, dev, m)
        del spec
        if var:  # every bias / LayerNorm-weight gradient (and KQLN's four leaves) from the partial rows, in one launch
            wmap: dict[str, torch.Tensor] = dict(zip(WEIGHTS, weights, strict=True))
            small = {
                "attention.ada_ln_in.ln_cond.weight": (qcols, 0),
                "attention.ada_ln_in.to_scale.bias": (pcols, 0),
                "attention.to_query.bias": (pcols, 768),
                "attention.ln_pair.weight": (dgamma.reshape(1, -1), 0),
                "attention.to_bias.weight": (dwb.reshape(1, -1), 0),
                "attention.to_scale.bias": (pcols, 512),
                "transition.ada_ln_in.ln_cond.weight": (qcols, 128),
                "transition.ada_ln_in.to_scale.bias": (pcols, 256),
                "transition.to_scale.bias": (pcols, 640),
            }
            jobs = [(src, col0, wmap[name]) for name, (src, col0) in small.items()]
            if kqln:
                assert cs_kq is not None
                jobs += [(cs_kq, DS_ * i, w) for i, w in enumerate(opt_leaves[:4])]
            outs = rows.col_jobs(jobs)
            fused.update(zip(small, outs[: len(small)], strict=True))
            per_weight = [fused[name] for name in WEIGHTS] + (outs[len(small) :] if kqln else []) + ([fused[_UP_NAME]] if up else [])
            dpair = _dpair(dz, pair)
            return [_unpadded(ds.view(a, 1, n, DS_), single.shape), _unpadded(dc.view(cond.shape[0], 1, n, DC_), cond.shape), dpair, *per_weight]
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
    if cross:
        assert dbs is not None and dg_kv is not None  # both are set by the cross branch above
        grads.update(
            {
                "attention.ada_ln_kv.ln_cond.weight": dg_kv,
                "attention.ada_ln_kv.to_scale.bias": dbs,
            }
        )
    per_weight = [
        fused[name] if name in fused else grads[name].to(w.dtype).reshape(w.shape).clone() for name, w in zip(names, weights, strict=True)
    ]
    dpair = dz[: pair.shape[1]].reshape(pair.shape).clone()
    return [
        _unpadded(ds.view(a, 1, n, DS_), single.shape),
        _unpadded(dc.view(cond.shape[0], 1, n, DC_), cond.shape),
        dpair,
        *per_weight,
    ]


# ------------------------------------------------------------------------------------------------------------ autograd
class _LocalBlock(torch.autograd.Function):
    """The whole block forward and backward on the sm_100a kernels; gradients for single, cond, pair and every weight. ``parallel``:
    AlphaFold 3's K/V AdaLN of the raw atoms."""

    @staticmethod
    def forward(ctx, parallel, window, mask, single, cond, pair, *weights):  # window: 0 clipped, 1 slid, 2 slid by query blocks (qblock)
        out, *saved = _block_fwd(single, cond, pair, mask, list(weights), True, parallel, window > 0, window == 2)
        ctx.save_for_backward(mask, single, cond, pair, *saved, *weights)
        ctx.n_saved = len(saved)
        ctx.parallel, ctx.window = parallel, window
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        mask, single, cond, pair, *rest = ctx.saved_tensors
        saved, weights = rest[: ctx.n_saved], rest[ctx.n_saved :]
        grads = _block_bwd(dy, single, cond, pair, mask, list(weights), list(saved), ctx.parallel, ctx.window > 0, ctx.window == 2)
        return (None, None, None, *grads)


class _VarBlock(torch.autograd.Function):
    """The variant block (Boltz-2 / RoseTTAFold3 / Chai-1) forward and backward on the sm_100a kernels' variant builds; gradients for
    single, cond, pair and every leaf of ``_variant_weights`` (a constant leaf's is dropped by autograd). ``flags``: (slide, ident, pars,
    up, kqln, kq_eps, eps); ``uid``: Chai-1's padded conformer ids [N] or None."""

    @staticmethod
    def forward(ctx, flags, mask, uid, single, cond, pair, *weights):
        slide, ident, pars, up, kqln, kq_eps, eps = flags
        out, *saved = _block_fwd(single, cond, pair, mask, list(weights), True, False, slide, False, True, ident, pars, up, kqln, kq_eps, eps, uid)
        ctx.save_for_backward(mask, single, cond, pair, *saved, *weights)
        ctx.n_saved = len(saved)
        ctx.flags = flags
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        mask, single, cond, pair, *rest = ctx.saved_tensors
        saved, weights = rest[: ctx.n_saved], rest[ctx.n_saved :]
        slide, ident, pars, up, kqln, kq_eps, eps = ctx.flags
        grads = _block_bwd(dy, single, cond, pair, mask, list(weights), list(saved), False, slide, False, True, ident, pars, up, kqln, kq_eps, eps)
        return (None, None, None, *grads)


def _infer(single, cond, pair, mask, weights, parallel=False, window=0):
    return _block_fwd(single, cond, pair, mask, list(weights), False, parallel, window > 0, window == 2)[0]


def _shared_cond(single, cond):
    """B200: a conditioning that is one [1, 1, N, d] tensor expanded over the A > 1 samples (stride 0 on the sample axis: FoldForge's and
    the diffusion trainers' call) -> that [1, 1, N, d] view, so the tables are made once and their gradient is summed over the samples (autograd
    sums the expanded view's gradient back); anything else unchanged."""
    if (
        single.shape[0] > 1
        and cond.shape[0] == single.shape[0]
        and cond.stride(0) == 0
        and single.is_cuda
        and torch.cuda.get_device_capability(single.device) == (10, 0)
    ):
        return cond[:1]
    return cond


# ============================================================================================================ the fp32 path (TF32)
# fp32 single / cond / pair and fp32 parameters: every stage on CUDA, products on tcgen05 kind::tf32 (``b200_atom.gemm32``, the
# ``b200_atom_local`` *_tf32 attention kernels), the residual stream, LayerNorms, gates, softmax and every saved activation fp32; the
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
        proj = ("query", "gate", "key", "value") if cross else ("query", "key", "value", "gate")
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


def _cached32(build, weights, infer=False, eager=False):
    """``build(weights)``, reused while every weight's tensor object and version are unchanged; scoped to the CUDA-graph capture
    (``kernels._capture``) as the bf16 packs are -- except under ``kernels._capture.static_inputs()``, whose contract (the weights AND the
    conditioning / pair keep their contents between replays) covers the weights: a capture then serves from, and fills, the eager entry and a
    replay runs no packing kernel (``static_weights()`` has the same effect through ``scoped``).
    ``infer`` (an inference call) / ``eager`` (the variant blocks' inference packs): a capture reads the entry its eager warm-up made
    (``capture_reads_eager``), as ``_weights``."""
    return _capture.param_forms(
        ("local_dit_tf32", build.__name__), weights, lambda: build(weights), inputs_scope=True, capture_reads_eager=infer or eager
    )


def _weights32(weights, infer=False):
    """The forward pack (``_pack32``), cached (``_cached32``)."""
    return _cached32(_pack32, weights, infer)


def _weights32_bwd(weights):
    """The backward pack (``_pack32_bwd``), cached (``_cached32``)."""
    return _cached32(_pack32_bwd, weights)


def _cond_tables32(cond, c2_fn, n, weights, pk, cross, save):
    """The fp32 conditioning tables of a call: mod32 [rows, 768] and, in the cross mode, mkv [rows, 256]; with ``save`` also the
    LayerNorms cn1, cn2 (and cnkv) the weight gradients read. An inference call takes them from the hoist (``_cond_tables``)."""
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom as rows

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
        b200_atom_local as local,
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


def _fwd_fake32(single, cond, pair, mask, weights, save, parallel=False, slide=False, qblock=False):
    out = torch.empty_like(single)
    if not save:
        return [out]
    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    m = a * n

    def tile(width=DS_):
        return single.new_empty((m, width))

    def ctile(width=DS_):  # the conditioning's rows (n when shared by the samples)
        return single.new_empty((cond.shape[0] * n, width))

    saved = [
        out,
        ctile(6 * DS_),
        ctile(),
        ctile(),
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
    return [*saved, ctile(), ctile(2 * DS_), tile()] if len(weights) > len(WEIGHTS) else saved


@opaque(fake=_fwd_fake32, name="local_dit_b200_fwd_tf32")
def _block_fwd32(
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    weights: list[torch.Tensor],
    save: bool,
    parallel: bool = False,
    slide: bool = False,
    qblock: bool = False,
) -> list[torch.Tensor]:
    """The fp32 path's forward: the block output [A, 1, N, 128] fp32 and, when ``save``, the activations the backward reads (mod32, cn1,
    cn2, x1, P32, bias, O, LSE, gated, y, a1, x2, u, t; cross mode: cnkv, mkv, xkv -- all fresh fp32 tensors). ``parallel`` / ``slide``:
    AlphaFold 3's K/V AdaLN of the raw atoms / slid key window."""
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom as rows
    from miniworld_engine.kernels.augmented_attention.cuda import (
        b200_atom_local as local,
    )

    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    m = a * n
    dev = single.device
    with torch.cuda.device(dev):
        cross = len(weights) > len(WEIGHTS)
        pk = _weights32(weights, infer=not save)
        qc, kc, vc, gc = _P_COLS[cross]
        s2 = _padded(single, n).reshape(m, DS_).contiguous()

        def tables():
            return _cond_tables32(
                cond,
                lambda: _padded(cond, n).reshape(cond.shape[0] * n, DC_).contiguous(),  # (training: one [1, 1, N] conditioning when shared)
                n,
                weights,
                pk,
                cross,
                save,
            )

        if not save and _fused32_ready(dev):  # inference: pre (AdaLNs + projections), attention, post (the rest)

            def chain():  # the atoms' chain up to the attention's operands
                mod, _, _, _, mkv = tables()
                return (
                    mod,
                    rows.pre32(
                        s2,
                        mod[:, _S1],
                        mod[:, _B1],
                        pk.wp,
                        pk.bp[:DS_],
                        _P_RND[cross],
                        mks=mkv[:, :DS_] if mkv is not None else None,  # (mkv: the cross mode's only)
                        mkb=mkv[:, DS_:] if mkv is not None else None,
                        parallel=parallel,
                        samples=a,
                    ),
                )

            if _early_bias(m, cond.shape[0] * n):  # made ready before the chain, launched beside atom_pre_tf32 (its programmatic dependent)
                bias, launch = _pair_bias_deferred(pair, n, weights)
                mod, p = chain()
                if launch is not None:
                    launch()
            else:
                mod, p = chain()
                bias = _pair_bias32(pair, n, weights, save)
            p3 = p.view(a, n, 4 * DS_)
            o, _ = local.attn_fwd32(
                p3[..., qc : qc + DS_],
                p3[..., kc : kc + DS_],
                p3[..., vc : vc + DS_],
                bias,
                _keys(mask, n0, n),
                slide=slide,
                qblock=qblock,
            )
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
        mod, cn1, cn2, cnkv, mkv = tables()
        x1 = rows.adaln32(s2, mod[:, _S1], mod[:, _B1])  # AdaLN 1
        p = torch.empty(m, 4 * DS_, device=dev, dtype=F32)
        xkv = None
        if cross:  # q | g from x1, k | v from a second AdaLN of x1
            rows.gemm32(x1, pk.wp, p, n=2 * DS_, bias=pk.bp, rndmask=_P_RND[cross])
            assert mkv is not None  # the cross mode's K / V tables
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
            slide=slide,
            qblock=qblock,
        )
        gated = rows.gate32(p[:, gc : gc + DS_], o.view(m, DS_))
        y = torch.empty(m, DS_, device=dev, dtype=F32) if save else None
        a1 = rows.gemm32(gated, pk.wo, mode=rows.RESID_GATE, res=s2, gate=mod[:, _SO], out2=y)  # a1 = s + so * (gated Wo^T)
        x2 = rows.adaln32(a1, mod[:, _S2], mod[:, _B2])  # AdaLN 2
        u = torch.empty(m, 4 * DS_, device=dev, dtype=F32) if save else None
        h = rows.gemm32(x2, pk.wu, mode=rows.SWIGLU, out2=u)  # h = silu(x2 Wa^T) * (x2 Wb^T)
        t = torch.empty(m, DS_, device=dev, dtype=F32) if save else None
        out = rows.gemm32(h, pk.ws, mode=rows.RESID_GATE, res=a1, gate=mod[:, _ST], out2=t)  # out = a1 + st * (h Ws^T)
    out = _unpadded(out.view(a, 1, n, DS_), single.shape)
    if not save:
        return [out]
    assert y is not None and u is not None and t is not None  # allocated above: `save` is true here
    saved = [out, mod, cn1, cn2, x1, p, bias, o, lse, gated, y, a1, x2, u, t]
    if not cross:
        return saved
    assert cnkv is not None and mkv is not None and xkv is not None  # the cross mode built all three
    return [*saved, cnkv, mkv, xkv]


def _gated_bwd(dyo, woT, g, o, dg, a, n):
    """dgated = dyo Wo, then the output gate's backward -> (dO, D): one launch (round 3 part 13, ``gemm_gate_bwd32``: dgated stays in TMEM;
    MW_LD13_GB=0: gemm32 + gate_bwd32, A / B)."""
    import os

    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom as rows

    if os.environ.get("MW_LD13_GB", "1") != "0":
        return rows.gemm_gate_bwd32(dyo, woT, g, o, dg, a, n)
    return rows.gate_bwd32(rows.gemm32(dyo, woT), g, o, dg, a, n)


def _mm32(dout, x):
    """dout^T x (a weight gradient), fp32 on TF32 tensor cores (the caller holds ``_tf32()``)."""
    return torch.mm(dout.t(), x)


@opaque(fake=_bwd_fake, name="local_dit_b200_bwd_tf32")
def _block_bwd32(
    dy: torch.Tensor,
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    weights: list[torch.Tensor],
    saved: list[torch.Tensor],
    parallel: bool = False,
    slide: bool = False,
    qblock: bool = False,
) -> list[torch.Tensor]:
    """The fp32 path's backward: d single, d cond, d pair, then one gradient per weight (``WEIGHTS`` order, the weight's dtype)."""
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom as rows
    from miniworld_engine.kernels.augmented_attention.cuda import (
        b200_atom_local as local,
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
        rc = cond.shape[0] * n  # the tables' rows: n when the samples share the conditioning
        c2 = _padded(cond, n).reshape(rc, DC_).contiguous()
        dy = _padded(dy, n).reshape(m, DS_).float().contiguous()
        dmod = torch.empty(
            rc, 6 * DS_, device=dev, dtype=F32
        )  # d [s1 | bi1 | s2 | bi2 | so | st] (pre-sigmoid for the gates; summed over the samples in the row kernels when shared)
        names = WEIGHTS + CROSS_WEIGHTS[: len(weights) - len(WEIGHTS)]
        wmap = dict(zip(names, weights, strict=True))

        def own(g, name):  # a vector gradient as its weight's own fresh tensor (shape, dtype)
            w = wmap[name]
            return g.to(w.dtype).reshape(w.shape).clone()

        dt = rows.tail_bwd32(dy, t, mod[:, _ST], dmod[:, _ST])  # out = a1 + st t
        dab, hh = rows.dh_swiglu_bwd32(dt, pk.wsT, u)  # dh = dt Ws (in TMEM) -> the SwiGLU backward (round 3 part 11: one launch)
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
        dp = torch.empty(m, 4 * DS_, device=dev, dtype=F32)  # d P32 (same column layout)
        do, dd = _gated_bwd(dyo, pk.woT, p[:, gc : gc + DS_], o.view(m, DS_), dp[:, gc : gc + DS_], a, n)
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
            slide=slide,
            qblock=qblock,
        )

        # the pair bias backward (it reads db, the pair and two weights only) beside the row kernels below: side stream 1
        def pair_side():  # -> d pair, {the two pair weights' gradients}: their copies run on the side stream too
            dz, dgamma, dwb = local.pair_bias_bwd32(_trunk(pair, n), weights[_GAMMA], weights[_WBIAS], db)
            return dz[: pair.shape[1]].reshape(pair.shape).clone(), {
                "attention.ln_pair.weight": own(dgamma, "attention.ln_pair.weight"),
                "attention.to_bias.weight": own(dwb, "attention.to_bias.weight"),
            }

        (dpair, pair_grads), pb_done = _fork(pair_side, 1)
        dmkv = None
        if cross:  # x1 -> q | g directly, -> k | v through the K / V AdaLN
            cnkv, mkv, xkv = extra
            dx1q = rows.gemm32(dp, pk.wpT, k=2 * DS_)
            dxkv = rows.gemm32(dp, pk.wpT, k=2 * DS_, a_col0=2 * DS_, b_col0=2 * DS_)
            dmkv = torch.empty(rc, 2 * DS_, device=dev, dtype=F32)
            if parallel:  # the K / V AdaLN of s: its LN backward joins the residual's gradient
                dres, _ = rows.adaln_bwd32(s2, mkv[:, :DS_], dxkv, da1, dmkv[:, :DS_], dmkv[:, DS_:])
                dx1 = dx1q
            else:
                dx1, _ = rows.adaln_bwd32(x1, mkv[:, :DS_], dxkv, dx1q, dmkv[:, :DS_], dmkv[:, DS_:])
                dres = da1
        else:
            dx1 = rows.gemm32(dp, pk.wpT)  # K = 512
            dres = da1
        ds, _ = rows.adaln_bwd32(s2, mod[:, _S1], dx1, dres, dmod[:, _S1], dmod[:, _B1])  # AdaLN 1 + the residual

        def colsums():  # the bias gradients: column sums of the finished dmod / dp / dmkv (memory-bound; side stream 0, beside the products)
            dcols = rows.colsum32(dmod)
            out = {
                "attention.ada_ln_in.to_scale.bias": dcols[0:128],
                "attention.to_query.bias": rows.colsum32(dp[:, qc : qc + DS_]),
                "attention.to_scale.bias": dcols[512:640],
                "transition.ada_ln_in.to_scale.bias": dcols[256:384],
                "transition.to_scale.bias": dcols[640:768],
            }
            if dmkv is not None:
                out["attention.ada_ln_kv.to_scale.bias"] = rows.colsum32(dmkv[:, :DS_])
            return {name: own(g, name) for name, g in out.items()}

        col_grads, cols_done = _fork(colsums, 0)
        dc = rows.gemm32(dmod, pk.wmodT, k=2 * DS_, a_col0=4 * DS_, b_col0=4 * DS_)  # so, st <- c
        dcn1 = rows.gemm32(dmod, pk.wmodT, k=2 * DS_)
        dcn2 = rows.gemm32(dmod, pk.wmodT, k=2 * DS_, a_col0=2 * DS_, b_col0=2 * DS_)
        dcn3 = rows.gemm32(dmkv, pk.wmT) if cross else None
        dgc = rows.ln_bwd32(c2, dc, dcn1, pk.g1, dcn2, pk.g2, dcn3, pk.gkv)  # dc += the LayerNorm backwards
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
        if rc != m:  # shared conditioning: the tables' products have n rows, their own launch
            cond_names = {
                "attention.ada_ln_in.to_scale.weight",
                "transition.ada_ln_in.to_scale.weight",
                "attention.to_scale.weight",
                "attention.ada_ln_kv.to_scale.weight",
            }
            cspec = [e for e in spec if e[2][0][0] in cond_names]
            fused = _fused_wgrads([e for e in spec if e[2][0][0] not in cond_names], weights, names, True, dev, m)
            fused.update(_fused_wgrads(cspec, weights, names, True, dev, rc))
        else:
            fused = _fused_wgrads(spec, weights, names, True, dev, m)
        _join(cols_done, pb_done)
    grads: dict[str, torch.Tensor] = {
        **col_grads,
        **pair_grads,
        "attention.ada_ln_in.ln_cond.weight": own(dgc[0:128], "attention.ada_ln_in.ln_cond.weight"),
        "transition.ada_ln_in.ln_cond.weight": own(dgc[128:256], "transition.ada_ln_in.ln_cond.weight"),
    }
    if cross:
        grads["attention.ada_ln_kv.ln_cond.weight"] = own(dgc[256:384], "attention.ada_ln_kv.ln_cond.weight")
    per_weight = [fused[name] if name in fused else grads[name] for name in names]
    return [
        _unpadded(ds.view(a, 1, n, DS_), single.shape),
        _unpadded(dc.view(cond.shape[0], 1, n, DC_), cond.shape),
        dpair,
        *per_weight,
    ]


class _LocalBlock32(torch.autograd.Function):
    """The fp32 path's block forward and backward (TF32 kernels); gradients for single, cond, pair and every weight."""

    @staticmethod
    def forward(ctx, parallel, window, mask, single, cond, pair, *weights):  # window: 0 clipped, 1 slid, 2 slid by query blocks (qblock)
        out, *saved = _block_fwd32(single, cond, pair, mask, list(weights), True, parallel, window > 0, window == 2)
        ctx.save_for_backward(mask, single, cond, pair, *saved, *weights)
        ctx.n_saved = len(saved)
        ctx.parallel, ctx.window = parallel, window
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        mask, single, cond, pair, *rest = ctx.saved_tensors
        saved, weights = rest[: ctx.n_saved], rest[ctx.n_saved :]
        grads = _block_bwd32(dy, single, cond, pair, mask, list(weights), list(saved), ctx.parallel, ctx.window > 0, ctx.window == 2)
        return (None, None, None, *grads)


def _infer32(single, cond, pair, mask, weights, parallel=False, window=0):
    return _block_fwd32(single, cond, pair, mask, list(weights), False, parallel, window > 0, window == 2)[0]


# ============================================================================================================ the module's entry points
def reason(module, single, cond, pair, mask=None, uid=None) -> str | None:
    """None when :func:`run_block` serves the call on the B200, else why not (the caller then keeps its other B200 route). The block-local
    atom block with the K / V AdaLN chained (Protenix) or parallel (AlphaFold 3 / OpenFold3), the clipped or slid key window, the key mask
    form; d 128 / cond 128 / pair 16, 4 heads x 32, transition n = 2, every LayerNorm eps 1e-5; one fold (``single`` [A, 1, N, 128])."""
    from miniworld_engine.kernels.integrations import local_dit_h100

    if not (single.is_cuda and torch.cuda.get_device_capability(single.device) == (10, 0)):
        return "the tcgen05 local atom block runs on a B200 (sm_100a) CUDA tensor only"
    if module.kv_norm == "shared":
        return _variant_reason(module, single, cond, pair, mask, uid)
    if uid is not None:
        return "conformer ids (Chai-1's uid) are not an input of the tcgen05 local atom block"
    if module.kv_norm not in ("chained", "parallel") or module.parallel:
        return f"K / V AdaLN {module.kv_norm!r} / parallel schedule (the tcgen05 block has the chained or parallel K / V AdaLN only)"
    if module.key_window not in ("pad", "slide", "slide_qblock"):
        return f"key window {module.key_window!r} (the tcgen05 block has the clipped, AlphaFold 3's slid and IntelliFold2's slide_qblock window)"
    # (AlphaFold 3's mask form, AtomDiTOpenFold3 mask_form="af3": the class widens the key mask before it asks -- still a key mask)
    opts = local_dit_h100.options(module)
    if opts is None or opts[1:] != (False, False, False, False, False, 1e-5, 1e-5):
        return (
            "block options outside the tcgen05 block (identity AdaLN, parallel schedule, no out projection, up-gate, q / k LayerNorm, eps)"
        )
    att, tr = module.attention, module.transition
    norms = [att.ada_ln_in, tr.ada_ln_in, att.ada_ln_kv]
    if any(x.ln_cond.eps != EPS or x.ln_in.eps != EPS for x in norms) or att.ln_pair.eps != EPS:
        return "a LayerNorm eps other than 1e-5"
    geometry = (
        att.n_head,
        att.to_query.weight.shape[0],
        att.to_query.weight.shape[1],
        att.to_bias.weight.shape[1],
        tr.expand_a.weight.shape[0],
        tr.d_cond,
    )
    if geometry != (NH_, DS_, DS_, DP_, 2 * DS_, DC_):
        return "geometry: the kernels are built for 128 / cond 128 / pair 16, 4 heads x 32, transition n = 2"
    if att.to_query.bias is None or any(not p.is_contiguous() for p in module.parameters()):
        return "a query projection without bias or a non-contiguous parameter"
    return _io_reason(module, single, cond, pair, mask)


def _io_reason(module, single, cond, pair, mask) -> str | None:
    """The dtype / shape part of :func:`reason` (the block's inputs and parameters)."""
    params = list(module.parameters())
    dts = (single.dtype, cond.dtype, pair.dtype)
    if dts == (F32, F32, F32):
        if torch.is_autocast_enabled("cuda"):
            return "float32 activations under autocast"
        if any(p.dtype != F32 for p in params):
            return "float32 activations need float32 parameters (the TF32 kernels)"
    elif dts == (BF, BF, BF):
        if any(p.dtype not in (BF, F32) for p in params):
            return "bfloat16 activations take bfloat16 or float32 parameters"
        if torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") != BF:
            return "an autocast dtype other than bfloat16"
    else:
        return f"dtypes {tuple(str(t) for t in dts)} (single / cond / pair all bfloat16 or all float32)"
    if single.ndim != 4 or single.shape[3] != DS_ or single.shape[2] == 0 or single.shape[1] == 0:
        return f"single must be [A, F, N, 128], got {tuple(single.shape)}"
    a, nf, n, _ = single.shape
    if nf > 1:  # B > 1 folds (trunk_seeds of one input, each its own pair): one inference call, fold f's pair bias for its rows
        if dts != (BF, BF, BF):
            return "F > 1 folds: bfloat16 activations only"
        if torch.is_grad_enabled() and any(t.requires_grad for t in (single, cond, pair, *params)):
            return "F > 1 folds: inference only"
    if cond.ndim != 4 or cond.shape[1:] != (nf, n, DC_) or cond.shape[0] not in (1, a):
        return f"cond must be [A, F, N, 128] (or [1, F, N, 128]), got {tuple(cond.shape)}"
    if tuple(pair.shape) != (nf, (n + QUERIES - 1) // QUERIES, QUERIES, KEYS, DP_):
        return f"pair must be [{nf}, {(n + QUERIES - 1) // QUERIES}, 32, 128, 16], got {tuple(pair.shape)}"
    if mask is not None and (
        mask.dtype != torch.bool
        or mask.ndim != 2
        or mask.shape[1] != n
        or mask.shape[0] not in (1, nf)
        or (mask.shape[0] > 1 and mask.stride(0) != 0)
        or mask.device != single.device
    ):
        return "mask must be a bool [1, N] key mask (F > 1 folds: the folds' one mask, [1, N] or expanded to [F, N])"
    return None


# The variant blocks (2026-11-17, B200 round 3): Boltz-2 / RoseTTAFold3 / Chai-1 on the same tcgen05 kernels, their differences as builds
# of them (``b200_atom.KERNELS_VAR``) and leaves (``local_dit_h100._leaf``: the parts a model lacks are constant leaves -- no gating query:
# gate weight 0, sigmoid(0) = 0.5 exactly; no out projection: Wo = 2 I; no query bias: 0; identity AdaLN: scale bias 1), as the H100's
# ``local_dit_h100.options``. Served so far: bf16 activations (bf16 / fp32 master parameters), inference.
def variant_options(module):
    """(ident, pars, up, kqln, kq_eps, eps) of a one-AdaLN variant block, or None when it is outside the B200 kernels."""
    from miniworld_engine.kernels.integrations import local_dit_h100

    opts = local_dit_h100.options(module)
    if opts is None or opts[0] != 2:
        return None
    _, ident, pars, _noout, up, kqln, kq_eps, eps = opts
    if pars and up:
        return None
    return ident, pars, up, kqln, kq_eps, eps


def _variant_reason(module, single, cond, pair, mask, uid) -> str | None:
    if variant_options(module) is None:
        return "block options outside the tcgen05 variant builds (local_dit_h100.options; the parallel schedule with the up-gate)"
    if module.key_window not in ("pad", "slide"):
        return f"key window {module.key_window!r} (the tcgen05 block has the clipped and AlphaFold 3's slid window)"
    att, tr = module.attention, module.transition
    if att.ln_pair.eps != EPS:
        return "a pair LayerNorm eps other than 1e-5"
    geometry = (att.n_head, att.to_query.weight.shape[0], att.to_query.weight.shape[1], att.to_bias.weight.shape[1], tr.expand_a.weight.shape[0], tr.d_cond)
    if geometry != (NH_, DS_, DS_, DP_, 2 * DS_, DC_):
        return "geometry: the kernels are built for 128 / cond 128 / pair 16, 4 heads x 32, transition n = 2"
    if any(not p.is_contiguous() for p in module.parameters()):
        return "a non-contiguous parameter"
    if uid is not None and (uid.ndim != 2 or tuple(uid.shape) != (1, single.shape[2]) or uid.device != single.device):
        return f"uid must be [1, N], got {tuple(uid.shape)}"
    return _io_reason(module, single, cond, pair, mask)


def _variant_weights(module):
    """The 23 canonical leaves (``local_dit_h100._leaf``), then the option leaves (KQLN: wq bq wk bk; UP: Wu)."""
    from miniworld_engine.kernels.integrations import local_dit_h100

    w = local_dit_h100._weights_of(module)
    return w[: len(WEIGHTS)] + w[len(WEIGHTS) + len(CROSS_WEIGHTS) :]


def run_block(module, single, cond, pair, mask=None, uid=None):
    """The block's output stream (both residuals included) on the tcgen05 kernels; :func:`reason` must have accepted the call. The caller
    runs it with autocast off (the kernels' precision is their own)."""
    if module.kv_norm == "shared":  # the variant blocks (``_variant_reason``: bf16 inference)
        return _run_variant(module, single, cond, pair, mask, uid)
    weights = [module.get_parameter(name) for name in (*WEIGHTS, *CROSS_WEIGHTS)]
    parallel, window = module.kv_norm == "parallel", {"pad": 0, "slide": 1, "slide_qblock": 2}[module.key_window]
    index = single.device.index if single.device.index is not None else torch.cuda.current_device()
    single, pair = single.contiguous(), pair.contiguous()
    mask = None if mask is None else mask[:1].contiguous()  # (F > 1 folds: their one mask)
    cond = _shared_cond(single, cond)  # one conditioning expanded over the samples: its tables once, their gradient summed in the kernels
    if cond.shape[0] != single.shape[0] or not cond.is_contiguous():
        cond = cond.contiguous() if cond.shape[0] == single.shape[0] else cond[:1].contiguous()
    train = torch.is_grad_enabled() and any(t.requires_grad for t in (single, cond, pair, *weights))
    if single.dtype == F32:  # the fp32 path (TF32 tensor cores)
        _kernels32_ready(index)
        if train:
            return _LocalBlock32.apply(parallel, window, mask, single, cond, pair, *weights)
        return _infer32(single, cond, pair, mask, weights, parallel, window)
    _kernels_ready(index)
    if train:
        return _LocalBlock.apply(parallel, window, mask, single, cond, pair, *weights)
    return _infer(single, cond, pair, mask, weights, parallel, window)


def _run_variant(module, single, cond, pair, mask, uid):
    opts = variant_options(module)
    assert opts is not None  # (``_variant_reason`` accepted the call)
    ident, pars, up, kqln, kq_eps, eps = opts
    weights = _variant_weights(module)
    index = single.device.index if single.device.index is not None else torch.cuda.current_device()
    single, pair = single.contiguous(), pair.contiguous()
    mask = None if mask is None else mask.contiguous()
    cond = _shared_cond(single, cond)
    if cond.shape[0] != single.shape[0] or not cond.is_contiguous():
        cond = cond.contiguous() if cond.shape[0] == single.shape[0] else cond[:1].contiguous()
    n0 = single.shape[2]
    n = _padded_length(n0)
    u = None
    if uid is not None:
        u = (uid[0] if n == n0 else F.pad(uid[0], (0, n - n0), value=-1)).long().contiguous()
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom as rows

    slide = module.key_window == "slide"
    if single.dtype == F32:  # the fp32 path (TF32 tensor cores), inference
        _kernels32_ready(index)
        for name in rows.KERNELS_FUSED32_VAR:
            rows.fused_kernel32(name, index)
        if torch.is_grad_enabled() and any(t.requires_grad for t in (single, cond, pair, *weights)):
            return _VarBlock32.apply((slide, ident, pars, up, kqln, kq_eps, eps), mask, u, single, cond, pair, *weights)
        return _var_fwd32(single, cond, pair, mask, weights, slide, ident, pars, up, kqln, kq_eps, eps, u)[0]
    _kernels_ready(index)
    for name in rows.KERNELS_VAR:
        rows.atom_kernel(name, index)
    if torch.is_grad_enabled() and any(t.requires_grad for t in (single, cond, pair, *weights)):
        return _VarBlock.apply((slide, ident, pars, up, kqln, kq_eps, eps), mask, u, single, cond, pair, *weights)
    return _block_fwd(
        single, cond, pair, mask, weights, False, False, slide, False, True, ident, pars, up, kqln, kq_eps, eps, u
    )[0]


# ------------------------------------------------------------------------------------------------ the variant blocks' fp32 path (inference)
def _var_pack32(weights, kqln, up):
    """The fp32 path's forward pack of a variant block (``_pack32`` of its 23 canonical leaves) plus its option operands: kq fp32 [4, 128]
    (KQLN) and, for UP, ``wu_up`` = [Wa_q; Wb_q; Wu_q] for the four hidden quarters q (64 rows each; [768, 128], tf32-rounded), the
    operand of ``atom_post_tf32``'s ``-DUP=1`` build. Cached like the other fp32 packs."""

    def build(ws):
        pk = _pack32(list(ws[: len(WEIGHTS)]))
        extra = list(ws[len(WEIGHTS) :])
        with torch.no_grad():
            pk.kq = torch.stack([w.detach().float() for w in extra[:4]]).contiguous() if kqln else None
            pk.wu_up = None
            if up:
                w = dict(zip(WEIGHTS, ws[: len(WEIGHTS)], strict=True))
                wa, wb = w["transition.expand_a.weight"].detach().float(), w["transition.expand_b.weight"].detach().float()
                wu = extra[4 if kqln else 0].detach().float()
                q4 = torch.cat([torch.cat([wa[64 * q : 64 * q + 64], wb[64 * q : 64 * q + 64], wu[64 * q : 64 * q + 64]]) for q in range(4)])
                pk.wu_up = tf32_grid(q4).contiguous()
        return pk

    build.__name__ = f"_var_pack32_{int(bool(kqln))}{int(bool(up))}"
    return _cached32(build, list(weights), eager=not torch.is_grad_enabled())


def _var_fwd32_fake(single, cond, pair, mask, weights, slide=False, ident=False, pars=False, up=False, kqln=False, kq_eps=1e-5, eps=1e-5, uid=None):
    return [torch.empty_like(single)]


@opaque(fake=_var_fwd32_fake, name="local_dit_b200_var_fwd32")
def _var_fwd32(
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    weights: list[torch.Tensor],
    slide: bool = False,
    ident: bool = False,
    pars: bool = False,
    up: bool = False,
    kqln: bool = False,
    kq_eps: float = 1e-5,
    eps: float = 1e-5,
    uid: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """The variant block's fp32 inference on the fused TF32 kernels (``_block_fwd32``'s inference with the variant builds): conditioning
    tables (IDENT: one GEMM of the raw c, the scales raw), ``atom_pre_tf32`` (KQLN), the attention, ``atom_post_tf32`` (PARS / UP)."""
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom as rows
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom_local as local

    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    m = a * n
    dev = single.device
    with torch.cuda.device(dev):
        # the pair bias (+ conformer mask) on a side stream, beside the tables and the input projections, as ``_block_fwd``; without the
        # conformer mask (round 3 part 8): made ready here and launched beside atom_pre_tf32 as its programmatic dependent (_EARLY_BIAS)
        launch = join = None
        if uid is None and _early_bias(m, cond.shape[0] * n):
            bias, launch = _pair_bias_deferred(pair, n, weights)
        else:
            bias, join = _fork_bias(
                dev,
                (lambda: _pair_bias_conf(pair, n, weights, False, uid, mask, n0, slide, f32=True))
                if uid is not None
                else (lambda: _pair_bias32(pair, n, weights, False)),
            )
        pk = _var_pack32(weights, kqln, up)
        s2 = _padded(single, n).reshape(m, DS_).contiguous()

        def tables():
            c2 = _padded(cond, n).reshape(cond.shape[0] * n, DC_).contiguous()
            if ident:  # Chai-1: every table from the raw c, only the output gates (so, st: blocks 4, 5) as sigmoids
                return rows.gemm32(c2, pk.wcond, bias=pk.bcond, sigmask=0b110000)
            cat = torch.empty(c2.shape[0], 3 * DS_, device=dev, dtype=F32)
            rows.ln32(c2, pk.g1, pk.g2, None, out=cat)
            return rows.gemm32(cat, pk.wcond, k=DS_, a_grp=2 * DS_, bias=pk.bcond, sigmask=_SIG_MOD)

        mod = (
            _hoisted(_HOIST_COND, cond, ("tf32var", tuple((w.data_ptr(), w._version) for w in weights), n, ident), tables)
            if _hoisting()
            else tables()
        )
        p = rows.pre32(s2, mod[:, _S1], mod[:, _B1], pk.wp, pk.bp[:DS_], _P_RND[False], eps=eps, kq=pk.kq, kq_eps=kq_eps, samples=a)
        if launch is not None:
            launch()
        if join is not None:
            torch.cuda.current_stream().wait_event(join)
        p3 = p.view(a, n, 4 * DS_)
        qc, kc, vc, gc = _P_COLS[False]
        o, _ = local.attn_fwd32(p3[..., qc : qc + DS_], p3[..., kc : kc + DS_], p3[..., vc : vc + DS_], bias, _keys(mask, n0, n), slide=slide)
        out = rows.post32(
            o.view(m, DS_),
            p[:, gc : gc + DS_],
            s2,
            mod[:, _SO],
            mod[:, _S2],
            mod[:, _B2],
            mod[:, _ST],
            pk.wo,
            pk.wu_up if up else pk.wu,
            pk.ws,
            eps=eps,
            pars=pars,
            up=up,
        )
    return [_unpadded(out.view(a, 1, n, DS_), single.shape)]


# ------------------------------------------------------------------------------------------------ the variant blocks' fp32 path (training)
# The composed fp32 block (``_block_fwd32`` / ``_block_bwd32``'s stages: TF32 GEMMs between fp32 row kernels) with the variants' options:
#   IDENT  the tables one GEMM of the raw c (cn1 = cn2 = c), the AdaLN backwards with a raw scale (``adaln_bwd32_id``), dc one K = 768 GEMM
#   PARS   AdaLN 2 normalises the block input; the attention branch's gate reads dy alone (``tail_bwd32`` on y, so)
#   KQLN   the flat q / k LayerNorm on the projection, in place (``kqln_fwd32`` / ``kqln_bwd32``)
#   UP     the up factor one more GEMM of x2, h = silu(a) b * up (its three products elementwise)
def _var_pack32_bwd(weights, kqln, up):
    """The fp32 training packs of a variant block in ONE launch (``b200_atom.var_pack32``; the torch pack -- ``_pack32``, its transposes --
    recorded ~12 launches into every captured step): ``_pack32``'s forward operands (wmod = wcond, bmod = bcond, g1, g2, wp, bp, wo, wu,
    ws; all on the TF32 grid), kq fp32 [4, 128] (KQLN), the backward's transposes (``_pack32_bwd``: wmodT, wpT, woT, wuT, wsT), the
    conditioning backward's grouped operand ``wgrpT`` [384, 256] and, for UP, the plain up-gate weight ``wup`` [256, 128] and ``wupT``."""
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom as rows

    def build(ws):
        w = dict(zip(WEIGHTS, ws[: len(WEIGHTS)], strict=True))
        extra = list(ws[len(WEIGHTS) :])
        a1, a2, at, tr = "attention.ada_ln_in.", "transition.ada_ln_in.", "attention.", "transition."
        mats = [w[a1 + "to_scale.weight"], w[a1 + "to_bias.weight"], w[a2 + "to_scale.weight"], w[a2 + "to_bias.weight"], w[at + "to_scale.weight"],
                w[tr + "to_scale.weight"], w[at + "to_query.weight"], w[at + "to_key.weight"], w[at + "to_value.weight"], w[at + "to_gate.weight"],
                w[at + "to_out.weight"], w[tr + "expand_a.weight"], w[tr + "expand_b.weight"], w[tr + "squeeze.weight"]]
        if up:
            mats.append(extra[4 if kqln else 0])
        vecs = [w[a1 + "to_scale.bias"], None, w[a2 + "to_scale.bias"], None, w[at + "to_scale.bias"], w[tr + "to_scale.bias"],
                w[at + "to_query.bias"], None, None, None, w[a1 + "ln_cond.weight"], w[a2 + "ln_cond.weight"], *(extra[:4] if kqln else [])]
        with torch.no_grad():
            bf, vec, bt, bg, bu = rows.var_pack32([t.detach() for t in mats], [None if t is None else t.detach() for t in vecs])

        def mat(buf, row0, nrow, shape):
            return buf[row0 * DS_ : (row0 + nrow) * DS_].view(shape)

        wmod, bmod = mat(bf, 0, 768, (6 * DS_, DS_)), vec[0:6].reshape(-1)
        return SimpleNamespace(
            wmod=wmod, wcond=wmod, bmod=bmod, bcond=bmod, bp=vec[6:10].reshape(-1), g1=vec[10], g2=vec[11], gkv=None, flat=bf,
            kq=vec[12:16] if kqln else None,
            wp=mat(bf, 768, 512, (4 * DS_, DS_)), wo=mat(bf, 1280, 128, (DS_, DS_)), wu=mat(bf, 1408, 512, (4 * DS_, DS_)),
            ws=mat(bf, 1920, 256, (DS_, 2 * DS_)), wup=mat(bf, 2176, 256, (2 * DS_, DS_)) if up else None, wu_up=None,
            wsT=mat(bt, 0, 256, (2 * DS_, DS_)), wuT=mat(bt, 256, 512, (DS_, 4 * DS_)), woT=mat(bt, 768, 128, (DS_, DS_)),
            wpT=mat(bt, 896, 512, (DS_, 4 * DS_)), wmodT=mat(bt, 1408, 768, (DS_, 6 * DS_)),
            wupT=mat(bt, 2176, 256, (DS_, 2 * DS_)) if up else None, wgrpT=bg, wuupT=bu,
        )

    build.__name__ = f"_var_pack32_bwd_{int(bool(kqln))}{int(bool(up))}"
    return _cached32(build, list(weights))


def _var_train_fwd32_fake(single, cond, pair, mask, weights, slide=False, ident=False, pars=False, up=False, kqln=False, kq_eps=1e-5, eps=1e-5, uid=None):
    saved = _fwd_fake32(single, cond, pair, mask, weights[: len(WEIGHTS)], True)
    m = single.shape[0] * _padded_length(single.shape[2])
    rc = cond.shape[0] * _padded_length(single.shape[2])
    saved[2], saved[3] = single.new_empty((0,) if ident else (rc, 3 * DS_)), single.new_empty((0,))  # [cn1, cn2, c] in one tensor
    return [*saved, *([single.new_empty((m, 2 * DS_))] if kqln else []), *([single.new_empty((m, 2 * DS_))] if up else [])]


@opaque(fake=_var_train_fwd32_fake, name="local_dit_b200_var_train_fwd32")
def _var_train_fwd32(
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    weights: list[torch.Tensor],
    slide: bool = False,
    ident: bool = False,
    pars: bool = False,
    up: bool = False,
    kqln: bool = False,
    kq_eps: float = 1e-5,
    eps: float = 1e-5,
    uid: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """The variant block's fp32 training forward: the output and the activations ``_var_bwd32`` reads (``_block_fwd32``'s list: mod32, cn1,
    cn2 (here: [cn1, cn2, c] as ONE [rows, 384] tensor and an empty one; both empty with IDENT), x1, P32, bias, O, LSE, gated, y, a1, x2, u, t; then the raw q | k (KQLN) and the up factor (UP))."""
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom as rows
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom_local as local

    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    m = a * n
    dev = single.device
    with torch.cuda.device(dev):
        pk = _var_pack32_bwd(weights, kqln, up)
        canon = list(weights[: len(WEIGHTS)])
        bias, join = _fork_bias(  # the pair bias (+ conformer mask) on a side stream
            dev,
            (lambda: _pair_bias_conf(pair, n, canon, True, uid, mask, n0, slide, f32=True)) if uid is not None else (lambda: _pair_bias32(pair, n, canon, True)),
        )
        qc, kc, vc, gc = _P_COLS[False]
        s2 = _padded(single, n).reshape(m, DS_).contiguous()
        c2 = _padded(cond, n).reshape(cond.shape[0] * n, DC_).contiguous()
        if ident:  # every table from the raw c, only the output gates (so, st: blocks 4, 5) as sigmoids
            mod = rows.gemm32(c2, pk.wcond, bias=pk.bcond, sigmask=0b110000)
            cn1, cn2 = c2.new_empty(0), c2.new_empty(0)  # (the weight gradients read c itself: an op output may not alias an input)
        else:  # cat = [cn1, cn2, c] (one LayerNorm launch), then ONE grouped GEMM (``_cond_tables32``'s inference form); cat is saved
            cn1 = torch.empty(c2.shape[0], 3 * DS_, device=dev, dtype=F32)
            rows.ln32(c2, pk.g1, pk.g2, None, out=cn1)
            mod = rows.gemm32(cn1, pk.wcond, k=DS_, a_grp=2 * DS_, bias=pk.bcond, sigmask=_SIG_MOD)
            cn2 = c2.new_empty(0)
        x1 = rows.adaln32(s2, mod[:, _S1], mod[:, _B1], eps=eps)  # AdaLN 1
        p = torch.empty(m, 4 * DS_, device=dev, dtype=F32)
        rows.gemm32(x1, pk.wp, p, bias=pk.bp, rndmask=0b0100 if kqln else _P_RND[False])  # (KQLN: q / k are rounded after their LayerNorm)
        qkr = rows.kqln_fwd32(p, pk.kq, kq_eps) if kqln else None
        torch.cuda.current_stream().wait_event(join)
        p3 = p.view(a, n, 4 * DS_)
        o, lse = local.attn_fwd32(p3[..., qc : qc + DS_], p3[..., kc : kc + DS_], p3[..., vc : vc + DS_], bias, _keys(mask, n0, n), slide=slide)
        gated = rows.gate32(p[:, gc : gc + DS_], o.view(m, DS_))
        y = torch.empty(m, DS_, device=dev, dtype=F32)
        a1 = rows.gemm32(gated, pk.wo, mode=rows.RESID_GATE, res=s2, gate=mod[:, _SO], out2=y)  # a1 = s + so * (gated Wo^T)
        x2 = rows.adaln32(s2 if pars else a1, mod[:, _S2], mod[:, _B2], eps=eps)  # AdaLN 2 (PARS: of the block input)
        u = torch.empty(m, 4 * DS_, device=dev, dtype=F32)
        h = rows.gemm32(x2, pk.wu, mode=rows.SWIGLU, out2=u)  # silu(x2 Wa^T) * (x2 Wb^T)
        upf = None
        if up:
            upf = rows.gemm32(x2, pk.wup)  # [m, 256]
            h = h * upf
        t = torch.empty(m, DS_, device=dev, dtype=F32)
        out = rows.gemm32(h, pk.ws, mode=rows.RESID_GATE, res=a1, gate=mod[:, _ST], out2=t)  # out = a1 + st * (h Ws^T)
    out = _unpadded(out.view(a, 1, n, DS_), single.shape)
    saved = [out, mod, cn1, cn2, x1, p, bias, o, lse, gated, y, a1, x2, u, t]
    if kqln:
        assert qkr is not None
        saved.append(qkr)
    if up:
        assert upf is not None
        saved.append(upf)
    return saved


@opaque(fake=_bwd_fake, name="local_dit_b200_var_bwd32")
def _var_bwd32(
    dy: torch.Tensor,
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    weights: list[torch.Tensor],
    saved: list[torch.Tensor],
    slide: bool = False,
    ident: bool = False,
    pars: bool = False,
    up: bool = False,
    kqln: bool = False,
    kq_eps: float = 1e-5,
    eps: float = 1e-5,
) -> list[torch.Tensor]:
    """The variant block's fp32 backward: d single, d cond, d pair, then one gradient per leaf of ``weights`` (the 23 canonical ones, then
    KQLN's wq bq wk bk and UP's Wu), in the leaf's dtype."""
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom as rows
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom_local as local

    mod, cn1, cn2, x1, p, bias, o, lse, gated, y, a1, x2, u, t, *extra = saved
    qkr = extra[0] if kqln else None
    upf = extra[-1] if up else None
    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    m = a * n
    dev = single.device
    canon, opt_leaves = list(weights[: len(WEIGHTS)]), list(weights[len(WEIGHTS) :])
    adaln_b = (lambda *args: (rows.adaln_bwd32_id(*args, eps=eps), None)) if ident else (lambda *args: rows.adaln_bwd32(*args, eps=eps))
    with torch.cuda.device(dev):
        pk = _var_pack32_bwd(weights, kqln, up)
        qc, kc, vc, gc = _P_COLS[False]
        s2 = _padded(single, n).reshape(m, DS_).contiguous()
        rc = cond.shape[0] * n  # the tables' rows: n when the samples share the conditioning
        c2 = _padded(cond, n).reshape(rc, DC_).contiguous()
        dy = _padded(dy, n).reshape(m, DS_).float().contiguous()
        dmod = torch.empty(rc, 6 * DS_, device=dev, dtype=F32)  # d [s1 | bi1 | s2 | bi2 | so | st] (pre-sigmoid for the gates)
        if ident:  # the tables are products of the raw c: it is the weight gradients' operand
            cn1 = cn2 = c2
        else:  # (saved as one [rows, 384] tensor [cn1, cn2, c])
            cn1, cn2 = cn1[:, :DS_], cn1[:, DS_ : 2 * DS_]
        dt = rows.tail_bwd32(dy, t, mod[:, _ST], dmod[:, _ST])  # out = a1 + st t
        dh = rows.gemm32(dt, pk.wsT)  # [m, 256]
        dup = None
        if up:  # h = silu(a) b up: [d a, d b, d up] in one launch, dx2 one K = 768 GEMM against [Wu^T, Wup^T]
            assert upf is not None
            dabu, hh = rows.swiglu_up_bwd32(dh, u, upf)
            dab, dup = dabu[:, : 4 * DS_], dabu[:, 4 * DS_ :]
            dx2 = rows.gemm32(dabu, pk.wuupT)
        else:
            dab, hh = rows.swiglu_bwd32(dh, u)
            dx2 = rows.gemm32(dab, pk.wuT)  # K = 512
        if pars:  # AdaLN 2 normalised the block input: its LN backward joins the residual's gradient there; the gate reads dy
            da1, _ = adaln_b(s2, mod[:, _S2], dx2, dy, dmod[:, _S2], dmod[:, _B2])
            dyo = rows.tail_bwd32(dy, y, mod[:, _SO], dmod[:, _SO])
        else:
            da1, dyo = rows.adaln_bwd32(a1, mod[:, _S2], dx2, dy, dmod[:, _S2], dmod[:, _B2], y=y, so=mod[:, _SO], dso=dmod[:, _SO], eps=eps)
        dp = torch.empty(m, 4 * DS_, device=dev, dtype=F32)  # d P32 (same column layout)
        do, dd = _gated_bwd(dyo, pk.woT, p[:, gc : gc + DS_], o.view(m, DS_), dp[:, gc : gc + DS_], a, n)
        p3 = p.view(a, n, 4 * DS_)
        db = local.attn_bwd32(
            p3[..., qc : qc + DS_], p3[..., kc : kc + DS_], p3[..., vc : vc + DS_], do.view(a, n, DS_), bias, lse, dd, dp, (qc, kc, vc),
            _keys(mask, n0, n), slide=slide,
        )
        dz, dgamma, dwb = local.pair_bias_bwd32(_trunk(pair, n), canon[_GAMMA], canon[_WBIAS], db)
        cs_kq = rows.kqln_bwd32(dp, qkr, pk.kq, kq_eps, partials=True) if kqln else None
        dx1 = rows.gemm32(dp, pk.wpT)  # K = 512
        ds, _ = adaln_b(s2, mod[:, _S1], dx1, da1, dmod[:, _S1], dmod[:, _B1])  # AdaLN 1 + the residual
        if ident:
            dc = rows.gemm32(dmod, pk.wmodT)  # K = 768: every table is a product of the raw c
            gparts = None
        else:  # ONE grouped GEMM [dcn1, dcn2, dc0] (K = 256 each), then the two LayerNorm backwards onto dc0
            dc, gparts = rows.ln_bwd32_grp(c2, rows.gemm32(dmod, pk.wgrpT, k=2 * DS_, a_grp=DS_), pk.g1, pk.g2)
        dcols = rows.colsum32(dmod, partials=True)  # (partial rows: summed into the gradients by ``col_jobs`` below)
        dbq = rows.colsum32(dp[:, qc : qc + DS_], partials=True)
        A_ = "attention."
        spec = [
            (dt, hh, [("transition.squeeze.weight", 0)]),
            (
                dab,
                x2,
                [("transition.expand_a.weight", 0), ("transition.expand_b.weight", 0), ("transition.expand_a.weight", 1), ("transition.expand_b.weight", 1)],
            ),  # u interleaves [a_0 | b_0 | a_1 | b_1]
            (dyo, gated, [(A_ + "to_out.weight", 0)]),
            (dmod[:, 0:256], cn1, [(A_ + "ada_ln_in.to_scale.weight", 0), (A_ + "ada_ln_in.to_bias.weight", 0)]),
            (dmod[:, 256:512], cn2, [("transition.ada_ln_in.to_scale.weight", 0), ("transition.ada_ln_in.to_bias.weight", 0)]),
            (dmod[:, 512:768], c2, [(A_ + "to_scale.weight", 0), ("transition.to_scale.weight", 0)]),
            (dp, x1, [(A_ + "to_query.weight", 0), (A_ + "to_key.weight", 0), (A_ + "to_value.weight", 0), (A_ + "to_gate.weight", 0)]),
        ]
        names, wg_weights = WEIGHTS, canon
        if up:
            assert dup is not None
            spec.append((dup, x2, [(_UP_NAME, 0), (_UP_NAME, 1)]))
            names, wg_weights = (*WEIGHTS, _UP_NAME), [*canon, opt_leaves[4 if kqln else 0]]
        if rc != m:  # shared conditioning: the tables' products have n rows, their own launch
            cond_names = {"attention.ada_ln_in.to_scale.weight", "transition.ada_ln_in.to_scale.weight", "attention.to_scale.weight"}
            fused = _fused_wgrads([e for e in spec if e[2][0][0] not in cond_names], wg_weights, names, True, dev, m)
            fused.update(_fused_wgrads([e for e in spec if e[2][0][0] in cond_names], wg_weights, names, True, dev, rc))
        else:
            fused = _fused_wgrads(spec, wg_weights, names, True, dev, m)
        del spec
        # every bias / LayerNorm-weight gradient (and KQLN's four leaves) from the partial rows, in one launch (``col_jobs``)
        if gparts is None:  # IDENT: the conditioning LayerNorm weights are constant leaves
            gparts = torch.zeros(1, 2 * DS_, device=dev, dtype=F32)
        wmap: dict[str, torch.Tensor] = dict(zip(WEIGHTS, canon, strict=True))
        small = {
            "attention.ada_ln_in.ln_cond.weight": (gparts, 0),
            "attention.ada_ln_in.to_scale.bias": (dcols, 0),
            "attention.to_query.bias": (dbq, 0),
            "attention.ln_pair.weight": (dgamma.reshape(1, -1), 0),
            "attention.to_bias.weight": (dwb.reshape(1, -1), 0),
            "attention.to_scale.bias": (dcols, 512),
            "transition.ada_ln_in.ln_cond.weight": (gparts, 128),
            "transition.ada_ln_in.to_scale.bias": (dcols, 256),
            "transition.to_scale.bias": (dcols, 640),
        }
        jobs = [(src, col0, wmap[name]) for name, (src, col0) in small.items()]
        if kqln:
            assert cs_kq is not None
            jobs += [(cs_kq, DS_ * i, w) for i, w in enumerate(opt_leaves[:4])]
        outs = rows.col_jobs(jobs)
        fused.update(zip(small, outs[: len(small)], strict=True))
    per_weight = [fused[name] for name in WEIGHTS]
    if kqln:
        per_weight += outs[len(small) :]
    if up:
        per_weight.append(fused[_UP_NAME])
    return [_unpadded(ds.view(a, 1, n, DS_), single.shape), _unpadded(dc.view(cond.shape[0], 1, n, DC_), cond.shape), _dpair(dz, pair), *per_weight]


class _VarBlock32(torch.autograd.Function):
    """The variant block's fp32 forward and backward (TF32 kernels); gradients for single, cond, pair and every leaf. ``flags`` / ``uid``:
    ``_VarBlock``'s."""

    @staticmethod
    def forward(ctx, flags, mask, uid, single, cond, pair, *weights):
        slide, ident, pars, up, kqln, kq_eps, eps = flags
        out, *saved = _var_train_fwd32(single, cond, pair, mask, list(weights), slide, ident, pars, up, kqln, kq_eps, eps, uid)
        ctx.save_for_backward(mask, single, cond, pair, *saved, *weights)
        ctx.n_saved = len(saved)
        ctx.flags = flags
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        mask, single, cond, pair, *rest = ctx.saved_tensors
        saved, weights = rest[: ctx.n_saved], rest[ctx.n_saved :]
        slide, ident, pars, up, kqln, kq_eps, eps = ctx.flags
        grads = _var_bwd32(dy, single, cond, pair, mask, list(weights), list(saved), slide, ident, pars, up, kqln, kq_eps, eps)
        return (None, None, None, *grads)
