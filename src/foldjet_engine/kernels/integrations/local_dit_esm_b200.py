"""ESMFold2's AF3-form atom block with the atom pair (``modules.atom_dit.AtomDiTESMFold2Pair``) on the B200 (sm_100a, card ``b200_sm100``):
``origin/b200/ladder-final``'s tcgen05 ESMFold2 block (dc425caa5, ``integrations/local_dit.py`` ``_esm_block`` / ``_EsmBlock`` /
``_EsmBlock32``) on the 2.5 module, B200 only.

    inference (bf16)  cond_fwd RMS (hoisted: mod + the K / V tables) -> pre_fwd RMS (+ q / k norm + rotary) -> esm_bias (hoisted) -> attn_esm
                      -> post_fwd RMS
    training (bf16)   the same forward with its saves; tr_bwd_gate RMS -> post_bwd RMS -> esm dQ (+ dbias) / dK dV -> q / k norm + rotary
                      backward -> esm pair-bias backward -> pre_bwd RMS -> [dmod | dmkv] [Wmod; Wkv] -> SiLU backward -> atom_dw
    fp32 (TF32)       the fp32 path's unfused kernels with their RMS forms around the TF32 ESMFold2 attention, forward and backward.

The row kernels are ``b200_atom``'s ``-DRMS=1`` builds (RMSNorm AdaLNs, tables of silu(c), the scales identity-centred by a bias of 1, raw
gates); the attention, the q / k norm + rotary and the position-indexed pair bias are ``b200_atom_local/esm.py``'s. The row kernels take the
AF3 block's 27 parameter slots (``local_dit_b200.WEIGHTS + CROSS_WEIGHTS``); ESMFold2 has no conditioning LayerNorm and no projection biases
but the query's, so those slots hold constants (the LN weights 1, the scale biases 1, the gate biases 0); the q / k RMSNorm is affine-free
(FoldForge ``rope_norm``): its kernel weights are a constant 1 and take no gradient. One fold (B = 1), the rotary tables [1, N, 16].
"""

from __future__ import annotations

import functools
import os
from types import SimpleNamespace
from typing import cast

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from foldjet_engine.kernels import _capture
from foldjet_engine.kernels._compile import device_constant, opaque
from foldjet_engine.kernels._device import num_sms
from foldjet_engine.kernels._tf32 import tf32_grid
from foldjet_engine.kernels.integrations.local_dit_b200 import (
    _B1,
    _B2,
    _GAMMA,
    _S1,
    _S2,
    _SO,
    _ST,
    _WBIAS,
    BF,
    CROSS_WEIGHTS,
    DC_,
    DS_,
    F32,
    NH_,
    QUERIES,
    WEIGHTS,
    _dpair,
    _fused_wgrads,
    _gated_bwd,
    _kernels32_ready,
    _kernels_ready,
    _keys,
    _kv_pack,
    _padded,
    _padded_length,
    _shared_cond,
    _side_stream,
    _tkey,
    _trunk,
    _unpadded,
    _weights,
)

RMS_EPS = 1.1920928955078125e-07  # torch.finfo(torch.float32).eps: every RMSNorm of the block (FoldForge's, whatever the dtype)
_HOIST_COND: dict = _capture.weight_cache("local_dit_esm_b200.hoist_cond")
_HOIST_BIAS: dict = _capture.weight_cache("local_dit_esm_b200.hoist_bias")
_ESM_ONES = {
    "attention.ada_ln_in.ln_cond.weight",
    "attention.ada_ln_in.to_scale.bias",
    "transition.ada_ln_in.ln_cond.weight",
    "transition.ada_ln_in.to_scale.bias",
    "attention.ada_ln_kv.ln_cond.weight",
    "attention.ada_ln_kv.to_scale.bias",
}
_ESM_ZEROS = {"attention.to_scale.bias", "transition.to_scale.bias"}
_ESM_CONST: dict = {}


def _esm_const(device, value: float, width: int = DS_):
    """A persistent fp32 constant vector per device (made eagerly and kept; inside a capture with none kept yet, a fresh one that is not kept)."""
    key = (str(device), value, width)
    c = _ESM_CONST.get(key)
    if c is None:
        c = torch.full((width,), value, device=device, dtype=F32)
        if not torch.cuda.is_current_stream_capturing():
            _ESM_CONST[key] = c
    return c


def _esm_weights(module):
    """The module's parameters in the 27 slots of ``WEIGHTS + CROSS_WEIGHTS`` (constants where ESMFold2 has none)."""
    dev = module.attention.to_query.weight.device
    out = []
    for name in (*WEIGHTS, *CROSS_WEIGHTS):
        if name in _ESM_ONES:
            out.append(_esm_const(dev, 1.0))
        elif name in _ESM_ZEROS:
            out.append(_esm_const(dev, 0.0))
        else:
            out.append(module.get_parameter(name))
    return out


#: ``esm.KERNELS_ESM32`` entries the fp32 inference launches (the TF32 attention, the q / k norm + rotary)
ESM32_INFER = ("esm32", "esm_qk32")


@device_constant
def _esm_ready(index: int) -> bool:
    """Build and load the bf16 ESMFold2 kernels (row kernels' RMS builds, attention, q / k norm, pair bias) on ``index`` once."""
    from foldjet_engine.kernels.augmented_attention.cuda import b200_atom as rows
    from foldjet_engine.kernels.augmented_attention.cuda.b200_atom_local import esm

    _kernels_ready(index)
    for name in rows.KERNELS_ESM:
        rows.atom_kernel(name, index)
    esm.load_esm(index)
    return True


@device_constant
def _esm_train_ready(index: int) -> bool:
    """Build and load ESMFold2's training kernels (bf16 and fp32) on ``index`` once."""
    from foldjet_engine.kernels.augmented_attention.cuda import b200_atom as rows
    from foldjet_engine.kernels.augmented_attention.cuda.b200_atom_local import esm

    esm.load_esm(index)
    esm.load_esm32(index)
    for name in rows.KERNELS_ESM_TRAIN:
        rows.atom_kernel(name, index)
    esm.load_esm_train(index)
    rows.atom_dw(index, False)
    rows.atom_dw(index, True)
    return True


@device_constant
def _esm32_ready(index: int) -> bool:
    """Build and load ESMFold2's fp32 inference kernels on ``index`` once (the training ones: ``_esm_train_ready``)."""
    from foldjet_engine.kernels.augmented_attention.cuda import b200_atom as rows
    from foldjet_engine.kernels.augmented_attention.cuda.b200_atom_local import esm

    _kernels32_ready(index)
    esm._load_esm("esm_bias", index)
    for name in ESM32_INFER:
        esm._load_esm32(name, index)
    rows.fused_kernel32("pre_rms", index)
    rows.fused_kernel32("post_rms", index)
    rows.fused_kernel32("post_rms_pair", index)
    for name in ("pre_rms_t", "post_rms_t", "post_rms_pair_t", "pre_rms_w", "post_rms_w"):  # (round 3 part 10 / 11: the transposed kernels)
        rows.fused_kernel32(name, index)
    return True


def _esm_cond_pack(weights, wc, infer=False):
    """The RMS cond_fwd operands: W [1152, 128] bf16 = [Wmod; Wkv_scale; Wkv_shift; 0], b [1152] fp32 = [bmod; 0; 0; 0] (the K / V scale's
    identity bias 1 is pre_fwd's ``bs``). ``infer``: a capture reads the eager warm-up's pack (``capture_reads_eager``), and the two AdaLN
    scales' identity bias is left out: their tables hold the raw s = W_s silu(c) and the ``-DRAWSC=1`` row kernels compute
    ((1 + s) x_n + b) in fp32 with one rounding, as the reference module does (rn(1 + s) costs s its low bits: 16 of the 30 bf16 inference
    cases were YELLOW with it, e / tol up to 1.43)."""

    def build():
        wm = _kv_pack(weights, infer)[0]
        dev = wm.device
        with torch.no_grad():
            w = torch.cat([wc[0], wm, torch.zeros(DS_, DS_, device=dev, dtype=BF)]).contiguous()
            b = torch.cat([wc[1], torch.zeros(3 * DS_, device=dev)]).contiguous()
            if infer:  # mod = [s1 | bi1 | so | s2 | bi2 | st]: the raw s1, s2
                b[0:DS_] = 0
                b[3 * DS_ : 4 * DS_] = 0
        return w, b

    return _capture.param_forms(("local_dit_esm_b200", bool(infer)), weights, build, capture_reads_eager=infer)


def _hoisted(store, tensor, key, build, also=()):
    return _capture.hoisted_table(store, tensor, key, build, limit=64, also=also)


def _hoisting() -> bool:
    return True


#: Training on the tcgen05 block (``_EsmBlock`` / ``_EsmBlock32``), routed since round 3 part 7 (2026-11-17): the AdaLNs on raw scale tables
#: (``-DRAWSC``, forward and backward), the packs in one launch, the q / k norm + rotary inside the projection kernel and side-stream forks
#: took bf16 1024 atoms 246 -> ~150 us (the sm_80 path 229). ``MW_ESM_TT=0`` sends training back to ``local_dit_esm`` (A / B only).
ROUTE_TRAINING = True


def _route_training() -> bool:
    return os.environ.get("MW_ESM_TT", "1" if ROUTE_TRAINING else "0") == "1"


@functools.lru_cache(maxsize=8)
def _side2(index: int) -> torch.cuda.Stream:
    """A second side stream per device (``_fork``: work independent of both the main chain and the pair bias' side stream)."""
    with torch.cuda.device(index):
        return torch.cuda.Stream()


def _fork_rows() -> int:
    """Up to how many rows the training block forks independent work onto side streams (round 3 part 7; ``MW_ESM_FORK`` overrides)."""
    return int(os.environ.get("MW_ESM_FORK", "4096"))


def _fork(dev, m, build, second=False):
    """``build()`` on a side stream beside what the caller launches next (a fork / join in a captured graph) when the block has at most
    ``_fork_rows()`` rows (few tiles: the main chain leaves most SMs idle), else on the current stream: -> (result, join or None) --
    ``_join(join)`` before the result is read."""
    if m > _fork_rows():
        return build(), None
    cur = torch.cuda.current_stream()
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    side = _side2(index) if second else _side_stream(index)
    fork = torch.cuda.Event()
    fork.record(cur)
    side.wait_event(fork)
    with torch.cuda.stream(side):
        out = build()
    join = torch.cuda.Event()
    join.record(side)
    if not torch.cuda.is_current_stream_capturing():
        for t in out if isinstance(out, (tuple, list)) else (out,):
            if isinstance(t, torch.Tensor):
                t.record_stream(cur)
    return out, join


def _join(*joins):
    for j in joins:
        if j is not None:
            torch.cuda.current_stream().wait_event(j)


def _esm_fwd_fake(single, cond, pair, mask, weights, qkw, cos, sin):
    return torch.empty_like(single)


@opaque(fake=_esm_fwd_fake, name="local_dit_esm_b200_fwd")
def _esm_fwd(
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    weights: list[torch.Tensor],
    qkw: list[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """ESMFold2's block output [A, 1, N, 128] bf16 (inference). ``cos`` / ``sin``: the rotary tables, fp32 [n, 16] at the padded length."""
    from foldjet_engine.kernels.augmented_attention.cuda import b200_atom as rows
    from foldjet_engine.kernels.augmented_attention.cuda.b200_atom import f8 as _f8
    from foldjet_engine.kernels.augmented_attention.cuda.b200_atom_local import esm as local

    f8 = _f8 if _f8.ENABLED else None  # FP8 (opt-in, ``b200_atom.f8``)
    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    m = a * n
    with torch.cuda.device(single.device):
        wc, wp, wo, _ = _weights(weights[: len(WEIGHTS)])
        wrms, brms = _esm_cond_pack(weights, wc, infer=True)
        s2 = _padded(single, n).reshape(m, DS_).contiguous()
        wkey = tuple((w.data_ptr(), w._version) for w in weights)

        def tables():
            c2 = _padded(cond, n).reshape(cond.shape[0] * n, DC_).contiguous()
            return rows.cond_fwd(c2, wrms, brms, wc[2], wc[3], rms=True)

        keys = _keys(mask, n0, n)

        def bias():
            return local.esm_bias(_trunk(pair, n), weights[_GAMMA], weights[_WBIAS], keys, n0)

        def side():  # the pair bias (hoisted)
            if not _hoisting():
                return bias()
            mk = None if mask is None else _tkey(mask)
            return _hoisted(_HOIST_BIAS, pair, ("esm", wkey[_GAMMA], wkey[_WBIAS], n, mk), bias, () if mask is None else (mask,))

        def chain():  # the atoms' chain up to the attention's operands
            mod, mkv = _hoisted(_HOIST_COND, cond, ("esm", wkey, n), tables) if _hoisting() else tables()
            bs = _kv_pack(weights, True)[1]
            wq, wk = qkw  # (the affine-free q / k RMSNorm: constant fp32 ones)
            # Q / K leave the projection kernel normalised and rotated (``-DQKROPE=1``; round 3 part 4: one launch fewer)
            if f8 is not None and f8.serves("pre", m, False):  # FP8 (opt-in): the e4m3 projections, same epilogue (RMS / rotary on the staged rows)
                qn, kn, v, sg = f8.pre_fwd(
                    s2, mod, wp[0], wp[1], mkv=mkv, bs=bs, eps=RMS_EPS, parallel=True, rms=True, raw=True, rope=(wq, wk, cos, sin, RMS_EPS)
                )
                return mod, v, sg, qn, kn
            qn, kn, v, sg, _, _ = cast(tuple, rows.pre_fwd_kv(
                s2, mod, wp[0], wp[1], mkv, bs, eps=RMS_EPS, parallel=True, rms=True, raw=True, rope=(wq, wk, cos, sin, RMS_EPS)
            ))
            return mod, v, sg, qn, kn

        mod, v, sg, qn, kn = chain()
        bias2 = side()
        o = local.attn_esm_sm100(qn.view(a, n, DS_), kn.view(a, n, DS_), v.view(a, n, DS_), bias2)
        if f8 is not None and f8.serves("post", m, False):
            out = f8.post_fwd(s2, o.view(m, DS_), sg, mod, *wo, eps=RMS_EPS, rms=True, raw=True)
        else:
            out = rows.post_fwd(s2, o.view(m, DS_), sg, mod, *wo, eps=RMS_EPS, rms=True, raw=True)[0]
    return _unpadded(out.view(a, 1, n, DS_), single.shape)


def run_block(module, single, cond, pair, mask=None, rope=None):
    """ESMFold2's block on the sm_100a kernels; :func:`reason` must have accepted the call (inference, or training through ``_EsmBlock`` /
    ``_EsmBlock32``). The caller runs it with autocast off (the kernels' precision is their own)."""
    n0 = single.shape[2]
    n = _padded_length(n0)
    assert rope is not None  # (``reason``: the block rotates q and k)
    cos, sin = ((t[0].float() if n == n0 else F.pad(t[0].float(), (0, 0, 0, n - n0))).contiguous() for t in rope)
    dev = single.device
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    weights = _esm_weights(module)
    qkw = [_esm_const(dev, 1.0, 32), _esm_const(dev, 1.0, 32)]  # the affine-free q / k RMSNorm (no gradient)
    single, pair = single.contiguous(), pair.contiguous()
    mask = None if mask is None else mask.contiguous()
    train = torch.is_grad_enabled() and any(t.requires_grad for t in (single, cond, pair, *weights))
    if single.dtype == F32:
        _esm32_ready(index)
    else:
        _esm_ready(index)
    if train:
        _esm_train_ready(index)
    if single.dtype == F32:  # the fp32 path (TF32 tensor cores)
        if torch.is_grad_enabled() and any(
            t.requires_grad for t in (single, cond, pair, *weights, *qkw)
        ):  # (shared conditioning: tables once)
            return _EsmBlock32.apply(mask, cos, sin, single, _shared_cond(single, cond), pair, *weights, *qkw)
        return _esm_fwd32(single, _shared_cond(single, cond), pair, mask, weights, qkw, cos, sin, False)[0]
    cond = _shared_cond(single, cond)
    if torch.is_grad_enabled() and any(t.requires_grad for t in (single, cond, pair, *weights, *qkw)):
        return _EsmBlock.apply(mask, cos, sin, single, cond, pair, *weights, *qkw)
    return _esm_fwd(single, cond, pair, mask, weights, qkw, cos, sin)


# ------------------------------------------------------------------------------------------------------------ ESMFold2 training (bf16)
# The forward saves what the backward reads; the backward runs
#
#     tr_bwd_gate RMS -> post_bwd RMS (dO, D) -> esm dq (+ dbias) / dkv -> q / k norm + rotary backward -> esm pair-bias backward
#     -> pre_bwd RMS (AdaLN 1 and the K / V AdaLN, dmkv) -> [dmod | dmkv] [Wmod; Wkv] (cuBLAS, K = 1024) -> SiLU backward -> atom_dw
#
# (``-DRMS=1`` builds of the row kernels; ``lattn_esm_dq.cu`` / ``lattn_esm_dkv.cu``, ``lesm.cu``, ``lbias.cu``). Every conditioning table of
# ESMFold2's block is a product of silu(c), so the conditioning's gradient is one K = 1024 product of the eight tables' gradients (the
# backward layout of dmod, then the K / V AdaLN's [scale | shift]) and the weight gradients of all eight are two grouped problems against
# sc = rn(silu(c)). The constant slots of ``_esm_weights`` get no gradient.
_ESM_SAVED = ("mod", "mkv", "qkr", "qn", "kn", "v", "sg", "x1", "xkv", "bias2", "o", "lse", "u", "a2", "x2", "t")
_ESM_GRAD = tuple(name for name in (*WEIGHTS, *CROSS_WEIGHTS) if name not in _ESM_ONES and name not in _ESM_ZEROS)
_ESM_COND_BWD = (  # the weights of [dmod | dmkv] in the backward layout: [dsc1 | dbi1 | dsc2 | dbi2 | dos | dts | dkv_scale | dkv_shift]
    "attention.ada_ln_in.to_scale.weight",
    "attention.ada_ln_in.to_bias.weight",
    "transition.ada_ln_in.to_scale.weight",
    "transition.ada_ln_in.to_bias.weight",
    "attention.to_scale.weight",
    "transition.to_scale.weight",
    "attention.ada_ln_kv.to_scale.weight",
    "attention.ada_ln_kv.to_bias.weight",
)


#: The bf16 / master training packs (``b200_atom.esm_pack``, one launch; round 3 part 7, 2026-11-17), rows of 128 elements of one bf16 buffer:
#: cond_fwd's [Wsc1; Wbi1; Wos; Wsc2; Wbi2; Wts; Wkv_scale; Wkv_shift; 0] at 0, Wp = [Wq; Wk; Wv; Wg] at 1152, Wo at 1664, Wu = [Wa; Wb] at
#: 1792, Ws [128, 256] at 2304, the backward layout of the tables' weights (``_ESM_COND_BWD``) at 2560; transposed: Ws^T [256, 128] at 0,
#: Wu^T [128, 512] at 256, Wo^T at 768, Wp^T [128, 512] at 896. The AdaLN scale tables are raw (``-DRAWSC=1`` row builds): the table bias
#: is zero everywhere, a constant.
_ESM_TAB_FWD = (
    "attention.ada_ln_in.to_scale.weight",
    "attention.ada_ln_in.to_bias.weight",
    "attention.to_scale.weight",
    "transition.ada_ln_in.to_scale.weight",
    "transition.ada_ln_in.to_bias.weight",
    "transition.to_scale.weight",
    "attention.ada_ln_kv.to_scale.weight",
    "attention.ada_ln_kv.to_bias.weight",
)


def _esm_train_packs(weights):
    """The bf16 operands of ESMFold2's training block from its parameters (fp32 master or bf16) in one launch, scoped to the capture like
    every training pack (``_capture.param_forms``: a captured step records the launch; the weights change between replays)."""
    named: dict[str, torch.Tensor] = dict(zip((*WEIGHTS, *CROSS_WEIGHTS), weights, strict=True))
    A_, T_ = "attention.", "transition."
    leaves = [*_ESM_TAB_FWD, *(A_ + f"to_{p}.weight" for p in ("query", "key", "value", "gate")), A_ + "to_out.weight",
              T_ + "expand_a.weight", T_ + "expand_b.weight", T_ + "squeeze.weight", A_ + "to_query.bias"]  # fmt: skip
    ws = [named[k] for k in leaves]

    def build():
        from foldjet_engine.kernels.augmented_attention.cuda import b200_atom as rows

        w = {k: t.detach() for k, t in zip(leaves, ws, strict=True)}
        R = DS_ * DS_
        units = []

        def unit(src, row0=0, sp=DS_, d0=0, p0=DS_, d1=-1, dt=-1, pt=DS_):
            units.append((src, row0, sp, d0, p0, d1, DS_, dt, pt))

        for i, name in enumerate(_ESM_TAB_FWD):  # cond_fwd's order and the backward's (``_ESM_COND_BWD``)
            unit(w[name], d0=i * R, d1=2560 * DS_ + _ESM_COND_BWD.index(name) * R)
        unit(None, d0=1024 * DS_)
        for i, p in enumerate(("query", "key", "value", "gate")):
            unit(w[A_ + f"to_{p}.weight"], d0=(1152 + 128 * i) * DS_, dt=896 * DS_ + 128 * i, pt=4 * DS_)
        unit(w[A_ + "to_out.weight"], d0=1664 * DS_, dt=768 * DS_)
        for i, (name, h) in enumerate((("expand_a", 0), ("expand_a", 1), ("expand_b", 0), ("expand_b", 1))):
            unit(w[T_ + name + ".weight"], row0=h * R, d0=(1792 + 128 * i) * DS_, dt=256 * DS_ + 128 * i, pt=4 * DS_)
        for c in range(2):
            unit(w[T_ + "squeeze.weight"], row0=128 * c, sp=2 * DS_, d0=2304 * DS_ + 128 * c, p0=2 * DS_, dt=128 * c * DS_)
        with torch.no_grad():
            bf, bt, vec = rows.esm_pack(units, [w[A_ + "to_query.bias"]], 3584 * DS_, 1408 * DS_, False)
        assert bt is not None

        def mat(buf, row0, nrow, shape):
            return buf[row0 * DS_ : (row0 + nrow) * DS_].view(shape)

        return SimpleNamespace(
            wrms=mat(bf, 0, 1152, (9 * DS_, DS_)),
            wp=mat(bf, 1152, 512, (4 * DS_, DS_)),
            wo=mat(bf, 1664, 128, (DS_, DS_)),
            wu=mat(bf, 1792, 512, (4 * DS_, DS_)),
            ws=mat(bf, 2304, 256, (DS_, 2 * DS_)),
            wcb=mat(bf, 2560, 1024, (8 * DS_, DS_)),
            wsT=mat(bt, 0, 256, (2 * DS_, DS_)),
            wuT=mat(bt, 256, 512, (DS_, 4 * DS_)),
            woT=mat(bt, 768, 128, (DS_, DS_)),
            wpT=mat(bt, 896, 512, (DS_, 4 * DS_)),
            bq=vec[0],
        )

    return _capture.param_forms(("local_dit_esm_b200_train",), ws, build)


def _esm_train_fwd_fake(single, cond, pair, mask, weights, qkw, cos, sin):
    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    m = a * n
    t = lambda w=DS_, d=BF: single.new_empty((m, w), dtype=d)
    nwin = n // QUERIES
    rc = cond.shape[0] * n
    shapes = {
        "mod": single.new_empty((rc, 6 * DS_)),
        "mkv": single.new_empty((rc, 2 * DS_)),
        "qkr": single.new_empty((m, 2 * DS_)),
        "bias2": single.new_empty((NH_, nwin, QUERIES, 160), dtype=F32),
        "lse": single.new_empty((a, NH_, n), dtype=F32),
    }
    return [torch.empty_like(single), *(shapes[k] if k in shapes else t() for k in _ESM_SAVED)]


@opaque(fake=_esm_train_fwd_fake, name="local_dit_esm_b200_train_fwd")
def _esm_train_fwd(
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    weights: list[torch.Tensor],
    qkw: list[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> list[torch.Tensor]:
    """ESMFold2's block output [A, 1, N, 128] bf16 and the activations its backward reads (``_ESM_SAVED``; nothing hoisted)."""
    from foldjet_engine.kernels.augmented_attention.cuda import b200_atom as rows
    from foldjet_engine.kernels.augmented_attention.cuda.b200_atom_local import esm as local

    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    m = a * n
    with torch.cuda.device(single.device):
        pk = _esm_train_packs(weights)
        one = _esm_const(single.device, 1.0)
        s2 = _padded(single, n).reshape(m, DS_).contiguous()
        c2 = _padded(cond, n).reshape(cond.shape[0] * n, DC_).contiguous()  # (once when the samples share it: rows r % N)
        keys = _keys(mask, n0, n)
        bias2, jb = _fork(single.device, m, lambda: local.esm_bias(_trunk(pair, n), weights[_GAMMA], weights[_WBIAS], keys, n0))
        mod, mkv = rows.cond_fwd(c2, pk.wrms, _esm_const(single.device, 0.0, 9 * DS_), one, one, rms=True)  # (raw scale tables: no bias)
        # Q / K leave normalised and rotated, the raw q | k saved for the norm's backward (``-DQKSAVE=1``: one launch fewer)
        qn, kn, v, sg, x1, xkv, qkr = cast(tuple, rows.pre_fwd_kv(
            s2, mod, pk.wp, pk.bq, mkv, one, save=True, eps=RMS_EPS, parallel=True, rms=True, raw=True, rope=(*qkw, cos, sin, RMS_EPS)
        ))
        lse = torch.empty(a, NH_, n, device=single.device, dtype=F32)
        _join(jb)
        o = local.attn_esm_sm100(qn.view(a, n, DS_), kn.view(a, n, DS_), v.view(a, n, DS_), bias2, lse=lse)
        out, u, a2, x2, t = rows.post_fwd(s2, o.view(m, DS_), sg, mod, pk.wo, pk.wu, pk.ws, save=True, eps=RMS_EPS, rms=True, raw=True)
    out = _unpadded(out.view(a, 1, n, DS_), single.shape)
    assert xkv is not None  # (``save``: pre_fwd_kv returned it)
    return [out, mod, mkv, qkr, qn, kn, v, sg, x1, xkv, bias2, o.view(m, DS_), lse, u, a2, x2, t]


def _esm_cond_bwd_pack(weights):
    """[Wsc1; Wbi1; Wsc2; Wbi2; Wos; Wts; Wkv_scale; Wkv_shift] [1024, 128] bf16: the weights of the backward layout of [dmod | dmkv]."""
    named: dict[str, torch.Tensor] = dict(zip((*WEIGHTS, *CROSS_WEIGHTS), weights, strict=True))
    ws = [named[name] for name in _ESM_COND_BWD]

    def build():
        with torch.no_grad():
            return torch.cat([w.to(BF) for w in ws]).contiguous()

    return _capture.param_forms(("local_dit_esm_cond_bwd",), ws, build)


def _esm_bwd_fake(dy, single, cond, pair, mask, weights, qkw, cos, sin, saved):
    named: dict[str, torch.Tensor] = dict(zip((*WEIGHTS, *CROSS_WEIGHTS), weights, strict=True))
    return [
        torch.empty_like(single),
        torch.empty_like(cond),
        torch.empty_like(pair),
        *(torch.empty_like(named[k]) for k in _ESM_GRAD),
        *(torch.empty_like(w) for w in qkw),
    ]


def _esm_bwd16_fake(dy, single, cond, pair, mask, weights, qkw, cos, sin, saved):
    return _esm_bwd_fake(dy, single, cond, pair, mask, weights, [], cos, sin, saved)


@opaque(fake=_esm_bwd16_fake, name="local_dit_esm_b200_bwd")
def _esm_bwd(
    dy: torch.Tensor,
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    weights: list[torch.Tensor],
    qkw: list[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    saved: list[torch.Tensor],
) -> list[torch.Tensor]:
    """d single, d cond, d pair and the gradients of ``_ESM_GRAD``, each in its parameter's dtype (fresh tensors). The q / k norm weights
    are constants (affine-free RMSNorm): no gradient."""
    from foldjet_engine.kernels.augmented_attention.cuda import b200_atom as rows
    from foldjet_engine.kernels.augmented_attention.cuda.b200_atom_local import esm as local

    mod, mkv, qkr, qn, kn, v, sg, x1, xkv, bias2, o, lse, u, a2, x2, t = saved
    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    m = a * n
    dev = single.device
    with torch.cuda.device(dev):
        pk = _esm_train_packs(weights)
        one = _esm_const(dev, 1.0)
        s2 = _padded(single, n).reshape(m, DS_).contiguous()
        shared = cond.shape[0] == 1 and a > 1  # one conditioning for the samples (``_shared_cond``)
        c2 = _padded(cond, n).reshape(cond.shape[0] * n, DC_).contiguous()
        dy = _padded(dy, n).reshape(m, DS_).to(BF).contiguous()
        dmod = torch.empty(m, 8 * DS_, device=dev, dtype=BF)  # [dsc1 | dbi1 | dsc2 | dbi2 | dos | dts | dkv_scale | dkv_shift]
        dm6 = dmod[:, : 6 * DS_]
        tsums = _table_sums(a, n, True, dev) if shared else None  # (shared: the row kernels sum the tables' gradient over the samples)
        dp = torch.empty(m, 4 * DS_, device=dev, dtype=BF)  # [dq | dk | dv | dg]
        # the row kernels' column sums: one partial row per CTA (``COLS_STRIDE`` = 1024 floats, at most one CTA per SM), summed in a fixed order
        # below (local_dit_b200's layout): [0, 768) the bias columns (ESMFold2 has none: discarded), [768, 896) d bq, [896, 1024) d bs (discarded)
        pcols = torch.zeros(num_sms(dev), 1024, device=dev)
        dbias_cols, dbq, dbs = pcols, pcols[:, 768:], pcols[:, 896:]
        dd = torch.empty(a, NH_, n, device=dev)
        dt, hh, dab = rows.tr_bwd_gate(dy, t, mod, x2, pk.wu, pk.wsT, dm6, pcols[:, 640:], rms=True, part=tsums)
        da, du, gated, do = rows.post_bwd(
            dab, dy, a2, mod, u, sg, o, pk.wuT, pk.woT, dm6, dp, dd, dbias_cols, n, eps=RMS_EPS, rms=True, part=tsums, raw=True
        )
        db2 = local.attn_esm_bwd_sm100(qn.view(a, n, DS_), kn.view(a, n, DS_), v.view(a, n, DS_), do.view(a, n, DS_), bias2, lse, dd, dp)
        wq, wk = qkw
        local.esm_qkrope_bwd(dp[:, :DS_], dp[:, DS_ : 2 * DS_], qkr[:, :DS_], qkr[:, DS_:], wq, wk, cos, sin, RMS_EPS, weights=False)
        keys = _keys(mask, n0, n)
        (dz, dgamma, dwb), jb = _fork(dev, m, lambda: local.esm_bias_bwd(_trunk(pair, n), weights[_GAMMA], weights[_WBIAS], db2, keys, n0))
        ds, _ = rows.pre_bwd_kv(
            dp,
            s2,
            mod,
            da,
            pk.wpT,
            dm6,
            dbias_cols,
            dbq,
            x1,
            mkv,
            one,
            dbs,
            eps=RMS_EPS,
            parallel=True,
            rms=True,
            dmkv=dmod[:, 6 * DS_ :],
            part=tsums,
            raw=True,
        )
        named: dict[str, torch.Tensor] = dict(zip((*WEIGHTS, *CROSS_WEIGHTS), weights, strict=True))
        # d bq: the partial rows summed in order (``col_jobs``, deterministic) into a fresh tensor of the bias' dtype; the pair bias' two
        # gradients are fresh fp32 tensors of their own (cast only for bf16 parameters)
        (dbq,), jq = _fork(dev, m, lambda: rows.col_jobs([(pcols, 768, named["attention.to_query.bias"])]), second=True)
        dtab = (
            tsums.finish() if tsums is not None else dmod
        )  # summed over the samples (in the row kernels, fixed order) when they share the conditioning
        g = torch.mm(dtab, pk.wcb, out_dtype=F32)  # the tables' gradient through their weights, fp32
        dc, sc = local.esm_silu_bwd(c2, g)
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
            (dp[:, 0:DS_], x1, [(A_ + "to_query.weight", 0)]),
            (dp[:, 3 * DS_ : 4 * DS_], x1, [(A_ + "to_gate.weight", 0)]),
            (dp[:, DS_ : 3 * DS_], xkv, [(A_ + "to_key.weight", 0), (A_ + "to_value.weight", 0)]),
        ]
        cond_spec = [
            (dtab[:, : 4 * DS_], sc, [(name, 0) for name in _ESM_COND_BWD[:4]]),
            (dtab[:, 4 * DS_ :], sc, [(name, 0) for name in _ESM_COND_BWD[4:]]),
        ]
        if shared:  # the conditioning's products have N rows: their own launch
            fused = _fused_wgrads(spec, weights, WEIGHTS + CROSS_WEIGHTS, False, dev, m)
            fused.update(_fused_wgrads(cond_spec, weights, WEIGHTS + CROSS_WEIGHTS, False, dev, c2.shape[0]))
        else:
            fused = _fused_wgrads(spec + cond_spec, weights, WEIGHTS + CROSS_WEIGHTS, False, dev, m)
        _join(jb, jq)
    extra = {"attention.to_query.bias": dbq, "attention.ln_pair.weight": dgamma, "attention.to_bias.weight": dwb}
    grads = [fused[name] if name in fused else _fresh(extra[name], named[name]) for name in _ESM_GRAD]
    return [
        _unpadded(ds.view(a, 1, n, DS_), single.shape),
        _unpadded(dc.view(cond.shape[0], 1, n, DC_), cond.shape),
        _dpair(dz, pair),
        *grads,
    ]


def _fresh(g, like):
    """A kernel's fresh gradient ``g`` as ``like``'s gradient: itself (viewed in ``like``'s shape) when the dtype matches, else a cast."""
    return g.view(like.shape) if g.dtype == like.dtype else g.to(like.dtype).reshape(like.shape)


class _EsmBlock(torch.autograd.Function):
    """ESMFold2's block forward and backward on the sm_100a kernels (bf16 activations); gradients for single, cond, pair and every parameter
    (the constant slots of ``_esm_weights`` get None)."""

    @staticmethod
    def forward(ctx, mask, cos, sin, single, cond, pair, *params):
        nw = len(WEIGHTS) + len(CROSS_WEIGHTS)
        weights, qkw = list(params[:nw]), list(params[nw:])
        out, *saved = _esm_train_fwd(single, cond, pair, mask, weights, qkw, cos, sin)
        ctx.save_for_backward(mask, cos, sin, single, cond, pair, *saved, *params)
        ctx.n_saved = len(saved)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        mask, cos, sin, single, cond, pair, *rest = ctx.saved_tensors
        saved, params = rest[: ctx.n_saved], rest[ctx.n_saved :]
        nw = len(WEIGHTS) + len(CROSS_WEIGHTS)
        ds, dc, dpair, *g = _esm_bwd(dy, single, cond, pair, mask, list(params[:nw]), list(params[nw:]), cos, sin, list(saved))
        it = iter(g)
        out = [None if (name in _ESM_ONES or name in _ESM_ZEROS) else next(it) for name in (*WEIGHTS, *CROSS_WEIGHTS)]
        return (None, None, None, ds, dc, dpair, *out, None, None)  # (the q / k norm weights: constants)


# ------------------------------------------------------------------------------------------------------------ ESMFold2: the fp32 path (TF32)
# fp32 activations and parameters (no autocast): the fp32 path's unfused kernels with their RMS builds (``sm100_atom`` rows_tf32.cu
# ``f32_adaln_rms`` ..., gemm_tf32.cu) around ESMFold2's TF32 attention (``lattn_esm_tf32.cu``) and the fp32 q / k norm + rotary:
#
#     silu32 -> gemm32 (the eight tables at once; hoisted) -> AdaLN 1 RMS -> q | g -> K / V AdaLN RMS (of the raw atoms) -> k | v
#     -> q / k norm + rotary (tf32-rounded) -> esm_bias (fp32 pair; hoisted) -> attention -> gate -> Wo + gated residual -> AdaLN 2 RMS
#     -> SwiGLU -> Ws + gated residual
#
# Layouts: tab [M, 1024] = [s1 | bi1 | s2 | bi2 | so | st | kv_scale | kv_shift] (the scales hold 1 + s: their bias is 1, nothing sigmoid),
# P32 [M, 512] = [q | g | k | v] (v rounded to tf32 in the epilogue: only the attention's MMAs read it).
_ESM_ONES32 = 0b01000101  # tab blocks whose bias is 1: s1, s2, the K / V scale


def _esm_pack32(weights, qkw, infer=False):
    """The fp32 forward operands of ESMFold2's block in one flat fp32 buffer (matrices tf32-rounded): wcond [1024, 128] (the
    ``_ESM_COND_BWD`` order = tab's), wp [512, 128] = [Wq; Wg; Wk; Wv], wu [512, 128] interleaved, wo, ws; bcond [1024], bp [512]; the q / k
    norm weights [32] each (unrounded)."""
    named: dict[str, torch.Tensor] = dict(zip((*WEIGHTS, *CROSS_WEIGHTS), weights, strict=True))

    def build():
        def m(name):
            return named[name].detach().float().reshape(-1)

        with torch.no_grad():
            dev = weights[0].device
            wa, wb = m("transition.expand_a.weight").view(2 * DS_, DS_), m("transition.expand_b.weight").view(2 * DS_, DS_)
            mats = [m(name) for name in _ESM_COND_BWD]
            mats += [m(f"attention.to_{p}.weight") for p in ("query", "gate", "key", "value")]
            mats += [wa[:DS_].reshape(-1), wb[:DS_].reshape(-1), wa[DS_:].reshape(-1), wb[DS_:].reshape(-1)]
            mats += [m("attention.to_out.weight"), m("transition.squeeze.weight")]
            ones = torch.ones(DS_, device=dev, dtype=F32)
            zeros = torch.zeros(DS_, device=dev, dtype=F32)
            vecs = [ones if (_ESM_ONES32 >> i) & 1 else zeros for i in range(8)]
            vecs += [m("attention.to_query.bias"), zeros, zeros, zeros]
            vecs += [w.detach().float().reshape(-1) for w in qkw]
            flat = torch.cat(mats + vecs)
            nmat = sum(t.numel() for t in mats)
            flat[:nmat].copy_(tf32_grid(flat[:nmat]))
            off = [0]

            def take(n):
                t = flat[off[0] : off[0] + n]
                off[0] += n
                return t

            pk = SimpleNamespace()
            pk.wcond = take(8 * DS_ * DS_).view(8 * DS_, DS_)
            pk.wp = take(4 * DS_ * DS_).view(4 * DS_, DS_)
            pk.wu = take(4 * DS_ * DS_).view(4 * DS_, DS_)
            pk.wo = take(DS_ * DS_).view(DS_, DS_)
            pk.ws = take(2 * DS_ * DS_).view(DS_, 2 * DS_)
            pk.bcond = take(8 * DS_)
            pk.bp = take(4 * DS_)
            pk.wq, pk.wk = take(32), take(32)
            assert off[0] == flat.numel()
            pk.flat = flat
        return pk

    return _capture.param_forms(("local_dit_esm_tf32",), [*weights, *qkw], build, inputs_scope=True, capture_reads_eager=infer)


_ESM_BCOND32: dict = {}


def _esm_bcond32(device) -> torch.Tensor:
    """The fp32 tables' bias [1024]: 1 on the blocks of ``_ESM_ONES32`` (the identity-centred scales), else 0 -- a constant (no parameter)."""
    key = str(device)
    c = _ESM_BCOND32.get(key)
    if c is None:
        c = torch.zeros(8, DS_, device=device, dtype=F32)
        for i in range(8):
            if (_ESM_ONES32 >> i) & 1:
                c[i] = 1.0
        c = c.view(-1)
        if not torch.cuda.is_current_stream_capturing():
            _ESM_BCOND32[key] = c
    return c


def _esm_train_packs32(weights, qkw):
    """The fp32 training operands of ESMFold2's block in ONE launch (``b200_atom.esm_pack``, fp32 outputs on the TF32 grid; round 3 part 7,
    2026-11-17): (pk, bk) as ``_esm_pack32`` / ``_esm_pack32_bwd`` give them -- rows of 128 fp32 of one buffer: wcond (``_ESM_COND_BWD``) at
    0, wp = [Wq; Wg; Wk; Wv] at 1024, wu = [Wa_0; Wb_0; Wa_1; Wb_1] at 1536, wo at 2048, ws [128, 256] at 2176; transposed: ws^T at 0,
    wu^T [128, 512] at 256, wo^T at 768, wp^T [128, 512] at 896, wcond^T [128, 1024] at 1408. Scoped to the capture (a step records it)."""
    named: dict[str, torch.Tensor] = dict(zip((*WEIGHTS, *CROSS_WEIGHTS), weights, strict=True))
    A_, T_ = "attention.", "transition."
    leaves = [*_ESM_COND_BWD, *(A_ + f"to_{p}.weight" for p in ("query", "gate", "key", "value")), T_ + "expand_a.weight",
              T_ + "expand_b.weight", A_ + "to_out.weight", T_ + "squeeze.weight", A_ + "to_query.bias"]  # fmt: skip
    ws = [named[k] for k in leaves]

    def build():
        from foldjet_engine.kernels.augmented_attention.cuda import b200_atom as rows

        w = {k: t.detach() for k, t in zip(leaves, ws, strict=True)}
        R = DS_ * DS_
        units = []

        def unit(src, row0=0, sp=DS_, d0=0, p0=DS_, dt=-1, pt=DS_):
            units.append((src, row0, sp, d0, p0, -1, DS_, dt, pt))

        for i, name in enumerate(_ESM_COND_BWD):
            unit(w[name], d0=i * R, dt=1408 * DS_ + 128 * i, pt=8 * DS_)
        for i, p in enumerate(("query", "gate", "key", "value")):
            unit(w[A_ + f"to_{p}.weight"], d0=(1024 + 128 * i) * DS_, dt=896 * DS_ + 128 * i, pt=4 * DS_)
        for i, (name, h) in enumerate((("expand_a", 0), ("expand_b", 0), ("expand_a", 1), ("expand_b", 1))):  # (interleaved halves)
            unit(w[T_ + name + ".weight"], row0=h * R, d0=(1536 + 128 * i) * DS_, dt=256 * DS_ + 128 * i, pt=4 * DS_)
        unit(w[A_ + "to_out.weight"], d0=2048 * DS_, dt=768 * DS_)
        for c in range(2):
            unit(w[T_ + "squeeze.weight"], row0=128 * c, sp=2 * DS_, d0=2176 * DS_ + 128 * c, p0=2 * DS_, dt=128 * c * DS_)
        with torch.no_grad():
            bf, bt, vec = rows.esm_pack(units, [w[A_ + "to_query.bias"], None, None, None], 2432 * DS_, 2432 * DS_, True)
        assert bt is not None

        def mat(buf, row0, nrow, shape):
            return buf[row0 * DS_ : (row0 + nrow) * DS_].view(shape)

        dev = bf.device
        pk = SimpleNamespace(
            wcond=mat(bf, 0, 1024, (8 * DS_, DS_)),
            wp=mat(bf, 1024, 512, (4 * DS_, DS_)),
            wu=mat(bf, 1536, 512, (4 * DS_, DS_)),
            wo=mat(bf, 2048, 128, (DS_, DS_)),
            ws=mat(bf, 2176, 256, (DS_, 2 * DS_)),
            bcond=_esm_bcond32(dev),
            bp=vec.view(-1),
            wq=qkw[0].detach().float().contiguous(),
            wk=qkw[1].detach().float().contiguous(),
        )
        bk = SimpleNamespace(
            wsT=mat(bt, 0, 256, (2 * DS_, DS_)),
            wuT=mat(bt, 256, 512, (DS_, 4 * DS_)),
            woT=mat(bt, 768, 128, (DS_, DS_)),
            wpT=mat(bt, 896, 512, (DS_, 4 * DS_)),
            wcondT=mat(bt, 1408, 1024, (DS_, 8 * DS_)),
        )
        return pk, bk

    return _capture.param_forms(("local_dit_esm_b200_train32",), ws, build)


_ESM_SAVED32 = ("tab", "x1", "xkv", "p", "qn", "kn", "bias2", "o", "lse", "gated", "y", "a1", "x2", "u", "t", "sc")


def _esm_fwd32_fake(single, cond, pair, mask, weights, qkw, cos, sin, save):
    out = torch.empty_like(single)
    if not save:
        return [out]
    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    m = a * n
    widths = {"tab": 8 * DS_, "p": 4 * DS_, "u": 4 * DS_}
    shapes = {"bias2": (NH_, n // QUERIES, QUERIES, 160), "lse": (a, NH_, n), "tab": (cond.shape[0] * n, 8 * DS_), "sc": (cond.shape[0] * n, DC_)}
    return [out, *(single.new_empty(shapes[k] if k in shapes else (m, widths.get(k, DS_))) for k in _ESM_SAVED32)]


#: fp32 inference (round 3 part 12): the pair bias as the programmatic dependent of pre32, beside it (``esm.esm_bias_early``; it ran
#: serially between the table GEMM and pre32: 5.9 us at 1024 atoms, 8.2 at 2048)
_ESM_EB32 = os.environ.get("MW_LD12_EE", "1") != "0"
#: ... up to this many rows of one sample (any with several): above, one sample's pre32 holds every SM and the bias beside it
#: rides its critical path (us: 7168 atoms 92.2 -> 98.3, 8192: 98.3 -> 104.4; 5120: 86.0 -> 79.9; 8192 x 5: 286.7 -> 278.5)
_ESM_EB32_ROWS = int(os.environ.get("MW_LD12_EER", "6144"))


@opaque(fake=_esm_fwd32_fake, name="local_dit_esm_b200_fwd_tf32")
def _esm_fwd32(
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    weights: list[torch.Tensor],
    qkw: list[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    save: bool,
) -> list[torch.Tensor]:
    """ESMFold2's block on the fp32 path: [out [A, 1, N, 128] fp32] and, when ``save``, the activations the backward reads (``_ESM_SAVED32``)."""
    from foldjet_engine.kernels.augmented_attention.cuda import b200_atom as rows
    from foldjet_engine.kernels.augmented_attention.cuda.b200_atom_local import esm as local

    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    m = a * n
    dev = single.device
    with torch.cuda.device(dev):
        pk = _esm_train_packs32(weights, qkw)[0] if save else _esm_pack32(weights, qkw, True)
        s2 = _padded(single, n).reshape(m, DS_).contiguous()
        wkey = tuple((w.data_ptr(), w._version) for w in weights)
        sc_ = []

        def tables():  # (training with a shared conditioning: [N, 1024], read at row r % N)
            c2 = _padded(cond, n).reshape(cond.shape[0] * n, DC_).contiguous()
            sc_.append(rows.silu32(c2))  # (training: saved for the tables' weight gradients)
            return rows.gemm32(sc_[0], pk.wcond, bias=pk.bcond)

        tab = _hoisted(_HOIST_COND, cond, ("esm32", wkey, n), tables) if (_hoisting() and not save) else tables()
        keys = _keys(mask, n0, n)

        def bias():
            return local.esm_bias(_trunk(pair, n), weights[_GAMMA], weights[_WBIAS], keys, n0)

        jb = None
        launch = None
        if not save and _ESM_EB32 and (m <= _ESM_EB32_ROWS or a > 1):  # round 3 part 12: beside pre32 (its PDL dependent)
            st: dict = {"defer": True}

            def bias_e():
                if st["defer"]:
                    b, st["launch"] = local.esm_bias_early(_trunk(pair, n), weights[_GAMMA], weights[_WBIAS], keys, n0)
                    return b
                return bias()

            mk = None if mask is None else _tkey(mask)
            key = ("esm32", wkey[_GAMMA], wkey[_WBIAS], n, mk)
            bias2 = _hoisted(_HOIST_BIAS, pair, key, bias_e, () if mask is None else (mask,)) if _hoisting() else bias_e()
            st["defer"] = False
            launch = st.pop("launch", None)
        elif _hoisting() and not save:
            mk = None if mask is None else _tkey(mask)
            bias2 = _hoisted(_HOIST_BIAS, pair, ("esm32", wkey[_GAMMA], wkey[_WBIAS], n, mk), bias, () if mask is None else (mask,))
        else:
            bias2, jb = _fork(dev, m, bias)
        if not save:  # inference: the fused fp32 row stages (``-DRMS=1`` builds): pre (AdaLNs + projections), attention, post (the rest)
            p = rows.pre32(
                s2,
                tab[:, _S1],
                tab[:, _B1],
                pk.wp,
                pk.bp[:DS_],
                0b1000,
                mks=tab[:, 6 * DS_ : 7 * DS_],
                mkb=tab[:, 7 * DS_ :],
                eps=RMS_EPS,
                parallel=True,
                rms=True,
            )  # P = [q | g | k | v], v tf32-rounded
            if launch is not None:
                launch()
            # (the q / k norm + rotary inside atom_pre_tf32's epilogue was measured slower, round 3 part 4: 3072 atoms 71.7 -> 86.0 us with
            # the silu GEMM A phase; the pre kernel alone 13.6 -> 18.5 us against this 2.8 us launch)
            qn, kn = local.esm_qkrope32(p[:, 0:DS_], p[:, 2 * DS_ : 3 * DS_], pk.wq, pk.wk, cos, sin, RMS_EPS)
            o = local.attn_esm32(qn.view(a, n, DS_), kn.view(a, n, DS_), p.view(a, n, 4 * DS_)[..., 3 * DS_ :], bias2)
            out = rows.post32(
                o.view(m, DS_),
                p[:, DS_ : 2 * DS_],
                s2,
                tab[:, _SO],
                tab[:, _S2],
                tab[:, _B2],
                tab[:, _ST],
                pk.wo,
                pk.wu,
                pk.ws,
                eps=RMS_EPS,
                rms=True,
            )
            return [_unpadded(out.view(a, 1, n, DS_), single.shape)]
        p = torch.empty(m, 4 * DS_, device=dev, dtype=F32)

        def kv():  # the K / V AdaLN of the raw atoms and k | v (v rounded): beside q | g on the second side stream
            xkv = rows.adaln32(s2, tab[:, 6 * DS_ : 7 * DS_], tab[:, 7 * DS_ :], eps=RMS_EPS, rms=True)
            rows.gemm32(xkv, pk.wp, p, n=2 * DS_, b_row0=2 * DS_, ocol0=2 * DS_, rndmask=0b1000)
            return xkv

        xkv, jk = _fork(dev, m, kv, second=True)
        x1 = rows.adaln32(s2, tab[:, _S1], tab[:, _B1], eps=RMS_EPS, rms=True)
        rows.gemm32(x1, pk.wp, p, n=2 * DS_, bias=pk.bp)  # q | g
        _join(jk)
        qn, kn = local.esm_qkrope32(p[:, 0:DS_], p[:, 2 * DS_ : 3 * DS_], pk.wq, pk.wk, cos, sin, RMS_EPS)
        p3 = p.view(a, n, 4 * DS_)
        lse = torch.empty(a, NH_, n, device=dev, dtype=F32) if save else None
        _join(jb)
        o = local.attn_esm32(qn.view(a, n, DS_), kn.view(a, n, DS_), p3[..., 3 * DS_ :], bias2, lse=lse)
        gated = rows.gate32(p[:, DS_ : 2 * DS_], o.view(m, DS_))
        y = torch.empty(m, DS_, device=dev, dtype=F32) if save else None
        a1 = rows.gemm32(gated, pk.wo, mode=rows.RESID_GATE, res=s2, gate=tab[:, _SO], out2=y)
        x2 = rows.adaln32(a1, tab[:, _S2], tab[:, _B2], eps=RMS_EPS, rms=True)
        u = torch.empty(m, 4 * DS_, device=dev, dtype=F32) if save else None
        h = rows.gemm32(x2, pk.wu, mode=rows.SWIGLU, out2=u)
        t = torch.empty(m, DS_, device=dev, dtype=F32) if save else None
        out = rows.gemm32(h, pk.ws, mode=rows.RESID_GATE, res=a1, gate=tab[:, _ST], out2=t)
    out = _unpadded(out.view(a, 1, n, DS_), single.shape)
    if not save:
        return [out]
    assert lse is not None and y is not None and u is not None and t is not None  # (``save``: allocated above)
    return [out, tab, x1, xkv, p, qn, kn, bias2, o.view(m, DS_), lse, gated, y, a1, x2, u, t, sc_[0]]


def _esm_pack32_bwd(weights, qkw):
    """The fp32 backward's transposed operands of ESMFold2's block (tf32-rounded as the forward pack): wsT [256, 128], wuT [128, 512],
    woT, wpT [128, 512], wcondT [128, 1024]."""
    pk = _esm_pack32(weights, qkw)

    def build():
        with torch.no_grad():

            def tr(x):
                return x.t().contiguous()

            return SimpleNamespace(wsT=tr(pk.ws), wuT=tr(pk.wu), woT=tr(pk.wo), wpT=tr(pk.wp), wcondT=tr(pk.wcond))

    return _capture.param_forms(("local_dit_esm_tf32_bwd",), [*weights, *qkw], build, inputs_scope=True)


def _esm_bwd32(dy, single, cond, pair, mask, weights, qkw, cos, sin, saved):
    """The fp32 backward of ESMFold2's block (``_EsmBlock32``): d single, d cond, d pair, the gradients of ``_ESM_GRAD`` and of the q / k norm
    weights (fp32)."""
    return _esm_bwd32_op(dy, single, cond, pair, mask, weights, qkw, cos, sin, saved)


def _esm_bwd32_fake(dy, single, cond, pair, mask, weights, qkw, cos, sin, saved):
    return _esm_bwd_fake(dy, single, cond, pair, mask, weights, [], cos, sin, saved)


@opaque(fake=_esm_bwd32_fake, name="local_dit_esm_b200_bwd_tf32")
def _esm_bwd32_op(
    dy: torch.Tensor,
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    weights: list[torch.Tensor],
    qkw: list[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    saved: list[torch.Tensor],
) -> list[torch.Tensor]:
    from foldjet_engine.kernels.augmented_attention.cuda import b200_atom as rows
    from foldjet_engine.kernels.augmented_attention.cuda.b200_atom_local import esm as local

    tab, x1, xkv, p, qn, kn, bias2, o, lse, gated, y, a1, x2, u, t, sc = saved
    a, _, n0, _ = single.shape
    n = _padded_length(n0)
    m = a * n
    dev = single.device
    with torch.cuda.device(dev):
        pk, bk = _esm_train_packs32(weights, qkw)
        s2 = _padded(single, n).reshape(m, DS_).contiguous()
        rc = cond.shape[0] * n  # the tables' rows: n when the samples share the conditioning
        c2 = _padded(cond, n).reshape(rc, DC_).contiguous()
        dy = _padded(dy, n).reshape(m, DS_).float().contiguous()
        dtab = torch.empty(
            rc, 8 * DS_, device=dev, dtype=F32
        )  # d [s1 | bi1 | s2 | bi2 | so | st | kv scale | kv shift] (summed over the samples)
        dt = rows.tail_bwd32(dy, t, tab[:, _ST], dtab[:, _ST], rms=True)
        dab, hh = rows.dh_swiglu_bwd32(dt, bk.wsT, u)  # (round 3 part 11: one launch, dh in TMEM)
        dx2 = rows.gemm32(dab, bk.wuT)
        da1, dyo = rows.adaln_bwd32(
            a1, tab[:, _S2], dx2, dy, dtab[:, _S2], dtab[:, _B2], y=y, so=tab[:, _SO], dso=dtab[:, _SO], eps=RMS_EPS, rms=True
        )
        dp = torch.empty(m, 4 * DS_, device=dev, dtype=F32)  # d [q | g | k | v]
        do, dd = _gated_bwd(dyo, bk.woT, p[:, DS_ : 2 * DS_], o, dp[:, DS_ : 2 * DS_], a, n)
        p3 = p.view(a, n, 4 * DS_)
        db2 = local.attn_esm_bwd32(
            qn.view(a, n, DS_), kn.view(a, n, DS_), p3[..., 3 * DS_ :], do.view(a, n, DS_), bias2, lse, dd, dp[:, :DS_], dp[:, DS_:]
        )  # dQ -> [:, 0:128], dK -> [:, 256:384], dV -> [:, 384:512]
        keys = _keys(mask, n0, n)  # (the pair bias' backward on a side stream; dx1 and d bq on the second)
        (dz, dgamma, dwb), jb = _fork(dev, m, lambda: local.esm_bias_bwd32(_trunk(pair, n), weights[_GAMMA], weights[_WBIAS], db2, keys, n0))
        local.esm_qkrope_bwd32(
            dp[:, :DS_], dp[:, 2 * DS_ : 3 * DS_], p[:, :DS_], p[:, 2 * DS_ : 3 * DS_], pk.wq, pk.wk, cos, sin, RMS_EPS, weights=False
        )  # (the q / k norm weights: constants, no gradient)
        (dx1, dbq), jx = _fork(dev, m, lambda: (rows.gemm32(dp, bk.wpT, k=2 * DS_), rows.colsum32(dp[:, :DS_])), second=True)  # [dq | dg] [Wq; Wg]
        dxkv = rows.gemm32(dp, bk.wpT, k=2 * DS_, a_col0=2 * DS_, b_col0=2 * DS_)  # [dk | dv] [Wk; Wv]
        dres, _ = rows.adaln_bwd32(
            s2, tab[:, 6 * DS_ : 7 * DS_], dxkv, da1, dtab[:, 6 * DS_ : 7 * DS_], dtab[:, 7 * DS_ :], eps=RMS_EPS, rms=True
        )  # the K / V AdaLN of the raw atoms (parallel)
        _join(jx)
        ds, _ = rows.adaln_bwd32(s2, tab[:, _S1], dx1, dres, dtab[:, _S1], dtab[:, _B1], eps=RMS_EPS, rms=True)
        # every table is a product of silu(c): K = 1024 (on the second side stream, beside the weight gradients)
        dc, jc = _fork(dev, m, lambda: rows.silu_bwd32(c2, rows.gemm32(dtab, bk.wcondT)), second=True)
        named: dict[str, torch.Tensor] = dict(zip((*WEIGHTS, *CROSS_WEIGHTS), weights, strict=True))
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
            (dtab[:, : 4 * DS_], sc, [(name, 0) for name in _ESM_COND_BWD[:4]]),
            (dtab[:, 4 * DS_ :], sc, [(name, 0) for name in _ESM_COND_BWD[4:]]),
            (dp[:, : 2 * DS_], x1, [(A_ + "to_query.weight", 0), (A_ + "to_gate.weight", 0)]),
            (dp[:, 2 * DS_ :], xkv, [(A_ + "to_key.weight", 0), (A_ + "to_value.weight", 0)]),
        ]
        if rc != m:  # shared conditioning: the tables' products have n rows, their own launch
            fused = _fused_wgrads(spec[:3] + spec[5:], weights, WEIGHTS + CROSS_WEIGHTS, True, dev, m)
            fused.update(_fused_wgrads(spec[3:5], weights, WEIGHTS + CROSS_WEIGHTS, True, dev, rc))
        else:
            fused = _fused_wgrads(spec, weights, WEIGHTS + CROSS_WEIGHTS, True, dev, m)
        _join(jb, jc)
    extra = {"attention.to_query.bias": dbq, "attention.ln_pair.weight": dgamma, "attention.to_bias.weight": dwb}
    grads = [fused[name] if name in fused else _fresh(extra[name], named[name]) for name in _ESM_GRAD]
    return [
        _unpadded(ds.view(a, 1, n, DS_), single.shape),
        _unpadded(dc.view(cond.shape[0], 1, n, DC_), cond.shape),
        _dpair(dz, pair),
        *grads,
    ]


class _EsmBlock32(torch.autograd.Function):
    """ESMFold2's block forward and backward on the fp32 path (TF32 tensor cores); the constant slots of ``_esm_weights`` get None."""

    @staticmethod
    def forward(ctx, mask, cos, sin, single, cond, pair, *params):
        nw = len(WEIGHTS) + len(CROSS_WEIGHTS)
        out, *saved = _esm_fwd32(single, cond, pair, mask, list(params[:nw]), list(params[nw:]), cos, sin, True)
        ctx.save_for_backward(mask, cos, sin, single, cond, pair, *saved, *params)
        ctx.n_saved = len(saved)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        mask, cos, sin, single, cond, pair, *rest = ctx.saved_tensors
        saved, params = rest[: ctx.n_saved], rest[ctx.n_saved :]
        nw = len(WEIGHTS) + len(CROSS_WEIGHTS)
        ds, dc, dpair, *g = _esm_bwd32(dy, single, cond, pair, mask, list(params[:nw]), list(params[nw:]), cos, sin, list(saved))
        it = iter(g)
        out = [None if (name in _ESM_ONES or name in _ESM_ZEROS) else next(it) for name in (*WEIGHTS, *CROSS_WEIGHTS)]
        return (None, None, None, ds, dc, dpair, *out, None, None)  # (the q / k norm weights: constants)


def _table_sums(a, n, cross, dev):
    """``b200_atom_local.TableSums``: the shared conditioning's table gradients summed over the samples in the row kernels."""
    from foldjet_engine.kernels.augmented_attention.cuda.b200_atom_local import TableSums

    return TableSums(a, n, cross, dev)


def reason(module, single, cond, pair, mask=None, rope=None) -> str | None:
    """Why the B200 tcgen05 ESMFold2 block does not take this call, or None: ``local_dit_esm.reason``'s contract (geometry, dtypes, one fold,
    the [1, N, 16] rotary tables without gradient) on a B200."""
    from foldjet_engine.kernels.integrations import local_dit_esm

    if not (single.is_cuda and torch.cuda.get_device_capability(single.device) == (10, 0)):
        return "the tcgen05 ESMFold2 atom block runs on a B200 (sm_100a) CUDA tensor only"
    why = local_dit_esm.reason(module, single, cond, pair, mask, rope)
    if why is not None:
        return why
    if not _route_training() and torch.is_grad_enabled() and any(t.requires_grad for t in (single, cond, pair, *module.parameters())):
        return "training: the tcgen05 ESMFold2 block is routed for inference only (``ROUTE_TRAINING``)"
    return None
