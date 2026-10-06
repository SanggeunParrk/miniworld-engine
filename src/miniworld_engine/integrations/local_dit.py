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

``serves()`` is the whole gate: B200, the engine's kernel backend (implementation MINIWORLD or TRITON), bf16 single / cond / pair,
the atom widths (d_single = d_cond = 128, 4 heads x 32, d_pair = 16, transition n = 2, no QK-norm), B == 1, any N, a [B, N] bool key
mask or none, LayerNorm eps 1e-5. N is padded to a multiple of 128 for the row kernels (zero single / cond rows, the padded atoms
masked as keys, the pair padded with zero windows) and the first N rows come back. ``MINIWORLD_LOCAL_DIT_SM100=0`` turns it off. A
build or load failure warns once and keeps the module path.

The block is two opaque ops, ``local_dit_block_fwd`` and ``local_dit_block_bwd`` (``kernels._compile.opaque``), so a ``torch.compile``d
model keeps them in its graph. The ops take the 23 parameters as a list (``WEIGHTS``); packing them for the kernels is reused while
every weight's tensor object and version are unchanged, except while a CUDA graph is captured. Parameters may be bf16 or fp32; their
gradients come back in their own dtype.
"""

from __future__ import annotations

import os
import warnings

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from miniworld_engine import settings
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
        hopper = rows._hopper(index)
        for name in (("cond", "pre", "post", "pack") if hopper else _ROW_KERNELS):
            rows.atom_kernel(name, index)
        for name in (("fwd", "pb_f", "cond_ln", "kv_f", "kv_pack") if hopper else local.KERNELS):
            local._load(name, index)
    except Exception as exc:  # a toolchain or driver problem keeps the module path
        if rows._hopper(index):
            raise RuntimeError("Hopper local DiT kernel build failed") from exc
        _FAILED = True
        warnings.warn(f"sm_100a local atom DiT kernels unavailable, keeping the module path: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    _LOADED.add(index)
    return True


def serves(module, single, cond, pair, mask) -> bool:
    qualified = _serves_hopper_training(module, single, cond, pair, mask)
    if qualified is not None:
        return qualified
    qualified = _serves_hopper_inference(module, single, cond, pair, mask)
    if qualified is not None:
        return qualified
    if _FAILED or os.environ.get("MINIWORLD_LOCAL_DIT_SM100", "1") == "0" or settings.current().engine_backend == "triton":
        return False
    from miniworld_engine.modules.dispatch import KernelBackend

    a, tr = module.attention, module.transition
    if a._backend != KernelBackend.TRITON or a.use_qk_norm:
        return False
    cap = torch.cuda.get_device_capability(single.device) if single.is_cuda else None
    hopper = cap == (9, 0) and os.environ.get("MINIWORLD_LOCAL_DIT_H100", "0") == "1"
    if hopper and torch.is_grad_enabled() and os.environ.get("MINIWORLD_LOCAL_DIT_H100_TRAIN", "0") != "1":
        return False
    if not (cap == (10, 0) or hopper):
        return False
    if (single.dtype, cond.dtype, pair.dtype) != (BF, BF, BF):
        return False
    if single.ndim != 4 or single.shape[1] != 1 or single.shape[-1] != DS_ or single.shape[2] == 0:
        return False
    n = single.shape[2]
    if hopper and torch.is_grad_enabled() and (n not in (4096, 8192) or not all(t.is_contiguous() for t in (single, cond, pair))):
        return False
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


def _serves_hopper_training(module, single, cond, pair, mask):
    """Required Hopper training geometry: native CUDA or an explicit error."""
    from miniworld_engine.modules.dispatch import KernelBackend

    a, tr = module.attention, module.transition
    if (a._backend != KernelBackend.TRITON or not torch.is_grad_enabled()
            or not single.is_cuda or torch.cuda.get_device_capability(single.device) != (9, 0)):
        return None
    # Preserve the explicit diagnostic comparison switch, outside production.
    if os.environ.get("MINIWORLD_LOCAL_DIT_H100_TRAIN", "1") == "0":
        return False

    def require(ok, detail):
        if not ok:
            raise NotImplementedError(f"H100 local atom DiT training requires {detail}; no composed fallback")

    require(settings.current().engine_backend != "triton", "the CUDA engine backend")
    require(module.cross_attention and not a.use_qk_norm, "cross-attention without QK norm")
    require(single.ndim == 4 and single.shape[0] in (1, 5, 48) and single.shape[1] == 1
            and single.shape[2] in (4096, 8192) and single.shape[3] == DS_,
            "[A,1,N,128] input with A=1/5/48 and N=4096/8192")
    augment, _, n, _ = single.shape
    require(tuple(cond.shape) == (augment, 1, n, DC_)
            and tuple(pair.shape) == (1, n // QUERIES, QUERIES, KEYS, DP_),
            "canonical conditioning and trunked pair shapes")
    require(all(t.dtype == BF and t.device == single.device and t.is_contiguous() for t in (single, cond, pair)),
            "contiguous bf16 activations on one CUDA device")
    require((a.n_head, a.to_query.weight.shape[0], a.to_query.weight.shape[1],
             a.to_bias.weight.shape[1], a.ada_ln_in.to_scale.weight.shape[1], tr.expand_a.weight.shape[0])
            == (NH_, DS_, DS_, DP_, DC_, 2 * DS_), "128/128/16 widths, four heads, transition n=2")
    weights = tuple(module.get_parameter(name) for name in (*WEIGHTS, *CROSS_WEIGHTS))
    require(all(w.dtype in (BF, torch.float32) and w.device == single.device and w.is_contiguous() for w in weights),
            "contiguous bf16 or fp32 parameters on the input device")
    require(mask is None or (mask.dtype == torch.bool and tuple(mask.shape) == (1, n)
            and mask.device == single.device and mask.is_contiguous()), "a contiguous [1,N] bool mask or None")
    norms = (a.ada_ln_in.ln_in, a.ada_ln_in.ln_cond, a.ln_pair, tr.ada_ln_in.ln_in,
             tr.ada_ln_in.ln_cond, a.ada_ln_kv.ln_in, a.ada_ln_kv.ln_cond)
    require(all(norm.eps == EPS for norm in norms), "LayerNorm eps=1e-5")
    index = single.device.index if single.device.index is not None else torch.cuda.current_device()
    if not _kernels_ready(index):
        raise RuntimeError("H100 local atom DiT CUDA kernels unavailable; no composed fallback")
    return True


def _serves_hopper_inference(module, single, cond, pair, mask):
    """The required inference cells use native CUDA, including recorded losses."""
    from miniworld_engine.modules.dispatch import KernelBackend

    a, tr = module.attention, module.transition
    if a._backend != KernelBackend.TRITON or torch.is_grad_enabled() or not single.is_cuda:
        return None
    if torch.cuda.get_device_capability(single.device) != (9, 0):
        return None
    def require(ok, detail):
        if not ok:
            raise NotImplementedError(f"H100 local atom DiT inference requires {detail}; no composed fallback")

    require(settings.current().engine_backend != "triton", "the CUDA engine backend")
    require(module.cross_attention and not a.use_qk_norm, "cross-attention without QK norm")
    require(single.ndim == 4 and (single.shape[0], single.shape[2])
            in ((1, 4096), (5, 2048), (5, 4096), (48, 4096)),
            "(A,N)=(1,4096),(5,2048),(5,4096),(48,4096)")
    augment, _, n, _ = single.shape
    require((a.n_head, a.to_query.weight.shape[0], a.to_query.weight.shape[1],
             a.to_bias.weight.shape[1], a.ada_ln_in.to_scale.weight.shape[1], tr.expand_a.weight.shape[0])
            == (NH_, DS_, DS_, DP_, DC_, 2 * DS_), "128/128/16 widths, four heads, transition n=2")
    require(tuple(single.shape) == (augment, 1, n, DS_) and tuple(cond.shape) == (augment, 1, n, DC_)
            and tuple(pair.shape) == (1, n // QUERIES, QUERIES, KEYS, DP_), "canonical single, conditioning and trunked pair shapes")
    require(all(t.dtype == BF and t.device == single.device and t.is_contiguous() for t in (single, cond, pair)),
            "contiguous bf16 activations on one CUDA device")
    weights = tuple(module.get_parameter(name) for name in (*WEIGHTS, *CROSS_WEIGHTS))
    require(all(w.dtype in (BF, torch.float32) and w.device == single.device and w.is_contiguous() for w in weights),
            "contiguous bf16 or fp32 parameters on the input device")
    require(mask is None or (mask.dtype == torch.bool and tuple(mask.shape) == (1, n)
            and mask.device == single.device and mask.is_contiguous()), "a contiguous [1,N] bool mask or None")
    norms = (a.ada_ln_in.ln_in, a.ada_ln_in.ln_cond, a.ln_pair, tr.ada_ln_in.ln_in,
             tr.ada_ln_in.ln_cond, a.ada_ln_kv.ln_in, a.ada_ln_kv.ln_cond)
    require(all(norm.eps == EPS for norm in norms), "LayerNorm eps=1e-5")
    index = single.device.index if single.device.index is not None else torch.cuda.current_device()
    if not _kernels_ready(index):
        raise RuntimeError("H100 local atom DiT CUDA kernels unavailable; no composed fallback")
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


def _kv_pack(weights, packed_pre=None):
    """The cross-attention mode's operands: Wm [256, 128] = [to_scale; to_bias] bf16, the scale bias and the LN weight (fp32), Wk, Wv (bf16)."""
    gamma, w_scale, b_scale, w_bias = weights[len(WEIGHTS):]
    if packed_pre is not None and torch.cuda.get_device_capability(gamma.device) == (9, 0):
        from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom_local as local
        return local.kv_pack_hopper(weights[len(WEIGHTS):], packed_pre)
    wk, wv = weights[WEIGHTS.index("attention.to_key.weight")], weights[WEIGHTS.index("attention.to_value.weight")]
    return torch.cat([w_scale, w_bias]).to(BF), b_scale.float(), gamma.float(), wk.to(BF), wv.to(BF)


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
        packed_infer = (not save and cross and a == 5 and n0 in (2048, 4096)
                        and rows._hopper(single.device.index))
        wc, wp, wo, _ = _weights(weights[: len(WEIGHTS)], backward=save)
        s2 = _padded(single, n).reshape(m, DS_).contiguous()
        c2 = _padded(cond, n).reshape(m, DC_).contiguous()
        mod = rows.cond_fwd(c2, *wc, hopper_waves=.5 if packed_infer else 2.)
        q, k, v, sg, x1 = rows.pre_fwd(s2, mod, *wp, save=save or cross, cross=cross and rows._hopper(single.device.index))
        if cross:       # keys and values from a second AdaLN of x1 (pre_fwd's own k / v are not used)
            wm, bs, gkv, wk, wv = _kv_pack(weights, wp)
            cn = local.cond_ln(c2, gkv)
            mkv = cn @ wm.t()
            xkv = local.kv_fwd(x1, mkv, bs)
            k, v = xkv @ wk.t(), xkv @ wv.t()
        bias = local.pair_bias_fwd(_trunk(pair, n), weights[_GAMMA], weights[_WBIAS])
        o, lse = local.attn_fwd(q.view(a, n, DS_), k.view(a, n, DS_), v.view(a, n, DS_), bias, _keys(mask, n0, n))
        out, u, a2, x2, t = rows.post_fwd(s2, o.view(m, DS_), sg, mod, *wo, save=save,
                                        packed_hopper=packed_infer)
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
        hopper = rows._hopper(dev.index)
        if hopper and (dy.dtype != BF or dy.device != dev or dy.shape != single.shape or not dy.is_contiguous()):
            raise NotImplementedError("H100 local DiT backward requires a contiguous bf16 upstream gradient matching single")
        wc, wp, wo, wb = _weights(weights[: len(WEIGHTS)], backward=True)
        s2 = _padded(single, n).reshape(m, DS_).contiguous()
        c2 = _padded(cond, n).reshape(m, DC_).contiguous()
        dy = _padded(dy, n).reshape(m, DS_).to(BF).contiguous()
        dmod = torch.empty(m, 6 * DS_, device=dev, dtype=BF)         # [dsc1 | dbi1 | dsc2 | dbi2 | dos | dts]
        dp = torch.empty(m, 4 * DS_, device=dev, dtype=BF)           # [dq | dk | dv | dg]
        allocate = torch.empty if hopper else torch.zeros
        dbias_cols = allocate(6 * DS_, device=dev)
        dbq = allocate(DS_, device=dev)
        dgamma_cond = allocate(2 * DS_, device=dev)
        dgparts = torch.empty((m + 63) // 64, 2 * DS_, device=dev) if hopper else dgamma_cond
        dd = torch.empty(a, NH_, n, device=dev)
        dt, hh, dab = rows.tr_bwd_gate(dy, t, mod, x2, wo[1], wb["WsT"], dmod, dbias_cols[640:])
        da, du, gated, do = rows.post_bwd(dab, dy, a2, mod, u, sg, o, wb["WuT"], wb["WoT"], dmod, dp, dd, dbias_cols, n)
        db = local.attn_bwd(q.view(a, n, DS_), k.view(a, n, DS_), v.view(a, n, DS_), do.view(a, n, DS_), bias, lse, dd, dp,
                            _keys(mask, n0, n))
        dz, dgamma, dwb = local.pair_bias_bwd(_trunk(pair, n), weights[_GAMMA], weights[_WBIAS], db)
        if cross:   # the K / V branch: dxkv -> its AdaLN -> dx1 (through AdaLN 1 below) and the conditioning
            cn, mkv, xkv = extra
            wm, bs, gkv, wk, wv = _kv_pack(weights, wp if hopper else None)
            dk, dv = dp[:, DS_:2 * DS_], dp[:, 2 * DS_:3 * DS_]
            dxkv = torch.addmm(dk @ wk, dv, wv)
            dwk, dwv = _wgrad(dk, xkv), _wgrad(dv, xkv)
            dx1_kv, dmkv = local.kv_bwd(x1, mkv, bs, dxkv)
            dwm = _wgrad(dmkv, cn)
            dbs = local.kv_bias_hopper(dmkv) if hopper else dmkv[:, :DS_].float().sum(0)
            dg_kv = allocate(DS_, device=dev)
            dc_kv = local.cond_ln_bwd(c2, dmkv @ wm, gkv, dg_kv)
            if hopper:
                local.zero_kv_hopper(dp)
            else:
                dp[:, DS_:3 * DS_].zero_()  # pre_bwd's dgrad then covers q and the gate only
        ds = rows.pre_bwd(dp, s2, mod, da, wb["WpT"], dmod, dbias_cols, dbq)
        if cross:
            local.adaln1_extra(s2, mod, dx1_kv, ds, dmod, dbias_cols[0:DS_])
        dc, cn1, cn2 = rows.cond_bwd(dmod, c2, wb["WmodT"], wc[2], wc[3], dgparts)
        if cross:
            if hopper:
                local.add_cond_hopper(dc, dc_kv)
            else:
                dc = dc + dc_kv
        if hopper:
            rows.finish_columns_hopper(dmod, dp, dgparts, dbias_cols, dbq, dgamma_cond)
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
    if hopper:
        per_weight = rows.grad_pack_hopper([grads[name] for name in WEIGHTS], weights[:len(WEIGHTS)])
        if cross:
            per_weight += local.cross_grad_pack_hopper([grads[name] for name in CROSS_WEIGHTS], weights[len(WEIGHTS):])
        dpair = dz.view(pair.shape)
    else:
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
        if single.is_cuda and torch.cuda.get_device_capability(single.device) == (9, 0):
            from miniworld_engine.integrations.token_dit_train import _activate_native_caller
            _activate_native_caller(force=True)
        return (None, *grads)


def _infer(single, cond, pair, mask, weights):
    return _block_fwd(single, cond, pair, mask, list(weights), False)[0]


def block(module, single, cond, pair, mask=None):
    """The block's output stream (both residuals included), on the sm_100a kernels; ``serves()`` must have accepted the call."""
    weights = [module.get_parameter(name) for name in (*WEIGHTS, *(CROSS_WEIGHTS if module.cross_attention else ()))]
    if torch.is_grad_enabled() and any(t.requires_grad for t in (single, cond, pair, *weights)):
        out = _LocalBlock.apply(mask, single, cond, pair, *weights)
        if single.is_cuda and torch.cuda.get_device_capability(single.device) == (9, 0):
            from miniworld_engine.integrations.token_dit_train import _native_caller_hooks
            _native_caller_hooks(single, cond, pair, *weights, force=True)
        return out
    return _infer(single, cond, pair, mask, weights)
