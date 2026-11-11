"""The AF3 atom transformer block with block-local attention on A100 (sm_80): inference and training.

One ``LocalDiTBlock`` call runs, in place of the module path,

    forward   cond_fwd -> pre_fwd -> pair bias (trunked) -> local attn_fwd -> post_fwd
    backward  tr_bwd -> post_bwd -> local attn_dq / attn_dkv (+ dbias) -> pair-bias backward -> pre_bwd -> cond_bwd,
              then the weight gradients as cuBLAS GEMMs on the activations those kernels write.

The row kernels (``kernels/augmented_attention/cuda/sm80_atom_rows``: conditioning, input projections, post-attention, transition, and their backward) are hand CUDA
(``mma.sync``); the attention and the pair bias are ``kernels/augmented_attention/cuda/sm80_atom_local``'s. The block is two opaque ops (``kernels._compile.opaque``) so a
``torch.compile``d model keeps them in its graph; both residuals are inside ``post_fwd`` (the block returns the stream). The twin of ``integrations/local_dit`` (B200) with the
same parameter order (``WEIGHTS`` / ``CROSS_WEIGHTS``) and the same activation save list.

``serves()`` is the whole gate: A100, the engine's kernel backend (implementation MINIWORLD or TRITON), bf16 single / cond / pair, the atom widths (d_single = d_cond = 128,
4 heads x 32, d_pair = 16, transition n = 2, no QK-norm), B == 1, any N, a [1, N] bool key mask or none, LayerNorm eps 1e-5.
Parameters may be bf16 or fp32 (an fp32 master under autocast); the kernels run on bf16 packs of them, packed once per set of weights (tensor objects and versions) and
scoped to a CUDA-graph capture; gradients come back in the parameter's dtype.
"""

from __future__ import annotations

import weakref
from types import SimpleNamespace
from typing import cast

import torch
from torch.autograd.function import once_differentiable

from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import device_constant, opaque
from miniworld_engine.kernels._nvcc import build_failed
from miniworld_engine.kernels.integrations.local_dit import (
    CROSS_WEIGHTS,
    KEYS,
    QUERIES,
    WEIGHTS,
)

BF = torch.bfloat16
F32 = torch.float32
EPS = 1e-5
DS_, DC_, DP_, NH_ = 128, 128, 16, 4  # the atom widths the kernels are built for
_GAMMA, _WBIAS = (
    WEIGHTS.index("attention.ln_pair.weight"),
    WEIGHTS.index("attention.to_bias.weight"),
)


@device_constant
def _ready(index: int) -> bool:
    """Build and load the row kernels and the local attention kernels on ``index`` once; a failed build / load raises ``_nvcc.KernelBuildError`` (never a fallback)."""
    try:
        from miniworld_engine.kernels.augmented_attention.cuda import (
            sm80_atom_local,
            sm80_atom_rows,
        )

        sm80_atom_rows.ready()
        sm80_atom_local.ready(index)
    except Exception as exc:  # a build / load failure is an error, never a fallback
        build_failed("sm_80 local atom DiT row", exc)
        return False  # reached only while planning (build_failed returns there)
    return True


def serves(module, single, cond, pair, mask) -> bool:
    a, tr = module.attention, module.transition
    if a.use_qk_norm:
        return False
    if not (single.is_cuda and torch.cuda.get_device_capability(single.device) == (8, 0)):
        return False
    if (single.dtype, cond.dtype, pair.dtype) != (BF, BF, BF):
        return False
    if single.ndim != 4 or single.shape[1] != 1 or single.shape[-1] != DS_ or single.shape[2] == 0:
        return False
    n = single.shape[2]
    if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != (1, n) or mask.device != single.device):
        return False
    if tuple(cond.shape) != (*single.shape[:3], DC_) or tuple(pair.shape) != (
        1,
        (n + QUERIES - 1) // QUERIES,
        QUERIES,
        KEYS,
        DP_,
    ):
        return False
    if cond.device != single.device or pair.device != single.device:
        return False
    if not pair.is_contiguous() or (mask is not None and not mask.is_contiguous()):
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
    return _ready(single.device.index if single.device.index is not None else torch.cuda.current_device())


# ------------------------------------------------------------------------------------------------------------ weights
_PACKS: dict = {}


def _pack(weights, cross):
    """The kernels' bf16 packs of the parameters: W_cond [1024, 128] = Wsc1 | Wbi1 | Wsckv | Wbikv | Wsc2 | Wbi2 | Wos | Wts (the order of the modulation columns), its bias [1024]
    and the three conditioning LayerNorm weights [3, 128] (fp32), W_pre [512, 128] = Wq | Wg | Wk | Wv with the query bias, Wo, Wu = Wa | Wb, Ws."""
    names = WEIGHTS + CROSS_WEIGHTS[: len(weights) - len(WEIGHTS)]
    w = dict(zip(names, weights, strict=True))
    dev = weights[0].device
    zero = torch.zeros(DS_, DC_, device=dev, dtype=BF)
    zb = torch.zeros(DS_, device=dev, dtype=F32)
    A, T, K = "attention.", "transition.", "attention.ada_ln_kv."

    def bf(name):
        return w[name].detach().to(BF)

    def fb(name):
        return w[name].detach().to(F32)

    with torch.no_grad():
        w_cond = torch.cat(
            [
                bf(A + "ada_ln_in.to_scale.weight"),
                bf(A + "ada_ln_in.to_bias.weight"),
                bf(K + "to_scale.weight") if cross else zero,
                bf(K + "to_bias.weight") if cross else zero,
                bf(T + "ada_ln_in.to_scale.weight"),
                bf(T + "ada_ln_in.to_bias.weight"),
                bf(A + "to_scale.weight"),
                bf(T + "to_scale.weight"),
            ]
        ).contiguous()
        b_cond = torch.cat(
            [
                fb(A + "ada_ln_in.to_scale.bias"),
                zb,
                fb(K + "to_scale.bias") if cross else zb,
                zb,
                fb(T + "ada_ln_in.to_scale.bias"),
                zb,
                fb(A + "to_scale.bias"),
                fb(T + "to_scale.bias"),
            ]
        ).contiguous()
        gamma = torch.stack(
            [
                fb(A + "ada_ln_in.ln_cond.weight"),
                fb(K + "ln_cond.weight") if cross else zb,
                fb(T + "ada_ln_in.ln_cond.weight"),
            ]
        ).contiguous()
        w_pre = torch.cat(
            [
                bf(A + "to_query.weight"),
                bf(A + "to_gate.weight"),
                bf(A + "to_key.weight"),
                bf(A + "to_value.weight"),
            ]
        ).contiguous()
        return SimpleNamespace(
            w_cond=w_cond,
            b_cond=b_cond,
            gamma=gamma,
            w_pre=w_pre,
            bq=fb(A + "to_query.bias").contiguous(),
            wo=bf(A + "to_out.weight").contiguous(),
            wu=torch.cat([bf(T + "expand_a.weight"), bf(T + "expand_b.weight")]).contiguous(),
            wd=bf(T + "squeeze.weight").contiguous(),
        )


def _weights(weights):
    """``_pack``, reused while every weight's tensor object and version are unchanged (scoped to the CUDA-graph capture: a capture packs once and records it)."""
    cross = len(weights) > len(WEIGHTS)
    try:
        versions = tuple(x._version for x in weights)
    except RuntimeError:  # inference tensors carry no version counter: nothing could invalidate an entry
        return _pack(weights, cross)
    key = _capture.scoped(tuple(id(x) for x in weights))
    if key is None:
        return _pack(weights, cross)
    hit = _PACKS.get(key)
    if hit is not None:
        refs, seen, pack = hit
        if seen == versions and all(r() is x for r, x in zip(refs, weights, strict=True)):
            return pack
    _capture.prune(_PACKS)
    if len(_PACKS) >= 64:  # a model's worth of blocks; drop the oldest
        _PACKS.pop(next(iter(_PACKS)))
    pack = _pack(weights, cross)
    _PACKS[key] = (tuple(weakref.ref(x) for x in weights), versions, pack)
    return pack


_PACKS_BWD: dict = {}


def _pack_bwd(weights, cross):
    """The backward kernels' transposed bf16 packs (rows = the output channel of the product the kernel runs): Ws^T [256, 128], Wu^T = [Wa^T | Wb^T] [128, 512], Wo^T, [Wq^T | Wg^T] and
    [Wk^T | Wv^T] [128, 256], and the conditioning weights' transposes [4, 128, 256] (per variant: its two [128, 128] matrices stacked on the units axis)."""
    names = WEIGHTS + CROSS_WEIGHTS[: len(weights) - len(WEIGHTS)]
    w = dict(zip(names, weights, strict=True))
    A, T = "attention.", "transition."

    def bt(name):
        return w[name].detach().to(BF).t()

    with torch.no_grad():
        pk = _weights(weights)
        return SimpleNamespace(
            wst=bt(T + "squeeze.weight").contiguous(),
            wut=torch.cat([bt(T + "expand_a.weight"), bt(T + "expand_b.weight")], 1).contiguous(),
            wot=bt(A + "to_out.weight").contiguous(),
            wqgt=torch.cat([bt(A + "to_query.weight"), bt(A + "to_gate.weight")], 1).contiguous(),
            wkvt=torch.cat([bt(A + "to_key.weight"), bt(A + "to_value.weight")], 1).contiguous(),
            wt=torch.stack([pk.w_cond[256 * i : 256 * (i + 1)].t().contiguous() for i in range(4)]).contiguous(),
        )


def _weights_bwd(weights):
    """``_pack_bwd``, reused like ``_weights``."""
    cross = len(weights) > len(WEIGHTS)
    try:
        versions = tuple(x._version for x in weights)
    except RuntimeError:
        return _pack_bwd(weights, cross)
    key = _capture.scoped(tuple(id(x) for x in weights))
    if key is None:
        return _pack_bwd(weights, cross)
    hit = _PACKS_BWD.get(key)
    if hit is not None:
        refs, seen, pack = hit
        if seen == versions and all(r() is x for r, x in zip(refs, weights, strict=True)):
            return pack
    _capture.prune(_PACKS_BWD)
    if len(_PACKS_BWD) >= 64:
        _PACKS_BWD.pop(next(iter(_PACKS_BWD)))
    pack = _pack_bwd(weights, cross)
    _PACKS_BWD[key] = (tuple(weakref.ref(x) for x in weights), versions, pack)
    return pack


_HOIST_COND: dict = {}  # (conditioning tensor, weights) -> mod: the per-item conditioning table of an inference call
_HOIST_BIAS: dict = {}  # (pair tensor, LN / bias weights) -> the windowed pair bias


def _tkey(t):
    """A tensor's identity for a cache key: storage address, in-place version, shape, dtype, device."""
    return (t.data_ptr(), t._version, tuple(t.shape), t.dtype, t.device)


def _hoisted(store, tensor, key, build):
    """The table ``build()`` makes from ``tensor``, once per tensor (pointer, version, ``key``): the entry keeps a weak reference to the tensor, so a new tensor that reuses a freed
    address is not served the old one's table, and entries of freed tensors are dropped (``kernels._capture.lookup_inputs``: under ``static_inputs()`` a capture serves from the eager
    entries and a replay runs none of the kernels that make them; otherwise the entry is scoped to the capture)."""
    entry = _capture.lookup_inputs(
        store,
        (_tkey(tensor), key),
        lambda: (weakref.ref(tensor), build()),
        limit=64,
        valid=lambda e: e[0]() is tensor,
        alive=lambda e: e[0]() is not None,
    )
    return entry[1]


def _keys(mask, n, device):
    """The valid-key bytes [n] the attention kernels read (an empty tensor: every key valid)."""
    if mask is None:
        return torch.empty(0, device=device, dtype=torch.bool)
    return mask.view(n)


# ------------------------------------------------------------------------------------------------------------ the forward op
def _fwd_fake(single, cond, pair, mask, weights, save):
    out = torch.empty_like(single)
    if not save:
        return [out]
    a, _, n, _ = single.shape
    m = a * n
    cross = len(weights) > len(WEIGHTS)

    def tile(width=DS_):
        return single.new_empty((m, width))

    return [
        out,
        tile(8 * DS_),  # mod
        tile(),  # q
        tile(),  # k
        tile(),  # v
        tile(),  # sg
        tile(),  # x1
        tile() if cross else single.new_empty(0),  # xkv
        single.new_empty((NH_, (n + QUERIES - 1) // QUERIES, QUERIES, KEYS), dtype=F32),  # bias
        tile(),  # o
        single.new_empty((a, NH_, n), dtype=F32),  # lse
        tile(),  # u
        tile(),  # a2
        tile(),  # x2
        tile(),  # t
    ]


@opaque(fake=_fwd_fake, name="local_rows_sm80_block_fwd")
def _block_fwd(
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    weights: list[torch.Tensor],
    save: bool,
) -> list[torch.Tensor]:
    """The block output [A, 1, N, 128] bf16 and, when ``save``, the activations the backward reads: mod, q, k, v, sg, x1, xkv, bias, O, LSE, u, a2, x2, t (fresh tensors)."""
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm80_atom_local as local,
    )
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm80_atom_rows as rows,
    )

    a, _, n, _ = single.shape
    m = a * n
    cross = len(weights) > len(WEIGHTS)
    with torch.cuda.device(single.device):
        pk = _weights(weights)
        s2 = single.reshape(m, DS_).contiguous()
        c2 = cond.reshape(m, DC_).contiguous()

        def make_mod():
            return rows.cond_fwd(c2, pk.w_cond, pk.b_cond, pk.gamma, EPS, cross)

        def make_bias():
            return local.pair_bias_fwd(pair[0], weights[_GAMMA], weights[_WBIAS])

        if save:
            mod, bias = make_mod(), make_bias()
        else:  # an inference call: both tables depend on one input tensor and the weights only (a sampler's steps share them)
            wkey = tuple((x.data_ptr(), x._version) for x in weights)
            mod = _hoisted(_HOIST_COND, cond, (wkey, cross), make_mod)
            bias = _hoisted(_HOIST_BIAS, pair, (wkey[_GAMMA], wkey[_WBIAS]), make_bias)
        pre = rows.pre_fwd(s2, mod, pk.w_pre, pk.bq, EPS, save, cross)
        q, k, v, sg = pre[:4]
        o, lse = local.attn_fwd(
            q.view(a, n, DS_),
            k.view(a, n, DS_),
            v.view(a, n, DS_),
            bias,
            _keys(mask, n, single.device),
        )
        post = rows.post_fwd(s2, o.view(m, DS_), sg, mod, pk.wo, pk.wu, pk.wd, EPS, save)
    if not save:
        return [cast(torch.Tensor, post).view(single.shape)]  # (no save: post_fwd returns the output alone)
    out, u, a2, x2, t = post
    saved_pre = cast("tuple[torch.Tensor, ...]", pre)  # (save: pre_fwd returns six tensors)
    x1 = saved_pre[4]
    xkv = saved_pre[5] if cross else single.new_empty(0)
    return [
        out.view(single.shape),
        mod,
        q,
        k,
        v,
        sg,
        x1,
        xkv,
        bias,
        o.view(m, DS_),
        lse,
        u,
        a2,
        x2,
        t,
    ]


def _bwd_fake(dy, single, cond, pair, mask, weights, saved):
    return [
        torch.empty_like(single),
        torch.empty_like(cond),
        torch.empty_like(pair),
        *(torch.empty_like(w) for w in weights),
    ]


@opaque(fake=_bwd_fake, name="local_rows_sm80_block_bwd")
def _block_bwd(
    dy: torch.Tensor,
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    weights: list[torch.Tensor],
    saved: list[torch.Tensor],
) -> list[torch.Tensor]:
    """d single, d cond, d pair, then one gradient per weight in ``WEIGHTS`` order, in the weight's dtype (all fresh tensors)."""
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm80_atom_local as local,
    )
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm80_atom_rows as rows,
    )

    mod, q, k, v, sg, x1, xkv, bias, o, lse, u, a2, x2, t = saved
    a, _, n, _ = single.shape
    m = a * n
    cross = len(weights) > len(WEIGHTS)
    dev = single.device
    with torch.cuda.device(dev):
        pk, pb = _weights(weights), _weights_bwd(weights)
        s2 = single.reshape(m, DS_).contiguous()
        c2 = cond.reshape(m, DC_).contiguous()
        dy2 = dy.reshape(m, DS_).to(BF).contiguous()
        dmod = torch.empty(m, 8 * DS_, device=dev, dtype=BF)  # [dsc1 | dbi1 | dsckv | dbikv | dsc2 | dbi2 | dos | dts]
        if not cross:
            dmod[:, 2 * DS_ : 4 * DS_] = 0
        dp = torch.empty(m, 4 * DS_, device=dev, dtype=BF)  # [dq | dg | dk | dv]
        dbias = torch.zeros(8 * DS_, device=dev, dtype=F32)
        dbq = torch.zeros(DS_, device=dev, dtype=F32)
        dgamma = torch.zeros(3, DS_, device=dev, dtype=F32)
        dt, hh, dab = rows.tr_bwd(dy2, t, x2, mod, pk.wu, pb.wst, dmod, dbias[7 * DS_ :])
        da, du, gated, do, dd = rows.post_bwd(dab, dy2, a2, mod, u, sg, o, pb.wut, pb.wot, dp, dmod, dbias, n, EPS)
        keymask = _keys(mask, n, dev)
        db = local.attn_bwd(
            q.view(a, n, DS_),
            k.view(a, n, DS_),
            v.view(a, n, DS_),
            do.view(a, n, DS_),
            bias,
            o.view(a, n, DS_),
            lse,
            keymask,
            delta=dd,
            packed=dp,
        )
        dz, dgamma_pair, dwb = local.pair_bias_bwd(pair[0], weights[_GAMMA], weights[_WBIAS], db)
        ds = rows.pre_bwd(dp, s2, x1, da, mod, pb.wqgt, pb.wkvt, dmod, dbias, dbq, EPS, cross)
        dc, cn = rows.cond_bwd(dmod, c2, pb.wt, pk.gamma, dgamma, EPS, cross)
        names = WEIGHTS + CROSS_WEIGHTS[: len(weights) - len(WEIGHTS)]
        A, T, K = "attention.", "transition.", "attention.ada_ln_kv."
        grads = {name: torch.empty(w.shape, device=dev, dtype=w.dtype) for name, w in zip(names, weights, strict=True)}
        # the weight gradients: D = A^T B products of the activations and gradients the kernels wrote (one grouped launch), and the small fp32 sums
        problems = [
            (dt, hh),
            (dab, x2),
            (du, gated),
            (dp[:, : 2 * DS_], x1),
            (dp[:, 2 * DS_ :], xkv if cross else x1),
            (dmod[:, 0 : 2 * DS_], cn[0]),
            (dmod[:, 4 * DS_ : 6 * DS_], cn[2]),
            (dmod[:, 6 * DS_ :], c2),
        ]
        index = {"sq": 0, "wu": 1, "wo": 2, "qg": 3, "kv": 4, "c0": 5, "c2": 6, "cg": 7}
        if cross:
            index["c1"] = len(problems)
            problems.append((dmod[:, 2 * DS_ : 4 * DS_], cn[1]))

        def two(key, first, second):
            return [
                (index[key], None, 0, DS_, grads[first]),
                (index[key], None, DS_, DS_, grads[second]),
            ]

        segments = [
            (index["sq"], None, 0, DS_, grads[T + "squeeze.weight"]),
            (index["wu"], None, 0, 2 * DS_, grads[T + "expand_a.weight"]),
            (index["wu"], None, 2 * DS_, 2 * DS_, grads[T + "expand_b.weight"]),
            (index["wo"], None, 0, DS_, grads[A + "to_out.weight"]),
            *two("qg", A + "to_query.weight", A + "to_gate.weight"),
            *two("kv", A + "to_key.weight", A + "to_value.weight"),
            *two("c0", A + "ada_ln_in.to_scale.weight", A + "ada_ln_in.to_bias.weight"),
            *two("c2", T + "ada_ln_in.to_scale.weight", T + "ada_ln_in.to_bias.weight"),
            *two("cg", A + "to_scale.weight", T + "to_scale.weight"),
            (None, dbias[:DS_], 0, 0, grads[A + "ada_ln_in.to_scale.bias"]),
            (
                None,
                dbias[4 * DS_ : 5 * DS_],
                0,
                0,
                grads[T + "ada_ln_in.to_scale.bias"],
            ),
            (None, dbias[6 * DS_ : 7 * DS_], 0, 0, grads[A + "to_scale.bias"]),
            (None, dbias[7 * DS_ :], 0, 0, grads[T + "to_scale.bias"]),
            (None, dbq, 0, 0, grads[A + "to_query.bias"]),
            (None, dgamma[0], 0, 0, grads[A + "ada_ln_in.ln_cond.weight"]),
            (None, dgamma[2], 0, 0, grads[T + "ada_ln_in.ln_cond.weight"]),
            (None, dgamma_pair, 0, 0, grads[A + "ln_pair.weight"]),
            (None, dwb.reshape(-1), 0, 0, grads[A + "to_bias.weight"]),
        ]
        if cross:
            segments += [
                *two("c1", K + "to_scale.weight", K + "to_bias.weight"),
                (None, dbias[2 * DS_ : 3 * DS_], 0, 0, grads[K + "to_scale.bias"]),
                (None, dgamma[1], 0, 0, grads[K + "ln_cond.weight"]),
            ]
        rows.wgrad(problems, segments, m)
        per_weight = [grads[name] for name in names]
    return [
        ds.view(single.shape),
        dc.view(cond.shape),
        dz.reshape(pair.shape),
        *per_weight,
    ]


class _RowsBlock(torch.autograd.Function):
    """The whole block forward and backward on the sm_80 kernels; gradients for single, cond, pair and every weight."""

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
    """The block's output stream (both residuals included), on the sm_80 kernels; ``serves()`` must have accepted the call."""
    weights = [module.get_parameter(name) for name in (*WEIGHTS, *(CROSS_WEIGHTS if module.cross_attention else ()))]
    if torch.is_grad_enabled() and any(t.requires_grad for t in (single, cond, pair, *weights)):
        return _RowsBlock.apply(mask, single, cond, pair, *weights)
    return _infer(single, cond, pair, mask, weights)
