"""Accuracy checks for the ``augmented_attention`` family.

The three attention families were one module (``checks_attn.py``). What the references have to
get right -- the ``(m, n)``-only bias indexing, the key mask, the log2-space ``m`` the forward
checkers also return, and why the backward references are fp32 autograd on a dense ``dy`` -- is
written out in ``checks/triangle_attention.py``; the helpers all three use are in
``checks/__init__.py``.
"""
from __future__ import annotations

import torch
import triton

from miniworld_engine.kernels.checks import (
    Pair,
    _fp32_matmul,
    _fwd_saved,
    _grads,
    _lse2,
    _rowsum,
)
from miniworld_engine.kernels.drivers import BF16, dev
from miniworld_engine.kernels.drivers.augmented_attention import A, _aug_qkvb
from miniworld_engine.kernels.drivers.triangle_attention import D, H, L


def _aug_logits(q, k, bias) -> torch.Tensor:
    """``[A,B,H,L,L]`` logits; bias ``[B,L,L,H]`` -> ``[B,H,L,L]``, shared across the A axis."""
    logits = torch.einsum("abmhd,abnhd->abhmn", q, k) * q.shape[-1] ** -0.5
    return logits + bias.permute(0, 3, 1, 2).unsqueeze(0)


def _aug_ref(q, k, v, bias) -> torch.Tensor:
    """q/k/v ``[A,B,L,H,D]``, bias ``[B,L,L,H]`` -> ``[B,H,L,L]``, shared across the A axis."""
    return torch.einsum("abhmn,abnhd->abmhd", _aug_logits(q, k, bias).softmax(-1), v)


# ── augmented_attention / triton / main.py ──────────────────────────────────────────────────


def augmented_attention_fwd_triton() -> dict[str, Pair]:
    from miniworld_engine.kernels.augmented_attention.triton.main import (
        TritonAugmentedAttentionFunction as Fn,
    )
    _fp32_matmul()
    # The driver passes no mask, so the forward substitutes all-ones and the where() over the key
    # axis is a no-op -- the reference carries no mask term (see the module docstring).
    q, k, v, bias = _aug_qkvb()
    # save order: (q, k, v, bias, mask, out, m) -> m last, at 6.
    out, m = _fwd_saved(Fn.apply, (q, k, v, bias), 6)
    qf, kf, vf, bf = (t.detach().float() for t in (q, k, v, bias))
    # m is [A,B,H,L]: it is stored at off_hz*N_CTX + off_m with off_hz enumerating (a, b, h), so
    # H sits in front of the query axis, which is the layout the [A,B,H,L,L] logits reduce to.
    return {"out": (out, _aug_ref(qf, kf, vf, bf)), "m": (m, _lse2(_aug_logits(qf, kf, bf)))}


def augmented_attention_bwd_pre_triton() -> Pair:
    from miniworld_engine.autotune.shape_key import atom_key
    from miniworld_engine.kernels.augmented_attention.triton.main import (
        _attn_bwd_preprocess,
    )
    B = 1
    o = torch.randn(A, B, L, H, D, device=dev(), dtype=BF16)
    do = torch.randn_like(o)
    delta = torch.empty(A, B, H, L, device=dev(), dtype=torch.float32)
    grid = lambda META: (triton.cdiv(L, META["BLOCK_M1"]), A * B, H)
    _attn_bwd_preprocess[grid](
        o, do, delta, L,
        o.stride(1), o.stride(2), o.stride(3), o.stride(4), H, D,
        shape_key=atom_key(L, H=H, HEAD_DIM=D), HEAD_DIM_PAD=triton.next_power_of_2(D),
    )
    # Delta is (A,B,H,L) -- stored at off_z*H*N_CTX + off_h*N_CTX + off_m with off_z over A*B --
    # while the rowsum over d leaves (A,B,L,H), so H moves in front of L.
    return delta, _rowsum(o, do).transpose(-1, -2)


def augmented_attention_bwd_split_triton() -> dict[str, Pair]:
    from miniworld_engine.kernels.augmented_attention.triton.main import (
        TritonAugmentedAttentionFunction as Fn,
    )
    # _attn_bwd writes dq into one slot of dq_expand per BLOCK_M2 block, so its dq is a PARTIAL;
    # the grad autograd returns is the post-_dq_reduce sum of those slots, which is the value the
    # split is supposed to add up to. dk/dv/dbias come out of this same kernel whole.
    return _grads(Fn.apply, _aug_qkvb(), _aug_ref, ("dq", "dk", "dv", "dbias"))


def augmented_attention_bwd_reduce_triton() -> Pair:
    from miniworld_engine.kernels.augmented_attention.triton.main import (
        _bwd_min_block_n,
        _dq_reduce,
        get_elem_group,
    )
    B = 1
    # The split count has to match the backward's: cdiv(L, min BLOCK_M2 over the split kernel's
    # configs). Random slots (rather than real partials) make the reference exact -- fp32 in,
    # fp32 out, and the only thing under test is the sum over the slot axis.
    num_splits = int(triton.cdiv(L, _bwd_min_block_n()))
    dq_expand = torch.randn(num_splits, A, B, L, H, D, device=dev(), dtype=torch.float32)
    dq = torch.empty(A, B, L, H, D, device=dev(), dtype=torch.float32)
    n_elem = A * B * L * H * D
    grid = lambda META: (triton.cdiv(n_elem, META["BLOCK_E"]),)
    _dq_reduce[grid](
        dq_expand, dq, num_splits, dq_expand.stride(0), 1, n_elem,
        shape_key=get_elem_group(n_elem),
    )
    return dq, dq_expand.sum(0)


# ── augmented_attention / triton / memory_efficient.py ──────────────────────────────────────


def augmented_attention_bwd_atomic_triton() -> dict[str, Pair]:
    from miniworld_engine.kernels.augmented_attention.triton.memory_efficient import (
        TritonAugmentedAttentionFunction as Fn,
    )
    # Same math as the split backward, dq accumulated by atomics into one fp32 buffer instead of
    # per-program slots -- so the same reference has to hold.
    return _grads(Fn.apply, _aug_qkvb(), _aug_ref, ("dq", "dk", "dv", "dbias"))


# ── augmented_attention / triton / window.py ────────────────────────────────────────────────


def _window_ref(keep, starts):
    from miniworld_engine.kernels.augmented_attention.reference import (
        window_attention_pytorch,
    )
    from miniworld_engine.kernels.drivers.augmented_attention import _WIN_KEYS
    return lambda q, k, v, bias: window_attention_pytorch(q, k, v, bias, starts, keep, keys=_WIN_KEYS)


def augmented_attention_fwd_window_triton() -> dict[str, Pair]:
    from miniworld_engine.kernels.augmented_attention.reference import (
        window_logits_pytorch,
    )
    from miniworld_engine.kernels.augmented_attention.triton.window import (
        WindowAttentionFunction as Fn,
    )
    from miniworld_engine.kernels.drivers.augmented_attention import (
        _WIN_KEYS,
        _window_inputs,
    )
    _fp32_matmul()
    q, k, v, bias, keep, starts = _window_inputs()
    # save order: (q, k, v, bias, starts, keep, uid, out, lse, lo, hi) -> lse at 8 (natural log, [A, B, H, nwin * 32])
    out, lse = _fwd_saved(lambda *t: Fn.apply(*t, starts, keep.to(torch.uint8), None, _WIN_KEYS, -1), (q, k, v, bias), 8)
    qf, kf, vf, bf = (t.detach().float() for t in (q, k, v, bias))
    logits, _, _ = window_logits_pytorch(qf, kf, bf, starts, keep, keys=_WIN_KEYS)
    a, b, h, nwin = logits.shape[:4]
    return {"out": (out, _window_ref(keep, starts)(qf, kf, vf, bf)),
            "lse": (lse, torch.logsumexp(logits, -1).reshape(a, b, h, nwin * 32))}


def augmented_attention_bwd_window_triton() -> dict[str, Pair]:
    from miniworld_engine.kernels.drivers.augmented_attention import (
        _window_apply,
        _window_inputs,
    )
    _fp32_matmul()
    q, k, v, bias, keep, starts = _window_inputs()
    return _grads(lambda *t: _window_apply(*t, keep, starts), (q, k, v, bias), _window_ref(keep, starts), ("dq", "dk", "dv", "dbias"))


def augmented_attention_bwd_reduce_window_triton() -> Pair:
    from miniworld_engine.kernels.augmented_attention.triton.window import (
        _window_fold,
        fold_range,
    )
    from miniworld_engine.kernels.drivers.augmented_attention import (
        _WIN_D,
        _WIN_H,
        _WIN_KEYS,
        _WIN_N,
        _window_inputs,
    )
    n, h, d, keys = _WIN_N, _WIN_H, _WIN_D, _WIN_KEYS
    _, _, _, _, _, starts = _window_inputs(False)
    nwin = starts.shape[1]
    ntile = (n + 31) // 32
    lo, hi = fold_range(starts, n, keys)
    dkp = torch.randn(A, 1, nwin, keys, h, d, device=dev(), dtype=torch.float32)
    dvp = torch.randn_like(dkp)
    dk = torch.empty(A, 1, n, h, d, device=dev(), dtype=torch.float32)
    dv = torch.empty_like(dk)
    from miniworld_engine.autotune.shape_key import atom_key, pack
    _window_fold[lambda meta: (ntile, A, h * triton.cdiv(d, meta["BLOCK_D"]))](
        dkp, dvp, dk, dv, starts, lo, hi, dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        n, nwin, ntile, 1, H=h, HEAD_DIM=d, KEYS=keys, shape_key=pack(atom_key(n), H=h, HEAD_DIM=d),
    )
    # reference: every window's partial added at its keys, dense and in fp32
    pos = (starts.long()[0][:, None] + torch.arange(keys, device=dev())).reshape(-1)         # [nwin * keys]
    ok = (pos >= 0) & (pos < n)
    ref_k = torch.zeros(A, n, h, d, device=dev())
    ref_v = torch.zeros_like(ref_k)
    ref_k.index_add_(1, pos[ok], dkp[:, 0].reshape(A, nwin * keys, h, d)[:, ok])
    ref_v.index_add_(1, pos[ok], dvp[:, 0].reshape(A, nwin * keys, h, d)[:, ok])
    return torch.stack([dk, dv]), torch.stack([ref_k[:, None], ref_v[:, None]])
