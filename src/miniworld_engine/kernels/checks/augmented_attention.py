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
        o, do, delta, A, B, L,
        o.stride(1), o.stride(2), o.stride(3), o.stride(4), H, D,
        shape_key=atom_key(L), HEAD_DIM_PAD=triton.next_power_of_2(D),
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
