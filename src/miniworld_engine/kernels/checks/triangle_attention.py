"""Accuracy checks for the ``triangle_attention`` family.

A checker takes no arguments and returns what ``autotune.run_all.check_one`` compares: one
``(actual, expected)`` pair, or a dict of them keyed by the tensor it names. Launching a kernel
proves it runs; only a reference says the number is right.

Shapes and the call pattern come from ``drivers_attn.py`` unchanged -- ``L``/``H``/``D``/``D32``/
``A``/``DH``/``DP`` and the ``_tri_qkvb``/``_aug_qkvb``/``_bias_only_vb`` input builders are imported
from it rather than restated, so a checker reaches the same kernel its driver reaches, and
``MINIWORLD_SHAPE_MODE=ragged`` moves both sides together. References are fp32 torch.

Nothing here restates an extent: the four ``bwd_pre``/``_dq_reduce`` checkers that launch a kernel
directly build their tensors out of those same constants, so a ragged run perturbs the direct
launches and the autograd launches by the same amount. ``D32`` is the atomic triangle path's fixed
head dim (its forward raises on anything else); ``DP`` is the gate-out GEMM's output width, which is
``wo.shape[0]`` and is deliberately NOT equal to the contraction width ``DH`` in ragged mode.

Three things the reference has to get right, all read off the kernels:

* ``triangle_attention`` and ``augmented_attention`` both compute
  ``softmax(q @ k.T * D**-0.5 + bias) @ v``. The bias is indexed ``(m, n)`` ONLY: the forward's
  bias base offset carries no row and no augmentation term (``b_base`` in
  ``triangle_attention/triton/main.py``, ``bias_offset`` in ``augmented_attention/triton/main.py``,
  neither of which adds ``off_t``/``off_a``), so bias is shared across the row axis of
  ``[B,H,L,L,D]`` and across the ``A`` axis of ``[A,B,L,H,D]``. Broadcasting it in the reference is
  what makes the reference ``dbias`` equal the sum the kernels build -- both backwards materialize
  a per-row ``dbias`` and reduce it (``reduce(dbias, "B (H L3) L L2 -> B H L L2", "sum")``).
* ``augmented_attention`` takes an optional ``mask``, a ``[A,B,L]`` KEY mask applied to the logits
  as ``bias = where(key_mask, bias, -inf)`` (``_attn_fwd_inner``). Both augmented forwards
  substitute an all-ones mask when it is ``None``, which is what the drivers pass, so the reference
  needs no mask term. ``triangle_attention`` and ``bias_only_attention`` take no mask at all; their
  only ``tl.where`` is an ``offs_n < N_CTX`` bound.
* ``bias_only_attention`` has no q/k. The logits ARE the bias (``logits = bias_val * 1/log(2)``),
  so the probability matrix is ``softmax(bias)`` -- one ``[B,H,L,L]`` matrix every row reuses, and
  the reference contracts it against ``v`` without ever forming a 5-D copy of it.

The four FORWARD checkers return BOTH tensors the kernel writes, ``out`` and ``m``. ``m`` is not
returned by ``forward`` -- it goes to the backward through ``ctx.save_for_backward`` -- so
``_fwd_saved`` reads it off the autograd node. Checking only ``out`` would leave the same hole the
backward checkers already leave: ``m`` is a log2-space log-sum-exp (``m_i += log2(l_i)`` closes
every one of these kernels), an ``m`` that is wrong while ``out`` looks right is a live failure
mode, and a poisoned tail tile can partially cancel between the running max and the running sum.

Backward references are fp32 autograd rather than a hand-derived formula: the reference forward
runs on fp32 copies of the same bf16 values and takes ``.backward(dy)`` with the SAME ``dy`` the
kernel got. ``dy`` is ``torch.randn_like(out)``, never ``out.sum().backward()`` -- ``sum()`` hands
the backward a stride-0 expanded grad whose storage holds one element, and the
contiguous-addressing preprocess kernels in ``triangle_attention/triton/atomic.py`` and
``bias_only_attention/triton/main.py`` take no strides and would read ``L*D`` past it. Both now
call ``grad_output.contiguous()`` first, so that path is safe either way; the dense grad keeps the
check off it regardless.

The ``bwd_pre`` kernels and ``_dq_reduce`` are never reached through autograd -- their outputs are
internal to a backward and are not returned. Those four checkers launch the kernel directly, with
the launch copied from the ``backward`` that owns it, against a reference exact enough to leave no
doubt: ``delta = rowsum(o * do)`` and ``dq = sum_s dq_expand[s]``.
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
from miniworld_engine.kernels.drivers.triangle_attention import D, D32, H, L, _tri_qkvb

def _tri_logits(q, k, bias) -> torch.Tensor:
    """``[B,H,L,L,L]`` pre-softmax logits; bias ``[B,H,L,L]`` shared across the row axis (dim 2)."""
    logits = torch.einsum("bhimd,bhind->bhimn", q, k) * q.shape[-1] ** -0.5
    return logits + bias.unsqueeze(2)


def _tri_ref(q, k, v, bias) -> torch.Tensor:
    """q/k/v ``[B,H,L,L,D]``, bias ``[B,H,L,L]`` shared across the row axis (dim 2)."""
    return torch.einsum("bhimn,bhind->bhimd", _tri_logits(q, k, bias).softmax(-1), v)


def _tri_fwd_pairs(Fn, d: int, m_at: int) -> dict[str, Pair]:
    """out + m for both triangle forwards: same math, same layouts, different save order.

    ``d`` is the head dim the driver reaches this file with (``D``, or ``D32`` for atomic.py).
    ``m`` is ``[B,H,L,L]`` indexed ``[b, h, row i, query j]``: the kernel's ``off_t`` (grid dim 2,
    extent L) is the row axis and the ``BLOCK_M1`` axis is the query, so a plain logsumexp over
    the key axis of the ``[B,H,L,L,L]`` logits lands in exactly that shape.
    """
    _fp32_matmul()
    q, k, v, bias = _tri_qkvb(d)
    out, m = _fwd_saved(Fn.apply, (q, k, v, bias), m_at)
    qf, kf, vf, bf = (t.detach().float() for t in (q, k, v, bias))
    return {"out": (out, _tri_ref(qf, kf, vf, bf)), "m": (m, _lse2(_tri_logits(qf, kf, bf)))}


# ── triangle_attention / triton / main.py ───────────────────────────────────────────────────


def triangle_attention_fwd_triton() -> dict[str, Pair]:
    from miniworld_engine.kernels.triangle_attention.triton.main import TritonTriangleAttentionPairBiasFunction as Fn
    # save order: (q, k, v, bias, m, out) -> m at 4.
    return _tri_fwd_pairs(Fn, D, 4)


def triangle_attention_bwd_pre_triton() -> Pair:
    from miniworld_engine.autotune.shape_key import token_key

    from miniworld_engine.kernels.triangle_attention.triton.main import _attn_bwd_preprocess
    from einops import rearrange
    B, HL = 1, H * L
    # main.py's preprocess takes BOTH stride sets, and at runtime the two differ: `out` is the
    # strided (B,H,L,L2,D) view over projection layout [B,L,L2,H*D] that the forward allocates,
    # while the grad arrives dense. Feed it that same mismatch.
    out = rearrange(
        torch.randn(B, L, L, H * D, device=dev(), dtype=BF16),
        "B L L2 (H D) -> B H L L2 D", H=H,
    )
    do = torch.randn(B, H, L, L, D, device=dev(), dtype=BF16)
    delta = torch.empty(B, HL, L, device=dev(), dtype=torch.float32)
    grid = lambda META: [triton.cdiv(L, META["BLOCK_M1"]), B * HL, 1]
    _attn_bwd_preprocess[grid](
        out, do, delta,
        *out.stride(), *do.stride(), HL, B, L, D,
        shape_key=token_key(L), HEAD_DIM_PAD=triton.next_power_of_2(D),
    )
    # Delta is addressed off_hz*N_CTX + off_m with off_hz enumerating (b, h, i_row) as
    # b*HL + h*L + i_row -- i.e. exactly the [B, H*L, L] flattening of the rowsum.
    return delta, _rowsum(out, do).reshape(B, HL, L)


def triangle_attention_bwd_dkdv_triton() -> dict[str, Pair]:
    from miniworld_engine.kernels.triangle_attention.triton.main import TritonTriangleAttentionPairBiasFunction as Fn
    # _attn_bwd_dkdv produces dk/dv/dbias; dq comes from the separate _attn_bwd_dq below.
    g = _grads(Fn.apply, _tri_qkvb(), _tri_ref, ("dq", "dk", "dv", "dbias"))
    return {n: g[n] for n in ("dk", "dv", "dbias")}


def triangle_attention_bwd_dq_triton() -> dict[str, Pair]:
    from miniworld_engine.kernels.triangle_attention.triton.main import TritonTriangleAttentionPairBiasFunction as Fn
    g = _grads(Fn.apply, _tri_qkvb(), _tri_ref, ("dq", "dk", "dv", "dbias"))
    return {"dq": g["dq"]}


# ── triangle_attention / triton / atomic.py ─────────────────────────────────────────────────


def triangle_attention_fwd_contig_triton() -> dict[str, Pair]:
    from miniworld_engine.kernels.triangle_attention.triton.atomic import TritonTriangleAttentionPairBiasFunction as Fn
    # D32, not D: this forward raises ValueError on any other head dim, so it is the only shape
    # the driver can reach it with. L is still ragged, and IS the axis that tiles here -- this
    # file's masking is a bias load with other=-inf plus EVEN_N/EVEN_D branches on q/k/v/store,
    # so a partial key tile contributes exp2(-inf) == 0 rather than a padded logit.
    # save order: (q, k, v, bias, o, M) -> M last, at 5 (main.py's is the other way round).
    return _tri_fwd_pairs(Fn, D32, 5)


def triangle_attention_bwd_pre_contig_triton() -> Pair:
    from miniworld_engine.autotune.shape_key import token_key

    from miniworld_engine.kernels.triangle_attention.triton.atomic import _attn_bwd_preprocess
    B, HL = 1, H * L
    # This preprocess takes no strides at all -- it addresses o/do as off_hz*D*L + m*D + d, a
    # contiguous [B, H*L, L, D], which is what its backward rearranges to before launching.
    # D32, not D: the driver reaches this kernel through atomic.py's forward, which raises on any
    # head dim but 32, so the checker has to build the same shape the driver does.
    o = torch.randn(B, HL, L, D32, device=dev(), dtype=BF16)
    do = torch.randn_like(o)
    delta = torch.empty(B, HL, L, device=dev(), dtype=torch.float32)
    grid = lambda META: [triton.cdiv(L, META["BLOCK_M1"]), B * HL, 1]
    _attn_bwd_preprocess[grid](
        o, do, delta, B, L, D32,
        shape_key=token_key(L), HEAD_DIM_PAD=triton.next_power_of_2(D32),
    )
    return delta, _rowsum(o, do)


def triangle_attention_bwd_atomic_triton() -> dict[str, Pair]:
    from miniworld_engine.kernels.triangle_attention.triton.atomic import TritonTriangleAttentionPairBiasFunction as Fn
    # One kernel emits all four grads here: dq lands via atomic accumulation into an fp32 buffer,
    # dk/dv/dbias are stored once. The atomics change the summation ORDER, not the value.
    return _grads(Fn.apply, _tri_qkvb(D32), _tri_ref, ("dq", "dk", "dv", "dbias"))
