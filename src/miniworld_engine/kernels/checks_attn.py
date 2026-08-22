"""Accuracy checks for the three attention families: triangle, augmented pair-bias, bias-only.

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

from collections.abc import Callable, Sequence

import torch
import triton

from .drivers import BF16, dev
from .drivers_attn import A, D, D32, DH, DP, H, L, _aug_qkvb, _bias_only_vb, _tri_qkvb

Pair = tuple[torch.Tensor, torch.Tensor]


# ── shared machinery ────────────────────────────────────────────────────────────────────────


def _grads(
    fn: Callable[..., torch.Tensor],
    inputs: Sequence[torch.Tensor],
    ref: Callable[..., torch.Tensor],
    names: Sequence[str],
) -> dict[str, Pair]:
    """Kernel grads vs fp32-autograd grads of ``ref``, on the same values and the same ``dy``.

    The kernel runs on bf16 leaves; the reference runs on fp32 copies of those exact bf16 values,
    so nothing but the arithmetic differs. One ``dy`` is drawn and both backwards get it.
    """
    leaves = [t.detach().clone().requires_grad_(True) for t in inputs]
    out = fn(*leaves)
    dy = torch.randn_like(out)          # dense: see the module docstring on out.sum().backward()
    out.backward(dy)

    refs = [t.detach().float().requires_grad_(True) for t in inputs]
    ref(*refs).backward(dy.float())
    return {n: (lf.grad, rf.grad) for n, lf, rf in zip(names, leaves, refs)}


def _rowsum(o: torch.Tensor, do: torch.Tensor) -> torch.Tensor:
    """``delta[..., m] = sum_d o[..., m, d] * do[..., m, d]`` -- what every bwd_pre kernel emits."""
    return (o.float() * do.float()).sum(-1)


def _fp32_matmul() -> None:
    """fp32 has to mean fp32: tf32 would hand the reference the kernel's own 10-bit mantissa."""
    torch.backends.cuda.matmul.allow_tf32 = False


#: The base-2 conversion every one of these kernels hardcodes (``qk_scale *= 1.44269504``).
_LOG2E = 1.44269504


def _lse2(logits: torch.Tensor) -> torch.Tensor:
    """The saved ``m``: a base-2 log-sum-exp over the key axis -- NOT the running row max.

    Each forward runs its softmax in log2 space (``p = exp2(logits*log2e - m_i)``) and then, after
    the key loop closes, folds the denominator in: ``m_i += tl.math.log2(l_i)``. So the stored
    value is ``log2(sum_n 2**(logits*log2e)) == logsumexp(logits) * log2e``, and the backward that
    consumes it recovers ``p`` as ``exp2(logits*log2e - m)`` with no separate ``l``. Checking ``m``
    against the max alone would be wrong by ``log2(l)`` on every row -- and would still "look
    plausible", which is the whole reason the indirect constraint through the backward is weak.
    """
    return torch.logsumexp(logits, -1) * _LOG2E


def _fwd_saved(fn, inputs, m_at: int) -> Pair:
    """``(out, m)`` for a forward whose ``m`` is saved for the backward and never returned.

    ``forward`` returns ``out`` alone; ``m`` reaches the backward through
    ``ctx.save_for_backward``, so the autograd node is the only place a checker can see the value
    the backward will actually consume. ``m_at`` is that tensor's index in the save order, which
    differs per file and is named at each call site. Reaching the kernel through ``.apply`` rather
    than re-launching it keeps the checker on the driver's exact path -- same ``.contiguous()``
    copies, same strides, same grid.
    """
    out = fn(*inputs)
    return out, out.grad_fn.saved_tensors[m_at]


def _tri_logits(q, k, bias) -> torch.Tensor:
    """``[B,H,L,L,L]`` pre-softmax logits; bias ``[B,H,L,L]`` shared across the row axis (dim 2)."""
    logits = torch.einsum("bhimd,bhind->bhimn", q, k) * q.shape[-1] ** -0.5
    return logits + bias.unsqueeze(2)


def _tri_ref(q, k, v, bias) -> torch.Tensor:
    """q/k/v ``[B,H,L,L,D]``, bias ``[B,H,L,L]`` shared across the row axis (dim 2)."""
    return torch.einsum("bhimn,bhind->bhimd", _tri_logits(q, k, bias).softmax(-1), v)


def _aug_logits(q, k, bias) -> torch.Tensor:
    """``[A,B,H,L,L]`` logits; bias ``[B,L,L,H]`` -> ``[B,H,L,L]``, shared across the A axis."""
    logits = torch.einsum("abmhd,abnhd->abhmn", q, k) * q.shape[-1] ** -0.5
    return logits + bias.permute(0, 3, 1, 2).unsqueeze(0)


def _aug_ref(q, k, v, bias) -> torch.Tensor:
    """q/k/v ``[A,B,L,H,D]``, bias ``[B,L,L,H]`` -> ``[B,H,L,L]``, shared across the A axis."""
    return torch.einsum("abhmn,abnhd->abmhd", _aug_logits(q, k, bias).softmax(-1), v)


def _bias_only_ref(v, bias) -> torch.Tensor:
    """No q/k: p = softmax(bias) is one ``[B,H,L,L]`` matrix, reused by every row of v."""
    return torch.einsum("bhmn,bhind->bhimd", bias.softmax(-1), v)


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
    from .triangle_attention.triton.main import TritonTriangleAttentionPairBiasFunction as Fn
    # save order: (q, k, v, bias, m, out) -> m at 4.
    return _tri_fwd_pairs(Fn, D, 4)


def triangle_attention_bwd_pre_triton() -> Pair:
    from miniworld_engine.autotune.shape_key import token_key

    from .triangle_attention.triton.main import _attn_bwd_preprocess
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
    from .triangle_attention.triton.main import TritonTriangleAttentionPairBiasFunction as Fn
    # _attn_bwd_dkdv produces dk/dv/dbias; dq comes from the separate _attn_bwd_dq below.
    g = _grads(Fn.apply, _tri_qkvb(), _tri_ref, ("dq", "dk", "dv", "dbias"))
    return {n: g[n] for n in ("dk", "dv", "dbias")}


def triangle_attention_bwd_dq_triton() -> dict[str, Pair]:
    from .triangle_attention.triton.main import TritonTriangleAttentionPairBiasFunction as Fn
    g = _grads(Fn.apply, _tri_qkvb(), _tri_ref, ("dq", "dk", "dv", "dbias"))
    return {"dq": g["dq"]}


# ── triangle_attention / triton / atomic.py ─────────────────────────────────────────────────


def triangle_attention_fwd_contig_triton() -> dict[str, Pair]:
    from .triangle_attention.triton.atomic import TritonTriangleAttentionPairBiasFunction as Fn
    # D32, not D: this forward raises ValueError on any other head dim, so it is the only shape
    # the driver can reach it with. L is still ragged, and IS the axis that tiles here -- this
    # file's masking is a bias load with other=-inf plus EVEN_N/EVEN_D branches on q/k/v/store,
    # so a partial key tile contributes exp2(-inf) == 0 rather than a padded logit.
    # save order: (q, k, v, bias, o, M) -> M last, at 5 (main.py's is the other way round).
    return _tri_fwd_pairs(Fn, D32, 5)


def triangle_attention_bwd_pre_contig_triton() -> Pair:
    from miniworld_engine.autotune.shape_key import token_key

    from .triangle_attention.triton.atomic import _attn_bwd_preprocess
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
    from .triangle_attention.triton.atomic import TritonTriangleAttentionPairBiasFunction as Fn
    # One kernel emits all four grads here: dq lands via atomic accumulation into an fp32 buffer,
    # dk/dv/dbias are stored once. The atomics change the summation ORDER, not the value.
    return _grads(Fn.apply, _tri_qkvb(D32), _tri_ref, ("dq", "dk", "dv", "dbias"))


# ── augmented_attention / triton / main.py ──────────────────────────────────────────────────


def augmented_attention_fwd_triton() -> dict[str, Pair]:
    from .augmented_attention.triton.main import TritonAugmentedAttentionFunction as Fn
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

    from .augmented_attention.triton.main import _attn_bwd_preprocess
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
    from .augmented_attention.triton.main import TritonAugmentedAttentionFunction as Fn
    # _attn_bwd writes dq into one slot of dq_expand per BLOCK_M2 block, so its dq is a PARTIAL;
    # the grad autograd returns is the post-_dq_reduce sum of those slots, which is the value the
    # split is supposed to add up to. dk/dv/dbias come out of this same kernel whole.
    return _grads(Fn.apply, _aug_qkvb(), _aug_ref, ("dq", "dk", "dv", "dbias"))


def augmented_attention_bwd_reduce_triton() -> Pair:
    from .augmented_attention.triton.main import (
        _bwd_min_block_n, _dq_reduce, get_elem_group,
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
    from .augmented_attention.triton.memory_efficient import (
        TritonAugmentedAttentionFunction as Fn,
    )
    # Same math as the split backward, dq accumulated by atomics into one fp32 buffer instead of
    # per-program slots -- so the same reference has to hold.
    return _grads(Fn.apply, _aug_qkvb(), _aug_ref, ("dq", "dk", "dv", "dbias"))


# ── bias_only_attention / triton / main.py ──────────────────────────────────────────────────


def bias_only_attention_fwd_triton() -> dict[str, Pair]:
    from .bias_only_attention.triton.main import TritonBiasOnlyAttentionFunction as Fn
    _fp32_matmul()
    v, bias = _bias_only_vb()
    # save order: (v, bias, m, out) -> m at 2.
    out, m = _fwd_saved(Fn.apply, (v, bias), 2)
    vf, bf = v.detach().float(), bias.detach().float()
    # There is no q/k, so the logits ARE the bias and m does not depend on the row axis at all:
    # the kernel's bias base offset carries no off_t, yet it still stores m per row into a
    # [B,H,L,L] buffer indexed [b, h, row i, query j]. So the reference is one [B,H,L] log-sum-exp
    # over the key axis, broadcast back over the L rows -- if the kernel ever let a row leak into
    # its softmax, this is where it would show.
    m_ref = _lse2(bf).unsqueeze(2).expand(-1, -1, L, -1)
    return {"out": (out, _bias_only_ref(vf, bf)), "m": (m, m_ref)}


def bias_only_attention_bwd_pre_triton() -> Pair:
    from miniworld_engine.autotune.shape_key import token_key

    from .bias_only_attention.triton.main import _attn_bwd_preprocess
    B, HL = 1, H * L
    # Contiguous addressing, no strides -- same contract as atomic.py's, and the reason its
    # backward calls grad_output.contiguous() before this launch.
    o = torch.randn(B, HL, L, D, device=dev(), dtype=BF16)
    do = torch.randn_like(o)
    delta = torch.empty(B, HL, L, device=dev(), dtype=torch.float32)
    grid = lambda META: [triton.cdiv(L, META["BLOCK_M1"]), B * HL, 1]
    _attn_bwd_preprocess[grid](
        o, do, delta, B, L, D,
        shape_key=token_key(L), HEAD_DIM_PAD=triton.next_power_of_2(D),
    )
    return delta, _rowsum(o, do)


def bias_only_attention_bwd_triton() -> dict[str, Pair]:
    from .bias_only_attention.triton.main import TritonBiasOnlyAttentionFunction as Fn
    # Only two grads exist on this path: there is no q/k to differentiate.
    return _grads(Fn.apply, _bias_only_vb(), _bias_only_ref, ("dv", "dbias"))


# ── bias_only_attention / triton / gate_out.py ──────────────────────────────────────────────


def gated_projection_gate_gemm_triton() -> Pair:
    from .bias_only_attention.triton.gate_out import _fwd
    kw = {"device": dev(), "dtype": BF16}
    gate = torch.randn(L * L, DH, **kw)
    out_r = torch.randn(L * L, DH, **kw)
    wo = torch.randn(DP, DH, **kw)          # to_out.weight [N, DH]; out = A @ wo.T
    # The fused kernel builds its A-tile as sigmoid(gate)*out_r in the GEMM prologue, so `gated`
    # never reaches HBM. The reference forms it and does the GEMM in fp32.
    ref = (torch.sigmoid(gate.float()) * out_r.float()) @ wo.float().t()
    return _fwd(gate, out_r, wo), ref


def gated_projection_bwd_dx_triton() -> dict[str, Pair]:
    from .bias_only_attention.triton.gate_out import _dgrad_epilogue
    kw = {"device": dev(), "dtype": BF16}
    do2 = torch.randn(L * L, DP, **kw)      # grad wrt [M, N], N == d_pair
    wo = torch.randn(DP, DH, **kw)
    g2 = torch.randn(L * L, DH, **kw)
    r2 = torch.randn(L * L, DH, **kw)
    dr, dg, a = _dgrad_epilogue(do2, wo, g2, r2)
    # out = (sigmoid(gate) * out_r) @ wo.T, so with da = do @ wo:
    #   d_out_r = s*da,  d_gate = da*r*s*(1-s),  and `gated` = s*r is handed to the d_wo GEMM.
    da = do2.float() @ wo.float()
    s, r = torch.sigmoid(g2.float()), r2.float()
    return {
        "d_out_r": (dr, s * da),
        "d_gate": (dg, da * r * s * (1.0 - s)),
        "gated": (a, s * r),
    }
