"""Drivers for the three attention families: triangle, augmented pair-bias, and bias-only.

Every kernel here is reached through an ``torch.autograd.Function``: the forward kernels by
``.apply(...)``, the backward kernels by calling ``.backward(dy)`` on what ``.apply`` returned.
The grad is an explicit ``torch.randn_like(out)``, never ``out.sum().backward()`` -- ``sum()``
hands the backward a stride-0 expanded tensor whose storage holds one element, and the
contiguous-addressing preprocess in ``triangle_attention/triton/atomic.py`` (which takes no
strides) reads L*D past it.

Shapes come from the kernel benches in ``benchmarks/runners/bench.py`` -- ``bench_kernel_tri_attn``
and ``bench_kernel_bias_attn`` use ``dh=32`` with ``H = d_pair // dh``, ``bench_kernel_aug_attn``
uses ``A=8, H=4, dh=32`` -- and ``d_pair=128`` from the module bench configs
(``benchmarks/modules/*/configs/bench.yaml``), so ``H=4``. ``L=128`` is the low end of the benched
sequence sweep (``augmented_attention/configs/bench.yaml``: ``min_seq_len: 128``); a driver only has
to reach the kernel, so the cheap end of the sweep is what it uses. The gate-out GEMM sizes follow
the module: ``to_out`` is ``Linear(d_hidden, d_pair)`` with ``d_hidden = d_pair = 128``, applied over
``M = B*L*L`` pair rows.

Tile alignment
--------------
Every extent above is a multiple of 128 or 32, and the five config sets tile at 16/32/64/128, so
nothing here has ever put a partial tile in front of a kernel. ``MINIWORLD_SHAPE_MODE=ragged``
subtracts 3 from each extent routed through ``drivers.ragged`` (see that module's docstring), which
makes the tail tile partial for all five config sets at once. Aligned mode is byte-identical to
what these drivers built before.

Which axis each extent perturbs:

* ``L`` -- the sequence length, and the axis that matters most here. It bounds BOTH loops of a
  flash-attention kernel: the query loop (``offs_m < N_CTX``) and the key/value loop
  (``for start_n in range(0, N_CTX, BLOCK_M2)``, ``offs_n < N_CTX``). An unmasked tail tile in the
  key loop poisons the softmax running max/sum -- ``m_i``/``l_i`` -- and that shows up as a
  plausible-looking wrong number, never a NaN. ``L`` also drives ``M = L*L`` for the gate-out GEMM
  and ``HL = H*L`` for the flattened (b, h, i_row) program axis of every backward.
* ``D`` -- the head dim. These kernels carry BOTH ``HEAD_DIM`` (the true extent) and
  ``HEAD_DIM_PAD = triton.next_power_of_2(HEAD_DIM)`` (the ``tl.arange``/``block_shape`` width, which
  must be a power of two), and every load/store on that axis is masked ``offs_k < HEAD_DIM`` or
  boundary-checked. ``HEAD_DIM_PAD`` is therefore the mechanism that *supports* a non-power-of-two
  head dim, not a contract forbidding one, so ``D`` is perturbed and the ``< HEAD_DIM`` masks get to
  run. The one exception is ``D32`` below.
* ``DH``/``DP`` -- the gate-out GEMM's contraction width (``d_hidden``) and output width
  (``d_pair == wo.shape[0]``). They are perturbed by DIFFERENT amounts so they stay unequal in ragged
  mode: ``_dgrad_epi`` tiles ``N`` as its contraction (``BLOCK_K``) and ``DH`` as a free axis
  (``BLOCK_N``), and with the two equal a swapped mask would still look right.

``H`` (head count) and ``A`` (augmentation count) are NOT routed through ``ragged``: neither is ever
blocked by a ``tl.arange``. Both appear only as grid extents and stride multipliers -- e.g.
``grid = lambda META: [triton.cdiv(L, META["BLOCK_M1"]), B * H, L]`` in every forward launcher -- so
there is no boundary mask on either axis to exercise. (``ragged``'s ``floor=16`` would also RAISE
them, 4 -> 16, rather than make any tail partial.)
"""

from __future__ import annotations

import torch

from .drivers import BF16, TensorKw, aligned_only, dev, driver_length, ragged

L = ragged(driver_length(128))   # sequence length: tiles in BOTH the query loop and the key/value loop
H = 4             # d_pair (128) // head dim (32); a grid extent, never a tl.arange block
D = ragged(32)    # head dim, masked against HEAD_DIM inside the HEAD_DIM_PAD block
A = 8             # augmentation dim, from bench_kernel_aug_attn; a grid extent, never blocked

#: The atomic triangle path refuses any other head dim, so it keeps a power-of-two 32.
D32 = aligned_only(
    "attn.triangle_atomic.head_dim",
    32,
    "triangle_attention/triton/atomic.py:518-519 -- forward raises "
    'ValueError(f"Only support D=32, but got {D=}") for every other head dim, so the atomic '
    "fwd/bwd_pre/bwd drivers cannot reach their kernels at any other D",
)

DH = ragged(128)          # d_hidden: gate/out_r width == the gate-out GEMM's contraction
DP = ragged(128, by=5)    # d_pair: the gate-out GEMM's output width N == wo.shape[0]


def _grad(out: torch.Tensor) -> None:
    """Reach the backward kernels with a dense grad (see the module docstring)."""
    out.backward(torch.randn_like(out))


def _tri_qkvb(d: int = D) -> tuple[torch.Tensor, ...]:
    """q/k/v [1, H, L, L, d] and bias [1, H, L, L] -- bench_kernel_tri_attn.

    ``d`` is the head dim: ``D`` (perturbable) for ``triton/main.py``, ``D32`` for
    ``triton/atomic.py``, whose forward rejects anything else.
    """
    kw: TensorKw = {"device": dev(), "dtype": BF16, "requires_grad": True}
    q, k, v = (torch.randn(1, H, L, L, d, **kw) for _ in range(3))
    return q, k, v, torch.randn(1, H, L, L, **kw)


def _aug_qkvb() -> tuple[torch.Tensor, ...]:
    """q/k/v [A, 1, L, H, D] and bias [1, L, L, H] -- bench_kernel_aug_attn."""
    kw: TensorKw = {"device": dev(), "dtype": BF16, "requires_grad": True}
    q, k, v = (torch.randn(A, 1, L, H, D, **kw) for _ in range(3))
    return q, k, v, torch.randn(1, L, L, H, **kw)


def _bias_only_vb() -> tuple[torch.Tensor, ...]:
    """v [1, H, L, L, D] and bias [1, H, L, L] -- bench_kernel_bias_attn."""
    kw: TensorKw = {"device": dev(), "dtype": BF16, "requires_grad": True}
    return torch.randn(1, H, L, L, D, **kw), torch.randn(1, H, L, L, **kw)


# ── triangle_attention / triton / main.py ───────────────────────────────────────────────────


def triangle_attention_fwd_triton() -> None:
    from .triangle_attention.triton.main import TritonTriangleAttentionPairBiasFunction as Fn
    Fn.apply(*_tri_qkvb())


def _tri_main_backward() -> None:
    from .triangle_attention.triton.main import TritonTriangleAttentionPairBiasFunction as Fn
    _grad(Fn.apply(*_tri_qkvb()))


def triangle_attention_bwd_pre_triton() -> None:
    _tri_main_backward()


def triangle_attention_bwd_dkdv_triton() -> None:
    _tri_main_backward()


def triangle_attention_bwd_dq_triton() -> None:
    _tri_main_backward()


# ── triangle_attention / triton / atomic.py ─────────────────────────────────────────────────


def triangle_attention_fwd_contig_triton() -> None:
    from .triangle_attention.triton.atomic import TritonTriangleAttentionPairBiasFunction as Fn
    Fn.apply(*_tri_qkvb(D32))   # this file's forward raises on any head dim but 32


def _tri_atomic_backward() -> None:
    from .triangle_attention.triton.atomic import TritonTriangleAttentionPairBiasFunction as Fn
    _grad(Fn.apply(*_tri_qkvb(D32)))


def triangle_attention_bwd_pre_contig_triton() -> None:
    _tri_atomic_backward()


def triangle_attention_bwd_atomic_triton() -> None:
    _tri_atomic_backward()


# ── augmented_attention / triton / main.py ──────────────────────────────────────────────────


def augmented_attention_fwd_triton() -> None:
    from .augmented_attention.triton.main import TritonAugmentedAttentionFunction as Fn
    Fn.apply(*_aug_qkvb())


def _aug_main_backward() -> None:
    from .augmented_attention.triton.main import TritonAugmentedAttentionFunction as Fn
    _grad(Fn.apply(*_aug_qkvb()))


def augmented_attention_bwd_pre_triton() -> None:
    _aug_main_backward()


def augmented_attention_bwd_split_triton() -> None:
    _aug_main_backward()


def augmented_attention_bwd_reduce_triton() -> None:
    _aug_main_backward()


# ── augmented_attention / triton / memory_efficient.py ──────────────────────────────────────


def augmented_attention_bwd_atomic_triton() -> None:
    # This file's forward and bwd_preprocess now come from main.py; the atomic dq/dk/dv/dbias
    # backward below is the only kernel it still defines.
    from .augmented_attention.triton.memory_efficient import TritonAugmentedAttentionFunction as Fn
    _grad(Fn.apply(*_aug_qkvb()))


# ── bias_only_attention / triton / main.py ──────────────────────────────────────────────────


def bias_only_attention_fwd_triton() -> None:
    from .bias_only_attention.triton.main import TritonBiasOnlyAttentionFunction as Fn
    Fn.apply(*_bias_only_vb())


def _bias_only_backward() -> None:
    from .bias_only_attention.triton.main import TritonBiasOnlyAttentionFunction as Fn
    _grad(Fn.apply(*_bias_only_vb()))


def bias_only_attention_bwd_pre_triton() -> None:
    _bias_only_backward()


def bias_only_attention_bwd_triton() -> None:
    _bias_only_backward()


# ── bias_only_attention / triton / gate_out.py ──────────────────────────────────────────────


def _pair_key() -> int:
    """``shape_key`` for the two gate-out launchers, computed the way the module computes it.

    ``_fwd`` and ``_dgrad_epilogue`` are INNER launchers: by the time they are called the
    activation is the flattened ``(M, DH)`` matrix and L is gone, so per
    ``autotune/shape_key.py::length_of`` they cannot derive the key themselves -- both take it as
    ``shape_key=`` and both fall back to ``token_key(0)`` (the BOTTOM bucket, 128) when it is
    omitted. The drivers below used to omit it, which pinned every driver length to bucket 128.
    Calling the module's own ``_key_of`` on the PRE-flatten pair shape ``(1, L, L, DH)`` is exactly
    what ``_FusedGateOut.forward``/``backward`` do, so the driver now records the bucket production
    records at this L.
    """
    from .bias_only_attention.triton.gate_out import _key_of
    return _key_of((1, L, L, DH))


def gated_projection_gate_gemm_triton() -> None:
    # M = L*L rows, DH the contraction, DP the output width -- all three tile, all three ragged.
    from .bias_only_attention.triton.gate_out import _fwd
    kw: TensorKw = {"device": dev(), "dtype": BF16}
    gate = torch.randn(L * L, DH, **kw)
    out_r = torch.randn(L * L, DH, **kw)
    wo = torch.randn(DP, DH, **kw)          # to_out.weight [d_pair, d_hidden]
    _fwd(gate, out_r, wo, shape_key=_pair_key())


def gated_projection_bwd_dx_triton() -> None:
    from .bias_only_attention.triton.gate_out import _dgrad_epilogue
    kw: TensorKw = {"device": dev(), "dtype": BF16}
    do2 = torch.randn(L * L, DP, **kw)      # grad wrt [M, N], N == d_pair
    wo = torch.randn(DP, DH, **kw)
    g2 = torch.randn(L * L, DH, **kw)
    r2 = torch.randn(L * L, DH, **kw)
    _dgrad_epilogue(do2, wo, g2, r2, shape_key=_pair_key())
