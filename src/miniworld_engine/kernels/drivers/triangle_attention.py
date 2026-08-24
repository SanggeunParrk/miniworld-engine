"""Drivers for the ``triangle_attention`` family -- and the shape block the other two attention
families import.

The three attention families were one module (``drivers_attn.py``) and still share ``L``/``H``/
``D`` (and ``drivers._grad``); the block lives here because triangle_attention has the most
kernels reading it. ``drivers/augmented_attention.py`` and ``drivers/bias_only_attention.py``
import it from here, and everything below applies to all three.

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

from miniworld_engine.kernels.drivers import (
    BF16,
    TensorKw,
    _grad,
    aligned_only,
    dev,
    driver_length,
    ragged,
)

L = ragged(driver_length(128))   # sequence length: tiles in BOTH the query loop and the key/value loop
H = 4             # d_pair (128) // head dim (32); a grid extent, never a tl.arange block
D = ragged(32)    # head dim, masked against HEAD_DIM inside the HEAD_DIM_PAD block

#: The atomic triangle path refuses any other head dim, so it keeps a power-of-two 32.
D32 = aligned_only(
    "attn.triangle_atomic.head_dim",
    32,
    "triangle_attention/triton/atomic.py:518-519 -- forward raises "
    'ValueError(f"Only support D=32, but got {D=}") for every other head dim, so the atomic '
    "fwd/bwd_pre/bwd drivers cannot reach their kernels at any other D",
)


def _tri_qkvb(d: int = D) -> tuple[torch.Tensor, ...]:
    """q/k/v [1, H, L, L, d] and bias [1, H, L, L] -- bench_kernel_tri_attn.

    ``d`` is the head dim: ``D`` (perturbable) for ``triton/main.py``, ``D32`` for
    ``triton/atomic.py``, whose forward rejects anything else.
    """
    kw: TensorKw = {"device": dev(), "dtype": BF16, "requires_grad": True}
    q, k, v = (torch.randn(1, H, L, L, d, **kw) for _ in range(3))
    return q, k, v, torch.randn(1, H, L, L, **kw)


# ── triangle_attention / triton / main.py ───────────────────────────────────────────────────


def triangle_attention_fwd_triton() -> None:
    from miniworld_engine.kernels.triangle_attention.triton.main import (
        TritonTriangleAttentionPairBiasFunction as Fn,
    )
    Fn.apply(*_tri_qkvb())


def _tri_main_backward() -> None:
    from miniworld_engine.kernels.triangle_attention.triton.main import (
        TritonTriangleAttentionPairBiasFunction as Fn,
    )
    _grad(Fn.apply(*_tri_qkvb()))


def triangle_attention_bwd_pre_triton() -> None:
    _tri_main_backward()


def triangle_attention_bwd_dkdv_triton() -> None:
    _tri_main_backward()


def triangle_attention_bwd_dq_triton() -> None:
    _tri_main_backward()


# ── triangle_attention / triton / atomic.py ─────────────────────────────────────────────────


def triangle_attention_fwd_contig_triton() -> None:
    from miniworld_engine.kernels.triangle_attention.triton.atomic import (
        TritonTriangleAttentionPairBiasFunction as Fn,
    )
    Fn.apply(*_tri_qkvb(D32))   # this file's forward raises on any head dim but 32


def _tri_atomic_backward() -> None:
    from miniworld_engine.kernels.triangle_attention.triton.atomic import (
        TritonTriangleAttentionPairBiasFunction as Fn,
    )
    _grad(Fn.apply(*_tri_qkvb(D32)))


def triangle_attention_bwd_pre_contig_triton() -> None:
    _tri_atomic_backward()


def triangle_attention_bwd_atomic_triton() -> None:
    _tri_atomic_backward()
