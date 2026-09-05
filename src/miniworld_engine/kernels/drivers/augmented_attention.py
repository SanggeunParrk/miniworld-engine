"""Drivers for the ``augmented_attention`` family.

The three attention families were one module (``drivers_attn.py``) and still share the
``L``/``H``/``D`` extents, which live in ``drivers/triangle_attention.py`` together with the
shape, grad and tile-alignment rationale for all three. ``A`` (the augmentation dim) is this
family's own and stays here.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import BF16, TensorKw, _grad, dev, driver_width, ragged
from miniworld_engine.kernels.drivers.triangle_attention import L

A = 8             # augmentation dim, from bench_kernel_aug_attn; a grid extent, never blocked

#: This family does NOT share triangle_attention's `H`/`D`, and sharing them was a measured bug.
#: That block is `H = driver_width(128) // 32, D = 32` -- head COUNT derived from the width at a
#: FIXED head dim -- which is right for triangle_attention (d_pair 128, 4 heads of 32) and wrong
#: here in the other direction: the DiT fixes the head count and lets the head dim follow d_single.
#: `builder.cases()` states it -- AugmentedAttentionPairBias at `d_single` 384 and 768, both with
#: `n_head: 16` -- so production runs HEAD_DIM 24 and 48. The shared block built (H=12, D=32) and
#: (H=24, D=32) instead: three plausible-looking buckets, none of them one the model ever asks for.
#: `dev audit --replay` measured 60 misses across this family's three kernels, every one of them
#: `(H=16, HEAD_DIM=24)` or `(H=16, HEAD_DIM=48)`.
#:
#: The atom side keeps the other rule: `d_single_atom` is 128 and the atom DiT runs 4 heads of 32,
#: which is what the width-derived form gives -- so it is kept for that width rather than replaced.
_N_HEAD_TOKEN = 16                        # cases(): AugmentedAttentionPairBias(n_head=16)
_W = driver_width(128)
H = _N_HEAD_TOKEN if _W >= 384 else _W // 32
D = ragged(_W // H)


def _aug_qkvb() -> tuple[torch.Tensor, ...]:
    """q/k/v [A, 1, L, H, D] and bias [1, L, L, H] -- bench_kernel_aug_attn."""
    kw: TensorKw = {"device": dev(), "dtype": BF16, "requires_grad": True}
    q, k, v = (torch.randn(A, 1, L, H, D, **kw) for _ in range(3))
    return q, k, v, torch.randn(1, L, L, H, **kw)


# ── augmented_attention / triton / main.py ──────────────────────────────────────────────────


def augmented_attention_fwd_triton() -> None:
    from miniworld_engine.kernels.augmented_attention.triton.main import (
        TritonAugmentedAttentionFunction as Fn,
    )
    Fn.apply(*_aug_qkvb())


def _aug_main_backward() -> None:
    from miniworld_engine.kernels.augmented_attention.triton.main import (
        TritonAugmentedAttentionFunction as Fn,
    )
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
    from miniworld_engine.kernels.augmented_attention.triton.memory_efficient import (
        TritonAugmentedAttentionFunction as Fn,
    )
    _grad(Fn.apply(*_aug_qkvb()))
