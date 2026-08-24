"""Drivers for the ``augmented_attention`` family.

The three attention families were one module (``drivers_attn.py``) and still share the
``L``/``H``/``D`` extents, which live in ``drivers/triangle_attention.py`` together with the
shape, grad and tile-alignment rationale for all three. ``A`` (the augmentation dim) is this
family's own and stays here.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import BF16, TensorKw, _grad, dev
from miniworld_engine.kernels.drivers.triangle_attention import D, H, L

A = 8             # augmentation dim, from bench_kernel_aug_attn; a grid extent, never blocked


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
