"""Numerically dead module instances of a checkpoint: skipped at inference.

Some released checkpoints carry residual modules whose every parameter has decayed to about 1e-37 (Protenix v2: 41 instances --
all four confidence-head blocks' triangle attentions, two triangle multiplications and a pair transition, and trunk attention-pair-bias
/ single transitions).  Such a module's update is exactly zero in bf16 and fp32, so ``x + update`` is ``x``; the Protenix-v2 kit
skips them (its ``deadskip`` manifest: every instance probed, output all zeros and ``x + out == x`` bitwise), which made its
confidence head about half of FoldForge's at 512-1536 tokens (5I28, 2026-10-29).

``mark_dead_modules(model)`` flags every such instance after the weights are loaded; a flagged module returns its residual input
unchanged when it runs without grad outside training (and runs normally otherwise: its parameters still train).  The flag is a
plain attribute, so torch.compile guards on it and CUDA graphs capture the skip.  Call it again after loading other weights, or
``clear_dead_modules(model)`` to run every module.
"""

from __future__ import annotations

import torch
from torch import nn

#: A module is dead when the largest |parameter| of every tensor it owns is below this (Protenix v2's are ~5e-37; the kit's bound).
THRESHOLD = 1e-30


def _skippable() -> tuple[type, ...]:
    """The residual modules whose forward honours ``dead`` (each returns its residual input when flagged)."""
    from miniworld_engine.modules.attention_pair_bias import AttentionPairBias
    from miniworld_engine.modules.msa_pair_weighted_averaging import (
        MSAPairWeightedAveraging,
    )
    from miniworld_engine.modules.outer_product import OuterProductMean
    from miniworld_engine.modules.transition import Transition
    from miniworld_engine.modules.triangle_attention import TriangleAttention
    from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication

    return (
        AttentionPairBias,
        MSAPairWeightedAveraging,
        OuterProductMean,
        Transition,
        TriangleAttention,
        TriangleMultiplication,
    )


@torch.no_grad()
def is_dead(module: nn.Module, threshold: float = THRESHOLD) -> bool:
    """Every parameter of ``module`` (recursively) below ``threshold`` in absolute value (a module without parameters is not)."""
    params = list(module.parameters())
    return bool(params) and all(float(p.detach().abs().max()) < threshold for p in params if p.numel())


def mark_dead_modules(model: nn.Module, threshold: float = THRESHOLD) -> list[str]:
    """Flag the dead instances of the skippable modules in ``model`` (and clear the flag on the others); returns their names."""
    kinds = _skippable()
    dead = []
    for name, module in model.named_modules():
        if isinstance(module, kinds):
            module.dead = is_dead(module, threshold)
            if module.dead:
                dead.append(name)
    return dead


def clear_dead_modules(model: nn.Module) -> None:
    """Run every module again."""
    kinds = _skippable()
    for module in model.modules():
        if isinstance(module, kinds):
            module.dead = False
