"""Precision of the bias-only token DiT's pair-bias logits ``to_bias(LN(pair))`` (``TokenDiTMiniWorldBiasOnly``): two modes.

``"fp32"`` (the default, and the only mode of training)
    LN0(pair) in fp32, the gamma-folded weights unrounded fp32, fp32 logits and an fp32 softmax; only the finished attention weights P are
    rounded to the activation dtype. User rule 2026-11-19: the logits are 1e3 .. 2e5 with a key spread of ~500, bf16 resolves 32 .. 1024
    there and broke the peptide C-N bonds. Every backend and card (H100 / A100 / 5090 ``cuda``, ``triton``, ``pytorch``) does this.

``"bf16"`` (opt-in, inference only)
    The logits are computed and stored in bf16 (LN0 output and the folded weights rounded to bf16, bf16 GEMM with fp32 accumulation, bf16
    store: the engine's bias-only path before 2026-11-19), the softmax runs in fp32 over those bf16 values and P is rounded to bf16 as before.
    Half the logit bytes and one pass less over them: faster and smaller at long L (docs/release-notes/, ``benchmarks``). A model trained
    with the fp32 logits showed no problem when its inference ran with bf16 logits (user experiment 2026-11-21); the engine does not
    re-validate that at model level.

    Only bf16 activations take it (fp32 blocks keep fp32 logits), and only calls that need no gradient: with autograd on (grad enabled and an
    input or parameter requires grad) the call runs the fp32 mode silently, so a training step is exactly the default whatever the switch.

The switch (per module, else process wide, else the environment, else ``"fp32"``)::

    block.set_pair_bias_logits("bf16")          # one TokenDiTMiniWorldBiasOnly (or BiasOnlyAttention); None: follow the process default
    foldjet_engine.bias_logits.set_default("bf16")   # every module that has not set its own; None: back to the environment
    FOLDJET_ENGINE_BIAS_LOGITS=bf16 python run.py     # the environment default (read at each call)
    with foldjet_engine.bias_logits.scope("bf16"): ...   # a block of code

:func:`pair_bias_all` follows the mode of its first block. The ``fp32`` mode is bit-identical to the engine before the switch existed, and the
mode is part of every cache key that holds a P or a bias, so switching at run time never serves the other mode's tensors.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

import torch

import foldjet_engine._envcompat  # noqa: F401  (old MINIWORLD_* environment names)

ENV = "FOLDJET_ENGINE_BIAS_LOGITS"
MODES = ("fp32", "bf16")
_DEFAULT: list[str | None] = [None]


def _check(mode: str, where: str) -> str:
    m = mode.strip().lower() if isinstance(mode, str) else mode
    if m not in MODES:
        raise ValueError(f"{where}: pair-bias logits mode {mode!r} (use one of {', '.join(MODES)})")
    return m


def default() -> str:
    """The process default: :func:`set_default`'s, else ``FOLDJET_ENGINE_BIAS_LOGITS`` (``fp32`` / ``bf16``; empty = unset), else ``"fp32"``."""
    if _DEFAULT[0] is not None:
        return _DEFAULT[0]
    v = os.environ.get(ENV, "").strip()
    return _check(v, ENV) if v else "fp32"


def set_default(mode: str | None) -> None:
    """Process-wide default for every module that has not set its own; ``None`` hands it back to ``FOLDJET_ENGINE_BIAS_LOGITS``."""
    _DEFAULT[0] = None if mode is None else _check(mode, "set_default")


@contextlib.contextmanager
def scope(mode: str) -> Iterator[None]:
    """``set_default(mode)`` for the duration of a ``with`` block."""
    old = _DEFAULT[0]
    set_default(mode)
    try:
        yield
    finally:
        _DEFAULT[0] = old


def requested(module_mode: str | None) -> str:
    """The mode a module asks for: its own setting, else the process default."""
    return default() if module_mode is None else _check(module_mode, "set_pair_bias_logits")


def effective(module_mode: str | None, *, grad: bool, dtype: torch.dtype) -> str:
    """The mode a call runs: ``"bf16"`` only when asked for, no autograd (``grad`` False) and bf16 activations; else ``"fp32"``."""
    if grad or dtype is not torch.bfloat16:
        return "fp32"
    return requested(module_mode)


__all__ = ["ENV", "MODES", "default", "effective", "requested", "scope", "set_default"]
