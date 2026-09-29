"""Drivers for the ``fused_ln_mask`` family.

fused_ln_mask, layernorm and layernorm_linear were one module (``drivers_ln.py``) and still
share the ``_D``/``_act`` shape block, which lives in ``drivers/layernorm_linear.py``.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import BF16, dev, norm_affine
from miniworld_engine.kernels.drivers.layernorm_linear import _D, _act

# ── fused_ln_mask ────────────────────────────────────────────────────────────────────────────


def layernorm_fwd_rowscale_triton() -> None:
    from miniworld_engine.kernels.fused_ln_mask.cute.fused_ln_mask import fused_ln_mask

    x = _act()
    mask = (torch.rand(*x.shape[:-1], device=dev()) > 0.1).to(BF16)  # (B, L, L), required
    # fp32 gamma/beta: this LN's affine comes from `primitives.LayerNorm` -- see
    # `drivers.norm_affine`. Replay showed production keying `bfloat16+float32` here
    # against a `bfloat16`-only cache, i.e. every masked-LN launch missing on dtype alone.
    fused_ln_mask(x, norm_affine(_D), norm_affine(_D), mask, 1e-5)
