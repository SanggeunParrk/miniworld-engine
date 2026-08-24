"""Accuracy checks for the ``fused_ln_mask`` family.

layernorm, layernorm_linear and fused_ln_mask were one module (``checks_ln.py``). The two rules
these references follow -- built from the same inputs the kernel saw, in fp32, and fed the same
saved statistics the kernel consumed -- are written out in ``checks/layernorm_linear.py``; the
helpers all three use are in ``checks/__init__.py``.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.checks import _EPS
from miniworld_engine.kernels.drivers import BF16, dev, pair, vec
from miniworld_engine.kernels.drivers.layernorm_linear import _D, _PAIR_N

# ── fused_ln_mask ────────────────────────────────────────────────────────────────────────────


def layernorm_fwd_rowscale_triton():
    """_fused_ln_mask_kernel: LN over the D axis, then a PER-ROW multiply by the pair mask.

    What the fold actually does, checked before writing the reference: the mask is (B, L, L) --
    ONE scalar per row of the flattened (B*L*L, D) matrix -- and the multiply happens AFTER the
    affine, so a masked-out row is exactly zero including the LayerNorm beta, and the mask never
    enters the D reduction. That is the math ``fused_ln_mask/reference.py`` states, so the
    reference is that function rather than a rewrite: it reduces in fp32 and reproduces the
    launcher's cast of the mask to x.dtype before the multiply.
    """
    from miniworld_engine.kernels.fused_ln_mask.cute.fused_ln_mask import fused_ln_mask
    from miniworld_engine.kernels.fused_ln_mask.reference import fused_ln_mask_pytorch

    x, w, b = pair(n=_PAIR_N, d=_D), vec(_D), vec(_D)
    mask = (torch.rand(*x.shape[:-1], device=dev()) > 0.1).to(BF16)  # (B, L, L), as the driver
    out = fused_ln_mask(x, w, b, mask, _EPS)
    ref = fused_ln_mask_pytorch(x, w, b, mask, _EPS)
    return out, ref
