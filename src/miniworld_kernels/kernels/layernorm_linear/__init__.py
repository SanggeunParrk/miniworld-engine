"""Fused LayerNorm + Linear (`te.LayerNormLinear` analogue).

LayerNorm over the last dim immediately followed by a Linear (GEMM + bias).
``reference.py`` holds the PyTorch math (and the ``torch.compile`` baseline);
the **cute** backend (``cute/``) is the SM90/Hopper fast path (forks quack's
``GemmSm90`` — WGMMA + TMA + clusters), and the **Triton** backend
(``triton/fused.py``) is the portable fallback for any other arch. ``layernorm_linear``
dispatches by GPU capability. See README.md.
"""

from __future__ import annotations

import torch

from .autograd import LayerNormLinearFn, layernorm_linear_fn
from .interface import layernorm_linear_triton
from .reference import LayerNormLinearRef, layernorm_linear_pytorch

__all__ = [
    "LayerNormLinearFn",
    "LayerNormLinearRef",
    "layernorm_linear",          # hardware-dispatched inference: SM90 cute fast path, else Triton
    "layernorm_linear_fn",       # trainable (autograd); v1 SM90/Hopper only
    "layernorm_linear_pytorch",
    "layernorm_linear_triton",
]


def layernorm_linear(x, ln_weight, ln_bias, weight, bias, eps: float = 1e-5, *,
                     save_stats: bool = False, prefolded=None):
    """Forward LayerNormLinear, dispatched by GPU capability.

    **SM90 (Hopper: H100/H200)** -> the cute fast path: fused M2 for N<=256 / M1 otherwise,
    with autotuned configs (and ``save_stats=True`` returns ``(Y, mean, rstd)`` via M1; see
    ``cute/__init__.py``). The cute backend (quack/WGMMA/TMA) is imported lazily so this
    package still imports on non-Hopper / non-cute machines.

    **Any other arch** (Ampere/Ada/Blackwell/ROCm) -> the portable Triton fallback
    (``triton/fused.py``). ``prefolded`` is cute-only and ignored here; ``save_stats=True``
    returns ``(Y, mean, rstd)`` with stats computed alongside.
    """
    if torch.cuda.is_available() and torch.cuda.get_device_capability(x.device)[0] == 9:
        from .cute import layernorm_linear as _cute_dispatch

        return _cute_dispatch(
            x, ln_weight, ln_bias, weight, bias, eps, save_stats=save_stats, prefolded=prefolded
        )

    # --- portable Triton fallback (non-Hopper) ---
    y = layernorm_linear_triton(x, ln_weight, ln_bias, weight, bias, eps)
    if save_stats:
        xf = x.reshape(-1, x.shape[-1]).float()
        mean = xf.mean(-1)
        rstd = torch.rsqrt(xf.var(-1, unbiased=False) + eps)
        return y, mean, rstd
    return y
