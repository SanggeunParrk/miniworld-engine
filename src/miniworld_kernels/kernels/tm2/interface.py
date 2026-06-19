"""CuTeDSL entry point for tm2.

Dispatches to the from-scratch fused dual-A SM90 kernel co-located in
``kernels/tm2/cute/tm2_cute_kernel.py``. The weights coming in match the
PyTorch reference (``Wg``/``Wo`` in ``(D, D)`` matmul form, i.e. ``x @ W``),
so we transpose to nn.Linear ``(N, K)`` form for the kernel.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_CUTE_DIR = Path(__file__).resolve().parent / "cute"


def _load_kernel_module():
    """Load ``tm2_cute_kernel`` from the co-located cute env without polluting sys.path."""
    mod_name = "_tm2_cute_kernel_impl"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = _CUTE_DIR / "tm2_cute_kernel.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None, f"failed to spec {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def tm2_cute(
    x_gate: torch.Tensor,
    x_out: torch.Tensor,
    W_gate: torch.Tensor,
    W_out: torch.Tensor,
) -> torch.Tensor:
    """tm2 forward: sigmoid(x_gate @ W_gate) * (x_out @ W_out).

    Args
    ----
    x_gate, x_out : (..., D), contiguous bf16
    W_gate, W_out : (D, D)   — matmul form (the reference is ``x @ W``)

    Returns
    -------
    out : (..., D), bf16
    """
    assert x_gate.shape == x_out.shape
    assert W_gate.shape == W_out.shape
    assert x_gate.is_contiguous() and x_out.is_contiguous()
    # Kernel wants weights in (N, K) nn.Linear form (last-dim contiguous = K).
    # The reference uses ``x @ W`` with W as (K, N), so transpose here.
    Wg_nk = W_gate.t().contiguous()
    Wo_nk = W_out.t().contiguous()
    mod = _load_kernel_module()
    return mod.tm2_dual_from_scratch(x_gate, x_out, Wg_nk, Wo_nk)
