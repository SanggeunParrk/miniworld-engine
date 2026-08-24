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


from .triton.main import triton_tm2

__all__ = ["tm2_cute", "triton_tm2"]


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

    D = x_gate.shape[-1]
    M = x_gate.numel() // D
    N, K = int(Wg_nk.shape[0]), int(Wg_nk.shape[1])
    tile_m = _resolve_tm2_tile_m(M, N, K, str(x_gate.dtype), x_gate.device)
    return mod.tm2_dual_from_scratch(x_gate, x_out, Wg_nk, Wo_nk, tile_m=tile_m)


def _tm2_smem_bytes(tile_m: int, N: int, K: int) -> int:
    """SMEM footprint of the tm2 kernel for a candidate tile_m (bf16, TILE_K=64, 2 K-stages).
    sX1+sX2 scale with tile_m; sW1+sW2 with N; sO with tile_m*N; +8B mbarrier +align slack."""
    tk, kloop = 64, K // 64
    b = 2  # bf16
    sX = tile_m * tk * kloop * b
    sW = N * tk * kloop * b
    sO = tile_m * N * b
    return 2 * sX + 2 * sW + sO + 8 + 4 * 1024  # +mbar +alignment padding headroom


# H100/sm90 hardware SMEM ceiling (opt-in max). Candidates above this are unlaunchable.
_SM90_SMEM_MAX = 232448


def _largest_valid_tile_m(M: int, N: int, K: int) -> int:
    """Largest tile_m in {256,192,128,64} that divides M and fits sm90 SMEM (falls to 64)."""
    for tm in (256, 192, 128, 64):
        if M % tm == 0 and _tm2_smem_bytes(tm, N, K) <= _SM90_SMEM_MAX:
            return tm
    return 64


def _resolve_tm2_tile_m(M: int, N: int, K: int, dtype: str, device) -> int:
    """Pick the autotuned tile_m for this shape; fall back to the largest valid divisor.

    The cache is bucketed by M, and buckets group multiple M — so a cached tile_m may not
    divide *this* M or may exceed SMEM. Since tile_m is performance-only, we validate the
    resolved pick and fall back to a guaranteed-valid default, never sacrificing correctness."""
    default = _largest_valid_tile_m(M, N, K)
    try:
        from miniworld_engine.autotune.buckets import bucket_mixed
        from miniworld_engine.autotune.cute_config import resolve_config, tm2_candidates
        from quack.gemm_config import GemmConfig

        dev_index = device.index if getattr(device, "index", None) is not None else 0
        cfg = resolve_config(
            "tm2_dual_fwd", tm2_candidates(), dtype=dtype,
            bucket=f"{bucket_mixed(M)}|k{K}",
            default=GemmConfig(tile_m=default, tile_n=N, pingpong=False,
                               is_dynamic_persistent=False, cluster_m=1, cluster_n=1,
                               swap_ab=False, max_swizzle_size=8, device_capacity=9),
            device_index=dev_index,
        )
        tm = int(cfg.tile_m)
    except Exception:  # noqa: BLE001 -- any resolve failure -> safe default
        return default
    # Validate the cached pick against THIS shape (divisibility + SMEM); else safe default.
    if M % tm == 0 and _tm2_smem_bytes(tm, N, K) <= _SM90_SMEM_MAX:
        return tm
    return default
