"""Automatic per-architecture dispatch for the triangle_multiplication cute path.

The ``miniworld`` (auto) implementation selects the best correct backend for the
running GPU, with no manual flag — following the repo's dispatch policy
(``docs/operations/dispatch-cache.md``): env override → static arch heuristic.

Unlike the layernorm / bias-only caches (which calibrate a *shape crossover* and
persist a per-GPU JSON), the trimul choice is an **architecture capability**
choice, not a shape crossover:

  * sm_100 (Blackwell / B200) → our from-scratch tcgen05 kernel (``bdll_sm100``),
    the measured winner here (beats cuequiv/dtv1 under compiled + CUDA-graph).
  * sm_90 (Hopper / H100) and other cute-capable archs → the quack M-major path
    (``bdll_direct``), the pre-existing best for those archs.
  * pre-Hopper (no tcgen05 / no supported cute GEMM) → the triton path.

Both cute out_layouts compute the SAME math (bf16 in / fp32 acc / bf16 out); the
choice is pure performance policy, so a wrong pick can only be slower, never
incorrect. Because the winner is fixed by arch capability (not by shape), no
per-shape calibration/cache is needed; the static heuristic IS the policy.

Env overrides (debug / manual pin):
  MINIWORLD_TRIMUL_IMPL       = cute | triton | pytorch | cuequivariance
  MINIWORLD_TRIMUL_OUT_LAYOUT = bdll_sm100 | bdll_direct | bdll | blld
"""

from __future__ import annotations

import os

import torch

from miniworld_kernels.modules.exceptions import ImplementationType


def _capability(device: torch.device | None = None) -> tuple[int, int]:
    if not torch.cuda.is_available():
        return (0, 0)
    idx = None
    if device is not None and device.type == "cuda":
        idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()
    return torch.cuda.get_device_capability(idx)


def resolve_impl(
    requested: ImplementationType,
    device: torch.device | None = None,
) -> ImplementationType:
    """Resolve the ``miniworld`` (auto) implementation to a concrete backend.

    Non-``miniworld`` requests pass through unchanged.
    """
    if requested != ImplementationType.MINIWORLD:
        return requested
    override = os.environ.get("MINIWORLD_TRIMUL_IMPL")
    if override:
        return ImplementationType(override.strip().lower())
    major, _ = _capability(device)
    # Hopper (sm_90) and Blackwell (sm_100) both run a cute path; the exact cute
    # kernel is chosen per-arch by resolve_out_layout(). Older archs -> triton.
    if major >= 9:
        return ImplementationType.CUTE
    return ImplementationType.TRITON


def resolve_out_layout(device: torch.device | None = None) -> str:
    """Pick the cute tm1 ``out_layout`` for the running GPU.

    sm_100 → ``bdll_sm100`` (our from-scratch tcgen05 gate GEMM);
    else    → ``bdll_direct`` (quack M-major, the H100/pre-existing path).
    """
    override = os.environ.get("MINIWORLD_TRIMUL_OUT_LAYOUT")
    if override:
        return override.strip()
    major, _ = _capability(device)
    if major >= 10:
        return "bdll_sm100"
    return "bdll_direct"
