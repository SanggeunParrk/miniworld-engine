"""Drivers for the ``tm2`` family.

trimul_inproj, tm1, tm2 and gated_projection were one module (``drivers_trimul.py``) and still
share ``D``/``L``/``IS_PAIR``/``M`` and the ``_x``/``_rows``/``_w``/``_bdll`` builders, which
live in ``drivers/trimul_inproj.py`` together with the shape and lazy-import rationale for all
four.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import BF16, aligned_only, dev
from miniworld_engine.kernels.drivers.trimul_inproj import _w, _x

# ── tm2 ──────────────────────────────────────────────────────────────────────────────────────

def trimul_outproj_gemm_gate_triton() -> None:
    """fused_sigmoid_gate2_fwd_kernel, via triton_tm2."""
    from miniworld_engine.kernels.tm2.triton.main import triton_tm2

    # _x(), not _rows(): TritonTM2Function reads token_key(length_of(x.shape)) before its own
    # rearrange, so a pre-flattened (M, d) gives it M = L*L -> clamped to 512.
    triton_tm2(_x(), _x(), _w(), _w())


def trimul_outproj_bwd_gate_recompute_triton() -> None:
    """fused_sigmoid_gate2_bwd_kernel, via TritonTM2Function.backward."""
    from miniworld_engine.kernels.tm2.triton.main import triton_tm2

    # _x(): same reason as the forward -- ctx.original_shape is what the backward keys on.
    x, y = _x().requires_grad_(), _x().requires_grad_()
    triton_tm2(x, y, _w(), _w()).sum().backward()


def tm2_dual_kernel() -> None:
    """TM2DualKernel.kernel, via tm2_dual_from_scratch (weights in (N,K) form, as the bench).

    This driver builds its OWN x/W instead of ``_x()``/``_w()``: both of this kernel's extents are
    pinned by asserts the launcher writes down itself, so neither can carry the ragged tail. The
    two ``aligned_only`` calls below record that, quoting the asserts.
    """
    from miniworld_engine.kernels.tm2.cute.tm2_cute_kernel import tm2_dual_from_scratch

    d = aligned_only(
        "tm2.trimul_outproj_gemm_gate_sm90_cute.K",
        128,
        "tm2/cute/tm2_cute_kernel.py:66 `assert K % _TILE_K == 0, f\"K={K} must be divisible by "
        "TILE_K={_TILE_K}\"`, with `_TILE_K = 64` at tm2_cute_kernel.py:49 and "
        "`self.k_loop = K // _TILE_K` at :72 -- the K-stage count is an exact division, so the "
        "channel width K = d_pair must be a multiple of 64",
    )
    lseq = aligned_only(
        "tm2.trimul_outproj_gemm_gate_sm90_cute.M",
        64,
        "tm2/cute/tm2_cute_kernel.py:406 `assert M % tile_m == 0, f\"M={M} must be divisible by "
        "tile_m={tile_m}\"`; tile_m is chosen at :404 as the largest of (256,192,128,64) dividing "
        "M and falls back to 64, so M = L*L must be a multiple of 64. Perturbing L to 61 makes "
        "that assert the launcher's first statement to fail (AssertionError: M=3721 must be "
        "divisible by tile_m=64), which is the assert speaking, not a masked tail",
    )
    x1 = torch.randn(1, lseq, lseq, d, device=dev(), dtype=BF16)
    x2 = torch.randn(1, lseq, lseq, d, device=dev(), dtype=BF16)
    wg = (torch.randn(d, d, device=dev(), dtype=BF16) * (d**-0.5)).contiguous()
    wp = (torch.randn(d, d, device=dev(), dtype=BF16) * (d**-0.5)).contiguous()
    tm2_dual_from_scratch(x1, x2, wg, wp)
