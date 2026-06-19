"""tm2 cute forward.

Math identical to ``team_gm.modules.kernels.tm2.triton_tm2``:

    out = sigmoid(x_gate @ Wg) * (x_out_normed @ Wp)

Backed by ``cuequivariance_ops_torch.gated_gemm_torch.fused_sigmoid_gated_dual_gemm_dual_x``
— the same dual-A dual-accumulator SM90 kernel that the cuequivariance
``triangle_multiplicative_update`` and the perf/trimul dtv1 wrapper use for
their output-gated GEMM. It also accepts an optional output mask so we can
fold the post-tm2 broadcast mask into the same kernel.

Convention:
    The underlying op takes weights in ``(N, K)`` form (nn.Linear-style).
    ``team_gm`` and our tm1 call sites pass them in ``(K, N)`` form (already
    transposed), so the wrapper accepts both and reorients via ``.T``.
"""

from __future__ import annotations

import torch
from cuequivariance_ops_torch.gated_gemm_torch import (
    fused_sigmoid_gated_dual_gemm_dual_x,
)


def tm2_cute_forward(
    x_gate: torch.Tensor,  # (..., D)  — first arg of tm2 (x_normed in TriMul)
    x_out_normed: torch.Tensor,  # (..., D) — second arg (LN_out of contraction)
    Wg_nk: torch.Tensor,  # (D, D)  — gate weight in (N, K) form, nn.Linear-style
    Wp_nk: torch.Tensor,  # (D, D)  — proj weight in (N, K) form, nn.Linear-style
    mask_arg: torch.Tensor | None = None,  # precomputed output mask, broadcast-shaped
) -> torch.Tensor:
    """Forward tm2 = ``sigmoid(x_gate @ Wg_nk.T) * (x_out_normed @ Wp_nk.T)``.

    Weights are expected in nn.Linear ``(N, K)`` form so they can be
    pre-transposed at setup time (matches cuequiv's convention; if you have
    ``(K, N)`` weights, pass ``W.T.contiguous()``).
    """
    assert x_gate.shape == x_out_normed.shape
    D = x_gate.shape[-1]
    orig_shape = x_gate.shape
    x1 = x_gate.reshape(-1, D)
    x2 = x_out_normed.reshape(-1, D)
    out = fused_sigmoid_gated_dual_gemm_dual_x(x1, x2, Wg_nk, Wp_nk, mask=mask_arg)
    return out.view(orig_shape)


def make_pair_mask_for_tm2(mask_1d: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Build a (M, 1) per-row mask for ``tm2_cute_forward(mask_arg=...)``."""
    m2 = (mask_1d.unsqueeze(-1) & mask_1d.unsqueeze(-2)).to(dtype)
    return m2.reshape(-1, 1)
