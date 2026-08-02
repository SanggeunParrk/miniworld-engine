"""Adapters onto quack's CuTeDSL normalization kernels, for head-to-head benching.

quack (Dao-AILab) ships a unified RMSNorm/LayerNorm CuTeDSL kernel:
  - `layernorm_fwd(x, weight, bias, eps)` : LayerNorm FORWARD (mean-subtracted),
    H100 cluster-aware `row_reduce` (clusters only engage for N >= 16k; our
    N <= 768 runs cluster_n=1). Requires fp32 weight/bias.
  - `rmsnorm(x, weight, ...)` : RMSNorm with autograd (fwd + bwd). The backward
    (`RMSNormBackward`) uses a PERSISTENT grid of `sm_count` blocks grid-striding
    over M, so it produces only ~132 partial dw rows (vs our triton partial path's
    cdiv(M, block_m) ~= 16k rows) then a `rms_final_reduce` pass.

quack does NOT ship a LayerNorm *backward* (only RMSNorm bwd; the bwd kernel takes
rstd but no mean). So for the cute-vs-triton backward question we bench quack
RMSNorm fwd+bwd as a *proxy* for "how fast is quack's cute norm backward" — it does
slightly less work than LayerNorm (no mean, no db), which we call out in the report.
"""

from __future__ import annotations

import torch

try:
    from quack.rmsnorm import layernorm_fwd as _quack_layernorm_fwd
    from quack.rmsnorm import rmsnorm as _quack_rmsnorm

    QUACK_AVAILABLE = True
except Exception as exc:  # pragma: no cover - import guard
    QUACK_AVAILABLE = False
    _QUACK_IMPORT_ERROR = exc


def quack_layernorm_fwd(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """quack CuTeDSL LayerNorm forward (forward-only). weight/bias upcast to fp32."""
    x2 = x.reshape(-1, x.shape[-1])
    out = _quack_layernorm_fwd(x2, weight.float(), bias.float(), eps=eps)
    return out.view_as(x)


def quack_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """quack CuTeDSL RMSNorm with autograd (fwd + bwd) — cute-backward speed proxy."""
    return _quack_rmsnorm(x, weight, eps=eps)
