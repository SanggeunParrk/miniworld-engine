"""``layer_norm_transpose`` — a cuequivariance-FREE drop-in for
``cuequivariance_ops_torch.fused_layer_norm_torch.layer_norm_transpose``.

Built on our own Triton LayerNorm, at the HBM bandwidth wall. Supports the two
layouts the trimul cute path uses:

    "nd->nd"   : x (M, D) -> LN over D -> (M, D)          (LN_in; our triton_layernorm)
    "dbn->bnd" : x (D, B, N) -> LN over D -> (B, N, D)    (LN_out; FUSED transpose-LN kernel)

The dbn->bnd case is a FUSED transpose+LN (one coalesced read of the channel-major
(D, M) input + one coalesced row-major write), NOT a materialised transpose then LN —
so it matches cuequiv's single-pass cost. Same channel-major-LN math as the trimul
back-half kernel (``_back_kernel``), just without the proj/gate.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from miniworld_kernels.kernels.layernorm.triton.main import triton_layernorm


_dbn_configs = [
    triton.Config({"BLOCK_M": bm}, num_warps=nw, num_stages=ns)
    for bm in (16, 32, 64, 128)
    for nw in (4, 8)
    for ns in (2, 3)
]


@triton.autotune(configs=_dbn_configs, key=["D"])
@triton.jit
def _ln_transpose_dbn_kernel(
    x_ptr,   # (D, M) channel-major: x[k, m] at k*M + m
    y_ptr,   # (M, D) row-major:     y[m, k] at m*D + k
    w_ptr, b_ptr,  # (D,)
    M, eps,
    D: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, D)
    mmask = rm[:, None] < M
    # channel-major load -> (BLOCK_M rows, D channels), coalesced along m within each channel
    x = tl.load(x_ptr + rk[None, :] * M + rm[:, None], mask=mmask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / D
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + rk).to(tl.float32)
    b = tl.load(b_ptr + rk).to(tl.float32)
    y = ((xc * rstd[:, None]) * w[None, :] + b[None, :]).to(y_ptr.dtype.element_ty)
    # row-major store -> (BLOCK_M, D), coalesced along k within each row (the transpose)
    tl.store(y_ptr + rm[:, None] * D + rk[None, :], y, mask=mmask)


def _ln_transpose_dbn_bnd(x, weight, bias, eps):
    """x: (D, B, N) -> LN over D -> (B, N, D), fused (no materialised transpose)."""
    d, b, n = x.shape
    M = b * n
    x_dm = x.reshape(d, M)  # (D, M) channel-major (contiguous view when x is contiguous)
    if not x_dm.is_contiguous():
        x_dm = x_dm.contiguous()
    y = torch.empty(M, d, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)  # noqa: E731
    _ln_transpose_dbn_kernel[grid](
        x_dm, y, weight.contiguous(), bias.contiguous(), M, float(eps), D=d,
    )
    return y.view(b, n, d)


def layer_norm_transpose(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float = 1e-5,
    layout: str = "nd->nd",
) -> torch.Tensor:
    """LayerNorm over the D axis with an optional layout transpose. Drop-in for the
    cuequiv op (same call signature); our Triton LN underneath, cuequiv-free."""
    if layout == "nd->nd":
        return triton_layernorm(x, weight, bias, eps)  # (M, D) -> (M, D)
    if layout == "dbn->bnd":
        return _ln_transpose_dbn_bnd(x, weight, bias, eps)  # (D, B, N) -> (B, N, D), fused
    msg = f"layer_norm_transpose: unsupported layout {layout!r} (have nd->nd, dbn->bnd)"
    raise ValueError(msg)
