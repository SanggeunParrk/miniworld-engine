"""Diagnose the large-N correctness bug: localize WHERE the cute kernel errs.

Sweeps N for fixed K, reports per-column error structure (which n-columns are
wrong, contiguous block vs scattered, tile alignment) to pin the root cause.

    python -m miniworld_kernels.kernels.layernorm_linear.diag_largeN
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from miniworld_kernels.kernels.layernorm_linear.cute import layernorm_linear_cute
from miniworld_kernels.kernels.layernorm_linear.reference import layernorm_linear_folded

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
TILE_N = 192  # default_config SM90


def run(M: int, K: int, N: int) -> None:
    eps = 1e-5
    torch.manual_seed(0)
    x = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    gamma = torch.randn(K, device=DEVICE, dtype=DTYPE)
    beta = torch.randn(K, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(N, K, device=DEVICE, dtype=DTYPE) / (K ** 0.5)
    bias = torch.randn(N, device=DEVICE, dtype=DTYPE)

    folded = layernorm_linear_folded(x, gamma, beta, weight, bias, eps).float()
    cute_y = layernorm_linear_cute(x, gamma, beta, weight, bias, eps).float()

    err = (cute_y - folded).abs()
    rel = (err.norm() / folded.norm()).item()
    # per-column (n) max error and per-row (m) max error
    col_err = err.amax(dim=0)  # (N,)
    row_err = err.amax(dim=1)  # (M,)
    bad_cols = (col_err > 0.5).nonzero(as_tuple=True)[0]
    bad_rows = (row_err > 0.5).nonzero(as_tuple=True)[0]
    n_tiles = (N + TILE_N - 1) // TILE_N
    print(f"=== M={M} K={K} N={N}  (N/{TILE_N}={N / TILE_N:.2f}, {n_tiles} n-tiles) ===")
    print(f"  rel_fro={rel:.3e}  max|abs|={err.max().item():.3e}")
    print(f"  bad cols(>0.5)={bad_cols.numel()}/{N}  bad rows(>0.5)={bad_rows.numel()}/{M}")
    if bad_cols.numel():
        lo, hi = bad_cols.min().item(), bad_cols.max().item()
        # which TILE_N blocks are bad
        blocks = sorted(set((bad_cols // TILE_N).tolist()))
        contiguous = bad_cols.numel() == (hi - lo + 1)
        print(f"  bad col range=[{lo},{hi}] contiguous={contiguous}  "
              f"bad tile-blocks({TILE_N})={blocks[:12]}{'...' if len(blocks) > 12 else ''}")


def main() -> None:
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}\n")
    for N in (4096, 4224, 6144, 8192, 12288):
        run(4096, 4096, N)


if __name__ == "__main__":
    main()
