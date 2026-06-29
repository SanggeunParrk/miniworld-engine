"""Milestone 0: validate the folded LayerNormLinear math (no kernel yet).

Compares `layernorm_linear_folded` (raw X@W2 + mean/rstd in epilogue) against the
true op `F.layer_norm(X) @ W + bias`, in bf16, and stress-tests the two known
numerical risks:
  1. naive variance  var = E[x^2] - E[x]^2   (vs torch unbiased=False)
  2. epilogue cancellation  acc - mean*S      (large common-mode input)

Run from repo root on a GPU node:
    python -m miniworld_kernels.kernels.layernorm_linear.verify_folding
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from miniworld_kernels.kernels.layernorm_linear.reference import layernorm_linear_folded

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


def compare(a: torch.Tensor, b: torch.Tensor) -> str:
    a32, b32 = a.float(), b.float()
    abs_err = (a32 - b32).abs().max().item()
    rel_fro = ((a32 - b32).norm() / (b32.norm() + 1e-12)).item()
    cos = F.cosine_similarity(a32.flatten(), b32.flatten(), dim=0).item()
    return f"max|abs|={abs_err:.3e}  rel_fro={rel_fro:.3e}  cos={cos:.6f}"


def true_op(x, gamma, beta, W, bias, eps):
    normed = F.layer_norm(x, (x.shape[-1],), gamma, beta, eps)
    return F.linear(normed, W, bias)


def run(M: int, K: int, N: int, *, x_mean: float = 0.0, x_scale: float = 1.0) -> None:
    eps = 1e-5
    torch.manual_seed(0)
    x = (torch.randn(M, K, device=DEVICE, dtype=DTYPE) * x_scale + x_mean)
    gamma = torch.randn(K, device=DEVICE, dtype=DTYPE)
    beta = torch.randn(K, device=DEVICE, dtype=DTYPE)
    W = torch.randn(N, K, device=DEVICE, dtype=DTYPE) / (K ** 0.5)
    bias = torch.randn(N, device=DEVICE, dtype=DTYPE)

    ref = true_op(x, gamma, beta, W, bias, eps)
    folded = layernorm_linear_folded(x, gamma, beta, W, bias, eps)

    # variance cancellation check (fp32): naive E[x^2]-E[x]^2 vs torch var
    xf = x.float()
    mean = xf.mean(1)
    var_naive = (xf * xf).mean(1) - mean * mean
    var_true = xf.var(1, unbiased=False)
    var_relerr = ((var_naive - var_true).abs() / (var_true.abs() + 1e-12)).max().item()

    tag = f"M={M} K={K} N={N} x_mean={x_mean} x_scale={x_scale}"
    print(f"=== {tag} ===")
    print(f"  folded vs true : {compare(folded, ref)}")
    print(f"  var naive vs torch: max rel_err={var_relerr:.3e}  (cancellation in stats)")


def main() -> None:
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}\n")
    # baseline shapes
    for K in (128, 768, 4096):
        run(4096, K, K)
    # QKV-style fan-out
    run(4096, 4096, 3 * 4096)
    print("\n--- stress: large common-mode mean (the acc - mean*S cancellation) ---")
    for xm in (1.0, 10.0, 100.0, 1000.0):
        run(4096, 4096, 4096, x_mean=xm, x_scale=1.0)


if __name__ == "__main__":
    main()
