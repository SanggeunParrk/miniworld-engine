"""Verify the fused CuTeDSL LayerNormLinear kernel vs the true op.

Compiles + runs `layernorm_linear_cute` and compares to
`F.layer_norm(X) @ W + bias` (bf16) and to the folded torch reference.

Run from repo root on a GPU node (with LD_LIBRARY_PATH=$CONDA_PREFIX/lib):
    python -m miniworld_kernels.kernels.layernorm_linear.verify_cute
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from miniworld_kernels.kernels.layernorm_linear.cute import layernorm_linear_cute
from miniworld_kernels.kernels.layernorm_linear.reference import layernorm_linear_folded

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


def compare(a: torch.Tensor, b: torch.Tensor) -> str:
    a32, b32 = a.float(), b.float()
    abs_err = (a32 - b32).abs().max().item()
    rel_fro = ((a32 - b32).norm() / (b32.norm() + 1e-12)).item()
    cos = F.cosine_similarity(a32.flatten(), b32.flatten(), dim=0).item()
    return f"max|abs|={abs_err:.3e}  rel_fro={rel_fro:.3e}  cos={cos:.6f}"


def run(M: int, K: int, N: int) -> None:
    eps = 1e-5
    torch.manual_seed(0)
    x = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    gamma = torch.randn(K, device=DEVICE, dtype=DTYPE)
    beta = torch.randn(K, device=DEVICE, dtype=DTYPE)
    weight = (torch.randn(N, K, device=DEVICE, dtype=DTYPE) / (K ** 0.5))
    bias = torch.randn(N, device=DEVICE, dtype=DTYPE)

    oracle = F.linear(F.layer_norm(x, (K,), gamma, beta, eps), weight, bias)
    folded = layernorm_linear_folded(x, gamma, beta, weight, bias, eps)
    cute_y = layernorm_linear_cute(x, gamma, beta, weight, bias, eps)

    print(f"=== M={M} K={K} N={N} ===")
    print(f"  cute   vs true  : {compare(cute_y, oracle)}")
    print(f"  cute   vs folded: {compare(cute_y, folded)}")


def main() -> None:
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}\n")
    run(4096, 768, 768)
    run(4096, 4096, 4096)
    run(4096, 4096, 3 * 4096)  # QKV
    run(2048, 1024, 1024)


if __name__ == "__main__":
    main()
