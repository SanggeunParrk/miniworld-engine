import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gra import gemm_resgate_adaln  # noqa
L, D, M, Kd = 768, 768, 3840, 768
dev, bf = "cuda", torch.bfloat16
a = torch.randn(M, 3072, device=dev, dtype=bf)[:, :Kd]
w = (torch.randn(D, Kd, device=dev) * Kd ** -0.5).to(bf)
x = torch.randn(M, D, device=dev); xa = torch.empty(M, D, device=dev, dtype=bf)
g = torch.randn(L, 4, D, device=dev, dtype=bf)
for _ in range(3):
    gemm_resgate_adaln(a, w, x, g[:, 0], g[:, 1], g[:, 2], xa, L, nwg=2)
    gemm_resgate_adaln(a, w, x, g[:, 0], None, None, None, L, nwg=2)
torch.cuda.synchronize()
