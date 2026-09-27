"""One launch of the CUDA core for NCU."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuda_core import attn_core  # noqa: E402

S, H, DS, NB, dev, bf = 5, 16, 768, 24, "cuda", torch.bfloat16
L = int(sys.argv[1]) if len(sys.argv) > 1 else 768
qkvg = (torch.randn(S * L, 4 * DS, device=dev) * DS ** -0.5).to(bf)
bias = (torch.randn(NB * H, L, L, device=dev) * 0.3).to(bf)
for _ in range(6):
    attn_core(qkvg, bias, 1, S, H)
torch.cuda.synchronize()
