"""Run custom (and cuBLAS) contraction a few times for ncu: python prof_contract.py L CH h"""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trimul_a100 as TA  # noqa: E402
L, CH, h = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
ext = TA.build(extra=sys.argv[4:])
a = torch.randn(CH, L, L, device="cuda", dtype=torch.bfloat16); b = torch.randn_like(a); x = torch.empty_like(a)
for _ in range(4):
    ext.contract(a, b, x, h, 0)
    if h: torch.bmm(a[:h], b[:h].transpose(1, 2), out=x[:h])
    if h < CH: torch.bmm(a[h:].transpose(1, 2), b[h:], out=x[h:])
torch.cuda.synchronize()
