"""Per-phase K1 cycles per weight-block step (build with -DK1_PROF=1): python k1_prof.py bidir 768"""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trimul_a100 as TA  # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType as I  # noqa: E402
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication  # noqa: E402
from miniworld_engine.modules.triangle_multiplication.bidirectional import BidirectionalTriangleMultiplication  # noqa: E402
var, L = sys.argv[1], int(sys.argv[2])
ext = TA.build(extra=["-DK1_PROF=1"] + sys.argv[3:])
mod = (BidirectionalTriangleMultiplication(128, implementation=I.PYTORCH) if var == "bidir" else TriangleMultiplication(128, implementation=I.PYTORCH)).cuda().bfloat16().eval()
z = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16); mask = torch.rand(1, L, device="cuda") > .1
with torch.no_grad():
    pk = TA.pack(mod); ch, T = pk["ch"], L * L
    zf = z.reshape(T, 128); ab = torch.empty(2 * ch, T, device="cuda", dtype=torch.bfloat16); m = mask.reshape(L).to(torch.uint8)
    prof = torch.zeros(16, dtype=torch.int64, device="cuda")
    run = lambda pr: ext.k1(zf, m, pk["w1"], pk["g_in"], pk["b_in"], ab, L, pk["eps_in"], 0, pr)
    for _ in range(3): run(torch.empty(0, device="cuda"))
    for _ in range(5): run(prof)
    torch.cuda.synchronize()
    v = prof.cpu().tolist(); steps = v[8]
    names = ["tile-start barrier", "frags/masks/z issue", "MMA + epilogue chunks", "per-step barrier", "weight wait", "plane stores", "z wait + spread LN", "prev tile tail"]
    tot = sum(v[:8])
    print(f"{var} L{L}: {tot / steps:.0f} cycles per warp-step (tensor minimum 1024)")
    for n, c in zip(names, v[:8]): print(f"  {n:24s} {c / steps:7.0f}  {100 * c / tot:5.1f}%")
