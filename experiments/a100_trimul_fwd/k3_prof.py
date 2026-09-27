"""Per-phase K3 cycles per warp-tile (build with -DK3_PROF=1): python k3_prof.py bidir 768"""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trimul_a100 as TA  # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType as I  # noqa: E402
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication  # noqa: E402
from miniworld_engine.modules.triangle_multiplication.bidirectional import BidirectionalTriangleMultiplication  # noqa: E402
var, L = sys.argv[1], int(sys.argv[2])
ext = TA.build(extra=["-DK3_PROF=1"] + sys.argv[3:])
mod = (BidirectionalTriangleMultiplication(128, implementation=I.PYTORCH) if var == "bidir" else TriangleMultiplication(128, implementation=I.PYTORCH)).cuda().bfloat16().eval()
z = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16)
with torch.no_grad():
    pk = TA.pack(mod); ch, T = pk["ch"], L * L
    x = torch.randn(ch, T, device="cuda", dtype=torch.bfloat16); zf = z.reshape(T, 128); out = torch.empty_like(zf)
    prof = torch.zeros(16, dtype=torch.int64, device="cuda")
    run = lambda pr: ext.k3(x, zf, pk["wo"], pk["wg"], pk["so"], pk["bo"], pk["sg"], pk["bg"], out, pk["eps_out"], 0, pr, torch.empty(0, device="cuda"), L)
    for _ in range(3): run(torch.empty(0, device="cuda"))
    for _ in range(5): run(prof)
    torch.cuda.synchronize()
    v = prof.cpu().tolist(); tiles = v[8]
    names = ["X wait", "X frags + x stats", "x exchange", "projection", "z wait+frags+stats", "z exchange", "gate+store", "end barrier"]
    tot = sum(v[:8])
    print(f"{var} L{L}: {tot / tiles:.0f} cycles per warp-tile")
    for n, c in zip(names, v[:8]): print(f"  {n:22s} {c / tiles:7.0f}  {100 * c / tot:5.1f}%")
