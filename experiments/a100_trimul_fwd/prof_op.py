"""Run the op a few times on one shape (for ncu / nsys): python prof_op.py bidir 768"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trimul_a100 as TA  # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType as I  # noqa: E402
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication  # noqa: E402
from miniworld_engine.modules.triangle_multiplication.bidirectional import BidirectionalTriangleMultiplication  # noqa: E402

var, L = sys.argv[1], int(sys.argv[2])
extra = sys.argv[3:]
ext = TA.build(extra=extra)
mod = (BidirectionalTriangleMultiplication(128, implementation=I.PYTORCH) if var == "bidir"
       else TriangleMultiplication(128, implementation=I.PYTORCH)).cuda().bfloat16().eval()
z = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16)
mask = torch.rand(1, L, device="cuda") > .1
with torch.no_grad():
    pk = TA.pack(mod)
    bufs = {}
    for _ in range(5):
        TA.forward(ext, z, mask, pk, bufs)
torch.cuda.synchronize()
