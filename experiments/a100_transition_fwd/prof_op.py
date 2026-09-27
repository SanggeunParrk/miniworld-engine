"""Run the op a few times on one shape (for ncu / nsys): python prof_op.py 384 [-DFOO=1 ...]"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transition_a100 as TA  # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType as I  # noqa: E402
from miniworld_engine.modules.transition import Transition  # noqa: E402

L = int(sys.argv[1])
ext = TA.build(extra=sys.argv[2:])
mod, x = TA.fixture(L)
with torch.no_grad():
    pk = TA.pack(mod, TA.chunk(sys.argv[2:]))
    out = torch.empty(x.shape[0], 128, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        TA.forward(ext, x, pk, out)
torch.cuda.synchronize()
