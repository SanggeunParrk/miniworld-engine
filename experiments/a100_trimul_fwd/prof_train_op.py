"""Run the training step (fwd + bwd) a few times on one shape (for ncu): python prof_train_op.py bidir 768 [extra flags]"""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trimul_a100 as TA, trimul_train as TT  # noqa: E401,E402
from miniworld_engine.modules.exceptions import ImplementationType as I  # noqa: E402
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication  # noqa: E402
from miniworld_engine.modules.triangle_multiplication.bidirectional import BidirectionalTriangleMultiplication  # noqa: E402

var, L = sys.argv[1], int(sys.argv[2])
ext = TA.build(extra=sys.argv[3:])
mod = (BidirectionalTriangleMultiplication if var == "bidir" else TriangleMultiplication)(128, implementation=I.PYTORCH, p_drop=.25)
mod = mod.cuda().bfloat16().train()
z = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
mask = torch.rand(1, L, device="cuda") > .1
ds = ((torch.rand(1, 1, L, 128, device="cuda") > .25).to(torch.bfloat16) / .75).to(torch.bfloat16)
dy = torch.randn_like(z).detach()
params = [dict(mod.named_parameters())[n] for n in TT.PARAMS]
for _ in range(5):
    torch.autograd.grad(TT.forward_train(ext, mod, z, mask, ds), [z] + params, dy)
torch.cuda.synchronize()
