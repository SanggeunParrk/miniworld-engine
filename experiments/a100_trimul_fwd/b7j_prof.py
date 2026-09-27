"""b7j role balance: python b7j_prof.py bidir 768 [C] [RINGS]  (build with -DB7J_PROF=1)"""
import os, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
var, L = sys.argv[1], int(sys.argv[2])
if len(sys.argv) > 3: os.environ["A100_B7J_C256"] = os.environ["A100_B7J_C128"] = sys.argv[3]
if len(sys.argv) > 4: os.environ["A100_B7J_RINGS"] = sys.argv[4]
import trimul_a100 as TA, trimul_train as TT  # noqa: E401
TA_build = TA.build
TA.build = lambda **k: TA_build(extra=["-DB7J_PROF=1"])
exec(open(Path(__file__).parent / "prof_train_op.py").read().split("var, L = ")[0])
from miniworld_engine.modules.exceptions import ImplementationType as I
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
from miniworld_engine.modules.triangle_multiplication.bidirectional import BidirectionalTriangleMultiplication
ext = TA.build()
mod = (BidirectionalTriangleMultiplication if var == "bidir" else TriangleMultiplication)(128, implementation=I.PYTORCH, p_drop=.25).cuda().bfloat16().train()
z = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
mask = torch.rand(1, L, device="cuda") > .1
ds = ((torch.rand(1, 1, L, 128, device="cuda") > .25).to(torch.bfloat16) / .75).to(torch.bfloat16)
dy = torch.randn_like(z).detach()
params = [dict(mod.named_parameters())[n] for n in TT.PARAMS]
step = lambda: torch.autograd.grad(TT.forward_train(ext, mod, z, mask, ds), [z] + params, dy)  # noqa: E731
for _ in range(3): step()
torch.cuda.synchronize(); ext.b7j_prof()
for _ in range(5): step()
torch.cuda.synchronize()
h = ext.b7j_prof()
nsrc = ch = None
print(f"{var} L{L} C={os.environ.get('A100_B7J_C256')}: source total {h[0]/5/1e6:.1f} Mcyc-CTA, slot waits {h[1]/5/1e6:.1f} ({100*h[1]/max(h[0],1):.1f}%) | "
      f"consumer total {h[2]/5/1e6:.1f}, flag waits {h[3]/5/1e6:.1f} ({100*h[3]/max(h[2],1):.1f}%)")
