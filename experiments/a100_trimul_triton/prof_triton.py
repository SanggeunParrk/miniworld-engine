"""Per-kernel GPU time of the Triton TriMul (inference + one training step): python prof_triton.py bidir 768"""
import collections, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trimul_triton as TT  # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType as I  # noqa: E402
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication  # noqa: E402
from miniworld_engine.modules.triangle_multiplication.bidirectional import BidirectionalTriangleMultiplication  # noqa: E402
var, L = sys.argv[1], int(sys.argv[2])
mod = (BidirectionalTriangleMultiplication if var == "bidir" else TriangleMultiplication)(128, implementation=I.PYTORCH, p_drop=.25).cuda().bfloat16().train()
z = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
mask = torch.rand(1, L, device="cuda") > .1
ds = ((torch.rand(1, 1, L, 128, device="cuda") > .25).to(torch.bfloat16) / .75).to(torch.bfloat16)
dy = torch.randn_like(z).detach()
params = [dict(mod.named_parameters())[n] for n in TT.PARAMS]
step = lambda: torch.autograd.grad(TT.forward_train(mod, z, mask, ds), [z] + params, dy)  # noqa: E731
for _ in range(3): step()
torch.cuda.synchronize()
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(3): step()
    torch.cuda.synchronize()
agg = collections.defaultdict(float)
for e in prof.events():
    if e.device_type.name == "CUDA": agg[e.name[:60]] += e.device_time / 3
for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:12]: print(f"  {v:9.1f} us  {k}")
for name in ("_k1", "_k3", "_b1e", "_b1d", "_src", "_con"):
    fn = getattr(TT, name)
    for k, v in fn.cache.items():
        print(name, "cfg", str(v)[:90])
