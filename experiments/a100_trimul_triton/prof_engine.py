"""Per-kernel GPU time of the ENGINE's Triton TriMul path (training step and inference) for comparison: python prof_engine.py bidir 768"""
import collections, sys
import torch
from miniworld_engine.modules.exceptions import ImplementationType as I
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
from miniworld_engine.modules.triangle_multiplication.bidirectional import BidirectionalTriangleMultiplication
var, L = sys.argv[1], int(sys.argv[2])
impl = getattr(I, "TRITON")
mod = (BidirectionalTriangleMultiplication if var == "bidir" else TriangleMultiplication)(128, implementation=impl, p_drop=.25).cuda().bfloat16().train()
z = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
mask = torch.rand(1, L, device="cuda") > .1
dy = torch.randn_like(z).detach()
step = lambda: torch.autograd.grad(mod(z, mask), [z] + list(mod.parameters()), dy, allow_unused=True)  # noqa: E731
for _ in range(3): step()
torch.cuda.synchronize()
from torch.profiler import profile, ProfilerActivity
for name, fn in (("train", step), ("infer", None)):
    if fn is None:
        mod.eval()
        fn = lambda: mod(z.detach(), mask)  # noqa: E731
        with torch.no_grad():
            for _ in range(3): fn()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        with torch.set_grad_enabled(name == "train"):
            for _ in range(3): fn()
        torch.cuda.synchronize()
    agg = collections.defaultdict(float)
    for e in prof.events():
        if e.device_type.name == "CUDA": agg[e.name[:70]] += e.device_time / 3
    print(f"== engine {name} {var} L{L}: total {sum(agg.values())/1000:.2f} ms")
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:14]: print(f"  {v:9.1f} us  {k}")
