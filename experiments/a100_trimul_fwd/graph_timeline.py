"""Kernel timeline inside CUDA-graph replays of the training step: busy time, idle gaps, the largest gaps.  python graph_timeline.py bidir 768"""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trimul_a100 as TA, trimul_train as TT  # noqa: E401
from miniworld_engine.modules.exceptions import ImplementationType as I
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
from miniworld_engine.modules.triangle_multiplication.bidirectional import BidirectionalTriangleMultiplication
var, L = sys.argv[1], int(sys.argv[2])
ext = TA.build()
mod = (BidirectionalTriangleMultiplication if var == "bidir" else TriangleMultiplication)(128, implementation=I.PYTORCH, p_drop=.25).cuda().bfloat16().train()
z = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
mask = torch.rand(1, L, device="cuda") > .1
ds = ((torch.rand(1, 1, L, 128, device="cuda") > .25).to(torch.bfloat16) / .75).to(torch.bfloat16)
dy = torch.randn_like(z).detach()
params = [dict(mod.named_parameters())[n] for n in TT.PARAMS]
fn = lambda: torch.autograd.grad(TT.forward_train(ext, mod, z, mask, ds), [z] + params, dy)  # noqa: E731
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): fn()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s): fn()
torch.cuda.current_stream().wait_stream(s)
for _ in range(20): g.replay()
torch.cuda.synchronize()
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(5): g.replay()
    torch.cuda.synchronize()
ev = sorted([(e.time_range.start, e.time_range.end, e.name) for e in prof.events() if e.device_type.name == "CUDA" and e.time_range.end > e.time_range.start])
t0, t1 = ev[0][0], max(e[1] for e in ev)
# union of busy intervals (streams may overlap)
busy, cur_s, cur_e, gaps = 0, ev[0][0], ev[0][1], []
for st, en, name in ev[1:]:
    if st > cur_e:
        busy += cur_e - cur_s; gaps.append((st - cur_e, name)); cur_s, cur_e = st, en
    else:
        cur_e = max(cur_e, en)
busy += cur_e - cur_s
span = (t1 - t0) / 5
print(f"{var} L{L}: per step span {span/1000:.3f} ms, busy {busy/5/1000:.3f} ms, idle {(t1 - t0 - busy)/5/1000:.3f} ms, kernels/step {len(ev)//5}")
agg = {}
for gap, name in gaps:
    agg.setdefault(name[:50], [0, 0]); agg[name[:50]][0] += gap; agg[name[:50]][1] += 1
for name, (gsum, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:12]:
    print(f"  gap before {name:50s} {gsum/5:8.1f} us/step ({n//5}x)")
