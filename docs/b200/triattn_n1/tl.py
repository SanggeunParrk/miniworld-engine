import sys, json, torch
from foldjet_engine.modules import triangle as T
kw = json.loads(sys.argv[1]); L = int(sys.argv[2])
torch.manual_seed(0)
mod = T.TriangleAttention(**kw, backend="cuda").cuda().float().train()
D = kw["d_pair"]
x = torch.randn(1, L, L, D, device="cuda", dtype=torch.bfloat16, requires_grad=True); dy = torch.randn_like(x)
def step():
    for p in mod.parameters(): p.grad = None
    x.grad = None
    with torch.autocast("cuda", torch.bfloat16): y = mod(x)
    y.backward(dy)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(4): step()
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): step()
flush = torch.empty(256<<20, dtype=torch.uint8, device="cuda")
for _ in range(5): g.replay()
torch.cuda.synchronize()
ts=[]
for _ in range(20):
    flush.zero_(); a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b)*1e3)
ts.sort(); print("graph median us", ts[len(ts)//2])
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as p:
    flush.zero_(); g.replay(); torch.cuda.synchronize()
ev = sorted([e for e in p.events() if e.device_type.name=="CUDA" and "memset" not in e.name.lower() and e.name!="" ], key=lambda e:e.time_range.start)
t0 = None; prev=None
for e in ev:
    if "elementwise" in e.name and t0 is None: continue
    s0=e.time_range.start; 
    if t0 is None: t0=s0
    dur=e.time_range.end-s0
    print(f"{s0-t0:8.1f} +{dur:6.1f} gap {(s0-prev) if prev is not None else 0:6.1f} {e.name[:90]}"); prev=e.time_range.end
