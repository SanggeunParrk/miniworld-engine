import sys, torch, time, os
from foldjet_engine.modules import triangle as T
from foldjet_engine.measure import speed
from torch.profiler import profile, ProfilerActivity
L=int(sys.argv[1]); tf=os.environ.get("TF","1")=="1"
torch.manual_seed(0)
m = T.BidirectionalTriangleAttention(d_pair=128, backend="cuda").cuda().float()
with torch.no_grad():
    [p.normal_(0,0.1) for p in m.parameters()]
inputs={"pair": torch.randn(1,L,L,128,device="cuda",dtype=torch.bfloat16), "mask": torch.ones(1,L,dtype=torch.bool,device="cuda")}
step = speed.make_step(m, inputs, backend="cuda", train=True, autocast=torch.bfloat16, seed=0, compile_mode=None)
torch.backends.cuda.matmul.allow_tf32 = tf
s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(5): step()
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
g=speed._capture(step, s); torch.cuda.synchronize()
for _ in range(5): g.replay()
torch.cuda.synchronize(); ts=[]
for _ in range(15):
    time.sleep(0.15); a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b)*1e3)
ts.sort(); print("L",L,"tf32",tf,"graph min %.0f med %.0f"%(ts[0],ts[7]))
with profile(activities=[ProfilerActivity.CUDA]) as p:
    g.replay(); torch.cuda.synchronize()
ev=[e for e in p.events() if e.device_type.name=="CUDA"]
ev.sort(key=lambda e:-(e.time_range.end-e.time_range.start))
for e in ev[:4]: print("   %8.1f us  %s"%(e.time_range.end-e.time_range.start, e.name[:90]))
