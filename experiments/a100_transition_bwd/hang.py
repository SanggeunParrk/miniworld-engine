"""Reproduce: sequential backward in a 20-call CUDA graph, replayed; reports progress (python hang.py L mode)"""
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import transition_bwd_a100 as TB  # noqa: E402
L = int(sys.argv[1]); mode = sys.argv[2]
ext = TB.build(extra=["-DDXP_FDS=1"])
mod, x = TB.TA.fixture(L)
dy = torch.randn_like(x)
pk = TB.pack(mod)
bufs = {}
fn = (lambda: TB.backward_seq(ext, x, dy, pk, bufs)) if mode == "seq" else (lambda: TB.backward_ring(ext, x, dy, pk, bufs, ndxp=64))
for i in range(30):
    fn(); torch.cuda.synchronize()
print("eager ok", flush=True)
g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
with torch.cuda.stream(s):
    fn(); torch.cuda.synchronize()
    with torch.cuda.graph(g, stream=s):
        for _ in range(20):
            fn()
torch.cuda.synchronize()
print("captured", flush=True)
for k in range(50):
    g.replay(); torch.cuda.synchronize()
    if k % 10 == 0: print("replay", k, flush=True)
print("done", flush=True)
