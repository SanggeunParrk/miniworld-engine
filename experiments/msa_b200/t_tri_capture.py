import sys, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from tri_b200 import _Core
L, H = 384, 4
q, k, v = (torch.randn(1, L, L, H, 32, device="cuda", dtype=torch.bfloat16).requires_grad_() for _ in range(3))
b = torch.randn(1, H, L, L, device="cuda", dtype=torch.bfloat16).requires_grad_()
go = torch.randn(1, L, L, H, 32, device="cuda", dtype=torch.bfloat16)
step = lambda: torch.autograd.grad(_Core.apply(q, k, v, b, 32 ** -0.5), [q, k, v, b], go)
for _ in range(3): step()
torch.cuda.synchronize()
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    step(); step()
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
try:
    with torch.cuda.graph(g):
        step()
    print("capture ok")
except Exception as e:
    print("capture failed:", str(e)[:300])
from bench_tri import make
from bench import timeit
m = make("ours"); m.train()
pair = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16)
gout = torch.randn_like(pair)
xb = pair.clone().requires_grad_(True); params = [xb, *m.parameters()]
st = lambda: torch.autograd.grad(m(xb), params, gout)
try:
    print("module train", timeit(st))
except Exception as e:
    print("module capture failed:", str(e)[:200])
    import traceback; traceback.print_exc()
