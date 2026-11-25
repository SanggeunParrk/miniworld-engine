import sys, json, torch
from foldjet_engine.modules import triangle as T
kw = json.loads(sys.argv[1]); L = int(sys.argv[2])
torch.manual_seed(0)
mod = T.TriangleAttention(**kw, backend="cuda").cuda().float().train()
x = torch.randn(1, L, L, kw["d_pair"], device="cuda", dtype=torch.bfloat16, requires_grad=True); dy = torch.randn_like(x)
with torch.autocast("cuda", torch.bfloat16): y = mod(x)
y.backward(dy); torch.cuda.synchronize(); print("step done", flush=True)
