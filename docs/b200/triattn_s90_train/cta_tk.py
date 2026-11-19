import sys, json, torch
from miniworld_engine.kernels.triangle_attention.cuda import b200_triattn as K
from miniworld_engine.kernels import _nvcc
orig = _nvcc.load_extension
def patched(*a, **kw):
    kw["extra_cuda_cflags"] = [*kw.get("extra_cuda_cflags", []), "-DS90_TL=1", "-DTA_TL=1"]; kw["name"] = kw["name"] + "_s90tl"; return orig(*a, **kw)
K.load_extension = patched
from miniworld_engine.modules import triangle as T
kw, L = json.loads(sys.argv[1]), int(sys.argv[2]); torch.manual_seed(0)
mod = T.TriangleAttention(**kw, backend="cuda").cuda().float().train()
x = torch.randn(1, L, L, kw["d_pair"], device="cuda", dtype=torch.bfloat16, requires_grad=True); dy = torch.randn_like(x)
def step():
    for p in mod.parameters(): p.grad = None
    x.grad = None
    with torch.autocast("cuda", torch.bfloat16): y = mod(x)
    y.backward(dy)
for _ in range(4): step()
torch.cuda.synchronize()
t = K._tallk().s90_tl().tolist()[5]; rows = [r for r in t if r[0] > 0]; t0 = min(r[0] for r in rows)
f = lambda i: sorted((r[i] - t0) / 1e3 for r in rows if r[i] > 0)
m = lambda v: v[len(v) // 2]
for i, nm in ((0, "entry"), (1, "prologue done"), (6, "first K block landed (MMA warp)"), (7, "last MMA issued (MMA warp)"), (4, "MMA complete seen by drain"), (3, "partial tiles in smem (cl_sync)"), (5, "reduction done"), (2, "end")):
    v = f(i)
    if not v: print(nm, "no stamps"); continue
    print(f"{nm:34s} n {len(v):3d} first {v[0]:.2f} median {m(v):.2f} last {v[-1]:.2f} us")
