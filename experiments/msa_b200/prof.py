"""Per-kernel device time of one module call (ours), from torch.profiler (CUPTI kernel records).
    python prof.py opm infer|train [L] [S]"""
import sys
import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench import make_inputs, make_module, module_fn

op, mode = sys.argv[1], sys.argv[2]
L = int(sys.argv[3]) if len(sys.argv) > 3 else 384
S = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
impl = sys.argv[5] if len(sys.argv) > 5 else "ours"
mod = make_module(op, impl)
msa, pair, mask = make_inputs(op, L, S)
if mode == "infer":
    mod.eval()
    f = module_fn(op, mod, msa, pair, mask)
    ctx = torch.no_grad
else:
    mod.train()
    msa.requires_grad_(True); pair.requires_grad_(True)
    params = [msa, pair, *mod.parameters()]
    g = module_fn(op, mod, msa, pair, mask)
    gout = torch.randn(msa.shape if op == "pwa" else pair.shape, device="cuda", dtype=torch.bfloat16)
    f = lambda: torch.autograd.grad(g(), params, gout)
    import contextlib
    ctx = contextlib.nullcontext
with ctx():
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(5):
            f()
        torch.cuda.synchronize()
rows = {}
for e in p.events():
    if e.device_type == torch.autograd.DeviceType.CUDA:
        r = rows.setdefault(e.name, [0, 0.0])
        r[0] += 1; r[1] += e.device_time
tot = sum(v[1] for v in rows.values()) / 5
print(f"{op} {mode} L{L} S{S}: summed kernel time {tot:.1f} us per call")
for k, (n, t) in sorted(rows.items(), key=lambda kv: -kv[1][1]):
    print(f"  {t/5:8.1f} us  x{n//5:<3d} {k[:110]}")
