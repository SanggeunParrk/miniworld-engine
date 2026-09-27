"""D64 at module level: our fused fwd (transition_fwd_d64x2, save) + fused bwd (tbwd_d64_r16 + reduce_partials) inside an
autograd.Function, timed exactly like bench_mod_bwd.py (fwd, fwd+bwd, bwd = difference) against modules.Transition(64).
   python bench_mod_d64.py --length 384"""
import argparse, copy, statistics, sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import drv  # noqa: E402
p = argparse.ArgumentParser(); p.add_argument("--length", type=int, default=384); p.add_argument("--repl", type=int, default=16); p.add_argument("--torch-saves", action="store_true"); a = p.parse_args()
from miniworld_engine import settings
from miniworld_engine.modules import Transition
D, H, L = 64, 256, a.length; M = L * L; SLICES = H // 64; NDW = SLICES * a.repl; NDX = 132 - NDW; bf = torch.bfloat16
kf = drv.Kernel(str(HERE / "build/transition_fwd_d64x2.cubin"), "transition_fwd_fused", 114944)
kb = drv.Kernel(str(HERE / f"build/tbwd_d64_r{a.repl}.cubin"), "transition_bwd_fused", 231424)
kr = drv.Kernel(str(HERE / f"build/tbwd_d64_r{a.repl}.cubin"), "reduce_partials", 0)
tm = lambda t, dims, stride, box: drv.TensorMap(t, dims=dims, stride_bytes=stride, box=box)
_maps = {}
def cmap(key, t, dims, stride, box):                      # descriptors bound to a base pointer: cache like the engine does
    k = (key, t.data_ptr(), tuple(dims), tuple(box))
    if k not in _maps: _maps[k] = tm(t, dims, stride, box)
    return _maps[k]

class Ours(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, wa, wb, ws):
        x2 = x.reshape(M, D)
        g, b = gamma.float().contiguous(), beta.float().contiguous()      # the module keeps the norm params fp32
        wst = ws.t().contiguous()
        out = torch.empty_like(x2); xn = torch.empty_like(x2)
        rstd = torch.empty(M, device=x.device); c1 = torch.empty(M, device=x.device)
        kf((264, 1, 1), (256, 1, 1), cmap("fx", x2, [D, M], D * 2, [64, 64]), cmap("fa", wa, [D, H], D * 2, [64, 64]),
           cmap("fb", wb, [D, H], D * 2, [64, 64]), tm(wst, [D, H], D * 2, [64, 64]), cmap("fo", out, [D, M], D * 2, [64, 64]),
           g, b, xn, out, rstd, c1, int(M), int(M // 128), 1e-5, 1)
        if a.torch_saves:                                 # diagnosis: replace the kernel's saves with fp32-computed ones
            xf = x2.float(); mu = xf.mean(-1); rs = torch.rsqrt(xf.var(-1, unbiased=False) + 1e-5)
            print("  saves vs torch: rstd", float((rstd - rs).abs().max() / rs.abs().max()), "c1", float((c1 - mu * rs).abs().max()),
                  "xn", float(((xn.float() - ((xf * rs[:, None] - (mu * rs)[:, None]) * g + b)).norm() / xn.float().norm())))
            rstd.copy_(rs); c1.copy_(mu * rs); xn.copy_(((xf * rs[:, None] - c1[:, None]) * g + b).to(bf))
        ctx.save_for_backward(x2, xn, rstd, c1, gamma, beta, wa, wb, ws)
        return out.reshape(x.shape)
    @staticmethod
    def backward(ctx, dy):
        x2, xn, rstd, c1, gamma, beta, wa, wb, ws = ctx.saved_tensors
        dy2 = dy.reshape(M, D).contiguous()
        gf = gamma.float().contiguous()
        dx = torch.empty_like(x2); dg = torch.zeros(D, device=dy.device); db = torch.zeros(D, device=dy.device)
        partw = torch.empty(NDW * 4 * 64 * D, device=dy.device); dgbw = torch.empty(NDX * 8 * 2 * D, device=dy.device)
        dWa, dWb, dWs = torch.empty_like(wa), torch.empty_like(wb), torch.empty_like(ws)
        kb((132, 1, 1), (256, 1, 1), tm(dy2, [D, M], D * 2, [64, 64]), cmap("bxn", xn, [D, M], D * 2, [64, 64]),
           cmap("bx", x2, [D, M], D * 2, [64, 64]), cmap("bws", ws, [H, D], H * 2, [64, D]), cmap("bwa", wa, [D, H], D * 2, [64, 64]),
           cmap("bwb", wb, [D, H], D * 2, [64, 64]), rstd, c1, gf, dx, dg, db, partw, dgbw, int(M), int(M // 128))
        kr(((3 * SLICES * 64 * D + 2 * D + 255) // 256, 1, 1), (256, 1, 1), partw, dWa, dWb, dWs, dgbw, dg, db)
        return dx.reshape(dy.shape), dg.to(gamma.dtype), db.to(beta.dtype), dWa, dWb, dWs

settings.configure(engine_backend="triton", transition_residual_fusion=True, transition_fused_sm90a=True)
torch.manual_seed(0)
m = Transition(D, n=4, implementation="triton").cuda().bfloat16()
with torch.no_grad():
    for prm in m.parameters():
        if prm.ndim == 2: prm.normal_(std=prm.shape[-1] ** -0.5)
        elif prm is m.ln_in.weight: prm.copy_(1 + 0.2 * torch.randn_like(prm))
        else: prm.normal_(std=0.2)
P = (m.ln_in.weight, m.ln_in.bias, m.expand_a.weight, m.expand_b.weight, m.squeeze.weight)
x = torch.randn(1, L, L, D, device="cuda", dtype=bf, requires_grad=True); dy = torch.randn_like(x)
ours = lambda: Ours.apply(x, *P)
# correctness: both against fp32 autograd
ref = copy.deepcopy(m).float(); xr = x.detach().float().requires_grad_(True); ref(xr).backward(dy.float())
R = [xr.grad] + [q.grad for q in (ref.ln_in.weight, ref.ln_in.bias, ref.expand_a.weight, ref.expand_b.weight, ref.squeeze.weight)]
del ref, xr
def grads(fn):
    for q in (x,) + P: q.grad = None
    fn().backward(dy); return [q.grad.clone() for q in (x,) + P]
G_e, G_o = grads(lambda: m(x)), grads(ours)
rel = lambda u, r: float((u.float().reshape(-1) - r.reshape(-1)).norm() / r.norm())
print(f"D64 L{L} grad rel vs fp32 (ours | engine): " + "  ".join(f"{n} {rel(o, r):.2e}|{rel(e, r):.2e}" for n, o, e, r in zip(("dx", "dg", "db", "dWa", "dWb", "dWs"), G_o, G_e, R)))
def t(fn, reps=5):
    for _ in range(3): fn()
    torch.cuda.synchronize(); o = []
    for _ in range(7):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
        for _ in range(reps): fn()
        e.record(); torch.cuda.synchronize(); o.append(s.elapsed_time(e) * 1e3 / reps)
    return statistics.median(o)
if a.torch_saves: sys.exit(0)
def graphed(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    return g.replay
res = {}
for name, f in (("engine", lambda: m(x)), ("ours", ours)):
    for q in (x,) + P: q.grad = None
    def step(f=f):
        for q in (x,) + P: q.grad = None
        f().backward(dy)
    fw = t(graphed(f)); fb = t(graphed(step)); res[name] = (fw, fb, fb - fw)
    print(f"  {name:6s} (CUDA graph): fwd {fw:.1f}  fwd+bwd {fb:.1f}  bwd {fb - fw:.1f} us")
e, o = res["engine"], res["ours"]
print(f"D64 L{L}: bwd {e[2]:.1f} -> {o[2]:.1f} x{e[2] / o[2]:.2f} | fwd x{e[0] / o[0]:.2f} | fwd+bwd x{e[1] / o[1]:.2f}")
