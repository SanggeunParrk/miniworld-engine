"""Module-level backward (fwd+bwd minus fwd) of modules.Transition, fused sm90a on vs off.  python bench_mod_bwd.py --width 128"""
import argparse, statistics, torch
p = argparse.ArgumentParser(); p.add_argument("--width", type=int, default=128); p.add_argument("--length", type=int, default=384); a = p.parse_args()
from miniworld_engine import settings
from miniworld_engine.modules import Transition
D, L = a.width, a.length
def t(fn, reps=5):
    for _ in range(3): fn()
    torch.cuda.synchronize(); o = []
    for _ in range(7):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
        for _ in range(reps): fn()
        e.record(); torch.cuda.synchronize(); o.append(s.elapsed_time(e) * 1e3 / reps)
    return statistics.median(o)
res = {}
for fused in (False, True):
    settings.configure(engine_backend="triton", transition_residual_fusion=True, transition_fused_sm90a=fused)
    torch.manual_seed(0)
    m = Transition(D, n=4, implementation="triton").cuda().bfloat16()
    with torch.no_grad():
        for prm in m.parameters():
            if prm.ndim == 2: prm.normal_(std=prm.shape[-1] ** -0.5)
    x = torch.randn(1, L, L, D, device="cuda", dtype=torch.bfloat16, requires_grad=True); dy = torch.randn_like(x)
    P = list(m.parameters())
    def step():
        for q in [x] + P: q.grad = None
        m(x).backward(dy)
    def graphed(fn):
        s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_):
            for _ in range(3): fn()
        torch.cuda.current_stream().wait_stream(s_)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): fn()
        return g.replay
    f = t(graphed(lambda: m(x))); fb = t(graphed(step))
    res[fused] = (f, fb, fb - f)
    print(f"D{D} L{L} fused={fused} (CUDA graph): fwd {f:.1f} fwd+bwd {fb:.1f} bwd {fb - f:.1f} us")
print(f"D{D} L{L}: bwd {res[False][2]:.1f} -> {res[True][2]:.1f} x{res[False][2] / res[True][2]:.2f}")
