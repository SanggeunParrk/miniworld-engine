"""The shipped Transition forward in INFERENCE (no_grad, the FWD_SAVE = 0 build) through `modules.Transition`:
bit-identity against the training build's forward, and timing against the training forward and the Triton path.

  python bench_infer.py --length 384
"""
import argparse, copy, statistics, torch

p = argparse.ArgumentParser(); p.add_argument("--length", type=int, default=384); p.add_argument("--reps", type=int, default=50)
a = p.parse_args()
from miniworld_engine import settings                      # noqa: E402
from miniworld_engine.modules import Transition           # noqa: E402

D, N, L = 128, 4, a.length
torch.manual_seed(2319)
base = Transition(D, n=N, implementation="triton").cuda().bfloat16()
with torch.no_grad():
    for prm in base.parameters():
        if prm.ndim == 2:
            prm.normal_(std=D ** -0.5)                      # squeeze is zero-init: a zero update would compare nothing
x = torch.randn(1, L, L, D, device="cuda", dtype=torch.bfloat16)


def module(fused):
    settings.configure(engine_backend="triton", transition_residual_fusion=True, transition_fused_sm90a=fused)
    return copy.deepcopy(base)


def timed(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            fn()
    torch.cuda.synchronize()
    o = []
    for _ in range(5):
        st, en = torch.cuda.Event(True), torch.cuda.Event(True)
        st.record()
        for _ in range(a.reps):
            g.replay()
        en.record(); torch.cuda.synchronize()
        o.append(st.elapsed_time(en) * 1e3 / a.reps)
    return statistics.median(o)


fused, trit = module(True), module(False)
settings.configure(engine_backend="triton", transition_residual_fusion=True, transition_fused_sm90a=True)
xg = x.clone().requires_grad_(True)
y_train = fused(xg).detach()                                 # training build (saves xn): the reference bytes
with torch.no_grad():
    y_inf = fused(x)                                         # inference build
print(f"L{L}: inference output == training-build output: {bool(torch.equal(y_inf, y_train))}  "
      f"(differing elements {int((y_inf != y_train).sum())})")


def inf_fused():
    with torch.no_grad():
        return fused(x)


def train_fwd():
    return fused(xg)


t_inf = timed(inf_fused)
t_train = timed(train_fwd)
settings.configure(engine_backend="triton", transition_residual_fusion=True, transition_fused_sm90a=False)


def inf_triton():
    with torch.no_grad():
        return trit(x)


t_trit = timed(inf_triton)
print(f"  inference fused  {t_inf:7.1f} us | training forward (fused, saves xn) {t_train:7.1f} us | "
      f"inference Triton path {t_trit:7.1f} us  ->  {t_trit / t_inf:.2f}x vs Triton")
