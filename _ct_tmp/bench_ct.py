"""Correctness + bench for the ConditionedTransition tail (TF32, fp32 io).

Covers, in team-gm style (parseable stdout -> table + graph later):
  - inference: dispatch (atom fused / token composed) vs torch eager + torch.compile
  - training : fwd+bwd (autograd Function) vs torch eager + torch.compile, both fwd+bwd timed

Streams: atom d=128 L in {2048,4096,8192}; token d=768 L in {384,512,768,1024}. M = L.
"""
import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from miniworld_kernels.kernels.conditioned_transition.triton.interface import (
    cond_transition_inference_dispatch,
)
from miniworld_kernels.kernels.conditioned_transition.triton.training import (
    cond_transition_train,
)


def cos(a, b):
    a = a.double().reshape(-1); b = b.double().reshape(-1)
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def ref(x, cond, Wa, Wb, Ws, Wsc, bsc):
    a = x @ Wa.t(); b = x @ Wb.t()
    h = F.silu(a) * b
    out = h @ Ws.t()
    scale = cond @ Wsc.t() + bsc
    return torch.sigmoid(scale) * out


def bench(fn, it=50, wu=20):
    for _ in range(wu):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(it):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it * 1e3  # us


def make(M, d, n=2, dc=384, dev="cuda", req=False):
    g = torch.Generator(dev).manual_seed(0)
    f = lambda *s: torch.randn(*s, device=dev, dtype=torch.float32, generator=g)
    x = f(M, d); cond = f(M, dc)
    Wa = f(n * d, d) / d ** 0.5; Wb = f(n * d, d) / d ** 0.5
    Ws = f(d, n * d) / (n * d) ** 0.5; Wsc = f(d, dc) / dc ** 0.5
    bsc = torch.full((d,), -2.0, device=dev)
    if req:
        for t in (x, cond, Wa, Wb, Ws, Wsc, bsc):
            t.requires_grad_(True)
    return x, cond, Wa, Wb, Ws, Wsc, bsc


STREAMS = [("atom", 128, (2048, 4096, 8192)), ("token", 768, (384, 512, 768, 1024))]


def inference_section():
    print("=== INFERENCE ===")
    print(f"{'stream':>6} {'M':>6} {'d':>5} | {'cos':>9} {'maxerr':>9} | {'ours_us':>9} {'eager_us':>9} {'compile_us':>10} | {'vs_eager':>8} {'vs_comp':>8}")
    print("-" * 100)
    ref_c = torch.compile(ref)
    for stream, d, Ms in STREAMS:
        for M in Ms:
            x, cond, Wa, Wb, Ws, Wsc, bsc = make(M, d)
            r = ref(x, cond, Wa, Wb, Ws, Wsc, bsc)
            y = cond_transition_inference_dispatch(x, cond, Wa, Wb, Ws, Wsc, bsc)
            c = cos(y, r); me = (y - r).abs().max().item()
            to = bench(lambda: cond_transition_inference_dispatch(x, cond, Wa, Wb, Ws, Wsc, bsc))
            te = bench(lambda: ref(x, cond, Wa, Wb, Ws, Wsc, bsc))
            tc = bench(lambda: ref_c(x, cond, Wa, Wb, Ws, Wsc, bsc))
            print(f"{stream:>6} {M:>6} {d:>5} | {c:9.6f} {me:9.2e} | {to:9.1f} {te:9.1f} {tc:10.1f} | {te/to:7.2f}x {tc/to:7.2f}x")


def fwdbwd_eager(x, cond, Wa, Wb, Ws, Wsc, bsc):
    y = ref(x, cond, Wa, Wb, Ws, Wsc, bsc)
    g = torch.ones_like(y)
    grads = torch.autograd.grad(y, (x, cond, Wa, Wb, Ws, Wsc, bsc), g, retain_graph=False)
    return y, grads


def training_section():
    print("=== TRAINING (fwd+bwd) ===")
    print(f"{'stream':>6} {'M':>6} {'d':>5} | {'cos_y':>8} {'cos_dx':>8} {'cos_dcond':>9} {'cos_dWa':>8} {'cos_dWs':>8} {'cos_dWsc':>8} {'cos_dbsc':>8} | {'ours_us':>9} {'eager_us':>9} {'compile_us':>10} | {'vs_eager':>8} {'vs_comp':>8}")
    print("-" * 150)
    ref_c = torch.compile(ref)

    def fwdbwd_ours(x, cond, Wa, Wb, Ws, Wsc, bsc):
        y = cond_transition_train(x, cond, Wa, Wb, Ws, Wsc, bsc)
        g = torch.ones_like(y)
        grads = torch.autograd.grad(y, (x, cond, Wa, Wb, Ws, Wsc, bsc), g)
        return y, grads

    def fwdbwd_compile(x, cond, Wa, Wb, Ws, Wsc, bsc):
        y = ref_c(x, cond, Wa, Wb, Ws, Wsc, bsc)
        g = torch.ones_like(y)
        grads = torch.autograd.grad(y, (x, cond, Wa, Wb, Ws, Wsc, bsc), g)
        return y, grads

    for stream, d, Ms in STREAMS:
        for M in Ms:
            t = make(M, d, req=True)
            x, cond, Wa, Wb, Ws, Wsc, bsc = t
            yr, gr = fwdbwd_eager(*t)
            yo, go = fwdbwd_ours(*t)
            cy = cos(yo, yr)
            cg = [cos(a, b) for a, b in zip(go, gr)]  # dx,dcond,dWa,dWb,dWs,dWsc,dbsc
            to = bench(lambda: fwdbwd_ours(*t))
            te = bench(lambda: fwdbwd_eager(*t))
            tc = bench(lambda: fwdbwd_compile(*t))
            print(f"{stream:>6} {M:>6} {d:>5} | {cy:8.5f} {cg[0]:8.5f} {cg[1]:9.5f} {cg[2]:8.5f} {cg[4]:8.5f} {cg[5]:8.5f} {cg[6]:8.5f} | {to:9.1f} {te:9.1f} {tc:10.1f} | {te/to:7.2f}x {tc/to:7.2f}x")


def main():
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name()}")
    inference_section()
    print()
    training_section()
    print("DONE")


if __name__ == "__main__":
    main()
