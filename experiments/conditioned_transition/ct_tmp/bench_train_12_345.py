"""Correctness + fwd+bwd timing for the 1+2|3+4+5-mirrored training autograd Function.

Correctness: 7 grads + cos_y vs autograd-through-torch reference (TF32), atom + token.
Timing: fwd+bwd under CUDA graph vs eager + torch.compile. TF32, fp32 io.
(perf is secondary — CUTLASS swap later; this is the reference number.)
"""
import torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
except Exception:
    pass
from miniworld_kernels.kernels.conditioned_transition.triton.train_12_345 import (
    cond_transition_train_12_345,
)


def cos(a, b):
    a = a.double().reshape(-1); b = b.double().reshape(-1)
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def ref(x, cond, Wa, Wb, Ws, Wsc, bsc):
    a = x @ Wa.t(); b = x @ Wb.t(); h = F.silu(a) * b
    out = h @ Ws.t(); scale = cond @ Wsc.t() + bsc
    return torch.sigmoid(scale) * out


def make(M, d, n=2, dc=384, dev="cuda"):
    g = torch.Generator(dev).manual_seed(0)
    f = lambda *s: torch.randn(*s, device=dev, generator=g)
    ts = [f(M, d), f(M, dc), f(n*d, d)/d**0.5, f(n*d, d)/d**0.5,
          f(d, n*d)/(n*d)**0.5, f(d, dc)/dc**0.5, torch.full((d,), -2.0, device=dev)]
    return tuple(t.detach().requires_grad_(True) for t in ts)


def fb(fn, t):
    y = fn(*t); return torch.autograd.grad(y, t, torch.ones_like(y))


def gbench(make_t, fn, it=100, wu=10):
    t = make_t()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fb(fn, t)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fb(fn, t)
    for _ in range(wu): g.replay()
    torch.cuda.synchronize()
    a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record()
    for _ in range(it): g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / it * 1e3


def cbench(fn, make_t, it=100, wu=30):
    t = make_t()
    def call():
        y = fn(*t); return torch.autograd.grad(y, t, torch.ones_like(y))
    for _ in range(wu): call()
    torch.cuda.synchronize()
    a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record()
    for _ in range(it): call()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / it * 1e3


STREAMS = [("atom", 128, (2048, 4096, 8192)), ("token", 768, (384, 512, 768, 1024))]
NAMES = ["dx", "dcond", "dWa", "dWb", "dWs", "dWsc", "dbsc"]


def main():
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name()}")
    shapes = [(s, d, M) for s, d, Ms in STREAMS for M in Ms]

    print("=== CORRECTNESS: cond_transition_train_12_345 vs autograd-through-torch ref ===")
    print(f"{'stream':>6} {'M':>6} {'d':>4} | {'cos_y':>8} " + " ".join(f"{n:>8}" for n in NAMES) + " | worst")
    cosmin = {}
    for s, d, M in shapes:
        t = make(M, d)
        yr = ref(*t); gr = torch.autograd.grad(yr, t, torch.ones_like(yr))
        yo = cond_transition_train_12_345(*t); go = torch.autograd.grad(yo, t, torch.ones_like(yo))
        cy = cos(yo, yr); cg = [cos(a, b) for a, b in zip(go, gr)]
        worst = min([cy] + cg); cosmin[(s, d, M)] = worst
        print(f"{s:>6} {M:>6} {d:>4} | {cy:8.5f} " + " ".join(f"{c:8.5f}" for c in cg) +
              f" | {worst:.5f} {'OK' if worst >= 0.999 else 'FAIL'}")
        del t, yr, gr, yo, go
    torch.cuda.synchronize(); torch.cuda.empty_cache()

    # timing: ours + eager via manual graph; compile last
    ours = {}; eager = {}
    for s, d, M in shapes:
        ours[(s, d, M)] = gbench(lambda: make(M, d), cond_transition_train_12_345)
        eager[(s, d, M)] = gbench(lambda: make(M, d), ref)
    refc = torch.compile(ref, mode="reduce-overhead")
    comp = {}
    for s, d, M in shapes:
        comp[(s, d, M)] = cbench(refc, lambda: make(M, d))

    print("\n=== TRAINING fwd+bwd (1+2|3+4+5 mirror): ours vs compile vs eager (CUDA graph, us) ===")
    print(f"{'stream':>6} {'M':>6} {'d':>4} | {'cos_min':>8} | {'ours_us':>8} {'compile_us':>10} {'eager_us':>8} | {'vs_compile':>10} {'vs_eager':>8}")
    for k in shapes:
        s, d, M = k
        print(f"{s:>6} {M:>6} {d:>4} | {cosmin[k]:8.5f} | {ours[k]:8.1f} {comp[k]:10.1f} {eager[k]:8.1f} | {comp[k]/ours[k]:9.2f}x {eager[k]/ours[k]:7.2f}x")
    print("DONE")


if __name__ == "__main__":
    main()
