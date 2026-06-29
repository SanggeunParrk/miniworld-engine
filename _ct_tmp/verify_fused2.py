"""Correctness of the fused-prologue dgrad training path; eager + CUDA-graph timing."""
import torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from miniworld_kernels.kernels.conditioned_transition.triton.train_fused import (
    cond_transition_train_fused,
)
from miniworld_kernels.kernels.conditioned_transition.triton.training import cond_transition_train


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
    x = f(M, d).requires_grad_(True); cond = f(M, dc).requires_grad_(True)
    Wa = (f(n * d, d) / d ** 0.5).requires_grad_(True); Wb = (f(n * d, d) / d ** 0.5).requires_grad_(True)
    Ws = (f(d, n * d) / (n * d) ** 0.5).requires_grad_(True); Wsc = (f(d, dc) / dc ** 0.5).requires_grad_(True)
    bsc = torch.full((d,), -2.0, device=dev, requires_grad=True)
    return (x, cond, Wa, Wb, Ws, Wsc, bsc)


def fb(fn, t):
    y = fn(*t); g = torch.ones_like(y)
    return y, torch.autograd.grad(y, t, g)


def bench_eager(fn, t, it=50, wu=20):
    for _ in range(wu): fn(t)
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(it): fn(t)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it * 1e3


def bench_graph(fn, t, it=50, wu=10):
    # capture fwd+bwd in a CUDA graph (kernels only; removes per-call launch overhead)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn(t)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(t)
    for _ in range(wu): g.replay()
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(True); ev1 = torch.cuda.Event(True); ev0.record()
    for _ in range(it): g.replay()
    ev1.record(); torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / it * 1e3


NAMES = ["dx", "dcond", "dWa", "dWb", "dWs", "dWsc", "dbsc"]
STREAMS = [("atom", 128, (2048, 4096, 8192)), ("token", 768, (384, 512, 768, 1024))]


def main():
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name()}")
    print("=== CORRECTNESS (fused-prologue dgrad) ===")
    for stream, d, Ms in STREAMS:
        for M in (Ms[0], Ms[-1]):
            t = make(M, d)
            yr, gr = fb(ref, t)
            yo, go = fb(cond_transition_train_fused, t)
            worst = min([cos(yo, yr)] + [cos(a, b) for a, b in zip(go, gr)])
            cgs = " ".join(f"{n}={cos(a,b):.5f}" for n, a, b in zip(NAMES, go, gr))
            print(f"  {stream:>5} M={M:>5} d={d:>3} cos_y={cos(yo,yr):.5f} {cgs} worst={worst:.5f} {'OK' if worst>=0.999 else 'FAIL'}")

    print("\n=== EAGER fwd+bwd (us) ===")
    print(f"{'stream':>6} {'M':>6} {'d':>4} | {'fused':>8} {'cublasTr':>9} {'eager':>8} {'compile':>8} | {'vs_eager':>8} {'vs_cTr':>7}")
    ref_c = torch.compile(ref)
    def fbc(t):
        y = ref_c(*t); return torch.autograd.grad(y, t, torch.ones_like(y))
    for stream, d, Ms in STREAMS:
        for M in Ms:
            t = make(M, d)
            ff = bench_eager(lambda tt: fb(cond_transition_train_fused, tt), t)
            cc = bench_eager(lambda tt: fb(cond_transition_train, tt), t)
            ee = bench_eager(lambda tt: fb(ref, tt), t)
            kk = bench_eager(fbc, t)
            print(f"{stream:>6} {M:>6} {d:>4} | {ff:8.1f} {cc:9.1f} {ee:8.1f} {kk:8.1f} | {ee/ff:7.2f}x {cc/ff:6.2f}x")

    print("\n=== CUDA-GRAPH fwd+bwd (us) — true kernel cost, no launch floor ===")
    print(f"{'stream':>6} {'M':>6} {'d':>4} | {'fused':>8} {'cublasTr':>9} {'eager':>8} | {'vs_eager':>8} {'vs_cTr':>7}")
    for stream, d, Ms in STREAMS:
        for M in Ms:
            t = make(M, d)
            try:
                ff = bench_graph(lambda tt: fb(cond_transition_train_fused, tt), t)
                cc = bench_graph(lambda tt: fb(cond_transition_train, tt), t)
                ee = bench_graph(lambda tt: fb(ref, tt), t)
                print(f"{stream:>6} {M:>6} {d:>4} | {ff:8.1f} {cc:9.1f} {ee:8.1f} | {ee/ff:7.2f}x {cc/ff:6.2f}x")
            except Exception as ex:
                print(f"{stream:>6} {M:>6} {d:>4} | GRAPH-FAIL {type(ex).__name__}: {str(ex)[:60]}")
    print("DONE")


if __name__ == "__main__":
    main()
