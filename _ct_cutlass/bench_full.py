"""FULL pure-CUTLASS e2e: cond_transition_train_full vs cond_transition_train vs compile.
CUDA graph, atom + token. 7 grads + cos_y correctness. Mirrors bench_e2e.py."""
import sys
import torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
except Exception:
    pass
sys.path.insert(0, "/home/psk6950/miniworld-kernels/_ct_cutlass")
from miniworld_kernels.kernels.conditioned_transition.triton.training import (
    cond_transition_train, set_forward_mode,
)
from ct_full import cond_transition_train_full


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


def main():
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name()}")
    set_forward_mode("auto")
    shapes = [(s, d, M) for s, d, Ms in STREAMS for M in Ms]
    cosmin = {}
    for s, d, M in shapes:
        print(f"  [correctness] {s} M={M} d={d} ...", flush=True)
        tc = make(M, d)
        yr = ref(*tc); gr = torch.autograd.grad(yr, tc, torch.ones_like(yr))
        yc = cond_transition_train_full(*tc)
        gc = torch.autograd.grad(yc, tc, torch.ones_like(yc))
        cosmin[(s, d, M)] = min([cos(yc, yr)] + [cos(a, b) for a, b in zip(gc, gr)])
        print(f"    cos_min={cosmin[(s,d,M)]:.5f}", flush=True)
        del tc, yr, gr, yc, gc
    torch.cuda.synchronize(); torch.cuda.empty_cache()

    cutlass = {}; ours = {}
    for s, d, M in shapes:
        cutlass[(s, d, M)] = gbench(lambda: make(M, d), cond_transition_train_full)
        ours[(s, d, M)] = gbench(lambda: make(M, d), cond_transition_train)
    compile_us = {}
    refc = torch.compile(ref, mode="reduce-overhead")
    for s, d, M in shapes:
        compile_us[(s, d, M)] = cbench(refc, lambda: make(M, d))

    print("\n=== FULL pure-CUTLASS fwd+bwd vs cond_transition_train vs compile (CUDA graph) ===")
    print(f"{'stream':>6} {'M':>6} {'d':>4} | {'cos_min':>8} | {'cutlass':>8} {'ours':>8} "
          f"{'compile':>8} | {'vs_ours':>8} {'vs_comp':>8}")
    for k in shapes:
        s, d, M = k
        flag = "" if cosmin[k] >= 0.999 else " FAIL"
        print(f"{s:>6} {M:>6} {d:>4} | {cosmin[k]:8.5f} | {cutlass[k]:8.1f} {ours[k]:8.1f} "
              f"{compile_us[k]:8.1f} | {ours[k]/cutlass[k]:7.2f}x {compile_us[k]/cutlass[k]:7.2f}x{flag}")
    print("FULL DONE")


if __name__ == "__main__":
    main()
