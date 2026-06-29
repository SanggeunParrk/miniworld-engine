"""Last-chance backward verdict at REAL M = A*L (A=48), CUDA graph, TF32.

Compares the fwd+bwd of:
  champion  = cond_transition_train          (cuBLAS dgrad + fused-triton elementwise)
  clean     = cond_transition_train_12_345_clean (NEW: elementwise + CLEAN triton dgrad GEMM)
  fused     = cond_transition_train_12_345   (gate/swiglu fused into dgrad-GEMM prologue)
  compile   = torch.compile(ref, reduce-overhead)
All wgrad on cuBLAS. Plus a per-stage dgrad GEMM micro (clean triton _dgemm vs torch.matmul)
to isolate the triton-vs-cuBLAS GEMM gap at the real dgrad shapes.
"""
import torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from miniworld_kernels.kernels.conditioned_transition.triton.training import cond_transition_train
from miniworld_kernels.kernels.conditioned_transition.triton.train_12_345 import (
    cond_transition_train_12_345,
    cond_transition_train_12_345_clean,
)
from miniworld_kernels.kernels.conditioned_transition.triton.train_fused import _dgemm

A = 48  # real batch multiplier: M = A * L


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


def gbench_call(make_args, fn, it=200, wu=20):
    """CUDA-graph time a pure callable fn(*args) (no autograd)."""
    args = make_args()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn(*args)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn(*args)
    for _ in range(wu): g.replay()
    torch.cuda.synchronize()
    a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record()
    for _ in range(it): g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / it * 1e3


STREAMS = [("atom", 128, (2048, 4096, 8192)), ("token", 768, (384, 512, 768, 1024))]
NAMES = ["dx", "dcond", "dWa", "dWb", "dWs", "dWsc", "dbsc"]


def main():
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name()}  A={A}")
    shapes = [(s, d, A * L) for s, d, Ls in STREAMS for L in Ls]

    # ---- correctness of the NEW clean variant ----
    print("\n=== CORRECTNESS clean-dgrad vs autograd ref (real M) ===")
    print(f"{'stream':>6} {'M':>7} {'d':>4} | {'cos_y':>8} " + " ".join(f"{n:>7}" for n in NAMES) + " | worst")
    cosmin = {}
    for s, d, M in shapes:
        t = make(M, d)
        yr = ref(*t); gr = torch.autograd.grad(yr, t, torch.ones_like(yr))
        yo = cond_transition_train_12_345_clean(*t); go = torch.autograd.grad(yo, t, torch.ones_like(yo))
        cy = cos(yo, yr); cg = [cos(a, b) for a, b in zip(go, gr)]
        worst = min([cy] + cg); cosmin[(s, d, M)] = worst
        print(f"{s:>6} {M:>7} {d:>4} | {cy:8.5f} " + " ".join(f"{c:7.4f}" for c in cg) +
              f" | {worst:.5f} {'OK' if worst >= 0.999 else 'FAIL'}")
        del t, yr, gr, yo, go
    torch.cuda.synchronize(); torch.cuda.empty_cache()

    # ---- end-to-end fwd+bwd (CUDA graph) ----
    champ = {}; clean = {}; fused = {}; eager = {}
    for s, d, M in shapes:
        champ[(s, d, M)] = gbench(lambda: make(M, d), cond_transition_train)
        clean[(s, d, M)] = gbench(lambda: make(M, d), cond_transition_train_12_345_clean)
        fused[(s, d, M)] = gbench(lambda: make(M, d), cond_transition_train_12_345)
        eager[(s, d, M)] = gbench(lambda: make(M, d), ref)
        torch.cuda.empty_cache()
    refc = torch.compile(ref, mode="reduce-overhead")
    comp = {}
    for s, d, M in shapes:
        comp[(s, d, M)] = cbench(refc, lambda: make(M, d)); torch.cuda.empty_cache()

    print("\n=== fwd+bwd (CUDA graph, us) — champion(cuBLAS-dg) vs clean(triton-dg) vs fused(prologue) vs compile ===")
    print(f"{'stream':>6} {'M':>7} {'d':>4} | {'cosmin':>7} | {'champ':>7} {'clean':>7} {'fused':>7} {'compile':>8} {'eager':>7} "
          f"| {'clean/champ':>11} {'clean/comp':>10}")
    for k in shapes:
        s, d, M = k
        print(f"{s:>6} {M:>7} {d:>4} | {cosmin[k]:7.4f} | {champ[k]:7.1f} {clean[k]:7.1f} {fused[k]:7.1f} {comp[k]:8.1f} {eager[k]:7.1f} "
              f"| {champ[k]/clean[k]:10.2f}x {comp[k]/clean[k]:9.2f}x")

    # ---- per-stage dgrad GEMM micro: clean triton _dgemm vs torch.matmul (cuBLAS), CUDA graph ----
    # dh = dout(M,D)@Ws(D,ND) ; dcond = dscale(M,D)@Wsc(D,DC) ; dx = dab(M,2ND)@Wcat(2ND,K)
    print("\n=== per-stage dgrad GEMM micro (CUDA graph, us): triton _dgemm vs cuBLAS matmul ===")
    print(f"{'stream':>6} {'M':>7} {'d':>4} {'gemm':>6} | {'M,K,N':>18} | {'triton':>8} {'cublas':>8} | {'tri/cub':>8}")
    for s, d, Ls in STREAMS:
        for L in Ls:
            M = A * L; ND = 2 * d; D = d; DC = 384; K = d
            dev = "cuda"
            dout = torch.randn(M, D, device=dev); ws = torch.randn(D, ND, device=dev)
            dscale = torch.randn(M, D, device=dev); wsc = torch.randn(D, DC, device=dev)
            dab = torch.randn(M, 2 * ND, device=dev); wcat = torch.randn(2 * ND, K, device=dev)
            cases = [
                ("dh", dout, ws, M, ND, D),
                ("dcond", dscale, wsc, M, DC, D),
                ("dx", dab, wcat, M, K, 2 * ND),
            ]
            for nm, aT, wT, Mm, Nn, Kk in cases:
                tri = gbench_call(lambda aT=aT, wT=wT, Mm=Mm, Nn=Nn, Kk=Kk:
                                  (aT, wT, Mm, Nn, Kk, wT.stride(0), wT.stride(1)),
                                  lambda aa, ww, mm, nn, kk, s0, s1: _dgemm(aa, ww, mm, nn, kk, s0, s1))
                cub = gbench_call(lambda aT=aT, wT=wT: (aT, wT), lambda aa, ww: aa @ ww)
                print(f"{s:>6} {M:>7} {d:>4} {nm:>6} | {f'{Mm},{Kk},{Nn}':>18} | {tri:8.1f} {cub:8.1f} | {tri/cub:7.2f}x")
            del dout, ws, dscale, wsc, dab, wcat
            torch.cuda.empty_cache()
    print("DONE")


if __name__ == "__main__":
    main()
