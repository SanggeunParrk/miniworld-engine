"""vs-COMPILE bench (primary baseline), CUDA graph apples-to-apples. TF32, fp32 io.

Baseline = torch.compile(reference, mode="reduce-overhead") of the PURE-PYTORCH reference.
We verify there are NO graph breaks (else it's a measurement bug). Ours is measured under our
own CUDA graph. Eager kept as a context column only.

Covers inference (dispatch) and training fwd+bwd, atom (d128) + token (d768).
"""
import torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from miniworld_kernels.kernels.conditioned_transition.triton.interface import (
    cond_transition_inference_dispatch,
)
from miniworld_kernels.kernels.conditioned_transition.triton.training import (
    cond_transition_train, set_forward_mode,
)


def cos(a, b):
    a = a.double().reshape(-1); b = b.double().reshape(-1)
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def ref(x, cond, Wa, Wb, Ws, Wsc, bsc):
    a = x @ Wa.t(); b = x @ Wb.t(); h = F.silu(a) * b
    out = h @ Ws.t(); scale = cond @ Wsc.t() + bsc
    return torch.sigmoid(scale) * out


def make(M, d, n=2, dc=384, dev="cuda", req=False):
    g = torch.Generator(dev).manual_seed(0)
    f = lambda *s: torch.randn(*s, device=dev, generator=g)
    x = f(M, d); cond = f(M, dc)
    Wa = f(n*d, d)/d**0.5; Wb = f(n*d, d)/d**0.5
    Ws = f(d, n*d)/(n*d)**0.5; Wsc = f(d, dc)/dc**0.5
    bsc = torch.full((d,), -2.0, device=dev)
    ts = [x, cond, Wa, Wb, Ws, Wsc, bsc]
    if req:
        ts = [t.requires_grad_(True) for t in ts]
    return tuple(ts)


def gbench(call, it=100, wu=10):
    """Time `call` under a fresh CUDA graph capture (our path: graph-break kernels)."""
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): call()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): call()
    for _ in range(wu): g.replay()
    torch.cuda.synchronize()
    a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record()
    for _ in range(it): g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / it * 1e3


def cbench(compiled_call, it=100, wu=30):
    """Time a reduce-overhead compiled fn (it manages its own cudagraphs); plain timed loop."""
    for _ in range(wu): compiled_call()
    torch.cuda.synchronize()
    a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record()
    for _ in range(it): compiled_call()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / it * 1e3


STREAMS = [("atom", 128, (2048, 4096, 8192)), ("token", 768, (384, 512, 768, 1024))]
NAMES = ["dx", "dcond", "dWa", "dWb", "dWs", "dWsc", "dbsc"]


def check_no_graph_breaks():
    import torch._dynamo as dyno
    t = make(2048, 128)
    expl = dyno.explain(ref)(*t)
    print(f"[graph-break check] ref: graph_count={expl.graph_count} break_count={expl.graph_break_count}")
    if expl.graph_break_count != 0:
        print("  !! WARNING: reference has graph breaks — measurement bug")


def inference_section():
    print("=== INFERENCE: ours vs torch.compile (CUDA graph) ===")
    print(f"{'stream':>6} {'M':>6} {'d':>4} | {'cos':>9} | {'ours_us':>8} {'compile_us':>10} {'eager_us':>8} | {'vs_compile':>10} {'vs_eager':>8}")
    refc = torch.compile(ref, mode="reduce-overhead")
    for stream, d, Ms in STREAMS:
        for M in Ms:
            t = make(M, d)
            r = ref(*t)
            y = cond_transition_inference_dispatch(*t)
            c = cos(y, r)
            ou = gbench(lambda: cond_transition_inference_dispatch(*t))
            cc = cbench(lambda: refc(*t))
            ee = gbench(lambda: ref(*t))
            print(f"{stream:>6} {M:>6} {d:>4} | {c:9.6f} | {ou:8.1f} {cc:10.1f} {ee:8.1f} | {cc/ou:9.2f}x {ee/ou:7.2f}x")


def fb(fn, t):
    y = fn(*t); return torch.autograd.grad(y, t, torch.ones_like(y))


def training_section():
    print("\n=== TRAINING fwd+bwd: ours(auto) vs torch.compile (both CUDA graph) ===")
    print(f"{'stream':>6} {'M':>6} {'d':>4} | {'cos_min':>8} | {'ours_us':>8} {'compile_us':>10} {'eager_us':>8} | {'vs_compile':>10} {'vs_eager':>8}")
    refc = torch.compile(ref, mode="reduce-overhead")
    set_forward_mode("auto")
    def fbc(t):
        y = refc(*t); return torch.autograd.grad(y, t, torch.ones_like(y))
    for stream, d, Ms in STREAMS:
        for M in Ms:
            t = make(M, d, req=True)
            yr, gr = ref(*t), None
            gr = torch.autograd.grad(yr, t, torch.ones_like(yr))
            go = fb(cond_transition_train, t)
            cmin = min([cos(cond_transition_train(*t), yr)] + [cos(a, b) for a, b in zip(go, gr)])
            ou = gbench(lambda: fb(cond_transition_train, t))
            cc = cbench(lambda: fbc(t))
            ee = gbench(lambda: fb(ref, t))
            print(f"{stream:>6} {M:>6} {d:>4} | {cmin:8.5f} | {ou:8.1f} {cc:10.1f} {ee:8.1f} | {cc/ou:9.2f}x {ee/ou:7.2f}x")


def main():
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name()}")
    check_no_graph_breaks()
    inference_section()
    training_section()
    print("DONE")


if __name__ == "__main__":
    main()
