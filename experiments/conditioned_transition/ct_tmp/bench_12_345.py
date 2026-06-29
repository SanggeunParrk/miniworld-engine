"""Forward-only bench: 1+2|3+4+5 two-kernel path vs b2b single-kernel vs eager vs compile.

CUDA graph for ours/b2b/eager; compile uses its own cudagraphs (reduce-overhead). TF32, fp32 io.
atom (d128, L 2048/4096/8192) + token (d768, L 384/512/768/1024). cos vs torch ref.
"""
import torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from miniworld_kernels.kernels.conditioned_transition.triton.composed import (
    cond_transition_fwd_12_345,
)
from miniworld_kernels.kernels.conditioned_transition.triton.inference import (
    cond_transition_inference,  # the atom b2b single-kernel
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
    x = f(M, d); cond = f(M, dc)
    Wa = f(n*d, d)/d**0.5; Wb = f(n*d, d)/d**0.5
    Ws = f(d, n*d)/(n*d)**0.5; Wsc = f(d, dc)/dc**0.5
    bsc = torch.full((d,), -2.0, device=dev)
    return (x, cond, Wa, Wb, Ws, Wsc, bsc)


def gbench(call, it=100, wu=10):
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


def cbench(call, it=100, wu=30):
    for _ in range(wu): call()
    torch.cuda.synchronize()
    a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record()
    for _ in range(it): call()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / it * 1e3


STREAMS = [("atom", 128, (2048, 4096, 8192)), ("token", 768, (384, 512, 768, 1024))]


def main():
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name()}")
    # PASS A: ours / b2b / eager via manual graph (no compile live)
    rows = []
    for stream, d, Ms in STREAMS:
        for M in Ms:
            t = make(M, d)
            r = ref(*t)
            y = cond_transition_fwd_12_345(*t)
            c = cos(y, r)
            us = gbench(lambda: cond_transition_fwd_12_345(*t))
            # b2b single-kernel: only valid for atom (d<=128); token won't compile
            try:
                yb = cond_transition_inference(*t)
                cb = cos(yb, r)
                b2b = gbench(lambda: cond_transition_inference(*t))
            except Exception as ex:
                cb = float('nan'); b2b = float('nan')
            ee = gbench(lambda: ref(*t))
            rows.append([stream, M, d, c, cb, us, b2b, ee])
    # PASS B: compile (reduce-overhead) last
    refc = torch.compile(ref, mode="reduce-overhead")
    comp = {}
    for stream, d, Ms in STREAMS:
        for M in Ms:
            t = make(M, d)
            comp[(stream, M)] = cbench(lambda: refc(*t))

    print("\n=== FORWARD: 1+2|3+4+5 vs b2b vs eager vs compile (CUDA graph, us) ===")
    print(f"{'stream':>6} {'M':>6} {'d':>4} | {'cos':>9} {'cos_b2b':>9} | {'12_345':>8} {'b2b':>8} {'eager':>8} {'compile':>8} | "
          f"{'vs_b2b':>7} {'vs_eager':>8} {'vs_comp':>8}")
    for stream, M, d, c, cb, us, b2b, ee in rows:
        cc = comp[(stream, M)]
        b2bs = f"{b2b/us:6.2f}x" if b2b == b2b else "   n/a"
        b2bus = f"{b2b:8.1f}" if b2b == b2b else "     n/a"
        cbs = f"{cb:9.6f}" if cb == cb else "      n/a"
        print(f"{stream:>6} {M:>6} {d:>4} | {c:9.6f} {cbs} | {us:8.1f} {b2bus} {ee:8.1f} {cc:8.1f} | "
              f"{b2bs} {ee/us:7.2f}x {cc/us:7.2f}x")
    print("DONE")


if __name__ == "__main__":
    main()
