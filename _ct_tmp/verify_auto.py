"""Confirm _FWD_MODE=auto: correctness + that e2e tracks the per-regime winner (CUDA graph)."""
import torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from miniworld_kernels.kernels.conditioned_transition.triton.training import (
    cond_transition_train, set_forward_mode, _pick_fwd,
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
    x = f(M, d).requires_grad_(True); cond = f(M, dc).requires_grad_(True)
    Wa = (f(n*d, d)/d**0.5).requires_grad_(True); Wb = (f(n*d, d)/d**0.5).requires_grad_(True)
    Ws = (f(d, n*d)/(n*d)**0.5).requires_grad_(True); Wsc = (f(d, dc)/dc**0.5).requires_grad_(True)
    bsc = torch.full((d,), -2.0, device=dev, requires_grad=True)
    return (x, cond, Wa, Wb, Ws, Wsc, bsc)


def fb(fn, t):
    y = fn(*t); return torch.autograd.grad(y, t, torch.ones_like(y))


def gbench(call, it=80, wu=10):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): call()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): call()
    for _ in range(wu): g.replay()
    torch.cuda.synchronize()
    a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
    for _ in range(it): g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b)/it*1e3


STREAMS = [("atom",128,(2048,4096,8192)), ("token",768,(384,512,768,1024))]
NAMES=["dx","dcond","dWa","dWb","dWs","dWsc","dbsc"]


def main():
    print(torch.cuda.get_device_name())
    set_forward_mode("auto")
    print("=== CORRECTNESS (_FWD_MODE=auto) ===")
    for stream,d,Ms in STREAMS:
        for M in (Ms[0],Ms[-1]):
            t=make(M,d); yr=ref(*t); gr=torch.autograd.grad(yr,t,torch.ones_like(yr))
            yo=cond_transition_train(*t); go=torch.autograd.grad(yo,t,torch.ones_like(yo))
            worst=min([cos(yo,yr)]+[cos(a,b) for a,b in zip(go,gr)])
            print(f"  {stream:>5} M={M:>5} d={d:>3} pick={_pick_fwd(d,M):>6} worst_cos={worst:.5f} {'OK' if worst>=0.999 else 'FAIL'}")
    print("\n=== AUTO e2e vs eager (CUDA graph, us) ===")
    print(f"{'stream':>6} {'M':>6} {'d':>4} | {'pick':>6} {'auto_us':>8} {'eager_us':>8} {'vs_eager':>8}")
    for stream,d,Ms in STREAMS:
        for M in Ms:
            t=make(M,d)
            au=gbench(lambda: fb(cond_transition_train,t))
            ee=gbench(lambda: fb(ref,t))
            print(f"{stream:>6} {M:>6} {d:>4} | {_pick_fwd(d,M):>6} {au:8.1f} {ee:8.1f} {ee/au:7.2f}x")
    print("DONE")


if __name__=="__main__":
    main()
