"""Correctness + bench for the fused ConditionedTransition INFERENCE kernel (TF32, fp32 io)."""
import torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from miniworld_kernels.kernels.conditioned_transition.triton.inference import cond_transition_inference


def cos(a, b):
    a = a.double().reshape(-1); b = b.double().reshape(-1)
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def ref(x, cond, Wa, Wb, Ws, Wsc, bsc):
    a = x @ Wa.t(); b = x @ Wb.t()
    h = F.silu(a) * b
    out = h @ Ws.t()
    scale = cond @ Wsc.t() + bsc
    return torch.sigmoid(scale) * out


def bench(fn, it=100, wu=30):
    for _ in range(wu): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it * 1e3  # us


def make(M, d, n=2, dc=384, dev="cuda"):
    torch.manual_seed(0); g = torch.Generator(dev).manual_seed(0)
    f = lambda *s: torch.randn(*s, device=dev, dtype=torch.float32, generator=g)
    x = f(M, d); cond = f(M, dc)
    Wa = f(n*d, d)/d**0.5; Wb = f(n*d, d)/d**0.5
    Ws = f(d, n*d)/(n*d)**0.5; Wsc = f(d, dc)/dc**0.5
    bsc = torch.full((d,), -2.0, device=dev)
    return x, cond, Wa, Wb, Ws, Wsc, bsc


def main():
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name()}")
    print(f"{'stream':>6} {'M':>6} {'d':>5} | {'cos':>9} {'maxerr':>9} | {'fused_us':>9} {'torch_us':>9} {'speedup':>7}")
    print("-"*72)
    for stream, d, Ms in [("atom",128,(2048,4096,8192)), ("token",768,(384,1024))]:
        for M in Ms:
            x,cond,Wa,Wb,Ws,Wsc,bsc = make(M,d)
            r = ref(x,cond,Wa,Wb,Ws,Wsc,bsc)
            try:
                y = cond_transition_inference(x,cond,Wa,Wb,Ws,Wsc,bsc)
                c = cos(y,r); me = (y-r).abs().max().item()
                tf = bench(lambda: cond_transition_inference(x,cond,Wa,Wb,Ws,Wsc,bsc))
                tt = bench(lambda: ref(x,cond,Wa,Wb,Ws,Wsc,bsc))
                print(f"{stream:>6} {M:>6} {d:>5} | {c:9.6f} {me:9.2e} | {tf:9.1f} {tt:9.1f} {tt/tf:6.2f}x")
            except Exception as ex:
                print(f"{stream:>6} {M:>6} {d:>5} | FAIL {type(ex).__name__}: {str(ex)[:50]}")
    print("DONE")


if __name__ == "__main__":
    main()
