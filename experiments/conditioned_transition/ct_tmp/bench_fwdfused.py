"""CUDA-graph fwd+bwd bench: NEW training (fused-triton forward + cuBLAS bwd + fused elem)
vs (a) OLD cuBLAS-forward training, (b) torch eager, (c) torch.compile. TF32, fp32 io.

Also a fwd-only CUDA-graph micro: fused-triton forward vs cuBLAS forward (where the lever is).
"""
import torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from miniworld_kernels.kernels.conditioned_transition.triton.training import (
    cond_transition_train,  # NEW: fused-triton forward + cuBLAS bwd
    _swiglu, _gate, _gate_bwd, _swiglu_bwd_packed,
)


# --- OLD cuBLAS-forward training (baseline) as an inline autograd Function ---
class _OldCublasFwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cond, wa, wb, ws, wsc, bsc):
        ND = wa.shape[0]
        wcat = torch.cat([wa, wb], dim=0)
        ab = x @ wcat.t()
        a, b = ab[:, :ND], ab[:, ND:]
        h = _swiglu(a, b)
        out = h @ ws.t()
        scale = torch.addmm(bsc, cond, wsc.t())
        y = _gate(out, scale)
        ctx.save_for_backward(x, cond, ab, h, out, scale, wcat, ws, wsc)
        ctx.ND = ND
        return y

    @staticmethod
    def backward(ctx, dy):
        x, cond, ab, h, out, scale, wcat, ws, wsc = ctx.saved_tensors
        ND = ctx.ND
        a, b = ab[:, :ND], ab[:, ND:]
        dy = dy.contiguous()
        dout, dscale = _gate_bwd(out, scale, dy)
        dcond = dscale @ wsc; dWsc = dscale.t() @ cond; db_sc = dscale.sum(0)
        dh = dout @ ws; dWs = dout.t() @ h
        dab = _swiglu_bwd_packed(a, b, dh)
        dx = dab @ wcat; dWcat = dab.t() @ x
        dWa, dWb = dWcat[:ND], dWcat[ND:]
        return dx, dcond, dWa.contiguous(), dWb.contiguous(), dWs, dWsc, db_sc


def old_train(x, cond, wa, wb, ws, wsc, bsc):
    return _OldCublasFwd.apply(x, cond, wa, wb, ws, wsc, bsc)


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


def gbench(call, it=80, wu=10):
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


NAMES = ["dx", "dcond", "dWa", "dWb", "dWs", "dWsc", "dbsc"]
STREAMS = [("atom", 128, (2048, 4096, 8192)), ("token", 768, (384, 512, 768, 1024))]


def correctness():
    print("=== CORRECTNESS (NEW fused-fwd training) vs autograd ref ===")
    for stream, d, Ms in STREAMS:
        for M in (Ms[0], Ms[-1]):
            t = make(M, d)
            yr, gr = fb(ref, t)
            yo, go = fb(cond_transition_train, t)
            worst = min([cos(yo, yr)] + [cos(a, b) for a, b in zip(go, gr)])
            cgs = " ".join(f"{n}={cos(a,b):.5f}" for n, a, b in zip(NAMES, go, gr))
            print(f"  {stream:>5} M={M:>5} d={d:>3} cos_y={cos(yo,yr):.5f} {cgs} worst={worst:.5f} {'OK' if worst>=0.999 else 'FAIL'}")


def fwd_micro():
    print("\n=== FWD-ONLY CUDA-graph (us): fused-triton fwd vs cuBLAS fwd (the lever) ===")
    from miniworld_kernels.kernels.conditioned_transition.triton.training import _fused_fwd_train
    print(f"{'stream':>6} {'M':>6} {'d':>4} | {'fused_fwd':>9} {'cublas_fwd':>10} {'speedup':>7}")
    for stream, d, Ms in STREAMS:
        for M in Ms:
            t = make(M, d)
            x, cond, Wa, Wb, Ws, Wsc, bsc = (z.detach() for z in t)
            def cub_fwd():
                wcat = torch.cat([Wa, Wb], 0); ab = x @ wcat.t()
                a, b = ab[:, :Wa.shape[0]], ab[:, Wa.shape[0]:]
                h = _swiglu(a, b); out = h @ Ws.t(); scale = torch.addmm(bsc, cond, Wsc.t())
                return _gate(out, scale)
            ff = gbench(lambda: _fused_fwd_train(x, cond, Wa, Wb, Ws, Wsc, bsc))
            cc = gbench(cub_fwd)
            print(f"{stream:>6} {M:>6} {d:>4} | {ff:9.1f} {cc:10.1f} {cc/ff:6.2f}x")


def e2e():
    print("\n=== CUDA-graph fwd+bwd (us): NEW(fused-fwd) vs OLD(cuBLAS-fwd) vs eager vs compile ===")
    print(f"{'stream':>6} {'M':>6} {'d':>4} | {'NEW':>8} {'OLD':>8} {'eager':>8} {'compile':>8} | {'vs_eager':>8} {'vs_OLD':>7} {'vs_comp':>7}")
    ref_c = torch.compile(ref)
    def fbc(t):
        y = ref_c(*t); return torch.autograd.grad(y, t, torch.ones_like(y))
    for stream, d, Ms in STREAMS:
        for M in Ms:
            t = make(M, d)
            nn = gbench(lambda: fb(cond_transition_train, t))
            oo = gbench(lambda: fb(old_train, t))
            ee = gbench(lambda: fb(ref, t))
            try:
                kk = gbench(lambda: fbc(t))
            except Exception:
                kk = float('nan')
            print(f"{stream:>6} {M:>6} {d:>4} | {nn:8.1f} {oo:8.1f} {ee:8.1f} {kk:8.1f} | {ee/nn:7.2f}x {oo/nn:6.2f}x {kk/nn:6.2f}x")


def main():
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name()}")
    correctness(); fwd_micro(); e2e(); print("DONE")


if __name__ == "__main__":
    main()
