"""Correctness + bench: NO-cuBLAS fused-triton training vs cuBLAS-training / eager / compile.

Also a per-wgrad-GEMM micro comparison (fused-triton vs cuBLAS). TF32, fp32 io.
"""
import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from miniworld_kernels.kernels.conditioned_transition.triton.train_fused import (
    cond_transition_train_fused, set_wgrad_backend,
)
from miniworld_kernels.kernels.conditioned_transition.triton.training import (
    cond_transition_train,  # existing cuBLAS-GEMM training path
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


def make(M, d, n=2, dc=384, dev="cuda"):
    g = torch.Generator(dev).manual_seed(0)
    f = lambda *s: torch.randn(*s, device=dev, dtype=torch.float32, generator=g)
    x = f(M, d).requires_grad_(True); cond = f(M, dc).requires_grad_(True)
    Wa = (f(n * d, d) / d ** 0.5).requires_grad_(True)
    Wb = (f(n * d, d) / d ** 0.5).requires_grad_(True)
    Ws = (f(d, n * d) / (n * d) ** 0.5).requires_grad_(True)
    Wsc = (f(d, dc) / dc ** 0.5).requires_grad_(True)
    bsc = torch.full((d,), -2.0, device=dev, requires_grad=True)
    return x, cond, Wa, Wb, Ws, Wsc, bsc


STREAMS = [("atom", 128, (2048, 4096, 8192)), ("token", 768, (384, 512, 768, 1024))]
NAMES = ["dx", "dcond", "dWa", "dWb", "dWs", "dWsc", "dbsc"]


def fb(fn, t):
    y = fn(*t)
    g = torch.ones_like(y)
    grads = torch.autograd.grad(y, t, g)
    return y, grads


def correctness():
    print("=== CORRECTNESS (fused-triton training, both wgrad backends) vs autograd ref ===")
    for be in ("cublas", "triton"):
        set_wgrad_backend(be)
        print(f"-- wgrad backend = {be} --")
        for stream, d, Ms in STREAMS:
            for M in (Ms[0], Ms[-1]):
                t = make(M, d)
                yr, gr = fb(ref, t)
                yo, go = fb(cond_transition_train_fused, t)
                cy = cos(yo, yr)
                cg = [cos(a, b) for a, b in zip(go, gr)]
                worst = min([cy] + cg)
                flag = "OK" if worst >= 0.999 else "FAIL"
                cgs = " ".join(f"{n}={c:.5f}" for n, c in zip(NAMES, cg))
                print(f"  {stream:>5} M={M:>5} d={d:>3} | cos_y={cy:.5f} {cgs} | worst={worst:.5f} {flag}")


def micro_wgrad():
    print("=== PER-WGRAD-GEMM MICRO: fused-triton vs cuBLAS (us, speedup=cublas/triton) ===")
    from miniworld_kernels.kernels.conditioned_transition.triton.train_fused import _wgrad
    print(f"{'stream':>6} {'M':>6} {'d':>4} | {'gemm':>5} {'N':>5} {'K':>5} | {'triton_us':>9} {'cublas_us':>9} {'tri/cub':>7}")
    for stream, d, Ms in STREAMS:
        for M in Ms:
            t = make(M, d)
            x, cond, Wa, Wb, Ws, Wsc, bsc = (z.detach() for z in t)
            ND = Wa.shape[0]; D = Ws.shape[0]; DC = cond.shape[1]; K = d
            dout = torch.randn(M, D, device="cuda")
            dscale = torch.randn(M, D, device="cuda")
            h = torch.randn(M, ND, device="cuda")
            dab = torch.randn(M, 2 * ND, device="cuda")
            cases = [
                ("dWs", D, ND, dout, h),
                ("dWsc", D, DC, dscale, cond),
                ("dWab", 2 * ND, K, dab, x),
            ]
            for name, N, Kk, g, xx in cases:
                tt = bench(lambda: _wgrad(g, xx, N, Kk))
                tc = bench(lambda: g.t() @ xx)
                print(f"{stream:>6} {M:>6} {d:>4} | {name:>5} {N:>5} {Kk:>5} | {tt:9.1f} {tc:9.1f} {tt/tc:6.2f}x")


def micro_dgrad():
    print("=== PER-DGRAD-GEMM MICRO: fused-triton vs cuBLAS (us, tri/cub) ===")
    from miniworld_kernels.kernels.conditioned_transition.triton.train_fused import _dgemm, _dx_fused
    print(f"{'stream':>6} {'M':>6} {'d':>4} | {'gemm':>6} | {'triton_us':>9} {'cublas_us':>9} {'tri/cub':>7}")
    for stream, d, Ms in STREAMS:
        for M in Ms:
            t = make(M, d)
            x, cond, Wa, Wb, Ws, Wsc, bsc = (z.detach() for z in t)
            ND = Wa.shape[0]; D = Ws.shape[0]; DC = cond.shape[1]
            dout = torch.randn(M, D, device="cuda")
            dscale = torch.randn(M, D, device="cuda")
            dh = torch.randn(M, ND, device="cuda")
            ab = torch.randn(M, 2 * ND, device="cuda")
            # dh = dout @ Ws
            tt = bench(lambda: _dgemm(dout, Ws, M, ND, D, Ws.stride(0), Ws.stride(1)))
            tc = bench(lambda: dout @ Ws)
            print(f"{stream:>6} {M:>6} {d:>4} | {'dh':>6} | {tt:9.1f} {tc:9.1f} {tt/tc:6.2f}x")
            # dcond = dscale @ Wsc
            tt = bench(lambda: _dgemm(dscale, Wsc, M, DC, D, Wsc.stride(0), Wsc.stride(1)))
            tc = bench(lambda: dscale @ Wsc)
            print(f"{stream:>6} {M:>6} {d:>4} | {'dcond':>6} | {tt:9.1f} {tc:9.1f} {tt/tc:6.2f}x")
            # dx fused vs cuBLAS dab@wcat
            wcat = torch.cat([Wa, Wb], 0)
            dab = torch.randn(M, 2 * ND, device="cuda")
            tt = bench(lambda: _dx_fused(dh, ab, Wa, Wb))
            tc = bench(lambda: dab @ wcat)
            print(f"{stream:>6} {M:>6} {d:>4} | {'dx':>6} | {tt:9.1f} {tc:9.1f} {tt/tc:6.2f}x")


def end_to_end():
    print("=== END-TO-END fwd+bwd (us) : fused-triton (best wgrad) vs cuBLAS-train / eager / compile ===")
    print(f"{'stream':>6} {'M':>6} {'d':>4} | {'cos_min':>8} | {'fused_us':>9} {'cublasTr':>9} {'eager_us':>9} {'compile':>9} | {'vs_cublasTr':>11} {'vs_eager':>9} {'vs_comp':>8}")
    ref_c = torch.compile(ref)

    def fb_compile(t):
        y = ref_c(*t)
        g = torch.ones_like(y)
        return torch.autograd.grad(y, t, g)

    for stream, d, Ms in STREAMS:
        # choose wgrad backend per stream from the micro results manually? Use cublas default,
        # but also report triton if it would be faster. Here: measure both, pick min.
        for M in Ms:
            t = make(M, d)
            yr, gr = fb(ref, t)
            # correctness with current default (set below per the verdict)
            best_us = None; best_be = None
            for be in ("cublas", "triton"):
                set_wgrad_backend(be)
                yo, go = fb(cond_transition_train_fused, t)
                cmin = min([cos(yo, yr)] + [cos(a, b) for a, b in zip(go, gr)])
                us = bench(lambda: fb(cond_transition_train_fused, t))
                if best_us is None or us < best_us:
                    best_us, best_be, best_cmin = us, be, cmin
            tc = bench(lambda: fb(cond_transition_train, t))   # existing cuBLAS-GEMM training
            te = bench(lambda: fb(ref, t))
            tk = bench(lambda: fb_compile(t))
            print(f"{stream:>6} {M:>6} {d:>4} | {best_cmin:8.5f} | {best_us:9.1f} {tc:9.1f} {te:9.1f} {tk:9.1f} | "
                  f"{tc/best_us:10.2f}x {te/best_us:8.2f}x {tk/best_us:7.2f}x  [{best_be}]")


def main():
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name()}")
    correctness()
    print()
    micro_dgrad()
    print()
    micro_wgrad()
    print()
    end_to_end()
    print("DONE")


if __name__ == "__main__":
    main()
