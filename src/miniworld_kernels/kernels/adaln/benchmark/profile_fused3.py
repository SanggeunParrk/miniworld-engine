"""Profile fused3: (mode=breakdown) per-kernel K1/K2/K3 timing+roofline; (mode=ncu) one K3 launch."""
from __future__ import annotations
import sys, torch, triton

from miniworld_kernels.kernels.adaln.triton.fused3 import _layernorm, _gemm_gate, adaln_fused3

DEVICE = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
HBM = 3.35e12
TF32 = 494e12
eps = 1e-5


def t(fn):
    m, _, _ = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8]); return m * 1000


def mk(M, d):
    x = torch.randn(M, d, device=DEVICE, dtype=torch.float32)
    cond = torch.randn(M, d, device=DEVICE, dtype=torch.float32)
    lnw = torch.randn(d, device=DEVICE, dtype=torch.float32)
    Ws = torch.randn(d, d, device=DEVICE, dtype=torch.float32) * d ** -0.5
    Wb = torch.randn(d, d, device=DEVICE, dtype=torch.float32) * d ** -0.5
    sb = torch.randn(d, device=DEVICE, dtype=torch.float32) * 0.1
    return x, cond, lnw, Ws, Wb, sb


def breakdown():
    print("torch", torch.__version__, torch.cuda.get_device_name(0))
    for tag, d, seq in [("TOKEN", 768, 1024), ("ATOM", 128, 8192)]:
        M = 32 * seq
        x, cond, lnw, Ws, Wb, sb = mk(M, d)
        xn = _layernorm(x, eps)
        cn = _layernorm(cond, eps, lnw)
        tk1 = t(lambda: _layernorm(x, eps))
        tk2 = t(lambda: _layernorm(cond, eps, lnw))
        tk3 = t(lambda: _gemm_gate(xn, cn, Ws, Wb, sb))
        tot = t(lambda: adaln_fused3(x, cond, lnw, Ws, sb, Wb, eps, eps))
        gf = 2 * 2 * M * d * d / 1e9  # 2 GEMMs
        k1_gb = (M * d * 2) * 4 / 1e9
        k3_tf = gf / (tk3 / 1e6) / 1e12
        print(f"\n#### {tag} d={d} M={M}  total={tot:.1f}us")
        print(f"  K1 LN(x)    : {tk1:6.1f}us  ({k1_gb/(tk1/1e6)/1e3*1000:.0f} GB/s, {k1_gb*1000/(tk1/1e6)/HBM*1e9*1e-9*100:.0f}% HBM)")
        print(f"  K2 LN(cond) : {tk2:6.1f}us")
        print(f"  K3 GEMM+gate: {tk3:6.1f}us  ({gf:.0f}GF → {k3_tf:.0f} TF/s, {k3_tf*1e12/TF32*100:.0f}% TF32peak)")
        print(f"  sum K1+K2+K3 = {tk1+tk2+tk3:.1f}us  (cfg K3={_gemm_gate.__module__ and 'autotuned'})")


def ncu_target():
    # Warmup (triggers autotune for K1/K2/K3), then a steady stream of K3 launches for ncu to grab.
    M, d = 32 * 1024, 768
    x, cond, lnw, Ws, Wb, sb = mk(M, d)
    xn = _layernorm(x, eps)
    cn = _layernorm(cond, eps, lnw)
    for _ in range(30):  # finish autotune + warm
        _gemm_gate(xn, cn, Ws, Wb, sb)
    torch.cuda.synchronize()
    for _ in range(400):  # steady-state launches; ncu skips early ones
        _gemm_gate(xn, cn, Ws, Wb, sb)
    torch.cuda.synchronize()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ncu":
        ncu_target()
    else:
        breakdown()
