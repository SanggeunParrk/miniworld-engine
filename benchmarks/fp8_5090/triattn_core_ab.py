"""Core-only: default gated rf core vs the e4m3 cores (PV; PV + QK) and their quant pass, d=128 4x32, no mask. python triattn_core_ab.py L ..."""
import sys, math, torch
from foldjet_engine.kernels.triangle_attention.cuda import sm80 as core, sm120_fp8 as F
flush = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
def bench(f, n=2):
    f(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n): flush.zero_(); f()
    gf = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gf):
        for _ in range(n): flush.zero_()
    def t(gr):
        gr.replay(); torch.cuda.synchronize(); ts = []
        for _ in range(7):
            e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True); e0.record(); gr.replay(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
        return sorted(ts)[3]
    return (t(g) - t(gf)) / n * 1e3
H = 4
for L in map(int, sys.argv[1:]):
    T = L * L
    y = torch.randn(T, 4 * 128 + 8, device="cuda", dtype=torch.bfloat16)
    q, k, v, g = y[:, :128], y[:, 128:256], y[:, 256:384], y[:, 384:512]
    bias = torch.randn(1, H, L, L, device="cuda", dtype=torch.bfloat16)
    a0 = torch.empty(T, 128, device="cuda", dtype=torch.bfloat16); a1 = torch.empty_like(a0); a2 = torch.empty_like(a0)
    scl = 1.4426950408889634 / math.sqrt(32)
    ext = core._ext(); f8 = F._ext()
    km = torch.empty(0, dtype=torch.uint8, device="cuda"); nul = torch.empty(0, dtype=torch.float32, device="cuda")
    o = torch.empty(0, dtype=torch.bfloat16, device="cuda"); lse = torch.empty(0, dtype=torch.float32, device="cuda")
    n = y.shape[1]
    def d(): ext.attn_fwd_gate_fast(q, k, v, bias, o, lse, g, a0, L, H, 1, n, n, n, 128, n, 128, scl, km)
    vinv = torch.full((H,), 8.0, device="cuda"); vsc = 1 / vinv
    ks = torch.full((H,), 1.0, device="cuda"); qs = 1 / ks
    v8 = torch.empty(L * H * L * 32, dtype=torch.uint8, device="cuda"); k8 = torch.empty_like(v8); e8 = v8.new_empty((0,)); peak = torch.zeros(4, dtype=torch.int32, device="cuda")
    def vq(): f8.quant(v, vinv, v8, v, nul, e8, peak, L, H, L, n, n)
    def kq(): f8.quant(v, vinv, v8, k, ks, k8, peak, L, H, L, n, n)
    def c8(): f8.attn_f8_gate(q, k, v8, vsc, e8, nul, peak, bias, g, a1, L, H, 1, n, n, n, 128, scl, km)
    def cqk(): f8.attn_f8_gate(q, k, v8, vsc, k8, qs, peak, bias, g, a2, L, H, 1, n, n, n, 128, scl, km)
    vq(); kq(); c8(); cqk(); d(); torch.cuda.synchronize()
    nr = a0.float().pow(2).mean().sqrt()
    r1 = (a0.float() - a1.float()).pow(2).mean().sqrt() / nr; r2 = (a0.float() - a2.float()).pow(2).mean().sqrt() / nr
    td, tv, tk, tc, tq = bench(d), bench(vq), bench(kq), bench(c8), bench(cqk)
    print(f"L{L}: default {td:.0f} us | PV: pass {tv:.0f} core {tc:.0f} ratio {td/tc:.3f} with pass {td/(tc+tv):.3f} diff {r1:.2e} | PV+QK: pass {tk:.0f} core {tq:.0f} ratio {td/tq:.3f} with pass {td/(tq+tk):.3f} diff {r2:.2e} peak {peak.tolist()}", flush=True)
