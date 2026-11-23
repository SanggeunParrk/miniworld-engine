"""Core-only: default gated rf core vs e4m3 core (+ vt_quant pass), d=128 4x32, no mask. python triattn_core_ab.py L ..."""
import sys, math, torch
from miniworld_engine.kernels.triangle_attention.cuda import sm80 as core, sm80_wide as W, sm120_fp8 as F
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _bench_util import bench
H = 4
for L in map(int, sys.argv[1:]):
    T = L * L
    y = torch.randn(T, 4 * 128 + 8, device="cuda", dtype=torch.bfloat16)
    q, k, v, g = y[:, :128], y[:, 128:256], y[:, 256:384], y[:, 384:512]
    bias = torch.randn(1, H, L, L, device="cuda", dtype=torch.bfloat16)
    a0 = torch.empty(T, 128, device="cuda", dtype=torch.bfloat16); a1 = torch.empty_like(a0)
    scl = 1.4426950408889634 / math.sqrt(32)
    ext = core._ext(); f8 = F._ext()
    km = torch.empty(0, dtype=torch.uint8, device="cuda")
    o = torch.empty(0, dtype=torch.bfloat16, device="cuda"); lse = torch.empty(0, dtype=torch.float32, device="cuda")
    n = y.shape[1]
    def d(): ext.attn_fwd_gate_fast(q, k, v, bias, o, lse, g, a0, L, H, 1, n, n, n, 128, n, 128, scl, km)
    vinv = torch.full((H,), 8.0, device="cuda"); vsc = 1 / vinv
    v8 = torch.empty(L * H * L * 32, dtype=torch.uint8, device="cuda"); peak = torch.zeros(2, dtype=torch.int32, device="cuda")
    def vq(): f8.vt_quant(v, vinv, v8, peak, L, H, L, n)
    def c8(): f8.attn_f8_gate(q, k, v8, vsc, bias, g, a1, L, H, 1, n, n, n, 128, scl, km)
    vq(); c8(); d(); torch.cuda.synchronize()
    r = (a0.float() - a1.float()).pow(2).mean().sqrt() / a0.float().pow(2).mean().sqrt()
    td, tq, tc = bench(d), bench(vq), bench(c8)
    print(f"L{L}: default core {td:.0f} us | vt_quant {tq:.0f} us | e4m3 core {tc:.0f} us | core ratio {td/tc:.3f} | with pass {td/(tc+tq):.3f} | rel diff {r:.3e}", flush=True)
