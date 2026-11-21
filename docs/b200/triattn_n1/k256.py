import sys, os, torch
from pathlib import Path
from miniworld_engine.kernels.triangle_attention.cuda import b200_bidir as BB
from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension
ensure_cuda_home()
old = BB.ext()
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
def tm(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): f()
    g.replay(); torch.cuda.synchronize(); ts = []
    for _ in range(n):
        flush.zero_(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    return sorted(ts)[n // 2]
L = int(sys.argv[1]); K = int(sys.argv[2]); LL = L * L; src = os.environ["EXP_SRC"]
variants = [[], ["ABL_MUFU"], ["ABL_MMA"], ["ABL_ST"], ["ABL_LD"], ["TANH_SIG"], ["HOIST"], ["HOIST", "TANH_SIG"]]
torch.manual_seed(0)
g = torch.randn(K, LL, device="cuda", dtype=torch.bfloat16); o = torch.randn(K, LL, device="cuda", dtype=torch.bfloat16)
wo = torch.randn(128, K, device="cuda", dtype=torch.bfloat16) * .1; x = torch.randn(LL, 128, device="cuda", dtype=torch.bfloat16); dy = torch.randn(LL, 128, device="cuda", dtype=torch.bfloat16)
big = torch.empty(K + 4, LL, device="cuda", dtype=torch.bfloat16); dg0 = big[:K]; dg1 = torch.empty(K, LL, device="cuda", dtype=torch.bfloat16)
out0 = torch.empty_like(x); u0 = torch.empty(K, LL, device="cuda", dtype=torch.bfloat16); out1 = torch.empty_like(x); u1 = torch.empty_like(u0)
old.tail(g, o, wo, x, out0, u0); r0 = old.gate_bwd_fused(dy, wo, g, o, dg0)
print(f"L{L} K{K} shared: tail(train) {tm(lambda: old.tail(g, o, wo, x, out0, u0)):.1f}  tail(eval) {tm(lambda: old.tail(g, o, wo, x, out0, None)):.1f}  gate_bwd {tm(lambda: old.gate_bwd_fused(dy, wo, g, o, dg0)):.1f}", flush=True)
for v in variants:
    e = load_extension(name="n1k256_" + "_".join(v), sources=[src], extra_cuda_cflags=[*host_flags(), *BB._FLAGS, *gencodes("100a"), *("-D" + d for d in v)], verbose=False)
    e.tail_x(g, o, wo, x, out1, u1); r1 = e.gb_x(dy, wo, g, o, dg1)
    eq = (torch.equal(out0, out1) and torch.equal(u0, u1) and torch.equal(r0, r1) and torch.equal(dg0, dg1)) if not any(d.startswith("ABL") or d == "TANH_SIG" for d in v) else None
    print(f"  {'+'.join(v) or 'base-copy':16s} tail(train) {tm(lambda: e.tail_x(g, o, wo, x, out1, u1)):.1f}  tail(eval) {tm(lambda: e.tail_x(g, o, wo, x, out1, None)):.1f}  gate_bwd {tm(lambda: e.gb_x(dy, wo, g, o, dg1)):.1f}  bitwise {eq}", flush=True)
