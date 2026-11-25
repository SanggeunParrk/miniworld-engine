# tlkv.py L [DEFS]: clock64 timeline of CTA 1 of triattn_bwd_kv (steady-state stage averages)
import sys, torch
from foldjet_engine.kernels.triangle_attention.cuda import b200_triattn as K
from foldjet_engine.kernels._nvcc import load_extension
L = int(sys.argv[1]); dv = [d for d in (sys.argv[2] if len(sys.argv) > 2 else "").split(",") if d]
H = 4; torch.manual_seed(0)
sh = (1, L, L, H, 32)
q, k, v, do = (torch.randn(sh, device="cuda", dtype=torch.bfloat16) for _ in range(4))
bias = torch.randn(1, H, L, L, device="cuda", dtype=torch.bfloat16)
r = K._rows_per_task(L, H)
tag = "_".join(d.replace("=", "") for d in dv)
e = load_extension(name=f"s90tl_r{r}_{tag}", sources=[str(K._SRC / "triattn_sm100.cu")], extra_cuda_cflags=[*K._flags(), f"-DTA_R={r}", "-DTA_TL=1", *(f"-D{d}" for d in dv)])
E = q.new_empty(0)
o, lse, _ = e.triattn_fwd(q, k, v, bias, 32 ** -0.5, True, E, E)
delta = e.triattn_delta(do, o)
for _ in range(3): e.triattn_bwd(q, k, v, bias, do, lse, delta, 32 ** -0.5, E.float(), E, E, E, E.float())
torch.cuda.synchronize()
t = e.triattn_tl().cpu().tolist()   # last launch = bwd_q overwrites? kv ran first, q has own TL? print kv only if untouched
import statistics as st
a, b = 40, 200
def avg(f): return st.mean(f(y) for y in range(a, b))
print(f"L{L} {tag}: steady stages {a}..{b}")
print(f"  stage period (S^T issuer ev0): {avg(lambda y: t[0][y]-t[0][y-1]):.0f} clk;  grad warp ev2: {avg(lambda y: t[2][y]-t[2][y-1]):.0f}")
print(f"  S^T issuer: waits+issue {avg(lambda y: t[1][y]-t[0][y]):.0f} clk")
print(f"  grad warp: wait for P {avg(lambda y: t[3][y]-t[2][y]):.0f} clk")
for gi in (0, 1):
    e4, e5, e6, e7 = 4 + gi * 6, 5 + gi * 6, 6 + gi * 6, 7 + gi * 6
    ys = range(a + gi - a % 2 + (a % 2) , b, 2)
    ys = [y for y in range(a, b) if y % 2 == gi]
    m = lambda f: st.mean(f(y) for y in ys)
    print(f"  group {gi}: period {m(lambda y: t[e4][y]-t[e4][y-2]):.0f} | wait S {m(lambda y: t[e5][y]-t[e4][y]):.0f} | math+ld/st {m(lambda y: t[e6][y]-t[e5][y]):.0f} | tmem_wait_st+arrive {m(lambda y: t[e7][y]-t[e6][y]):.0f}"
          f" | S issued -> S seen {m(lambda y: t[e5][y]-t[1][y]):.0f} | P arrive -> grad warp saw {m(lambda y: t[3][y]-t[e7][y]):.0f}")
n = 2; m2 = 20
print(f"  task epilogue (group 0, tasks 3..20): last stage pf-arrive -> dfr seen {st.mean(t[20][l]-t[7][l*16+14] for l in range(3,14)):.0f} (stage idx approx) | dfr wait {st.mean(t[21][l]-t[20][l] for l in range(3,14)):.0f} | ld+arrive de {st.mean(t[22][l]-t[21][l] for l in range(3,14)):.0f} | pack+st.shared+sync+TMA {st.mean(t[23][l]-t[22][l] for l in range(3,14)):.0f} | task period {st.mean(t[20][l+1]-t[20][l] for l in range(3,14)):.0f}")
print(f"  S^T issuer probes: q {avg(lambda y: t[16][y]-t[0][y]):.0f} | se {avg(lambda y: t[17][y]-t[16][y]):.0f} | bias {avg(lambda y: t[18][y]-t[17][y]):.0f} | waits(kv..) {avg(lambda y: t[19][y]-t[18][y]):.0f} | issue 4 MMA {avg(lambda y: t[1][y]-t[19][y]):.0f}")
for gi in (0, 1):
    ys = [y for y in range(a, b) if y % 2 == gi]
    e4, e5, e6, e7 = 4 + gi * 6, 5 + gi * 6, 6 + gi * 6, 7 + gi * 6
    m = lambda f: st.mean(f(y) for y in ys)
    print(f"  g{gi}: S(y) issue-ready(ev19) - group start of y-2 (ev5) {m(lambda y: t[19][y]-t[e5][y-2]):.0f} | - end math y-2 (ev6) {m(lambda y: t[19][y]-t[e6][y-2]):.0f} | S(y) MMA issued(ev1) - ev5(y-2) {m(lambda y: t[1][y]-t[e5][y-2]):.0f} | group asks(ev4,y) - ev1(y) {m(lambda y: t[e4][y]-t[1][y]):.0f} | ev4(y) - ev7(y-2) {m(lambda y: t[e4][y]-t[e7][y-2]):.0f} | issuer ev0(y) - ev1(y-1) {m(lambda y: t[0][y]-t[1][y-1]):.0f}")
R = range(3, 14)
print("  epilogue split: ld->bulk_wait_read %.0f | named_sync1 %.0f | pack+STS %.0f | fence+sync2 %.0f" % tuple(st.mean(t[b][l]-t[a_][l] for l in R) for a_, b in ((22, 24), (24, 25), (25, 26), (26, 27))))
for gi in (0, 1):
    ys = [y for y in range(a, b) if y % 2 == gi]
    m = lambda f: st.mean(f(y) for y in ys)
    print(f"  g{gi}: se-arrive - group start(ev5) {m(lambda y: t[28+gi][y]-t[5+6*gi][y]):.0f} | S issue-ready(ev19, y+2) - se-arrive(y) {m(lambda y: t[19][y+2]-t[28+gi][y]):.0f} | S MMA issued(ev1,y+2) - se-arrive(y) {m(lambda y: t[1][y+2]-t[28+gi][y]):.0f}")
