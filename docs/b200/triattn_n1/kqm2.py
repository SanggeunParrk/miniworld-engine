import sys, os, torch
from miniworld_engine.kernels.triangle_attention.cuda import b200_triattn as K
from miniworld_engine.kernels._nvcc import load_extension
L = 128; H = 4; which = int(sys.argv[1]); nkv = int(sys.argv[2]) if len(sys.argv) > 2 else 0
r = 2
DEF = [d for d in os.environ.get("N1_DEF", "").split(",") if d]
e = load_extension(name="n1kqm_r2_" + "_".join(DEF), sources=[str(K._SRC / "triattn_sm100.cu")], extra_cuda_cflags=[*K._flags(), "-DTA_R=2", "-DN1_STAMP", *("-D" + d for d in DEF)])
torch.manual_seed(0); sh = (1, L, L, H, 32)
q, k, v, do = (torch.randn(sh, device="cuda", dtype=torch.bfloat16) for _ in range(4))
bias = torch.randn(1, H, L, L, device="cuda", dtype=torch.bfloat16); E = q.new_empty(0)
o, lse, _ = e.triattn_fwd(q, k, v, bias, 32 ** -0.5, True, E, E); delta = e.triattn_delta(do, o)
bT = (bias.float().transpose(-1, -2) * 1.4426950408889634).contiguous()
mk = torch.zeros(2, dtype=torch.int64)
mkd = torch.zeros(2, dtype=torch.int64, device="cuda")
f = lambda: e.triattn_bwd(q, k, v, bias, do, lse, delta, 32 ** -0.5, E.float(), E, E, E, bT, False, False, which, E)
for _ in range(3): f()
def body():
    e.n1_mark(mkd, 0); f(); e.n1_mark(mkd, 1)
body(); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): body()
for rep in range(3):
    torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize()
    m = mkd.cpu().double(); s = e.n1_stamps().view(-1, 8)[:, :5].double(); s = s[s[:, 0] > 0]
    if rep == 2:
        t0 = m[0].item()
        rel = lambda x: (x - t0) / 1000.0
        print(f"which {which}: mark0 -> mark1 {rel(m[1]):.2f} us; first CTA entry {rel(s[:,0].min()) if len(s) else 0:.2f}  last entry {rel(s[:,0].max()):.2f}  prologue-done max {rel(s[:,2].max()):.2f}  last body-end(3) {rel(s[:,3].max()) if len(s) else 0:.2f}")
    if rep == 2 and os.environ.get("W4"):
        w = e.n1_wstamps().view(-1, 32).double()
        for c in (1, 20, 130):
            print(f" CTA{c} pre-barrier/warp:", " ".join(f"{rel(x):.2f}" for x in w[c][:11]))
            print(f" CTA{c} last-instr /warp:", " ".join(f"{rel(x):.2f}" for x in w[c][16:27]), "| body-end(3) thread0", f"{rel(s[c,3]):.2f}")
