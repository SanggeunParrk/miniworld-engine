import sys, torch
from miniworld_engine.kernels.triangle_attention.cuda import b200_triattn as K
from miniworld_engine.kernels._nvcc import load_extension
L = 128; H = 4; cta = int(sys.argv[1])
r = 2
e = load_extension(name=f"n1tl_r{r}_c{cta}", sources=[str(K._SRC / "triattn_sm100.cu")], extra_cuda_cflags=[*K._flags(), f"-DTA_R={r}", f"-DTA_TL={cta}"])
torch.manual_seed(0); sh = (1, L, L, H, 32)
q, k, v, do = (torch.randn(sh, device="cuda", dtype=torch.bfloat16) for _ in range(4))
bias = torch.randn(1, H, L, L, device="cuda", dtype=torch.bfloat16); E = q.new_empty(0)
o, lse, _ = e.triattn_fwd(q, k, v, bias, 32 ** -0.5, True, E, E); delta = e.triattn_delta(do, o)
bT = (bias.float().transpose(-1, -2) * 1.4426950408889634).contiguous()
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
f = lambda: e.triattn_bwd(q, k, v, bias, do, lse, delta, 32 ** -0.5, E.float(), E, E, E, bT, False, False, 1, E)
for _ in range(3): f()
flush.zero_(); torch.cuda.synchronize(); f(); torch.cuda.synchronize()
t = e.triattn_tl().view(24, 256)
base = t[t > 0].min().item()
names = ["TMA/S^T-warp loop top", "S^T issued", "gradMMA top", "gradMMA got P^T", "g0 s0", "g0 s1", "g0 s2", "g0 s3", "", "", "g1 s0", "g1 s1", "g1 s2", "g1 s3"]
for ev in range(16):
    row = t[ev, :8]
    if (row > 0).any(): print(f"ev{ev:2d}", " ".join(f"{(x.item()-base):6d}" if x.item() > 0 else "     -" for x in row))
