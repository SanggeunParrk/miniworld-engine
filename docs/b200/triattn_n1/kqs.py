import sys, torch
from foldjet_engine.kernels.triangle_attention.cuda import b200_triattn as K
from foldjet_engine.kernels._nvcc import load_extension
L = int(sys.argv[1]); H = 4; which = int(sys.argv[2]); nkv = int(sys.argv[3]) if len(sys.argv) > 3 else 0
r = K._rows_per_task(L, H)
e = load_extension(name=f"n1kqs_r{r}", sources=[str(K._SRC / "triattn_sm100.cu")], extra_cuda_cflags=[*K._flags(), f"-DTA_R={r}", "-DN1_STAMP"])
torch.manual_seed(0); sh = (1, L, L, H, 32)
q, k, v, do = (torch.randn(sh, device="cuda", dtype=torch.bfloat16) for _ in range(4))
bias = torch.randn(1, H, L, L, device="cuda", dtype=torch.bfloat16); E = q.new_empty(0)
o, lse, _ = e.triattn_fwd(q, k, v, bias, 32 ** -0.5, True, E, E); delta = e.triattn_delta(do, o)
bT = (bias.float().transpose(-1, -2) * 1.4426950408889634).contiguous()
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
f = lambda: e.triattn_bwd(q, k, v, bias, do, lse, delta, 32 ** -0.5, E.float(), E, E, E, bT, False, False, which, E)
for _ in range(3): f()
for rep in range(3):
    flush.zero_(); torch.cuda.synchronize(); f(); torch.cuda.synchronize()
    s = e.n1_stamps().view(-1, 4).double()
    ok = s[:, 0] > 0
    s = s[ok]; t0 = s[:, 0].min()
    s = (s - t0) / 1000.0
    n = s.shape[0]
    if rep == 2:
        print(f"which {which}: CTAs {n}; entry min/max {s[:,0].min():.2f}/{s[:,0].max():.2f}  prologue-done min/med/max {s[:,2].min():.2f}/{s[:,2].median():.2f}/{s[:,2].max():.2f}  end min/med/max {s[:,3].min():.2f}/{s[:,3].median():.2f}/{s[:,3].max():.2f}")
        if which == 3 and nkv:
            for nm, sl in (("kv", slice(0, nkv)), ("q", slice(nkv, n))):
                x = s[sl]; print(f"  {nm}: {x.shape[0]} CTAs  entry {x[:,0].min():.2f}..{x[:,0].max():.2f}  pro {x[:,2].min():.2f}..{x[:,2].max():.2f}  body(end-pro) med {(x[:,3]-x[:,2]).median():.2f}  end {x[:,3].min():.2f}..{x[:,3].max():.2f}")
        else:
            print("  body med", float((s[:,3]-s[:,2]).median()), "pro-entry med", float((s[:,2]-s[:,0]).median()))
        import numpy as np
        end = s[:, 3].numpy(); print("  end by CTA idx (every 4th):", " ".join(f"{x:.1f}" for x in end[::4]))
