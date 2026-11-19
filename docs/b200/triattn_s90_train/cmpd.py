import sys, torch
from miniworld_engine.kernels.triangle_attention.cuda import b200_triattn as K
from miniworld_engine.kernels._nvcc import load_extension
L = int(sys.argv[2]); dv = [d for d in sys.argv[1].split(",") if d]; H = 4
def build(defs, r):
    tag = "_".join(d.replace("=", "") for d in defs) or "base"
    return load_extension(name=f"s90cmp_r{r}_{tag}", sources=[str(K._SRC / "triattn_sm100.cu")], extra_cuda_cflags=[*K._flags(), f"-DTA_R={r}", *(f"-D{d}" for d in defs)])
torch.manual_seed(0); sh = (1, L, L, H, 32)
q, k, v, do = (torch.randn(sh, device="cuda", dtype=torch.bfloat16) for _ in range(4))
bias = torch.randn(1, H, L, L, device="cuda", dtype=torch.bfloat16); E = q.new_empty(0)
r = K._rows_per_task(L, H); out = {}
for nm, d in (("base", []), ("var", dv)):
    e = build(d, r); o, lse, _ = e.triattn_fwd(q, k, v, bias, 32 ** -0.5, True, E, E); delta = e.triattn_delta(do, o)
    out[nm] = [x.clone() for x in e.triattn_bwd(q, k, v, bias, do, lse, delta, 32 ** -0.5, E.float(), E, E, E, E.float())[:3]]
for i, nm in ((1, "dk"), (2, "dv")):
    bad = (out["base"][i] != out["var"][i]).float()   # [1, N(row i), S(key j), H, 32]
    print(nm, "frac bad", float(bad.mean()), "by row i (first 12):", [round(float(x), 2) for x in bad.mean((0, 2, 3, 4))[:12]], "by key tile (j//64):", [round(float(bad[:, :, a:a+64].mean()), 2) for a in range(0, L, 64)], "by head:", [round(float(bad[:, :, :, h].mean()), 2) for h in range(H)])
for i, nm in ((1, "dk"),):
    bad = (out["base"][i] != out["var"][i]).float()[:, 0::2]   # even rows
    print(nm, "even rows: by key quadrant (j//32):", [round(float(bad[:, :, a:a+32].mean()), 2) for a in range(0, L, 32)], "by channel group (8):", [round(float(bad[..., a:a+8].mean()), 2) for a in range(0, 32, 8)])
    print("rows with any good element:", int((bad.mean((2,3,4)) < 1).sum()), "of", bad.shape[1])
