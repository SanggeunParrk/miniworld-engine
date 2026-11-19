import sys, torch
from miniworld_engine.kernels.triangle_attention.cuda import b200_triattn as K
from miniworld_engine.kernels._nvcc import load_extension
from torch.profiler import profile, ProfilerActivity
L = int(sys.argv[1]); H = int(sys.argv[2]) if len(sys.argv) > 2 else 4; B = int(sys.argv[3]) if len(sys.argv) > 3 else 1
r = K._rows_per_task(L, H*B)
e = load_extension(name=f"n1kq_r{r}", sources=[str(K._SRC / "triattn_sm100.cu")], extra_cuda_cflags=[*K._flags(), f"-DTA_R={r}"])
torch.manual_seed(0); sh = (B, L, L, H, 32)
q, k, v, do = (torch.randn(sh, device="cuda", dtype=torch.bfloat16) for _ in range(4))
bias = torch.randn(B, H, L, L, device="cuda", dtype=torch.bfloat16); E = q.new_empty(0)
o, lse, _ = e.triattn_fwd(q, k, v, bias, 32 ** -0.5, True, E, E); delta = e.triattn_delta(do, o)
bT = e.triattn_biasT(bias, L) if hasattr(e, "triattn_biasT") else E.float()
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
for nm, w in (("two", 0), ("kv", 1), ("q", 2), ("one", 3)):
    f = lambda: e.triattn_bwd(q, k, v, bias, do, lse, delta, 32 ** -0.5, E.float(), E, E, E, bT if bT.numel() else E.float(), False, False, w, E)
    for _ in range(3): f()
    torch.cuda.synchronize(); ds = []
    for _ in range(5):
        flush.zero_(); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p: f(); torch.cuda.synchronize()
        ev = sorted([x for x in p.events() if x.device_type.name == "CUDA"], key=lambda x: x.time_range.start)
        ds.append([(x.name[:28], round(x.time_range.end - x.time_range.start, 1)) for x in ev if "bwd" in x.name or "slab" in x.name])
    print(nm, ds[-1], flush=True)
