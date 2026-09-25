import sys, torch, statistics
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench import timeit
from miniworld_engine.integrations import pwa_train as P
torch.manual_seed(0)
N, H, DZ = 384, 8, 128
z = torch.randn(N, N, DZ, device="cuda", dtype=torch.bfloat16)
w16 = torch.softmax(torch.randn(H, N, N, device="cuda"), -1).to(torch.bfloat16)
dw = torch.randn(H, N, N, device="cuda")
lw = 1 + 0.1 * torch.randn(DZ, device="cuda"); lb = 0.1 * torch.randn(DZ, device="cuda")
wb = (torch.randn(H, DZ, device="cuda") * 0.1).to(torch.bfloat16)
ref = P.pair_bwd(z, w16, dw, lw, lb, 1e-5, wb, BJ=32)
for bj, bjo, nw in [(32, None, 4), (32, 128, 4), (16, 128, 4), (16, 96, 4), (32, 96, 4), (16, 48, 2), (32, 128, 2), (16, 128, 2), (32, 192, 4), (16, 192, 4)]:
    out = P.pair_bwd(z, w16, dw, lw, lb, 1e-5, wb, BJ=bj, BJO=bjo, num_warps=nw)
    err = max((a.float() - b.float()).abs().max().item() / (b.float().abs().max().item() + 1e-30) for a, b in zip(out, ref))
    ts = [timeit(lambda: P.pair_bwd(z, w16, dw, lw, lb, 1e-5, wb, BJ=bj, BJO=bjo, num_warps=nw), rounds=3) for _ in range(5)]
    print(f"BJ={bj} BJO={bjo} warps={nw}: {statistics.median(ts)*1e3:7.1f} us   max rel diff vs default {err:.1e}", flush=True)

