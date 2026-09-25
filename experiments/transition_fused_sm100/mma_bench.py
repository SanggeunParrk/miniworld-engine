import torch, drv
k = {}
for n in ("mma_ss64", "mma_ss128", "mma_ts128"):
    kk = drv.Kernel("build/mma_bench.cubin", n, 65536 + 1024)
    o = torch.zeros(148, dtype=torch.int64, device="cuda")
    it = 2000
    kk((148, 1, 1), (128, 1, 1), o, it); torch.cuda.synchronize()
    kk((148, 1, 1), (128, 1, 1), o, it); torch.cuda.synchronize()
    N = int(n[6:])
    flop = 2 * 128 * N * 128 * it
    clk = o.float().median().item()
    print(f"{n}: {clk/it:.0f} clk per K=128 chain, {flop/clk:.0f} FLOP/clk/SM (peak 8192)")
