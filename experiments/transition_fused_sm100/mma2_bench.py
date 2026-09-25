import torch, drv
for n, N in (("mma2_ss128", 128), ("mma2_ss256", 256), ("mma2_ts128", 128)):
    k = drv.Kernel("build/mma2_bench.cubin", n, 65536 + 1024, cluster=2)
    o = torch.zeros(148, dtype=torch.int64, device="cuda")
    it = 2000
    k((148, 1, 1), (128, 1, 1), o, it); torch.cuda.synchronize()
    clk = o[0::2].float().median().item()
    print(f"{n}: {clk / it:.0f} clk per 8-MMA chain -> {2 * 256 * N * 128 * it / clk / 2:.0f} FLOP/clk per SM (peak 8192)")
