import torch, drv
for n, N in (("mn_ts128", 128), ("mn_ss64", 64), ("mn_ss128", 128)):
    k = drv.Kernel("build/mma_bench.cubin", n, 65536 + 1024)
    o = torch.zeros(148, dtype=torch.int64, device="cuda")
    it = 2000
    k((148, 1, 1), (128, 1, 1), o, it); torch.cuda.synchronize()
    clk = o.float().median().item()
    print(f"{n}: {clk / it:.0f} clk per 8-MMA chain (ideal {N * 4:.0f}), {2 * 128 * N * 128 * it / clk:.0f} FLOP/clk")
