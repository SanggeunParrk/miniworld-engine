import torch, drv
k = {}
for n in ("mma_ss16", "mma_ss32", "mma_ss64", "mma_ss128", "mma_ts128", "mma_ss128_busy"):
    kk = drv.Kernel("build/mma_bench.cubin", n, 65536 + 1024)
    o = torch.zeros(148, dtype=torch.int64, device="cuda")
    it = 2000
    th = 512 if "busy" in n else 128
    kk((148, 1, 1), (th, 1, 1), o, it); torch.cuda.synchronize()
    kk((148, 1, 1), (th, 1, 1), o, it); torch.cuda.synchronize()
    N = int(n[6:].split("_")[0])
    flop = 2 * 128 * N * 128 * it
    clk = o.float().median().item()
    print(f"{n}: {clk/it:.0f} clk per K=128 chain, {flop/clk:.0f} FLOP/clk/SM (peak 8192)")
for n in ("pat0", "pat1", "pat2"):
    kk = drv.Kernel("build/mma_bench.cubin", n, 65536 + 1024)
    o = torch.zeros(148, dtype=torch.int64, device="cuda")
    it = 512
    kk((148, 1, 1), (128, 1, 1), o, it); torch.cuda.synchronize()
    print(f"{n}: issue {o.float().median().item()/it:.0f} clk per chunk (tensor work 768)")
for n, N, nw in (("ind32", 32, 1), ("ind64", 64, 1), ("ind128", 128, 1), ("ind32w2", 32, 2), ("ind64w2", 64, 2)):
    kk = drv.Kernel("build/mma_bench.cubin", n, 65536 + 1024)
    o = torch.zeros(148, dtype=torch.int64, device="cuda")
    it = 1000
    kk((148, 1, 1), (128, 1, 1), o, it); torch.cuda.synchronize()
    clk = o.float().median().item()
    print(f"{n}: {clk/(it*8*nw):.1f} clk per MMA (tensor time {N/2:.0f}), {2*128*N*16*it*8*nw/clk:.0f} FLOP/clk")
