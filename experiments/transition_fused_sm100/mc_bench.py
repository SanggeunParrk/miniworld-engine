import torch, drv
rows = 65536
t = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
m = drv.TensorMap(t, [128, rows], 256, [64, 64])
for n in ("uni", "mc"):
    k = drv.Kernel("build/mc_bench.cubin", n, 6 * 32768 + 1024)
    o = torch.zeros(148, dtype=torch.int64, device="cuda")
    k((148, 1, 1), (32, 1, 1), m, o, 20, rows); torch.cuda.synchronize()
    clk = o.float().median().item()
    print(f"{n}: burst of 6 x 32 KB received per SM in {clk:.0f} clk -> {6*32768/clk:.1f} B/clk/SM received")
