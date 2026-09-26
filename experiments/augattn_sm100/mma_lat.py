import torch, drv
k = drv.Kernel("build/mma_lat.cubin", "mma_lat", 65536 + 1024)
out = torch.zeros(8, dtype=torch.int64, device="cuda")
k((1, 1, 1), (128, 1, 1), out, 50); torch.cuda.synchronize()
names = ["1 MMA", "S: 3 chained", "S then dP (6)", "S/dP interleaved (6)", "dQ TS 4 chained", "6 independent"]
for n, v in zip(names, out.tolist()): print(f"{n:24s} {v:6d} clk")
