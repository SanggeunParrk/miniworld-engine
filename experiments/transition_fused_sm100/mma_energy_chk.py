import sys, torch, drv
n = sys.argv[1]
k = drv.Kernel("build/mma_energy.cubin", n, 65536 + 1024)
o = torch.zeros(148, dtype=torch.int64, device="cuda")
k((148, 1, 1), (128, 1, 1), o, 10); torch.cuda.synchronize(); print(n, "ok", o.float().median().item())
