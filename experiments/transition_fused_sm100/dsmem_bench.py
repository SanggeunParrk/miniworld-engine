import torch, drv
k = drv.Kernel("build/dsmem_bench.cubin", "rs8", 2 * 65536)
o = torch.zeros(120, dtype=torch.int64, device="cuda")
it = 500
k((120, 1, 1), (128, 1, 1), o, it); torch.cuda.synchronize()
k((120, 1, 1), (128, 1, 1), o, it); torch.cuda.synchronize()
print(f"cluster-8 reduce-scatter of a 128x128 fp32 partial: {o.float().median().item() / it:.0f} clk per tile (120 CTAs)")
