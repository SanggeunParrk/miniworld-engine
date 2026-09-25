import torch, drv
k = drv.Kernel("build/ldshape.cubin", "ldshape", 0)
o = torch.zeros(32 * 6, dtype=torch.int32, device="cuda")
k((1, 1, 1), (128, 1, 1), o); torch.cuda.synchronize()
o = o.view(32, 6).cpu()
for t in range(32):
    print(t, "16x256b:", [(int(v) // 1000, int(v) % 1000) for v in o[t, :4]], " 16x128b:", [(int(v) // 1000, int(v) % 1000) for v in o[t, 4:]])
