"""The cuBLAS kernels torch.bmm runs for the TriMul contraction shapes (names from the profiler)."""

import torch
from torch.profiler import ProfilerActivity, profile

for h, L in ((128, 768), (128, 1024), (256, 1024), (128, 2048)):
    a = torch.randn(h, L, L, device="cuda").bfloat16()
    b = torch.randn(h, L, L, device="cuda").bfloat16()
    out = torch.empty_like(a)
    for mode in (0, 1):
        A, B = (a, b.transpose(1, 2)) if mode == 0 else (a.transpose(1, 2), b)
        torch.bmm(A, B, out=out)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            torch.bmm(A, B, out=out)
            torch.cuda.synchronize()
        names = {e.name for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA}
        print(h, L, mode, sorted(names), flush=True)
