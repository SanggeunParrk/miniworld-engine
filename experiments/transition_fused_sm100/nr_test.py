import torch, drv
k = drv.Kernel("build/ex2_test.cubin", "ex2_test", 0)
d = torch.cat([torch.linspace(1, 2, 2_000_001), torch.logspace(0, 37.5, 2_000_001)]).cuda()
n = d.numel(); outs = [torch.empty_like(d) for _ in range(7)]
k(((n + 255) // 256, 1, 1), (256, 1, 1), d, *outs, n); torch.cuda.synchronize()
ref = 1 / d.double()
for name, o in (("rcp_nr", outs[5]), ("rcp.approx", outs[6])):
    r = ((o.double() - ref).abs() / ref)
    print(f"{name}: max rel err {r.max().item():.3e} mean {r.mean().item():.3e}")
a = torch.linspace(-60, 60, 4_000_001).cuda(); na = a.numel(); oa = [torch.empty_like(a) for _ in range(7)]
k(((na + 255) // 256, 1, 1), (256, 1, 1), a, *oa, na); torch.cuda.synchronize()
refs = torch.sigmoid(a.double())
for name, o in (("sigmoid_nr", oa[4]), ("sigmoid_kit", oa[3])):
    r = ((o.double() - refs).abs() / refs.clamp_min(1e-30))
    print(f"{name}: max rel err {r[refs > 1e-30].max().item():.3e}")
print("h-level bf16 equal fraction nr vs kit:", ((a * oa[4]).bfloat16() == (a * oa[3]).bfloat16()).float().mean().item())
