import torch, drv
k = drv.Kernel("build/ex2_test.cubin", "ex2_test", 0)
x = torch.cat([torch.linspace(-126, 126, 4_000_001), torch.linspace(-4, 4, 4_000_001)]).cuda()
n = x.numel(); outs = [torch.empty_like(x) for _ in range(4)]
k(((n + 255) // 256, 1, 1), (256, 1, 1), x, *outs, n); torch.cuda.synchronize()
ref = torch.exp2(x.double())
m = (x > -125) & (x < 125)
for name, o in (("ex2_poly", outs[0]), ("ex2.approx", outs[1])):
    r = ((o.double() - ref).abs() / ref)[m]
    print(f"{name}: max rel err {r.max().item():.3e}, mean {r.mean().item():.3e}")
a = torch.linspace(-60, 60, 4_000_001).cuda(); na = a.numel(); oa = [torch.empty_like(a) for _ in range(4)]
k(((na + 255) // 256, 1, 1), (256, 1, 1), a, *oa, na); torch.cuda.synchronize()
refs = torch.sigmoid(a.double())
for name, o in (("sigmoid_poly", oa[2]), ("sigmoid_kit", oa[3])):
    r = ((o.double() - refs).abs() / refs.clamp_min(1e-30))
    print(f"{name}: max rel err {r[refs > 1e-30].max().item():.3e}")
print("sigmoid_poly vs kit: bitwise equal fraction", (oa[2] == oa[3]).float().mean().item(),
      " h-level bf16 equal fraction", ((a * oa[2]).bfloat16() == (a * oa[3]).bfloat16()).float().mean().item())
