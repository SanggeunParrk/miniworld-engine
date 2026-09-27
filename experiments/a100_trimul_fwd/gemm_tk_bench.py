"""Sweep the split-K form of the weight-gradient GEMMs a^T b (K = T = L^2)."""
import torch
T = 768 * 768
ao = torch.randn(T, 128, device="cuda", dtype=torch.bfloat16)
x = torch.randn(256, T, device="cuda", dtype=torch.bfloat16)
z = torch.randn(T, 128, device="cuda", dtype=torch.bfloat16)
def tm(f, n=20):
    for _ in range(3): f()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
    for _ in range(n): f()
    e.record(); e.synchronize(); return s.elapsed_time(e) / n * 1000
def chunk(a, b, S, form):
    if form == "atb":          # [S, M, T/S] x [S, T/S, N]
        return torch.bmm(a.unflatten(0, (S, T // S)).transpose(1, 2), b.unflatten(0, (S, T // S)), out_dtype=torch.float32).sum(0)
    else:                      # (b^T a)^T: [S, N, T/S] x [S, T/S, M]
        return torch.bmm(b.unflatten(0, (S, T // S)).transpose(1, 2), a.unflatten(0, (S, T // S)), out_dtype=torch.float32).sum(0).t()
ref = (ao.float().t() @ x.float().t())
for name, a, b in (("G  128x256", ao, x.t()), ("Gg 128x128", ao, z)):
    print(name, "mm32 %.0f us" % tm(lambda: torch.mm(a.t(), b, out_dtype=torch.float32)))
    for form in ("atb", "bta"):
        for S in (16, 32, 64, 128, 256, 512):
            print(f"  {form} S={S:4d} {tm(lambda: chunk(a, b, S, form)):7.0f} us")
g = chunk(ao, x.t(), 64, "atb"); print("check rel", float((g - ref).norm() / ref.norm()))
