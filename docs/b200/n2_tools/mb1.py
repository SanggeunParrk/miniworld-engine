import torch, statistics
dev = torch.device("cuda"); m = 4096; nf = 133; nfs = 136
dp = torch.randn(m, 128, device=dev); F = torch.randn(m, nfs, device=dev); f = F[:, :nf]; fc = torch.randn(m, nf, device=dev)
out = torch.empty(128, nf, device=dev); flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
def bench(fn, name, reps=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps): fn()
    ts = []
    for _ in range(15):
        flush.zero_(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record(); g.replay(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1000 / reps)
    print(f"{name:40s} {statistics.median(ts):7.2f} us/launch", flush=True)
bench(lambda: torch.mm(dp.t(), fc, out=out), "mm unpadded")
bench(lambda: torch.mm(dp.t(), f, out=out), "mm padded view")
for ks in (4, 16, 32):
    part = torch.empty(ks, 128, nf, device=dev)
    bench(lambda: torch.bmm(dp.view(ks, m // ks, 128).transpose(1, 2), fc.view(ks, m // ks, nf), out=part), f"bmm ks{ks} unpadded")
    bench(lambda: torch.bmm(dp.view(ks, m // ks, 128).transpose(1, 2), F.view(ks, m // ks, nfs)[:, :, :nf], out=part), f"bmm ks{ks} padded")
    bench(lambda: torch.sum(part, 0, out=out), f"sum ks{ks}")
# fp32 highest vs tf32
torch.backends.cuda.matmul.allow_tf32 = True
bench(lambda: torch.mm(dp.t(), fc, out=out), "mm unpadded tf32-allowed")
bench(lambda: torch.mm(dp.t(), f, out=out), "mm padded tf32-allowed")
