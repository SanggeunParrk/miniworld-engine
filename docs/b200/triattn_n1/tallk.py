import sys, torch
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
def tm(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): f()
    g.replay(); torch.cuda.synchronize(); ts = []
    for _ in range(n):
        flush.zero_(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    return sorted(ts)[n // 2]
for L in [int(x) for x in sys.argv[1:]]:
    LL = L * L; torch.manual_seed(0)
    dt = torch.randn(LL, 128, device="cuda", dtype=torch.bfloat16); u = torch.randn(128, LL, device="cuda", dtype=torch.bfloat16)   # dWo = dt^T u^T: [128 x LL][LL x 128]
    dc = torch.randn(260, LL, device="cuda", dtype=torch.bfloat16); y = torch.randn(LL, 128, device="cuda", dtype=torch.bfloat16)  # dW = dc y: [260 x LL][LL x 128]
    r64 = (dt.double().t() @ u.double().t())
    out = f"L{L} dWo: mm {tm(lambda: torch.mm(dt.t(), u.t(), out_dtype=torch.float32)):.1f}"
    for s in (2, 4, 8, 16):
        def f():
            a = dt.view(s, LL // s, 128).transpose(1, 2)           # [s, 128, K']
            b = u.view(128, s, LL // s).permute(1, 2, 0)           # [s, K', 128]
            return torch.bmm(a, b, out_dtype=torch.float32).sum(0)
        e = float((f().double() - r64).abs().max()); e0 = float((torch.mm(dt.t(), u.t(), out_dtype=torch.float32).double() - r64).abs().max())
        out += f" | s{s} {tm(f):.1f} (err {e:.1e} vs {e0:.1e})"
    r64b = dc.double() @ y.double()
    out += f" || dW: mm {tm(lambda: torch.mm(dc, y, out_dtype=torch.float32)):.1f}"
    for s in (2, 4, 8):
        def g():
            a = dc.view(260, s, LL // s).permute(1, 0, 2)           # [s, 260, K']
            b = y.view(s, LL // s, 128)
            return torch.bmm(a, b, out_dtype=torch.float32).sum(0)
        out += f" | s{s} {tm(g):.1f}"
    print(out, flush=True)
