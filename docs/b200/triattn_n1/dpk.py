import sys, torch
H, dh = 4, 32
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
    torch.manual_seed(0)
    dos = torch.randn(H, dh * L, L, device="cuda", dtype=torch.bfloat16); c = torch.randn(H, dh * L, L, device="cuda", dtype=torch.bfloat16)
    ref = torch.bmm(dos.transpose(1, 2), c, out_dtype=torch.float32)
    def mk(s):
        def f():
            if s == 1: return torch.bmm(dos.transpose(1, 2), c, out_dtype=torch.float32)
            a = dos.view(H * s, dh * L // s, L).transpose(1, 2); b = c.view(H * s, dh * L // s, L)
            o = torch.bmm(a, b, out_dtype=torch.float32)
            return o.view(H, s, L, L).sum(1)
        return f
    base = tm(mk(1)); out = f"L{L}: s1 {base:.1f}"
    r64 = torch.bmm(dos.transpose(1, 2).double(), c.double())
    for s in (2, 3, 4, 8):
        if (dh * L) % s: continue
        o = mk(s)(); e1 = float((o.double() - r64).abs().max()); e0 = float((ref.double() - r64).abs().max())
        out += f" | s{s} {tm(mk(s)):.1f} (err {e1:.1e} vs s1 {e0:.1e})"
    print(out, flush=True)
