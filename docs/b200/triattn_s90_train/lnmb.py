# lnmb.py: bias-only glue kernels (ln_fwd, ln_bwd, gate_u, tri_scale_rows) vs byte floors; every timing with the L2 flushed
import torch, sys
from foldjet_engine.kernels.triangle_attention.cuda import b200_bidir, b200_triattn
g = b200_bidir.glue(); w = b200_triattn._wide()
C = 128
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
def t(fn, n=20):
    ts = []
    for _ in range(n):
        flush.zero_(); torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    ts.sort(); return ts[len(ts) // 2]
for L in (256, 512, 768):
    M = L * L; torch.manual_seed(0)
    x = torch.randn(M, C, device="cuda", dtype=torch.bfloat16); dy = torch.randn_like(x)
    lw = torch.randn(C, device="cuda"); lb = torch.randn(C, device="cuda")
    y, mean, rstd = g.ln_fwd(x, lw, lb, 1e-5, L, False)
    dyln = torch.randn(M, C, device="cuda")
    pb = M * C * 2 / 1e6
    cp = t(lambda: x.clone())
    f = t(lambda: g.ln_fwd(x, lw, lb, 1e-5, L, False))
    b = t(lambda: g.ln_bwd(dyln, x, dy, lw, mean, rstd, L, False))
    print(f"L{L}: pair {pb:.0f} MB | clone (r+w {2*pb:.0f} MB) {cp:.1f} us ({2*pb/cp/1e3:.2f} TB/s) | ln_fwd {f:.1f} us ({2*pb/f/1e3:.2f} TB/s; floor@clone {cp:.1f}) | "
          f"ln_bwd {b:.1f} us ({(pb*2+pb*2+pb)/b/1e3:.2f} TB/s, {5*pb:.0f} MB; floor@clone-rate {5*pb/ (2*pb/cp):.1f})", flush=True)
