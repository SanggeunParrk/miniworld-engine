"""Per-phase timeline of gemm_resgate_adaln_ws (GRA_DEFS=PP_TIMES): MMA and epilogue warpgroups side by side."""
import os, sys
from pathlib import Path
import torch
os.environ["GRA_DEFS"] = "PP_TIMES"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "token_dit_fused"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gra  # noqa
fn, ext = gra.fused("ws")
dev, bf, D = "cuda", torch.bfloat16, 768
PH = [("turn+yfree", 0, 1), ("mainloop", 1, 2), ("(tail)", 2, 3), ("y write", 3, 4),
      ("yfull wait", 4, 5), ("residual", 5, 6), ("stats wait", 6, 7), ("adaln", 7, 8)]
for L, Kd, nm in ((768, 768, "Wo"), (768, 1536, "squeeze"), (384, 768, "Wo")):
    M = 5 * L
    a = torch.randn(M, Kd, device=dev, dtype=bf); w = (torch.randn(D, Kd, device=dev) * Kd ** -0.5).to(bf)
    x = torch.randn(M, D, device=dev); xa = torch.empty(M, D, device=dev, dtype=bf)
    g = torch.randn(L, 4, D, device=dev, dtype=bf); gl, ms, mb = g[:, 0], g[:, 1], g[:, 2]
    for _ in range(3): fn(a, w, x, gl, ms, mb, xa, L)
    torch.cuda.synchronize()
    t = ext().debug_times()
    ncta = int((t[:, 15, 0] > 0).sum()); t0 = int(t[:ncta, 15, 0].min())
    nt = (M // 64 + ncta // 4 - 1) // (ncta // 4)
    ends = t[:ncta, :nt, 8]; span = (int(ends[ends > 0].max()) - t0) / 1e3
    print(f"L{L} {nm}: {ncta} CTAs, <= {nt} tiles a CTA, span {span:.1f} us", flush=True)
    for tile in range(nt):
        row = t[:ncta, tile].double(); ok = row[:, 8] > row[:, 0]
        if int(ok.sum()) == 0: break
        r = row[ok]
        st = float(((r[:, 1] - t0) / 1e3).median())
        print(f"  tile {tile} ({int(ok.sum())}) ML starts +{st:5.1f}: " +
              "  ".join(f"{n} {float(((r[:, b] - r[:, a_]) / 1e3).median()):4.1f}" for n, a_, b in PH), flush=True)
