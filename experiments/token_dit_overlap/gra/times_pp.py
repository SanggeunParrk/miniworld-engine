"""Per-phase timeline of gemm_resgate_adaln_pp (build with GRA_DEFS=PP_TIMES): where a tile's time goes."""
import os, sys, statistics
from pathlib import Path
import torch
os.environ["GRA_DEFS"] = "PP_TIMES"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "token_dit_fused"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gra  # noqa
from gra import gemm_resgate_adaln_pp  # noqa

dev, bf, D = "cuda", torch.bfloat16, 768
NAMES = ["turn wait", "mainloop", "acc drain", "x wait", "residual", "stats", "adaln", "free"]
for L, Kd, nm in ((768, 768, "Wo"), (768, 1536, "squeeze"), (384, 768, "Wo")):
    M = 5 * L
    a = torch.randn(M, Kd, device=dev, dtype=bf); w = (torch.randn(D, Kd, device=dev) * Kd ** -0.5).to(bf)
    x = torch.randn(M, D, device=dev); xa = torch.empty(M, D, device=dev, dtype=bf)
    g = torch.randn(L, 4, D, device=dev, dtype=bf); gl, ms, mb = g[:, 0], g[:, 1], g[:, 2]
    for _ in range(3): gemm_resgate_adaln_pp(a, w, x, gl, ms, mb, xa, L)
    torch.cuda.synchronize()
    t = gra._ext_pp().debug_times()
    ncta = int((t[:, 15, 0] > 0).sum())
    t0 = int(t[:ncta, 15, 0].min())
    ends = t[:ncta, :15, 8]; span = (int(ends[ends > 0].max()) - t0) / 1e3
    print(f"L{L} {nm}: {ncta} CTAs, kernel span {span:.1f} us (first CTA start -> last tile freed)", flush=True)
    for tile in range(4):
        row = t[:ncta, tile]
        ok = row[:, 8] > 0
        if int(ok.sum()) == 0: break
        r = row[ok].double()
        d = [(r[:, k + 1] - r[:, k]) / 1e3 for k in range(8)]
        start = (r[:, 0] - t0) / 1e3
        print(f"  tile {tile} ({int(ok.sum())} CTAs) starts +{float(start.median()):5.1f} us: " +
              "  ".join(f"{n} {float(v.median()):4.1f}" for n, v in zip(NAMES, d)), flush=True)
