"""Per-CTA phase timestamps (build with GRA_DEFS=GRA_TIMES): where the fused kernel's time goes."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gra  # noqa
from gra import gemm_resgate_adaln  # noqa
dev, bf, D = "cuda", torch.bfloat16, 768
NAMES = ["start", "mainloop", "x ready", "resgate+store", "stats+cluster", "end"]
for L, Kd, sa in ((768, 768, 3072), (768, 1536, 1536), (384, 768, 3072)):
    M = 5 * L
    a = torch.randn(M, sa, device=dev, dtype=bf)[:, :Kd]
    w = (torch.randn(D, Kd, device=dev) * Kd ** -0.5).to(bf)
    x = torch.randn(M, D, device=dev); xa = torch.empty(M, D, device=dev, dtype=bf)
    g = torch.randn(L, 4, D, device=dev, dtype=bf)
    for nwg in (1, 2):
        for _ in range(5):
            gemm_resgate_adaln(a, w, x, g[:, 0], g[:, 1], g[:, 2], xa, L, nwg=nwg)
        torch.cuda.synchronize()
        t = gra._ext().debug_times()
        n = 4 * M // (64 * nwg)
        t = t[:n, 1:7].double()
        t0 = t[:, 0].min()
        rel = (t - t0) / 1e3
        med = rel.median(0).values.tolist()
        mx = rel.max(0).values.tolist()
        print(f"L{L} K{Kd} nwg{nwg}: " + "  ".join(f"{nm} {m:5.1f}/{x_:5.1f}" for nm, m, x_ in zip(NAMES, med, mx)) + "  us (median/max CTA)")
