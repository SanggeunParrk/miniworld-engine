"""Fine phase stamps of the AdaLN epilogue (GRA_DEFS=GRA_TIMES), thread 0 of each CTA, median over CTAs, us from start."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gra  # noqa
from gra import gemm_resgate_adaln  # noqa
dev, bf, D = "cuda", torch.bfloat16, 768
ORDER = [(1, "start"), (2, "mainloop"), (3, "x ready"), (4, "resgate+xstore"), (7, "m2+stats"), (8, "arrive"),
         (9, "ldg issued"), (5, "cl wait"), (10, "merge"), (11, "xa math"), (13, "store wait"), (6, "end")]
for L, Kd, sa, nwg in ((768, 768, 3072, 2), (768, 1536, 1536, 2), (384, 768, 3072, 1)):
    M = 5 * L
    a = torch.randn(M, sa, device=dev, dtype=bf)[:, :Kd]
    w = (torch.randn(D, Kd, device=dev) * Kd ** -0.5).to(bf)
    x = torch.randn(M, D, device=dev); xa = torch.empty(M, D, device=dev, dtype=bf)
    g = torch.randn(L, 4, D, device=dev, dtype=bf)
    for _ in range(5):
        gemm_resgate_adaln(a, w, x, g[:, 0], g[:, 1], g[:, 2], xa, L, nwg=nwg)
    torch.cuda.synchronize()
    t = gra._ext().debug_times()[: 4 * M // (64 * nwg)].double()
    t0 = t[:, 1].min()
    print(f"L{L} K{Kd} nwg{nwg}: " + "  ".join(f"{nm} {float(((t[:, k] - t0) / 1e3).median()):.1f}" for k, nm in ORDER))
