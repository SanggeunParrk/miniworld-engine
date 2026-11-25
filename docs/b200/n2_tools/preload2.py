# preload2.py K YAML OUT : launch K dummy gattn_fwd split modules once (distinct cubins gz_X*), then run measure.speed YAML --out OUT in this process
import runpy, sys, torch
from foldjet_engine.kernels.swa_dit.cuda import sm100
K = int(sys.argv[1]); yaml, out = sys.argv[2], sys.argv[3]
dev = torch.device("cuda"); S, H, D, C, N = 1024, 4, 32, 128, 1
Qh, Kh, Vh = (torch.randn(N, H, S, D, device=dev, dtype=torch.bfloat16) for _ in range(3))
seq = torch.full((N,), S, device=dev, dtype=torch.int32)
tags = ["XA", "XB", "XC", "XD", "XE", "XF", "XG", "XH", "XI", "XJ"]
keep = []
for t in tags[:K]:
    nm = "gz_" + t
    sm100.CUBINS[nm] = ("gattn_fwd", ("-DSPLITB=1", f"-D{t}=1"))
    g = sm100.GattnFwd(split_cubin=nm); g.split = True; g.cs = 0
    O = torch.empty(N * S, C, device=dev, dtype=torch.bfloat16); L = torch.empty(N, H, S, device=dev, dtype=torch.float32)
    run, _, _ = g.bind(Qh, Kh, Vh, seq, O=O, LSE=L, log2=True); run(); torch.cuda.synchronize()
    keep.append((g, O, L))
sys.argv = ["speed", yaml, "--out", out]
runpy.run_module("foldjet_engine.measure.speed", run_name="__main__")
