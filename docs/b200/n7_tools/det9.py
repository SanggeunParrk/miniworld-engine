# det9.py: forward race hunt + bit comparison of the default (CMBS 2) build against the CMBS=0 build; R launches per shape (default 200), full and ragged seqused
import os, sys, torch
from miniworld_engine.kernels.swa_dit.cuda import sm100
R = int(os.environ.get("R", "200")); H, D, C = 4, 32, 128; dev = torch.device("cuda")
sm100.CUBINS["o_w"] = ("gattn_fwd", ("-DSPLITB=0", "-DCMBS=0")); sm100.CUBINS["o_s"] = ("gattn_fwd", ("-DSPLITB=1", "-DCMBS=0")); sm100.CUBINS["o_c"] = ("gattn_fwd", ("-DSPLITB=0", "-DCSPLIT=1", "-DCMBS=0"))
sm100.CUBINS["n_w"] = ("gattn_fwd", ("-DSPLITB=0",)); sm100.CUBINS["n_s"] = ("gattn_fwd", ("-DSPLITB=1",)); sm100.CUBINS["n_c"] = ("gattn_fwd", ("-DSPLITB=0", "-DCSPLIT=1"))
go = sm100.GattnFwd("o_w", split_cubin="o_s", csplit_cubin="o_c"); gn = sm100.GattnFwd("n_w", split_cubin="n_s", csplit_cubin="n_c")
bad = 0
for S, N, ragged in [(6144, 1, 0), (5120, 1, 0), (7168, 1, 1), (4096, 5, 0), (2048, 5, 1), (6144, 5, 1), (3072, 3, 1), (8192, 1, 0)]:
    torch.manual_seed(S + N)
    x = torch.randn(N, H, S, D, device=dev); x = (x / x.norm(dim=-1, keepdim=True) * D ** 0.5).bfloat16()
    Kh = x.roll(1, 2).contiguous(); Vh = torch.randn(N, H, S, D, device=dev).bfloat16()
    seq = torch.tensor([S - (97 * (i + 1) if ragged else 0) for i in range(N)], device=dev, dtype=torch.int32)
    res = []
    for g in (go, gn):
        O = torch.empty(N * S, C, device=dev, dtype=torch.bfloat16); L = torch.empty(N, H, S, device=dev)
        run, _, _ = g.bind(x, Kh, Vh, seq, O=O, LSE=L, log2=True); res.append((run, O, L))
    ref = None; nb = 0
    for i in range(R):
        for run, O, L in res[1:]:
            L.fill_(7.0); O.zero_(); run()
        torch.cuda.synchronize()
        cur = (res[1][1].clone(), res[1][2].clone())
        if ref is None:
            res[0][0](); torch.cuda.synchronize(); ref = (res[0][1].clone(), res[0][2].clone())
            if not (torch.equal(cur[0], ref[0]) and torch.equal(cur[1], ref[1])): nb += 1; print("DIFF vs CMBS=0")
        elif not (torch.equal(cur[0], ref[0]) and torch.equal(cur[1], ref[1])): nb += 1
    print(f"S {S} N {N} ragged {ragged} split {gn.splits(N, S)}: {R} launches, mismatches {nb}"); bad += nb
print("TOTAL BAD", bad)
