# cta_tl.py : per-CTA %globaltimer timeline (entry / prologue done / end) of triattn_bwd_kv, triattn_bwd_q, triattn_db_slabs in the L128 / L256 core backward
import sys, torch
from miniworld_engine.kernels.triangle_attention.cuda import b200_triattn as K
from miniworld_engine.kernels._nvcc import load_extension
H = 4
for L in [int(x) for x in sys.argv[1:]] or [128]:
    torch.manual_seed(0); sh = (1, L, L, H, 32)
    q, k, v, do = (torch.randn(sh, device="cuda", dtype=torch.bfloat16) for _ in range(4))
    bias = torch.randn(1, H, L, L, device="cuda", dtype=torch.bfloat16); E = q.new_empty(0)
    r = K._rows_per_task(L, H)
    e = load_extension(name=f"s90cta_r{r}", sources=[str(K._SRC / "triattn_sm100.cu")], extra_cuda_cflags=[*K._flags(), f"-DTA_R={r}", "-DS90_TL=1", "-DTA_TL=1"])
    o, lse, _ = e.triattn_fwd(q, k, v, bias, 32 ** -0.5, True, E, E); delta = e.triattn_delta(do, o)
    f = lambda: e.triattn_bwd(q, k, v, bias, do, lse, delta, 32 ** -0.5, E.float(), E, E, E, E.float())
    for _ in range(5): f()
    torch.cuda.synchronize()
    for kid, nm in ((0, "bwd_kv"), (1, "bwd_q"), (2, "db_slabs")):
        t = e.s90_tl().tolist()[kid]
        rows = [x for x in t if x[0] > 0]
        if not rows: continue
        t0 = min(x[0] for x in rows)
        ent = sorted((x[0] - t0) / 1e3 for x in rows)
        pro = [(x[1] - x[0]) / 1e3 for x in rows if x[1] > x[0]]
        body = [(x[2] - x[1]) / 1e3 for x in rows if x[2] > x[1] and x[1] > 0]
        end = sorted((x[2] - t0) / 1e3 for x in rows if x[2] > 0)
        print(f"L{L} {nm}: CTAs {len(rows)} | entry spread {ent[0]:.2f}..{ent[-1]:.2f} us | prologue (entry -> tmem alloc + barriers + sync) median {sorted(pro)[len(pro)//2] if pro else 0:.2f} us | body median {sorted(body)[len(body)//2] if body else 0:.2f} us (min {min(body) if body else 0:.2f} max {max(body) if body else 0:.2f}) | CTA end first {end[0]:.2f} median {end[len(end)//2]:.2f} last {end[-1]:.2f} us from the first entry", flush=True)
