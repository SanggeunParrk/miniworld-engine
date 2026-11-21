# s90t.py D,n,M,FLAG,FLAG ... : the fused wide forward (inference, save 0) built per argument with the extra -D flags; eager L2-flushed median and
# back-to-back time per launch (timing only; probe flags give wrong outputs).  Flag syntax NAME=VAL (VAL optional: NAME alone = -DNAME).
import sys, statistics
import torch
from miniworld_engine.kernels.transition.cuda import fused_wide_sm100a as fw
from miniworld_engine.kernels.transition.cuda.fused_sm100a import build_cubins
SRC = {64: "widths/tfwd_d64.cu", 128: "tr_fwd_sm100.cu", 256: "widths/tfwd_d256.cu"}
FIRST = {}
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
for i, c in enumerate(sys.argv[1:]):
    f = c.split(","); D, n, M = (int(v) for v in f[:3]); flags = tuple(f"-D{v}" for v in f[3:])
    H = n * D
    torch.manual_seed(0)
    x = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
    k = fw._Launch(x, H)
    cn = f"fwdv{i}"
    p = build_cubins(f"d{D}n{n}", ((cn, SRC[D], (f"-DHID={H}", *flags)),))
    k.ext.load_cubin(f"d{D}n{n}_{cn}", p[cn])
    import os; os.system(f"cuobjdump -res-usage {p[cn]} 2>&1 | grep -i reg | head -3")
    wa, wb = (torch.randn(H, D, device="cuda") * D ** -0.5).bfloat16(), (torch.randn(H, D, device="cuda") * D ** -0.5).bfloat16()
    ws = (torch.randn(D, H, device="cuda") * H ** -0.5).bfloat16()
    g, b = torch.ones(D, device="cuda"), torch.zeros(D, device="cuda")
    out = torch.empty_like(x); rstd = torch.empty(1, device="cuda"); mout = k.rows(out)
    args = (k.rows(x), k.tmap(wa, D, H), k.tmap(wb, D, H), k.tmap(ws, H, D, 64, D // 2), mout, mout, g, b, x, out if "X2=1" in f else rstd, k.tiles, 1e-5, 0)
    blk = 512
    xd = x.double(); mu = xd.mean(-1, keepdim=True); var = xd.var(-1, unbiased=False, keepdim=True)
    xn = (xd - mu) / torch.sqrt(var + 1e-5)
    ref = xd + (torch.nn.functional.silu(xn @ wa.double().t()) * (xn @ wb.double().t())) @ ws.double().t()
    k.run(cn, fw._FUSED_FWD[D], k.grid, args, block=blk); torch.cuda.synchronize()
    err = (out.double() - ref)
    if M not in FIRST: FIRST[M] = out.clone()
    print(f"   max|diff vs first| {(out.float() - FIRST[M].float()).abs().max().item():.3e}", flush=True)
    print(f"   err max/max|ref| {err.abs().max().item() / ref.abs().max().item():.3e} rms {err.pow(2).mean().sqrt().item():.3e}", flush=True)
    ts = []
    for _ in range(40):
        flush.zero_(); a, e = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); k.run(cn, fw._FUSED_FWD[D], k.grid, args, block=blk); e.record(); e.synchronize(); ts.append(a.elapsed_time(e) * 1000)
    a, e = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
    for _ in range(50): k.run(cn, fw._FUSED_FWD[D], k.grid, args, block=blk)
    e.record(); e.synchronize(); am = a.elapsed_time(e) * 1000 / 50
    print(f"{c:44s} median {statistics.median(ts):7.1f} us  back-to-back {am:7.2f}  tiles/CTA {k.tiles / k.grid:.1f}  us/tile {am / (k.tiles / k.grid):.3f}", flush=True)
