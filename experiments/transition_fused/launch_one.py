"""Launch ONE wide-Transition kernel a few times on realistic data, for NCU fixed-clock A/B (ncu_one.sh).
   python launch_one.py --kind lnsg|squeeze|gate|dxln|d256fwd --cubin build/X.cubin --width 512 --length 384 [--n 4]"""
import argparse, sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import drv  # noqa: E402
p = argparse.ArgumentParser(); p.add_argument("--kind", required=True); p.add_argument("--cubin", required=True)
p.add_argument("--width", type=int, default=512); p.add_argument("--length", type=int, default=384); p.add_argument("--n", type=int, default=4)
p.add_argument("--grid", type=int, default=132); p.add_argument("--tbk", type=int, default=32); p.add_argument("--hb", type=int, default=128); a = p.parse_args()
D, H, M = a.width, 4 * a.width, a.length ** 2; bf = torch.bfloat16
torch.manual_seed(1)
x = torch.randn(M, D, device="cuda").to(bf); g = (1 + 0.1 * torch.randn(D, device="cuda")).contiguous(); b = (0.1 * torch.randn(D, device="cuda")).contiguous()
wa = (torch.randn(H, D, device="cuda") * D ** -0.5).to(bf); wb = (torch.randn(H, D, device="cuda") * D ** -0.5).to(bf)
ws = (torch.randn(D, H, device="cuda") * H ** -0.5).to(bf).contiguous()
pack = lambda blk: torch.stack([wa.view(H // blk, blk, D), wb.view(H // blk, blk, D)], 1).reshape(2 * H, D).contiguous()
tm = lambda t, dims, stride, box, sw=128: drv.TensorMap(t, dims=dims, stride_bytes=stride, box=box, swizzle=sw)
h = torch.randn(M, H, device="cuda").to(bf); out = torch.empty_like(x)
xn = torch.empty_like(x); rstd = torch.empty(M, device="cuda"); c1 = torch.empty(M, device="cuda")
G = (a.grid, 1, 1); T = (384, 1, 1)
if a.kind == "lnsg":
    k = drv.Kernel(a.cubin, "ln_swiglu_gemm", 231424); w1 = pack(64)
    maps = (tm(x, [D, M], D * 2, [64, 64]), tm(w1, [D, 2 * H], D * 2, [64, 128]), tm(h, [H, M], H * 2, [64, 64]))
    run = lambda: k(G, T, *maps, g, b, xn, rstd, c1, int(M), int(H), 1e-5, 1)
elif a.kind == "squeeze":
    k = drv.Kernel(a.cubin, "squeeze_gemm", 231424); bn = 256 if D % 256 == 0 else 192
    maps = (tm(h, [H, M], H * 2, [64, 64]), tm(ws, [H, D], H * 2, [64, bn]), tm(x, [D, M], D * 2, [64, 64]), tm(out, [D, M], D * 2, [64, 64]))
    run = lambda: k(G, T, *maps, int(M), int(H), int(D))
elif a.kind == "d256fwd":
    k = drv.Kernel(a.cubin, "transition_fwd_fused", 213248); wst = ws.t().contiguous()
    maps = (tm(x, [D, M], D * 2, [64, 64]), tm(wa, [D, H], D * 2, [64, 32]), tm(wb, [D, H], D * 2, [64, 32]), tm(wst, [D, H], D * 2, [64, 32]), tm(out, [D, M], D * 2, [64, 64]))
    run = lambda: k(G, T, *maps, g, b, xn, out, rstd, c1, int(M), int(M // 128), 1e-5, 1)
elif a.kind == "gate":
    k = drv.Kernel(a.cubin, "gate_gemm", 231424); w1 = pack(a.hb); wst = ws.t().contiguous(); T_ = a.tbk
    dy = torch.randn(M, D, device="cuda").to(bf); dab = torch.empty(M, 2 * H, device="cuda", dtype=bf)
    maps = (tm(x, [D, M], D * 2, [T_, 64], 2 * T_), tm(dy, [D, M], D * 2, [T_, 64], 2 * T_), tm(w1, [D, 2 * H], D * 2, [T_, 128], 2 * T_),
            tm(wst, [D, H], D * 2, [T_, a.hb], 2 * T_), tm(h, [H, M], H * 2, [64, 64]), tm(dab, [2 * H, M], 2 * H * 2, [64, 64]))
    run = lambda: k(G, T, *maps, int(M), int(D), int(H), h, dab)
else:
    raise SystemExit("kind")
for _ in range(a.n): run()
torch.cuda.synchronize(); print("ok")
