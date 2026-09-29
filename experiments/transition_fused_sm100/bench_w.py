"""Forward check + timing for the width-specific sm_100a Transition forwards (D = 64, ...): n = 4, bf16.

  python bench_w.py --dim 64 --lengths 128 384 768 [--cubin build/tfwd_d64.cubin]

Checks the output (and, with saves, xn / rstd / c1) against the contract emulated in torch (same rounding points: xn bf16, a / b fp32,
h bf16, acc fp32, out = bf16(x + acc)) and against the fp32 module; times inference (no saves) and the training build (saves) by
CUDA-graph replay; alongside, torch.compile of the bf16 module and the engine's Triton residual path when importable."""
import argparse, statistics, torch
import torch.nn.functional as F
import drv
from common import graph_time

p = argparse.ArgumentParser()
p.add_argument("--dim", type=int, default=64)
p.add_argument("--lengths", type=int, nargs="+", default=[128, 384, 768])
p.add_argument("--rows", type=int, nargs="*", default=[], help="extra checks at M rows (whole 128-row tiles; odd tile counts)")
p.add_argument("--cubin", default=None)
p.add_argument("--no-time", action="store_true")
p.add_argument("--baselines", action="store_true")
a = p.parse_args()
D, H = a.dim, 4 * a.dim
cubin = a.cubin or f"build/tfwd_d{D}.cubin"
SMEM = {64: 115712, 256: 231936}                              # SMEM_BYTES of each kernel (static_assert-checked in the .cu)
torch.backends.cuda.matmul.allow_tf32 = False


def inputs(L, seed=2319, dev="cuda", M=None):
    g = torch.Generator().manual_seed(seed)
    M = M or L * L
    x = torch.randn(M, D, generator=g).to(dev, torch.bfloat16)
    wa = (torch.randn(H, D, generator=g) * D ** -0.5).to(dev, torch.bfloat16)
    wb = (torch.randn(H, D, generator=g) * D ** -0.5).to(dev, torch.bfloat16)
    ws = (torch.randn(D, H, generator=g) * H ** -0.5).to(dev, torch.bfloat16)
    gamma = (1 + 0.1 * torch.randn(D, generator=g)).to(dev, torch.float32)
    beta = (0.1 * torch.randn(D, generator=g)).to(dev, torch.float32)
    return x, wa, wb, ws, gamma, beta


def contract(x, wa, wb, ws, gamma, beta, eps=1e-5):
    xf = x.float()
    mean = xf.mean(-1, keepdim=True)
    rs = torch.rsqrt(((xf - mean) ** 2).mean(-1, keepdim=True) + eps)
    xn = ((xf - mean) * rs * gamma + beta).to(torch.bfloat16)
    av, bv = xn.float() @ wa.float().t(), xn.float() @ wb.float().t()
    h = (av * torch.sigmoid(av) * bv).to(torch.bfloat16)
    return (xf + h.float() @ ws.float().t()).to(torch.bfloat16), xn, rs.squeeze(-1), (mean * rs).squeeze(-1)


def fp32(x, wa, wb, ws, gamma, beta, eps=1e-5):
    xf = x.float()
    xn = F.layer_norm(xf, (D,), gamma, beta, eps)
    return xf + (F.silu(xn @ wa.float().t()) * (xn @ wb.float().t())) @ ws.float().t()


def rel(a_, b_):
    return float((a_.float() - b_.float()).norm() / b_.float().norm())


class Fwd:
    def __init__(self):
        self.k = drv.Kernel(cubin, f"transition_fwd_d{D}_sm100", SMEM[D], cluster=2)
        self.nsm = torch.cuda.get_device_properties(0).multi_processor_count

    def bind(self, x, wa, wb, ws, gamma, beta, save, eps=1e-5):
        M = x.shape[0]
        tm = drv.TensorMap
        out = torch.empty_like(x); xn = torch.empty_like(x)
        rstd = torch.empty(M, device=x.device, dtype=torch.float32); c1 = torch.empty_like(rstd)
        maps = (tm(x, [D, M], D * 2, [64, 64]), tm(wa, [D, H], D * 2, [64, 64]), tm(wb, [D, H], D * 2, [64, 64]),
                tm(ws, [H, D], H * 2, [64, D // 2]), tm(out, [D, M], D * 2, [64, 64]), tm(xn, [D, M], D * 2, [64, 64]))
        tiles = M // 128
        g = min(self.nsm, tiles); g = max(2, g - g % 2)

        def run():
            self.k((g, 1, 1), (512, 1, 1), *maps, gamma, beta, rstd, c1, int(tiles), float(eps), int(save))
        run.keep = maps
        return run, out, xn, rstd, c1


f = Fwd()
print(f"D{D}: regs {f.k.regs}, local {f.k.lmem} B")
for L in [-m for m in a.rows] + a.lengths:
    x, wa, wb, ws, gamma, beta = inputs(L, M=-L if L < 0 else None)
    ref, xnr, rsr, c1r = contract(x, wa, wb, ws, gamma, beta)
    r32 = fp32(x, wa, wb, ws, gamma, beta)
    run, out, xn, rstd, c1 = f.bind(x, wa, wb, ws, gamma, beta, save=True)
    run(); torch.cuda.synchronize()
    o1 = out.clone(); run(); torch.cuda.synchronize()
    line = (f"L{L}: out vs contract {rel(out, ref):.2e} (mismatch {(out != ref).float().mean().item() * 100:.2f} %), vs fp32 {rel(out, r32):.2e}"
            f" (contract {rel(ref, r32):.2e}) | xn {rel(xn, xnr):.2e} rstd {rel(rstd, rsr):.2e} c1 {rel(c1, c1r):.2e}"
            f" | repro {torch.equal(o1, out)} finite {bool(torch.isfinite(out.float()).all())}")
    print(line, flush=True)
    if a.no_time or L < 0:
        continue
    ri, *_ = f.bind(x, wa, wb, ws, gamma, beta, save=False)
    msg = f"L{L}: inference {graph_time(ri):7.1f} us, training build {graph_time(run):7.1f} us"
    if a.baselines:
        g16, b16 = gamma.bfloat16(), beta.bfloat16()
        tf = lambda: x + F.linear(F.silu(F.linear(F.layer_norm(x, (D,), g16, b16), wa)) * F.linear(F.layer_norm(x, (D,), g16, b16), wb), ws)
        cf = torch.compile(tf)
        with torch.no_grad():
            cf(); msg += f" | torch.compile {graph_time(cf):7.1f} us"
    print(msg, flush=True)
