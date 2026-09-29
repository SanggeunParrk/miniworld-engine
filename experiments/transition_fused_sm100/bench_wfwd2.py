"""The two-kernel Transition forward for D = 384 / 512 on B200: LayerNorm (tln_w.cu) -> expand + SwiGLU (tswiglu_w.cu, writes h) ->
squeeze + residual (cuBLAS addmm: fp32 accumulation, x added, rounded once). Checked against the contract, timed next to
torch.compile.

  python bench_wfwd2.py --dim 384 --lengths 128 384 768 [--rows 128 384 1920]"""
import argparse, json, torch
import torch.nn.functional as F
import drv
from common import graph_time

p = argparse.ArgumentParser()
p.add_argument("--dim", type=int, default=384)
p.add_argument("--lengths", type=int, nargs="*", default=[128, 384])
p.add_argument("--rows", type=int, nargs="*", default=[])
p.add_argument("--no-time", action="store_true")
p.add_argument("--addmm", action="store_true", help="squeeze by cuBLAS addmm (rounds acc to bf16 before adding x)")
a = p.parse_args()
D, H = a.dim, 4 * a.dim
torch.backends.cuda.matmul.allow_tf32 = False
nsm = torch.cuda.get_device_properties(0).multi_processor_count
kln = drv.Kernel(f"build/tln_d{D}.cubin", "transition_ln_w", 0)
ksg = drv.Kernel(f"build/tswiglu_d{D}.cubin", "transition_swiglu_w_sm100", 5 * 32768 + 32768 + 512, cluster=2)
GEMM_SMEM = {256: 197120, 384: 229888, 512: 213504}
kgm = drv.Kernel(f"build/tgemm_sq_d{D}.cubin", "transition_gemm_nd_sm100", GEMM_SMEM[D], cluster=2)
print(f"D{D}: swiglu regs {ksg.regs} lmem {ksg.lmem} | squeeze gemm regs {kgm.regs} lmem {kgm.lmem}", flush=True)


def inputs(M, seed=2319, dev="cuda"):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, D, generator=g).to(dev, torch.bfloat16)
    wa = (torch.randn(H, D, generator=g) * D ** -0.5).to(dev, torch.bfloat16)
    wb = (torch.randn(H, D, generator=g) * D ** -0.5).to(dev, torch.bfloat16)
    ws = (torch.randn(D, H, generator=g) * H ** -0.5).to(dev, torch.bfloat16)
    gamma = (1 + 0.1 * torch.randn(D, generator=g)).to(dev, torch.float32)
    beta = (0.1 * torch.randn(D, generator=g)).to(dev, torch.float32)
    return x, wa, wb, ws, gamma, beta


def rel(a_, b_):
    return float((a_.float() - b_.float()).norm() / b_.float().norm().clamp_min(1e-30))


def contract(x, wa, wb, ws, gamma, beta, eps=1e-5):
    xf = x.float(); mean = xf.mean(-1, keepdim=True)
    rs = torch.rsqrt(((xf - mean) ** 2).mean(-1, keepdim=True) + eps)
    xn = ((xf - mean) * rs * gamma + beta).to(torch.bfloat16)
    av, bv = xn.float() @ wa.float().t(), xn.float() @ wb.float().t()
    h = (av * torch.sigmoid(av) * bv).to(torch.bfloat16)
    return (xf + h.float() @ ws.float().t()).to(torch.bfloat16), xn, h


def bind(x, wa, wb, ws, gamma, beta, save, eps=1e-5):
    M = x.shape[0]; tiles = M // 128; tm = drv.TensorMap
    xn, out = torch.empty_like(x), torch.empty_like(x)
    h = torch.empty(M, H, device=x.device, dtype=torch.bfloat16)
    rstd = torch.empty(M, device=x.device); c1 = torch.empty_like(rstd)
    maps = (tm(xn, [D, M], D * 2, [64, 64]), tm(wa, [D, H], D * 2, [64, 128]), tm(wb, [D, H], D * 2, [64, 128]), tm(h, [H, M], H * 2, [64, 64]))
    gmaps = (maps[3], tm(ws, [H, D], H * 2, [64, 64]), tm(x, [D, M], D * 2, [64, 64]), tm(out, [D, M], D * 2, [64, 64]))
    g = min(nsm, tiles); g = max(2, g - g % 2)
    wst = ws.t()

    def run():
        kln((min(M // 8, nsm * 16), 1, 1), (256, 1, 1), x, gamma, beta, xn, rstd, c1, int(M), float(eps), int(save))
        ksg((g, 1, 1), (512, 1, 1), *maps, int(tiles))
        if a.addmm:
            torch.addmm(x, h, wst, out=out)
        else:
            kgm((g, 1, 1), (512, 1, 1), *gmaps, int(tiles))
    run.keep = maps
    run.gkeep = gmaps
    return run, out, xn, h


for M in list(a.rows) + [L * L for L in a.lengths]:
    x, wa, wb, ws, gamma, beta = inputs(M)
    run, out, xn, h = bind(x, wa, wb, ws, gamma, beta, True)
    run(); torch.cuda.synchronize()
    ref, xnr, hr = contract(x, wa, wb, ws, gamma, beta)
    xf = x.float(); r32 = xf + (F.silu(F.layer_norm(xf, (D,), gamma, beta) @ wa.float().t()) * (F.layer_norm(xf, (D,), gamma, beta) @ wb.float().t())) @ ws.float().t()
    res = {"M": M, "out_vs_contract": f"{rel(out, ref):.1e}", "vs_fp32": f"{rel(out, r32):.1e} ({rel(ref, r32):.1e})",
           "xn": f"{rel(xn, xnr):.1e}", "h": f"{rel(h, hr):.1e}", "finite": bool(torch.isfinite(out.float()).all())}
    o1 = out.clone(); run(); torch.cuda.synchronize(); res["repro"] = torch.equal(o1, out)
    if not a.no_time and M >= 128 * 128:
        res["us_fwd"] = round(graph_time(run), 1)
        res["us_swiglu"] = round(graph_time(lambda: ksg((max(2, min(nsm, M // 128) - min(nsm, M // 128) % 2), 1, 1), (512, 1, 1), *run.keep, int(M // 128))), 1)
        g16, b16 = gamma.bfloat16(), beta.bfloat16()
        cf = torch.compile(lambda x_: x_ + F.linear(F.silu(F.linear(F.layer_norm(x_, (D,), g16, b16), wa)) * F.linear(F.layer_norm(x_, (D,), g16, b16), wb), ws))
        with torch.no_grad():
            cf(x); res["us_compile"] = round(graph_time(lambda: cf(x)), 1)
    print(json.dumps(res), flush=True)
