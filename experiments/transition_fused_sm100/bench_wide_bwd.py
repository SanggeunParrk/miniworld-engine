"""The wide-width (D = 256 / 384 / 512) split backward on B200: gate kernel (tgate_w.cu) -> fp32 cuBLAS weight gradients ->
d_xn GEMM -> LayerNorm backward (the engine's Triton kernel) + residual; checked against the contract and fp32 autograd, timed by
CUDA-graph replay next to torch.compile of the module's forward + backward.

  python bench_wide_bwd.py --dim 256 --lengths 128 384 768 [--rows 128 384 1920]
The D = 256 forward is the fused tfwd_d256 kernel (with saves); at D = 384 / 512 xn / rstd / c1 come from torch."""
import argparse, json, torch
import torch.nn.functional as F
import drv
from common import graph_time

p = argparse.ArgumentParser()
p.add_argument("--dim", type=int, default=256)
p.add_argument("--lengths", type=int, nargs="*", default=[128, 384])
p.add_argument("--rows", type=int, nargs="*", default=[])
p.add_argument("--gcubin", default=None)
p.add_argument("--no-time", action="store_true")
p.add_argument("--no-check", action="store_true")
p.add_argument("--no-compile", action="store_true")
p.add_argument("--breakdown", action="store_true")
p.add_argument("--old", action="store_true", help="d_xn by cuBLAS + the Triton LayerNorm backward")
p.add_argument("--dxln", action="store_true", help="D256: d_xn fused with the LayerNorm backward (tdxln_w)")
a = p.parse_args()
D, H, HS = a.dim, 4 * a.dim, 64
torch.backends.cuda.matmul.allow_tf32 = False
nsm = torch.cuda.get_device_properties(0).multi_processor_count
GSMEM = 229888
kg = drv.Kernel(a.gcubin or f"build/tgate_d{D}.cubin", "transition_gate_w_sm100", GSMEM, cluster=2)
kf = drv.Kernel("build/tfwd_d256.cubin", "transition_fwd_d256_sm100", 231936, cluster=2) if D == 256 else None
print(f"D{D}: gate regs {kg.regs} lmem {kg.lmem}", flush=True)
GEMM_SMEM = {256: 197120, 384: 229888, 512: 213504}
kdx = drv.Kernel(f"build/tgemm_dxn_d{D}.cubin", "transition_gemm_nd_sm100", GEMM_SMEM[D], cluster=2)
LNT = {256: "_l16", 384: "_l16", 512: "_l32"}[D]
kln = drv.Kernel(f"build/tlnbwd_d{D}{LNT}.cubin", "transition_lnbwd_w", 0)
klr = drv.Kernel(f"build/tlnbwd_d{D}{LNT}.cubin", "transition_lnbwd_w_reduce", 0)
kdl = drv.Kernel("build/tdxln_d256.cubin", "transition_dxln_w_sm100", 231936, cluster=2) if a.dxln else None
kdr = drv.Kernel("build/tdxln_d256.cubin", "transition_dxln_w_reduce", 0) if a.dxln else None
from miniworld_engine.kernels.transition.triton.fused import _transition_ln_bwd


def inputs(M, seed=2319, dev="cuda"):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, D, generator=g).to(dev, torch.bfloat16)
    wa = (torch.randn(H, D, generator=g) * D ** -0.5).to(dev, torch.bfloat16)
    wb = (torch.randn(H, D, generator=g) * D ** -0.5).to(dev, torch.bfloat16)
    ws = (torch.randn(D, H, generator=g) * H ** -0.5).to(dev, torch.bfloat16)
    gamma = (1 + 0.1 * torch.randn(D, generator=g)).to(dev, torch.float32)
    beta = (0.1 * torch.randn(D, generator=g)).to(dev, torch.float32)
    dy = (torch.randn(M, D, generator=torch.Generator().manual_seed(7)) * 0.1).to(dev, torch.bfloat16)
    return x, wa, wb, ws, gamma, beta, dy


def rel(a_, b_):
    return float((a_.float() - b_.float()).norm() / b_.float().norm().clamp_min(1e-30))


def mm32(x_, y_):
    try:
        return torch.mm(x_, y_, out_dtype=torch.float32)          # cuBLAS with an fp32 output
    except TypeError:
        return torch.mm(x_, y_).float()


def ln_stats(x, gamma, beta, eps=1e-5):
    xf = x.float(); mean = xf.mean(-1, keepdim=True)
    rs = torch.rsqrt(((xf - mean) ** 2).mean(-1, keepdim=True) + eps)
    return ((xf - mean) * rs * gamma + beta).bfloat16(), rs.squeeze(-1).contiguous(), (mean * rs).squeeze(-1).contiguous()


def bind(x, wa, wb, ws, gamma, beta, dy, eps=1e-5):
    M = x.shape[0]; tiles = M // 128; tm = drv.TensorMap
    wst = ws.t().contiguous(); wab = torch.cat((wa, wb), 0).contiguous(); wabT = wab.t().contiguous()
    dxn = torch.empty_like(x); dxo = torch.empty_like(x)
    nb = min(M // 16, nsm * 4); lpart = torch.empty(nb, 2 * D, device=x.device); dgb = torch.empty(2 * D, device=x.device)
    out, xn = torch.empty_like(x), torch.empty_like(x)
    rstd = torch.empty(M, device=x.device); c1 = torch.empty_like(rstd)
    h = torch.empty(M, H, device=x.device, dtype=torch.bfloat16); dab = torch.empty(M, 2 * H, device=x.device, dtype=torch.bfloat16)
    mdy, mxn = tm(dy, [D, M], D * 2, [64, 64]), tm(xn, [D, M], D * 2, [64, 64])
    maps = (mdy, mxn, tm(wst, [D, H], D * 2, [64, 32]), tm(wa, [D, H], D * 2, [64, 64]), tm(wb, [D, H], D * 2, [64, 64]),
            tm(h, [H, M], H * 2, [64, 64]), tm(dab, [2 * H, M], 4 * H, [64, 64]))
    g = min(nsm, tiles); g = max(2, g - g % 2)
    dlpart = torch.empty(nsm * 4, 2 * D, device=x.device)
    dmaps = (maps[6], tm(wabT, [2 * H, D], 4 * H, [64, 64]), mdy, tm(dxn, [D, M], D * 2, [64, 64]))
    if kf is not None:
        fmaps = (tm(x, [D, M], D * 2, [64, 64]), tm(wa, [D, H], D * 2, [64, 64]), tm(wb, [D, H], D * 2, [64, 64]),
                 tm(ws, [H, D], H * 2, [64, D // 2]), tm(out, [D, M], D * 2, [64, 64]), mxn)

        def fwd():
            kf((g, 1, 1), (512, 1, 1), *fmaps, gamma, beta, rstd, c1, int(tiles), float(eps), 1)
    else:
        xn0, rs0, c10 = ln_stats(x, gamma, beta, eps)

        def fwd():
            xn.copy_(xn0); rstd.copy_(rs0); c1.copy_(c10)
    res = {}
    lmaps = (dmaps[0], dmaps[1], tm(x, [D, M], D * 2, [64, 64]), mdy, tm(dxo, [D, M], D * 2, [64, 64]))

    def gate():
        kg((g, 1, 1), (512, 1, 1), *maps, int(tiles), 1)

    def bwd():
        gate()
        res["dws"] = mm32(dy.t(), h)
        dwab = mm32(dab.t(), xn)
        res["dwa"], res["dwb"] = dwab[:H], dwab[H:]
        if a.old:
            dxo_ = torch.mm(dab, wab)
            dx, res["dgamma"], res["dbeta"] = _transition_ln_bwd(dxo_, x, rstd, c1, gamma)
            res["dx"] = dx.add_(dy)
        elif kdl is not None:
            kdl((g, 1, 1), (512, 1, 1), *lmaps, rstd, c1, gamma, dlpart, int(tiles))
            kdr(((2 * D + 255) // 256, 1, 1), (256, 1, 1), dlpart, dgb, int(g * 4))
            res["dx"], res["dgamma"], res["dbeta"] = dxo, dgb[:D], dgb[D:]
        else:
            kdx((g, 1, 1), (512, 1, 1), *dmaps, int(tiles))
            kln((nb, 1, 1), (256, 1, 1), dxn, x, dy, rstd, c1, gamma, dxo, lpart, int(M))
            klr(((2 * D + 255) // 256, 1, 1), (256, 1, 1), lpart, dgb, int(nb))
            res["dx"], res["dgamma"], res["dbeta"] = dxo, dgb[:D], dgb[D:]

    def step():
        fwd(); bwd()
    step.keep = (maps, wst, wab, dmaps, lmaps)
    step.fwd, step.bwd, step.gate, step.res, step.inter = fwd, bwd, gate, res, (h, dab, xn, rstd, c1)
    step.parts = dict(dws=lambda: mm32(dy.t(), h), dwab=lambda: mm32(dab.t(), xn),
                      dxn=lambda: kdx((g, 1, 1), (512, 1, 1), *dmaps, int(tiles)),
                      lnbwd=lambda: kln((nb, 1, 1), (256, 1, 1), dxn, x, dy, rstd, c1, gamma, dxo, lpart, int(M)),
                      dxln=lambda: kdl((g, 1, 1), (512, 1, 1), *lmaps, rstd, c1, gamma, dlpart, int(tiles)) if kdl is not None else None)
    return step


def references(x, wa, wb, ws, gamma, beta, dy, eps=1e-5):
    xf, dyf = x.float(), dy.float()
    xnb, rs, c1 = ln_stats(x, gamma, beta, eps); xnf = xnb.float()
    dh = (dyf @ ws.float()).bfloat16().float()
    A = xnf @ wa.float().t(); Bv = xnf @ wb.float().t()
    s = torch.sigmoid(A); l = A * s
    h = (l * Bv).bfloat16()
    dA = ((dh * Bv) * (s + l * (1 - s))).bfloat16(); dB = (dh * l).bfloat16()
    c = dict(h=h, dA=dA, dB=dB, dws=dyf.t() @ h.float(), dwa=dA.float().t() @ xnf, dwb=dB.float().t() @ xnf)
    dxn = (dA.float() @ wa.float() + dB.float() @ wb.float()).bfloat16().float()
    mean = xf.mean(-1, keepdim=True); xhat = (xf - mean) * rs[:, None]
    w = gamma * dxn; ca = (xhat * w).mean(-1, keepdim=True); cb = w.mean(-1, keepdim=True)
    c["dx"] = (((w - xhat * ca - cb) * rs[:, None]).bfloat16().float() + dyf).bfloat16()
    c["dgamma"] = (dxn * xhat).sum(0); c["dbeta"] = dxn.sum(0)
    xr = xf.clone().requires_grad_(True)
    prm = [t.float().requires_grad_(True) for t in (wa, wb, ws, gamma, beta)]
    xnr = F.layer_norm(xr, (D,), prm[3], prm[4], eps)
    (xr + (F.silu(xnr @ prm[0].t()) * (xnr @ prm[1].t())) @ prm[2].t()).backward(dyf)
    return c, dict(dx=xr.grad, dwa=prm[0].grad, dwb=prm[1].grad, dws=prm[2].grad, dgamma=prm[3].grad, dbeta=prm[4].grad)


for M in list(a.rows) + [L * L for L in a.lengths]:
    x, wa, wb, ws, gamma, beta, dy = inputs(M)
    st = bind(x, wa, wb, ws, gamma, beta, dy)
    st(); torch.cuda.synchronize()
    out = {"M": M}
    if not a.no_check:
        c, ref = references(x, wa, wb, ws, gamma, beta, dy)
        h, dab = st.inter[0], st.inter[1]
        out.update(h=f"{rel(h, c['h']):.1e}", dA=f"{rel(dab[:, :H], c['dA']):.1e}", dB=f"{rel(dab[:, H:], c['dB']):.1e}")
        out.update({k: f"{rel(st.res[k], c[k]):.1e} / {rel(st.res[k], ref[k]):.1e} ({rel(c[k], ref[k]):.1e})" for k in ref})
        g0 = [t.clone() for t in (h, dab)]; st(); torch.cuda.synchronize()
        out["repro"] = all(torch.equal(u, v) for u, v in zip(g0, (h, dab)))
    if not a.no_time and M >= 128 * 128:
        out["us_gate"] = round(graph_time(st.gate), 1); out["us_bwd"] = round(graph_time(st.bwd), 1)
        if a.breakdown:
            for k_, f_ in st.parts.items():
                out["us_" + k_] = round(graph_time(f_), 1)
        if kf is not None:
            out["us_step"] = round(graph_time(st), 1)
        if not a.no_compile:
            g16, b16 = gamma.bfloat16(), beta.bfloat16()
            prm = [t.clone().requires_grad_(True) for t in (wa, wb, ws)]
            xl = x.clone().requires_grad_(True)
            f = torch.compile(lambda x_: x_ + F.linear(F.silu(F.linear(F.layer_norm(x_, (D,), g16, b16), prm[0])) * F.linear(F.layer_norm(x_, (D,), g16, b16), prm[1]), prm[2]))

            def cstep():
                f(xl).backward(dy)
            for _ in range(3):
                cstep()
            for t in [xl] + prm:
                t.grad = None
            with torch.no_grad():
                out["us_compile_fwd"] = round(graph_time(lambda: f(x)), 1)
            out["us_compile_step"] = round(graph_time(cstep), 1)
    print(json.dumps(out), flush=True)
print("(gradients: vs contract / vs fp32 (contract vs fp32)); h / dA / dB vs contract")
