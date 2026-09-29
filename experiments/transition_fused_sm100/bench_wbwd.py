"""Backward check + timing for the width-specific sm_100a Transition (D = 64, ...): forward with saves -> fused backward -> reduction.

  python bench_wbwd.py --dim 64 --lengths 128 384 [--repl 16] [--rows 128 384 1920]

Gradients vs the contract emulated in torch (same rounding points as bench_bwd.py) and vs the fp32 autograd module; replay
bit-reproducibility; CUDA-graph timing of the backward alone and of the training step (forward with saves + backward)."""
import argparse, json, torch
import torch.nn.functional as F
import drv
from common import graph_time

p = argparse.ArgumentParser()
p.add_argument("--dim", type=int, default=64)
p.add_argument("--lengths", type=int, nargs="*", default=[128, 384])
p.add_argument("--rows", type=int, nargs="*", default=[])
p.add_argument("--repl", type=int, nargs="+", default=[16])
p.add_argument("--fcubin", default=None)
p.add_argument("--bcubin", default=None)
p.add_argument("--no-time", action="store_true")
p.add_argument("--no-check", action="store_true")
p.add_argument("--compile", action="store_true", help="also time torch.compile of the module's forward + backward")
a = p.parse_args()
D, H, HS = a.dim, 4 * a.dim, 64
NSL = H // HS
FSMEM, BSMEM = {64: 115712}[D], {64: 164864}[D]
torch.backends.cuda.matmul.allow_tf32 = False
nsm = torch.cuda.get_device_properties(0).multi_processor_count
kf = drv.Kernel(a.fcubin or f"build/tfwd_d{D}.cubin", f"transition_fwd_d{D}_sm100", FSMEM, cluster=2)
bc = a.bcubin or f"build/tbwd_d{D}.cubin"
kb = drv.Kernel(bc, f"transition_bwd_d{D}_sm100", BSMEM, cluster=2)
kr = drv.Kernel(bc, f"transition_bwd_d{D}_reduce", 0)
print(f"D{D}: fwd regs {kf.regs} lmem {kf.lmem} | bwd regs {kb.regs} lmem {kb.lmem}")


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


def bind(x, wa, wb, ws, gamma, beta, dy, repl, eps=1e-5):
    M = x.shape[0]; tiles = M // 128; tm = drv.TensorMap
    out, xn, dx = torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32); c1 = torch.empty_like(rstd)
    ndw = NSL * repl; ndx = nsm - ndw
    assert ndw % 2 == 0 and ndx >= 2 and ndx % 2 == 0
    partab = torch.empty(ndw, 128, D, device=x.device); parts = torch.empty(ndw, D, HS, device=x.device)
    dgbw = torch.empty(ndx * 4, 2 * D, device=x.device)
    g = dict(dx=dx, dwa=torch.empty_like(wa), dwb=torch.empty_like(wb), dws=torch.empty_like(ws),
             dgamma=torch.empty(D, device=x.device), dbeta=torch.empty(D, device=x.device))
    mx, mo, mxn, mdy, mdx = (tm(t, [D, M], D * 2, [64, 64]) for t in (x, out, xn, dy, dx))
    mwa, mwb = tm(wa, [D, H], D * 2, [64, 64]), tm(wb, [D, H], D * 2, [64, 64])
    mws_f, mws_b = tm(ws, [H, D], H * 2, [64, D // 2]), tm(ws, [H, D], H * 2, [64, D])
    gf = min(nsm, tiles); gf = max(2, gf - gf % 2)
    nred = 3 * H * D + 2 * D

    def fwd():
        kf((gf, 1, 1), (512, 1, 1), mx, mwa, mwb, mws_f, mo, mxn, gamma, beta, rstd, c1, int(tiles), float(eps), 1)

    def bwd():
        kb((nsm, 1, 1), (512, 1, 1), mdy, mxn, mx, mws_b, mwa, mwb, mdx, rstd, c1, gamma, partab, parts, dgbw, int(tiles), int(ndw))
        kr(((nred + 255) // 256, 1, 1), (256, 1, 1), partab, parts, dgbw, g["dwa"], g["dwb"], g["dws"], g["dgamma"], g["dbeta"], int(ndw), int(ndx * 4))

    def step():
        fwd(); bwd()
    step.keep = (mx, mo, mxn, mdy, mdx, mwa, mwb, mws_f, mws_b, partab, parts, dgbw)
    step.bwd, step.grads = bwd, g
    return step


def references(x, wa, wb, ws, gamma, beta, dy, eps=1e-5):
    xf, dyf = x.float(), dy.float()
    mean = xf.mean(-1, keepdim=True)
    rs = torch.rsqrt(((xf - mean) ** 2).mean(-1, keepdim=True) + eps)
    xnf = ((xf - mean) * rs * gamma + beta).bfloat16().float()
    dh = (dyf @ ws.float()).bfloat16().float()
    A = xnf @ wa.float().t(); Bv = xnf @ wb.float().t()
    s = torch.sigmoid(A); l = A * s
    h = (l * Bv).bfloat16().float()
    dA = ((dh * Bv) * (s + l * (1 - s))).bfloat16().float()
    dB = (dh * l).bfloat16().float()
    c = dict(dws=(dyf.t() @ h).bfloat16(), dwa=(dA.t() @ xnf).bfloat16(), dwb=(dB.t() @ xnf).bfloat16())
    dxn = (dA @ wa.float() + dB @ wb.float()).bfloat16().float()
    xhat = (xf - mean) * rs
    w = gamma * dxn
    ca = (xhat * w).mean(-1, keepdim=True); cb = w.mean(-1, keepdim=True)
    c["dx"] = (((w - xhat * ca - cb) * rs).bfloat16().float() + dyf).bfloat16()
    c["dgamma"] = (dxn * xhat).sum(0); c["dbeta"] = dxn.sum(0)
    xr = xf.clone().requires_grad_(True)
    prm = [t.float().requires_grad_(True) for t in (wa, wb, ws, gamma, beta)]
    xnr = F.layer_norm(xr, (D,), prm[3], prm[4], eps)
    y = xr + (F.silu(xnr @ prm[0].t()) * (xnr @ prm[1].t())) @ prm[2].t()
    y.backward(dyf)
    return c, dict(dx=xr.grad, dwa=prm[0].grad, dwb=prm[1].grad, dws=prm[2].grad, dgamma=prm[3].grad, dbeta=prm[4].grad)


for M in [r for r in a.rows] + [L * L for L in a.lengths]:
    x, wa, wb, ws, gamma, beta, dy = inputs(M)
    if not a.no_check:
        c, ref = references(x, wa, wb, ws, gamma, beta, dy)
    for R in a.repl:
        st = bind(x, wa, wb, ws, gamma, beta, dy, R)
        st(); torch.cuda.synchronize()
        g = st.grads
        res = {"M": M, "R": R}
        if not a.no_check:
            res.update({k: f"{rel(g[k], c[k]):.1e} / {rel(g[k], ref[k]):.1e} ({rel(c[k], ref[k]):.1e})" for k in ref})
            g0 = {k: v.clone() for k, v in g.items()}; st(); torch.cuda.synchronize()
            res["repro"] = all(torch.equal(g0[k], g[k]) for k in g)
            res["finite"] = all(bool(torch.isfinite(v.float()).all()) for v in g.values())
        if not a.no_time and M >= 128 * 128:
            res["us_bwd"] = round(graph_time(st.bwd), 1); res["us_step"] = round(graph_time(st), 1)
        if a.compile and not a.no_time and M >= 128 * 128 and R == a.repl[0]:
            # torch.compile of the bf16 module, forward + backward (one shape per process: run one length per invocation)
            g16, b16 = gamma.bfloat16(), beta.bfloat16()
            prm = [t.clone().requires_grad_(True) for t in (wa, wb, ws)]
            xl = x.clone().requires_grad_(True)
            f = torch.compile(lambda x_: x_ + F.linear(F.silu(F.linear(F.layer_norm(x_, (D,), g16, b16), prm[0])) * F.linear(F.layer_norm(x_, (D,), g16, b16), prm[1]), prm[2]), dynamic=False)

            def cstep():
                f(xl).backward(dy)
            for _ in range(3):
                cstep()
            for t in [xl] + prm:
                t.grad = None
            res["us_compile_step"] = round(graph_time(cstep), 1)
        print(json.dumps(res), flush=True)
print("(gradient columns: vs contract / vs fp32 (contract vs fp32))")
