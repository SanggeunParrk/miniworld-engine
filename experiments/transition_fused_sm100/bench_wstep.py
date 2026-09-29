"""D = 384 / 512 training step WITHOUT the fp32 recompute: the forward saves a and b (bf16), the backward's gate is elementwise.

  forward : tln_w -> tswiglu_w (SAVE_AB: h, a, b) -> tgemm_nd EPI_RESID (out = bf16(h Ws^T + x))
  backward: dh = bf16(dy Ws) (cuBLAS) -> tgate_elem (h, dA | dB from dh, a, b) -> dWs = dy^T h, [dWa; dWb] = [dA|dB]^T xn (cuBLAS, fp32)
            -> tgemm_nd EPI_PLAIN (d_xn) -> tlnbwd_w (+ residual)
Accuracy vs the fp32 autograd module (and the recompute contract's own error, for reference); CUDA-graph timing next to torch.compile.

  python bench_wstep.py --dim 384 --lengths 128 384 768 [--rows 384 1920]"""
import argparse, json, sys, torch
import torch.nn.functional as F
import drv
from common import graph_time

p = argparse.ArgumentParser()
p.add_argument("--dim", type=int, default=384)
p.add_argument("--lengths", type=int, nargs="*", default=[128, 384])
p.add_argument("--rows", type=int, nargs="*", default=[])
p.add_argument("--no-check", action="store_true")
p.add_argument("--no-compile", action="store_true")
p.add_argument("--breakdown", action="store_true")
p.add_argument("--fused-gate", action="store_true", help="tgate_ab: dh GEMM + gate in one kernel")
p.add_argument("--keep-h", action="store_true", help="the backward uses the forward's h (kept) instead of re-storing it")
p.add_argument("--sg-item", action="store_true", help="SwiGLU forward on the item schedule (tswiglu_abis_d*, grid = SM count)")
p.add_argument("--gate-item", action="store_true", help="--fused-gate --keep-h gate on the item schedule (tgate_abnhis_d*)")
p.add_argument("--ab-item", type=int, default=0, help="A/B tile / item SwiGLU / item SwiGLU + gate in this process, N alternating rounds")
a = p.parse_args()
D, H = a.dim, 4 * a.dim
torch.backends.cuda.matmul.allow_tf32 = False
nsm = torch.cuda.get_device_properties(0).multi_processor_count
GEMM_SMEM = {256: 197120, 384: 229888, 512: 213504}
LNT = {256: "_l16", 384: "_l16", 512: "_l32"}[D]
kln = drv.Kernel(f"build/tln_d{D}{LNT}.cubin", "transition_ln_w", 0)
ksg = drv.Kernel(f"build/tswiglu_ab_d{D}.cubin", "transition_swiglu_w_sm100", 229888, cluster=2)
kga_is = (drv.Kernel(f"build/tgate_abnhis_d{D}.cubin", "transition_gate_ab_sm100", 3 * 32768 + 2 * 4 * 16384 + 512, cluster=2)
          if (a.gate_item or a.ab_item) and a.fused_gate and a.keep_h else None)
ksg_is = (drv.Kernel(f"build/tswiglu_abis_d{D}.cubin", "transition_swiglu_w_sm100", 229888, cluster=2)
          if a.sg_item or a.ab_item else None)
ksq = drv.Kernel(f"build/tgemm_sq_d{D}.cubin", "transition_gemm_nd_sm100", GEMM_SMEM[D], cluster=2)
kdx = drv.Kernel(f"build/tgemm_dxn_d{D}.cubin", "transition_gemm_nd_sm100", GEMM_SMEM[D], cluster=2)
kel = drv.Kernel("build/tgate_elem.cubin", "transition_gate_elem", 0)
kga = (drv.Kernel(f"build/tgate_abnh_d{D}.cubin", "transition_gate_ab_sm100", 3 * 32768 + 2 * 4 * 16384 + 512, cluster=2) if a.keep_h else
       drv.Kernel(f"build/tgate_ab_d{D}.cubin", "transition_gate_ab_sm100", 229888, cluster=2)) if a.fused_gate else None
klb = drv.Kernel(f"build/tlnbwd_d{D}{LNT}.cubin", "transition_lnbwd_w", 0)
klr = drv.Kernel(f"build/tlnbwd_d{D}{LNT}.cubin", "transition_lnbwd_w_reduce", 0)
print(f"D{D}: swiglu(ab) regs {ksg.regs} lmem {ksg.lmem}", flush=True)


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
    return torch.mm(x_, y_, out_dtype=torch.float32)


def bind(x, wa, wb, ws, gamma, beta, dy, eps=1e-5, item=None, gitem=None):
    item = a.sg_item if item is None else item
    gitem = a.gate_item if gitem is None else gitem
    M = x.shape[0]; tiles = M // 128; tm = drv.TensorMap
    bf = dict(device=x.device, dtype=torch.bfloat16)
    xn, out, dxo = torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)
    h, av, bv, dh = (torch.empty(M, H, **bf) for _ in range(4))
    dab = torch.empty(M, 2 * H, **bf); dxn = torch.empty_like(x)
    rstd = torch.empty(M, device=x.device); c1 = torch.empty_like(rstd)
    wab = torch.cat((wa, wb), 0).contiguous(); wabT = wab.t().contiguous()
    nb = min(M // 16, nsm * 4); lpart = torch.empty(nb, 2 * D, device=x.device); dgb = torch.empty(2 * D, device=x.device)
    g = min(nsm, tiles); g = max(2, g - g % 2)
    sg, gs = (ksg_is, nsm) if item else (ksg, g)
    ga, gg = (kga_is, nsm) if gitem else (kga, g)
    mh = tm(h, [H, M], H * 2, [64, 64])
    smaps = (tm(xn, [D, M], D * 2, [64, 64]), tm(wa, [D, H], D * 2, [64, 128]), tm(wb, [D, H], D * 2, [64, 128]), mh,
             tm(av, [H, M], H * 2, [64, 64]), tm(bv, [H, M], H * 2, [64, 64]))
    qmaps = (mh, tm(ws, [H, D], H * 2, [64, 64]), tm(x, [D, M], D * 2, [64, 64]), tm(out, [D, M], D * 2, [64, 64]))
    dmaps = (tm(dab, [2 * H, M], 4 * H, [64, 64]), tm(wabT, [2 * H, D], 4 * H, [64, 64]), qmaps[2], tm(dxn, [D, M], D * 2, [64, 64]))
    nv = M * H // 8
    wst = ws.t().contiguous()
    gmaps = (tm(dy, [D, M], D * 2, [64, 64]), tm(wst, [D, H], D * 2, [64, 64]), smaps[4], smaps[5], mh, dmaps[0])
    res = {}

    def gate():
        if kga is not None:
            ga((gg, 1, 1), (512, 1, 1), *gmaps, int(tiles), int(not a.keep_h))
        else:
            torch.mm(dy, ws, out=dh)
            kel((min((nv + 255) // 256, nsm * 32), 1, 1), (256, 1, 1), dh, av, bv, h, dab, int(M), int(H))

    def fwd(save=True):
        kln((min(M // 16, nsm * 4), 1, 1), (256, 1, 1), x, gamma, beta, xn, rstd, c1, int(M), float(eps), int(save))
        sg((gs, 1, 1), (512, 1, 1), *smaps, int(save), int(tiles))
        ksq((g, 1, 1), (512, 1, 1), *qmaps, int(tiles))

    def bwd():
        gate()
        res["dws"] = mm32(dy.t(), h)
        dwab = mm32(dab.t(), xn); res["dwa"], res["dwb"] = dwab[:H], dwab[H:]
        kdx((g, 1, 1), (512, 1, 1), *dmaps, int(tiles))
        klb((nb, 1, 1), (256, 1, 1), dxn, x, dy, rstd, c1, gamma, dxo, lpart, int(M))
        klr(((2 * D + 255) // 256, 1, 1), (256, 1, 1), lpart, dgb, int(nb))
        res["dx"], res["dgamma"], res["dbeta"] = dxo, dgb[:D], dgb[D:]

    def step():
        fwd(); bwd()
    step.keep = (smaps, qmaps, dmaps, wab, wabT, gmaps, wst)
    step.fwd, step.bwd, step.res, step.out = fwd, bwd, res, out
    step.parts = dict(gate=gate,
                      dws=lambda: mm32(dy.t(), h), dwab=lambda: mm32(dab.t(), xn),
                      dxn=lambda: kdx((g, 1, 1), (512, 1, 1), *dmaps, int(tiles)),
                      lnbwd=lambda: klb((nb, 1, 1), (256, 1, 1), dxn, x, dy, rstd, c1, gamma, dxo, lpart, int(M)),
                      fwd_infer=lambda: fwd(False),
                      f_ln=lambda: kln((min(M // 16, nsm * 4), 1, 1), (256, 1, 1), x, gamma, beta, xn, rstd, c1, int(M), float(eps), 1),
                      f_swiglu_ab=lambda: sg((gs, 1, 1), (512, 1, 1), *smaps, 1, int(tiles)),
                      f_swiglu=lambda: sg((gs, 1, 1), (512, 1, 1), *smaps, 0, int(tiles)),
                      f_squeeze=lambda: ksq((g, 1, 1), (512, 1, 1), *qmaps, int(tiles)))
    return step


def fp32_ref(x, wa, wb, ws, gamma, beta, dy, eps=1e-5):
    xr = x.float().requires_grad_(True)
    prm = [t.detach().float().clone().requires_grad_(True) for t in (wa, wb, ws, gamma, beta)]
    xnr = F.layer_norm(xr, (D,), prm[3], prm[4], eps)
    y = xr + (F.silu(xnr @ prm[0].t()) * (xnr @ prm[1].t())) @ prm[2].t()
    y.backward(dy.float())
    return y.detach(), dict(dx=xr.grad, dwa=prm[0].grad, dwb=prm[1].grad, dws=prm[2].grad, dgamma=prm[3].grad, dbeta=prm[4].grad)


def eager_bf16(x, wa, wb, ws, gamma, beta, dy, eps=1e-5):
    """PyTorch's own bf16 module (saves a, b in bf16): the error level the engine's other paths live at."""
    xr = x.clone().requires_grad_(True)
    prm = [t.clone().requires_grad_(True) for t in (wa, wb, ws)]
    g16, b16 = gamma.detach().bfloat16().requires_grad_(True), beta.detach().bfloat16().requires_grad_(True)
    xnr = F.layer_norm(xr, (D,), g16, b16, eps)
    y = xr + F.linear(F.silu(F.linear(xnr, prm[0])) * F.linear(xnr, prm[1]), prm[2])
    y.backward(dy)
    return y.detach(), dict(dx=xr.grad, dwa=prm[0].grad, dwb=prm[1].grad, dws=prm[2].grad, dgamma=g16.grad, dbeta=b16.grad)


if a.ab_item:
    import statistics
    for L in a.lengths:
        M = L * L
        x, wa, wb, ws, gamma, beta, dy = inputs(M)
        V = ("tile", "item", "item2") if kga_is is not None else ("tile", "item")
        sts = {v: bind(x, wa, wb, ws, gamma, beta, dy, item=(v != "tile"), gitem=(v == "item2")) for v in V}
        grads = {}
        for v, st in sts.items():
            st(); torch.cuda.synchronize(); grads[v] = {k: t.clone() for k, t in st.res.items()}
        same = all(torch.equal(grads["tile"][k], grads[v][k]) for v in V for k in grads["tile"])
        r = {(v, w): [] for v in sts for w in ("fwd", "step")}
        for _ in range(a.ab_item):
            for v, st in sts.items():
                r[(v, "fwd")].append(graph_time(st.fwd)); r[(v, "step")].append(graph_time(st))
        md = {k: round(statistics.median(t), 1) for k, t in r.items()}
        print(json.dumps({"L": L, **{f"{w}_{v}": md[(v, w)] for w in ("fwd", "step") for v in V}, "same": same}), flush=True)
    sys.exit(0)

for M in list(a.rows) + [L * L for L in a.lengths]:
    x, wa, wb, ws, gamma, beta, dy = inputs(M)
    st = bind(x, wa, wb, ws, gamma, beta, dy)
    st(); torch.cuda.synchronize()
    res = {"M": M}
    if not a.no_check:
        y32, r32 = fp32_ref(x, wa, wb, ws, gamma, beta, dy)
        y16, r16 = eager_bf16(x, wa, wb, ws, gamma, beta, dy)
        res["out"] = f"{rel(st.out, y32):.1e} (eager {rel(y16, y32):.1e})"
        res.update({k: f"{rel(st.res[k], r32[k]):.1e} (eager {rel(r16[k], r32[k]):.1e})" for k in r32})
        g0 = {k: v.clone() for k, v in st.res.items()}; st(); torch.cuda.synchronize()
        res["repro"] = all(torch.equal(g0[k], st.res[k]) for k in g0)
        res["finite"] = all(bool(torch.isfinite(v.float()).all()) for v in st.res.values())
    if M >= 128 * 128:
        res["us_fwd_infer"] = round(graph_time(st.parts["fwd_infer"]), 1)
        res["us_fwd"] = round(graph_time(st.fwd), 1); res["us_bwd"] = round(graph_time(st.bwd), 1); res["us_step"] = round(graph_time(st), 1)
        if a.breakdown:
            for k_, f_ in st.parts.items():
                if k_ != "fwd_infer":
                    res["us_" + k_] = round(graph_time(f_), 1)
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
            res["us_compile_step"] = round(graph_time(cstep), 1)
    print(json.dumps(res), flush=True)
print("(errors vs the fp32 module; in parentheses PyTorch's own bf16 module)")
