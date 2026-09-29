"""Inference (forward, no saves) of the sm_100a Transition at every width and L = 128 .. 768: ours vs torch.compile and eager PyTorch
(bf16 module math), with the error vs the fp32 module. CUDA-graph replay medians, µs.

  python bench_infer_all.py [--dims 64 128 256 384 512] [--lengths 128 256 384 512 640 768] [--save records/infer-all.json]

ours: D64 tfwd_d64, D128 tfwd2 (2-CTA), D256 tfwd_d256 (fused, one kernel each); D384 / D512 tln_w -> tswiglu_w -> tgemm_nd (squeeze +
residual). The SwiGLU schedule is chosen per L (rounds/s1.md, `swiglu_sched`): whole tiles per cluster pair, or (tile pair, 128-unit
chunk) items dealt over all pairs (tswiglu_is_d*, -DITEM_SCHED) up to L = 384."""
import argparse, json, os, sys, torch
import torch.nn.functional as F
import drv
from common import graph_time

p = argparse.ArgumentParser()
p.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256, 384, 512])
p.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 384, 512, 640, 768])
p.add_argument("--save", default=None)
p.add_argument("--no-eager", action="store_true")
p.add_argument("--no-compile", action="store_true")
p.add_argument("--anthropic-root", default=os.environ.get("ANTHROPIC_ROOT", "/NHNHOME/WORKSPACE/26mohw002_A/psk6950/refs/uplifting-biomolecular-modeling"))
p.add_argument("--anthropic-rows", nargs="+", default=["v2", "esm_t16", "pf", "lnl", "af3_fused", "v1"])
a = p.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
nsm = torch.cuda.get_device_properties(0).multi_processor_count
sys.path.insert(0, os.path.join(a.anthropic_root, "common", "opt_core"))
from opt_core.kernels import transition as T
tm = drv.TensorMap
FUSED = {64: ("build/tfwd_d64.cubin", "transition_fwd_d64_sm100", 115712),
         128: ("build/tfwd2.cubin", "transition_fwd2_sm100", 230912),
         256: ("build/tfwd_d256.cubin", "transition_fwd_d256_sm100", 231936)}
GEMM_SMEM = {384: 229888, 512: 213504}
LNT = {384: "_l16", 512: "_l32"}


def swiglu_sched(D, tiles):
    """'item' = (tile pair, 128-unit chunk) items dealt over all cluster pairs (tswiglu_is_d*) up to L = 384 (1152 tiles): the whole
    forward measured 5-15 % faster at L256 / L384 (the last wave of whole tile pairs is partly idle), equal from L512 (rounds/s1.md)."""
    return "item" if tiles <= 1152 else "tile"


def inputs(D, M, seed=2319, dev="cuda"):
    H = 4 * D
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, D, generator=g).to(dev, torch.bfloat16)
    wa = (torch.randn(H, D, generator=g) * D ** -0.5).to(dev, torch.bfloat16)
    wb = (torch.randn(H, D, generator=g) * D ** -0.5).to(dev, torch.bfloat16)
    ws = (torch.randn(D, H, generator=g) * H ** -0.5).to(dev, torch.bfloat16)
    gamma = (1 + 0.1 * torch.randn(D, generator=g)).to(dev, torch.float32)
    beta = (0.1 * torch.randn(D, generator=g)).to(dev, torch.float32)
    return x, wa, wb, ws, gamma, beta


def rel(a_, b_):
    return float((a_.float() - b_.float()).norm() / b_.float().norm())


def bind_ours(D, x, wa, wb, ws, gamma, beta, eps=1e-5):
    H, M = 4 * D, x.shape[0]; tiles = M // 128
    g = min(nsm, tiles); g = max(2, g - g % 2)
    out = torch.empty_like(x)
    rstd = torch.empty(M, device=x.device); c1 = torch.empty_like(rstd)
    if D in FUSED:
        cub, fn, smem = FUSED[D]
        k = drv.Kernel(cub, fn, smem, cluster=2)
        xn = torch.empty_like(x)
        maps = (tm(x, [D, M], D * 2, [64, 64]), tm(wa, [D, H], D * 2, [64, 64]), tm(wb, [D, H], D * 2, [64, 64]),
                tm(ws, [H, D], H * 2, [64, 64 if D == 128 else D // 2]), tm(out, [D, M], D * 2, [64, 64]), tm(xn, [D, M], D * 2, [64, 64]))

        def run():
            k((g, 1, 1), (512, 1, 1), *maps, gamma, beta, rstd, c1, int(tiles), float(eps), 0)
        run.keep = (maps, k)
        return run, out
    kln = drv.Kernel(f"build/tln_d{D}{LNT[D]}.cubin", "transition_ln_w", 0)
    item = swiglu_sched(D, tiles) == "item"
    ksg = drv.Kernel(f"build/tswiglu{'_is' if item else ''}_d{D}.cubin", "transition_swiglu_w_sm100", 5 * 32768 + 32768 + 512, cluster=2)
    gs = nsm if item else g
    ksq = drv.Kernel(f"build/tgemm_sq_d{D}.cubin", "transition_gemm_nd_sm100", GEMM_SMEM[D], cluster=2)
    xn = torch.empty_like(x); h = torch.empty(M, H, device=x.device, dtype=torch.bfloat16)
    smaps = (tm(xn, [D, M], D * 2, [64, 64]), tm(wa, [D, H], D * 2, [64, 128]), tm(wb, [D, H], D * 2, [64, 128]), tm(h, [H, M], H * 2, [64, 64]))
    qmaps = (smaps[3], tm(ws, [H, D], H * 2, [64, 64]), tm(x, [D, M], D * 2, [64, 64]), tm(out, [D, M], D * 2, [64, 64]))

    def run():
        kln((min(M // 16, nsm * 4), 1, 1), (256, 1, 1), x, gamma, beta, xn, rstd, c1, int(M), float(eps), 0)
        ksg((gs, 1, 1), (512, 1, 1), *smaps, int(tiles))
        ksq((g, 1, 1), (512, 1, 1), *qmaps, int(tiles))
    run.keep = (smaps, qmaps, kln, ksg, ksq)
    return run, out


rows = []
for D in a.dims:
    for L in a.lengths:
        M = L * L
        x, wa, wb, ws, gamma, beta = inputs(D, M)
        xf = x.float(); xnf = F.layer_norm(xf, (D,), gamma, beta, 1e-5)
        r32 = xf + (F.silu(xnf @ wa.float().t()) * (xnf @ wb.float().t())) @ ws.float().t()
        run, out = bind_ours(D, x, wa, wb, ws, gamma, beta)
        run(); torch.cuda.synchronize()
        o1 = out.clone(); run(); torch.cuda.synchronize()
        anth = {}
        try:
            W = T.pack(w_o=ws, w_a=wa, w_b=wb, ln_w=gamma, ln_b=beta, eps=1e-5, device=x.device)
            for word in a.anthropic_rows:
                try:
                    with torch.no_grad():
                        fa = lambda: T.transition(x, W, word=word, residual=True, n_tokens=L, timing="graph")[0]
                        ya = fa()
                        anth[word] = (round(graph_time(fa), 1), round(rel(ya, r32), 5))
                except Exception as exc:  # noqa: BLE001 -- a refused row is recorded, not fatal
                    anth[word] = (None, repr(exc)[:60])
        except Exception as exc:  # noqa: BLE001
            anth["pack"] = (None, repr(exc)[:60])
        ok = {w_: v for w_, v in anth.items() if v[0] is not None}
        best = min(ok.items(), key=lambda kv: kv[1][0]) if ok else None
        g16, b16 = gamma.bfloat16(), beta.bfloat16()
        ref = lambda x_: x_ + F.linear(F.silu(F.linear(F.layer_norm(x_, (D,), g16, b16), wa)) * F.linear(F.layer_norm(x_, (D,), g16, b16), wb), ws)
        torch._dynamo.reset()                                     # a fresh static-shape compile per (D, L): no automatic-dynamic fallback
        cf = torch.compile(ref, dynamic=False)
        with torch.no_grad():
            yc = cf(x) if not a.no_compile else ref(x); ye = ref(x)
            row = {"D": D, "L": L, "ours_us": round(graph_time(run), 1),
                   "swiglu_sched": swiglu_sched(D, M // 128) if D not in FUSED else None,
                   "compile_us": None if a.no_compile else round(graph_time(lambda: cf(x)), 1),
                   "anthropic_best": best[0] if best else None, "anthropic_us": best[1][0] if best else None,
                   "err_anthropic": best[1][1] if best else None, "anthropic_rows": anth,
                   "eager_us": None if a.no_eager else round(graph_time(lambda: ref(x)), 1),
                   "err_ours": round(rel(out, r32), 5), "err_compile": round(rel(yc, r32), 5), "err_eager": round(rel(ye, r32), 5),
                   "repro": torch.equal(o1, out), "finite": bool(torch.isfinite(out.float()).all())}
        row["x_compile"] = round(row["compile_us"] / row["ours_us"], 2) if row["compile_us"] else None
        row["x_anthropic"] = round(row["anthropic_us"] / row["ours_us"], 2) if row["anthropic_us"] else None
        rows.append(row)
        print(json.dumps(row), flush=True)
        del run, out, x, xf, xnf, r32
        torch.cuda.empty_cache()
if a.save:
    json.dump({"device": torch.cuda.get_device_name(), "rows": rows}, open(a.save, "w"), indent=1)
