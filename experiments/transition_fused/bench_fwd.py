"""Correctness and timing for the fused Transition forward (D = 128, H = 512, bf16).

Checks the fused forward against an fp32 reference and against the engine's own training forward (LayerNorm, expand-SwiGLU,
squeeze-residual), including the tensors the backward needs (xn, rstd, c1), and times all three paths.

  python bench_fwd.py --length 384 [--engine] [--save records/fwd-L384.json]
"""
from __future__ import annotations

import argparse, json, statistics, sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import drv  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=384)
p.add_argument("--width", type=int, default=128)
p.add_argument("--ncta", type=int, default=132)
p.add_argument("--smem", type=int, default=231424)
p.add_argument("--wbox", type=int, default=64, help="rows per TMA box of the weight maps (the D = 256 kernel loads 32-unit chunks)")
p.add_argument("--cubin", default="")
p.add_argument("--rounds", type=int, default=3)
p.add_argument("--reps", type=int, default=20)
p.add_argument("--eps", type=float, default=1e-5)
p.add_argument("--engine", action="store_true")
p.add_argument("--save", default="")
p.add_argument("--ref-cubin", default="", help="also run this cubin once and report how many output elements differ from it")
p.add_argument("--no-save", action="store_true", help="pass save = 0 (kernels with a trailing save argument; others ignore it)")
p.add_argument("--threads", type=int, default=256, help="block size (a variant with producer warps launches 320)")
a = p.parse_args()
D, H, ROWS, NCTA = a.width, 4 * a.width, 128, a.ncta
M = a.length * a.length
tiles = M // ROWS
dev = "cuda"
torch.manual_seed(2319)
x = torch.randn(M, D, device=dev, dtype=torch.bfloat16)
gamma = (torch.rand(D, device=dev) + 0.5).contiguous()
beta = (torch.randn(D, device=dev) * 0.1).contiguous()
wa = (torch.randn(H, D, device=dev) * D ** -0.5).to(torch.bfloat16).contiguous()
wb = (torch.randn(H, D, device=dev) * D ** -0.5).to(torch.bfloat16).contiguous()
ws = (torch.randn(D, H, device=dev) * H ** -0.5).to(torch.bfloat16).contiguous()
wst = ws.t().contiguous()

cubin = Path(a.cubin) if a.cubin else HERE / "build" / "transition_fwd.cubin"
SMEM = a.smem
k = drv.Kernel(str(cubin), "transition_fwd_fused", SMEM)
print(f"{cubin.name}: regs {k.regs} lmem {k.lmem} smem {SMEM} | grid {NCTA}x{a.threads} | M {M} = {tiles} tiles", flush=True)
tm = lambda t, dims, stride, box: drv.TensorMap(t, dims=dims, stride_bytes=stride, box=box)
out_k = torch.empty_like(x)
maps = (tm(x, [D, M], D * 2, [64, 64]), tm(wa, [D, H], D * 2, [64, a.wbox]),
        tm(wb, [D, H], D * 2, [64, a.wbox]), tm(wst, [D, H], D * 2, [64, a.wbox]), tm(out_k, [D, M], D * 2, [64, 64]))
xn_k = torch.empty_like(x)
rstd_k = torch.empty(M, device=dev, dtype=torch.float32)
c1_k = torch.empty(M, device=dev, dtype=torch.float32)


def fused():
    k((NCTA, 1, 1), (a.threads, 1, 1), *maps, gamma, beta, xn_k, out_k, rstd_k, c1_k, int(M), int(tiles), float(a.eps), int(not a.no_save))
    return out_k, xn_k, rstd_k, c1_k


with torch.no_grad():
    xf = x.float()
    mean_r = xf.mean(-1)
    rstd_r = torch.rsqrt(xf.var(-1, unbiased=False) + a.eps)
    xn_r = (xf - mean_r[:, None]) * rstd_r[:, None] * gamma + beta
    out_r = (F.silu(xn_r @ wa.float().t()) * (xn_r @ wb.float().t())) @ ws.float().t() + xf

NAMES = ("out", "xn", "rstd", "c1")
rec = {"length": a.length, "M": M, "regs": k.regs, "cmp": {}}


def compare(tag, got, want):
    d = {}
    for n, g, w in zip(NAMES, got, want):
        gf, wf = g.float(), w.float()
        e = (gf - wf).abs()
        d[n] = dict(rel_rms=float(e.norm() / wf.norm().clamp_min(1e-20)), max_abs=float(e.max()),
                    finite=bool(torch.isfinite(gf).all()))
        print(f"  {tag:>16s} {n:>5s}: rel_rms {d[n]['rel_rms']:.3e}  max_abs {d[n]['max_abs']:.3e}  finite {d[n]['finite']}", flush=True)
    return d


got = [t.clone() for t in fused()]
torch.cuda.synchronize()
rec["cmp"]["fused_vs_fp32"] = compare("fused vs fp32", got, (out_r, xn_r, rstd_r, mean_r * rstd_r))
if a.ref_cubin:
    kr = drv.Kernel(a.ref_cubin, "transition_fwd_fused", SMEM)
    out_r2 = torch.empty_like(x)
    kr((NCTA, 1, 1), (256, 1, 1), *maps[:4], tm(out_r2, [D, M], D * 2, [64, 64]), gamma, beta, torch.empty_like(x), out_r2,
       torch.empty(M, device=dev, dtype=torch.float32), torch.empty(M, device=dev, dtype=torch.float32), int(M), int(tiles), float(a.eps))
    torch.cuda.synchronize()
    ne = int((got[0] != out_r2).sum()); d = (got[0].float() - out_r2.float())
    rec["vs_ref"] = dict(frac_differ=ne / got[0].numel(), rel_rms=float(d.norm() / out_r2.float().norm()), max_abs=float(d.abs().max()))
    print(f"  vs {Path(a.ref_cubin).name}: out elements differing {100 * ne / got[0].numel():.4f} %  rel_rms {rec['vs_ref']['rel_rms']:.3e}  max_abs {rec['vs_ref']['max_abs']:.3e}", flush=True)
again = [t.clone() for t in fused()]
torch.cuda.synchronize()
rec["reproducible"] = {n: bool(torch.equal(u, v)) for n, u, v in zip(NAMES, got, again)}
print("  bit-reproducible on replay:", rec["reproducible"], flush=True)

engine = None
if a.engine:
    try:
        from miniworld_engine.autotune.shape_key import both_key
        from miniworld_engine.kernels.layernorm.triton.main import _ln_fwd
        from miniworld_engine.kernels.transition.triton.main import _expand_swiglu
        from miniworld_engine.kernels.transition.triton.residual import squeeze_residual
        key = both_key(M)
        gb, bb = gamma.to(torch.bfloat16), beta.to(torch.bfloat16)

        def engine():
            xn, mean, rstd = _ln_fwd(x, gamma, beta, None, a.eps, False, key)
            h = _expand_swiglu(xn, wa, wb, key)
            return squeeze_residual(h, ws, x, key), xn, rstd, mean * rstd
        eng = [t.clone() for t in engine()]
        torch.cuda.synchronize()
        rec["cmp"]["engine_vs_fp32"] = compare("engine vs fp32", eng, (out_r, xn_r, rstd_r, mean_r * rstd_r))
        rec["cmp"]["fused_vs_engine"] = compare("fused vs engine", got, eng)
    except Exception as exc:  # noqa: BLE001
        print(f"engine forward unavailable ({exc!r})", flush=True)
        engine = None


def graph_of(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            fn()
    torch.cuda.synchronize()
    return g


def time_us(g, reps):
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    g.replay(); torch.cuda.synchronize()
    st.record()
    for _ in range(reps):
        g.replay()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) * 1000.0 / reps


graphs = {"fused": graph_of(fused)}
if engine is not None:
    graphs["engine"] = graph_of(engine)
runs = {n: [] for n in graphs}
for r in range(a.rounds):
    for n in (list(graphs) if r % 2 == 0 else list(graphs)[::-1]):
        runs[n].append(time_us(graphs[n], a.reps))
floor = 6.0 * M * D * H / (132 * 4096 * 1.755e9) * 1e6
rec["tensor_floor_us"] = floor
rec["time_us"] = {n: dict(median=statistics.median(v), min=min(v), samples=v) for n, v in runs.items()}
for n, v in rec["time_us"].items():
    print(f"RESULT {n:>8s}: {v['median']:8.1f} us (min {v['min']:.1f}) | tensor floor {floor:.1f} us -> {floor / v['median'] * 100:.1f}%", flush=True)
if a.save:
    out = Path(a.save); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print("saved", out, flush=True)
