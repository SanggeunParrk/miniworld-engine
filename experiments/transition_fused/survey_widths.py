"""Transition(D, n=4) across channel widths: where every existing path stands against the tensor floor.

  python survey_widths.py --width 256 --lengths 384,768 [--no-compile]

Rows (same weights, bf16, x of shape [1, L, L, D] so M = L^2 rows; squeeze drawn non-zero):
  engine_default   modules.Transition as a training run gets it (fused sm_90a where its gate admits, else Triton)
  engine_triton    modules.Transition with the fused sm_90a path off
  torch_eager      the engine's transition_pytorch: F.layer_norm + 3 cuBLAS GEMMs + silu*b + add
  torch_compile    the same under torch.compile
  cublas_cat       LN + ONE cuBLAS GEMM over the packed [Wa; Wb] + silu*b + cuBLAS GEMM + add (the lean unfused reference)
Modes: infer = no_grad forward; train = forward + backward.  CUDA-graph replay median (compile rows: plain event timing).
Floors: forward 24 M D^2 FLOP, training 72 M D^2, at 989 TFLOP/s dense bf16.
"""
import argparse, copy, json, statistics, sys, torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--width", type=int, required=True)
p.add_argument("--lengths", default="384,768")
p.add_argument("--reps", type=int, default=10)
p.add_argument("--no-compile", action="store_true")
p.add_argument("--output", default="")
a = p.parse_args()
from miniworld_engine import settings                                   # noqa: E402
from miniworld_engine.kernels.transition.reference import transition_pytorch  # noqa: E402
from miniworld_engine.modules import Transition                        # noqa: E402

D, N = a.width, 4
PEAK = 989e12
res = {"width": D, "rows": []}


def graph_time(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            fn()
    torch.cuda.synchronize()
    o = []
    for _ in range(5):
        st, en = torch.cuda.Event(True), torch.cuda.Event(True)
        st.record()
        for _ in range(a.reps):
            g.replay()
        en.record(); torch.cuda.synchronize()
        o.append(st.elapsed_time(en) * 1e3 / a.reps)
    return statistics.median(o)


def event_time(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    o = []
    for _ in range(5):
        st, en = torch.cuda.Event(True), torch.cuda.Event(True)
        st.record()
        for _ in range(a.reps):
            fn()
        en.record(); torch.cuda.synchronize()
        o.append(st.elapsed_time(en) * 1e3 / a.reps)
    return statistics.median(o)


for L in [int(v) for v in a.lengths.split(",")]:
    M = L * L
    torch.manual_seed(2319)
    base = Transition(D, n=N, implementation="triton").cuda().bfloat16()
    with torch.no_grad():
        for prm in base.parameters():
            if prm.ndim == 2:
                prm.normal_(std=prm.shape[-1] ** -0.5)
    x = torch.randn(1, L, L, D, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    lw, lb = base.ln_in.weight, base.ln_in.bias
    wa, wb, ws = base.expand_a.weight, base.expand_b.weight, base.squeeze.weight
    wab = torch.cat([wa, wb], 0).detach().clone().requires_grad_(True)
    wsd = ws.detach().clone().requires_grad_(True)
    H = wa.shape[0]

    def cublas_cat(t, wab_=wab, ws_=wsd):
        xn = F.layer_norm(t, (D,), lw.to(t.dtype), lb.to(t.dtype), base.ln_in.eps)
        ab = F.linear(xn, wab_)
        return t + F.linear(F.silu(ab[..., :H]) * ab[..., H:], ws_)

    def engine(fused):
        settings.configure(engine_backend="triton", transition_residual_fusion=True, transition_fused_sm90a=fused)
        return copy.deepcopy(base)

    mods = {"engine_default": (engine(True), True), "engine_triton": (engine(False), False)}
    tp = lambda t: transition_pytorch(t, lw, lb, wa, wb, ws, N, base.ln_in.eps)
    fns = {"torch_eager": tp, "cublas_cat": cublas_cat}
    if not a.no_compile:
        fns["torch_compile"] = torch.compile(tp)
    floors = {"infer": 24 * M * D * D / PEAK * 1e6, "train": 72 * M * D * D / PEAK * 1e6}
    for mode in ("infer", "train"):
        xg = x.clone().requires_grad_(mode == "train")
        cands = []
        for name, (mod, fused) in mods.items():
            def f(mod=mod, fused=fused):
                settings.configure(engine_backend="triton", transition_residual_fusion=True, transition_fused_sm90a=fused)
                if mode == "infer":
                    with torch.no_grad():
                        return mod(x)
                y = mod(xg); y.backward(dy); return y
            cands.append((name, f, "graph"))
        for name, fn in fns.items():
            def f(fn=fn):
                if mode == "infer":
                    with torch.no_grad():
                        return fn(x)
                y = fn(xg); y.backward(dy); return y
            cands.append((name, f, "event" if name == "torch_compile" else "graph"))
        for name, f, how in cands:
            try:
                us = (graph_time if how == "graph" else event_time)(f)
                pct = 100 * floors[mode] / us
                print(f"D{D} L{L} {mode:5s} {name:15s} {us:9.1f} us   floor {floors[mode]:8.1f} us  -> {pct:5.1f} % of floor", flush=True)
                res["rows"].append(dict(L=L, mode=mode, row=name, us=us, floor_us=floors[mode], pct=pct))
            except Exception as e:  # noqa: BLE001
                print(f"D{D} L{L} {mode:5s} {name:15s} FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
            torch.cuda.empty_cache()
    del mods, fns, base, x, dy
    torch.cuda.empty_cache()
if a.output:
    json.dump(res, open(a.output, "w"), indent=1)
