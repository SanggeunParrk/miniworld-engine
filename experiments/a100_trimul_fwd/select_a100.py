"""Exact-shape selection for the A100 TriMul path (the native TriMul's per-shape K1 x K3 table, over this path's knobs).

For each shape, every candidate is run once and compared with the default's outputs: out, and in training dz plus all 10 parameter
gradients. A candidate is kept only if every tensor is finite and within rel-L2 5e-3 of the default's. The kept candidates are then
timed by CUDA-graph replay. The fastest is recorded only if it beats the default by more than --margin. Otherwise the shape keeps its
defaults. There are two stages:
  1. runtime knobs, on the default build;
  2. the compile variants in --extras, run with stage 1's knobs.

  python select_a100.py --mode train infer --variant bidir single --length 384 768 [--extras "-DK1_FRAGLN=1" ...] [--write]
"""
import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trimul_a100 as TA  # noqa: E402
import trimul_train as TT  # noqa: E402
from train_bench_fixture import make  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mode", nargs="+", default=["train", "infer"])
ap.add_argument("--variant", nargs="+", default=["bidir", "single"])
ap.add_argument("--length", nargs="+", type=int, default=[384, 768])
ap.add_argument("--extras", nargs="*", default=[])
ap.add_argument("--margin", type=float, default=0.01)
ap.add_argument("--tol", type=float, default=5e-3)
ap.add_argument("--write", action="store_true")
a = ap.parse_args()


def gtime(fn, reps=7, n=20):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
        torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s): fn()
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(20): g.replay()
    r = []
    for _ in range(reps):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(n): g.replay()
        en.record(); en.synchronize(); r.append(st.elapsed_time(en) / n)
    del g
    return statistics.median(r)


def train_candidates(ch, L):
    cs = (6, 8, 10) if ch == 256 else (4, 6, 8)
    out = []
    for b1g, ov, con in itertools.product((False, True), (True, False), (None, "cublas")):
        base = dict(b1g=b1g, overlap=ov, contract=con)
        out.append(dict(base, b7j=False))
        out += [dict(base, b7j=True, c=c, rings=r) for c in cs for r in (4, 8, 12)]
    return out


def infer_candidates(ch, L):
    return [dict(contract=c) for c in ("auto", "cublas") + (("custom",) if L % 128 == 0 else ())]


def rel(x, r):
    return float((x.float() - r.float()).norm() / (r.float().norm() + 1e-30))


def run_shape(mode, var, L):
    mod, z, mask, ds, dy = make(var, L)
    bidir = var == "bidir"
    ch = 2 * 128 if bidir else 128
    key = TA.selection_key(bidir, L, mode)
    params = [dict(mod.named_parameters())[n] for n in TT.PARAMS]
    if mode == "infer":
        mod = mod.eval()
        pk = TA.pack(mod)
        bufs = {}
        run = lambda ext: [TA.forward(ext, z.detach(), mask, pk, bufs)]  # noqa: E731
    else:
        def run(ext):
            y = TT.forward_train(ext, mod, z, mask, ds)
            return [y.detach()] + list(torch.autograd.grad(y, [z] + params, dy))
    TA.CFG = {}
    ext0 = TA.build()
    with torch.no_grad() if mode == "infer" else torch.enable_grad():
        ref = [t.detach().clone() for t in run(ext0)]

    def measure(ext, kn):
        TA.CFG = dict(kn)
        try:
            with torch.no_grad() if mode == "infer" else torch.enable_grad():
                outs = run(ext)
                errs = [rel(o, r) for o, r in zip(outs, ref)]
                ok = all(torch.isfinite(o).all().item() for o in outs) and max(errs) <= a.tol
                del outs
                ms = gtime(lambda: run(ext)) if ok else None
        except RuntimeError as e:                                  # e.g. a B7J group that does not fit
            ok, errs, ms = False, [float("nan")], None
            print(f"    {kn}: {str(e).splitlines()[0][:100]}", flush=True)
        finally:
            TA.CFG = {}
        return ok, max(errs), ms

    ok, _, t_def = measure(ext0, {})
    rows = [dict(extra="", knobs={}, ms=t_def, err=0.0)]
    print(f"== {key}: default {t_def:.4f} ms", flush=True)
    best = rows[0]
    cands = train_candidates(ch, L) if mode == "train" else infer_candidates(ch, L)
    for kn in cands:
        ok, err, ms = measure(ext0, kn)
        rows.append(dict(extra="", knobs=kn, ms=ms, err=err))
        print(f"   {'  ' if ok else 'x '}{ms if ms else float('nan'):8.4f} ms  err {err:.1e}  {kn}", flush=True)
        if ok and ms < best["ms"]:
            best = rows[-1]
    for ex in a.extras:
        ext = TA.build(extra=tuple(ex.split()))
        ok, err, ms = measure(ext, best["knobs"])
        rows.append(dict(extra=ex, knobs=best["knobs"], ms=ms, err=err))
        print(f"   {'  ' if ok else 'x '}{ms if ms else float('nan'):8.4f} ms  err {err:.1e}  [{ex}] {best['knobs']}", flush=True)
        if ok and ms < best["ms"]:
            best = rows[-1]
    win = best if best["ms"] < t_def * (1 - a.margin) else rows[0]
    print(f"-> {key}: {win['ms']:.4f} ms ({t_def / win['ms']:.3f}x the default)  [{win['extra']}] {win['knobs']}", flush=True)
    return key, dict(knobs=win["knobs"], extra=win["extra"], ms=win["ms"], default_ms=t_def, candidates=rows)


table = json.loads(TA.SELECT_PATH.read_text()) if TA.SELECT_PATH.exists() else {}
TA._SELECT = {}                                                    # measure against the defaults, not a previous table
for mode in a.mode:
    for var in a.variant:
        for L in a.length:
            k, entry = run_shape(mode, var, L)
            table[k] = entry
            torch.cuda.empty_cache()
if a.write:
    TA.SELECT_PATH.write_text(json.dumps(table, indent=1))
    print("wrote", TA.SELECT_PATH)
