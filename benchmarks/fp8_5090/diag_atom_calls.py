"""Diagnostics: every e4m3 atom DiT block call of a FoldForge predictor run (route_census.py arguments after ``--``), checked against the default block
on the same inputs: per call M / N / A / table rows, finite flags, max |y8 - y| / max |y|, and for a non-finite call the part (pre / post) that makes it
and the input / table ranges. Run eagerly (``--no-compile``) so every call passes through Python.

    python diag_atom_calls.py --model af3 --sizes 4 --no-compile [--out DIR]
"""

import os
import runpy
import sys

for key in [k for k in os.environ if k.startswith("MINIWORLD_") and "FP8" in k]:
    os.environ.pop(key)
os.environ["MINIWORLD_FP8"] = "safe"

import torch  # noqa: E402

from miniworld_engine.kernels.dit_variants.cuda import atom_fp8_sm120 as F  # noqa: E402

_orig = F.block
_n = [0]


def _stats(t):
    t = t.float()
    fin = torch.isfinite(t)
    return f"finite {bool(fin.all())} absmax {t[fin].abs().max().item() if fin.any() else float('nan'):.3e}"


def block(x, tab, bias2, widx, keys, ebias, tinfo, tb, wq, bq, wk, wv, wg, wo, wexp, wsq, qkw, cfg, use):
    args = (x, tab, bias2, widx, keys, ebias, tinfo, tb, wq, bq, wk, wv, wg, wo, wexp, wsq, qkw, cfg)
    y8 = _orig(*args, use)
    with torch.autocast("cuda", enabled=False):
        y0 = F._block(*args, frozenset())
    _n[0] += 1
    i = _n[0]
    ok8, ok0 = bool(torch.isfinite(y8).all()), bool(torch.isfinite(y0).all())
    err = ((y8.float() - y0.float()).abs().max() / y0.float().abs().max()).item() if ok8 and ok0 else float("nan")
    print(f"DIAG call {i} M {x.shape[0]} N {cfg[7]} A {x.shape[0] // cfg[7]} TR {tab.shape[0]} use {sorted(use)} finite e4m3 {ok8} default {ok0} rel {err:.3e} "
          f"cfg {cfg}", flush=True)  # fmt: skip
    if not ok8:
        for part in ("pre", "post"):
            with torch.autocast("cuda", enabled=False):
                yp = F._block(*args, frozenset({part}))
            bad = ~torch.isfinite(yp.float())
            rows = bad.any(1).nonzero().flatten()
            print(f"DIAG   part {part}: non-finite rows {rows.numel()} first {rows[:8].tolist()}", flush=True)
        print(f"DIAG   x {_stats(x)}  tab {_stats(tab)} dtype {tab.dtype} shape {tuple(tab.shape)}  bias2 {_stats(bias2)}", flush=True)
        for nm, w in (("wq", wq), ("wk", wk), ("wv", wv), ("wg", wg), ("wo", wo), ("wsq", wsq), *((f"wexp{j}", w) for j, w in enumerate(wexp))):
            if w.numel():
                print(f"DIAG   {nm} {tuple(w.shape)} {w.dtype} {_stats(w)}", flush=True)
        torch.save({"args": [a.detach().cpu() if torch.is_tensor(a) else a for a in args[:8]], "cfg": cfg}, f"diag_atom_call{i}.pt")
    return y8


F.block = block
sys.argv = [str(os.path.join(os.path.dirname(os.path.abspath(__file__)), "route_census.py")), *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name="__main__")
