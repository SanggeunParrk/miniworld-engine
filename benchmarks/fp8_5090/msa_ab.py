"""RTX 5090: the opt-in e4m3 OuterProductMean (opm_fp8_sm120, MINIWORLD_OPM_FP8=bound) and PWA (pwa_fp8_sm120, MINIWORLD_PWA_FP8=1) routes against the
default route, same process, alternating A-B; and their error against the module in float64 (docs/fp8.md "RTX 5090", OPM / PWA).

    python msa_ab.py [--rounds 6] [--err] [--noab] [--nogate] opm:cm:cz:L:S[:a] ... pwa:D:C:cz:S:N ...

Time: CUDA-graph replay of 3 calls, 128 MB L2 flush before each call (a flush-only graph subtracted), median of 7 replays; ``--rounds`` rounds, the order of
the two routes flips every round; ratio = median default / median FP8 (> 1 = FP8 faster). The FP8 copy is calibrated by two eager calls first; the
capture runs the lazy guard. ``--nogate``: every shape gate of the routes removed (the shipped gates otherwise: a row whose gate declines reads ~1.00).
Error (``--err``; the H100 tables' columns): OPM rms = rms(y - y64) / rms(y64), judge = max|y - y64| / max|y64|, tol = max(PyTorch bf16 judge, 8e-3);
PWA upd rms / upd max of (y - y64) relative to the update y64 - x, judge. Random weights N(0, 1 / fan_in), LayerNorm weight 1 +- 0.1, biases +- 0.1."""
import argparse
import copy
import json
import statistics
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _msa_common import err, graph_time, opm_case, pwa_case  # noqa: E402

from miniworld_engine.kernels.integrations import opm_fp8_sm120 as FO  # noqa: E402
from miniworld_engine.kernels.integrations import pwa_fp8_sm120 as FP  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("cases", nargs="+")
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--err", action="store_true")
ap.add_argument("--noab", action="store_true")
ap.add_argument("--nogate", action="store_true")
ap.add_argument("--json", default=None)
args = ap.parse_args()
if args.nogate:
    FO.MIN_S = 0
    FP.MIN_SN_ALL = 0
    FP.BIG_MIN_SN = 0


def route(kind, on):
    if kind == "opm":
        FO.POLICY = "bound" if on else None
    else:
        FP.ENABLED = bool(on)


def fired_count(kind):
    return FO.GUARD_STATS["calls"] if kind == "opm" else FP.STATS["calls"] + FP.STATS["wide_calls"]


print(torch.cuda.get_device_name(), flush=True)
for c in args.cases:
    kind, spec = c.split(":", 1)
    m, inp = (opm_case if kind == "opm" else pwa_case)(spec)

    def run():
        with torch.no_grad():
            return m(*inp)

    route(kind, False)
    y0 = run()
    route(kind, True)
    n0 = fired_count(kind)
    run()
    y1 = run()
    fired = fired_count(kind) > n0
    rec = {"case": c, "fired": fired}
    if args.err:
        with torch.no_grad():
            ref = copy.deepcopy(m).double().set_backend("pytorch")
            d_inp = [t.double() if t.is_floating_point() else t for t in inp]
            y64 = ref(*d_inp)
            del ref
            mp = copy.deepcopy(m).set_backend("pytorch")
            with torch.autocast("cuda", enabled=False):
                yp = mp(*inp)
            del mp
        x = inp[0] if kind == "pwa" else None
        rec["pytorch"], rec["default"], rec["fp8"] = err(yp, y64, x), err(y0, y64, x), err(y1, y64, x)
        rec["tol"] = max(rec["pytorch"]["judge"], 8e-3)
        del y64, yp
    if not args.noab:
        ta, tb = [], []
        for r in range(args.rounds):
            for on in ((False, True) if r % 2 == 0 else (True, False)):
                route(kind, on)
                (tb if on else ta).append(graph_time(run))
        a, b = statistics.median(ta), statistics.median(tb)
        rec.update(default_us=round(a, 1), fp8_us=round(b, 1), ratio=round(a / b, 4))
    rec["peak"] = FO.overflow_peak() if kind == "opm" else FP.overflow_peak()
    FP.take_overflow()
    route(kind, False)
    line = f"{c}: fired {fired}"
    if "ratio" in rec:
        line += f" | default {rec['default_us']:.0f} us e4m3 {rec['fp8_us']:.0f} us ratio {rec['ratio']:.3f}"
    if args.err:
        f = lambda e: (f"upd {100 * e['upd_rms']:.2f} / {100 * e['upd_max']:.2f} / judge {100 * e['judge']:.3f} %" if "upd_rms" in e else f"rms {e['rms']:.2e} / judge {e['judge']:.2e}")  # noqa: E731
        line += f" | pytorch {f(rec['pytorch'])} | default {f(rec['default'])} | e4m3 {f(rec['fp8'])} (tol {rec['tol']:.1e})"
    line += f" | peak {rec['peak']:.1f}"
    print(line, flush=True)
    if args.json:
        with open(args.json, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    del m, inp, y0, y1
    torch.cuda.empty_cache()
