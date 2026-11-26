"""RTX 5090 TriMul e4m3 planes (``trimul_inproj.cuda.sm120_fp8``) against the default, same process, alternating A-B over >= 6 rounds (``dit_common.ab``:
CUDA graphs, 128 MB L2 flush, order reversed every round, a same-cubin control graph), and the update error against the float64 module.

    python trimul_ab6.py [--rounds 6] [--err] D:kind:L ...        # kind 1 outgoing, 2 incoming, 0 bidirectional

Module: ``TriangleMultiplication`` / ``BidirectionalTriangleMultiplication`` (master weights, bf16 autocast, eval), weights N(0, 1 / D), 5 masked tokens.
The route runs with every gate removed (``FOLDJET_ENGINE_TRIMUL_FP8_MINL`` = 0 semantics), static scales calibrated on the eager warm-up, lazy guard.
Ratio = default / variant (> 1: e4m3 faster). Error: update error rms(y - y64) / rms(y64 - x) (docs/fp8.md column), and max |y - y64| / max |y64 - x|."""

import argparse

import torch
from dit_common import ab, graph

from foldjet_engine.kernels.trimul_inproj.cuda import sm120_fp8 as F
from foldjet_engine.modules.triangle import BidirectionalTriangleMultiplication, TriangleMultiplication

ap = argparse.ArgumentParser()
ap.add_argument("cases", nargs="+")
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--err", action="store_true")
args = ap.parse_args()
print(torch.cuda.get_device_name(), flush=True)


def make(d, kind):
    torch.manual_seed(7)
    if kind == 0:
        m = BidirectionalTriangleMultiplication(d, d_hidden=d, master_weights=True, backend="cuda")
    else:
        m = TriangleMultiplication(d, outgoing=kind == 1, master_weights=True, backend="cuda")
    m = m.cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() > 1:
                p.normal_(0, d**-0.5)
    return m


def gold(m, d, kind, x, mask):
    r = BidirectionalTriangleMultiplication(d, d_hidden=d, backend="pytorch") if kind == 0 else TriangleMultiplication(d, outgoing=kind == 1, backend="pytorch")
    r = r.cuda().eval().double()
    r.load_state_dict({k: v.double() for k, v in m.state_dict().items()})
    with torch.no_grad():
        return r(x.double(), mask)


for spec in args.cases:
    d, kind, L = (int(v) for v in spec.split(":"))
    m = make(d, kind)
    torch.manual_seed(3)
    x = torch.randn(1, L, L, d, device="cuda").bfloat16()
    mask = torch.ones(1, L, dtype=torch.bool, device="cuda")
    mask[:, L - 5 :] = False

    def step():
        with torch.autocast("cuda", torch.bfloat16):
            return m(x, mask)

    F._CACHES.clear()
    F.GUARD, F._MINL_ENV = "lazy", "0"
    F.POLICY = None
    with torch.no_grad():
        y0 = step().float()
    F.POLICY = "calib"
    for k in F.GUARD_STATS:
        F.GUARD_STATS[k] = 0
    with torch.no_grad():
        y1 = step().float()
    peak = F.overflow_peak()
    st = dict(F.GUARD_STATS)
    line = f"D{d} {'bidir' if kind == 0 else 'out' if kind == 1 else 'in'} L{L}"
    if args.err:
        y64 = gold(m, d, kind, x, mask)
        def e(y):
            return ((y.double() - y64).norm() / (y64 - x.double()).norm()).item(), ((y.double() - y64).abs().max() / (y64 - x.double()).abs().max()).item()

        (a0, b0), (a1, b1) = e(y0), e(y1)
        line += f" | err default {a0 * 100:.2f} / {b0 * 100:.2f} % e4m3 {a1 * 100:.2f} / {b1 * 100:.2f} % (peak {peak:.1f})"
        del y64
    gs = {}
    F.POLICY = None
    gs["default"] = graph(step)
    gs["control"] = graph(step)
    F.POLICY = "calib"
    gs["e4m3"] = graph(step)
    F.POLICY = None
    t = ab(gs, rounds=args.rounds)
    print(f"{line} | default {t['default']:.1f} us control {t['default'] / t['control']:.3f} e4m3 {t['e4m3']:.1f} us ratio {t['default'] / t['e4m3']:.3f} | stats {st}", flush=True)
    del m, x, gs, y0, y1
    torch.cuda.empty_cache()
