"""RTX 5090: the e4m3 Transition (fp8_sm120, fp8.SM120_ROUTES; MINIWORLD_TRANSITION_FP8 / the safe preset) against the default route, same process, alternating A-B.

    python transition_ab.py [--rounds 6] [--err] d,n,rows ...      (rows: the call's token rows, e.g. 65536 = an L256 pair stack)

Time: CUDA-graph replay of 3 calls, 128 MB L2 flush before each call (subtracted with a flush-only graph), median of 7 replays; per round default
then FP8, the ratio of the per-route medians over the rounds (default / FP8, > 1 = FP8 faster). Error (--err, rows <= 262144): against the module
in float64 on the PyTorch backend, random weights N(0, 1 / fan_in), LayerNorm weight 1 +- 0.1, bias +- 0.1, x ~ N(0, 1): judge = max|y - y64| / max|y64|,
rms = rms(y - y64) / rms(y64), update = rms(y - y64) / rms(y64 - x) (docs/fp8.md, the H100 Transition table's columns)."""
import argparse
import copy
import statistics

import torch

from foldjet_engine.kernels.transition.cuda import fp8_sm120 as F
from foldjet_engine.modules import Transition

ap = argparse.ArgumentParser()
ap.add_argument("cases", nargs="+")
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--err", action="store_true")
args = ap.parse_args()
F.MIN_ROWS = 0
flush = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")


def graph_time(f, n=3):
    f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            flush.zero_()
            f()
    gf = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gf):
        for _ in range(n):
            flush.zero_()

    def t(gr):
        gr.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(7):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            gr.replay()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        return sorted(ts)[3]

    return (t(g) - t(gf)) / n * 1e3


def module(d, n):
    m = Transition(d, n, master_weights=True, backend="cuda").cuda().eval()
    g = torch.Generator(device="cpu").manual_seed(d + n)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.dim() > 1:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5)
            elif name.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
    return m


print(torch.cuda.get_device_name(), "guard", F.GUARD, flush=True)
for c in args.cases:
    d, n, rows = map(int, c.split(","))
    torch.manual_seed(0)
    m = module(d, n)
    x = torch.randn(1, rows, d, device="cuda").bfloat16()

    def run():
        with torch.no_grad():
            return m(x)

    F.POLICY = None
    y0 = run()
    F.POLICY = "calib"
    calls = F.GUARD_STATS["calls"]
    y1 = run()
    fired = F.GUARD_STATS["calls"] > calls
    err = ""
    if args.err and rows <= 262144:
        ref = copy.deepcopy(m).double()
        ref.set_backend("pytorch")
        with torch.no_grad():
            y64 = ref(x.double())
        xd = x.double()

        def e(y):
            y = y.double()
            return ((y - y64).abs().max() / y64.abs().max()).item(), ((y - y64).pow(2).mean().sqrt() / y64.pow(2).mean().sqrt()).item(), (
                (y - y64).pow(2).mean().sqrt() / (y64 - xd).pow(2).mean().sqrt()
            ).item()

        e0, e1 = e(y0), e(y1)
        err = f" | default judge {e0[0]:.4f} rms {e0[1]:.4f} upd {e0[2]:.4f} | e4m3 judge {e1[0]:.4f} rms {e1[1]:.4f} upd {e1[2]:.4f}"
        del ref, y64
    ta, tb = [], []
    for _ in range(args.rounds):
        F.POLICY = None
        ta.append(graph_time(run))
        F.POLICY = "calib"
        tb.append(graph_time(run))
    a, b = statistics.median(ta), statistics.median(tb)
    fl = rows * (2 * d * 2 * d * n + 2 * d * n * d)
    print(
        f"d{d} n{n} rows {rows}: default {a:.0f} us ({fl / a / 1e6:.0f} TFLOPS) e4m3 {b:.0f} us ({fl / b / 1e6:.0f} TFLOPS) ratio {a / b:.3f} fired {fired}"
        f" peak-guard {F.overflow_peak():.0f}{err}",
        flush=True,
    )
    F.POLICY = None
    del m, x, y0, y1
    torch.cuda.empty_cache()
