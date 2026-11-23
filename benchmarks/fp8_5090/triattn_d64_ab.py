"""Chai-1 TriAttn (64-channel heads) module: default vs the e4m3 d64 route (MINIWORLD_FP8=safe, model chai1), same process, alternating, CUDA graph, 128 MB L2 flush, gates
cleared; update error against the float64 module (L <= 512).
python triattn_d64_ab.py [--rounds N] [--qk] [--model chai1|chai1_confidence] [--ending] [--nomask] L ..."""
import copy
import sys

import torch

import miniworld_engine  # noqa: F401
from miniworld_engine import fp8
from miniworld_engine.kernels.triangle_attention.cuda import sm120_fp8 as F
from miniworld_engine.modules.triangle import TriangleAttention

args = sys.argv[1:]
rounds, model = 6, "chai1"
for flag in ("--rounds", "--model"):
    if flag in args:
        i = args.index(flag)
        val = args[i + 1]
        del args[i : i + 2]
        if flag == "--rounds":
            rounds = int(val)
        else:
            model = val
starting = "--ending" not in args
masked = "--nomask" not in args
F.D64_QK = "--qk" in args
args = [a for a in args if not a.startswith("--")]
F.D64_MIN_L = 0
flush = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")


def graph_time(f, n=2):
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
        for _ in range(5):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            gr.replay()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        return sorted(ts)[len(ts) // 2]

    return (t(g) - t(gf)) / n * 1e3


print("arch", fp8.arch(), "model", model, "starting", starting, "mask", masked, "qk", F.D64_QK, flush=True)
for L in map(int, args):
    m = TriangleAttention(256, 4, model=model, starting=starting, p_drop=0.0).cuda().eval()
    gen = torch.Generator(device="cpu").manual_seed(1)
    with torch.no_grad():
        for nm, p in m.named_parameters():
            if p.dim() > 1:
                p.copy_(torch.randn(p.shape, generator=gen) * p.shape[-1] ** -0.5)
            elif nm.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=gen))
            else:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=gen))
    m64 = copy.deepcopy(m).double().set_backend("pytorch") if L <= 512 else None
    m = m.to(torch.bfloat16)
    m.set_backend("cuda")
    x = torch.randn(1, L, L, 256, generator=torch.Generator(device="cpu").manual_seed(5)).cuda().to(torch.bfloat16)
    mk = torch.ones(1, L, dtype=torch.bool, device="cuda")
    if masked:
        mk[:, L - L // 8 :] = False

    def run():
        with torch.no_grad():
            return m(x, mk)

    F._SITES.clear()
    fp8.disable()
    y0 = run()
    fp8.enable("safe", model="chai1")
    fp8.reset_route_stats()
    run()
    y1 = run()
    st = dict(fp8.route_stats()["triattn"])
    chk = fp8.check()
    if m64 is not None:
        with torch.no_grad():
            y64 = m64(x.double(), mk).double()

        def err(y):
            u = y64 - x.double()
            e = y.double() - y64
            return (e.pow(2).mean().sqrt() / u.pow(2).mean().sqrt()).item(), (e.abs().max() / u.abs().max()).item()

        e0, e1 = err(y0), err(y1)
    else:
        e0 = e1 = (float("nan"),) * 2
    ta, tb = [], []
    for _ in range(rounds):
        fp8.disable()
        ta.append(graph_time(run))
        fp8.enable("safe", model="chai1")
        tb.append(graph_time(run))
    a_, b_ = sorted(ta)[len(ta) // 2], sorted(tb)[len(tb) // 2]
    print(
        f"L{L} | default upd rms {e0[0]:.3e} max {e0[1]:.3e} | fp8 rms {e1[0]:.3e} max {e1[1]:.3e} | pv_d64={st.get('pv_d64')} qk={st.get('qk', 0)} sat={chk} | "
        f"default {a_:.0f} us fp8 {b_:.0f} us ratio {a_ / b_:.3f} (min-based {min(ta) / min(tb):.3f})",
        flush=True,
    )
    torch.cuda.empty_cache()
