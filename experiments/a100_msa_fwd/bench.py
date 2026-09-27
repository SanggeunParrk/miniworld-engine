"""Correctness + timing of the A100 MSA forwards against the fp32 PyTorch module (the a100_anthropic_baseline fixture).

  python bench.py --op opm pwa --length 384 768 [--out records/x.json] [-D FOO=1 ...] [--no-time]
"""
import argparse
import collections
import copy
import json
import os
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import msa_a100 as MA  # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType as I  # noqa: E402
from miniworld_engine.modules.msa_pair_weighted_averaging import MSAPairWeightedAveraging  # noqa: E402
from miniworld_engine.modules.outer_product import OuterProductMean  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--op", nargs="+", default=["opm"])
p.add_argument("--length", nargs="+", type=int, default=[384, 768])
p.add_argument("--msa-depth", type=int, default=1024)
p.add_argument("--mask-frac", type=float, default=.1, help="PWA: fraction of masked keys")
p.add_argument("--mask-mode", default="rand", choices=["rand", "none", "all"], help="PWA key mask: random / None / every key masked")
p.add_argument("--out", default=None)
p.add_argument("--define", "-D", action="append", default=[], help="NAME=VAL compile definition (repeatable)")
p.add_argument("--no-time", action="store_true")
p.add_argument("--repeat", type=int, default=1, help="extra launches before the check (for ncu -s)")
a = p.parse_args()
extra = [f"-D{d}" for d in a.define]
ext = MA.build(extra=extra)
record = dict(gpu=torch.cuda.get_device_name(), host=os.uname().nodename, extra=extra, rows=[])
S = a.msa_depth


def init_(m):
    torch.manual_seed(1234)
    with torch.no_grad():
        for n, t in m.named_parameters():
            if t.ndim >= 2:
                t.normal_(std=t.shape[-1] ** -.5)
            elif "weight" in n:
                t.copy_(1 + .1 * torch.randn_like(t))
            else:
                t.normal_(std=.05)
    return m


def graph_ms(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g.replay()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(10):
        g.replay()
    en.record()
    en.synchronize()
    warm = min(10000, max(30, int(300 / max(st.elapsed_time(en) / 10, 1e-3))))
    for _ in range(warm):
        g.replay()
    rounds = []
    for _ in range(7):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(50):
            g.replay()
        en.record()
        en.synchronize()
        rounds.append(st.elapsed_time(en) / 50)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
    kt = collections.defaultdict(float)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            kt[e.name[:70]] += e.device_time_total / 20
    return statistics.median(rounds), rounds, dict(sorted(kt.items(), key=lambda kv: -kv[1]))


def rel(y, ref):
    d = y.float() - ref.float()
    return float(d.square().mean().sqrt() / ref.float().square().mean().sqrt()), float(d.abs().max())


def case_opm(L, pipe=None):
    mod = init_(OuterProductMean(64, 128, 32, implementation=I.PYTORCH).cuda().bfloat16()).eval()
    torch.manual_seed(90323)
    kw = dict(device="cuda", dtype=torch.bfloat16)
    msa, mask, pair = torch.randn(1, S, L, 64, **kw), torch.rand(1, S, L, device="cuda") > .1, torch.randn(1, L, L, 128, **kw)
    ref = copy.deepcopy(mod).float()(msa.float(), mask, residual=pair.float())
    yb = mod(msa, mask, residual=pair)
    pk = MA.pack_opm(mod)
    bufs = {}
    if pipe:
        return (lambda: MA.opm_forward_pipelined(ext, msa, mask, pk, pair, bufs, *pipe)), ref, yb, MA.opm_sol_us(S, L)
    return (lambda: MA.opm_forward(ext, msa, mask, pk, pair, bufs)), ref, yb, MA.opm_sol_us(S, L)


def case_pwa(L, split=False, pipe=None):
    mod = init_(MSAPairWeightedAveraging(64, 128, 8, 32, implementation=I.PYTORCH).cuda().bfloat16()).eval()
    torch.manual_seed(90323)
    kw = dict(device="cuda", dtype=torch.bfloat16)
    msa, pair = torch.randn(1, S, L, 64, **kw), torch.randn(1, L, L, 128, **kw)
    mask = {"rand": torch.rand(1, L, device="cuda") > a.mask_frac, "none": None,
            "all": torch.zeros(1, L, dtype=torch.bool, device="cuda")}[a.mask_mode]
    ref = copy.deepcopy(mod).float()(msa.float(), pair.float(), mask)
    yb = mod(msa, pair, mask)
    pk = MA.pack_pwa(mod)
    bufs = {}
    if pipe:
        cs, ring = pipe
        return (lambda: MA.pwa_forward_pipelined(ext, msa, pair, mask, pk, bufs, cs, ring)), ref, yb, MA.pwa_sol_us(S, L)
    return (lambda: MA.pwa_forward(ext, msa, pair, mask, pk, bufs, split)), ref, yb, MA.pwa_sol_us(S, L)


for op in a.op:
    for L in a.length:
        with torch.no_grad():
            fn, ref, yb, sol = (case_pwa(L, pipe=tuple(int(t) for t in op.split("_")[1:3])) if op.startswith("pwapipe_") else
                              case_opm(L, pipe=tuple(int(t) for t in op.split("_")[1:3])) if op.startswith("opmpipe_") else
                              {"opm": case_opm, "pwa": case_pwa, "pwa_split": lambda L: case_pwa(L, True)}[op](L))
            for _ in range(a.repeat - 1):
                fn()
            y = fn()
            torch.cuda.synchronize()
            r, mx = rel(y, ref)
            rb, _ = rel(yb, ref)
        del yb
        row = dict(op=op, L=L, S=S, rel_rms=r, max_abs=mx, finite=bool(torch.isfinite(y).all()), bf16_module_rel_rms=rb, sol_us=sol)
        msg = f"{op} L{L}: rel {r:.3e} (bf16 module {rb:.3e}) max {mx:.3e}"
        if not a.no_time:
            with torch.no_grad():
                ms, rounds, kt = graph_ms(fn)
            row.update(us=ms * 1000, rounds_ms=rounds, kernels_us=kt, pct_of_sol=sol["total"] / (ms * 1000) * 100)
            msg = (f"{op} L{L}: {ms*1000:8.1f} us (SoL {sol['total']:.0f} -> {row['pct_of_sol']:.0f}%)  " + msg + "\n    "
                   + "  ".join(f"{k[:40]} {v:.1f}" for k, v in list(kt.items())[:6]))
        print(msg, flush=True)
        record["rows"].append(row)
        del fn, ref, y
        torch.cuda.empty_cache()
if a.out:
    Path(a.out).write_text(json.dumps(record, indent=1))
