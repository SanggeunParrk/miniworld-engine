"""Correctness + timing of the A100 TriMul forward against the fp32 PyTorch module, with the SoL composite per kernel.

  python bench.py --variant bidir single --length 384 768 [--out records/x.json] [--extra -DFOO=1]
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
import trimul_a100 as TA  # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType as I  # noqa: E402
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication  # noqa: E402
from miniworld_engine.modules.triangle_multiplication.bidirectional import BidirectionalTriangleMultiplication  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--variant", nargs="+", default=["bidir", "single"])
p.add_argument("--length", nargs="+", type=int, default=[384, 768])
p.add_argument("--out", default=None)
p.add_argument("--extra", nargs="*", default=[])
p.add_argument("--no-time", action="store_true")
p.add_argument("--nomask", action="store_true")
a = p.parse_args()
a.extra = [x for e in a.extra for x in e.split()]          # one quoted string may carry several flags
ext = TA.build(extra=a.extra)
record = dict(gpu=torch.cuda.get_device_name(), host=os.uname().nodename, extra=a.extra, rows=[])


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
            kt[e.name[:60]] += e.device_time_total / 20
    return statistics.median(rounds), rounds, dict(sorted(kt.items(), key=lambda kv: -kv[1]))


for var in a.variant:
    for L in a.length:
        mod = (BidirectionalTriangleMultiplication(128, implementation=I.PYTORCH) if var == "bidir"
               else TriangleMultiplication(128, implementation=I.PYTORCH))
        mod = init_(mod.cuda().bfloat16()).eval()
        torch.manual_seed(90323)
        z = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16)
        mask = None if a.nomask else torch.rand(1, L, device="cuda") > .1
        with torch.no_grad():
            ref = copy.deepcopy(mod).float()(z.float(), mask)
            pk = TA.pack(mod)
            y = TA.forward(ext, z, mask, pk)
            torch.cuda.synchronize()
            d = y.float() - ref
            rel = float(d.square().mean().sqrt() / ref.square().mean().sqrt())
            mx = float(d.abs().max())
            yb = mod(z, mask)                               # the bf16 module itself, for scale
            relb = float((yb.float() - ref).square().mean().sqrt() / ref.square().mean().sqrt())
        row = dict(variant=var, L=L, ch=pk["ch"], rel_rms=rel, max_abs=mx, finite=bool(torch.isfinite(y).all()),
                   bf16_module_rel_rms=relb, sol_us=TA.sol_us(L, pk["ch"]))
        if not a.no_time:
            bufs = {}
            with torch.no_grad():
                ms, rounds, kt = graph_ms(lambda: TA.forward(ext, z, mask, pk, bufs))
            row.update(us=ms * 1000, rounds_ms=rounds, kernels_us=kt)
            s = row["sol_us"]
            k1 = sum(v for k, v in kt.items() if k.startswith("void a100::k1_kernel"))
            k3 = sum(v for k, v in kt.items() if k.startswith("void a100::k3_kernel"))
            ct = sum(v for k, v in kt.items() if "gemm" in k or "sm80_xmma" in k or "cutlass" in k or "contract_kernel" in k)
            row["split_us"] = dict(k1=k1, contraction=ct, k3=k3)
            row["pct_of_sol"] = dict(k1=s["k1"] / k1 * 100 if k1 else None, contraction=s["contraction"] / ct * 100 if ct else None,
                                     k3=s["k3"] / k3 * 100 if k3 else None, op=s["total"] / (ms * 1000) * 100)
            print(f"{var:6s} L{L}: {ms*1000:8.1f} us  (SoL {s['total']:.0f} us -> {row['pct_of_sol']['op']:.1f}%)  "
                  f"K1 {k1:.1f} ({row['pct_of_sol']['k1'] or 0:.0f}%)  C {ct:.1f} ({row['pct_of_sol']['contraction'] or 0:.0f}%)  "
                  f"K3 {k3:.1f} ({row['pct_of_sol']['k3'] or 0:.0f}%)  rel {rel:.3e} (bf16 module {relb:.3e}) max {mx:.3e}", flush=True)
        else:
            print(f"{var:6s} L{L}: rel {rel:.3e} (bf16 module {relb:.3e}) max {mx:.3e} finite {row['finite']}", flush=True)
        record["rows"].append(row)
if a.out:
    Path(a.out).write_text(json.dumps(record, indent=1))
