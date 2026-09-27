"""Correctness (all six gradients vs an fp32 autograd run of the module) + timing of the A100 Transition backward.
    python bench_bwd.py --length 384 768 [--no-time] [--extra="-DFOO=1"]"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transition_bwd_a100 as TB  # noqa: E402
from bench_common import graph_ms, rel_rms  # noqa: E402

TA = TB.TA
p = argparse.ArgumentParser()
p.add_argument("--length", nargs="+", type=int, default=[384, 768])
p.add_argument("--out", default=None)
p.add_argument("--extra", nargs="*", default=[])
p.add_argument("--no-time", action="store_true")
p.add_argument("--ring", type=int, default=0, help="one-launch L2-ring backward with this many DXP CTAs")
p.add_argument("--two", type=int, default=-1, help="two-kernel PX + W backward (PX grid, 0 = all SMs)")
p.add_argument("--seq", type=int, default=0, help="one-launch sequential PX -> W backward, W stages per PX tile (wr)")
p.add_argument("--pwx", type=int, default=0, help="two-kernel PW + X backward with this many replicas (8 slices each)")
p.add_argument("--nrep", type=int, default=27)
p.add_argument("--fused", type=int, default=0, help="one-launch H100-layout backward with this many DX CTAs")
a = p.parse_args()
a.extra = " ".join(a.extra).split()
ext = TB.build(extra=a.extra)
TB.RING_K = int(next((f.split('=')[1] for f in a.extra if f.startswith('-DRING_K=')), 8))
rec = dict(gpu=torch.cuda.get_device_name(), host=os.uname().nodename, job=os.getenv("SLURM_JOB_ID"), extra=a.extra, rows=[])
names = ["dx", "dgamma", "dbeta", "dWa", "dWb", "dWs"]
for L in a.length:
    mod, x = TA.fixture(L)
    torch.manual_seed(7)
    dy = torch.randn_like(x)
    m32 = copy.deepcopy(mod).float().train()
    x32 = x.float().requires_grad_(True)
    p32 = [m32.ln_in.weight, m32.ln_in.bias, m32.expand_a.weight, m32.expand_b.weight, m32.squeeze.weight]
    ref = torch.autograd.grad(m32(x32), [x32] + p32, dy.float())
    pk = TB.pack(mod)
    bufs = {}
    bwd = (lambda: TB.backward_fused(ext, x, dy, pk, bufs, ndx=a.fused)) if a.fused else (lambda: TB.backward(ext, x, dy, pk, bufs))
    if a.ring:
        bwd = lambda: TB.backward_ring(ext, x, dy, pk, bufs, ndxp=a.ring)  # noqa: E731
    if a.two >= 0:
        bwd = lambda: TB.backward_2k(ext, x, dy, pk, bufs, nrep=a.nrep, gpx=a.two)  # noqa: E731
    if a.pwx:
        bwd = lambda: TB.backward_pwx(ext, x, dy, pk, bufs, nrep=a.pwx)  # noqa: E731
    if a.seq:
        bwd = lambda: TB.backward_seq(ext, x, dy, pk, bufs, wr=a.seq)  # noqa: E731
    got = bwd()
    torch.cuda.synchronize()
    rels = {n: rel_rms(g, r) for n, g, r in zip(names, got, ref)}
    got2 = bwd()
    bitrep = all(torch.equal(u, v) for u, v in zip(got, got2))
    row = dict(L=L, rel_rms=rels, bit_reproducible=bitrep, finite=all(bool(torch.isfinite(g).all()) for g in got))
    msg = f"L{L}: rel " + " ".join(f"{k} {v:.2e}" for k, v in rels.items()) + f"  bitrep {bitrep} finite {row['finite']}"
    if not a.no_time:
        ms = graph_ms(bwd)[0]
        sol = TB.sol_us(x.shape[0])
        row.update(us=ms * 1e3, sol_us=sol, pct_of_sol=sol / (ms * 1e3) * 100)
        msg = f"L{L}: bwd {ms*1e3:8.1f} us  SoL {sol:.0f} -> {row['pct_of_sol']:.1f}%  " + msg
    print(msg, flush=True)
    rec["rows"].append(row)
if a.out:
    Path(a.out).write_text(json.dumps(rec, indent=1))
