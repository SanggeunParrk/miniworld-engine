"""Correctness + timing of the A100 Transition forward against the fp32 PyTorch module, with the SoL floor.

  python bench.py --length 384 768 [--anth pf] [--out records/x.json] [--extra -DFOO=1]

Fixture = ../a100_anthropic_baseline (Transition(128, n=4), same init and input seeds); timing = CUDA-graph replay, median of 7 x 50.
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
import transition_a100 as TA  # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType as I  # noqa: E402
from miniworld_engine.modules.transition import Transition  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--length", nargs="+", type=int, default=[384, 768])
p.add_argument("--out", default=None)
p.add_argument("--extra", nargs="*", default=[])
p.add_argument("--anth", nargs="*", default=[], help="Anthropic transition rows to time in the same process (e.g. pf)")
p.add_argument("--no-time", action="store_true")
p.add_argument("--grid", type=int, default=0)
a = p.parse_args()
a.extra = " ".join(a.extra).split()                  # --extra="-DA=1 -DB=1" (argparse would read a bare -D... as an option)
ext = TA.build(extra=a.extra)
record = dict(gpu=torch.cuda.get_device_name(), host=os.uname().nodename, job=os.getenv("SLURM_JOB_ID"), extra=a.extra, rows=[])


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


def rel_rms(y, ref):
    d = y.float() - ref.float()
    return float(d.square().mean().sqrt() / ref.float().square().mean().sqrt())


for L in a.length:
    mod = init_(Transition(128, n=4, implementation=I.PYTORCH).cuda().bfloat16()).eval()
    torch.manual_seed(90323)
    x = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16)
    M = L * L
    sol = TA.sol_us(M)
    with torch.no_grad():
        ref = copy.deepcopy(mod).float()(x.float())
        pk = TA.pack(mod, TA.chunk(a.extra))
        y = TA.forward(ext, x, pk, grid=a.grid)
        torch.cuda.synchronize()
        rel, mx = rel_rms(y, ref), float((y.float() - ref).abs().max())
        relb = rel_rms(mod(x), ref)
        y2 = TA.forward(ext, x, pk, grid=a.grid)
        bitrep = bool(torch.equal(y, y2))
    row = dict(L=L, M=M, rel_rms=rel, max_abs=mx, finite=bool(torch.isfinite(y).all()), bit_reproducible=bitrep,
               bf16_module_rel_rms=relb, sol_us=sol)
    msg = f"L{L}: rel {rel:.3e} (bf16 module {relb:.3e}) max {mx:.3e} finite {row['finite']} bitrep {bitrep}"
    if not a.no_time:
        out = torch.empty(M, 128, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            ms, rounds, kt = graph_ms(lambda: TA.forward(ext, x, pk, out, grid=a.grid))
        us = ms * 1000
        row.update(us=us, rounds_ms=rounds, kernels_us=kt, pct_of_sol=sol["total"] / us * 100)
        msg = f"L{L}: {us:8.1f} us  SoL {sol['total']:.1f} -> {row['pct_of_sol']:.1f}%  (SoL90 <= {sol['total'] / .9:.1f})  " + msg
        for word in a.anth:
            from opt_core.kernels import transition as TR
            W = TR.pack(w_a=mod.expand_a.weight, w_b=mod.expand_b.weight, w_o=mod.squeeze.weight, ln_w=mod.ln_in.weight,
                        ln_b=mod.ln_in.bias, eps=mod.ln_in.eps)
            with torch.no_grad():
                ya, _ = TR.transition(x, W, word=word, residual=True, n_tokens=L)
                rela = rel_rms(ya, ref)
                msa, _, _ = graph_ms(lambda: TR.transition(x, W, word=word, residual=True, n_tokens=L))
            row.setdefault("anthropic", {})[word] = dict(us=msa * 1000, rel_rms=rela)
            msg += f"\n      anth:{word} {msa * 1000:8.1f} us ({sol['total'] / msa / 10:.1f}% SoL) rel {rela:.3e} -> ours {msa * 1000 / us:.2f}x"
    print(msg, flush=True)
    record["rows"].append(row)
if a.out:
    Path(a.out).write_text(json.dumps(record, indent=1))
