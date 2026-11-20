"""A100: pwa_sm80 past L 2048 -- accuracy vs a chunked fp64 gold and time vs the pytorch / triton backends (same process, A-B-A-B, CUDA graph, L2 flush).
usage: python pwa_long_bench.py d_msa d_pair d_hidden L [L ...]   (one line per case, flushed)"""

import copy
import sys

import torch

sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from miniworld_engine.measure import config as C
from miniworld_engine.modules import MSAPairWeightedAveraging

sys.path.insert(0, "tests/gpu")
from test_pwa_long_inference_sm80 import gold_f64

import miniworld_engine.modules.msa as _msa_mod


def _chunked_random(shape, *, dtype, device, generator, scale=1.0):
    """random_input drawn a slab at a time (an fp32 draw of an L 8192 pair would not fit next to its bf16 result)."""
    out = torch.empty(tuple(shape), dtype=dtype, device=device)
    flat = out.view(-1)
    step = 1 << 27
    for i in range(0, flat.numel(), step):
        part = flat[i : i + step]
        part.copy_(scale * torch.randn(part.shape, device=device, generator=generator))
    return out


_msa_mod.random_input = _chunked_random
dm, dp, dh = map(int, sys.argv[1:4])
lengths = [int(x) for x in sys.argv[4:]]
FLUSH = torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda")


def graph_of(mod, inp):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(2):
            mod(**inp)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = mod(**inp)
    return g, out


def time_graph(g):
    FLUSH.zero_()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b)


def med(x):
    return sorted(x)[len(x) // 2]


for length in lengths:
    gen = torch.Generator(device="cuda").manual_seed(0)
    base = MSAPairWeightedAveraging(dm, dp, 8, d_hidden=dh, p_drop=0.25).cuda().eval()
    C.randomize(base, gen)
    inp = base.make_inputs(length=length, samples=1, depth=32, dtype=torch.bfloat16, device="cuda", generator=gen)
    rows = {}
    with torch.no_grad():
        gold = gold_f64(base, inp["msa"], inp["pair"], inp["mask"])
        sc, rm = gold.abs().max(), gold.pow(2).mean().sqrt()
        graphs = {}
        for name in ("cuda", "pytorch", "triton"):
            k = copy.deepcopy(base).set_backend(name)
            why = k.unsupported(**inp) if name != "pytorch" else None
            if why:
                rows[name] = f"unsupported: {why[:90]}"
                continue
            try:
                g, out = graph_of(k, inp)
            except Exception as e:
                rows[name] = f"error: {str(e)[:90]}"
                continue
            g.replay()
            torch.cuda.synchronize()
            o = out.double()
            rows[name] = (float((o - gold).abs().max() / sc), float(((o - gold) ** 2).mean().sqrt() / rm))
            graphs[name] = (g, out, k)
        t = {n: [] for n in graphs}
        for _ in range(5):
            for n in list(graphs) + list(graphs)[::-1]:
                t[n].append(time_graph(graphs[n][0]))
    line = f"d_msa {dm} d_pair {dp} d_hidden {dh} L {length}: "
    for n, r in rows.items():
        if isinstance(r, tuple):
            line += f"| {n} max {r[0]:.2e} rms {r[1]:.2e} {med(t[n]):.2f} ms "
        else:
            line += f"| {n} {r} "
    print(line, flush=True)
    del graphs, gold, inp, base
    torch.cuda.empty_cache()
