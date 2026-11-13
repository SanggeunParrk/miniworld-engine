"""TriangleMultiplication (cuda backend, inference, master weights, bf16 pair) before / after the bounded contraction: the same build timed
with ``_bounded.CELLS`` emptied (the cuBLAS contraction, the state before) and as shipped, on an unpadded mask, the last L // 8 padded and
~25 % padded. CUDA-graph timing, interleaved rounds, median. One line per (d, L, direction)."""

import os
import statistics

import torch

from miniworld_engine.kernels.trimul_inproj.cuda import _bounded
from miniworld_engine.modules.triangle import TriangleMultiplication

ROUNDS = int(os.environ.get("ROUNDS", "7"))
CASES = [tuple(map(int, x.split("x"))) for x in os.environ.get("CASES", "128x768,128x1024,128x1536,128x2048,256x1024,256x2048").split(",")]
SHIPPED = dict(_bounded.CELLS)


def graph(fn, it=5):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(it):
            fn()
    g.replay()
    torch.cuda.synchronize()
    return g, it


def once(gi):
    g, it = gi
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / it


for d, L in CASES:
    for outgoing in (True, False):
        torch.manual_seed(0)
        m = TriangleMultiplication(d, outgoing=outgoing, master_weights=True, backend="cuda").cuda().eval()
        with torch.no_grad():
            for p in m.parameters():
                p.normal_(0, 0.05) if p.dim() == 2 else p.normal_(1.0, 0.1)
        x = torch.randn(1, L, L, d, device="cuda").bfloat16()
        row = []
        for label, pad in (("unpadded", 0), ("L//8", L // 8), ("25%", L // 4)):
            mask = torch.ones(1, L, dtype=torch.bool, device="cuda")
            if pad:
                mask[:, L - pad :] = False
            with torch.no_grad():
                _bounded.CELLS = {}
                ref = m(x, mask)
                before = graph(lambda: m(x, mask))
                _bounded.CELLS = SHIPPED
                got = m(x, mask)
                after = graph(lambda: m(x, mask))
            tb, ta = [], []
            for _ in range(ROUNDS):
                tb.append(once(before))
                ta.append(once(after))
            b, a = statistics.median(tb), statistics.median(ta)
            row.append(f"{label} {b:.3f}->{a:.3f}ms ({b / a:.2f}x){'' if torch.equal(ref, got) else ' DIFF'}")
            del before, after
        print(f"d{d} L{L} {'out' if outgoing else 'in '}: " + " | ".join(row), flush=True)
        del m, x
        torch.cuda.empty_cache()
