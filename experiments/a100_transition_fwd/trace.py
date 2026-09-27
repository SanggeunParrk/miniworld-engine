"""clock64 trace of one launch (build -DTR_TRACE): per-warp time split into x wait / LayerNorm / chunks / store+turnover.
    python trace.py 768 [-DFOO=1 ...]"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transition_a100 as TA  # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType as I  # noqa: E402
from miniworld_engine.modules.transition import Transition  # noqa: E402

L = int(sys.argv[1])
ext = TA.build(extra=["-DTR_TRACE", *sys.argv[2:]])
mod, x = TA.fixture(L)
M = x.shape[0]
nsm = torch.cuda.get_device_properties(0).multi_processor_count
tr = torch.zeros(nsm * 8 * 32 * 40, dtype=torch.int64, device="cuda")
with torch.no_grad():
    pk = TA.pack(mod, TA.chunk(sys.argv[2:]))
    out = torch.empty(M, 128, device="cuda", dtype=torch.bfloat16)
    for _ in range(20):
        TA.forward(ext, x, pk, out, trace=tr)
    tr.zero_()
    TA.forward(ext, x, pk, out, trace=tr)
torch.cuda.synchronize()
t = tr.view(nsm * 8, 32, 40).cpu().double()
valid = t[:, :, 0] > 0
xw, ln, ch, last, turn, tile, stq, wt = [], [], [], [], [], [], [], []
for wi in range(t.shape[0]):
    n = int(valid[wi].sum())
    for it in range(n):
        r = t[wi, it]
        xw.append(r[1] - r[0]); ln.append(r[2] - r[1])
        ch += (r[4:19] - r[3:18]).tolist(); last.append(r[19] - r[18]); stq.append(r[20] - r[19]); wt += (r[3:19] - r[21:37]).tolist()
        if it + 1 < n:
            turn.append(t[wi, it + 1, 0] - r[20]); tile.append(t[wi, it + 1, 0] - r[0])
mean = lambda v: sum(v) / max(len(v), 1)  # noqa: E731
T = mean(tile)
ch_sorted = sorted(ch)
print(f"L{L}: tile {T:.0f} clk | x wait {mean(xw):.0f} ({100*mean(xw)/T:.1f}%)  LN {mean(ln):.0f} ({100*mean(ln)/T:.1f}%)  "
      f"chunks 16 x {mean(ch):.0f} (p10 {ch_sorted[len(ch)//10]:.0f}, p50 {ch_sorted[len(ch)//2]:.0f}, p90 {ch_sorted[9*len(ch)//10]:.0f}) "
      f"({100*16*mean(ch)/T:.1f}%)  store {mean(stq):.0f} ({100*mean(stq)/T:.1f}%)  turn {mean(turn):.0f} ({100*mean(turn)/T:.1f}%)  last chunk {mean(last):.0f}  chunk-start->weights ready {mean(wt):.0f} (p90 {sorted(wt)[9*len(wt)//10]:.0f})")
# per-warp span and imbalance
starts = t[:, 0, 0][valid[:, 0]]
ends = torch.stack([t[wi, int(valid[wi].sum()) - 1, 19] for wi in range(t.shape[0]) if valid[wi].any()])
print(f"span per warp: min {float((ends - starts).min()):.0f}  max {float((ends - starts).max()):.0f} clk; "
      f"chunk ideal (96 HMMA x 2 warps x 8 clk x 2 steps) = {2 * 96 * 2 * 8}")
