"""Timeline of the sequential one-launch backward (build -DSEQ_PROF): per CTA PX end, W start/end.  python prof_seq.py 384 [wr]"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transition_bwd_a100 as TB  # noqa: E402

L = int(sys.argv[1]); wr = int(sys.argv[2]) if len(sys.argv) > 2 else 32
ext = TB.build(extra=["-DSEQ_PROF", *sys.argv[3:]])
mod, x = TB.TA.fixture(L)
dy = torch.randn_like(x)
pk = TB.pack(mod)
nsm = torch.cuda.get_device_properties(0).multi_processor_count
prof = torch.zeros(nsm, 4, dtype=torch.int64, device="cuda")
bufs = {}
for _ in range(20):
    TB.backward_seq(ext, x, dy, pk, bufs, wr=wr, prof=prof)
torch.cuda.synchronize()
p = prof.cpu()
t0 = p[:, 0].min()
s, e1, e2, n = (p[:, 0] - t0) / 1e3, (p[:, 1] - t0) / 1e3, (p[:, 2] - t0) / 1e3, p[:, 3]
ntile = x.shape[0] // 256
nt = torch.tensor([(ntile - b + nsm - 1) // nsm for b in range(nsm)])
for k in sorted(set(nt.tolist())):
    m = nt == k
    print(f"L{L} wr{wr} CTAs with {k} PX tiles: {int(m.sum())}  start {s[m].mean():.1f}  PX end {e1[m].min():.1f}..{e1[m].max():.1f} (per tile {(e1[m]-s[m]).mean()/k:.1f})"
          f"  W stages {n[m].float().mean():.0f}  W {(e2[m]-e1[m]).mean():.1f} us ({((e2[m]-e1[m])/n[m]).mean()*1e3:.0f} ns/stage)  end {e2[m].min():.1f}..{e2[m].max():.1f}")
