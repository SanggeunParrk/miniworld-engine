"""Where the ring kernel waits (build -DRING_PROF): python prof_ring.py L ndxp"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transition_bwd_a100 as TB  # noqa: E402

L, ndxp = int(sys.argv[1]), int(sys.argv[2])
ext = TB.build(extra=["-DRING_PROF", *sys.argv[3:]])
mod, x = TB.TA.fixture(L)
dy = torch.randn_like(x)
pk = TB.pack(mod)
bufs = {}
nsm = torch.cuda.get_device_properties(0).multi_processor_count
prof = torch.zeros(nsm * 4, dtype=torch.int64, device="cuda")
for _ in range(5):
    TB.backward_ring(ext, x, dy, pk, bufs, ndxp=ndxp, prof=prof)
torch.cuda.synchronize()
prof.zero_()
TB.backward_ring(ext, x, dy, pk, bufs, ndxp=ndxp, prof=prof)
torch.cuda.synchronize()
pr = prof.view(nsm, 4).cpu().double()
d, w = pr[:ndxp], pr[ndxp:]
print(f"L{L} ndxp {ndxp}: DXP total {d[:,1].mean()/1e3:.0f}k clk (max {d[:,1].max()/1e3:.0f}k), waiting for W {d[:,0].mean()/1e3:.0f}k ({100*d[:,0].mean()/d[:,1].mean():.1f}%)")
print(f"         W   total {w[:,1].mean()/1e3:.0f}k clk (max {w[:,1].max()/1e3:.0f}k), waiting for DXP {w[:,0].mean()/1e3:.0f}k ({100*w[:,0].mean()/w[:,1].mean():.1f}%), items {w[:,2].mean():.1f}")
