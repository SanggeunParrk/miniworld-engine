import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import transition_bwd_a100 as TB  # noqa: E402
ext = TB.build(extra=sys.argv[2:])
TB.RING_K = int(next((f.split('=')[1] for f in sys.argv[2:] if f.startswith('-DRING_K=')), 8))
mod, x = TB.TA.fixture(int(sys.argv[1]))
dy = torch.randn_like(x)
pk = TB.pack(mod)
bufs = {}
for _ in range(4):
    TB.backward_ring(ext, x, dy, pk, bufs, ndxp=64)
torch.cuda.synchronize()
