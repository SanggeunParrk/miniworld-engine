"""The one-launch sm80 pack equals the reference Python packing (experiments' TA.pack / TB.pack), bit for bit."""
import copy
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import transition_bwd_a100 as TB  # noqa: E402
from miniworld_engine.kernels.transition.cuda import fused_sm80 as F  # noqa: E402
mod, x = TB.TA.fixture(384)
fpk, pk = TB.TA.pack(mod), TB.pack(mod)
wa, wb, ws = (m.weight.to(torch.bfloat16) for m in (mod.expand_a, mod.expand_b, mod.squeeze))
w, gb, wdw, wx, gf, bf = F._pack(mod.ln_in.weight, mod.ln_in.bias, wa, wb, ws)
torch.cuda.synchronize()
wx_ref = TB.pack(copy.deepcopy(mod).requires_grad_(False))["wx"]
half = copy.deepcopy(mod)
with torch.no_grad():
    half.expand_a.weight.mul_(0.5)
wx_ref = TB.pack(half)["wx"]                                   # X's Wa carries the 0.5 (PW writes 2 dA)
for n, a, b in [("w", w, fpk["w"]), ("gb", gb, fpk["gb"]), ("wdw", wdw, pk["wdw"]), ("wx", wx, wx_ref)]:
    print(n, a.shape, b.shape, torch.equal(a.view(torch.int16) if a.dtype != torch.float32 else a, b.view(torch.int16) if b.dtype != torch.float32 else b), flush=True)
