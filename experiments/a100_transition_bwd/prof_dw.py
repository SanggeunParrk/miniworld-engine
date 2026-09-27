import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import transition_bwd_a100 as TB  # noqa: E402
ext = TB.build()
mod, x = TB.TA.fixture(int(sys.argv[1]))
dy = torch.randn_like(x)
pk = TB.pack(mod)
xn = torch.nn.functional.layer_norm(x.float(), (128,), mod.ln_in.weight.float(), mod.ln_in.bias.float(), mod.ln_in.eps).bfloat16()
part = torch.empty(13, 3, 512, 128, device="cuda")
for _ in range(4):
    ext.bwd_dw(xn, dy, pk["wdw"], pk["gamma"], pk["beta"], part, pk["eps"])
torch.cuda.synchronize()
