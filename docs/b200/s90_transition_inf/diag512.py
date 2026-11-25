import torch
from foldjet_engine.kernels.transition.cuda import fused_wide_sm100a as fw
from foldjet_engine.modules import Transition
D=512; M=256
torch.manual_seed(0)
m = Transition(D, n=2, backend="cuda").cuda().eval()
for p in m.parameters():
    if p.dim() > 1: torch.nn.init.normal_(p, 0, D ** -0.5)
    else: torch.nn.init.normal_(p, 1.0 if p is m.ln_in.weight else 0.0, 0.3)
x = torch.randn(1, M, D, device="cuda").to(torch.bfloat16)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    fw.FWD512 = True; yf = m(x).float()[0]
    fw.FWD512 = False; yb = m(x).float()[0]
df, db = yf - x[0].float(), yb - x[0].float()
print("delta rms fused", df.pow(2).mean().sqrt().item(), "base", db.pow(2).mean().sqrt().item())
e = (df - db)
for c in range(0, 512, 64): print("col", c, "err rms", e[:, c:c+64].pow(2).mean().sqrt().item(), "fused rms", df[:, c:c+64].pow(2).mean().sqrt().item())
for r in (0, 1, 31, 32, 127, 128, 255): print("row", r, e[r].pow(2).mean().sqrt().item())
# correlation / scale
print("ratio", (df*db).sum().item()/(db*db).sum().item())
