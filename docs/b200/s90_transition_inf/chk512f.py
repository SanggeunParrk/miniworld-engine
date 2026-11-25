# chk512f.py n M mean ... : the fused D 512 pair kernel (FWD512) vs the generic chain vs fp64
import sys, torch
from foldjet_engine.kernels.transition.cuda import fused_wide_sm100a as fw
from foldjet_engine.modules import Transition
D = 512
for c in sys.argv[1:]:
    n, M, mu = c.split(","); n, M, mu = int(n), int(M), float(mu)
    torch.manual_seed(0)
    m = Transition(D, n=n, backend="cuda").cuda().eval()
    for p in m.parameters():
        if p.dim() > 1: torch.nn.init.normal_(p, 0, D ** -0.5)
        else: torch.nn.init.normal_(p, 1.0 if p is m.ln_in.weight else 0.0, 0.3)
    x = (torch.randn(1, M, D, device="cuda") + mu).to(torch.bfloat16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        fw.FWD512 = True; yf = m(x).float(); torch.cuda.synchronize()
        fw.FWD512 = False; yb = m(x).float()
    xd = x.double(); ln = torch.nn.functional.layer_norm(xd, (D,), m.ln_in.weight.double(), m.ln_in.bias.double(), 1e-5)
    wa, wb, ws = m.expand_a.weight.double(), m.expand_b.weight.double(), m.squeeze.weight.double()
    ref = xd + (torch.nn.functional.silu(ln @ wa.t()) * (ln @ wb.t())) @ ws.t()
    ef, eb = (yf.double() - ref), (yb.double() - ref)
    print(f"n {n} M {M} mean {mu}: fused-vs-base max {(yf - yb).abs().max().item():.3e}; err/max|ref|: fused {ef.abs().max().item() / ref.abs().max().item():.3e} rms {ef.pow(2).mean().sqrt().item():.3e}; base {eb.abs().max().item() / ref.abs().max().item():.3e} rms {eb.pow(2).mean().sqrt().item():.3e}", flush=True)
