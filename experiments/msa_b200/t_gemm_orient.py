"""OPM backward GEMMs, sustained (power-capped): dA = BT^T dO^T [S, (i,c)] as shipped vs its transpose dO BT [(i,c), S], same for dB."""
import sys, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from energy_sol import sustained
bf = torch.bfloat16; N, S, CH = 384, 1024, 32; M = N * CH
g = torch.Generator(device="cuda").manual_seed(0)
a2 = (torch.randn(M, S, device="cuda", generator=g) * (torch.rand(M, S, device="cuda", generator=g) > 0.1)).to(bf)
bt = (torch.randn(M, S, device="cuda", generator=g) * (torch.rand(M, S, device="cuda", generator=g) > 0.1)).to(bf)
dO = (torch.randn(M, M, device="cuda", generator=g) * 0.01).to(bf)
cases = {
    "dA  = mm(bt.t(), dO.t())  [S,(i,c)]": lambda: torch.mm(bt.t(), dO.t()),
    "dA^T= mm(dO, bt)          [(i,c),S]": lambda: torch.mm(dO, bt),
    "dB  = mm(a2.t(), dO)      [S,(j,e)]": lambda: torch.mm(a2.t(), dO),
    "dB^T= mm(dO.t(), a2)      [(j,e),S]": lambda: torch.mm(dO.t(), a2),
    "O   = mm(a2, bt.t())      fwd":       lambda: torch.mm(a2, bt.t()),
}
for k, f in cases.items():
    r = sustained(f, secs=2.0)
    print(f"{k}: {r['ms']*1e3:7.1f} us  {r['J']*1e3:6.1f} mJ  {r['W']:.0f} W  {2*M*M*S/(r['ms']*1e-3)/1e15:.3f} PF", flush=True)
