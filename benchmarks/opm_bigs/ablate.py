"""Removal ablations of opm_fused_kernel vs cuBLAS product, cuda-event timed, standalone. python ablate.py L S"""
import sys, torch
from foldjet_engine.kernels.integrations import opm_train as ot
L, S = int(sys.argv[1]), int(sys.argv[2]); dev = torch.device("cuda"); bf = torch.bfloat16
torch.manual_seed(0)
m = torch.randn(S, L, 64, device=dev, dtype=bf); mk = torch.rand(S, L, device=dev) > 0.1
gam = torch.ones(64, device=dev); bet = torch.zeros(64, device=dev)
wa = (torch.randn(32, 64, device=dev) * 0.1).to(bf); wb = (torch.randn(32, 64, device=dev) * 0.1).to(bf)
wo = (torch.randn(128, 1024, device=dev) * 0.02).to(bf); ob = torch.zeros(128, device=dev)
ext = ot._ext(dev, native=True, tma=False, cm=64, cz=128, fused=True)
a2, bt, stats, mask16, norm = ext.opm_prologue_wide(m, mk, gam, bet, wa, wb, 1e-5, False, None, False)
wp = ot._fused_weight(wo)
def t(f, n=10):
    for _ in range(3): f()
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(n): f()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / n
fl = 2 * S * L * L * 1024
for dbg in (0, 1, 2, 3):
    ms = t(lambda: ext.opm_fused_infer(a2, bt, norm, wp, ob, None, 16 * dbg + 1))
    print(f"fused DBG={dbg}: {ms:.3f} ms  {fl/ms/1e9:.0f} TF-equiv", flush=True)
ms = t(lambda: torch.mm(a2, bt.t())); print(f"cuBLAS product only: {ms:.3f} ms {fl/ms/1e9:.0f} TF", flush=True)
