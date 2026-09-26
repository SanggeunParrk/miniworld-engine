"""One module call (ours) between cudaProfilerStart / Stop, after warm-up: for `ncu --profile-from-start off`.
    python prof_ncu.py pwa train"""
import sys, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bench import make_inputs, make_module, module_fn
op, mode = sys.argv[1], sys.argv[2]
mod = make_module(op, "ours"); msa, pair, mask = make_inputs(op, 384, 1024)
mod.train(mode == "train")
if mode == "train":
    msa.requires_grad_(True); pair.requires_grad_(True); params = [msa, pair, *mod.parameters()]
    g = module_fn(op, mod, msa, pair, mask); gout = torch.randn(msa.shape if op == "pwa" else pair.shape, device="cuda", dtype=torch.bfloat16)
    f = lambda: torch.autograd.grad(g(), params, gout)
else:
    torch.set_grad_enabled(False); f = module_fn(op, mod, msa, pair, mask)
for _ in range(5): f()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
f(); torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print("done")
