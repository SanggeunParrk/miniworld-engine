"""CUDA path edge checks: training at T % 128 != 0 and inference at large L."""
import copy, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trimul_a100 as TA, trimul_train as TC  # noqa: E401,E402
from miniworld_engine.modules.exceptions import ImplementationType as I  # noqa: E402
from miniworld_engine.modules.triangle_multiplication.bidirectional import BidirectionalTriangleMultiplication  # noqa: E402
rel = lambda u, v: float((u.float() - v.float()).norm() / (v.float().norm() + 1e-30))  # noqa: E731
ext = TA.build()
torch.manual_seed(0)
mod = BidirectionalTriangleMultiplication(128, implementation=I.PYTORCH, p_drop=0.0).cuda().bfloat16()
with torch.no_grad():
    for n, t in mod.named_parameters():
        t.normal_(std=t.shape[-1] ** -.5) if t.ndim >= 2 else t.copy_(1 + .1 * torch.randn_like(t))
for L in [int(a) for a in sys.argv[1:]]:
    z = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, L, device="cuda") > .1
    try:
        if L <= 256:
            ref = copy.deepcopy(mod).float().train()
            zr = z.float().requires_grad_()
            dy = torch.randn_like(z)
            gr = torch.autograd.grad(ref(zr, mask), zr, dy.float())[0]
            zt = z.clone().requires_grad_()
            mod.train()
            g = torch.autograd.grad(TC.forward_train(ext, mod, zt, mask, None), zt, dy)[0]
            print(f"CUDA L{L} T%128={L*L % 128}: train dz rel {rel(g, gr):.2e}", flush=True)
        else:
            mod.eval()
            with torch.no_grad():
                y = TA.forward(ext, z, mask, TA.pack(mod), {})
                yb = mod(z, mask)
            print(f"CUDA L{L}: infer rel vs bf16 module {rel(y, yb):.2e}", flush=True)
    except Exception as e:
        print(f"CUDA L{L}: {type(e).__name__}: {str(e).splitlines()[0][:150]}", flush=True)
