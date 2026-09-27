"""Edge checks: (a) T % 128 != 0 in training, (b) int32 offsets at large L (inference vs the bf16 module)."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trimul_triton as TT  # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType as I  # noqa: E402
from miniworld_engine.modules.triangle_multiplication.bidirectional import BidirectionalTriangleMultiplication  # noqa: E402
rel = lambda u, v: float((u.float() - v.float()).norm() / (v.float().norm() + 1e-30))  # noqa: E731
torch.manual_seed(0)
mod = BidirectionalTriangleMultiplication(128, implementation=I.PYTORCH, p_drop=0.0).cuda().bfloat16()
with torch.no_grad():
    for n, t in mod.named_parameters():
        t.normal_(std=t.shape[-1] ** -.5) if t.ndim >= 2 else t.copy_(1 + .1 * torch.randn_like(t))
for L in [int(a) for a in sys.argv[1:]]:
    z = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, L, device="cuda") > .1
    if L <= 256:                                   # training: dz of the tail tokens
        zr = z.float().requires_grad_()
        ref = __import__("copy").deepcopy(mod).float().train()
        dy = torch.randn_like(z)
        gr = torch.autograd.grad(ref(zr, mask), zr, dy.float())[0]
        zt = z.clone().requires_grad_()
        mod.train()
        g = torch.autograd.grad(TT.forward_train(mod, zt, mask, None), zt, dy)[0]
        T = L * L
        tail = T - (T // 128) * 128
        print(f"L{L} T={T} (T % 128 = {tail}): train dz rel {rel(g, gr):.2e}; last-128-token dz rel "
              f"{rel(g.reshape(T, 128)[-128:], gr.reshape(T, 128)[-128:]):.2e}", flush=True)
    else:                                          # inference at large L: offsets channel * T + token beyond 2^31
        mod.eval()
        with torch.no_grad():
            y = TT.forward(mod, z, mask)
            yb = mod(z, mask)
        print(f"L{L}: NP*T = {512 * L * L} ({'>' if 512 * L * L > 2**31 else '<='} 2^31): infer rel vs bf16 module {rel(y, yb):.2e}", flush=True)
