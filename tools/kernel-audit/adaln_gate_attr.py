"""Is adaln_fwd_saveact's Gate=2.77e-02 the documented bf16 cast, or something else?

The checker compares Gate against an fp32 reference. main.adaln_fwd_kernel computes
cond_aff = cond_norm*lnw in fp32 registers, then casts it to bf16 before the two tl.dot calls
(USE_BF16), so `scale` carries the cast's error before the sigmoid ever sees it. If that is the
whole story, then rebuilding the reference with the SAME bf16 operand -- fp32 everywhere else --
must collapse the Gate error to bf16 storage roundoff, i.e. into the ~2e-03 band the other pairs
sit in. If it does not collapse, the attribution is wrong and something else is going on.

Reported side by side, same inputs, same run:
  fp32 reference      -- what the checker uses
  bf16-operand ref    -- fp32 reference with cond_aff cast to bf16 before the GEMMs
"""
from __future__ import annotations
import sys
sys.path.insert(0, "src")

import torch
import torch.nn.functional as F


def main() -> int:
    from miniworld_engine.kernels.drivers import SHAPE_MODE
    from miniworld_engine.kernels.drivers.adaln import _EPS, _adaln_args
    from miniworld_engine.kernels.drivers.conditioned_transition import _D, _DC
    from miniworld_engine.kernels.adaln.triton.main import triton_adaptive_layer_norm

    print(f"device={torch.cuda.get_device_name()}  mode={SHAPE_MODE}  NX={_D} NC={_DC}")
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for seed in (1234, 4321):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            args = _adaln_args()
            # without requires_grad the Function does not build an autograd node and `grad_fn` is
            # None, so the saved activations are unreachable -- the saved Gate is the whole point
            for t in args:
                t.requires_grad_(True)
            x, cond, lnw, ws, sb, wb = args
            y = triton_adaptive_layer_norm(x, cond, lnw, ws, sb, wb, _EPS, _EPS)
            saved = y.grad_fn.saved_tensors
            gate = saved[2].float()

            aff32 = F.layer_norm(cond.detach().float(), (_DC,), eps=_EPS) * lnw.detach().float()
            g32 = torch.sigmoid(torch.addmm(sb.detach().float(), aff32, ws.detach().float().t()))
            # the kernel's own operand dtype: cond_aff -> bf16 before the dot
            affbf = aff32.to(torch.bfloat16).float()
            gbf = torch.sigmoid(torch.addmm(sb.detach().float(), affbf, ws.detach().float().t()))

            def rel(a, b):
                return ((a - b).abs().max() / (b.abs().max() or 1.0)).item()

            print(f"  seed={seed}  Gate vs fp32 ref      = {rel(gate, g32):.3e}")
            print(f"  seed={seed}  Gate vs bf16-operand  = {rel(gate, gbf):.3e}"
                  f"   (fp32 vs bf16-operand refs differ by {rel(g32, gbf):.3e})")
            print(f"  seed={seed}  scale std={torch.logit(g32.clamp(1e-6, 1-1e-6)).std().item():.2f}"
                  f"  max|Gate|={gate.abs().max().item():.4f}")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return 0


if __name__ == "__main__":
    sys.exit(main())
