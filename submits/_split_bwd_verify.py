"""Verify the split-path backward swap (stacked savedxn) is CORRECT (grad cos vs pytorch) and
faster. FORCE_SPLIT routes all d through TritonTransitionFunction (the modified backward)."""
import os
os.environ["MINIWORLD_TRANSITION_FORCE_SPLIT"] = "1"

import torch
import triton

from miniworld_kernels.modules import Transition, ImplementationType as IT

dev = "cuda"; bf16 = torch.bfloat16
torch.manual_seed(0)
def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()

print(f"{'shape':<14}{'fwdcos':>9}{'dXcos':>9}{'dWa':>9}{'dWb':>9}{'dWs':>9}{'full ms':>10}")
print("-" * 70)
for d in (128, 256, 512):
    for L in (384, 512):
        M = L * L
        mw = Transition(d, implementation=IT.MINIWORLD).to(dev).to(bf16)
        with torch.no_grad():
            mw.squeeze.weight.normal_(0, 0.02)   # lift zero-init -> non-degenerate grads
        ref = Transition(d, implementation=IT.PYTORCH).to(dev).to(bf16)
        ref.load_state_dict(mw.state_dict())
        x = (torch.randn(1, L, L, d, device=dev, dtype=bf16) * 0.1)
        g = torch.randn(1, L, L, d, device=dev, dtype=bf16)

        xa = x.clone().requires_grad_(True); xb = x.clone().requires_grad_(True)
        ya = mw(xa); yb = ref(xb)
        ya.backward(g); yb.backward(g)
        fc = cos(ya, yb)
        dxc = cos(xa.grad, xb.grad)
        dwa = cos(mw.expand_a.weight.grad, ref.expand_a.weight.grad)
        dwb = cos(mw.expand_b.weight.grad, ref.expand_b.weight.grad)
        dws = cos(mw.squeeze.weight.grad, ref.squeeze.weight.grad)

        def step():
            mw.zero_grad(set_to_none=True); xa.grad = None
            mw(xa).backward(g)
        ms = triton.testing.do_bench(step, warmup=20, rep=60, return_mode="median")
        bad = "" if min(fc, dxc, dwa, dwb, dws) > 0.99 else "  <-- LOW COS !!"
        print(f"d={d} L={L:<7}{fc:>9.4f}{dxc:>9.4f}{dwa:>9.4f}{dwb:>9.4f}{dws:>9.4f}{ms:>10.3f}{bad}")
print("\nSPLIT BWD VERIFY DONE")
