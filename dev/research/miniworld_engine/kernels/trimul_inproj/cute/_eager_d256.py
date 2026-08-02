"""Throwaway: eager (no compile) v6 vs dtv1 fwd+bwd at D=256 — isolate compile artifact."""
import sys
from pathlib import Path

_s = Path(__file__).resolve()
while _s.name != "src" and _s.parent != _s:
    _s = _s.parent
sys.path.insert(0, str(_s))

import torch
import torch.nn as nn
import triton

from miniworld_engine.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_engine.kernels.trimul_inproj.cute.v6_training import V6TriMul
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_multiplication.baseline_dtv1 import (
    fused_triangle_multiplicative_update_dtv1,
)
from miniworld_engine.modules.triangle_multiplication.module import TriangleMultiplication

_bdll_patch.apply()
import os; D = int(os.environ.get("ED","256"))
dt = torch.bfloat16
base = TriangleMultiplication(d_pair=D, outgoing=True,
                              implementation=ImplementationType.PYTORCH).cuda()
torch.manual_seed(0)
for l in (base.to_left, base.to_left_gate, base.to_right, base.to_right_gate,
          base.to_gate, base.to_out):
    nn.init.normal_(l.weight, std=D**-0.5)
bf = base.to(dt)
v6 = V6TriMul(bf, direction="out")
kw = dict(norm_in_weight=bf.ln_pair.weight, norm_in_bias=bf.ln_pair.bias,
          p_in_weight=torch.cat([bf.to_left.weight, bf.to_right.weight], 0),
          g_in_weight=torch.cat([bf.to_left_gate.weight, bf.to_right_gate.weight], 0),
          norm_out_weight=bf.ln_out.weight, norm_out_bias=bf.ln_out.bias,
          p_out_weight=bf.to_out.weight, g_out_weight=bf.to_gate.weight)


def bench(fn, p, g):
    def step():
        p.grad = None
        fn(p).backward(g)
    for _ in range(3):
        step()
    return triton.testing.do_bench(step, warmup=10, rep=50, return_mode="median", grad_to_none=[p])


for L in (512, 1024):
    p = torch.randn(1, L, L, D, device="cuda", dtype=dt, requires_grad=True)
    g = torch.randn_like(p)
    tv = bench(lambda x: v6(x), p, g)
    td = bench(lambda x: fused_triangle_multiplicative_update_dtv1(x, "outgoing", None, eps=1e-5, **kw), p, g)
    print(f"  EAGER D=256 L={L}: v6={tv:.3f}  dtv1={td:.3f}  v6/dtv1={tv/td:.2f}x", flush=True)
