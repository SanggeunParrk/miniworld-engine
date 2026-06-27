"""Profile v6 AND dtv1 fwd+bwd at L=1024 with the SAME profiler — compare the
contraction bmm time op-by-op to settle 'same bmm, why slower'."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_here = _Path(__file__).resolve().parent
_sys.path[:] = [p for p in _sys.path if _Path(p).resolve() != _here]
_src_root = _here
while _src_root.name != "src" and _src_root.parent != _src_root:
    _src_root = _src_root.parent
if str(_src_root) not in _sys.path:
    _sys.path.insert(0, str(_src_root))

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile

from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.cute.v6_training import V6TriMul
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication.baseline_dtv1 import (
    fused_triangle_multiplicative_update_dtv1,
)
from miniworld_kernels.modules.triangle_multiplication.module import TriangleMultiplication

D = 128


def run(tag, fn, p, g):
    for _ in range(5):
        p.grad = None
        fn(p).backward(g)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            p.grad = None
            fn(p).backward(g)
        torch.cuda.synchronize()
    print(f"\n========== {tag} (10 fwd+bwd, L=1024) — top 12 by CUDA ==========", flush=True)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12), flush=True)


def main():
    assert torch.cuda.is_available()
    _bdll_patch.apply()
    torch.manual_seed(0)
    base = TriangleMultiplication(d_pair=D, outgoing=True,
                                  implementation=ImplementationType.PYTORCH).cuda()
    for lin in (base.to_left, base.to_left_gate, base.to_right, base.to_right_gate,
                base.to_gate, base.to_out):
        nn.init.normal_(lin.weight, std=D**-0.5)
    base_bf = base.to(torch.bfloat16)
    v6 = V6TriMul(base_bf, direction="out")

    dtv1_kw = dict(
        norm_in_weight=base_bf.ln_pair.weight, norm_in_bias=base_bf.ln_pair.bias,
        p_in_weight=torch.cat([base_bf.to_left.weight, base_bf.to_right.weight], dim=0),
        g_in_weight=torch.cat([base_bf.to_left_gate.weight, base_bf.to_right_gate.weight], dim=0),
        norm_out_weight=base_bf.ln_out.weight, norm_out_bias=base_bf.ln_out.bias,
        p_out_weight=base_bf.to_out.weight, g_out_weight=base_bf.to_gate.weight,
    )

    L = 1024
    p = torch.randn(1, L, L, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn_like(p)
    run("ours_v6", lambda x: v6(x), p, g)
    run("dtv1", lambda x: fused_triangle_multiplicative_update_dtv1(x, "outgoing", None, eps=1e-5, **dtv1_kw),
        p, g)


if __name__ == "__main__":
    main()
