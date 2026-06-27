"""Profile ONE v6 single-dir backward at L=1024 to find the hot op (the 37ms bwd)."""

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
from miniworld_kernels.modules.triangle_multiplication.module import TriangleMultiplication

import os
D = int(os.environ.get("V6_D", "128"))


def main():
    assert torch.cuda.is_available()
    _bdll_patch.apply()
    torch.manual_seed(0)
    base = TriangleMultiplication(d_pair=D, outgoing=True,
                                  implementation=ImplementationType.PYTORCH).cuda()
    for lin in (base.to_left, base.to_left_gate, base.to_right, base.to_right_gate,
                base.to_gate, base.to_out):
        nn.init.normal_(lin.weight, std=D**-0.5)
    v6 = V6TriMul(base.to(torch.bfloat16), direction="out")

    import sys
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    print(f"### L={L} ###", flush=True)
    p = torch.randn(1, L, L, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn_like(p)

    for _ in range(5):                       # warmup
        p.grad = None
        v6(p).backward(g)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        for _ in range(10):
            p.grad = None
            y = v6(p)
            y.backward(g)
        torch.cuda.synchronize()

    print(f"=== L={L}: top 18 by CUDA self time (÷10 = per fwd+bwd) ===", flush=True)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=18), flush=True)


if __name__ == "__main__":
    main()
