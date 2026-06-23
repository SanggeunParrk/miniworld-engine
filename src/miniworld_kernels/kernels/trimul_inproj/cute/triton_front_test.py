"""Triton single-launch front (left+right+gate, proper sigmoid) — verify + bench.

Compares the front THREE ways:
  quack-2L : left+right glu launch  +  fused-sigmoid gate launch  (x read twice)
  triton-1L: one Triton kernel, left+right+gate, x read once, gate proper sigmoid
Front-only timing + correctness, then e2e (compile K=8) of the full pipeline with
each front. B=1, D=128, bf16. Run with QUACK_CACHE_ENABLED=0. COMPUTE NODE only.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_src_root = _Path(__file__).resolve()
while _src_root.name != "src" and _src_root.parent != _src_root:
    _src_root = _src_root.parent
if str(_src_root) not in _sys.path:
    _sys.path.insert(0, str(_src_root))

import torch
import torch.nn as nn
import triton

from miniworld_kernels.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import (
    layernorm_linear_cute_fused,
)
from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.cute.launch import (
    prepack_lr_operand, trimul_inproj_cute_forward,
)
from miniworld_kernels.kernels.trimul_inproj.triton.front import trimul_front_triton
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication.module import (
    TriangleMultiplication, _load_cute_fns,
)

K = 8


def _bench(fn, *, warmup=20, rep=50):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def main():
    assert torch.cuda.is_available()
    print(f"triton single-launch front on {torch.cuda.get_device_name(0)}", flush=True)
    _bdll_patch.apply()
    dtype, D = torch.bfloat16, 128
    _t1, tm2_cute_forward, _f, layer_norm_transpose = _load_cute_fns()

    mod = TriangleMultiplication(d_pair=D, implementation=ImplementationType.CUTE).cuda()
    torch.manual_seed(0)
    for lin in (mod.to_left, mod.to_left_gate, mod.to_right, mod.to_right_gate,
                mod.to_gate, mod.to_out):
        nn.init.normal_(lin.weight, std=D**-0.5)
    mod = mod.to(dtype)
    WL, WLg = mod.to_left.weight.T, mod.to_left_gate.weight.T
    WR, WRg, Wg = mod.to_right.weight.T, mod.to_right_gate.weight.T, mod.to_gate.weight.T
    gln_w, gln_b, gW, eps = mod.ln_out.weight, mod.ln_out.bias, mod.to_out.weight, mod.ln_out.eps
    b_lr = prepack_lr_operand(WL, WLg, WR, WRg)

    # --- correctness: triton front vs fp32 reference ---
    x = torch.randn(1, 256, 256, D, device="cuda", dtype=dtype)
    with torch.no_grad():
        left, right, gate = trimul_front_triton(x, WL, WLg, WR, WRg, Wg)
        xf = x.float()
        lref = (torch.sigmoid(xf @ WLg.float()) * (xf @ WL.float())).permute(0, 3, 1, 2)
        rref = (torch.sigmoid(xf @ WRg.float()) * (xf @ WR.float())).permute(0, 3, 1, 2)
        gref = torch.sigmoid(xf @ Wg.float())  # blld
        el = (left.float() - lref).abs().max().item()
        er = (right.float() - rref).abs().max().item()
        eg = (gate.float() - gref).abs().max().item()
    print(f"correctness  left={el:.2e} right={er:.2e} gate={eg:.2e}  "
          f"{'OK' if max(el, er, eg) < 1e-1 else 'FAIL'}", flush=True)

    # --- front-only timing: quack 2-launch vs triton 1-launch ---
    print(f"\n{'L':>5} | {'quack-2L(ms)':>12} | {'triton-1L(ms)':>13} | {'q2L/t1L':>7}")
    print("-" * 48)
    for L in (384, 512, 768, 1024):
        xx = torch.randn(1, L, L, D, device="cuda", dtype=dtype)

        def quack_2l():
            return trimul_inproj_cute_forward(xx, WL, WLg, WR, WRg, Wg,
                                              bdll_direct=True, compute_gate=True, b_lr=b_lr)

        def triton_1l():
            return trimul_front_triton(xx, WL, WLg, WR, WRg, Wg)

        tq = _bench(quack_2l)
        tt = _bench(triton_1l)
        print(f"{L:>5} | {tq:>12.3f} | {tt:>13.3f} | {tq/tt:>6.2f}x", flush=True)


if __name__ == "__main__":
    main()
