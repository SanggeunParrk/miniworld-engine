"""Version comparison UNDER torch.compile (reduce-overhead), stacked K layers.

All numbers assume torch.compile (no eager). Per-layer ms for a K-deep stack
(realistic: AlphaFold stacks many trimul blocks; this is where compile's per-call
overhead amortizes). Variants:
  current = _forward_cute math (tm1 2-launch -> bmm -> LN_out -> tm2)
  v2      = trimul_inproj left+right fused -> bmm -> LN_out -> tm2
  v3      = trimul_inproj(+gate) -> bmm -> layernorm_linear + torch gate mul
B=1, D=128, bf16, forward-only. COMPUTE NODE only.
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

from miniworld_engine.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import (
    layernorm_linear_cute_fused,
)
from miniworld_engine.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_engine.kernels.trimul_inproj.cute.launch import trimul_inproj_cute_forward
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_multiplication.module import (
    TriangleMultiplication,
    _load_cute_fns,
)

K = 8


def _bench(fn, *, warmup=20, rep=50):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def main():
    assert torch.cuda.is_available()
    print(f"version table UNDER torch.compile (K={K} stack) on "
          f"{torch.cuda.get_device_name(0)}", flush=True)
    _bdll_patch.apply()
    dtype, D = torch.bfloat16, 128
    tm1_cute_forward, tm2_cute_forward, _f, layer_norm_transpose = _load_cute_fns()

    mod = TriangleMultiplication(d_pair=D, implementation=ImplementationType.CUTE).cuda()
    torch.manual_seed(0)
    for lin in (mod.to_left, mod.to_left_gate, mod.to_right, mod.to_right_gate,
                mod.to_gate, mod.to_out):
        nn.init.normal_(lin.weight, std=D**-0.5)
    mod = mod.to(dtype)

    def _ln_in(pair):
        b, l1, l2, d = pair.shape
        o = layer_norm_transpose(pair.reshape(b * l1 * l2, d), mod.ln_pair.weight,
                                 mod.ln_pair.bias, eps=mod.ln_pair.eps, layout="nd->nd")
        return (o[0] if isinstance(o, tuple) else o).view(b, l1, l2, d)

    def _ln_out_t(tri, b, l1, l2, d):
        tri_dbn = tri.permute(1, 0, 2, 3).reshape(d, b, l1 * l2)
        o2 = layer_norm_transpose(tri_dbn, mod.ln_out.weight, mod.ln_out.bias,
                                  eps=mod.ln_out.eps, layout="dbn->bnd")
        return (o2[0] if isinstance(o2, tuple) else o2).view(b, l1, l2, d)

    def current_one(pair):
        b, l1, l2, d = pair.shape
        x_normed = _ln_in(pair)
        left, right = tm1_cute_forward(
            x_normed, mod.to_left.weight.T, mod.to_left_gate.weight.T,
            mod.to_right.weight.T, mod.to_right_gate.weight.T, out_layout="bdll_direct")
        tri = torch.einsum("bdik,bdjk->bdij", left, right)
        out_normed = _ln_out_t(tri, b, l1, l2, d)
        return tm2_cute_forward(x_normed, out_normed, mod.to_gate.weight, mod.to_out.weight)

    def v2_one(pair):
        b, l1, l2, d = pair.shape
        x_normed = _ln_in(pair)
        left, right, _ = trimul_inproj_cute_forward(
            x_normed, mod.to_left.weight.T, mod.to_left_gate.weight.T,
            mod.to_right.weight.T, mod.to_right_gate.weight.T, None,
            bdll_direct=True, compute_gate=False)
        tri = torch.einsum("bdik,bdjk->bdij", left, right)
        out_normed = _ln_out_t(tri, b, l1, l2, d)
        return tm2_cute_forward(x_normed, out_normed, mod.to_gate.weight, mod.to_out.weight)

    def v3_one(pair):
        b, l1, l2, d = pair.shape
        x_normed = _ln_in(pair)
        left, right, gate = trimul_inproj_cute_forward(
            x_normed, mod.to_left.weight.T, mod.to_left_gate.weight.T,
            mod.to_right.weight.T, mod.to_right_gate.weight.T, mod.to_gate.weight.T,
            bdll_direct=True, compute_gate=True)
        tri = torch.einsum("bdik,bdjk->bijd", left, right).contiguous()
        proj = layernorm_linear_cute_fused(
            tri.reshape(b * l1 * l2, d), mod.ln_out.weight, mod.ln_out.bias,
            mod.to_out.weight, None, eps=mod.ln_out.eps)
        return proj.view(b, l1, l2, d) * gate

    def stacked(one):
        def stack(pair):
            for _ in range(K):
                pair = one(pair)
            return pair
        return stack

    variants = (("current", current_one), ("v2", v2_one), ("v3", v3_one))
    print(f"\n{'L':>5} | {'current/lyr':>11} | {'v2/lyr':>8} | {'v3/lyr':>8} | "
          f"{'v2/cur':>6} | {'v3/cur':>6}   (ms, compile-RO, per layer)")
    print("-" * 70)
    for L in (384, 512, 768, 1024):
        torch.manual_seed(L)
        pair = torch.randn(1, L, L, D, device="cuda", dtype=dtype)
        times = {}
        for name, one in variants:
            with torch.no_grad():
                torch._dynamo.reset()
                cfn = torch.compile(stacked(one), mode="reduce-overhead")
                for _ in range(6):
                    cfn(pair)
                times[name] = _bench(lambda: cfn(pair)) / K
        c, v2, v3 = times["current"], times["v2"], times["v3"]
        print(f"{L:>5} | {c:>11.3f} | {v2:>8.3f} | {v3:>8.3f} | "
              f"{c / v2:>5.2f}x | {c / v3:>5.2f}x", flush=True)


if __name__ == "__main__":
    main()
