"""Cute dual-gemm back (v6) vs triton dual-gemm back (v4) vs tm2 (v2).

v6 = quack front (left+right) -> bmm -> CUTE dual-gemm back (one gated GEMM:
     LN(tri)@Wp ⊙ sigmoid(x_n@Wg), gate computed in-kernel, not materialized).
compile K=8, per-layer ms. B=1, D=128, bf16. QUACK_CACHE_DIR fresh. COMPUTE NODE.
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
import torch.nn.functional as F
import triton

from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.cute.launch import (
    prepack_lr_operand, trimul_inproj_cute_forward,
)
from miniworld_kernels.kernels.trimul_inproj.cute.dualgemm_kernel import (
    dualgemm_back_cute, prepack_dualgemm,
)
from miniworld_kernels.kernels.trimul_inproj.triton.back import trimul_back_triton
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication.module import (
    TriangleMultiplication, _load_cute_fns,
)

K = 8


def _bench(fn, *, warmup=20, rep=50):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def main():
    assert torch.cuda.is_available()
    print(f"cute dual-gemm back (v6) vs v4/v2 (K={K}) on {torch.cuda.get_device_name(0)}", flush=True)
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
    gln_w, gln_b, eps = mod.ln_out.weight, mod.ln_out.bias, mod.ln_out.eps
    Wp_t, Wg_t = mod.to_out.weight.T, mod.to_gate.weight.T
    b_lr = prepack_lr_operand(WL, WLg, WR, WRg)
    PRE = prepack_dualgemm(Wp_t, Wg_t, gln_w, gln_b, dtype=dtype, device="cuda", eps=eps)

    # correctness: full v6 vs reference (one layer)
    pair0 = torch.randn(1, 256, 256, D, device="cuda", dtype=dtype)

    def _ln_in(pair):
        b, l1, l2, d = pair.shape
        o = layer_norm_transpose(pair.reshape(b * l1 * l2, d), mod.ln_pair.weight,
                                 mod.ln_pair.bias, eps=mod.ln_pair.eps, layout="nd->nd")
        return (o[0] if isinstance(o, tuple) else o).view(b, l1, l2, d)

    def front_lr(xn):
        return trimul_inproj_cute_forward(xn, WL, WLg, WR, WRg, None,
                                          bdll_direct=True, compute_gate=False, b_lr=b_lr)

    def v2_one(pair):
        b, l1, l2, d = pair.shape
        xn = _ln_in(pair)
        left, right, _ = front_lr(xn)
        tri = torch.einsum("bdik,bdjk->bdij", left, right)
        tri_dbn = tri.permute(1, 0, 2, 3).reshape(d, b, l1 * l2)
        o2 = layer_norm_transpose(tri_dbn, gln_w, gln_b, eps=eps, layout="dbn->bnd")
        out_n = (o2[0] if isinstance(o2, tuple) else o2).view(b, l1, l2, d)
        return tm2_cute_forward(xn, out_n, mod.to_gate.weight, mod.to_out.weight)

    def v4_one(pair):
        b, l1, l2, d = pair.shape
        xn = _ln_in(pair)
        left, right, _ = front_lr(xn)
        tri = torch.einsum("bdik,bdjk->bdij", left, right)
        return trimul_back_triton(tri, xn, Wp_t, Wg_t, gln_w, gln_b, eps)

    def v6_one(pair):
        b, l1, l2, d = pair.shape
        xn = _ln_in(pair)
        left, right, _ = front_lr(xn)
        tri = torch.einsum("bdik,bdjk->bdij", left, right)
        return dualgemm_back_cute(tri, xn, Wp_t, Wg_t, gln_w, gln_b, eps, prepacked=PRE)

    with torch.no_grad():
        yv2 = v2_one(pair0); yv6 = v6_one(pair0)
        print(f"v6 vs v2 max_abs={ (yv2.float()-yv6.float()).abs().max().item():.3e}", flush=True)

    def stacked(one):
        def stack(pair):
            for _ in range(K):
                pair = one(pair)
            return pair
        return stack

    print(f"\n{'L':>5} | {'v2':>7} | {'v4(triton)':>10} | {'v6(cute)':>9} | "
          f"{'v6/v2':>6} | {'v6/v4':>6}  (ms/layer)")
    print("-" * 66)
    for L in (512, 1024):
        torch.manual_seed(L)
        pair = torch.randn(1, L, L, D, device="cuda", dtype=dtype)
        t = {}
        for name, one in (("v2", v2_one), ("v4", v4_one), ("v6", v6_one)):
            with torch.no_grad():
                torch._dynamo.reset()
                cfn = torch.compile(stacked(one), mode="reduce-overhead")
                for _ in range(6):
                    cfn(pair)
                t[name] = _bench(lambda: cfn(pair)) / K
        print(f"{L:>5} | {t['v2']:>7.3f} | {t['v4']:>10.3f} | {t['v6']:>9.3f} | "
              f"{t['v6']/t['v2']:>5.2f}x | {t['v6']/t['v4']:>5.2f}x", flush=True)


if __name__ == "__main__":
    main()
