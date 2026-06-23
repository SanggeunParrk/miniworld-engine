"""Cute fused-back: layernorm_linear with folded gate-mul (gate via C). Verify + bench.

v5 = front quack (left+right + fused-sigmoid gate) -> bmm
     -> layernorm_linear_cute_fused(tri, gate=gate)   # (LN(tri)@Wp) ⊙ gate, ONE quack kernel
v4 = triton fused back (gate computed in back)
v2 = layer_norm_transpose + tm2
compile K=8, per-layer ms. B=1, D=128, bf16. QUACK_CACHE_ENABLED=0. COMPUTE NODE.
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

from miniworld_kernels.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import (
    layernorm_linear_cute_fused,
)
from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.cute.launch import (
    prepack_lr_operand, trimul_inproj_cute_forward,
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
    print(f"cute fused-back (layernorm_linear+gatemul) test (K={K}) on "
          f"{torch.cuda.get_device_name(0)}", flush=True)
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

    # --- correctness: layernorm_linear with fused gate-mul vs fp32 ref ---
    L0, M0 = 256, 256 * 256
    tri = torch.randn(1, D, L0, L0, device="cuda", dtype=dtype)
    gate = torch.sigmoid(torch.randn(M0, D, device="cuda", dtype=dtype))
    with torch.no_grad():
        view = tri.reshape(D, M0).t()  # (M,D) m-major
        y = layernorm_linear_cute_fused(view, gln_w, gln_b, mod.to_out.weight, None,
                                        eps=eps, gate=gate)
        tri_md = tri.reshape(D, M0).t().float()
        proj = F.linear(F.layer_norm(tri_md, (D,), gln_w.float(), gln_b.float(), eps),
                        mod.to_out.weight.float())
        yref = proj * gate.float()
        e = (y.float() - yref).abs().max().item()
    print(f"correctness  y max_abs={e:.3e}  {'OK' if e < 2e-1 else 'FAIL'}", flush=True)

    def _ln_in(pair):
        b, l1, l2, d = pair.shape
        o = layer_norm_transpose(pair.reshape(b * l1 * l2, d), mod.ln_pair.weight,
                                 mod.ln_pair.bias, eps=mod.ln_pair.eps, layout="nd->nd")
        return (o[0] if isinstance(o, tuple) else o).view(b, l1, l2, d)

    def v2_one(pair):
        b, l1, l2, d = pair.shape
        xn = _ln_in(pair)
        left, right, _ = trimul_inproj_cute_forward(
            xn, WL, WLg, WR, WRg, None, bdll_direct=True, compute_gate=False, b_lr=b_lr)
        tri = torch.einsum("bdik,bdjk->bdij", left, right)
        tri_dbn = tri.permute(1, 0, 2, 3).reshape(d, b, l1 * l2)
        o2 = layer_norm_transpose(tri_dbn, gln_w, gln_b, eps=eps, layout="dbn->bnd")
        out_n = (o2[0] if isinstance(o2, tuple) else o2).view(b, l1, l2, d)
        return tm2_cute_forward(xn, out_n, mod.to_gate.weight, mod.to_out.weight)

    def v4_one(pair):  # triton fused back
        b, l1, l2, d = pair.shape
        xn = _ln_in(pair)
        left, right, _ = trimul_inproj_cute_forward(
            xn, WL, WLg, WR, WRg, None, bdll_direct=True, compute_gate=False, b_lr=b_lr)
        tri = torch.einsum("bdik,bdjk->bdij", left, right)
        return trimul_back_triton(tri, xn, Wp_t, Wg_t, gln_w, gln_b, eps)

    def v5_one(pair):  # cute fused back: layernorm_linear + folded gate-mul
        b, l1, l2, d = pair.shape
        xn = _ln_in(pair)
        left, right, gate = trimul_inproj_cute_forward(
            xn, WL, WLg, WR, WRg, Wg, bdll_direct=True, compute_gate=True, b_lr=b_lr)
        tri = torch.einsum("bdik,bdjk->bdij", left, right)
        view = tri.reshape(b, d, l1 * l2)[0].t()
        y = layernorm_linear_cute_fused(view, gln_w, gln_b, mod.to_out.weight, None,
                                        eps=eps, gate=gate.reshape(b * l1 * l2, d))
        return y.view(b, l1, l2, d)

    def stacked(one):
        def stack(pair):
            for _ in range(K):
                pair = one(pair)
            return pair
        return stack

    print(f"\n{'L':>5} | {'v2':>7} | {'v4(triton)':>10} | {'v5(cute)':>9} | "
          f"{'v5/v2':>6} | {'v5/v4':>6}  (ms/layer)")
    print("-" * 66)
    for L in (512, 1024):
        torch.manual_seed(L)
        pair = torch.randn(1, L, L, D, device="cuda", dtype=dtype)
        t = {}
        for name, one in (("v2", v2_one), ("v4", v4_one), ("v5", v5_one)):
            with torch.no_grad():
                torch._dynamo.reset()
                cfn = torch.compile(stacked(one), mode="reduce-overhead")
                for _ in range(6):
                    cfn(pair)
                t[name] = _bench(lambda: cfn(pair)) / K
        print(f"{L:>5} | {t['v2']:>7.3f} | {t['v4']:>10.3f} | {t['v5']:>9.3f} | "
              f"{t['v5']/t['v2']:>5.2f}x | {t['v5']/t['v4']:>5.2f}x", flush=True)


if __name__ == "__main__":
    main()
