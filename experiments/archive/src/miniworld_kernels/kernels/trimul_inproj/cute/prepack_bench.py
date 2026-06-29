"""Verify + bench the pre-packed b_lr path (no per-forward cat/interleave).

v2  = builds (D,4D) b_lr inside every forward (torch.cat + interleave)
v2p = b_lr pre-packed ONCE (prepack_lr_operand), passed in via b_lr=
Both under torch.compile(reduce-overhead), K=8 stack, per-layer ms.
Also checks v2p == fp32 reference. B=1, D=128, bf16, forward-only. COMPUTE NODE.
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

from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.cute.launch import (
    prepack_lr_operand,
    trimul_inproj_cute_forward,
)
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication.module import (
    TriangleMultiplication,
    _load_cute_fns,
)

K = 8


def _bench(fn, *, warmup=20, rep=50):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def main():
    assert torch.cuda.is_available()
    print(f"prepack b_lr verify+bench (K={K}) on {torch.cuda.get_device_name(0)}", flush=True)
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
    WR, WRg = mod.to_right.weight.T, mod.to_right_gate.weight.T
    b_lr = prepack_lr_operand(WL, WLg, WR, WRg)  # (D, 4D), built ONCE

    # --- correctness: packed left/right == fp32 reference ---
    torch.manual_seed(1)
    xchk = torch.randn(1, 128, 128, D, device="cuda", dtype=dtype)
    with torch.no_grad():
        lp, rp, _ = trimul_inproj_cute_forward(
            xchk, None, None, None, None, None, bdll_direct=True,
            compute_gate=False, b_lr=b_lr)
        xf = xchk.float()
        lref = (torch.sigmoid(xf @ WLg.float()) * (xf @ WL.float())).permute(0, 3, 1, 2)
        rref = (torch.sigmoid(xf @ WRg.float()) * (xf @ WR.float())).permute(0, 3, 1, 2)
        el = (lp.float() - lref).abs().max().item()
        er = (rp.float() - rref).abs().max().item()
    print(f"correctness (packed vs ref): left max_abs={el:.3e}  right max_abs={er:.3e}  "
          f"{'OK' if max(el, er) < 1e-1 else 'FAIL'}", flush=True)

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

    def back(x_normed, left, right, b, l1, l2, d):
        tri = torch.einsum("bdik,bdjk->bdij", left, right)
        out_normed = _ln_out_t(tri, b, l1, l2, d)
        return tm2_cute_forward(x_normed, out_normed, mod.to_gate.weight, mod.to_out.weight)

    def v2_one(pair):  # builds b_lr every call
        b, l1, l2, d = pair.shape
        xn = _ln_in(pair)
        left, right, _ = trimul_inproj_cute_forward(
            xn, WL, WLg, WR, WRg, None, bdll_direct=True, compute_gate=False)
        return back(xn, left, right, b, l1, l2, d)

    def v2p_one(pair):  # uses pre-packed b_lr
        b, l1, l2, d = pair.shape
        xn = _ln_in(pair)
        left, right, _ = trimul_inproj_cute_forward(
            xn, None, None, None, None, None, bdll_direct=True,
            compute_gate=False, b_lr=b_lr)
        return back(xn, left, right, b, l1, l2, d)

    def stacked(one):
        def stack(pair):
            for _ in range(K):
                pair = one(pair)
            return pair
        return stack

    print(f"\n{'L':>5} | {'v2 (cat)':>9} | {'v2p (prepack)':>13} | {'v2/v2p':>7}  "
          f"(ms/layer, compile-RO)")
    print("-" * 50)
    for L in (512, 1024):
        torch.manual_seed(L)
        pair = torch.randn(1, L, L, D, device="cuda", dtype=dtype)
        with torch.no_grad():
            torch._dynamo.reset()
            c_v2 = torch.compile(stacked(v2_one), mode="reduce-overhead")
            for _ in range(6):
                c_v2(pair)
            t_v2 = _bench(lambda: c_v2(pair)) / K

            torch._dynamo.reset()
            c_v2p = torch.compile(stacked(v2p_one), mode="reduce-overhead")
            for _ in range(6):
                c_v2p(pair)
            t_v2p = _bench(lambda: c_v2p(pair)) / K
        print(f"{L:>5} | {t_v2:>9.3f} | {t_v2p:>13.3f} | {t_v2 / t_v2p:>6.2f}x", flush=True)


if __name__ == "__main__":
    main()
