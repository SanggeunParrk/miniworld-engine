"""Does layernorm_linear now accept a bdll (M-major) input -> kills v3's transpose?

v3  : tri = einsum(...->bijd).contiguous()  (bdll->blld TRANSPOSE) -> layernorm_linear
v3p : tri = einsum(...->bdij)  (bdll, contiguous)
      -> M-major view (M,D) strides (1,L*L) -> layernorm_linear  (NO transpose)
Verifies the M-major-input path matches reference, then benches v2 / v3 / v3p
under torch.compile(reduce-overhead), K=8 stack, per-layer ms. COMPUTE NODE only.
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
from miniworld_kernels.kernels.trimul_inproj.cute.launch import trimul_inproj_cute_forward
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication.module import (
    TriangleMultiplication, _load_cute_fns,
)

K = 8


def _bench(fn, *, warmup=20, rep=50):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def main():
    assert torch.cuda.is_available()
    print(f"bdll-input layernorm_linear test (K={K}) on {torch.cuda.get_device_name(0)}", flush=True)
    _bdll_patch.apply()
    dtype, D = torch.bfloat16, 128
    _t1, tm2_cute_forward, _f, layer_norm_transpose = _load_cute_fns()

    mod = TriangleMultiplication(d_pair=D, implementation=ImplementationType.CUTE).cuda()
    torch.manual_seed(0)
    for lin in (mod.to_left, mod.to_left_gate, mod.to_right, mod.to_right_gate,
                mod.to_gate, mod.to_out):
        nn.init.normal_(lin.weight, std=D**-0.5)
    mod = mod.to(dtype)
    gln_w, gln_b, gW = mod.ln_out.weight, mod.ln_out.bias, mod.to_out.weight
    eps = mod.ln_out.eps

    # --- 1. correctness: M-major bdll view into layernorm_linear vs reference ---
    L0 = 256
    tri_bdll = torch.randn(1, D, L0, L0, device="cuda", dtype=dtype)  # mimic bmm out
    view = tri_bdll.reshape(1, D, L0 * L0)[0].t()  # (M, D) strides (1, L*L) — no copy
    with torch.no_grad():
        y_view = layernorm_linear_cute_fused(view, gln_w, gln_b, gW, None, eps=eps)
        y_cont = layernorm_linear_cute_fused(view.contiguous(), gln_w, gln_b, gW, None, eps=eps)
        # reference: LN over D then @W_out, row = tri[:, :, i, j]
        trif = tri_bdll.reshape(1, D, L0 * L0)[0].t().float()  # (M, D)
        ref = torch.nn.functional.linear(
            torch.nn.functional.layer_norm(trif, (D,), gln_w.float(), gln_b.float(), eps),
            gW.float())
        e_view = (y_view.float() - ref).abs().max().item()
        e_cont = (y_cont.float() - ref).abs().max().item()
    print(f"correctness  M-major view vs ref: {e_view:.3e}  |  contiguous vs ref: {e_cont:.3e}  "
          f"{'OK' if e_view < 2e-1 else 'FAIL'}", flush=True)

    def _ln_in(pair):
        b, l1, l2, d = pair.shape
        o = layer_norm_transpose(pair.reshape(b * l1 * l2, d), mod.ln_pair.weight,
                                 mod.ln_pair.bias, eps=mod.ln_pair.eps, layout="nd->nd")
        return (o[0] if isinstance(o, tuple) else o).view(b, l1, l2, d)

    def v2_one(pair):
        b, l1, l2, d = pair.shape
        xn = _ln_in(pair)
        left, right, _ = trimul_inproj_cute_forward(
            xn, mod.to_left.weight.T, mod.to_left_gate.weight.T,
            mod.to_right.weight.T, mod.to_right_gate.weight.T, None,
            bdll_direct=True, compute_gate=False)
        tri = torch.einsum("bdik,bdjk->bdij", left, right)
        tri_dbn = tri.permute(1, 0, 2, 3).reshape(d, b, l1 * l2)
        o2 = layer_norm_transpose(tri_dbn, mod.ln_out.weight, mod.ln_out.bias,
                                  eps=mod.ln_out.eps, layout="dbn->bnd")
        out_normed = (o2[0] if isinstance(o2, tuple) else o2).view(b, l1, l2, d)
        return tm2_cute_forward(xn, out_normed, mod.to_gate.weight, mod.to_out.weight)

    def v3_one(pair):  # OLD: transpose bdll->blld
        b, l1, l2, d = pair.shape
        xn = _ln_in(pair)
        left, right, gate = trimul_inproj_cute_forward(
            xn, mod.to_left.weight.T, mod.to_left_gate.weight.T,
            mod.to_right.weight.T, mod.to_right_gate.weight.T, mod.to_gate.weight.T,
            bdll_direct=True, compute_gate=True)
        tri = torch.einsum("bdik,bdjk->bijd", left, right).contiguous()
        proj = layernorm_linear_cute_fused(tri.reshape(b * l1 * l2, d), gln_w, gln_b, gW, None, eps=eps)
        return proj.view(b, l1, l2, d) * gate

    def v3p_one(pair):  # NEW: bdll M-major view, no transpose
        b, l1, l2, d = pair.shape
        xn = _ln_in(pair)
        left, right, gate = trimul_inproj_cute_forward(
            xn, mod.to_left.weight.T, mod.to_left_gate.weight.T,
            mod.to_right.weight.T, mod.to_right_gate.weight.T, mod.to_gate.weight.T,
            bdll_direct=True, compute_gate=True)
        tri = torch.einsum("bdik,bdjk->bdij", left, right)  # bdll [B,D,L,L]
        view = tri.reshape(b, d, l1 * l2)[0].t()  # (M,D) M-major view, no copy
        proj = layernorm_linear_cute_fused(view, gln_w, gln_b, gW, None, eps=eps)
        return proj.view(b, l1, l2, d) * gate

    def stacked(one):
        def stack(pair):
            for _ in range(K):
                pair = one(pair)
            return pair
        return stack

    print(f"\n{'L':>5} | {'v2':>7} | {'v3(transp)':>10} | {'v3p(bdll)':>9} | "
          f"{'v3p/v2':>6} | {'v3p/v3':>6}  (ms/layer, compile-RO)")
    print("-" * 70)
    for L in (512, 1024):
        torch.manual_seed(L)
        pair = torch.randn(1, L, L, D, device="cuda", dtype=dtype)
        t = {}
        for name, one in (("v2", v2_one), ("v3", v3_one), ("v3p", v3p_one)):
            with torch.no_grad():
                torch._dynamo.reset()
                cfn = torch.compile(stacked(one), mode="reduce-overhead")
                for _ in range(6):
                    cfn(pair)
                t[name] = _bench(lambda: cfn(pair)) / K
        print(f"{L:>5} | {t['v2']:>7.3f} | {t['v3']:>10.3f} | {t['v3p']:>9.3f} | "
              f"{t['v3p']/t['v2']:>5.2f}x | {t['v3p']/t['v3']:>5.2f}x", flush=True)


if __name__ == "__main__":
    main()
