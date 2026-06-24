"""Whole trimul FORWARD, end-to-end: ours_opt (triton LN_in + front + bmm + fused
back) vs ours_old (cuequiv LN + front + bmm + unfused back) vs dt-v1 vs cuequiv.
Forward-only, eager do_bench (cute kernels don't torch.compile), bf16, B=1, D=128.
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

from miniworld_kernels.kernels.layernorm.triton.main import triton_layernorm
from miniworld_kernels.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import (
    layernorm_linear_cute_fused,
)
from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.cute.launch import (
    prepack_lr_operand, trimul_inproj_cute_forward,
)
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication.baseline_dtv1 import (
    fused_triangle_multiplicative_update_dtv1,
)
from miniworld_kernels.modules.triangle_multiplication.module import (
    TriangleMultiplication, _load_cute_fns,
)

D = 128
EPS = 1e-5


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


def main():
    assert torch.cuda.is_available()
    print(f"whole trimul forward on {torch.cuda.get_device_name(0)}", flush=True)
    _bdll_patch.apply()
    _t1, _t2, _f, lnt = _load_cute_fns()
    torch.manual_seed(0)
    dt = torch.bfloat16

    base = TriangleMultiplication(d_pair=D, implementation=ImplementationType.PYTORCH).cuda()
    for lin in (base.to_left, base.to_left_gate, base.to_right, base.to_right_gate,
                base.to_gate, base.to_out):
        nn.init.normal_(lin.weight, std=D**-0.5)
    base = base.to(dt)
    WLt, WLgt = base.to_left.weight.T.contiguous(), base.to_left_gate.weight.T.contiguous()
    WRt, WRgt = base.to_right.weight.T.contiguous(), base.to_right_gate.weight.T.contiguous()
    Wg = base.to_gate.weight.T.contiguous()
    Wp = base.to_out.weight.T.contiguous()
    Wp_lin = base.to_out.weight.contiguous()
    lw_in, lb_in = base.ln_pair.weight, base.ln_pair.bias
    lw_o, lb_o = base.ln_out.weight, base.ln_out.bias
    b_lr = prepack_lr_operand(WLt, WLgt, WRt, WRgt)

    cq = TriangleMultiplication(d_pair=D, implementation=ImplementationType.CUEQUIVARIANCE).cuda()
    cq.load_state_dict(TriangleMultiplication(d_pair=D, implementation=ImplementationType.PYTORCH).cuda().state_dict()
                       if False else base.float().state_dict())
    cq = cq.to(dt)
    dtv1_kw = dict(
        norm_in_weight=base.ln_pair.weight, norm_in_bias=base.ln_pair.bias,
        p_in_weight=torch.cat([base.to_left.weight, base.to_right.weight], 0),
        g_in_weight=torch.cat([base.to_left_gate.weight, base.to_right_gate.weight], 0),
        norm_out_weight=base.ln_out.weight, norm_out_bias=base.ln_out.bias,
        p_out_weight=base.to_out.weight, g_out_weight=base.to_gate.weight)

    for L in (256, 512, 1024):
        M = L * L
        x = torch.randn(1, L, L, D, device="cuda", dtype=dt)
        xf = x.reshape(M, D)

        def ours_old():
            o = lnt(xf, lw_in, lb_in, eps=EPS, layout="nd->nd")
            xn = (o[0] if isinstance(o, tuple) else o).view(1, L, L, D)
            lft, rgt, _ = trimul_inproj_cute_forward(xn, WLt, WLgt, WRt, WRgt, None,
                                                     bdll_direct=True, compute_gate=False, b_lr=b_lr)
            tri = torch.einsum("bdik,bdjk->bdij", lft, rgt)
            o2 = lnt(tri.reshape(D, M), lw_o, lb_o, eps=EPS, layout="dn->nd")
            on = (o2[0] if isinstance(o2, tuple) else o2)
            gate = torch.sigmoid(xn.reshape(M, D) @ Wg)
            return (on @ Wp) * gate

        def ours_opt():
            xn = triton_layernorm(xf, lw_in, lb_in, EPS)
            lft, rgt, _ = trimul_inproj_cute_forward(xn.view(1, L, L, D), WLt, WLgt, WRt, WRgt, None,
                                                     bdll_direct=True, compute_gate=False, b_lr=b_lr)
            tri = torch.einsum("bdik,bdjk->bdij", lft, rgt)
            gate = torch.sigmoid(xn @ Wg)
            return layernorm_linear_cute_fused(tri.reshape(D, M).t(), lw_o, lb_o, Wp_lin, None,
                                               eps=EPS, gate=gate)

        if L == 256:
            print(f"  cos(opt, old) = {cos(ours_opt(), ours_old().reshape(M, D)):.5f}", flush=True)

        def b_(fn):
            try:
                for _ in range(5):
                    fn()
                return triton.testing.do_bench(fn, warmup=10, rep=50, return_mode="median")
            except Exception as e:  # noqa: BLE001
                print(f"   fail {type(e).__name__}: {str(e)[:60]}", flush=True)
                return float("nan")

        t_old = b_(ours_old)
        t_opt = b_(ours_opt)
        t_dt = b_(lambda: fused_triangle_multiplicative_update_dtv1(x, "outgoing", None, eps=EPS, **dtv1_kw))
        t_cq = b_(lambda: cq(x))
        print(f"  L={L:>4}: ours_old {t_old:.3f} | ours_OPT {t_opt:.3f} | dtv1 {t_dt:.3f} | cuequiv {t_cq:.3f} ms",
              flush=True)


if __name__ == "__main__":
    main()
