"""cuequiv vs triton LayerNorm (LN_in, no transpose), and the front total each gives:
   triton_LN + gemm_act  vs  cuequiv_LN + gemm_act  vs  fold(stats + gemm_gated_ln).
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
import triton

from quack.gemm_config import GemmConfig
from miniworld_kernels.kernels.layernorm.triton.main import triton_layernorm
from miniworld_kernels.kernels.layernorm_linear.triton.stats import stats_triton
from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.cute.gemm_gated_ln import (
    prepack_front_folded, trimul_front_lnfold,
)
from miniworld_kernels.kernels.trimul_inproj.cute.launch import (
    prepack_lr_operand, trimul_inproj_cute_forward,
)
from miniworld_kernels.modules.triangle_multiplication.module import _load_cute_fns

D = 128
EPS = 1e-5


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


def main():
    assert torch.cuda.is_available()
    print(f"LN bench on {torch.cuda.get_device_name(0)}", flush=True)
    _bdll_patch.apply()
    _t1, _t2, _f, lnt = _load_cute_fns()
    torch.manual_seed(0)
    dt = torch.bfloat16
    cfg = GemmConfig(tile_m=128, tile_n=256, pingpong=False, is_dynamic_persistent=False,
                     cluster_m=1, cluster_n=1, swap_ab=False, max_swizzle_size=8, device_capacity=9)

    for L in (256, 512, 1024):
        M = L * L
        x = torch.randn(1, L, L, D, device="cuda", dtype=dt) * 0.6
        xf = x.reshape(M, D)
        g = torch.randn(D, device="cuda", dtype=dt)
        b = torch.randn(D, device="cuda", dtype=dt) * 0.1
        WL, WLg, WR, WRg = (torch.randn(D, D, device="cuda", dtype=dt) * D**-0.5 for _ in range(4))
        WLt, WLgt, WRt, WRgt = (w.T.contiguous() for w in (WL, WLg, WR, WRg))
        b_lr = prepack_lr_operand(WLt, WLgt, WRt, WRgt)
        Bf, S, B2 = prepack_front_folded(WL, WLg, WR, WRg, g, b)

        # correctness: triton LN vs cuequiv
        yt = triton_layernorm(xf, g, b, EPS)
        oc = lnt(xf, g, b, eps=EPS, layout="nd->nd")
        yc = (oc[0] if isinstance(oc, tuple) else oc)
        if L == 256:
            print(f"  cos(triton_LN, cuequiv_LN) = {cos(yt, yc):.5f}", flush=True)

        def b_(fn):
            for _ in range(5):
                fn()
            return triton.testing.do_bench(fn, warmup=10, rep=50, return_mode="median")

        t_cq = b_(lambda: lnt(xf, g, b, eps=EPS, layout="nd->nd"))
        t_tr = b_(lambda: triton_layernorm(xf, g, b, EPS))

        def front_cq():
            o = lnt(xf, g, b, eps=EPS, layout="nd->nd")
            xn = (o[0] if isinstance(o, tuple) else o).view(1, L, L, D)
            return trimul_inproj_cute_forward(xn, WLt, WLgt, WRt, WRgt, None,
                                              bdll_direct=True, compute_gate=False, b_lr=b_lr)

        def front_tr():
            xn = triton_layernorm(xf, g, b, EPS).view(1, L, L, D)
            return trimul_inproj_cute_forward(xn, WLt, WLgt, WRt, WRgt, None,
                                              bdll_direct=True, compute_gate=False, b_lr=b_lr)

        def front_fold():
            return trimul_front_lnfold(x, Bf, S, B2, EPS, config=cfg)

        fc = b_(front_cq)
        ft = b_(front_tr)
        ff = b_(front_fold)
        print(f"  L={L:>4}: LN cuequiv={t_cq:.3f} triton={t_tr:.3f} | "
              f"front: cuequiv+inproj={fc:.3f} triton+inproj={ft:.3f} fold={ff:.3f}", flush=True)


if __name__ == "__main__":
    main()
