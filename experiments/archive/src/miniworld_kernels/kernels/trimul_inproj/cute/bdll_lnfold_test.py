"""bdll LN-fold gated GEMM (trimul_front_lnfold): correctness + config sweep vs
current (cuequiv LN + trimul_inproj). Should win large L now (bdll direct, tuned)."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_src_root = _Path(__file__).resolve()
while _src_root.name != "src" and _src_root.parent != _src_root:
    _src_root = _src_root.parent
if str(_src_root) not in _sys.path:
    _sys.path.insert(0, str(_src_root))

import torch
import torch.nn.functional as F
import triton
from quack.gemm_config import GemmConfig

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


def _cfg(tm, tn, pp, cn):
    return GemmConfig(tile_m=tm, tile_n=tn, pingpong=pp, is_dynamic_persistent=False,
                      cluster_m=1, cluster_n=cn, swap_ab=False, max_swizzle_size=8, device_capacity=9)


_CFGS = [
    _cfg(256, 128, False, 2), _cfg(192, 128, True, 2), _cfg(128, 128, True, 1),
    _cfg(128, 128, False, 2), _cfg(128, 256, False, 1), _cfg(256, 256, False, 1),
    _cfg(192, 256, False, 1), _cfg(64, 128, True, 2), _cfg(256, 128, True, 2),
    _cfg(192, 128, False, 2), _cfg(128, 128, False, 1),
]


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


def main():
    assert torch.cuda.is_available()
    print(f"bdll LN-fold on {torch.cuda.get_device_name(0)}", flush=True)
    _bdll_patch.apply()
    _t1, _t2, _f, layer_norm_transpose = _load_cute_fns()
    torch.manual_seed(0)
    dt = torch.bfloat16

    for L in (256, 512, 1024):
        M = L * L
        x = torch.randn(1, L, L, D, device="cuda", dtype=dt) * 0.6
        g = torch.randn(D, device="cuda", dtype=dt)
        b = torch.randn(D, device="cuda", dtype=dt) * 0.1
        WL, WLg, WR, WRg = (torch.randn(D, D, device="cuda", dtype=dt) * D**-0.5 for _ in range(4))
        Bf, S, B2 = prepack_front_folded(WL, WLg, WR, WRg, g, b)
        WLt, WLgt, WRt, WRgt = (w.T.contiguous() for w in (WL, WLg, WR, WRg))
        b_lr = prepack_lr_operand(WLt, WLgt, WRt, WRgt)

        # correctness (use a known-good config)
        cfg0 = _cfg(128, 256, False, 1)
        left_f, right_f = trimul_front_lnfold(x, Bf, S, B2, EPS, config=cfg0)
        xn = F.layer_norm(x.float(), (D,), g.float(), b.float(), EPS).reshape(M, D)
        left_r = torch.sigmoid(xn @ WLg.float().T) * (xn @ WL.float().T)
        right_r = torch.sigmoid(xn @ WRg.float().T) * (xn @ WR.float().T)
        lf = left_f[0].reshape(D, M).T   # bdll -> (M,D)
        rf = right_f[0].reshape(D, M).T
        if L == 256:
            print(f"  cos left={cos(lf, left_r):.5f} right={cos(rf, right_r):.5f}", flush=True)

        def cur():
            o = layer_norm_transpose(x.reshape(M, D), g, b, eps=EPS, layout="nd->nd")
            xn2 = (o[0] if isinstance(o, tuple) else o).view(1, L, L, D)
            return trimul_inproj_cute_forward(xn2, WLt, WLgt, WRt, WRgt, None,
                                              bdll_direct=True, compute_gate=False, b_lr=b_lr)

        for _ in range(5):
            cur()
        tc = triton.testing.do_bench(cur, warmup=10, rep=50, return_mode="median")
        best, bc = float("inf"), None
        for cfg in _CFGS:
            try:
                for _ in range(3):
                    trimul_front_lnfold(x, Bf, S, B2, EPS, config=cfg)
                t = triton.testing.do_bench(lambda: trimul_front_lnfold(x, Bf, S, B2, EPS, config=cfg),
                                            warmup=8, rep=30, return_mode="median")
            except Exception as e:  # noqa: BLE001
                continue
            if t < best:
                best, bc = t, cfg
        if bc is None:
            print(f"  L={L:>4}: ALL CONFIGS FAILED", flush=True)
            continue
        print(f"  L={L:>4}: current {tc:.3f} ms | lnfold-bdll best {best:.3f} ms "
              f"(tm={bc.tile_m} tn={bc.tile_n} pp={bc.pingpong} cn={bc.cluster_n})  "
              f"{'FOLD WINS' if best < tc else 'current wins'}", flush=True)


if __name__ == "__main__":
    main()
