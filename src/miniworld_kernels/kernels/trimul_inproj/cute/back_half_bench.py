"""Back-half: current (cuequiv LN_out + out_n@Wp + gate-mul) vs
layernorm_linear_cute_fused (LN_out + @Wp + gate-mul in ONE kernel, = v5).
gate = sigmoid(x_n@Wg) precomputed (common to both). bf16, B=1, D=128.
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
import torch.nn.functional as F
import triton

from miniworld_kernels.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import (
    layernorm_linear_cute_fused,
)
from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.modules.triangle_multiplication.module import _load_cute_fns

D = 128
EPS = 1e-5


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


def main():
    assert torch.cuda.is_available()
    print(f"back-half bench on {torch.cuda.get_device_name(0)}", flush=True)
    _bdll_patch.apply()
    _t1, _t2, _f, lnt = _load_cute_fns()
    torch.manual_seed(0)
    dt = torch.bfloat16

    for L in (256, 512, 1024):
        M = L * L
        tri = torch.randn(1, D, L, L, device="cuda", dtype=dt) * 0.3   # bdll
        gln_w = torch.randn(D, device="cuda", dtype=dt)
        gln_b = torch.randn(D, device="cuda", dtype=dt) * 0.1
        Wp = torch.randn(D, D, device="cuda", dtype=dt) * D**-0.5       # nn.Linear (N,K)
        gate = torch.rand(M, D, device="cuda", dtype=dt)               # = sigmoid(x_n@Wg), precomputed
        tri_view = tri.reshape(D, M).t()                               # (M,D) strided (m-major)

        # fused (v5): LN_out + @Wp + gate-mul in one
        y_f = layernorm_linear_cute_fused(tri_view, gln_w, gln_b, Wp, None, eps=EPS, gate=gate)

        # reference
        trin = tri.permute(0, 2, 3, 1).reshape(M, D).float()
        out_n = F.layer_norm(trin, (D,), gln_w.float(), gln_b.float(), EPS)
        y_r = (out_n @ Wp.float().T) * gate.float()
        if L == 256:
            print(f"  cos(fused, ref) = {cos(y_f, y_r):.5f}", flush=True)

        def b_(fn):
            for _ in range(5):
                fn()
            return triton.testing.do_bench(fn, warmup=10, rep=50, return_mode="median")

        def cur():
            o = lnt(tri.reshape(D, M), gln_w, gln_b, eps=EPS, layout="dn->nd")  # (M,D) blld, fuse transpose
            on = o[0] if isinstance(o, tuple) else o
            return (on @ Wp.t()) * gate

        def fused():
            return layernorm_linear_cute_fused(tri_view, gln_w, gln_b, Wp, None, eps=EPS, gate=gate)

        try:
            tc = b_(cur)
        except Exception as e:  # noqa: BLE001
            tc = float("nan"); print(f"   cur fail {type(e).__name__}: {str(e)[:60]}", flush=True)
        tf = b_(fused)
        print(f"  L={L:>4}: current(cuequiv LN_out + matmul + mul) {tc:.3f} ms | "
              f"layernorm_linear fused {tf:.3f} ms  {'FUSED WINS' if tf < tc else 'current wins'}",
              flush=True)


if __name__ == "__main__":
    main()
