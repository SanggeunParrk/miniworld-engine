"""Front gated-GEMM efficiency: blld (contiguous write) vs bdll (strided write)
vs roofline. Isolates whether the bdll M-major postact write is the bottleneck.

left+right fused GLU GEMM only: x_flat(M,D) @ b_lr(D,4D) -> glu -> (M,2D).
  blld = natural n-major postact (M,2D) contiguous
  bdll = M-major view of [2D,L,L] (strided n, stride L*L) -> the direct path
Reports kernel ms (do_bench), achieved TFLOPS and GB/s, % of mem roofline.
B=1, D=128, bf16. COMPUTE NODE only.
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
from quack.gemm_interface import gemm_act

from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.cute.launch import prepack_lr_operand

PEAK_TFLOPS = 990.0   # H100 SXM bf16 dense (approx)
PEAK_TBPS = 3.35      # H100 HBM3 ~3.35 TB/s


def _bench(fn, *, warmup=25, rep=100):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def main():
    assert torch.cuda.is_available()
    print(f"front gated-GEMM roofline on {torch.cuda.get_device_name(0)}", flush=True)
    _bdll_patch.apply()
    dtype, D = torch.bfloat16, 128
    scale = D**-0.5

    def w():
        return (torch.randn(D, D, device="cuda", dtype=dtype) * scale).contiguous()

    b_lr = prepack_lr_operand(w(), w(), w(), w())  # (D, 4D)

    print(f"\n{'L':>5} | {'blld(ms)':>8} | {'bdll(ms)':>8} | {'bdll/blld':>9} | "
          f"{'blld TF/GBs':>13} | {'bdll TF/GBs':>13} | {'bdll %roof':>10}")
    print("-" * 86)
    for L in (384, 512, 768, 1024):
        M = L * L
        x = torch.randn(M, D, device="cuda", dtype=dtype)
        flops = 2 * M * D * (4 * D)               # acc compute (N=4D)
        bytes_ = M * D * 2 + M * (2 * D) * 2       # read x + write postact(2D)
        mem_floor = bytes_ / (PEAK_TBPS * 1e12) * 1e3  # ms

        def blld():
            _, out = gemm_act(A=x, B=b_lr, activation="glu", store_preact=False)
            return out

        lr = torch.empty(1, 2 * D, L, L, device="cuda", dtype=dtype)
        lr_view = lr.view(2 * D, M).T  # (M, 2D) strides (1, M)

        def bdll():
            gemm_act(A=x, B=b_lr, activation="glu", store_preact=False, postact_out=lr_view)
            return lr

        t_blld = _bench(blld)
        t_bdll = _bench(bdll)

        def tf_gbs(t):
            return flops / (t * 1e-3) / 1e12, bytes_ / (t * 1e-3) / 1e9

        tf_b, gb_b = tf_gbs(t_blld)
        tf_d, gb_d = tf_gbs(t_bdll)
        roof_pct = mem_floor / t_bdll * 100
        print(f"{L:>5} | {t_blld:>8.3f} | {t_bdll:>8.3f} | {t_bdll / t_blld:>8.2f}x | "
              f"{tf_b:>5.0f}/{gb_b:>6.0f} | {tf_d:>5.0f}/{gb_d:>6.0f} | "
              f"{roof_pct:>9.0f}%", flush=True)
    print("\n(TF=achieved TFLOPS, GBs=achieved GB/s; %roof = mem-floor/bdll-time)", flush=True)


if __name__ == "__main__":
    main()
