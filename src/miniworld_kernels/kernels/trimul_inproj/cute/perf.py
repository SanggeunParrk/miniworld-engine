"""Micro-benchmark for the trimul_inproj input-projection kernel.

Kernel-development diagnostic (like tm1's ``triton/perf.py``) — NOT the module
bench (`benchmarks/runners/bench.py`). Answers two design questions for the left+right+gate
front kernel:

  1. Does the `[B,D,L,L]` direct write beat the permute fallback?
     -> `new (bdll_direct)` vs `new (fallback)`
  2. Is fusing left+right into ONE wide gated GEMM faster than tm1's two
     separate launches (1 wide launch + register pressure  vs  2 narrow launches
     + a redundant x read)?
     -> `new (bdll_direct)` vs `tm1 2-launch`

All three include the same plain-torch `gate = sigmoid(x@Wg)`, so the delta is
purely the left/right path. Times are forward-only, B=1, D=128, bf16.

Run on a COMPUTE NODE only (srun) — see verify.py for the full command.
"""

from __future__ import annotations


# --- put src/ on the path; use package imports (both kernels have a `launch.py`,
#     so a bare `import launch` would collide). ---
import sys as _sys
from pathlib import Path as _Path

_src_root = _Path(__file__).resolve()
while _src_root.name != "src" and _src_root.parent != _src_root:
    _src_root = _src_root.parent
if str(_src_root) not in _sys.path:
    _sys.path.insert(0, str(_src_root))
# -----------------------------------------------------------------------------

import torch
import triton
from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.cute.launch import trimul_inproj_cute_forward
from miniworld_kernels.kernels.tm1.cute.launch import tm1_cute_forward


def _bench(fn, *, warmup=25, rep=100):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def main():
    assert torch.cuda.is_available()
    print(f"trimul_inproj micro-bench on {torch.cuda.get_device_name(0)}")
    # tm1's bdll_direct also needs our shim (stock quack rejects m-major postact).
    _bdll_patch.apply()

    dtype = torch.bfloat16
    D = 128
    scale = D**-0.5
    Ls = [384, 512, 768, 1024]

    def w():
        return (torch.randn(D, D, device="cuda", dtype=dtype) * scale).contiguous()

    print(f"\n{'L':>5} | {'new bdll(ms)':>13} | {'new fallbk(ms)':>15} | "
          f"{'tm1 2-launch(ms)':>17} | {'bdll/fallbk':>11} | {'bdll/tm1':>9}")
    print("-" * 86)
    for L in Ls:
        torch.manual_seed(0)
        x = torch.randn(1, L, L, D, device="cuda", dtype=dtype)
        WL, WLg, WR, WRg, Wg = w(), w(), w(), w(), w()

        def new_direct():
            return trimul_inproj_cute_forward(x, WL, WLg, WR, WRg, Wg, bdll_direct=True)

        def new_fallback():
            return trimul_inproj_cute_forward(x, WL, WLg, WR, WRg, Wg, bdll_direct=False)

        def tm1_style():
            left, right = tm1_cute_forward(
                x, WL, WLg, WR, WRg, out_layout="bdll_direct"
            )
            gate = torch.sigmoid(x.reshape(L * L, D) @ Wg).view(1, L, L, D)
            return left, right, gate

        t_direct = _bench(new_direct)
        t_fallback = _bench(new_fallback)
        t_tm1 = _bench(tm1_style)
        print(f"{L:>5} | {t_direct:>13.3f} | {t_fallback:>15.3f} | "
              f"{t_tm1:>17.3f} | {t_fallback / t_direct:>10.2f}x | "
              f"{t_tm1 / t_direct:>8.2f}x")


if __name__ == "__main__":
    main()
