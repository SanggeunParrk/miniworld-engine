"""Does _bidir_front_kernel's K-loop read memory OUTSIDE its operands when K % BLOCK_K_D != 0?

Why this script exists
----------------------
The ragged sweep gave two different answers for ``trimul_gemm_gate_mmajor_triton`` at K=125:

    single process, 25 kernels in registry order   rel left=4.19e+00   FAIL
    --isolate (one fresh subprocess per kernel)    rel left=2.01e-03   ok

Both runs used the same shapes and the same config (BLOCK_K_D=64). The difference is what was
already in device memory. The kernel's K-loop (bidirectional.py:88 and :110) bounds rows and
columns but never bounds ``rk``, so at K=125 the second trip reads k = 64..127:

  * x_flat is (M, K) row-major, so k = 125..127 of row m is row m+1's first three channels --
    in-bounds real data, always non-zero.
  * Wlr is (K, 4*H2), so k = 125..127 is 3 * 4*H2 elements PAST THE END of the weight tensor.

The contaminating term is ``x_extra * w_extra``. x_extra is always non-zero, so the wrong answer
appears only when w_extra -- freshly allocated device memory just past the weight -- is non-zero.
In a fresh process that memory is still zero, the garbage products are 0, and the bug is INVISIBLE.
After other kernels have cycled the caching allocator it is non-zero and the result is wrong.
(``tm1/cute/sm100_gate_gemm_collective.py`` already documents this exact failure mode for its own
partial last tile: "once the caching allocator has served non-zero memory, leaks NaN".)

So a green ``--isolate`` row does not clear this kernel; it only says the fresh-process heap
happened to be zero. This script decides it by controlling the one variable that matters, in ONE
process, with everything else held fixed:

    for D in (128, 125):            # 128: K divides BLOCK_K_D. 125: it does not.
        for dirty in (False, True): # dirty = the allocator has served non-zero memory
            run the kernel on identical inputs and compare to a torch reference

A kernel that reads only its operands cannot notice ``dirty``. Its two rows must be BIT-IDENTICAL.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "src")
os.environ.setdefault("MINIWORLD_CONFIG_DIR", os.path.abspath("configs/accuracy"))

import torch


def make_inputs(D: int, L: int):
    """Identical every call: same seed, same shapes, same values."""
    torch.manual_seed(1234)
    h2 = 2 * D
    x = torch.randn(1, L, L, D, device="cuda", dtype=torch.bfloat16)
    WL, WLg, WR, WRg = (
        (torch.randn(D, h2, device="cuda", dtype=torch.bfloat16) * (D**-0.5)).contiguous()
        for _ in range(4)
    )
    return x, WL, WLg, WR, WRg


def dirty_the_allocator() -> None:
    """Leave the caching allocator holding freed blocks full of NON-ZERO bytes.

    No empty_cache(): the freed blocks stay in the pool, so the small (K, 4*H2) weight that
    ``bidir_front_triton`` allocates next is carved out of memory that was written with 7.0, and
    the bytes just past its end are non-zero instead of a fresh zero page.
    """
    for nbytes_elems in (1 << 26, 1 << 24, 1 << 22, 1 << 20, 1 << 18):
        junk = torch.full((nbytes_elems,), 7.0, device="cuda", dtype=torch.bfloat16)
        del junk
    torch.cuda.synchronize()


def one(D: int, L: int, dirty: bool):
    from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import (
        bidir_front_triton,
    )

    x, WL, WLg, WR, WRg = make_inputs(D, L)
    if dirty:
        dirty_the_allocator()
    left, _right, _preact = bidir_front_triton(x, WL, WLg, WR, WRg)

    M, h2 = L * L, 2 * D
    xf = x.float().reshape(M, D)
    ref_l = (torch.sigmoid(xf @ WLg.float()) * (xf @ WL.float())).t()   # (h2, M)
    got_l = left.reshape(h2, M).float()
    rel = (got_l - ref_l).abs().max().item() / (ref_l.abs().max().item() or 1.0)
    return rel, left.reshape(h2, M).clone()


def main() -> int:
    print(f"device={torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability(0)}")
    from miniworld_engine.autotune.configs import configs_for
    cfgs = configs_for("trimul_gemm_gate_mmajor_triton")
    print(f"config(s) for trimul_gemm_gate_mmajor_triton: "
          f"{[c.kwargs for c in cfgs] if cfgs else cfgs}")
    L = 61
    verdicts = []
    for D in (128, 125):
        rows = {}
        for dirty in (False, True):
            rel, out = one(D, L, dirty)
            rows[dirty] = (rel, out)
            print(f"  D(K)={D:4d}  L={L}  K%BLOCK_K_D={D % 64:3d}  dirty={int(dirty)}  "
                  f"rel_left={rel:.3e}")
        same = torch.equal(rows[False][1], rows[True][1])
        ndiff = int((rows[False][1] != rows[True][1]).sum().item())
        print(f"  D(K)={D:4d}: clean vs dirty bit-identical = {same}   differing elements = "
              f"{ndiff} / {rows[False][1].numel()}")
        verdicts.append((D, same, rows[False][0], rows[True][0]))
        print()
    print("=== verdict ===")
    for D, same, rel_clean, rel_dirty in verdicts:
        if same:
            print(f"  K={D}: output does NOT depend on unrelated device memory "
                  f"(rel {rel_clean:.3e} both ways) -- reads only its operands")
        else:
            print(f"  K={D}: output DEPENDS on unrelated device memory "
                  f"(rel {rel_clean:.3e} clean -> {rel_dirty:.3e} dirty) -- "
                  f"OUT-OF-BOUNDS READ on the contraction axis")
    return 0


if __name__ == "__main__":
    sys.exit(main())
