"""Speed characterisation of the B>1-generalised TRITON trimul (bidir + unidirectional).

For each (path, L, B) reports fwd and fwd+bwd latency for:
  - triton BATCHED : one call on the (B,L,L,d) tensor (the new single-launch path)
  - triton LOOP    : B separate B=1 calls (the naive alternative the batched grid replaces)
  - pytorch        : the module's PYTORCH backend (eager reference)
plus ms/batch (batched) to show scaling, and the batched-vs-pytorch speedup.

A100, bf16, d=128. NOTE: no tuned autotune cache exists for these A100 shapes, so triton
runs the full autotune grid (warmup absorbs it) — numbers are best-of-grid, representative
but not the tuned-cache optimum.
"""

from __future__ import annotations

import statistics
import time

import torch

from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication import (
    BidirectionalTriangleMultiplication,
)
from miniworld_kernels.modules.triangle_multiplication.module import (
    TriangleMultiplication,
)

DEVICE, BF16 = "cuda", torch.bfloat16
IMPL = ImplementationType


def _mk(cls, impl, **kw):
    m = cls(128, implementation=impl, **kw)
    torch.manual_seed(0)
    for p in m.parameters():
        p.data.normal_(0, 0.5)
    return m.to(DEVICE, BF16)


def _time(fn, n=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def bench_path(name, tri, py, extra_kw):
    print(f"\n== {name} ==")
    print(f"{'L':>4} {'B':>2} | {'tri_bat':>8} {'tri_loop':>9} {'pytorch':>8} "
          f"| {'ms/batch':>8} {'bat/py':>7} | fwd+bwd: {'tri_bat':>8} {'pytorch':>8}")
    for L in (256, 384):
        for B in (1, 2, 4, 8):
            x = torch.randn(B, L, L, 128, device=DEVICE, dtype=BF16, requires_grad=True)
            mask = torch.ones(B, L, dtype=torch.bool, device=DEVICE)
            x1 = [torch.randn(1, L, L, 128, device=DEVICE, dtype=BF16) for _ in range(B)]
            m1 = torch.ones(1, L, dtype=torch.bool, device=DEVICE)

            def bat_fwd():
                with torch.no_grad():
                    tri(x, mask)

            def loop_fwd():
                with torch.no_grad():
                    for xb in x1:
                        tri(xb, m1)

            def py_fwd():
                with torch.no_grad():
                    py(x, mask)

            def bat_fb():
                x.grad = None
                tri(x, mask).float().sum().backward()

            def py_fb():
                xr = x.detach().clone().requires_grad_(True)
                py(xr, mask).float().sum().backward()

            tb, tl, tp = _time(bat_fwd), _time(loop_fwd), _time(py_fwd)
            fbb, fbp = _time(bat_fb), _time(py_fb)
            print(f"{L:>4} {B:>2} | {tb:8.3f} {tl:9.3f} {tp:8.3f} "
                  f"| {tb / B:8.3f} {tb / tp:7.2f} | {'':>9}{fbb:8.3f} {fbp:8.3f}")


def main():
    assert torch.cuda.is_available()
    print(f"device={torch.cuda.get_device_name(0)}  (fwd-only ms unless labelled fwd+bwd)")
    bidir_tri = _mk(BidirectionalTriangleMultiplication, IMPL.TRITON)
    bidir_py = _mk(BidirectionalTriangleMultiplication, IMPL.PYTORCH)
    bench_path("bidirectional", bidir_tri, bidir_py, {})
    uni_tri = _mk(TriangleMultiplication, IMPL.TRITON, outgoing=True)
    uni_py = _mk(TriangleMultiplication, IMPL.PYTORCH, outgoing=True)
    bench_path("unidirectional (outgoing)", uni_tri, uni_py, {})
    print("\nSPEED DONE")


if __name__ == "__main__":
    main()
