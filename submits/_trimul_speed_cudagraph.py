"""Speed of the B>1 triton trimul UNDER CUDA GRAPHS (the deployment regime; matches
bench.py cudagraph=manual). Eager numbers include host/alloc/dispatch overhead that
graphs remove — so batched-vs-loop must be judged here, not eagerly.

Captures each callable in a per-shape CUDA graph (side-stream warmup -> capture, exactly
bench.py's capture_cudagraph), then times replay. Inference forward (no_grad), the regime
the module's _*_infer path is designed for. batched (1 call on B) vs loop (B calls on B=1)
vs pytorch. A100, bf16, d=128.
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


def _mk(cls, impl, **kw):
    m = cls(128, implementation=impl, **kw)
    torch.manual_seed(0)
    for p in m.parameters():
        p.data.normal_(0, 0.5)
    return m.to(DEVICE, BF16)


def _capture(step, warmup=8):
    """side-stream warmup -> capture, mirroring bench.py capture_cudagraph (inference)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            with torch.no_grad():
                step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.no_grad():
        step()
    return g


def _time_replay(g, n=50):
    for _ in range(8):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def _prewarm(fn, n=6):
    """Main-stream eager warmup to trigger torch.compile tracing + autotune BEFORE the
    side-stream cudagraph capture (compiling inside capture warmup is fragile)."""
    with torch.no_grad():
        for _ in range(n):
            fn()
    torch.cuda.synchronize()


def bench(name, tri, py):
    # pytorch baseline COMPILED (matches bench.py compile=true, which does ref_model.compile()).
    # The triton path is @torch.compiler.disable so compile is a no-op there -> tri left eager.
    py = torch.compile(py)
    print(f"\n== {name} (CUDA-GRAPH replay, fwd inference; pytorch=torch.compile'd) ==")
    print(f"{'L':>4} {'B':>2} | {'tri_bat':>8} {'tri_loop':>9} {'py_comp':>8} "
          f"| {'ms/batch':>8} {'bat/py':>7} {'bat/loop':>8}")
    for L in (256, 384):
        for B in (1, 2, 4, 8):
            x = torch.randn(B, L, L, 128, device=DEVICE, dtype=BF16)
            mask = torch.ones(B, L, dtype=torch.bool, device=DEVICE)
            x1 = [torch.randn(1, L, L, 128, device=DEVICE, dtype=BF16) for _ in range(B)]
            m1 = torch.ones(1, L, dtype=torch.bool, device=DEVICE)

            _prewarm(lambda: py(x, mask))     # force compile trace before capture
            g_bat = _capture(lambda: tri(x, mask))
            g_loop = _capture(lambda: [tri(xb, m1) for xb in x1])
            g_py = _capture(lambda: py(x, mask))
            tb, tl, tp = _time_replay(g_bat), _time_replay(g_loop), _time_replay(g_py)
            print(f"{L:>4} {B:>2} | {tb:8.3f} {tl:9.3f} {tp:8.3f} "
                  f"| {tb / B:8.3f} {tb / tp:7.2f} {tb / tl:8.2f}")
            del g_bat, g_loop, g_py
            torch.cuda.empty_cache()


def main():
    assert torch.cuda.is_available()
    print(f"device={torch.cuda.get_device_name(0)}")
    bench("bidirectional",
          _mk(BidirectionalTriangleMultiplication, ImplementationType.TRITON),
          _mk(BidirectionalTriangleMultiplication, ImplementationType.PYTORCH))
    bench("unidirectional (outgoing)",
          _mk(TriangleMultiplication, ImplementationType.TRITON, outgoing=True),
          _mk(TriangleMultiplication, ImplementationType.PYTORCH, outgoing=True))
    print("\nSPEED CUDAGRAPH DONE")


if __name__ == "__main__":
    main()
