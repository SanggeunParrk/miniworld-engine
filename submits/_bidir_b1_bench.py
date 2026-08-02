"""B=1 fwd/bwd latency for the TRITON bidirectional trimul — the A/B regression probe.

Run once on the working tree (with the B>1 generalisation) and once on HEAD (pre-change,
via `git stash`); the driver sbatch diffs the two. B=1 is the production path on A100
(cute is sm90+), and the generalised kernel must not regress it. A `TAG` env var labels
the output line so the driver can grep both runs.
"""

from __future__ import annotations

import os
import statistics
import time

import torch

from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_multiplication import (
    BidirectionalTriangleMultiplication,
)

DEVICE, BF16, TAG = "cuda", torch.bfloat16, os.environ.get("TAG", "run")


def _mod(d_pair: int):
    m = BidirectionalTriangleMultiplication(d_pair, implementation=ImplementationType.TRITON)
    torch.manual_seed(0)
    for p in m.parameters():
        p.data.normal_(0, 0.5)
    return m.to(DEVICE, BF16)


def _time(fn, n=50, warmup=10) -> float:
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


def main() -> None:
    assert torch.cuda.is_available()
    d = 128
    mod = _mod(d)
    mask = None
    for L in (256, 384, 512):
        pair = torch.randn(1, L, L, d, device=DEVICE, dtype=BF16, requires_grad=True)

        def fwd():
            with torch.no_grad():
                mod(pair, mask)

        def fwdbwd():
            pair.grad = None
            mod(pair, mask).float().sum().backward()

        f = _time(fwd)
        fb = _time(fwdbwd)
        print(f"BENCH tag={TAG} L={L} fwd={f:.4f}ms fwd+bwd={fb:.4f}ms")


if __name__ == "__main__":
    main()
