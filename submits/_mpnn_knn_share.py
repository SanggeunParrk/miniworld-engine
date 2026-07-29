"""How much of a training step is the neighbour search, and could it be hoisted out?

The neighbour graph is a function of the input coordinates alone. With
``coordinate_grad`` off nothing flows back through it, so it is preprocessing that
happens to live inside the forward. This measures what moving it out would be worth:
the search on its own, against the whole features module, at the real packed size.

  python submits/_mpnn_knn_share.py [seq_len] [batch]
"""

from __future__ import annotations

import sys

import torch

from miniworld_kernels.modules.mpnn import BackboneFeatures

DEVICE = "cuda"


def timed(fn, warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def main() -> int:
    length = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    total = batch * length
    print(f"{torch.cuda.get_device_name(0)}  packed B={batch} T={length} "
          f"-> {total:,} nodes")

    torch.manual_seed(0)
    lengths = torch.full((batch,), length, dtype=torch.long, device=DEVICE)
    coordinates = torch.randn(1, total, 3, device=DEVICE) * 6.0
    mask = torch.ones(1, total, device=DEVICE)

    print(f"\n  {'knn backend':<14}{'search ms':>12}{'vs cdist':>11}")
    baseline = None
    for backend in ("cdist", "segment"):
        features = BackboneFeatures(
            edge_width=128, num_rbf=16, k_neighbors=48,
            coordinate_noise=0.0, knn_backend=backend,
        ).to(DEVICE)
        features.eval()

        def search(f=features):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                return f._nearest_neighbors(coordinates, mask, lengths)

        try:
            ms = timed(search)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  {backend:<14}{'OOM':>12}")
            continue
        baseline = baseline or ms
        print(f"  {backend:<14}{ms:>12.2f}{baseline / ms:>10.2f}x")
        del features
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
