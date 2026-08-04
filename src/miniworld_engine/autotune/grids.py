"""Full brute-force autotune grids (Triton). ONE place defines the exhaustive sweep.

Triton config choice is performance-only — it never changes numerics (unlike the CuTe /
CUDA GEMM configs, where cluster/pingpong/tile choices can be numerically wrong). So the
autotuner may explore the ENTIRE launchable grid: the device-shared-memory prune
(``make_device_smem_prune``) drops configs that cannot launch, so an over-wide grid here
costs only tuning time, never correctness or a crash. Widen the ranges below to broaden
the search for every kernel at once.
"""

from __future__ import annotations

import itertools

import triton

# Comprehensive per-dimension candidate sets (powers of two spanning the range these
# kernels actually run). Unlaunchable combinations are pruned at runtime by smem, so these
# are intentionally wide — "no assumptions" beyond hardware-representable tile sizes.
BLOCK_M = (16, 32, 64, 128, 256)
BLOCK_N = (16, 32, 64, 128, 256)
BLOCK_K = (16, 32, 64, 128)
BLOCK_1D = (64, 128, 256, 512, 1024, 2048)
WARPS = (1, 2, 4, 8, 16)
STAGES = (1, 2, 3, 4, 5, 6)


def brute(block_dims: dict[str, tuple[int, ...]],
          warps: tuple[int, ...] = WARPS,
          stages: tuple[int, ...] = STAGES) -> list:
    """Full cartesian product of the given block dims x num_warps x num_stages.

    ``block_dims`` maps each kernel-specific constexpr name (``BLOCK_M`` / ``BM`` /
    ``BLK`` / ...) to its candidate tuple. Example::

        configs = brute({"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N})            # 2-D tile
        configs = brute({"BM": BLOCK_M, "BK": BLOCK_K, "BN": BLOCK_N})       # 3-D GEMM tile
    """
    names = list(block_dims)
    value_lists = [block_dims[n] for n in names]
    out = []
    for combo in itertools.product(*value_lists):
        kw = dict(zip(names, combo))
        for w, s in itertools.product(warps, stages):
            out.append(triton.Config(kw, num_warps=w, num_stages=s))
    return out
