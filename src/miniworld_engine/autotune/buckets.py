"""Autotune-key bucketing by the kernel's actual row count M — bucketed by SIZE.

A kernel's optimal config depends on the real number of rows M it iterates, and nothing
else: two launches with the same M tile the same way whether that M arose as a linear
length N or as a pair count L*L. So we never inspect *which* caller produced M — we bucket
the raw M against edges placed at the scales a kernel runs at:

* SQUARED edges — pair L×L work (M ~ L²): 128²…512². 512² is a SATURATING cap: any L>512
  reuses the 512² config (these saturate early), so no distinct >512² bucket is ever tuned.
* LINEAR  edges — 1st-order work (M ~ N): 128…2048 (2048 likewise saturates for larger M).
* MIXED (transition / layernorm / gated_projection) run on BOTH, so bucket M against the
  UNION of the two edge sets — a small M lands among the linear edges, a large M among the
  squared edges, purely by magnitude.

Each kernel is registered to whichever family matches its M range; the cache is per-op, so
the bucket indices never collide across kernels.
"""

from __future__ import annotations

SQUARED_EDGES = tuple(e * e for e in (128, 256, 384, 512))   # 16384, 65536, 147456, 262144
LINEAR_EDGES = (128, 256, 384, 512, 768, 1024, 1536, 2048)
# union of both scales, for kernels whose M is sometimes ~N and sometimes ~L²
COMBINED_EDGES = tuple(sorted(set(LINEAR_EDGES + SQUARED_EDGES)))


def _idx(m: int, edges: tuple[int, ...]) -> int:
    # The top edge is a SATURATING ceiling, not a threshold with an extra overflow bucket:
    # any M above the largest edge reuses the largest edge's config (configs stop changing
    # once M is big enough — "saturate early"). So bucket_squared caps at 512² — L>512 pair
    # work reuses the 512² config instead of tuning a distinct >512² bucket.
    for i, e in enumerate(edges):
        if m <= e:
            return i
    return len(edges) - 1


def bucket_squared(pair_rows: int) -> int:
    """Pair L×L kernel that passes M = L*L (row count)."""
    return _idx(pair_rows, SQUARED_EDGES)




def bucket_linear(n: int) -> int:
    """1st-order kernel whose row count scales ~ N."""
    return _idx(n, LINEAR_EDGES)


def bucket_mixed(rows: int) -> int:
    """Kernel run on both single (M~N) and pair (M~L²) tensors: bucket raw M by size."""
    return _idx(rows, COMBINED_EDGES)
