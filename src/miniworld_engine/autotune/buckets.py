"""Autotune-key bucketing by the kernel's actual row count M — bucketed by SIZE.

A kernel's optimal config depends on the real number of rows M it iterates, and nothing
else: two launches with the same M tile the same way whether that M arose as a linear
length N or as a pair count L*L. So we never inspect *which* caller produced M — we bucket
the raw M against edges placed at the scales a kernel runs at:

* SQUARED edges — pair L×L work (M ~ L²): 128²…512² (capped at 512²; these saturate early).
* LINEAR  edges — 1st-order work (M ~ N): 128…2048.
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
    for i, e in enumerate(edges):
        if m <= e:
            return i
    return len(edges)


def bucket_squared(pair_rows: int) -> int:
    """Pair L×L kernel that passes M = L*L (row count)."""
    return _idx(pair_rows, SQUARED_EDGES)




def bucket_linear(n: int) -> int:
    """1st-order kernel whose row count scales ~ N."""
    return _idx(n, LINEAR_EDGES)


def bucket_mixed(rows: int) -> int:
    """Kernel run on both single (M~N) and pair (M~L²) tensors: bucket raw M by size."""
    return _idx(rows, COMBINED_EDGES)


def elem_bucket_of(*names: str):
    """``bucket_of`` for a flat 1-D elementwise kernel keyed on a raw element count.

    ``names`` are the kernel args whose PRODUCT is that count — one name when the kernel already
    takes ``n_elem``, several (``"M", "ND"``) when it takes the extents instead.

    A flat EW kernel has no shape-defining constexpr to bucket on — the only thing that moves
    its best BLOCK is the total element count, which is continuous. ``key_bucket_of()`` with no
    keys builds the SAME empty bucket for every shape (one tuned config shared by a 256-element
    launch and a 64M-element one); the raw count is the other extreme (a bucket per exact value,
    missing on every shape the build did not visit). Bucket it against the canonical mixed edges,
    which is what ``bias_only``'s hand-rolled ``_sigmul_bucket`` already did — this is that, once,
    for every EW kernel instead of copied per file.
    """
    def f(named_args, kwargs):
        from .cache import shape_bucket

        get = (lambda k: named_args.get(k, kwargs.get(k))) if hasattr(named_args, "get") \
            else (lambda k: kwargs.get(k))
        n = 1
        for k in names:
            v = get(k)
            if v is None:
                return shape_bucket(NE=0)
            n *= int(v)
        return shape_bucket(NE=bucket_mixed(n))

    f._miniworld_keys = tuple(names)  # noqa: SLF001 -- lets build.audit introspect the extractor
    return f
