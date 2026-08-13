"""Per-kernel autotuning DISPATCH cache: pick the fastest implementation VARIANT per problem
shape and remember it (like the layernorm atomic-vs-partial dispatch, but across backends —
cuBLAS / quack / triton). Some trimul ops have >1 valid impl whose winner flips with shape
(e.g. the input-grad GEMM: cuBLAS wins small L, quack ≈ cuBLAS at large L; the gate: triton vs
the fused-quack epilogue). Always-one-backend leaves the small-L (or large-L) case slow; this
benchmarks the candidates ONCE per shape (first call) and caches the winner.

`pick(name, key, candidates)`:
  candidates = [(label, thunk), ...]  where thunk() runs that variant and returns its result.
  On a cache MISS for `key`: time every thunk (do_bench), cache the fastest label, return its
  result. On a HIT: just run the cached winner. Pure thunks only (no input mutation) — the
  losers' outputs are discarded. In-process cache (amortized over a training/inference run).

Env: TRIMUL_DISPATCH=0 disables (always uses candidate[0]); TRIMUL_DISPATCH_LOG=1 prints picks.
"""

from __future__ import annotations

import os

import torch
import triton
from miniworld_engine import settings

_CACHE: dict[str, dict] = {}
_ENABLED = settings.current().trimul_cute_dispatch
_LOG = settings.current().trimul_dispatch_log


def pick(name, key, candidates):
    """Run the fastest of `candidates` for `key`, caching the choice. candidates: list of
    (label, thunk()->result)."""
    if not _ENABLED or len(candidates) == 1:
        return candidates[0][1]()
    cache = _CACHE.setdefault(name, {})
    idx = cache.get(key)
    if idx is None:
        best_i, best_t = 0, float("inf")
        for i, (label, thunk) in enumerate(candidates):
            try:
                for _ in range(3):       # warm (compile/autotune the variant)
                    thunk()
                t = triton.testing.do_bench(thunk, warmup=10, rep=30, return_mode="median")
            except Exception:            # noqa: BLE001 — a variant may not support this shape
                t = float("inf")
            if t < best_t:
                best_t, best_i = t, i
        cache[key] = best_i
        idx = best_i
        if _LOG:
            print(f"[dispatch] {name} key={key} -> {candidates[idx][0]} "
                  f"({best_t:.4f} ms)", flush=True)
    return candidates[idx][1]()


def reset():
    """Clear the dispatch cache (e.g. between bench configs)."""
    _CACHE.clear()


# --- GEMM/bmm primitives that dispatch cuBLAS vs quack (cute) per shape ----------------------
# Generic so EVERY matmul in the pipeline can autotune its backend. quack candidates lazy-import
# and are guarded by pick()'s try/except — a shape quack can't take (e.g. odd strides) just
# scores inf and cuBLAS is chosen. dW huge-K reductions reliably pick cuBLAS; M-major input-grad
# GEMMs can pick quack. Keys include the operand shapes so each distinct matmul caches its own.

def mm(name, A, B):
    """A @ B, dispatched cuBLAS vs quack."""
    def _q():
        from miniworld_engine.kernels._quack_compat import gemm as qg
        return qg(A, B)
    return pick(name, (A.shape[-2], A.shape[-1], B.shape[-1]),
                [("cublas", lambda: A @ B), ("quack", _q)])


def addmm(name, C, A, B):
    """A @ B + C, dispatched cuBLAS addmm vs quack gemm_act (C-add epilogue)."""
    def _q():
        from miniworld_engine.kernels._quack_compat import gemm_act as qga
        return qga(A, B, C=C, activation=None, store_preact=False)[1]
    return pick(name, (A.shape[-2], A.shape[-1], B.shape[-1]),
                [("cublas", lambda: torch.addmm(C, A, B)), ("quack", _q)])


def bmm(name, A, B):
    """batched A @ B, dispatched cuBLAS torch.bmm vs quack batched gemm."""
    def _q():
        from miniworld_engine.kernels._quack_compat import gemm as qg
        return qg(A, B)
    return pick(name, (A.shape[0], A.shape[-2], A.shape[-1], B.shape[-1]),
                [("cublas", lambda: torch.bmm(A, B)), ("quack", _q)])
