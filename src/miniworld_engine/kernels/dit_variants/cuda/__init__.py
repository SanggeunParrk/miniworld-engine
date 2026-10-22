"""CUDA row kernels of the DiT block variants (``rows.cu``), sm_90a. Built on first use (``load_extension``), never at import."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_dir = Path(__file__).parent


@functools.lru_cache(maxsize=None)
def ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="dit_variants_rows_sm90a",
        sources=[str(_dir / "rows.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            *gencodes("90a"),
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def ln_mod_rows(x, mods, L, eps, sig, *, xin=None):
    """xhat = LN(x) (no affine; x = xin first when given); for each (ms, mb, out) of ``mods`` (one to three): out = xhat * f(ms) + mb,
    f = sigmoid when ``sig`` else identity; ms / mb [L, D] bf16 table views (one row stride) indexed by row % L. D = 768 or 128."""
    ms, mb, out = (list(t) for t in zip(*mods, strict=True))
    ext().ln_mod_rows(xin, x, ms, mb, out, int(L), float(eps), bool(sig))


def res_rows(x, ya, ga, L, *, yt=None, gt=None, ms=None, mb=None, out_ln=None, final=None, eps=1e-5, sig=True):
    """x += sigmoid(ga) ya [+ sigmoid(gt) yt]; into ``final`` (bf16) when given, else back into x (fp32); then
    out_ln = LN(x) * f(ms) + mb when ``out_ln`` is given."""
    ext().res_rows(x, ya, ga, yt, gt, ms, mb, out_ln, final, int(L), float(eps), bool(sig))


def qk_ln_rows(q, k, wq, bq, wk, bk, eps):
    """In place: q <- LN(q) * wq + bq, k <- LN(k) * wk + bk over D (768 or 128) columns (bf16 row views; fp32 [D] affine)."""
    ext().qk_ln_rows(q, k, wq, bq, wk, bk, float(eps))


def gate_rows(o, g):
    """In place: o <- o * sigmoid(g) (bf16 row views)."""
    ext().gate_rows(o, g)


def glu_rows(ab, out, up):
    """out = silu(g) * v [* u] for ab = [g | v | u] (bf16)."""
    ext().glu_rows(ab, out, bool(up))


def ln_rows(z, out, eps=1e-5):
    """out = LN(z), no affine, z / out bf16 [rows, 128 | 256]."""
    ext().ln_rows(z, out, float(eps))


def available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def parallel_residual(x, outputs):
    """x + sum_i (y_i - x) with fp32 sums and one rounding: a parallel block's output from the residual outputs y_i = x + f_i(x) of its
    parts (each engine module owns its residual). x / y_i contiguous, one shape, bf16 or fp32; one to eight outputs."""
    return ext().parallel_residual(x, list(outputs))
