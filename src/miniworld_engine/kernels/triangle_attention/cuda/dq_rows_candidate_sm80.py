"""Own CUDA DQ query-tile/row-group stage candidate; established module unchanged."""

import functools, torch
from pathlib import Path
from . import sm80_arch as _arch
from ..._nvcc import ensure_cuda_home, host_flags, load_extension

CONFIGS = (
    (4, 3, 32, 1, 128, True),
    (8, 3, 32, 1, 128, True),
    (8, 2, 32, 1, 64, True),
    (4, 2, 32, 1, 64, True),
    (8, 3, 32, 1, 64, True),
    (8, 2, 32, 1, 64, False),
    (8, 3, 32, 1, 128, False),
)


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    d = Path(__file__).parent / "sm80"
    return load_extension(
        name=_arch.name("triattn_dq_rows_candidate_sm80"),
        sources=[str(d / "dq_rows_candidate.cu")],
        extra_include_paths=[str(d)],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-lineinfo",
            *_arch.gencode(),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def run(q, k, v, dov, bias, lse, delta, dq, partial, db, cfg=0):
    z, L, _, c = q.shape
    h = c // 32
    r, nkv, bn, minb, bm, cached = CONFIGS[cfg]
    _ext().run(
        q,
        k,
        v,
        dov,
        bias,
        lse,
        delta,
        dq,
        partial,
        L,
        h,
        z,
        q.stride(2),
        k.stride(2),
        v.stride(2),
        dov.stride(2),
        dq.stride(2),
        1.4426950408889634 / (32**0.5),
        1 / (32**0.5),
        r,
        nkv,
        bn,
        minb,
        bm,
        cached,
    )
    _ext().reduce(partial, db, L // r)
