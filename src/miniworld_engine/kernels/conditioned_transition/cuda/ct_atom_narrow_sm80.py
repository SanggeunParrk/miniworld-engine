"""Own CUDA atom ConditionedTransition warp-count ablation; no default dispatch."""

import functools
from pathlib import Path
from ..._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    d = Path(__file__).parent
    a = Path(__file__).parents[2] / "adaln/cuda/sm80"
    return load_extension(
        name="ct_atom_narrow_sm80",
        sources=[str(d / "ct_atom_narrow_sm80.cu")],
        extra_include_paths=[str(d), str(a)],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "-Xptxas=-v",
            "-gencode=arch=compute_80,code=sm_80",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=True,
    )


def run(x, c, ps, wab, xa, y, cfg, eps_x=1e-5, eps_c=1e-5, residual=True):
    _ext().run(x, c, list(ps), wab, xa, y, cfg, eps_x, eps_c, residual)
