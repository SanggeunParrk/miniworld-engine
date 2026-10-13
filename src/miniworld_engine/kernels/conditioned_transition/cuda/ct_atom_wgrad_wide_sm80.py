"""Own CUDA BF16 transposed weight-gradient GEMM, FP32 output, explicit splitK."""

import functools
from pathlib import Path
import torch
from ..._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    d = Path(__file__).parent
    return load_extension(
        name="ct_atom_wgrad_wide_sm80",
        sources=[str(d / "ct_atom_wgrad_wide_sm80.cu")],
        extra_include_paths=[str(d)],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "-Xptxas=-v",
            "-gencode=arch=compute_80,code=sm_80",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=True,
    )


def run(a, b, partial, out, cfg=0, splits=32):
    _ext().run(a, b, partial, out, cfg, splits)


def result(a, b):
    """Qualified physical dWs geometry: BF16 K=48*8192, unrounded FP32."""
    if a.shape != (48 * 8192, 128) or b.shape != (48 * 8192, 256):
        raise ValueError("qualified atom dWs physical dimensions required")
    partial = torch.empty((32, 128, 256), dtype=torch.float32, device=a.device)
    out = torch.empty((128, 256), dtype=torch.float32, device=a.device)
    run(a, b, partial, out, 8, 32)
    return out
