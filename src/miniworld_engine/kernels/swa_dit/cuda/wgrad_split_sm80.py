"""Own CUDA BF16 transposed weight-gradient GEMM, FP32 output, explicit splitK."""

import functools
from pathlib import Path
import torch
from ..._nvcc import ensure_cuda_home, host_flags
from miniworld_engine.kernels._sm120 import sm80_load_extension as load_extension  # sm_120a: same sources (kernels/_sm120.py)


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    d = Path(__file__).parent / "sm80"
    return load_extension(
        name="swa_wgrad_split_sm80",
        sources=[str(d / "wgrad_split.cu")],
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


def down_weight_gradient(dffn: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """Unrounded FP32 dWd for the two frozen A100 training row counts."""
    partial = torch.empty((32, 128, 256), device=dffn.device, dtype=torch.float32)
    out = torch.empty((128, 256), device=dffn.device, dtype=torch.float32)
    run(dffn, hidden, partial, out, cfg=3, splits=32)
    return out
