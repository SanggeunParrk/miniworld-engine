"""Own-CUDA split-K weight-gradient candidate, not a production dispatch."""

import functools
from pathlib import Path
import torch
from ..._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    directory = Path(__file__).parent / "sm80"
    return load_extension(
        name="transition_weight_grad_splitk_sm80",
        sources=[str(directory / "weight_grad_splitk.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-std=c++17",
            "-O3",
            "-gencode=arch=compute_80,code=sm_80",
            f"-I{directory}",
        ],
        extra_cflags=["-std=c++17", "-O3"],
        verbose=False,
    )


def run(a, b, part, out, replicas=16, cfg=0):
    _ext().run(a, b, part, out, replicas, cfg)


def result(a, b, replicas=16, cfg=0, dtype=torch.bfloat16):
    part = torch.empty(
        (replicas, a.shape[1], b.shape[1]), device=a.device, dtype=torch.float32
    )
    out = torch.empty((a.shape[1], b.shape[1]), device=a.device, dtype=dtype)
    run(a, b, part, out, replicas, cfg)
    return out
