"""Own CUDA 128x128 output-weight gradient with BF16 epilogue; unconnected candidate."""

import functools, torch
from pathlib import Path
from ..._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    d = Path(__file__).parent / "sm80"
    return load_extension(
        name="triattn_native_wo_tn_sm80",
        sources=[str(d / "native_wo_tn.cu")],
        extra_include_paths=[str(d)],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "-gencode=arch=compute_80,code=sm_80",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def run(a, b, partial, out, cfg=4, chunk=8192):
    _ext().run(a, b, partial, out, cfg, chunk)


def matmul(a, b, cfg=4, chunk=8192):
    out = torch.empty((a.shape[1], b.shape[1]), device=a.device, dtype=torch.bfloat16)
    partial = torch.empty(
        ((a.shape[0] + chunk - 1) // chunk, *out.shape),
        device=a.device,
        dtype=torch.float32,
    )
    run(a, b, partial, out, cfg, chunk)
    return out
