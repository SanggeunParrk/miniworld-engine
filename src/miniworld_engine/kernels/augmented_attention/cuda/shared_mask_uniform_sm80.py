"""Own CUDA correction for shared all-masked softmax and score gradients."""

import functools
from pathlib import Path
from ..._nvcc import ensure_cuda_home, host_flags
from miniworld_engine.kernels._sm120 import sm80_load_extension as load_extension  # sm_120a: same sources (kernels/_sm120.py)


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    directory = Path(__file__).parent / "sm80"
    return load_extension(
        name="augattn_shared_mask_uniform_sm80",
        sources=[str(directory / "shared_mask_uniform.cu")],
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


def forward(mask, value, gate, out, lse, length, padded, samples, heads):
    _ext().fwd(
        mask.contiguous(),
        value,
        gate,
        out,
        lse,
        length,
        padded,
        samples,
        heads,
        value.shape[1],
    )


def backward(mask, dout, dq, dk, dv, db, length, padded, samples, heads):
    _ext().bwd(
        mask.contiguous(),
        dout,
        dq,
        dk,
        dv,
        db,
        length,
        padded,
        samples,
        heads,
        dout.shape[1],
    )
