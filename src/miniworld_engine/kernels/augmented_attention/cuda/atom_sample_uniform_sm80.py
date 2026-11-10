"""Private E326 per-sample fullymasked correction; no shared/default change."""

import functools
from pathlib import Path
from ..._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    directory = Path(__file__).parent / "sm80"
    return load_extension(
        name="dense_atom_sample_uniform326_sm80",
        sources=[str(directory / "atom_sample_uniform.cu")],
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


def backward(mask, dout, dq, dk, dv, length, padded, samples, heads):
    _ext().bwd(
        mask.contiguous(),
        dout,
        dq,
        dk,
        dv,
        length,
        padded,
        samples,
        heads,
        dout.shape[1],
    )
