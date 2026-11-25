"""A100 shared Token conditioning gate; called inside the opaque forward."""

import functools
from pathlib import Path
from torch.utils.cpp_extension import load


@functools.lru_cache(maxsize=1)
def ext():
    return load(
        name="miniworld_token_shared_scale_sm80_v1",
        sources=[str(Path(__file__).with_suffix(".cu"))],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "-gencode=arch=compute_80,code=sm_80",
        ],
        extra_cflags=["-O3"],
        extra_include_paths=[str(Path(__file__).parent)],
        verbose=False,
    )


def forward(y, gate, residual, out, length, padded):
    empty = y.new_empty((0,)) if residual is None else residual
    ext().shared_res_gate(y, gate, empty, out, length, padded)
