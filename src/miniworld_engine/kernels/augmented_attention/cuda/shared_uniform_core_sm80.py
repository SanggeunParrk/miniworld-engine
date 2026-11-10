"""Experimental shared-mask classification fused into the BF16 attention core."""

import functools, math
from pathlib import Path
import torch
from ..._nvcc import ensure_cuda_home, host_flags, load_extension
from .sm80 import _fwd_schedule


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    directory = Path(__file__).parent / "sm80_shared_uniform"
    return load_extension(
        name="augmented_attention_shared_uniform_sm80",
        sources=[str(directory / "ops.cu")],
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


def forward(
    q,
    k,
    v,
    g,
    bias,
    mask,
    length,
    padded,
    samples,
    heads,
    head_dim,
    save=False,
    out=None,
    lse=None,
):
    mode = 2 if save else 3
    if out is None:
        out = torch.empty(
            samples * padded, heads * head_dim, device=q.device, dtype=torch.bfloat16
        )
    if lse is None:
        lse = torch.empty(
            (samples, heads, padded) if save else (0,),
            device=q.device,
            dtype=torch.float32,
        )
    empty = torch.empty(0)
    _ext().attn_fwd_shared(
        q,
        k,
        v,
        bias,
        q if g is None else g,
        out,
        lse,
        samples,
        padded,
        heads,
        head_dim,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        (q if g is None else g).stride(0),
        out.stride(0),
        1.4426950408889634 / math.sqrt(head_dim),
        1.4426950408889634,
        mode,
        *_fwd_schedule(samples),
        0,
        empty,
        0,
        mask.contiguous(),
        length,
    )
    return out, lse
