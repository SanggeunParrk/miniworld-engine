"""Own CUDA row/column dropout candidate with graph-aware Philox state."""

import functools
from pathlib import Path
import torch
from miniworld_engine import settings
from ..._compile import opaque, device_constant
from ..._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    return load_extension(
        name="dropout_row_scale_sm80",
        sources=[str(Path(__file__).with_suffix(".cu"))],
        extra_cuda_cflags=[
            *host_flags(),
            "-std=c++17",
            "-O3",
            "-gencode=arch=compute_80,code=sm_80",
        ],
        extra_cflags=["-std=c++17", "-O3"],
        verbose=False,
    )


@device_constant
def _ready():
    _ext()
    return True


def supports(reference):
    return (
        settings.current().dropout_scale_sm80
        and reference.is_cuda
        and reference.dtype in (torch.bfloat16, torch.float32)
        and torch.cuda.get_device_capability(reference.device) == (8, 0)
        and _ready()
    )


def _scale_fake(reference, probability, dimension):
    """Fresh tensor shaped like ``reference`` with extent 1 along ``dimension``."""
    shape = list(reference.shape)
    shape[dimension] = 1
    return reference.new_empty(shape)


@opaque(
    fake=_scale_fake,
    name="triangle_attention_dropout_row_scale_sm80",
    tags=(torch.Tag.nondeterministic_seeded,),
)
def _scale(reference: torch.Tensor, probability: float, dimension: int) -> torch.Tensor:
    """Row/column dropout keep-scale (0 or 1/(1-p)) from graph-aware Philox state, one value per slice along ``dimension``."""
    return _ext().row_scale(reference, probability, dimension)


def scale(reference, probability, dimension):
    with torch.no_grad():
        return _scale(reference, probability, dimension)
