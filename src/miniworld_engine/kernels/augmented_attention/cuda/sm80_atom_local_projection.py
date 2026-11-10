"""Source-owned eight FP32 projection operand RN casts for A100 inference."""

import functools
from pathlib import Path
import torch
import torch.nn.functional as F
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def ext():
    ensure_cuda_home()
    return load_extension(
        name="af3_local_projection_cast_sm80",
        sources=[str(Path(__file__).with_suffix(".cu"))],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "-gencode=arch=compute_80,code=sm_80",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=True,
    )


def _cast_fake(parameters):
    """One fresh bf16 tensor per parameter, same shapes."""
    return [torch.empty_like(p, dtype=torch.bfloat16) for p in parameters]


@opaque(fake=_cast_fake, name="augmented_attention_local_projection_cast_sm80")
def _cast(parameters: list[torch.Tensor]) -> list[torch.Tensor]:
    """Batched fp32 -> bf16 cast of the eight AF3 local attention projection parameters; every result is a fresh tensor."""
    assert len(parameters) == 8 and all((p.dtype is torch.float32 for p in parameters))
    result = list(ext().result(parameters))
    assert all(
        (a.data_ptr() != b.data_ptr() for a, b in zip(parameters, result, strict=True))
    )
    return result


def projection_parameters(att):
    parameters = [
        att.to_query.weight,
        att.to_query.bias,
        att.to_key.weight,
        att.to_value.weight,
        att.to_gate.weight,
        att.to_out.weight,
        att.to_scale.weight,
        att.to_scale.bias,
    ]
    assert len(parameters) == 8
    if parameters[0].dtype is torch.float32:
        assert all((p.dtype is torch.float32 for p in parameters))
        return _cast(parameters)
    assert all((p.dtype is torch.bfloat16 for p in parameters))
    return parameters
