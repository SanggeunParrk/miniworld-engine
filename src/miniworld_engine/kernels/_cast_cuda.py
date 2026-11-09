"""fp32 / bf16 casts of up to twelve tensors in one CUDA launch (``_cast_cuda.cu``): the one implementation behind the H100 launchers'
weight casts (Transition, TriMul, PWA, ConditionedTransition, AttentionPairBias, the fp32-master projections). Round to nearest even, so
a cast here is bit-identical to ``tensor.to(dtype)``. The portable (Triton) paths keep ``kernels/_cast.py``.

``cast(tensors, dtypes, force=False)``: one output per input, of the dtype asked (``torch.bfloat16`` / ``torch.float32``); an input that
already is that dtype and contiguous comes back as it is unless ``force``. ``cast_into(srcs, dsts)``: into preallocated contiguous
destinations (row blocks of one packed buffer); a source may be a strided 2-D matrix.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_TYPES = {torch.bfloat16: 0, torch.float32: 1}


@functools.lru_cache(maxsize=1)
def _ext():
    from miniworld_engine.kernels._nvcc import (
        ensure_cuda_home,
        gencodes,
        host_flags,
        load_extension,
    )

    ensure_cuda_home()
    return load_extension(
        name="weights_cast_cuda_v1",
        sources=[str(Path(__file__).with_name("_cast_cuda.cu"))],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("80", "90", "100")],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def _fake(tensors) -> bool:
    """FakeTensors (the derivation, ``torch.compile`` tracing): the cast is written as the torch ops it is bit-identical to, so no extension
    is built or launched on a tensor that has no storage (a launch there read a null pointer: a sticky illegal address)."""
    from torch._subclasses.fake_tensor import is_fake

    return any(is_fake(t) for t in tensors)


def cast(tensors, dtypes, force: bool = False) -> list[torch.Tensor]:
    """``[t.to(d).contiguous() for t, d in zip(tensors, dtypes)]`` in one launch (an input already of its dtype is returned as it is)."""
    tensors = list(tensors)
    if _fake(tensors):
        return [t.to(d).contiguous().clone() if force or t.dtype != d or not t.is_contiguous() else t for t, d in zip(tensors, dtypes, strict=True)]
    return list(_ext().cast(tensors, [_TYPES[d] for d in dtypes], force))


def cast_into(srcs, dsts) -> None:
    """``d.copy_(s)`` for each pair in one launch: fp32 / bf16 either way, contiguous destinations."""
    srcs, dsts = list(srcs), list(dsts)
    if _fake([*srcs, *dsts]):
        for s, d in zip(srcs, dsts, strict=True):
            d.copy_(s)
        return
    _ext().cast_into(srcs, dsts)


__all__ = ["cast", "cast_into"]


def prebuild(target, rows):
    """The build of this module (kernels/_prebuild.py): one multi-arch extension (the dtype is a launch argument, not a build
    switch), built when a requirement row needs a cast."""
    if rows:
        yield _ext
