"""The hand-written CUDA kernels of the fused parallel residual (``modules/residual.py``, ``parallel_residual.cu``): ONE source built as
``parallel_residual_sm90`` (H100, sm_90a), ``parallel_residual_sm80`` (A100, sm_80), ``parallel_residual_sm120`` (RTX 5090, sm_120a)
and ``parallel_residual_sm100`` (B200, sm_100a); the dtype and the branch count are template instances of one extension, the sizes launch
arguments (no shape-dependent build). Launched on torch's current stream (CUDA-graph capturable); built ahead of time by the
prebuild (:func:`prebuild`), never at import."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from foldjet_engine.kernels._compile import opaque

_dir = Path(__file__).parent
_ARCH = {
    "sm90": "90a",
    "sm80": "80",
    "sm120": "120a",
    "sm100": "100a",
}  # sm120: the GeForce RTX 5090 (sm_120a); sm100: the B200 (sm_100a, 2026-11-16); the same source
# A100: plain loads and stores (PR_HINT=1): the streaming-hinted ones (ld.global.cs / st.global.cs, the H100 default) run 5-8 % under the Triton kernel there (1.60 vs 1.71 TB/s of 1.94)
# H100: read-only loads (PR_HINT=2: ld.global.nc, plain stores): the streaming-hinted ones ran 3-4 % under the Triton kernel (msa d 64 L 2048: 0.363 vs 0.350 ms;
# pair d 256 L 2048 n 2: 2.827 vs 2.728 ms); nc loads / plain loads equal it (bit-identical outputs; 2 a hair ahead at small sizes). sm120 keeps the hinted default.
# sm100 (B200): its own source (parallel_residual_sm100.cu), default hints (no PR_HINT flag)
# sm120 (RTX 5090, 2026-11-22): PR_UNROLL=4 (4 vectors a thread and tensor; the default 1 left the small calls under the Triton kernel: pair d 256 L 128, 3 branches 30.7 vs 28.7 us, 2 branches 24.9 vs 24.6 us -> 28.7 / 22.5 us)
_FLAGS = {"sm80": ("-DPR_HINT=1",), "sm90": ("-DPR_HINT=2",), "sm120": ("-DPR_UNROLL=8", "-DPR_HINT=2", "-DPR_UNROLL3=4")}


@functools.lru_cache(maxsize=None)
def ext(arch: str):
    """The extension of ``arch`` (``"sm90"`` / ``"sm80"`` / ``"sm120"`` / ``"sm100"``)."""
    from foldjet_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name=f"parallel_residual_{arch}",
        # B200: parallel_residual_sm100.cu (the shared source; tensors past the L2 read with cached loads, measured on the B200)
        sources=[str(_dir / ("parallel_residual_sm100.cu" if arch == "sm100" else "parallel_residual.cu"))],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes(_ARCH[arch]), *_FLAGS.get(arch, ())],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def _device_ext(device: torch.device):
    return ext({(8, 0): "sm80", (12, 0): "sm120", (10, 0): "sm100"}.get(torch.cuda.get_device_capability(device), "sm90"))


def _forward_fake(x: torch.Tensor, branches: list[torch.Tensor]) -> torch.Tensor:
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def _backward_fake(g: torch.Tensor, n: int, copies: int = 0) -> list[torch.Tensor]:
    return [torch.empty_like(g, memory_format=torch.contiguous_format) for _ in range(1 + copies)]


# opaque to torch.compile (kernels/_compile.py): a PairformerBlock(parallel=True) calls the forward inside its (compiled) block
@opaque(fake=_forward_fake, name="parallel_residual_cuda_forward")
def forward(x: torch.Tensor, branches: list[torch.Tensor]) -> torch.Tensor:
    """``x + sum_i (y_i - x)`` for 1 .. 5 contiguous tensors of ``x``'s shape, dtype and device (the module checks)."""
    return _device_ext(x.device).forward(x, branches)


@opaque(fake=_backward_fake, name="parallel_residual_cuda_backward")
def backward(g: torch.Tensor, n: int, copies: int = 0) -> list[torch.Tensor]:
    """``[(1 - n) g, g, ..., g]`` in one pass: the gradient of ``x`` (fp32 product, one rounding to ``g``'s dtype) and ``copies`` fresh tensors
    equal to ``g`` (the gradients of the branches that are autograd leaves: ``modules/residual.py``)."""
    return _device_ext(g.device).backward(g, int(n), int(copies))


def prebuild(target, rows):
    """The build of the ``parallel_residual`` family (kernels/_prebuild.py): the target card's extension, once, when a requirements row
    (of ``ParallelResidual``) needs the family."""
    if rows and target.arch in _ARCH:
        yield functools.partial(ext, target.arch)
