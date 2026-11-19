"""B200 (sm_100a) builds of the bias-only TriangleAttention's kernels (``kernels/integrations/bidir_triattn_h100``, which serves
:class:`~miniworld_engine.modules.triangle.BidirectionalTriangleAttention` and the one-direction ``use_self_attention=False`` form): the
width-generic glue (``h100_sources/triattn_generic_sm90.cu``: LayerNorm forward / backward, the gate) and the gate's backward on channel
planes (``h100_sources/triattn_bidir_sm90.cu``), each source compiled unchanged for sm_100a (plain CUDA-core kernels, as the A100 /
RTX 5090 builds in ``sm80_bidir``: the sm_90-only helpers of ``sm90.cuh`` it includes are never called). Built on first use."""

from __future__ import annotations

import functools
from pathlib import Path

from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

_SRC = Path(__file__).parent / "h100_sources"
_FLAGS = ("-std=c++17", "-O3", "--use_fast_math", "-DNDEBUG")


@functools.lru_cache(maxsize=None)
def glue():
    ensure_cuda_home()
    return load_extension(
        name="triattn_generic_sm100",
        sources=[str(_SRC / "triattn_generic_sm90.cu")],
        extra_cuda_cflags=[*host_flags(), *_FLAGS, *gencodes("100a")],
        verbose=False,
    )


@functools.lru_cache(maxsize=None)
def ext():
    ensure_cuda_home()
    return load_extension(
        name="triattn_bidir_sm100",
        sources=[str(_SRC / "triattn_bidir_sm90.cu")],
        extra_cuda_cflags=[*host_flags(), *_FLAGS, *gencodes("100a")],
        verbose=False,
    )


@functools.lru_cache(maxsize=None)
def bo():
    """The B200-native kernels of the bias-only path (``b200_sources/bidir_sm100.cu``, tcgen05 / TMA): the ending node's per-head
    contractions over every head's channels in one launch each (``bo_pv``: O_e = P_e V_e and dV_e = P_e^T dO_e; ``bo_dp``: dP_e[h] =
    sum_c dO_e[c] V_e[c]^T, fp32, a deterministic cluster reduction), and the softmax backward's row dot (``bo_rowdot``)."""
    ensure_cuda_home()
    src = Path(__file__).parent / "b200_sources"
    return load_extension(
        name="triattn_b200_bidir_v3",
        sources=[str(src / "bidir_sm100.cu")],
        extra_cuda_cflags=[*host_flags(), *_FLAGS, *gencodes("100a"), f"-I{src}"],
        verbose=False,
    )


@functools.lru_cache(maxsize=None)
def lnb():
    """The B200 LayerNorm backward of the bias-only paths (``b200_sources/lnbwd_sm100.cu``, C = 128): the shared glue's ``ln_bwd`` with its
    row loads issued together and a row step ahead, a persistent grid (fewer d gamma / d beta partial rows). dx bitwise equal."""
    ensure_cuda_home()
    src = Path(__file__).parent / "b200_sources"
    return load_extension(
        name="triattn_b200_ln_v1",
        sources=[str(src / "lnbwd_sm100.cu")],
        extra_cuda_cflags=[*host_flags(), *_FLAGS, *gencodes("100a")],
        verbose=False,
    )


@functools.lru_cache(maxsize=None)
def ltmm():
    """Tuned cublasLt bf16 -> fp32 GEMMs (``b200_sources/ltmm_sm100.cu``): the first call of a shape outside a graph capture times the heuristic's 32
    candidates (deterministic and accurate ones only) and keeps the fastest; used for the bias-only backward's tall-K contractions."""
    ensure_cuda_home()
    src = Path(__file__).parent / "b200_sources"
    return load_extension(
        name="triattn_b200_ltmm_v2",
        sources=[str(src / "ltmm_sm100.cu")],
        extra_cuda_cflags=[*host_flags(), *_FLAGS, *gencodes("100a")],
        verbose=False,
    )
