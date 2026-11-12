"""Own-CUDA batched FP32 master parameter casts; BF16 inputs retain their tensors.  Built for the A100 (``parameter_cast_sm80``) and the
GeForce RTX 5090 (``parameter_cast_sm120``, the same source for sm_120a); ``_ext`` loads the current card's."""

from pathlib import Path

from ..._nvcc import ensure_cuda_home, host_flags, load_extension
from ._arch import per_arch


def _build(suffix: str, gencode: str):
    ensure_cuda_home()
    directory = Path(__file__).parent
    return load_extension(
        name=f"parameter_cast_{suffix}",
        sources=[str(directory / "parameter_cast_sm80.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-lineinfo",
            gencode,
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


_ext, _ext_sm80, _ext_sm120 = per_arch(_build)


def result(parameters):
    return tuple(_ext().result(list(parameters)))


def result_wab(wa, wb, ws):
    """(wab [2H, D] bf16 = [wa; wb], ws bf16) from fp32 masters in one launch."""
    return tuple(_ext().result_wab(wa, wb, ws))
