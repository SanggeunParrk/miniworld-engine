"""CUDA implementation of the fused Transition b2b forward."""

from pathlib import Path

from torch.utils.cpp_extension import load

_dir = Path(__file__).parent

transition_b2b_cuda = load(
    name="transition_b2b_cuda",
    sources=[str(_dir / "transition_b2b_kernel.cu")],
    extra_cuda_cflags=[
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-gencode",
        "arch=compute_90a,code=sm_90a",
        "-I/home/psk6950/mathdx_dl/extracted/nvidia/mathdx/include",
        "-I/home/psk6950/mathdx_dl/extracted/nvidia/mathdx/external/cutlass/include",
        "-DCUBLASDX_IGNORE_NVBUG_5218000_ASSERT",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    ],
    extra_cflags=["-std=c++17"],
    verbose=False,
)


def transition_b2b_fwd(x, rstd, c1, g, beta, wa, wb, ws):
    """Fused LN + SwiGLU expand + squeeze forward for fixed AF3 transition shapes."""
    return transition_b2b_cuda.transition_b2b_fwd(x, rstd, c1, g, beta, wa, wb, ws)
