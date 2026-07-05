# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/cuda/layernorm/__init__.py
"""CUDA implementation of LayerNorm."""

from pathlib import Path

from torch.utils.cpp_extension import load

_dir = Path(__file__).parent

layer_norm_cuda = load(
    name="layer_norm_cuda",
    sources=[str(_dir / "layer_norm_cuda_kernel.cu")],
    # Explicit -gencode so the JIT build never relies on torch's arch autodetect
    # (which can misreport the device, e.g. "Unknown CUDA arch (10.1)" on H100).
    extra_cuda_cflags=[
        "-O3", "--use_fast_math",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_90,code=sm_90",
    ],
    verbose=False,
)


def layer_norm_bwd_cuda(dy, x, weight, mean, rstd):
    """Standalone CUDA LayerNorm backward candidate.

    Signature matches compile_native._bwd_persistent_impl:
    (dy, x, weight, mean, rstd) -> (dx, dw, db).
    """
    return layer_norm_cuda.layer_norm_bwd(dy, x, weight, mean, rstd)
