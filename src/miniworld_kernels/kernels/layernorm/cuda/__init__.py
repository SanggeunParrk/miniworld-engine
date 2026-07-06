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
        # B200 / Blackwell (sm_100): emit SASS for sm_100 plus a compute_100 PTX
        # fallback so the plain-CUDA kernels (shfl + vector loads, no arch-only
        # features) load and JIT on sm_10.x. Without this the module builds but the
        # kernel launch fails with "no kernel image is available" on B200.
        "-gencode=arch=compute_100,code=sm_100",
        "-gencode=arch=compute_100,code=compute_100",
    ],
    verbose=False,
)


def layer_norm_bwd_cuda(dy, x, weight, mean, rstd):
    """Standalone CUDA LayerNorm backward candidate.

    Signature matches compile_native._bwd_persistent_impl:
    (dy, x, weight, mean, rstd) -> (dx, dw, db).
    """
    return layer_norm_cuda.layer_norm_bwd(dy, x, weight, mean, rstd)
