# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/cuda/layernorm/__init__.py
"""CUDA implementation of LayerNorm."""

from pathlib import Path


from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

ensure_cuda_home()

_dir = Path(__file__).parent

layer_norm_cuda = load_extension(
    name="layer_norm_cuda",
    sources=[str(_dir / "layer_norm_cuda_kernel.cu")],
    # Explicit -gencode so the JIT build never relies on torch's arch autodetect
    # (which can misreport the device, e.g. "Unknown CUDA arch (10.1)" on H100).
    # Arch list filtered against what the local nvcc actually supports -- see kernels/_nvcc.py.
    # A hard-coded compute_90 made this build fail outright when PATH resolved nvcc to CUDA 11.7.
    extra_cuda_cflags=[*host_flags(), "-O3", "--use_fast_math", *gencodes("80", "90", "100", ptx=("100",))],
    verbose=False,
)


def layer_norm_bwd_cuda(dy, x, weight, mean, rstd, row_scale=None):
    """Standalone CUDA LayerNorm backward candidate.

    Signature matches compile_native._bwd_persistent_impl:
    (dy, x, weight, mean, rstd) -> (dx, dw, db).

    Optional ``row_scale`` [M] folds a per-row scale into the backward of
    ``y = LN(x) * row_scale`` (AF triangle pair-mask). The incoming grad is
    scaled by row_scale per row; dx/dw/db all follow (matches the triton path).
    """
    return layer_norm_cuda.layer_norm_bwd(dy, x, weight, mean, rstd, row_scale)
