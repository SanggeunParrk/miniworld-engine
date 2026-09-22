"""gemm_resgate_adaln: the residual GEMM with the gated residual and the next AdaLN in its epilogue (sm_90a, bf16)."""
import functools
import os
from pathlib import Path

import torch

_dir = Path(__file__).resolve().parent
_v5 = Path("/home/psk6950/miniworld-engine-dit2/src/miniworld_engine/kernels/transition/cuda/anthropic_v5")


def _defs():
    """GRA_DEFS="EPI=0 ..." -- diagnostic builds, each under its own module name."""
    return os.environ.get("GRA_DEFS", "").split()


@functools.lru_cache(maxsize=1)
def _ext():
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension
    ensure_cuda_home()
    return load_extension(
        name="tdit_gemm_resgate_adaln" + "".join("_" + d.replace("=", "") for d in _defs()),
        sources=[str(_dir / "gemm_resgate_adaln.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("90a"), f"-I{_v5}", "--expt-relaxed-constexpr",
                           "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                           "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-Xptxas=-v", *("-D" + d for d in _defs())],
        extra_cflags=["-std=c++17"], verbose=True)


def gemm_resgate_adaln(a, w, x, gl, ms, mb, xa, L, eps=1e-5, nwg=None):
    """x += sigmoid(gl) * (a @ w.T) in place (fp32); then, when ms is given, xa = AdaLN(x) (bf16)."""
    if nwg is None:
        nwg = 2 if a.shape[0] >= 3072 else 1
    _ext().gemm_resgate_adaln(a, w, x, gl, ms, mb, xa, L, eps, nwg)
