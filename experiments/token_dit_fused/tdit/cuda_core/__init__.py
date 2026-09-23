"""Fused attention core (sm_90a): gated attention with the hoisted bias multicast to the S same-bias CTAs."""
import functools
import os
from pathlib import Path


def _defs():
    """ATTN_DEFS="NWG=2 PIPE=1" -- build variants, each under its own module name."""
    return os.environ.get("ATTN_DEFS", "").split()

_dir = Path(__file__).resolve().parent
_v5 = Path("/home/psk6950/miniworld-engine-dit2/src/miniworld_engine/kernels/transition/cuda/anthropic_v5")


@functools.lru_cache(maxsize=1)
def _ext():
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension
    ensure_cuda_home()
    return load_extension(
        name="tdit_attn_core" + "".join("_" + d.replace("=", "") for d in _defs()),
        sources=[str(_dir / "attn_core.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("90a"), f"-I{_v5}", "--expt-relaxed-constexpr",
                           "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                           "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-Xptxas=-v", *("-D" + d for d in _defs())],
        extra_cflags=["-std=c++17"], verbose=True)


def attn_core(qkvg, bias, block, S, H=16, dbg=None):
    """qkvg [S*L, 4*768] bf16 (q|k|v|g); bias [nb*H, L, L] bf16. Gated output written over q."""
    _ext().attn_core(qkvg, bias, block, S, H, dbg)
