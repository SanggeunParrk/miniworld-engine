"""Token DiT attention core, training kernels (sm_90a, bf16 operands in an fp32 model)."""
import functools
import os
from pathlib import Path

_dir = Path(__file__).resolve().parent
_v5 = Path("/home/psk6950/miniworld-engine-dit2/src/miniworld_engine/kernels/transition/cuda/anthropic_v5")


def _defs(var):
    return os.environ.get(var, "").split()


@functools.lru_cache(maxsize=None)
def ext(name, extra=()):
    """Build ``<name>.cu``; ``<NAME>_DEFS="A=1 B=2"`` and ``extra`` add defines (each variant its own module)."""
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension
    ensure_cuda_home()
    defs = [*_defs(name.upper() + "_DEFS"), *extra]
    return load_extension(
        name="tdt_" + name + "".join("_" + d.replace("=", "") for d in defs),
        sources=[str(_dir / f"{name}.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("90a"), f"-I{_v5}", "--expt-relaxed-constexpr",
                           "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                           "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-Xptxas=-v", *("-D" + d for d in defs)],
        extra_cflags=["-std=c++17"], verbose=bool(os.environ.get("TDT_VERBOSE")))
