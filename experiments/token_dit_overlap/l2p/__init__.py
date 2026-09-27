"""L2 persisting window for the step's residual buffers."""
import functools
from pathlib import Path

_dir = Path(__file__).resolve().parent


@functools.lru_cache(maxsize=1)
def _ext():
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension
    ensure_cuda_home()
    return load_extension(name="tdit_l2_persist", sources=[str(_dir / "l2_persist.cu")],
                          extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("90a")],
                          extra_cflags=["-std=c++17"], verbose=False)


def set_window(t, hit_ratio=1.0):
    _ext().set_window(t, hit_ratio)


def clear_window():
    _ext().clear_window()


def max_persist_bytes():
    return _ext().max_persist_bytes()
