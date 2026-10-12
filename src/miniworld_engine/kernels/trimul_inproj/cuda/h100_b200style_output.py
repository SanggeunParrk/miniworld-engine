"""Experimental folded output with a deeper WGMMA pipeline."""

from . import _h100_runtime as T
from . import h100_b200style as K
from . import h100_wide_inference as BI

WAIT = 2


@T.device_cache
def _kernel(d, h, bn, stages, wait):
    flags = [
        "-std=c++17",
        "-O3",
        "-arch=sm_90a",
        "--cubin",
        "-lineinfo",
        f"-I{K._headers()}",
        f"-I{T.SOURCES / 'wide_forward'}",
        f"-DWIDTH={d}",
        f"-DHIDDEN={h}",
        f"-DBLOCKN={bn}",
        f"-DSTAGES={stages}",
        f"-DWAITN={wait}",
        "-DGFOLD=1",
        "-DTMN_SIGMOID_TANH=1",
        "-DMW_MINB=1",
        "-DMW_K1_STREAM=1",
    ]
    kernel = T.load_unit(
        str(T.compile(K.R / "k3deep.cu", flags)), "b200style_k3deep"
    ).kernel("mw_b200style_k3deep")
    smem = BI._k3_smem(d, bn, stages)
    kernel.set_max_dynamic_smem(smem)
    return kernel, smem


def output_kernel(d, h, cfg):
    return _kernel(d, h, *cfg, WAIT)
