from . import _h100_runtime as T
from . import h100_b200style as K
from . import h100_wide_inference as BI


@T.device_cache
def output_kernel(d, h, cfg):
    bn, stages = cfg
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
        "-DGFOLD=1",
        "-DTMN_SIGMOID_TANH=1",
        "-DMW_MINB=1",
        "-DMW_K1_STREAM=1",
    ]
    kernel = T.load_unit(
        str(T.compile(K.R / "k3d384_tmasave.cu", flags)), "b200style_k3_d384_tmasave"
    ).kernel("mw_b200style_k3_d384_tmasave")
    smem = BI._k3_smem(d, bn, stages)
    kernel.set_max_dynamic_smem(smem)
    return kernel, smem
