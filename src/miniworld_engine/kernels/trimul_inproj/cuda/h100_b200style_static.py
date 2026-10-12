"""Compile the inference-only K3 branch without saved activations or scale loads."""

from . import _h100_runtime as T
from . import h100_b200style as K
from . import h100_wide_inference as BI


@T.device_cache
def output_kernel(d, h, cfg):
    bn, stages = cfg
    source = K.R / ("k3reorder.cu" if d == 512 else "k3.cu")
    body = source.read_text()
    for anchor in (
        "if (p.pv)",
        "if(p.mu && (lane&3)==0)" if d == 512 else "if (p.mu)",
        "if (p.ds)",
    ):
        assert body.count(anchor) == 1, anchor
        body = body.replace(anchor, "if constexpr (false)")
    old = "mw_b200style_k3reorder" if d == 512 else "mw_b200style_k3"
    name = "mw_b200style_k3_static"
    assert old in body
    body = body.replace(old, name)
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
        str(T.compile_text(body, flags)), "b200style_k3_static"
    ).kernel(name)
    smem = BI._k3_smem(d, bn, stages)
    kernel.set_max_dynamic_smem(smem)
    return kernel, smem
