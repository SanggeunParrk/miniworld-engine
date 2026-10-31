"""Compile the inference-only K3 branch without saved activations or scale loads."""

from . import _h100_runtime as T
from . import h100_b200style as K
from . import h100_wide_inference as BI


@T.device_cache
def output_kernel(d, h, cfg, rowpad=False, xbuf=1, extra=(), split=None):
    """rowpad: the any-L one-sample form (k3.cu ROWPAD: 64-token chunks of one pair row, padded planes); xbuf: residual / output
    tiles per warpgroup (k3.cu XBUF); split: (A stages, weight stages) of the split operand rings (k3.cu NSA / NSB), None = one ring of
    cfg[1] stages."""
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
        *(["-DROWPAD=1"] if rowpad else []),
        *([f"-DXBUF={xbuf}"] if xbuf != 1 else []),
        *extra,   # measurement builds only (e.g. -DK3_DIAG=1)
        *([f"-DNSA={split[0]}", f"-DNSB={split[1]}"] if split else []),
    ]
    kernel = T.load_unit(
        str(T.compile_text(body, flags)), "b200style_k3_static"
    ).kernel(name)
    smem = BI._k3_smem(d, bn, stages) + (xbuf - 1) * (2 * 64 * bn * 2 + 16)
    if split:   # the one ring's stages give way to NSA A chunks (128 tok x 64 k) and NSB weight chunks (BN x 64 k)
        smem += split[0] * 128 * 128 + split[1] * bn * 128 + 16 * (split[0] + split[1]) - stages * (128 * 128 + bn * 128 + 16)
    if smem > 232448:
        raise ValueError("K3 resources")
    kernel.set_max_dynamic_smem(smem)
    return kernel, smem
