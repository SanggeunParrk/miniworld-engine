"""Experimental front with independent Hopper compute/epilogue warpgroups."""

from . import _h100_runtime as T
from . import h100_uni_wide_inference as I


def _config(d, backward):
    return (128, 2)


@T.device_cache
def _kernel(d, h, backward, cfg):
    bn, stages = cfg
    smem = 128 * d * 2 + stages * bn * 128 + 2 * (bn // 2) * 128 * 4 + 512
    name = f"mw_b200style_overlap{'wb' if backward else ''}_n{bn}s{stages}"
    flags = [
        "-std=c++17",
        "-O3",
        "-arch=sm_90a",
        "--cubin",
        "-lineinfo",
        f"-I{I._headers()}",
        f"-I{T.SOURCES / 'wide_forward'}",
        f"-DWIDTH={d}",
        f"-DHIDDEN={h}",
        f"-DBACKWARD={int(backward)}",
        f"-DBLOCKN={bn}",
        f"-DSTAGES={stages}",
        f"-DKERNEL_NAME={name}",
        "-DMW_MINB=1",
        "-DMW_K1_STREAM=1",
        "-DTMN_SIGMOID_TANH=1",
    ]
    kernel = T.load_unit(
        str(T.compile(T.SOURCES / "b200style/k1overlap.cu", flags)), "b200style_overlap"
    ).kernel(name)
    kernel.set_max_dynamic_smem(smem)
    return kernel, smem


def front(x, w1, mask, gi, bi, ab, xs, dpl=None, dcat=None, xn=None):
    n, d, h = x.shape[1], x.shape[-1], ab.shape[0] // 2
    m = n * n
    backward = dcat is not None
    cfg = _config(d, backward)
    bn, _ = cfg
    kernel, smem = _kernel(d, h, backward, cfg)
    am = I._map(ab, [64, 32], [m, 2 * h], [m * 2])
    fields = [
        I._map(x, [64, 128], [d, m], [d * 2]),
        I._map(w1, [64, bn], [d, 4 * h], [d * 2]),
        am,
        I._map(dpl, [64, bn // 2], [m, 2 * h], [m * 2]) if backward else am,
        I._map(dcat, [64, 64], [dcat.shape[1], m], [dcat.shape[1] * 2])
        if backward
        else am,
        I._map(xn, [64, 128], [d, m], [d * 2]) if xn is not None else am,
        mask,
        gi,
        bi,
        xs if backward else None,
        None if backward else xs,
        m,
        n,
        m // 128,
        int(xn is not None),
    ]
    kernel.launch(
        (min(m // 128, I._sms(x.device.index)), 1, 1),
        (640, 1, 1),
        [T._launch_module().Struct(fields)],
        smem,
    )
    return xn is not None
