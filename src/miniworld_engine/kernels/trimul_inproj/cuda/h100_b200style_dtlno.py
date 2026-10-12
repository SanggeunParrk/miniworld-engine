"""Experimental projection-gradient GEMM fused with output-LN backward."""

import torch

from . import _h100_runtime as T
from . import h100_b200style as K
from . import h100_b200style_aux as A
from . import h100_uni_wide_inference as I


def _config(d, h):
    return (192 if h % 128 else 256 if h % 256 == 0 else 128, 2, 1)


@T.device_cache
def _kernel(d, h, cfg):
    bn, stages, _ = cfg[:3]
    mt = cfg[3] if len(cfg) > 3 else 128
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
        f"-DTOKENS={mt}",
        "-DMW_MINB=1",
        "-DMW_K1_STREAM=1",
    ]
    nwg = mt // 64
    smem = (
        stages * (mt * 128 + bn * 128)
        + nwg * bn * 128
        + nwg * 4 * 2 * bn * 4
        + nwg * 2 * bn * 4
        + 512
    )
    kernel = T.load_unit(
        str(T.compile(K.R / "dtlno.cu", flags)), "b200style_dtlno"
    ).kernel("mw_b200style_dtlno")
    kernel.set_max_dynamic_smem(smem)
    if mt == 64:
        drv = kernel.unit.drv
        active = drv._unwrap(
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
            drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                drv.d.CUfunction(int(kernel.handle)), 256, smem
            ),
        )
        if active < 2:
            raise RuntimeError(f"fused output needs two resident CTAs, got {active}")
    return kernel, smem


def dtlno(wp, dpr, tri, mu, rs, s1, s2, go, dt, dgb):
    d, h = wp.shape
    m = dpr.shape[0]
    cfg = _config(d, h)
    bn, _, mult = cfg[:3]
    mt = cfg[3] if len(cfg) > 3 else 128
    nwg = mt // 64
    kernel, smem = _kernel(d, h, cfg)
    blocks = min(m // mt, I._sms(wp.device.index) * mult)
    partial = torch.empty((nwg * blocks, 2 * h), device=wp.device, dtype=torch.float32)
    wt = wp.t().contiguous()
    fields = [
        I._map(dpr, [64, mt], [d, m], [d * 2]),
        I._map(wt, [64, bn], [d, h], [d * 2]),
        I._map(tri, [64, 32], [m, h], [m * 2]),
        I._map(dt, [64, 32], [m, h], [m * 2]),
        mu,
        rs,
        s1,
        s2,
        go,
        partial,
        m,
        m // mt,
    ]
    kernel.launch(
        (blocks, h // bn, 1),
        (128 * (nwg + 1), 1, 1),
        [T._launch_module().Struct(fields)],
        smem,
    )
    A._launch(
        "mw_b200style_colsum",
        ((2 * h + 31) // 32, 1, 1),
        [partial, dgb, nwg * blocks, 2 * h],
    )


class FusedAux(A.DriverAux):
    def lnout_bwd(self, operands, tri, mu, rs, s1, s2, go, dt, dgb):
        wp, dpr = operands
        dtlno(wp, dpr, tri, mu, rs, s1, s2, go, dt, dgb)
