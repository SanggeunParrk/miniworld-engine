"""Experimental atomic-free gate and bounded output-LN partial reductions."""

import torch

from . import _h100_runtime as T
from . import h100_b200style_aux as A
from . import h100_uni_wide_inference as I


def _config(d):
    # Gate rows in flight, gate CTA multiplier, output channels per CTA,
    # output token CTA multiplier (zero retains the original full grid).
    return (2, 4, 64, 4)


@T.device_cache
def _unit(cfg):
    rr, _, channels, _ = cfg
    source = A._device_source("wide_bwd.cu", "wbwd").replace(
        "constexpr int LOB_CH = 64;", f"constexpr int LOB_CH = {channels};"
    )
    body = (
        "#include <stdint.h>\n"
        + A._device_source("wide_aux.cu", "wide")
        + "\n"
        + source
        + A._STRUCTS
        + (T.SOURCES / "b200style/gate_row.cuh").read_text()
    )
    for d in (256, 384, 512):
        body += f'extern "C" __global__ void mw_b200style_rowgate_{d}(const Gate p) {{ gate_row<{d},{rr}>(p); }}\n'
    body = body.replace("__global__ void", "__global__ void __launch_bounds__(256)")
    return T.load_unit(
        str(
            T.compile_text(
                body, ["-std=c++17", "-O3", "-arch=sm_90a", "--cubin", "-lineinfo"]
            )
        ),
        "b200style_row_reductions",
    )


def _launch(cfg, name, grid, fields):
    _unit(cfg).kernel(name).launch(
        grid, (256, 1, 1), [T._launch_module().Struct(fields)]
    )


class ReductionAux(A.DriverAux):
    def gate_bwd(self, dy, pv, gv, ds, vec, mu, rs, dpr, dg, s1, s2, part, length):
        m, d = dy.shape
        cfg = _config(d)
        nb = min((m + 63) // 64, I._sms(dy.device.index) * cfg[1])
        partial = torch.empty((nb, 2 * d), device=dy.device, dtype=torch.float32)
        _launch(
            cfg,
            f"mw_b200style_rowgate_{d}",
            (nb, 1, 1),
            [
                dy,
                pv,
                gv,
                ds,
                vec,
                mu,
                rs,
                dpr,
                dg,
                s1,
                s2,
                partial,
                dg.stride(0),
                m,
                length,
            ],
        )
        _launch(
            cfg,
            "mw_b200style_colsum",
            ((2 * d + 31) // 32, 1, 1),
            [partial, part, nb, 2 * d],
        )

    def lnout_bwd(self, dor, tri, mu, rs, s1, s2, go, dt, dgb):
        h, m = tri.shape[0], tri.numel() // tri.shape[0]
        cfg = _config(h if h in (256, 384, 512) else h // 2)
        gx = min(m // 256, I._sms(tri.device.index) * cfg[3]) if cfg[3] else m // 256
        partial = torch.empty((gx, 2 * h), device=tri.device, dtype=torch.float32)
        _launch(
            cfg,
            "mw_b200style_lnout",
            (gx, h // cfg[2], 1),
            [dor, tri, mu, rs, s1, s2, go, dt, partial, m, h],
        )
        _launch(
            cfg,
            "mw_b200style_colsum",
            ((2 * h + 31) // 32, 1, 1),
            [partial, dgb, gx, 2 * h],
        )
