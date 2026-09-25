"""Host side of the relaxed-precision (e4m3 operand) backward tbwd8.cu: weight quantizer, scales, tensor maps, launch.

Scales (dequantization, value = q * s) follow delayed scaling: activation / gradient amax values come from a calibration of the same
inputs (in training, the previous step's amax), weight and xn scales are exact for the current tensors."""
from pathlib import Path
import math, torch
import drv
from common import D, H

HERE = Path(__file__).resolve().parent
SMEM_BWD8 = 206336
E4M3_MAX = 448.0


def scales_for(x, wa, wb, ws, gamma, beta, dy, eps=1e-5):
    """sc[0..6]: s_x, s_wab, s_ws, s_dy, s_h, s_dab, s_hf from the tensors (and a torch pass for the activation amax values)."""
    s_x = (gamma.abs().max().item() * math.sqrt(D - 1) + beta.abs().max().item()) / E4M3_MAX
    s_wab = max(wa.abs().max().item(), wb.abs().max().item()) / E4M3_MAX
    s_ws = ws.abs().max().item() / E4M3_MAX
    s_dy = dy.abs().max().item() / E4M3_MAX
    xn = torch.nn.functional.layer_norm(x.float(), (D,), gamma, beta, eps)
    a = xn @ wa.float().t(); b = xn @ wb.float().t()
    s = torch.sigmoid(a); l = a * s; h = l * b
    g = dy.float() @ ws.float()
    da = g * b * (s + l * (1 - s)); db = g * l
    s_h = h.abs().max().item() / E4M3_MAX
    s_dab = max(da.abs().max().item(), db.abs().max().item()) / E4M3_MAX
    return torch.tensor([s_x, s_wab, s_ws, s_dy, s_h, s_dab, s_h, 0.0], device=x.device, dtype=torch.float32)


class Quant8:
    def __init__(self, cubin=HERE / "build" / "tbwd8.cubin"):
        self.k = drv.Kernel(str(cubin), "quant8_weights", 0)
        self.v2 = "8x" in str(cubin)

    def bind(self, wa, wb, ws, sc):
        dev = wa.device
        wab_q = torch.empty(2 * H, D, device=dev, dtype=torch.uint8)
        wst_q = torch.empty(H, D, device=dev, dtype=torch.uint8)
        ws_q = torch.empty(D, H, device=dev, dtype=torch.uint8)
        n = 3 * H * D // 2

        nb = 160 if self.v2 else (n + 255) // 256

        def run():
            self.k((nb, 1, 1), (256, 1, 1), wa, wb, ws, sc, wab_q, wst_q, ws_q)
        return run, wab_q, wst_q, ws_q


class FusedBwd8:
    def __init__(self, cubin=HERE / "build" / "tbwd8.cubin", repl=9):
        self.k = drv.Kernel(str(cubin), "transition_bwd8_sm100", SMEM_BWD8, cluster=2)
        self.red = drv.Kernel(str(cubin), "transition_bwd8_reduce", 0)
        self.nsm = torch.cuda.get_device_properties(0).multi_processor_count
        self.repl = repl

    def bind(self, dy, xq, x, rstd, c1, gamma, wab_q, wst_q, sc, wa, wb, ws):
        M = x.shape[0]; tiles = M // 128
        tm = drv.TensorMap
        ndw = 8 * self.repl; ndx = self.nsm - ndw
        dev = x.device
        dx = torch.empty_like(x)
        partab = torch.empty(ndw, 128, 128, device=dev, dtype=torch.float32)
        parts = torch.empty(ndw, 128, 64, device=dev, dtype=torch.float32)
        dgbw = torch.zeros(self.nsm * 4, 256, device=dev, dtype=torch.float32)   # rows by block (DYN) or DX index
        dwa = torch.empty_like(wa); dwb = torch.empty_like(wb); dws = torch.empty_like(ws)
        dgam = torch.empty(D, device=dev, dtype=torch.float32); dbeta = torch.empty_like(dgam)
        maps = (tm(dy, [D, M], D * 2, [64, 64]), tm(xq, [D, M], D, [128, 64], dtype="u8"), tm(x, [D, M], D * 2, [64, 64]),
                tm(wst_q, [D, H], D, [128, 64], dtype="u8"), tm(wst_q, [D, H], D, [128, 32], dtype="u8"),
                tm(wab_q, [D, 2 * H], D, [128, 64], dtype="u8"), tm(dx, [D, M], D * 2, [64, 64]))
        dyq = torch.zeros(M, D, device=dev, dtype=torch.uint8)
        flags = torch.zeros(tiles, device=dev, dtype=torch.int32)
        epoch = torch.ones(1, device=dev, dtype=torch.int32)
        maps = maps + (tm(dyq, [D, M], D, [128, 64], dtype="u8"),)
        nred = 3 * H * D + 256

        def run():
            self.k((self.nsm, 1, 1), (512, 1, 1), *maps, rstd, c1, gamma, sc, partab, parts, dgbw, dyq, flags, epoch, int(tiles), int(ndw))
            self.red(((nred + 255) // 256, 1, 1), (256, 1, 1), partab, parts, dgbw, sc, dwa, dwb, dws, dgam, dbeta, int(ndw), int(ndx * 4), epoch)
        run.keep = maps + (dyq, flags, epoch)
        run.bufs = (partab, parts, dgbw)
        return run, dict(dx=dx, dwa=dwa, dwb=dwb, dws=dws, dgamma=dgam, dbeta=dbeta)


class FusedBwd8x(FusedBwd8):
    """tbwd8x: the DW slices publish e4m3 [dA | dB] blocks; the DX CTAs only run d_xn + the LayerNorm backward (no recompute, one gate)."""
    def __init__(self, cubin=HERE / "build" / "tbwd8x.cubin", repl=14, pdl=None):
        pdl = ("pdl" in str(cubin)) if pdl is None else pdl
        self.cubin = cubin
        self.k = drv.Kernel(str(cubin), "transition_bwd8x_sm100", 232448, cluster=2, pdl=pdl)
        self.red = drv.Kernel(str(cubin), "transition_bwd8_reduce", 0, pdl=pdl)
        self.nsm = torch.cuda.get_device_properties(0).multi_processor_count
        self.repl = repl

    def bind(self, dy, xq, x, rstd, c1, gamma, wab_q, wst_q, sc, wa, wb, ws):
        run0, grads = super().bind(dy, xq, x, rstd, c1, gamma, wab_q, wst_q, sc, wa, wb, ws)
        M = x.shape[0]; tiles = M // 128
        maps, (dyq, flags, epoch) = list(run0.keep[:8]), run0.keep[8:]
        if "bf16dy" in str(self.cubin):
            maps[3] = drv.TensorMap(ws, [H, D], H * 2, [64, 64])   # the DW role's Ws slice: bf16, MN-major
        maps = tuple(maps)
        dab = torch.zeros(tiles * 8 * 128, D, device=x.device, dtype=torch.uint8)
        dflags = torch.zeros(tiles * 8, device=x.device, dtype=torch.int32)
        mdab = drv.TensorMap(dab, [D, tiles * 8 * 128], D, [128, 64], dtype="u8")
        ndw = 8 * self.repl; ndx = self.nsm - ndw
        partab, parts, dgbw = run0.bufs
        ctr = torch.zeros(1, device=x.device, dtype=torch.int32)
        nrows = self.nsm * 4 if "dyn" in str(self.cubin) else ndx * 4
        nred = 3 * H * D + 256

        def run():
            self.k((self.nsm, 1, 1), (512, 1, 1), *maps, rstd, c1, gamma, sc, partab, parts, dgbw, dyq, flags, epoch, int(tiles), int(ndw),
                   mdab, dab, dflags, dy, ctr)
            self.red((800, 1, 1), (256, 1, 1), partab, parts, dgbw, sc, grads["dwa"], grads["dwb"], grads["dws"],
                     grads["dgamma"], grads["dbeta"], int(ndw), int(nrows), epoch, ctr)
        run.keep = maps + (dyq, flags, epoch, mdab, dab, dflags, ctr)
        return run, grads
