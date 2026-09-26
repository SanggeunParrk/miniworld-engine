"""Host side of the sm_100a fused backward + the training step (fused forward with saves -> fused backward -> reduction)."""
from pathlib import Path
import torch
import drv
from common import D, H

HERE = Path(__file__).resolve().parent
SMEM_BWD = 230656


class FusedBwd:
    def __init__(self, cubin=HERE / "build" / "tbwd.cubin", repl=9):
        self.k = drv.Kernel(str(cubin), "transition_bwd_sm100", SMEM_BWD, cluster=2)
        self.red = drv.Kernel(str(cubin), "transition_bwd_reduce", 0)
        self.nsm = torch.cuda.get_device_properties(0).multi_processor_count
        self.repl = repl

    xch = False                                                   # EXCH_SIM builds take one more tensor map (the exchange buffer)

    def bind(self, dy, xn, x, rstd, c1, gamma, wa, wb, ws):
        M = x.shape[0]
        tiles = M // 128
        tm = drv.TensorMap
        ndw = 8 * self.repl
        ndx = self.nsm - ndw
        dev = x.device
        dx = torch.empty_like(x)
        partab = torch.empty(ndw, 128, 128, device=dev, dtype=torch.float32)
        parts = torch.empty(ndw, 128, 64, device=dev, dtype=torch.float32)
        dgbw = torch.empty(ndx * 4, 256, device=dev, dtype=torch.float32)
        dwa = torch.empty_like(wa); dwb = torch.empty_like(wb); dws = torch.empty_like(ws)
        dgam = torch.empty(D, device=dev, dtype=torch.float32); dbeta = torch.empty_like(dgam)
        maps = (tm(dy, [D, M], D * 2, [64, 64]), tm(xn, [D, M], D * 2, [64, 64]), tm(x, [D, M], D * 2, [64, 64]),
                tm(ws, [H, D], H * 2, [64, 64]), tm(wa, [D, H], D * 2, [64, 64]), tm(wb, [D, H], D * 2, [64, 64]),
                tm(dx, [D, M], D * 2, [64, 64]))
        nred = 3 * H * D + 256
        extra = ()
        if self.xch:
            xbuf = torch.zeros(tiles * 8 * 128, 64, device=dev, dtype=torch.bfloat16)
            extra = (tm(xbuf, [64, tiles * 8 * 128], 128, [64, 64]),)

        def run():
            self.k((self.nsm, 1, 1), (512, 1, 1), *maps, rstd, c1, gamma, x, partab, parts, dgbw, int(tiles), int(ndw), *extra)
            self.red(((nred + 255) // 256, 1, 1), (256, 1, 1), partab, parts, dgbw, dwa, dwb, dws, dgam, dbeta, int(ndw), int(ndx * 4))
        run.keep = maps + extra
        run.bufs = (partab, parts, dgbw)
        return run, dict(dx=dx, dwa=dwa, dwb=dwb, dws=dws, dgamma=dgam, dbeta=dbeta)


class FusedTrain:
    """One training step of the module: y = fused forward (saves xn, rstd, c1), then the fused backward for a given dy."""
    def __init__(self, fwd, repl=9, cubin=None, v2=False, x=False):
        self.f = fwd
        if x:
            self.b = FusedBwdX(cubin, repl) if cubin else FusedBwdX(repl=repl)
        elif v2:
            self.b = FusedBwd2(cubin, repl) if cubin else FusedBwd2(repl=repl)
        else:
            self.b = FusedBwd(cubin, repl) if cubin else FusedBwd(repl=repl)

    def bind(self, x, gamma, beta, dy):
        wa, wb, ws = self.f.w
        run_f, out, xn, rstd, c1 = self.f.bind(x, gamma, beta, save=True)
        run_b, grads = self.b.bind(dy, xn, x, rstd, c1, gamma, wa, wb, ws)

        def step():
            run_f(); run_b()
        step.keep = (run_f, run_b)
        step.out, step.grads = out, grads
        return step


class FusedBwd2(FusedBwd):
    """tbwd2: DX writes [dA | dB | h] per (tile, chunk) to an L2-resident exchange buffer, DW only runs the weight gradients."""
    def __init__(self, cubin=HERE / "build" / "tbwd2.cubin", repl=4):
        self.k = drv.Kernel(str(cubin), "transition_bwd2_sm100", SMEM_BWD, cluster=2)
        self.red = drv.Kernel(str(cubin), "transition_bwd_reduce", 0)
        self.nsm = torch.cuda.get_device_properties(0).multi_processor_count
        self.repl = repl

    def bind(self, dy, xn, x, rstd, c1, gamma, wa, wb, ws):
        run0, grads = super().bind(dy, xn, x, rstd, c1, gamma, wa, wb, ws)
        M = x.shape[0]; tiles = M // 128
        gbuf = torch.empty(tiles * 8 * 3 * 128, 64, device=x.device, dtype=torch.bfloat16)
        flags = torch.zeros(tiles * 8, device=x.device, dtype=torch.int32)
        mg = drv.TensorMap(gbuf, [64, tiles * 8 * 3 * 128], 128, [64, 64])
        maps, (partab, parts, dgbw) = run0.keep, run0.bufs
        ndw = 8 * self.repl; ndx = self.nsm - ndw
        nred = 3 * H * D + 256

        def run():
            self.k((self.nsm, 1, 1), (512, 1, 1), *maps, rstd, c1, gamma, x, partab, parts, dgbw, int(tiles), int(ndw), mg, gbuf, flags)
            self.red(((nred + 255) // 256, 1, 1), (256, 1, 1), partab, parts, dgbw, grads["dwa"], grads["dwb"], grads["dws"],
                     grads["dgamma"], grads["dbeta"], int(ndw), int(ndx * 4))
        run.keep = (maps, mg, gbuf, flags)
        return run, grads


class FusedBwdX(FusedBwd):
    """tbwdx: bf16 exchange backward — DW slices publish bf16 [dA | dB] blocks, DX CTAs run only d_xn + the LayerNorm backward."""
    def __init__(self, cubin=HERE / "build" / "tbwdx.cubin", repl=14):
        self.k = drv.Kernel(str(cubin), "transition_bwdx_sm100", 232448, cluster=2)
        self.red = drv.Kernel(str(cubin), "transition_bwdx_reduce", 0)
        self.nsm = torch.cuda.get_device_properties(0).multi_processor_count
        self.repl = repl

    def bind(self, dy, xn, x, rstd, c1, gamma, wa, wb, ws):
        M = x.shape[0]; tiles = M // 128
        tm = drv.TensorMap
        ndw = 8 * self.repl; ndx = self.nsm - ndw
        dev = x.device
        dx = torch.empty_like(x)
        partab = torch.empty(ndw, 128, 128, device=dev, dtype=torch.float32)
        parts = torch.empty(ndw, 128, 64, device=dev, dtype=torch.float32)
        dgbw = torch.zeros(ndx * 4, 256, device=dev, dtype=torch.float32)
        dwa = torch.empty_like(wa); dwb = torch.empty_like(wb); dws = torch.empty_like(ws)
        dgam = torch.empty(D, device=dev, dtype=torch.float32); dbeta = torch.empty_like(dgam)
        dab = torch.zeros(tiles * 8 * 128, D, device=dev, dtype=torch.bfloat16)
        dflags = torch.zeros(tiles * 8, device=dev, dtype=torch.int32)
        epoch = torch.ones(1, device=dev, dtype=torch.int32)
        maps = (tm(dy, [D, M], D * 2, [64, 64]), tm(xn, [D, M], D * 2, [64, 64]), tm(x, [D, M], D * 2, [64, 64]),
                tm(ws, [H, D], H * 2, [64, 64]), tm(wa, [D, H], D * 2, [64, 64]), tm(wb, [D, H], D * 2, [64, 64]),
                tm(dx, [D, M], D * 2, [64, 64]), tm(dab, [D, tiles * 8 * 128], D * 2, [64, 64]))

        def run():
            self.k((self.nsm, 1, 1), (512, 1, 1), *maps, rstd, c1, gamma, partab, parts, dgbw, dab, dflags, epoch, int(tiles), int(ndw))
            self.red((800, 1, 1), (256, 1, 1), partab, parts, dgbw, dwa, dwb, dws, dgam, dbeta, int(ndw), int(ndx * 4), epoch)
        run.keep = maps + (dab, dflags, epoch)
        run.bufs = (partab, parts, dgbw)
        return run, dict(dx=dx, dwa=dwa, dwb=dwb, dws=dws, dgamma=dgam, dbeta=dbeta)
