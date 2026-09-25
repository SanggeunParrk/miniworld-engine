"""Host side of the sm_100a fused backward + the training step (fused forward with saves -> fused backward -> reduction)."""
from pathlib import Path
import torch
import drv
from common import D, H

HERE = Path(__file__).resolve().parent
SMEM_BWD = 230656


class FusedBwd:
    def __init__(self, cubin=HERE / "build" / "tbwd.cubin", repl=10):
        self.k = drv.Kernel(str(cubin), "transition_bwd_sm100", SMEM_BWD)
        self.red = drv.Kernel(str(cubin), "transition_bwd_reduce", 0)
        self.nsm = torch.cuda.get_device_properties(0).multi_processor_count
        self.repl = repl

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

        def run():
            self.k((self.nsm, 1, 1), (384, 1, 1), *maps, rstd, c1, gamma, partab, parts, dgbw, int(tiles), int(ndw))
            self.red(((nred + 255) // 256, 1, 1), (256, 1, 1), partab, parts, dgbw, dwa, dwb, dws, dgam, dbeta, int(ndw), int(ndx * 4))
        run.keep = maps
        return run, dict(dx=dx, dwa=dwa, dwb=dwb, dws=dws, dgamma=dgam, dbeta=dbeta)


class FusedTrain:
    """One training step of the module: y = fused forward (saves xn, rstd, c1), then the fused backward for a given dy."""
    def __init__(self, fwd, repl=10, cubin=None):
        self.f = fwd
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
