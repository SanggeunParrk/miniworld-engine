"""Host side of the e4m3 forward tfwd8.cu and of the relaxed-precision training step (quantize weights -> forward -> backward)."""
from pathlib import Path
import torch
import drv
from common import D, H
from bwd8_op import FusedBwd8, FusedBwd8x, Quant8, scales_for

HERE = Path(__file__).resolve().parent


class FusedFwd8:
    def __init__(self, cubin=HERE / "build" / "tfwd8.cubin"):
        self.k = drv.Kernel(str(cubin), "transition_fwd8_sm100", 232448, cluster=2, pdl="pdl" in str(cubin))
        self.nsm = torch.cuda.get_device_properties(0).multi_processor_count

    def bind(self, x, gamma, beta, sc, wab_q, ws_q, eps=1e-5, save=True):
        M = x.shape[0]; tiles = M // 128
        tm = drv.TensorMap
        out = torch.empty_like(x)
        xq = torch.empty(M, D, device=x.device, dtype=torch.uint8)
        rstd = torch.empty(M, device=x.device, dtype=torch.float32); c1 = torch.empty_like(rstd)
        maps = (tm(x, [D, M], D * 2, [64, 64]), tm(wab_q, [D, 2 * H], D, [128, 64], dtype="u8"),
                tm(ws_q, [H, D], H, [64, 64], swizzle=64, dtype="u8"), tm(out, [D, M], D * 2, [64, 64]),
                tm(xq, [D, M], D, [128, 64], dtype="u8"))
        g = min(self.nsm, tiles); g -= g % 2

        def run():
            self.k((g, 1, 1), (512, 1, 1), *maps, gamma, beta, sc, rstd, c1, int(tiles), float(eps), int(bool(save)))
        run.keep = maps
        return run, out, xq, rstd, c1


class Train8:
    """One relaxed-precision training step: weights -> e4m3 (quant8), e4m3 forward (saves xq, rstd, c1), e4m3 backward + reduction.
    Scales: delayed-scaling emulation, computed once at bind from the step's own tensors."""
    def __init__(self, repl=7, fcubin=HERE / "build" / "tfwd8.cubin", bcubin=HERE / "build" / "tbwd8.cubin"):
        self.f = FusedFwd8(fcubin); self.q = Quant8(bcubin)
        self.b = FusedBwd8x(bcubin, repl) if "tbwd8x" in str(bcubin) or "b8x" in str(bcubin) or "8x" in str(bcubin) else FusedBwd8(bcubin, repl)

    def bind(self, x, wa, wb, ws, gamma, beta, dy):
        sc = scales_for(x, wa, wb, ws, gamma, beta, dy)
        run_q, wab_q, wst_q, ws_q = self.q.bind(wa, wb, ws, sc)
        run_f, out, xq, rstd, c1 = self.f.bind(x, gamma, beta, sc, wab_q, ws_q, save=True)
        run_i, out_i, *_ = self.f.bind(x, gamma, beta, sc, wab_q, ws_q, save=False)
        run_b, grads = self.b.bind(dy, xq, x, rstd, c1, gamma, wab_q, wst_q, sc, wa, wb, ws)

        def step():
            run_q(); run_f(); run_b()
        step.keep = (run_q, run_f, run_b)
        step.out, step.grads, step.sc, step.xq = out, grads, sc, xq
        step.infer, step.infer_out = run_i, out_i
        return step
