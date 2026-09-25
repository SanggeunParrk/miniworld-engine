"""Host side of the sm_100a fused forward: tensor maps + launch through drv (cubin, cuda.bindings)."""
from pathlib import Path
import torch
import drv
from common import D, H

HERE = Path(__file__).resolve().parent
SMEM = 2 * 49152 + 4 * 32768 + 1024 + 256


class FusedFwd:
    def __init__(self, cubin=HERE / "build" / "tfwd.cubin", name="transition_fwd_sm100", smem=SMEM, threads=512, cluster=None):
        self.threads = threads
        self.cluster = cluster
        self.k = drv.Kernel(str(cubin), name, smem, cluster=cluster)
        self.nsm = torch.cuda.get_device_properties(0).multi_processor_count
        self._wmaps = None

    def set_weights(self, wa, wb, ws):
        tm = drv.TensorMap
        self.w = (wa, wb, ws)
        self._wmaps = (tm(wa, [D, H], D * 2, [64, 64]), tm(wb, [D, H], D * 2, [64, 64]), tm(ws, [H, D], H * 2, [64, 64]))

    def bind(self, x, gamma, beta, eps=1e-5, save=True):
        """Allocate outputs and encode activation maps once, return a launch closure (capture-safe)."""
        M = x.shape[0]
        assert M % 128 == 0 and x.shape[1] == D
        out = torch.empty_like(x); xn = torch.empty_like(x)
        rstd = torch.empty(M, device=x.device, dtype=torch.float32); c1 = torch.empty_like(rstd)
        tm = drv.TensorMap
        mx, mo, mxn = (tm(t, [D, M], D * 2, [64, 64]) for t in (x, out, xn))
        tiles = M // 128
        grid = (min(self.nsm, tiles), 1, 1)
        if self.cluster:
            grid = (max(self.cluster, grid[0] - grid[0] % self.cluster), 1, 1)
        keep = (mx, mo, mxn)

        def run():
            self.k(grid, (self.threads, 1, 1), mx, *self._wmaps, mo, mxn, gamma, beta, rstd, c1, int(tiles), float(eps), int(bool(save)))
        run.keep = keep
        return run, out, xn, rstd, c1


def FusedFwd2(cubin=HERE / "build" / "tfwd2.cubin"):
    """The 2-CTA (cta_group::2) forward: same host contract, launched as 2-CTA clusters."""
    return FusedFwd(cubin, "transition_fwd2_sm100", 230912, cluster=2)
