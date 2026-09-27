"""Host side of the sm_100a SWA atom block kernels (cubins through drv). Rows are ((a * B + b) * S + s); C = 128, H = 4, D = 32."""
from pathlib import Path
import torch
import drv
from common import C, H, D, EPS

HERE = Path(__file__).resolve().parent
_NSM = [None]


def nsm():
    if _NSM[0] is None:
        _NSM[0] = torch.cuda.get_device_properties(0).multi_processor_count
    return _NSM[0]


def tiling(A):
    """(SP augments, AT atoms) of a 128-row tile: SP = min(A, 16), AT = 128 // SP (the modulation tile holds AT <= 25 rows)."""
    SP = min(A, 16)
    AT = 128 // SP
    assert AT <= 25, "A >= 6 needed by the resident-weight layout"
    return SP, AT


def map_rows(t, A, B, S, SP, AT):
    """[A, B, S, 128] bf16 rows -> 4-D TMA map, box (64 channels, AT atoms, 1, SP augments), 128-B swizzle."""
    return drv.TensorMapND(t, [C, S, B, A], [C * 2, S * C * 2, B * S * C * 2], [64, AT, 1, SP])


class QkvgFwd:
    def __init__(self, cubin=HERE / "build" / "qkvg_fwd.cubin"):
        self.k = drv.Kernel(str(cubin), "swa_qkvg_fwd_sm100", 232448)

    def bind(self, q, mod, cos, sin, wqkv, wg, A, B, save=False):
        """q [N, S, C] bf16 (N = A B); mod [B S, 6C] fp32; cos / sin [B S, D/2] fp32 -> run(), (Qh, Kh, Vh [N, H, S, D], G [N S, C], X, PQ, PK)."""
        N, S, _ = q.shape
        SP, AT = tiling(A)
        dev = q.device
        W = torch.cat([wqkv, wg]).contiguous()
        Qh, Kh, Vh = (torch.empty(N, H, S, D, device=dev, dtype=torch.bfloat16) for _ in range(3))
        G = torch.empty(N * S, C, device=dev, dtype=torch.bfloat16)
        X, PQ, PK = ((torch.empty(N * S, C, device=dev, dtype=torch.bfloat16) for _ in range(3)) if save else (G, G, G))
        maps = (map_rows(q, A, B, S, SP, AT), drv.TensorMap(W, [C, 4 * C], C * 2, [64, 128]),
                drv.TensorMapND(mod, [32, B * S, 24], [6 * C * 4, 128], [32, AT, 8], swizzle=128, dtype="f32"),   # (32 ch, rows, 24 blocks)
                drv.TensorMapND(cos, [D // 2, B * S], [D // 2 * 4], [D // 2, AT], swizzle=0, dtype="f32"),
                drv.TensorMapND(sin, [D // 2, B * S], [D // 2 * 4], [D // 2, AT], swizzle=0, dtype="f32"))
        nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
        ntile = nab * nag * B
        grid = (min(nsm(), ntile), 1, 1)

        def run():
            self.k(grid, (384, 1, 1), *maps, int(S), int(A), int(B), int(SP), int(AT), int(nab), int(nag), int(ntile), float(EPS), float(EPS),
                   int(save), Qh, Kh, Vh, G, X, PQ, PK)
        run.keep = (maps, W)
        return run, (Qh, Kh, Vh, G, X, PQ, PK)
