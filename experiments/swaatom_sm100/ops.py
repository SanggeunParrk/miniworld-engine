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

    def bind(self, q, mod, cos, sin, wqkv, wg, A, B, save=False, W=None):
        """q [N, S, C] bf16 (N = A B); mod [B S, 6C] fp32; cos / sin [B S, D/2] fp32 -> run(), (Qh, Kh, Vh [N, H, S, D], G [N S, C], X, PQ, PK)."""
        N, S, _ = q.shape
        SP, AT = tiling(A)
        dev = q.device
        W = torch.cat([wqkv, wg]).contiguous() if W is None else W
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


def pack_ffn64(wu):
    """rows per 64-wide hidden chunk j: [Wu[64 j .. 64 j + 63] ; Wu[256 + 64 j ..]] (the forward's a|b tile)."""
    NH = wu.shape[0] // 2
    return wu.view(2, NH // 64, 64, wu.shape[1]).permute(1, 0, 2, 3).reshape(2 * NH, wu.shape[1]).contiguous()


class FfnFwd:
    def __init__(self, cubin=HERE / "build" / "ffn_fwd.cubin"):
        self.k = drv.Kernel(str(cubin), "swa_ffn_fwd_sm100", 232448)

    def bind(self, q, g, o, mod, wo, wu, wd, A, B, save=False, packed=False):
        """q (residual), g (gate logits), o (attention output): [N S, C] bf16; mod [B S, 6C] fp32 -> run(), (out, q1, att, y, ffn).
        packed: wu is already pack_ffn64(Wu)."""
        M = q.numel() // C
        S = M // (A * B)
        SP, AT = tiling(A)
        dev = q.device
        out = torch.empty(M, C, device=dev, dtype=torch.bfloat16)
        Q1, Att, Y, Ff = ((torch.empty(M, C, device=dev, dtype=torch.bfloat16) for _ in range(4)) if save else (out, out, out, out))
        wab = wu if packed else pack_ffn64(wu)
        maps = (map_rows(q, A, B, S, SP, AT), map_rows(g, A, B, S, SP, AT), map_rows(o, A, B, S, SP, AT),
                drv.TensorMapND(mod, [32, B * S, 24], [6 * C * 4, 128], [32, AT, 16], swizzle=128, dtype="f32"),
                drv.TensorMap(wo, [C, C], C * 2, [64, 128]), drv.TensorMap(wab, [C, 512], C * 2, [64, 128]),
                drv.TensorMap(wd, [256, C], 256 * 2, [64, 128]), map_rows(out, A, B, S, SP, AT))
        nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
        ntile = nab * nag * B
        grid = (min(nsm(), ntile), 1, 1)
        rs = min(6, (232448 - 6 * 16384 - 16 * AT * 128 - 1024 - 512) // 16384)   # ring slots that fit next to the tile

        def run():
            self.k(grid, (384, 1, 1), *maps, int(S), int(A), int(B), int(SP), int(AT), int(nab), int(nag), int(ntile), float(EPS), int(save), int(rs),
                   Q1, Att, Y, Ff)
        run.keep = (maps, wab)
        return run, (out, Q1, Att, Y, Ff)


def pack_ffn32(wu):
    """rows per 32-wide hidden chunk j: [Wu[32 j .. 32 j + 31] ; Wu[256 + 32 j ..]] (the backward's a|b tile)."""
    NH = wu.shape[0] // 2
    return wu.view(2, NH // 32, 32, wu.shape[1]).permute(1, 0, 2, 3).reshape(2 * NH, wu.shape[1]).contiguous()


class FfnBwd:
    def __init__(self, cubin=HERE / "build" / "ffn_bwd.cubin"):
        self.k = drv.Kernel(str(cubin), "swa_ffn_bwd_sm100", 232448)

    def bind(self, dq2, q1, y, ffn, mod, wu, wd, A, B, dmod=None, packed=None):
        """dq2 (grad of the block output), q1, y, ffn (forward saves): [N S, C] bf16; mod [B S, 6C] fp32 ->
        run(), (dq1, dffn, h, dab, dmod). dmod [B S, 6C] fp32: given -> accumulated into as is; None -> a fresh one zeroed by run.
        packed: (wab = pack_ffn32(Wu), Wd^T, wab^T) prepared by the caller."""
        M = dq2.numel() // C
        S = M // (A * B)
        SP, AT = tiling(A)
        dev = dq2.device
        dq1, dffn = (torch.empty(M, C, device=dev, dtype=torch.bfloat16) for _ in range(2))
        hh = torch.empty(M, 256, device=dev, dtype=torch.bfloat16)
        dab = torch.empty(M, 512, device=dev, dtype=torch.bfloat16)
        own = dmod is None
        if own:
            dmod = torch.zeros(B * S, 6 * C, device=dev)
        if packed is None:
            wab = pack_ffn32(wu)
            wdt, wabt = wd.t().contiguous(), wab.t().contiguous()
        else:
            wab, wdt, wabt = packed
        maps = (map_rows(y, A, B, S, SP, AT), map_rows(dq2, A, B, S, SP, AT), map_rows(q1, A, B, S, SP, AT),
                drv.TensorMapND(mod, [32, B * S, 24], [6 * C * 4, 128], [32, AT, 8], swizzle=128, dtype="f32"),
                drv.TensorMap(wab, [C, 512], C * 2, [64, 64]), drv.TensorMap(wdt, [C, 256], C * 2, [64, 32]),
                drv.TensorMap(wabt, [512, C], 512 * 2, [64, 128]))
        nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
        ntile = nab * nag * B
        grid = (min(nsm(), ntile), 1, 1)
        rs = min(6, (232448 - 6 * 16384 - 8 * AT * 128 - 1024 - 512) // 16384)

        def run():
            if own:
                dmod.zero_()
            self.k(grid, (384, 1, 1), *maps, int(S), int(A), int(B), int(SP), int(AT), int(nab), int(nag), int(ntile), float(EPS), int(rs),
                   ffn, dq1, dffn, hh, dab, dmod)
        run.keep = (maps, wab, wdt, wabt)
        return run, (dq1, dffn, hh, dab, dmod)


class AttnFwd:
    def __init__(self, cubin=HERE / "build" / "attn_fwd.cubin"):
        self.k = drv.Kernel(str(cubin), "swa_attn_fwd_sm100", 232448)

    def bind(self, Qh, Kh, Vh, seqused):
        """Qh, Kh, Vh head-major [N, H, S, D] bf16; seqused [N] int32 -> run(), O [N S, C] bf16, LSE [N, H, S] fp32 (natural log)."""
        N, _, S, _ = Qh.shape
        assert S % 128 == 0
        O = torch.empty(N * S, C, device=Qh.device, dtype=torch.bfloat16)
        LSE = torch.empty(N, H, S, device=Qh.device)
        rows = N * H * S
        maps = (drv.TensorMap(Qh.view(rows, D), [D, rows], D * 2, [D, 128]), drv.TensorMap(Kh.view(rows, D), [D, rows], D * 2, [D, 64]),
                drv.TensorMap(Vh.view(rows, D), [D, rows], D * 2, [D, 64]), drv.TensorMap(O, [C, N * S], C * 2, [D, 128], swizzle=64))
        items = N * (S // 128) * 2
        grid = (min(nsm(), items), 1, 1)

        def run():
            self.k(grid, (384, 1, 1), *maps, seqused, LSE, int(S), int(N), float(D ** -0.5))
        run.keep = maps
        return run, O, LSE
