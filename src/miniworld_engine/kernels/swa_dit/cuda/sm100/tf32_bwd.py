"""sm_100a (B200) backward of the fused SWA atom DiT block in fp32: hand-written tcgen05 / TMEM / TMA kernels with TF32 tensor-core MMAs
(``kind::tf32``, fp32 accumulation), the fp32 counterparts of the bf16 backward kernels (``sm100.block_bwd``), built from the bf16
kernels' structure and the forward kernels' (``tf32_fwd``) TF32 layouts. Host side: tensor maps, launches, weight forms."""

from __future__ import annotations

import functools

import torch

from miniworld_engine.kernels._device import num_sms
from miniworld_engine.kernels.swa_dit.cuda import sm100 as _sm
from miniworld_engine.kernels.swa_dit.cuda.sm100 import tf32_fwd as _f

C, H, D, NHID = _f.C, _f.H, _f.D, _f.NHID


def _round_cols(t: torch.Tensor) -> torch.Tensor:
    return _f._round_tf32(t.float().t().contiguous())


class WprepBwdTf32:
    """Wd^T [NHID, C] and Wo^T [C, C], TF32-rounded, in one launch (wprep_bwd_tf32.cu): views of one packed buffer."""

    def __init__(self):
        self.k = _f._load("wprep_bwd_tf32")

    def forms(self, wd, wo):
        wd, wo = (w.float().contiguous() for w in (wd, wo))
        out = torch.empty(NHID * C + C * C, device=wd.device, dtype=torch.float32)
        self.k(((out.numel() + 255) // 256, 1, 1), (256, 1, 1), wd, wo, out)
        return out[:NHID * C].view(NHID, C), out[NHID * C:].view(C, C)


def _bwd_forms(wd, wo):
    """(WdT [NHID, C], WoT [C, C]): the backward's K-major B operands, one launch; cached per weight version."""
    with torch.cuda.device(wd.device):
        return kernels_bwd(_index(wd)).wprep.forms(wd, wo)


class FfnBwdGateTf32:
    """y, dq2 [M, C] fp32, mod [B S, 6C] fp32, Wab [2 NHID, C] (TF32-rounded, forward form), WdT [NHID, C] -> DFFN [M, C], HH [M, NHID],
    DAB [M, 2 NHID] (fp32 holding TF32 values)."""

    def __init__(self):
        self.k = _f._load("ffn_bwd_gate_tf32")

    def bind(self, y, dq2, mod, wab, wdt, A, B):
        M = y.numel() // C
        S = M // (A * B)
        SP, AT = _f.tiling(A)
        dev = y.device
        dffn = torch.empty(M, C, device=dev, dtype=torch.float32)
        hh = torch.empty(M, NHID, device=dev, dtype=torch.float32)
        dab = torch.empty(M, 2 * NHID, device=dev, dtype=torch.float32)
        def out_map(t, width):                                              # [A, B, S, width] rows, box (16 channels, AT atoms, 1, SP), SW64
            return _f._f32map(t, [width, S, B, A], [width * 4, S * width * 4, B * S * width * 4], [16, AT, 1, SP], swizzle=64)

        maps = (_f._rows_map(y, A, B, S, SP, AT), _f._rows_map(dq2, A, B, S, SP, AT), _f._mod_map(mod, AT),
                _f._f32map(wab, [C, 2 * NHID], C * 4, [32, 64]), _f._f32map(wdt, [C, NHID], C * 4, [32, 32]),
                out_map(dffn, C), out_map(hh, NHID), out_map(dab, 2 * NHID))
        nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
        ntile = nab * nag * B
        grid = (min(num_sms(), ntile), 1, 1)

        def run():
            self.k(grid, (384, 1, 1), *maps, int(S), int(A), int(B), int(SP), int(AT), int(nab), int(nag), int(ntile))
        run.keep = (maps, mod, y, dq2, dffn, hh, dab)
        return run, (dffn, hh, dab)


class AdalnBwdTf32:
    """The RMS-adaLN backward after the GEMM that gives dy (adaln_bwd_tf32.cu): x, dy, res [M, C] fp32, gin [M, C] fp32 or None, mod
    [B S, 6C] -> out [M, C] = res + the norm's backward, and the modulation gradient written into dmod [B S, 6C] columns out_col
    (shift), out_col + C (scale: also where mod's scale is read), out_col + 2 C (gate, with gin) by plain stores."""

    def __init__(self):
        self.k = _f._load("adaln_bwd_tf32")

    def bind(self, x, dy, res, gin, mod, dmod, A, B, eps, out_col, att=None, att_col=0):
        M = x.numel() // C
        S = M // (A * B)
        out = torch.empty(M, C, device=x.device, dtype=torch.float32)

        def run():
            self.k((B * S, 1, 1), (256, 1, 1), x, dy, res, gin if gin is not None else 0, mod, out, dmod, int(S), int(A), int(B),
                   float(eps), int(out_col),
                   att if att is not None else 0, int(att_col))
        run.keep = (x, dy, res, gin, mod, dmod, att)
        return run, out


class OprojBwdFusedTf32:
    """The whole out-projection backward in one kernel (oproj_bwd_fused_tf32.cu): dq1, o, g [M, C] fp32, mod [B S, 6C] fp32, Wo^T [C, C] (TF32-
    rounded, K-major B operand) -> datt, gated, dO [M, C] (TF32-rounded), Dv [N, H, S] fp32, and dG written into dP [M, 4C] at column 3C."""

    def __init__(self):
        self.k = _f._load("oproj_bwd_fused_tf32")

    def bind(self, dq1, o, g, mod, wot, dp, A, B):
        M = dq1.numel() // C
        S = M // (A * B)
        SP, AT = _f.tiling(A)
        dev = dq1.device
        datt, gated, do = (torch.empty(M, C, device=dev, dtype=torch.float32) for _ in range(3))
        dv = torch.empty(M // S, H, S, device=dev, dtype=torch.float32)
        mdp = _f._f32map(dp, [4 * C, S, B, A], [4 * C * 4, S * 4 * C * 4, B * S * 4 * C * 4], [32, AT, 1, SP])
        maps = (_f._rows_map(dq1, A, B, S, SP, AT), _f._mod_map(mod, AT), _f._rows_map(o, A, B, S, SP, AT), _f._rows_map(g, A, B, S, SP, AT),
                _f._f32map(wot, [C, C], C * 4, [32, 128]), _f._out_map(datt, A, B, S, SP, AT), _f._rows_map(gated, A, B, S, SP, AT),
                _f._rows_map(do, A, B, S, SP, AT), mdp)
        nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
        ntile = nab * nag * B
        grid = (min(num_sms(), ntile), 1, 1)

        def run():
            self.k(grid, (384, 1, 1), *maps, dv, int(S), int(A), int(B), int(SP), int(AT), int(nab), int(nag), int(ntile))
        run.keep = (maps, dq1, o, g, mod, wot, dp, datt, gated, do, dv)
        return run, (datt, gated, do, dv)


class ModBwdTf32:
    """silu / silu' of the hoisted modulation's backward (mod_bwd_tf32.cu)."""

    def __init__(self):
        self.ks = _f._load("mod_bwd_tf32", "swa_mod_silu_tf32_sm100")
        self.kd = _f._load("mod_bwd_tf32", "swa_mod_dsilu_tf32_sm100")

    def silu(self, c):
        R = c.shape[0]
        a = torch.empty_like(c)
        self.ks(((R + 7) // 8, 1, 1), (256, 1, 1), c, a, int(R))
        return a

    def dsilu(self, acc, c):
        R = c.shape[0]
        dc = torch.empty_like(c)
        self.kd(((R + 7) // 8, 1, 1), (256, 1, 1), acc, c, dc, int(R))
        return dc


class AttnBwdTf32:
    """Window-attention backward with the head-wise tail of the qkvg backward in its epilogues (attn_dq_tf32.cu, attn_dkv_tf32.cu): Qh, Kh, Vh
    head-major [N, H, S, D] fp32 (TF32-valued), dO [N S, C] fp32 (TF32-valued), lse and Dv [N, H, S] fp32, seqused [N] int32, the raw
    projections pq / pk [N S, C] and cos / sin [B S, D/2] -> dpq | dpk | dpv written into dP [N S, 4C] columns 0 .. 3C (TF32-rounded; dpq: the
    dQ pass, dpk and dpv: the dK / dV pass; column 3C .. 4C is dG, written by the out-projection kernel)."""

    def __init__(self):
        self.kq = _f._load("attn_dq_tf32")
        self.kk = _f._load("attn_dkv_tf32")

    @staticmethod
    def _mat(t, rows, cols, box_rows, swizzle=128):
        return _f._f32map(t, [cols, rows], cols * 4, [32, box_rows], swizzle=swizzle)

    def bind(self, qh, kh, vh, do, lse, dv, seqused, pq, pk, cos, sin, dp, B):
        N, _, S, _ = qh.shape
        assert S % 128 == 0 and dp.shape == (N * S, 4 * C)
        rows = N * H * S
        q2, k2, v2 = (t.view(rows, D) for t in (qh, kh, vh))
        mdp = self._mat(dp, N * S, 4 * C, 128)
        mq = (self._mat(q2, rows, D, 128), self._mat(do, N * S, C, 128), self._mat(k2, rows, D, 32), self._mat(v2, rows, D, 32),
              self._mat(k2, rows, D, 32, "128a32"))
        mk = (self._mat(q2, rows, D, 32), self._mat(q2, rows, D, 32, "128a32"), self._mat(k2, rows, D, 128), self._mat(v2, rows, D, 128),
              self._mat(do, N * S, C, 32), self._mat(do, N * S, C, 32, "128a32"))
        grid = (min(num_sms(), N * (S // 128) * H), 1, 1)
        scale = float(D ** -0.5)

        def run():
            self.kq(grid, (512, 1, 1), *mq, lse, dv, seqused, pq, cos, sin, mdp, int(S), int(N), int(B), scale, float(_f.EPS))
            self.kk(grid, (512, 1, 1), *mk, lse, dv, seqused, pk, cos, sin, mdp, int(S), int(N), int(B), scale, float(_f.EPS))
        run.keep = (mq, mk, mdp, qh, kh, vh, do, lse, dv, seqused, pq, pk, cos, sin, dp)
        return run


class DwTf32:
    """The five weight gradients in one launch (dw_tf32.cu) + a deterministic reduction of the split-K partial sums (dw_reduce_tf32):
    dP [M, 4C], x [M, C], datt, gated [M, C], dab [M, 2 NHID], y [M, C], dffn [M, C], hh [M, NHID] (all row-major fp32) ->
    dP^T x [4C, C], datt^T gated [C, C], dab^T y [2 NHID, C], dffn^T hh [C, NHID]. A CTA takes a row slice of one problem; the CTAs go to
    the problems in proportion to their bytes (0.297 | 0.189 | 0.297 | 0.216, measured on B200: 44 / 28 / 44 / 32 of 148)."""

    FRAC = (44 / 148, 28 / 148, 44 / 148, 32 / 148)
    SHAPES = ((4 * C, C), (C, C), (2 * NHID, C), (C, NHID))

    def __init__(self):
        self.kd = _f._load("dw_tf32")
        self.kr = _f._load("dw_reduce_tf32")

    def bind(self, dp, x, datt, gated, dab, y, dffn, hh):
        M = dp.shape[0]
        pairs = ((dp, x), (datt, gated), (dab, y), (dffn, hh))
        for a, b in pairs:
            assert a.is_contiguous() and b.is_contiguous() and a.shape[0] == M and b.shape[0] == M and M % 16 == 0
        nsm = num_sms()
        ns = tuple(max(1, min(round(f * nsm), M // 256)) for f in self.FRAC)
        dev = dp.device
        maps = []
        for a, b in pairs:
            maps += [AttnBwdTf32._mat(a, M, a.view(M, -1).shape[1], 16, "128a32"), AttnBwdTf32._mat(b, M, b.view(M, -1).shape[1], 16, "128a32")]
        ws = torch.empty(sum(n * m * k for n, (m, k) in zip(ns, self.SHAPES)), device=dev, dtype=torch.float32)
        outs = tuple(torch.empty(m, k, device=dev, dtype=torch.float32) for m, k in self.SHAPES)
        total4 = sum(m * k // 4 for m, k in self.SHAPES)

        def run():
            self.kd((sum(ns), 1, 1), (256, 1, 1), *maps, ws, int(M), *(int(n) for n in ns))
            self.kr(((total4 + 255) // 256, 1, 1), (256, 1, 1), ws, *outs, *(int(n) for n in ns))
        run.keep = (maps, ws, dp, x, datt, gated, dab, y, dffn, hh)
        return run, outs


class _Kernels:
    def __init__(self) -> None:
        self.wprep = WprepBwdTf32()
        self.mod = ModBwdTf32()
        self.attn = AttnBwdTf32()
        self.ffn_gate = FfnBwdGateTf32()
        self.adaln = AdalnBwdTf32()
        self.oproj_fused = OprojBwdFusedTf32()
        self.dw = DwTf32()


@functools.lru_cache(maxsize=16)
def kernels_bwd(index: int) -> _Kernels:
    """Build and load every TF32 backward kernel on device ``index``."""
    with torch.cuda.device(index):
        torch.zeros(1, device=f"cuda:{index}")             # a torch CUDA call first: autograd's backward thread has no current context
        return _Kernels()


def _index(t: torch.Tensor) -> int:
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


def mm_tf32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b of fp32 operands on the TF32 tensor cores (cuBLAS; operands that are TF32 values already, as the kernels here write them,
    are exact): fp32 sums, ~3e-4 relative to fp64 for random operands, the precision of the TF32 forward kernels."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        return torch.mm(a, b)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def mod_bwd_tf32(g: torch.Tensor, c: torch.Tensor, wmod: torch.Tensor):
    """(dc [R, C], dWmod [6C, C]) fp32 of mod = silu(c) Wmod^T from g = d mod [R, 6C] fp32: dc = (g Wmod) silu'(c) and dWmod = g^T silu(c),
    both GEMMs on cuBLAS TF32, the silu terms on mod_bwd_tf32.cu."""
    with torch.cuda.device(c.device):
        K = kernels_bwd(_index(c))
        a = K.mod.silu(c)
        dc = K.mod.dsilu(mm_tf32(g, wmod), c)
        return dc, mm_tf32(g.t(), a)


def block_bwd_tf32(dy, q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, qh, kh, vh, g, o, lse, q1, x, pq, pk, att, y, ffn, B,
                   half_window=_f.HW):
    """The fp32 backward of ``tf32_fwd.block_fwd_tf32``'s saves on the TF32 kernels: [dq, dmod, dWqkv, dWg, dWo, dWu, dWd] -- the layout
    of ``dispatch.swa_dit_block_bwd``. FFN (gate kernel, dy = [da | db] Wu on cuBLAS TF32, adaLN backward) -> out-projection (prep kernel,
    dgated = datt Wo, post kernel) -> window-attention backward (dQ and dK / dV TF32 kernels) -> qkvg (prep kernel,
    dx = dP [Wqkv; Wg], adaLN backward) -> the five weight gradients as cuBLAS TF32 GEMMs over TF32-valued operands. Each modulation
    gradient column is written by exactly one kernel (plain stores, no atomics, bit-reproducible)."""
    N, S, _ = q.shape
    _sm.need_s128(S)
    M = N * S
    A = N // B
    dev = q.device
    eps = _f.EPS
    with torch.cuda.device(dev):
        K = kernels_bwd(_index(q))
        w, _, wab, _ = _sm._cached(_f._tf32_block_forms, wqkv, wg, wo, wu, wd)
        wdt, wot = _sm._cached(_bwd_forms, wd, wo)
        dmod = torch.empty(B * S, 6 * C, device=dev, dtype=torch.float32)
        # FFN
        dq2 = dy.reshape(M, C)
        run, (dffn, hh, dab) = K.ffn_gate.bind(y, dq2, mod, wab, wdt, A, B)
        run()
        run, dq1 = K.adaln.bind(q1, mm_tf32(dab, wu), dq2, ffn, mod, dmod, A, B, eps, 3 * C, att, 2 * C)   # + d gate_a = sum_aug dq1 att
        run()
        # out-projection: one kernel (datt, dgated = datt Wo, dO, dG, gated, Dv)
        dP = torch.empty(M, 4 * C, device=dev, dtype=torch.float32)       # [dpq | dpk | dpv | dg]
        run, (datt, gated, dO, Dv) = K.oproj_fused.bind(dq1, o, g, mod, wot, dP, A, B)
        run()
        # window attention (TF32 kernels, fp32 operands: the forward's saves as they are)
        # (the epilogues run the head-wise tail of the qkvg backward and write dpq | dpk | dpv into dP)
        run = K.attn.bind(qh, kh, vh, dO, lse, Dv, seqused, pq, pk, cos, sin, dP, B)
        run()
        # qkvg
        run, dq = K.adaln.bind(q.reshape(M, C), mm_tf32(dP, w), dq1, None, mod, dmod, A, B, eps, 0)
        run()
        # the five weight gradients: one launch (dWqkv | dWg is one problem over dP, x read once) + the fixed-order reduction
        run, (dwqkvg, dwo, dwu, dwd) = K.dw.bind(dP, x.view(M, C), datt, gated.view(M, C), dab, y.view(M, C), dffn, hh)
        run()
        return [dq.view(N, S, C), dmod, dwqkvg[:3 * C], dwqkvg[3 * C:].clone(), dwo, dwu, dwd]
