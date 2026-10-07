"""sm_100a (B200) backward of the fused SWA atom DiT block in fp32: hand-written tcgen05 / TMEM / TMA kernels with TF32 tensor-core MMAs
(``kind::tf32``, fp32 accumulation), the fp32 counterparts of the bf16 backward kernels (``sm100.block_bwd``), built from the bf16
kernels' structure and the forward kernels' (``tf32_fwd``) TF32 layouts. Host side: tensor maps, launches, weight forms."""

from __future__ import annotations

import functools

import torch

from miniworld_engine.kernels.swa_dit.cuda import sm100 as _sm
from miniworld_engine.kernels.swa_dit.cuda.sm100 import tf32_fwd as _f

C, H, D, NHID = _f.C, _f.H, _f.D, _f.NHID


def _round_cols(t: torch.Tensor) -> torch.Tensor:
    return _f._round_tf32(t.float().t().contiguous())


def _ffn_bwd_forms(wu, wd):
    """(WdT [NHID, C], WuT [C, 2 NHID]): Wd^T and Wu^T rounded to TF32, K-major B operands of the FFN backward."""
    return _round_cols(wd), _round_cols(wu)


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
        grid = (min(_sm.nsm(), ntile), 1, 1)

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

    def bind(self, x, dy, res, gin, mod, dmod, A, B, eps, out_col):
        M = x.numel() // C
        S = M // (A * B)
        out = torch.empty(M, C, device=x.device, dtype=torch.float32)

        def run():
            self.k((B * S, 1, 1), (256, 1, 1), x, dy, res, gin if gin is not None else 0, mod, out, dmod, int(S), int(A), int(B),
                   float(eps), int(out_col))
        run.keep = (x, dy, res, gin, mod, dmod)
        return run, out


class OprojBwdTf32:
    """The out-projection backward around dgated = datt Wo: :meth:`prep` (dq1, o, g, att, mod -> datt, gated [M, C] TF32-rounded and
    dmod column 2C) and :meth:`post` (dgated, o, g -> dO [M, C] (fp32 TF32-rounded, or bf16), Dv [N, H, S] fp32, and dG into dP [M, 4C])."""

    def __init__(self):
        self.kp = _f._load("oproj_bwd_tf32", "swa_oproj_bwd_prep_tf32_sm100")
        self.kf = _f._load("oproj_bwd_tf32", "swa_oproj_bwd_post_tf32_sm100")
        self.kb = _f._load("oproj_bwd_tf32", "swa_oproj_bwd_post_bf16o_tf32_sm100")

    def prep(self, dq1, o, g, att, mod, dmod, A, B):
        M = dq1.numel() // C
        S = M // (A * B)
        datt = torch.empty(M, C, device=dq1.device, dtype=torch.float32)
        gated = torch.empty_like(datt)

        def run():
            self.kp((B * S, 1, 1), (256, 1, 1), dq1, o, g, att, mod, datt, gated, dmod, int(S), int(A), int(B))
        run.keep = (dq1, o, g, att, mod, dmod)
        return run, (datt, gated)

    def post(self, dgated, o, g, S, bf16_do, dp):
        """dp [M, 4C] fp32 receives dG (TF32-rounded) at columns 3C .. 4C."""
        M = dgated.numel() // C
        N = M // S
        dev = dgated.device
        dout = torch.empty(M, C, device=dev, dtype=torch.bfloat16 if bf16_do else torch.float32)
        dv = torch.empty(N, H, S, device=dev, dtype=torch.float32)
        k = self.kb if bf16_do else self.kf

        def run():
            k(((M + 7) // 8, 1, 1), (256, 1, 1), dgated, o, g, dout, dp, dv, int(S), int(M))
        run.keep = (dgated, o, g, dp)
        return run, (dout, dv)


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
        grid = (min(_sm.nsm(), N * (S // 128) * H), 1, 1)
        scale = float(D ** -0.5)

        def run():
            self.kq(grid, (384, 1, 1), *mq, lse, dv, seqused, pq, cos, sin, mdp, int(S), int(N), int(B), scale, float(_f.EPS))
            self.kk(grid, (384, 1, 1), *mk, lse, dv, seqused, pk, cos, sin, mdp, int(S), int(N), int(B), scale, float(_f.EPS))
        run.keep = (mq, mk, mdp, qh, kh, vh, do, lse, dv, seqused, pq, pk, cos, sin, dp)
        return run


class _Kernels:
    def __init__(self) -> None:
        self.mod = ModBwdTf32()
        self.attn = AttnBwdTf32()
        self.ffn_gate = FfnBwdGateTf32()
        self.adaln = AdalnBwdTf32()
        self.oproj = OprojBwdTf32()


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
        w, wo_t, wab, _ = _sm._cached(_f._tf32_block_forms, wqkv, wg, wo, wu, wd)
        wdt, _ = _sm._cached(_ffn_bwd_forms, wu, wd)
        dmod = torch.empty(B * S, 6 * C, device=dev, dtype=torch.float32)
        # FFN
        dq2 = dy.reshape(M, C)
        run, (dffn, hh, dab) = K.ffn_gate.bind(y, dq2, mod, wab, wdt, A, B)
        run()
        run, dq1 = K.adaln.bind(q1, mm_tf32(dab, wu), dq2, ffn, mod, dmod, A, B, eps, 3 * C)
        run()
        # out-projection
        run, (datt, gated) = K.oproj.prep(dq1, o, g, att, mod, dmod, A, B)
        run()
        dP = torch.empty(M, 4 * C, device=dev, dtype=torch.float32)       # [dpq | dpk | dpv | dg]
        run, (dO, Dv) = K.oproj.post(mm_tf32(datt, wo_t), o, g, S, False, dP)
        run()
        # window attention (TF32 kernels, fp32 operands: the forward's saves as they are)
        # (the epilogues run the head-wise tail of the qkvg backward and write dpq | dpk | dpv into dP)
        run = K.attn.bind(qh, kh, vh, dO, lse, Dv, seqused, pq, pk, cos, sin, dP, B)
        run()
        # qkvg
        run, dq = K.adaln.bind(q.reshape(M, C), mm_tf32(dP, w), dq1, None, mod, dmod, A, B, eps, 0)
        run()
        dwqkvg = mm_tf32(dP.t(), x)                        # dWqkv | dWg as one GEMM over dP (x read once)
        return [dq.view(N, S, C), dmod, dwqkvg[:3 * C], dwqkvg[3 * C:].clone(), mm_tf32(datt.t(), gated), mm_tf32(dab.t(), y),
                mm_tf32(dffn.t(), hh)]
