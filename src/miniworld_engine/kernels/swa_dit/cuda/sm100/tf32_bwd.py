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
        return out[: NHID * C].view(NHID, C), out[NHID * C :].view(C, C)


def _bwd_forms(wd, wo):
    """(WdT [NHID, C], WoT [C, C]): the backward's K-major B operands, one launch; cached per weight version."""
    with torch.cuda.device(wd.device):
        return kernels_bwd(_index(wd)).wprep.forms(wd, wo)


class FfnBwdGateTf32:
    """y, dq2 [M, C] fp32, mod [B S, 6C] fp32, Wab [2 NHID, C] (TF32-rounded, forward form), WdT [NHID, C] -> DFFN [M, C], HH [M, NHID],
    DAB [M, 2 NHID] (fp32 holding TF32 values)."""

    NGMAX = 8  # development switch (1: one item per tile)

    def __init__(self):
        self.k = _f._load("ffn_bwd_gate_tf32")
        self.kdy = None  # the DYF build (dy partials inside), loaded on first use

    def bind(self, y, dq2, mod, wab, wdt, A, B, dyf=False, late_trigger=False):
        """``late_trigger`` (r3-swa part 19): the dependents triggered after the kernel's wait (the next launch is ``sm100.save32_late``).
        ``dyf`` (round 3 part 6): the DYF build -- also DYP [NG, M, C] fp32, the items' partial dy = [da | db] Wab (sum over NG in
        group order: ``AdalnBwdTf32.bind(ndy=NG)``); returns (dffn, hh, dab, dyp) then."""
        M = y.numel() // C
        S = M // (A * B)
        SP, AT = _f.tiling(A)
        dev = y.device
        dffn = torch.empty(M, C, device=dev, dtype=torch.float32)
        hh = torch.empty(M, NHID, device=dev, dtype=torch.float32)
        dab = torch.empty(M, 2 * NHID, device=dev, dtype=torch.float32)

        def out_map(t, width):  # [A, B, S, width] rows, box (16 channels, AT atoms, 1, SP), SW64
            return _f._f32map(t, [width, S, B, A], [width * 4, S * width * 4, B * S * width * 4], [16, AT, 1, SP], swizzle=64)

        nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
        ntile = nab * nag * B
        NG = 1  # chunk groups: few tiles spread their hidden chunks over the SMs (ffn_bwd_gate_tf32.cu)
        while NG < 8 and ntile * NG * 2 <= num_sms():
            NG *= 2
        NG = min(NG, self.NGMAX)
        k = self.k
        if dyf:
            if self.kdy is None:
                self.kdy = _f._load("ffn_bwd_gate_tf32_dyf")
            k = self.kdy
            dyp = torch.empty(NG, M, C, device=dev, dtype=torch.float32)
            W5 = [C * 4, S * C * 4, B * S * C * 4, A * B * S * C * 4]
            extra = (
                _f._f32map(wab, [C, 2 * NHID], C * 4, [32, 64], swizzle="128a32"),  # Wab[j] as the MN-major dy operand
                _f._f32map(dyp, [C, S, B, A, NG], W5, [16, AT, 1, SP, 1], swizzle=64),
            )
        maps = (
            _f._rows_map_any(y, A, B, S, SP, AT),
            _f._rows_map_any(dq2, A, B, S, SP, AT),
            _f._mod_map(mod, AT),
            _f._f32map(wab, [C, 2 * NHID], C * 4, [32, 64]),
            _f._f32map(wdt, [C, NHID], C * 4, [32, 32]),
            out_map(dffn, C),
            out_map(hh, NHID),
            out_map(dab, 2 * NHID),
        )
        maps = (*maps, *(extra if dyf else maps[-2:]))  # (the plain build ignores the last two)
        grid = (min(num_sms(), ntile * NG), 1, 1)

        def run():
            k(
                grid,
                (384, 1, 1),
                *maps,
                int(S),
                int(A),
                int(B),
                int(SP),
                int(AT),
                int(nab),
                int(nag),
                int(ntile),
                int(NG),
                _f._b16(dq2) | (_f._b16(y) << 1) | (int(late_trigger) << 2),
            )

        run.keep = (maps, mod, y, dq2, dffn, hh, dab)
        if dyf:
            run.keep += (dyp,)
            return run, (dffn, hh, dab, dyp)
        return run, (dffn, hh, dab)


class AdalnBwdTf32:
    """The RMS-adaLN backward after the GEMM that gives dy (adaln_bwd_tf32.cu): x, dy, res [M, C] fp32, gin [M, C] fp32 or None, mod
    [B S, 6C] -> out [M, C] = res + the norm's backward, and the modulation gradient written into dmod [B S, 6C] columns out_col
    (shift), out_col + C (scale: also where mod's scale is read), out_col + 2 C (gate, with gin) by plain stores."""

    def __init__(self):
        self.k = _f._load("adaln_bwd_tf32")
        self.k1 = _f._load("adaln_bwd_tf32", "swa_adaln_bwd1_tf32_sm100")  # one augment per atom: a warp per row
        # round 3 part 6 (master training, bf16 in place): res bf16 (the FFN norm: res = dq2) / x and out bf16 (the qkvg norm)
        self.kv = {
            sfx: (
                _f._load("adaln_bwd_tf32", f"swa_adaln_bwd_tf32{sfx}_sm100"),
                _f._load("adaln_bwd_tf32", f"swa_adaln_bwd1_tf32{sfx}_sm100"),
            )
            for sfx in ("_r16", "_xo16")
        }

    def bind(self, x, dy, res, gin, mod, dmod, A, B, eps, out_col, att=None, att_col=0, ndy=1):
        """``ndy`` (round 3 part 6): dy [ndy, M, C] partials summed in order (``FfnBwdGateTf32.bind(dyf=True)``)."""
        M = x.numel() // C
        assert dy.numel() == ndy * M * C and dy.is_contiguous()
        S = M // (A * B)
        out = torch.empty(M, C, device=x.device, dtype=x.dtype)
        b16 = torch.bfloat16
        assert dy.dtype == torch.float32 and (x.dtype != b16 or res.dtype != b16)
        k, k1 = self.kv["_xo16"] if x.dtype == b16 else self.kv["_r16"] if res.dtype == b16 else (self.k, self.k1)

        def run():
            if A == 1:
                g0, a0 = gin, att  # None -> a null pointer of 8 bytes (an int 0 is passed as 4)
                k1(
                    ((M + 7) // 8, 1, 1),
                    (256, 1, 1),
                    x,
                    dy,
                    res,
                    g0,
                    mod,
                    out,
                    dmod,
                    int(M),
                    float(eps),
                    int(out_col),
                    a0,
                    int(att_col),
                    int(ndy),
                    M * C if ndy > 1 else 0,
                )
                return
            k(
                (B * S, 1, 1),
                (256, 1, 1),
                x,
                dy,
                res,
                gin,  # None -> a null pointer of 8 bytes
                mod,
                out,
                dmod,
                int(S),
                int(A),
                int(B),
                float(eps),
                int(out_col),
                att,
                int(att_col),
                int(ndy),
                M * C if ndy > 1 else 0,
            )

        run.keep = (x, dy, res, gin, mod, dmod, att)
        return run, out


class OprojBwdFusedTf32:
    """The whole out-projection backward in one kernel (oproj_bwd_fused_tf32.cu): dq1, o, g [M, C] fp32, mod [B S, 6C] fp32, Wo^T [C, C] (TF32-
    rounded, K-major B operand) -> datt, gated, dO [M, C] (TF32-rounded), Dv [N, H, S] fp32, and dG written into dP [M, 4C] at column 3C."""

    NGMAX = 4  # development switch (1: one item per tile)

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
        maps = (
            _f._rows_map(dq1, A, B, S, SP, AT),
            _f._mod_map(mod, AT),
            _f._rows_map(o, A, B, S, SP, AT),
            _f._rows_map(g, A, B, S, SP, AT),
            _f._f32map(wot, [C, C], C * 4, [32, 128]),
            _f._out_map(datt, A, B, S, SP, AT),
            _f._rows_map(gated, A, B, S, SP, AT),
            _f._rows_map(do, A, B, S, SP, AT),
            mdp,
        )
        nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
        ntile = nab * nag * B
        NG = 1  # head groups: few tiles spread their heads over the SMs (oproj_bwd_fused_tf32.cu)
        while NG < 4 and ntile * NG * 2 <= num_sms():
            NG *= 2
        NG = min(NG, self.NGMAX)
        grid = (min(num_sms(), ntile * NG), 1, 1)

        def run():
            self.k(grid, (384, 1, 1), *maps, dv, int(S), int(A), int(B), int(SP), int(AT), int(nab), int(nag), int(ntile), int(NG))

        run.keep = (maps, dq1, o, g, mod, wot, dp, datt, gated, do, dv)
        return run, (datt, gated, do, dv)


class ModBwdTf32:
    """silu / silu' of the hoisted modulation's backward (mod_bwd_tf32.cu)."""

    def __init__(self):
        self.ks = _f._load("mod_bwd_tf32", "swa_mod_silu_tf32_sm100")
        self.kd = _f._load("mod_bwd_tf32", "swa_mod_dsilu_tf32_sm100")
        # c bf16 (round 3 part 6, master training): read in place, dc in bf16
        self.ks16 = _f._load("mod_bwd_tf32", "swa_mod_silu_tf32_c16_sm100")
        self.kd16 = _f._load("mod_bwd_tf32", "swa_mod_dsilu_tf32_c16_sm100")
        # round 3 part 14: dc summed over a stack's blocks inside the kernel (DcAcc)
        self.kda = _f._load("mod_bwd_tf32", "swa_mod_dsilu_acc_sm100")

    def silu(self, c):
        R = c.shape[0]
        a = torch.empty_like(c, dtype=torch.float32)
        (self.ks16 if c.dtype == torch.bfloat16 else self.ks)(((R + 7) // 8, 1, 1), (256, 1, 1), c, a, int(R))
        return a

    def dsilu(self, acc, c):
        """``acc`` [R, C] or [np, R, C] partials (summed in order: :func:`mm_tf32_ks`)."""
        R = c.shape[0]
        dc = torch.empty_like(c)
        np_ = acc.shape[0] if acc.dim() == 3 else 1
        assert acc.is_contiguous()
        (self.kd16 if c.dtype == torch.bfloat16 else self.kd)(((R + 7) // 8, 1, 1), (256, 1, 1), acc, c, dc, int(R), np_, R * C)
        return dc

    def dsilu_acc(self, acc, c, prev, out_dtype, xadd=None):
        """(round 3 part 14) out = (prev + dc) [+ xadd] in ``out_dtype`` (fp32 sums, one rounding); ``prev`` None or fp32 [R, C]."""
        R = c.shape[0]
        out = torch.empty(c.shape, device=c.device, dtype=out_dtype)
        np_ = acc.shape[0] if acc.dim() == 3 else 1
        assert acc.is_contiguous() and c.is_contiguous() and (xadd is None or (xadd.is_contiguous() and xadd.numel() == c.numel()))
        b16 = torch.bfloat16
        self.kda(((R + 7) // 8, 1, 1), (256, 1, 1), acc, c, prev, out, xadd, int(R), np_, R * C, int(c.dtype == b16), int(out_dtype == b16),
                 int(xadd is not None and xadd.dtype == b16))
        return out


class DcAcc:
    """B200 round 3 part 14: the c gradient of a stack's blocks that share one c, summed inside the blocks' dsilu launches (the backward runs
    the last block first, the first block last: the dependency through x orders them) instead of autograd's adds and casts. ``n`` blocks;
    block ``i``'s backward adds its dc to the running fp32 sum (``buf``) -- block n - 1 starts it, block 0 ends it in ``dtype`` -- and with
    ``fold`` (the stack's x and c are one tensor) block 0 also adds its x gradient and returns the whole as x's gradient (c's None)."""

    def __init__(self, n: int, dtype: torch.dtype, fold: bool):
        self.n, self.dtype, self.fold = n, dtype, fold
        self.buf: torch.Tensor | None = None
        self.keep: list = []  # what a side-stream step reads, alive until the last step joins that stream
        self.ev = None
        self.held: list = []  # events of deferred side streams (hold), waited by the last step

    def hold(self, streams, keep):
        """Defer the join of ``streams`` to the last step (block 0's dc launch); ``keep`` alive until then."""
        for s in streams:
            ev = torch.cuda.Event()
            ev.record(s)
            self.held.append(ev)
        self.keep += keep

    def step(self, K, i: int, make_acc, c, dq, dmod=None):
        """Block ``i``'s dc launches (``make_acc()``: the GEMM g Wmod, its partials); returns (x gradient, c gradient) for that block's
        Function. Every block but the first runs them on a stream of their own (DEFER: nothing of the stack's x-gradient chain waits for
        them -- the next block's backward starts after this block's dq); the first block's (the last step) joins that stream."""
        first, last = i == self.n - 1, i == 0
        if not last:
            if DEFER_DC:
                cur = torch.cuda.current_stream()
                s = _dc_stream(cur.device.index)
                s.wait_stream(cur)
                with torch.cuda.stream(s):
                    acc = make_acc()
                    self.buf = K.mod.dsilu_acc(acc, c, None if first else self.buf, torch.float32)
                self.keep += [dmod, c, acc]  # (acc may be allocated on the current stream: _bwd_few's early half)
                self.ev = torch.cuda.Event()
                self.ev.record(s)
            else:
                self.buf = K.mod.dsilu_acc(make_acc(), c, None if first else self.buf, torch.float32)
            return dq, None
        if self.ev is not None:
            torch.cuda.current_stream().wait_event(self.ev)
        for ev in self.held:  # the deferred weight-gradient streams of the later blocks (hold)
            torch.cuda.current_stream().wait_event(ev)
        self.held = []
        prev, self.buf, self.ev = (None if first else self.buf), None, None
        acc = make_acc()
        self.keep = []
        if self.fold:
            return K.mod.dsilu_acc(acc, c, prev, dq.dtype, dq.reshape(c.shape)).view(dq.shape), None
        return dq, K.mod.dsilu_acc(acc, c, prev, self.dtype)

    def late(self, i: int, run, dq):
        """:meth:`step` for :class:`DcgTf32`'s late launch (r3-swa part 19): ``run(prev, out_dtype, xadd)`` is the GEMM's last third with the
        silu' and the running sum in one launch -- on the dc stream for every block but the first (DEFER_DC, as :meth:`step`)."""
        first, last = i == self.n - 1, i == 0
        if not last:
            prev = None if first else self.buf
            if DEFER_DC:
                cur = torch.cuda.current_stream()
                s = _dc_stream(cur.device.index)
                s.wait_stream(cur)
                with torch.cuda.stream(s):
                    self.buf = run(prev, torch.float32, None)
                self.keep += [run.keep]
                self.ev = torch.cuda.Event()
                self.ev.record(s)
            else:
                self.buf = run(prev, torch.float32, None)
            return dq, None
        cur = torch.cuda.current_stream()
        if self.ev is not None:
            cur.wait_event(self.ev)
        for ev in self.held:
            cur.wait_event(ev)
        self.held = []
        prev, self.buf, self.ev = (None if first else self.buf), None, None
        self.keep = []
        if self.fold:
            return run(prev, dq.dtype, dq.reshape(-1, C)).view(dq.shape), None
        return dq, run(prev, self.dtype, None)

    def fused(self, i: int, run, dq):
        """:meth:`step` for the one-kernel tail (round 3 part 16, ``DcxTf32``): ``run(prev, out_dtype, fold)`` makes this block's dc sum on
        the current stream (no side stream: the kernel that makes dq makes it). Returns (x gradient, c gradient)."""
        first, last = i == self.n - 1, i == 0
        cur = torch.cuda.current_stream()
        if self.ev is not None:  # (a block before on the side-stream form)
            cur.wait_event(self.ev)
            self.ev = None
        prev = None if first else self.buf
        if not last:
            self.buf = run(prev, torch.float32, False)
            return dq, None
        for ev in self.held:
            cur.wait_event(ev)
        self.held, self.buf, self.keep = [], None, []
        if self.fold:
            return run(prev, dq.dtype, True).view(dq.shape), None
        return dq, run(prev, self.dtype, False)


#: round 3 part 14 (development switch): a stack's non-final dc launches on a stream of their own (DcAcc.step)
DEFER_DC = True


@functools.lru_cache(maxsize=16)
def _dc_stream(index: int):
    with torch.cuda.device(index):
        return torch.cuda.Stream()


class AttnBwdTf32:
    """Window-attention backward with the head-wise tail of the qkvg backward in its epilogues (attn_dq_tf32.cu, attn_dkv_tf32.cu): Qh, Kh, Vh
    head-major [N, H, S, D] fp32 (TF32-valued), dO [N S, C] fp32 (TF32-valued), lse and Dv [N, H, S] fp32, seqused [N] int32, the raw
    projections pq / pk [N S, C] and cos / sin [B S, D/2] -> dpq | dpk | dpv written into dP [N S, 4C] columns 0 .. 3C (TF32-rounded; dpq: the
    dQ pass, dpk and dpv: the dK / dV pass; column 3C .. 4C is dG, written by the out-projection kernel)."""

    def __init__(self):
        self.kq = _f._load("attn_dq_tf32")
        self.kk = _f._load("attn_dkv_tf32")
        self.kqx = self.kkx = None

    @staticmethod
    def _mat(t, rows, cols, box_rows, swizzle=128):
        return _f._f32map(t, [cols, rows], cols * 4, [32, box_rows], swizzle=swizzle)

    @staticmethod
    def dxp_ok(N, S) -> bool:
        """r3-swa part 21: the DXP builds need one item a CTA (both passes' grids = their items)."""
        return N * (S // 128) * H <= num_sms()

    def bind(self, qh, kh, vh, do, lse, dv, seqused, pq, pk, cos, sin, dp, B, w=None, xp=None):
        """``w`` [4C, C] fp32 (TF32-valued) and ``xp`` [8, N S, C] fp32 (r3-swa part 21, DXP): the passes also write dx = dP w as eight
        per-head partials into ``xp`` (dQ pass: [dpq_h | dg_h] [Wq_h; Wg_h] at h, dK / dV pass: [dpk_h | dpv_h] [Wk_h; Wv_h] at 4 + h)."""
        N, _, S, _ = qh.shape
        assert S % 128 == 0 and dp.shape == (N * S, 4 * C)
        rows = N * H * S
        q2, k2, v2 = (t.view(rows, D) for t in (qh, kh, vh))
        mdp = self._mat(dp, N * S, 4 * C, 128)
        mq = (
            self._mat(q2, rows, D, 128),
            self._mat(do, N * S, C, 128),
            self._mat(k2, rows, D, 32),
            self._mat(v2, rows, D, 32),
            self._mat(k2, rows, D, 32, "128a32"),
        )
        mk = (
            self._mat(q2, rows, D, 32),
            self._mat(q2, rows, D, 32, "128a32"),
            self._mat(k2, rows, D, 128),
            self._mat(v2, rows, D, 128),
            self._mat(do, N * S, C, 32),
            self._mat(do, N * S, C, 32, "128a32"),
        )
        grid = (min(num_sms(), N * (S // 128) * H), 1, 1)
        scale = float(D**-0.5)
        kq, kk, xa = self.kq, self.kk, ()
        if xp is not None:
            assert self.dxp_ok(N, S) and xp.shape == (8, N * S, C) and xp.is_contiguous()
            assert w.shape == (4 * C, C) and w.is_contiguous() and w.dtype == torch.float32
            if self.kqx is None:
                self.kqx, self.kkx = _f._load("attn_dq_tf32_x"), _f._load("attn_dkv_tf32_x")
            kq, kk = self.kqx, self.kkx
            xa = (
                _f._f32map(w, [32, 4 * C, 4], [C * 4, 128], [32, 32, 4], swizzle="128a32"),  # 32 rows of W, MN-major (4 atoms)
                self._mat(xp.view(8 * N * S, C), 8 * N * S, C, 128),
            )

        def run_q():
            kq(grid, (512, 1, 1), *mq, lse, dv, seqused, pq, cos, sin, mdp, int(S), int(N), int(B), scale, float(_f.EPS), *xa)

        def run_k():
            kk(grid, (512, 1, 1), *mk, lse, dv, seqused, pk, cos, sin, mdp, int(S), int(N), int(B), scale, float(_f.EPS), *xa)

        def run():
            run_q()
            run_k()

        run.q, run.k = run_q, run_k

        run.keep = (mq, mk, mdp, qh, kh, vh, do, lse, dv, seqused, pq, pk, cos, sin, dp, xa, w, xp)
        return run


class DwTf32:
    """The five weight gradients in one launch (dw_tf32.cu) + a deterministic reduction of the split-K partial sums (dw_reduce_tf32):
    dP [M, 4C], x [M, C], datt, gated [M, C], dab [M, 2 NHID], y [M, C], dffn [M, C], hh [M, NHID] (all row-major fp32) ->
    dP^T x [4C, C], datt^T gated [C, C], dab^T y [2 NHID, C], dffn^T hh [C, NHID]. A CTA takes a row slice of one problem; the CTAs go to
    the problems in proportion to their bytes (0.297 | 0.189 | 0.297 | 0.216, measured on B200: 44 / 28 / 44 / 32 of 148)."""

    FRAC = (44 / 148, 28 / 148, 44 / 148, 32 / 148)
    SHAPES = ((4 * C, C), (C, C), (2 * NHID, C), (C, NHID))
    #: least rows of a slice (B200 2026-11-17, round 3 part 4: 64, was 256 -- one sample of 1024 atoms ran each problem on 4 CTAs, 16 ring
    #: stages through a ring of 5 at ~2.3 us a round)
    MINROWS = 64
    #: round 3 part 14 (development A/B): True -- the partial sums leave by the 128 epilogue threads' coalesced stores (dw_tf32.cu
    #: DW_DIRECT); measured level against the single-thread TMA stores (ESMFold2 master L 1024: part 0 ~10 us after the attention either way)
    DIRECT_STORE = False
    #: round 3 part 14: a ring stage's operands as two boxes of 3-D maps (dw_tf32.cu DW_BOX3) instead of aa + bb (up to 20) 2-KB boxes
    BOX3 = True
    #: round 3 part 15: the H100's form -- one cluster of CLUSTER CTAs per 128-row output tile (10 tiles), rank k the k-th row slice, the
    #: partials summed in rank order over DSMEM inside the kernel (dw_tf32.cu DW_CL): no 256-KB partial per CTA, no workspace, no reduce
    #: launch. Up to CL_MAX_ROWS rows (more rows: the split-K form over every SM + the reduce launch). 0: off; -1: 4 CTAs a cluster up
    #: to 2048 rows, 8 above (B200 2026-11-18, graph replay of the step: ESMFold2 master L 2048 102.2 -> 96.2 us with 4 or 8, SWA 3
    #: blocks L 1024 236.3 -> 231.2 (4) / 233.2 (8), embedder global L 2048 620.3 -> 610.7 / 612.2; L 4096 / 8192 with 8: ESMFold2
    #: 139.0 -> 132.9 / 184.3 -> 176.2, SWA L 8192 540.5 -> 520.9; 16 CTAs: slower, its six-cluster launch waits for whole GPCs)
    CLUSTER = int(__import__("os").environ.get("SWA_DWCL", "-1"))
    CL_MAX_ROWS = int(__import__("os").environ.get("SWA_DWCL_MAX", "16384"))
    CL_ROWS = int(__import__("os").environ.get("SWA_DWCL_R", "32"))  # reduction rows a ring stage of the cluster build
    CL_X = int(__import__("os").environ.get("SWA_DWCL_X", "0"))  # development builds (dw_tf32.cu DW_CLX)
    #: r3-swa part 21: the cluster build's partials summed through a global workspace (L2) instead of DSMEM (dw_tf32.cu DW_GX; the same
    #: bits). Measured (clock64 trace, 2 tiles x 8 CTAs, 1024 rows): the DSMEM exchange was ~9700 of ~14000 clocks after the MMAs
    GX = __import__("os").environ.get("SWA_DWGX", "1") == "1"
    GX_CL_MAX = 8  # workspace per tile: GX_CL_MAX x [128][256] fp32 (dw_tf32.cu: any cluster size <= 8 of a launch fits)

    @classmethod
    def _gws(cls, tiles, dev):
        return torch.empty(tiles * cls.GX_CL_MAX * 128 * 256, device=dev, dtype=torch.float32) if cls.GX else None

    _TILES = (0, 4, 5, 9, 10)  # first output tile of each problem (dWqkvg 4 m-blocks, dWo, dWu 4 m-blocks, dWd)

    def __init__(self):
        self.kd = _f._load("dw_tf32")
        self.kdt = None
        self.kb3 = None
        self.kcl = {}
        self.kr = _f._load("dw_reduce_tf32")
        # round 3 part 14: dWg into a buffer of its own (no clone launch per block backward)
        self.kr5 = _f._load("dw_reduce_tf32", "swa_dw_reduce5_tf32_sm100")

    def bind(self, dp, x, datt, gated, dab, y, dffn, hh, split=False):
        """``split`` (round 3 part 14): outs = (dWqkv [3C, C], dWo, dWu, dWd) and ``run.dwg`` [C, C] a tensor of its own (the reduce
        writes it; no copy); else outs[0] is dWqkv | dWg [4C, C]."""
        M = dp.shape[0]
        pairs = ((dp, x), (datt, gated), (dab, y), (dffn, hh))
        for a, b in pairs:
            assert a.is_contiguous() and b.is_contiguous() and a.shape[0] == M and b.shape[0] == M and M % 16 == 0
        nsm = num_sms()
        ns = tuple(max(1, min(round(f * nsm), M // self.MINROWS)) for f in self.FRAC)
        dev = dp.device
        maps = []
        box3 = self.BOX3
        cl = self.CLUSTER
        if cl < 0:
            cl = 4 if M <= 2048 else 8
        if cl and M <= self.CL_MAX_ROWS and M >= 16 * cl:
            return self._bind_cluster(cl, pairs, M, dev, split)

        def m3(t):  # round 3 part 14: [32 fp32][16 rows][all column atoms] in one box (the kernel's stage layout)
            k = t.view(M, -1).shape[1]
            return _f._f32map(t, [32, M, k // 32], [k * 4, 128], [32, 16, k // 32], swizzle="128a32")

        for a, b in pairs:
            if box3:
                maps += [m3(a), m3(b)]
                continue
            maps += [
                AttnBwdTf32._mat(a, M, a.view(M, -1).shape[1], 16, "128a32"),
                AttnBwdTf32._mat(b, M, b.view(M, -1).shape[1], 16, "128a32"),
            ]
        ws = torch.empty(sum(n * m * k for n, (m, k) in zip(ns, self.SHAPES)), device=dev, dtype=torch.float32)
        outs = tuple(torch.empty(m, k, device=dev, dtype=torch.float32) for m, k in self.SHAPES)
        dwg = None
        if split:
            outs = (torch.empty(3 * C, C, device=dev, dtype=torch.float32), *outs[1:])
            dwg = torch.empty(C, C, device=dev, dtype=torch.float32)
        total4 = sum(m * k // 4 for m, k in self.SHAPES)
        off3 = sum(n * m * k for n, (m, k) in zip(ns[:3], self.SHAPES[:3]))  # the partial sums leave by TMA: rows of 128 / of 256 floats
        wmaps = (
            _f._f32map(ws[:off3].view(-1, C), [C, off3 // C], C * 4, [32, 128]),
            _f._f32map(ws[off3:].view(-1, NHID), [NHID, (ws.numel() - off3) // NHID], NHID * 4, [32, 128]),
        )

        def part(p0, p1):  # the slices of problems p0 .. p1 - 1 (their operands are ready)
            lo, n = sum(ns[:p0]), sum(ns[p0:p1])
            kd = self.kd
            if box3:
                if self.kb3 is None:
                    self.kb3 = _f._load("dw_tf32_b3")
                kd = self.kb3
            elif self.DIRECT_STORE:  # development A/B (round 3 part 14)
                if self.kdt is None:
                    self.kdt = _f._load("dw_tf32_direct")
                kd = self.kdt
            kd((n, 1, 1), (256, 1, 1), *maps, ws, int(M), *(int(n) for n in ns), int(lo), *wmaps)

        def reduce():
            if split:
                self.kr5(((total4 + 255) // 256, 1, 1), (256, 1, 1), ws, *outs, *(int(n) for n in ns), dwg)
            else:
                self.kr(((total4 + 255) // 256, 1, 1), (256, 1, 1), ws, *outs, *(int(n) for n in ns))

        def run():
            part(0, 4)
            reduce()

        run.keep = (maps, wmaps, ws, dp, x, datt, gated, dab, y, dffn, hh, dwg)  # (dwg last: DcAcc.hold keeps keep[:-1])
        run.dwg = dwg
        run.part, run.reduce, run.ctas = part, reduce, lambda p0, p1: sum(ns[p0:p1])
        return run, outs

    def _clk(self, cl):
        gx = "g" if self.GX and cl <= self.GX_CL_MAX else ""
        name = f"dw_tf32_cl{cl}_r{self.CL_ROWS}" + (f"x{self.CL_X}" if self.CL_X else "") + gx
        k = self.kcl.get(name)
        if k is None:
            k = self.kcl[name] = _f._load(name, cluster=cl)
        return k

    def _bind_cluster(self, cl, pairs, M, dev, split):
        """:meth:`bind` on the cluster build (round 3 part 15): grid = tiles x cl CTAs, cluster (cl, 1, 1); the final dW written by the
        kernel (``run.reduce`` does nothing)."""
        br = self.CL_ROWS
        k = self._clk(cl)
        maps = []
        for i, (a, b) in enumerate(pairs):
            for j, t in enumerate((a, b)):
                kk = t.view(M, -1).shape[1]
                box = 4 if (j == 0 and i in (0, 2)) else kk // 32  # dP / dab: one 128-row m-block of the output a box (4 column atoms)
                maps.append(_f._f32map(t, [32, M, kk // 32], [kk * 4, 128], [32, br, box], swizzle="128a32"))
        outs = tuple(torch.empty(m, kk, device=dev, dtype=torch.float32) for m, kk in self.SHAPES)
        dwg = None
        if split:
            outs = (torch.empty(3 * C, C, device=dev, dtype=torch.float32), *outs[1:])
            dwg = torch.empty(C, C, device=dev, dtype=torch.float32)
        T = self._TILES
        gws = self._gws(10, dev)

        def part(p0, p1):
            lo, hi = T[p0], T[p1]
            k(((hi - lo) * cl, 1, 1), (256, 1, 1), *maps, gws, int(M), 0, 0, 0, 0, int(lo), maps[0], maps[1], *outs, dwg)

        def reduce():
            pass

        def run():
            part(0, 4)

        def tiles(lo, hi, c=cl):  # r3-swa part 21: the output tiles lo .. hi - 1 on clusters of c CTAs
            assert M >= 16 * c
            self._clk(c)(((hi - lo) * c, 1, 1), (256, 1, 1), *maps, gws, int(M), 0, 0, 0, 0, int(lo), maps[0], maps[1], *outs, dwg)

        run.keep = (maps, pairs, gws, dwg)
        run.dwg = dwg
        run.part, run.reduce, run.ctas, run.tiles = part, reduce, lambda p0, p1: (T[p1] - T[p0]) * cl, tiles
        return run, outs


#: N2 (development switch): the modulation's weight gradient dWmod = dmod^T silu(c) on the cluster dW kernel (DwModTf32) instead of a cuBLAS GEMM with K = rows
DWM_KERNEL = __import__("os").environ.get("SWA_DWM_KERNEL", "1") == "1"


class DwModTf32:
    """dWmod [6C, C] = dmod^T a (dmod [M, 6C], a = silu(c) [M, C] fp32): dw_tf32.cu built with DWM=1 -- six 128-row output tiles, each a cluster of 8
    CTAs (rank k the k-th row slice of the reduction, the partials through a global workspace and summed in rank order: deterministic). Was a cutlass
    sm80-style TF32 GEMM on 12 CTAs (11 - 20 us on a side stream beside the chain) and its operands' roundings: the kernel's operands are truncated by the MMA."""

    CL = 8

    def __init__(self):
        self.k = None

    def ok(self, dmod, a) -> bool:
        M = dmod.shape[0]
        return (
            dmod.dtype == torch.float32 and a.dtype == torch.float32 and dmod.is_contiguous() and a.is_contiguous() and a.shape[0] == M
            and tuple(dmod.shape) == (M, 6 * C) and tuple(a.shape) == (M, C) and M >= 16 * self.CL * 2 and M % 16 == 0
        )

    def run(self, dmod, a):
        if self.k is None:
            self.k = _f._load("dw_tf32_cl8_r32g_dwm", cluster=self.CL)
        M = dmod.shape[0]
        dev = dmod.device
        br = 32
        ma = _f._f32map(dmod, [32, M, 6 * C // 32], [6 * C * 4, 128], [32, br, 4], swizzle="128a32")
        mb = _f._f32map(a, [32, M, C // 32], [C * 4, 128], [32, br, C // 32], swizzle="128a32")
        out = torch.empty(6 * C, C, device=dev, dtype=torch.float32)
        gws = torch.empty(6 * 8 * 128 * 256, device=dev, dtype=torch.float32)
        self.k((6 * self.CL, 1, 1), (256, 1, 1), ma, mb, ma, mb, ma, mb, ma, mb, gws, int(M), 0, 0, 0, 0, 0, ma, mb, out, out, out, out, None)
        return out, (ma, mb, gws, dmod, a)


class DcxTf32:
    """B200 round 3 part 16: the few-tile backward's tail in one kernel (dcx_tf32.cu) -- dy = dP W, the qkvg RMS-adaLN backward (dq and
    dmod columns 0 .. 2C), dc = (dmod Wmod) silu'(c) [+ prev] [+ dq] -- on one cluster of CL CTAs per 128-row tile (rank k: the K slices k
    of both GEMMs, rows 16 k .. 16 k + 15 of the row stages; the partials exchanged by bulk DSMEM copies and summed in rank order). Was four
    dependent launches (cuBLAS dx, adaln_bwd1, cuBLAS g Wmod, dsilu). One augment per atom (A = 1), every tile's cluster in one wave."""

    CL = 8
    NAME = __import__("os").environ.get("SWA_DCX_NAME", "dcx_tf32")  # development: the build (dcx_tf32_sta: st.async exchanges)

    def __init__(self):
        self.k = None

    def ok(self, A, M, x, res, c, w, wmod, mod) -> bool:
        return (
            A == 1
            and mod.is_contiguous()
            and M % 128 == 0
            and (M // 128) * self.CL <= num_sms()
            and res.dtype == torch.float32
            and mod.dtype == torch.float32
            and wmod.dtype == torch.float32
            and w.dtype == torch.float32
            and x.dtype in (torch.float32, torch.bfloat16)
            and c.dtype in (torch.float32, torch.bfloat16)
            and wmod.is_contiguous()
            and c.is_contiguous()
            and res.is_contiguous()
            and x.is_contiguous()
            and tuple(wmod.shape) == (6 * C, C)
        )

    def bind(self, dp, w, dmod, wmod, x, res, mod, c, eps):
        """-> run(prev, out_dtype, fold) -> dc (dtype ``out_dtype`` or c's), and dq (x's dtype). ``prev`` None or fp32 [M, C]."""
        if self.k is None:
            self.k = _f._load(self.NAME, cluster=self.CL)
        M = dp.shape[0]
        dev = dp.device
        assert dp.is_contiguous() and dmod.is_contiguous() and mod.is_contiguous() and w.stride(1) == 1 and w.stride(0) == C
        dq = torch.empty(M, C, device=dev, dtype=x.dtype)
        kd = 4 * C // self.CL
        maps = (
            _f._f32map(dp, [4 * C, M], 4 * C * 4, [32, 128]),
            _f._f32map(w, [32, 4 * C, 4], [C * 4, 128], [32, kd, 4], swizzle="128a32"),
            _f._f32map(dmod, [6 * C, dmod.shape[0]], 6 * C * 4, [32, 128]),
            _f._f32map(wmod, [32, 6 * C, 4], [C * 4, 128], [32, kd, 4], swizzle="128a32"),
            _f._f32map(wmod, [32, 6 * C, 4], [C * 4, 128], [32, 2 * C // self.CL, 4], swizzle="128a32"),
        )
        b16 = torch.bfloat16

        def run(prev=None, out_dtype=None, fold=False):
            od = out_dtype or c.dtype
            dc = torch.empty(M, C, device=dev, dtype=od)
            assert prev is None or (prev.dtype == torch.float32 and prev.is_contiguous())
            flags = int(x.dtype == b16) | (int(c.dtype == b16) << 1) | (int(od == b16) << 2) | (int(fold) << 3)
            self.k(((M // 128) * self.CL, 1, 1), (256, 1, 1), *maps, x, res, mod, dq, dmod, c, dc, prev, int(M), float(eps), flags)
            run.keep = (maps, dp, w, dmod, wmod, x, res, mod, c, prev)
            return dc

        return run, dq


class DcgTf32:
    """r3-swa part 19 (dcg_tf32.cu): dc = (dmod Wmod) silu'(c) [+ prev] [+ xadd] as two launches -- ``early``: E = dmod[:, 2C:] Wmod[2C:]
    (fp32 [M, C]; off the chain: its columns are made by the FFN adaLN backward), ``late`` (after the qkvg adaLN backward): (E + dmod[:, :2C]
    Wmod[:2C]) silu'(c) with the stack's running sum, in place of cuBLAS g Wmod (K 6C) + the dsilu launch on the chain."""

    def __init__(self):
        self.k = None

    @staticmethod
    def ok(M, dmod, wmod, c) -> bool:
        return (
            M % 128 == 0
            and dmod.shape[0] == M
            and dmod.is_contiguous()
            and dmod.dtype == torch.float32
            and wmod.dtype == torch.float32
            and wmod.is_contiguous()
            and tuple(wmod.shape) == (6 * C, C)
            and c.is_contiguous()
            and c.dtype in (torch.float32, torch.bfloat16)
            and M <= DCG_MAX_ROWS
        )

    NS_MAX = int(__import__("os").environ.get("SWA_DCG_NS", "4"))  # development: the most column splits
    NS_EARLY = 1  # the early launch: one CTA a tile

    @classmethod
    def _ns(cls, M):
        ns = 1
        while ns < cls.NS_MAX and (M // 128) * ns * 2 <= num_sms():
            ns *= 2
        return ns

    def _maps(self, dmod, wmod, ns):
        M = dmod.shape[0]
        return (
            _f._f32map(dmod, [6 * C, M], 6 * C * 4, [32, 128]),
            _f._f32map(wmod, [32, 6 * C, 4], [C * 4, 128], [32, 32, 4 // ns], swizzle="128a32"),
        )

    def early(self, dmod, wmod, e):
        """E = dmod[:, 2C:] Wmod[2C:] into ``e`` (fp32 [M, C]); returns what the launch reads (keep it alive)."""
        if self.k is None:
            self.k = _f._load("dcg_tf32")
        M = dmod.shape[0]
        ns = self.NS_EARLY  # (off the chain: as few SMs as it can, beside the attention backward)
        maps = self._maps(dmod, wmod, ns)
        self.k(((M // 128) * ns, 1, 1), (256, 1, 1), *maps, 2 * C, 4 * C // 32, ns, None, None, None, None, e, 1)
        return (maps, dmod, wmod)

    def late(self, dmod, wmod, e, c):
        """-> run(prev, out_dtype, fold_x) (``DcAcc.fused``'s protocol; ``fold_x``: the x gradient added, or None) -> dc."""
        if self.k is None:
            self.k = _f._load("dcg_tf32")
        M = dmod.shape[0]
        ns = self._ns(M)
        maps = self._maps(dmod, wmod, ns)
        b16 = torch.bfloat16

        def run(prev=None, out_dtype=None, xadd=None):
            od = out_dtype or c.dtype
            out = torch.empty(c.shape, device=c.device, dtype=od)
            assert prev is None or (prev.dtype == torch.float32 and prev.is_contiguous())
            assert xadd is None or (xadd.is_contiguous() and xadd.numel() == c.numel())
            flags = (int(c.dtype == b16) << 1) | (int(od == b16) << 2) | (int(xadd is not None and xadd.dtype == b16) << 3)
            self.k(((M // 128) * ns, 1, 1), (256, 1, 1), *maps, 0, 2 * C // 32, ns, e, c, prev, xadd, out, flags)
            run.keep = (maps, dmod, wmod, e, c, prev, xadd)
            return out

        return run


#: r3-swa part 19 (development switch): the c gradient as DcgTf32's early / late launches in the few-tile backward (_bwd_few)
DCG = __import__("os").environ.get("SWA_DCG", "1") == "1"
#: ... up to this many rows
DCG_MAX_ROWS = int(__import__("os").environ.get("SWA_DCG_MAX_ROWS", "8192"))


@functools.lru_cache(maxsize=16)
def _dce_stream(index: int):
    with torch.cuda.device(index):
        return torch.cuda.Stream()


#: round 3 part 16 (development switch): the few-tile tail (dx GEMM -> adaLN -> dc GEMM -> silu') as one cluster kernel (DcxTf32)
DCX = __import__("os").environ.get("SWA_DCX", "0") == "1"
#: ... with dWmod's rows 2C .. 6C early on a stream of their own (_bwd_few)
PROBE_SKIP = int(__import__("os").environ.get("SWA_PROBE_SKIP", "0"))  # timing probe (wrong gradients): 1 no dWmod GEMM, 2 no dc GEMM
#: N2 (development switch): silu(c) of the modulation's weight gradient made once for the blocks of a stack (they share c)
SILU_ONCE = __import__("os").environ.get("SWA_SILU_ONCE", "1") == "1"
DWM_SPLIT = __import__("os").environ.get("SWA_DWM_SPLIT", "1") == "1"
#: r3-swa part 21 (development switch): the few-tile backward's dx = dP W as per-head partials in the window attention passes' epilogues
#: (attn_dq_tf32 / attn_dkv_tf32 -DDXP=1), summed by the qkvg adaLN backward -- no dx GEMM launch after dK / dV
DXP = __import__("os").environ.get("SWA_DXP", "1") == "1"
#: r3-swa part 21 (development switch): dWqkvg's m-blocks launched as their operands arrive where the early weight gradients fit beside
#: the attention passes (DWS: dWg with the early tiles after the out-projection, dWq after the dQ pass, dWk / dWv after dK / dV on clusters
#: of DWS_CL)
DWS = __import__("os").environ.get("SWA_DWS", "1") == "1"
DWS_CL = int(__import__("os").environ.get("SWA_DWS_CL", "8"))


@functools.lru_cache(maxsize=16)
def _xstream(index: int, k: int):
    """r3-swa part 21: the few-tile backward's extra side stream (1: dWq after the dQ pass)."""
    with torch.cuda.device(index):
        return torch.cuda.Stream()


class _Kernels:
    def __init__(self) -> None:
        self.wprep = WprepBwdTf32()
        self.mod = ModBwdTf32()
        self.attn = AttnBwdTf32()
        self.ffn_gate = FfnBwdGateTf32()
        self.adaln = AdalnBwdTf32()
        self.oproj_fused = OprojBwdFusedTf32()
        self.dw = DwTf32()
        self.dcx = DcxTf32()
        self.dcg = DcgTf32()
        self.dwmod = DwModTf32()


@functools.lru_cache(maxsize=16)
def kernels_bwd(index: int) -> _Kernels:
    """Build and load every TF32 backward kernel on device ``index``."""
    with torch.cuda.device(index):
        torch.zeros(1, device=f"cuda:{index}")  # a torch CUDA call first: autograd's backward thread has no current context
        return _Kernels()


def _index(t: torch.Tensor) -> int:
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


def mm_tf32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b of fp32 operands on the TF32 tensor cores (cuBLAS; operands that are TF32 values already, as the kernels here write them,
    are exact): fp32 sums, ~3e-4 relative to fp64 for random operands, the precision of the TF32 forward kernels. Autocast off: a
    master-training backward runs under the caller's bf16 autocast (it would make these bf16 GEMMs)."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with torch.autocast("cuda", enabled=False):
            return torch.mm(a, b)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


#: B200 (2026-11-18, round 3 part 12): the few-tile backward's dx = dP W and dc = dmod Wmod GEMMs as K-split batched GEMMs (KS
#: slices of K on KS x the CTAs; the partials summed in slice order by the adaLN / dsilu kernels that read them -- deterministic). One
#: sample of 1024 rows ran each on 16 CTAs of cuBLAS's 128 x 64 tiles (5 / 7 us for K = 512 / 768). 0: the plain GEMMs. Measured
#: (ESMFold2 float32 training L 2048 / 3072 114.5 -> 112.5 / 126.8 -> 125.1 us, master 3072 120.6 -> 118.9, SWA 3 blocks L 2048 348 -> 340):
#: 2 / 3 slices; 4 / 6 were slower at 1024 - 2048 rows.
KSPLIT_DX = int(__import__("os").environ.get("SWA_KSPLIT_DX", "2"))
KSPLIT_DC = int(__import__("os").environ.get("SWA_KSPLIT_DC", "3"))
#: ... from this many rows (1024 rows: level -- ESMFold2 float32 training 104.4 -> 104.5 us, SWA 3 blocks 311 -> 313)
KSPLIT_MIN_ROWS = 2048


def _ks(M: int, ks: int) -> int:
    """The largest divisor d of ks whose batched GEMM (cuBLAS 128 x 64 tiles of an [M, C] output, d of them) fits the SMs."""
    for d in range(ks, 0, -1):
        if ks % d == 0 and ((M + 127) // 128) * (C // 64) * d <= num_sms():
            return d
    return 1


def mm_tf32_out(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """:func:`mm_tf32` into ``out`` (round 3 part 14)."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with torch.autocast("cuda", enabled=False):
            torch.mm(a, b, out=out)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


#: round 3 part 14 (development switch): the few-tile backward's dc GEMM split at the modulation column 2C, the FFN half early on a
#: stream of its own. Measured and left off (ESMFold2 master L 1024 84 -> 96 us): the early half beside the attention backward stretched
#: dK / dV 12.5 -> 17 us, and the late half (a strided K slice) ran on cuBLAS's sm_80 64 x 64 kernel (10 us against the full GEMM's 6.7)
EARLY_DC = False


def mm_tf32_ks(a: torch.Tensor, b: torch.Tensor, ks: int) -> torch.Tensor:
    """[ks, M, N] partials of a @ b (a [M, K], b [K, N]) over ks K-slices, one strided batched TF32 GEMM (no copies)."""
    M, Kd = a.shape
    kc = Kd // ks
    assert kc * ks == Kd and a.stride(1) == 1 and b.is_contiguous()
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with torch.autocast("cuda", enabled=False):
            return torch.bmm(a.as_strided((ks, M, kc), (kc, a.stride(0), 1)), b.view(ks, kc, b.shape[1]))
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


def _head_tail(p: torch.Tensor, dh: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """The head-wise tail of the qkvg backward for one of q / k: the head = rope(RMSNorm_head(p)) (affine-free, fp32 eps; the rotary on the
    head's leading 2 r channels, channel c with c + r, ``functional.apply_rotary_emb_3d``) -> d p [N S, C] fp32 from d head [N, H, S, D] fp32.
    The TF32 window kernels run this in their epilogues; global attention runs it here (elementwise torch, fp32)."""
    n, _, s, _ = dh.shape
    r = cos.shape[-1]
    c4, s4 = cos.view(n, s, 1, r).float(), sin.view(n, s, 1, r).float()
    g = dh.transpose(1, 2)  # [N, S, H, D]
    # the rotary's transpose: (a, b) -> (a cos + b sin, b cos - a sin) on the channel pairs (c, c + r)
    ga, gb = g[..., :r], g[..., r : 2 * r]
    dy = torch.cat([ga * c4 + gb * s4, gb * c4 - ga * s4, g[..., 2 * r :]], dim=-1)
    x = p.view(n, s, H, D).float()
    rs = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + _f.EPS)
    dx = rs * (dy - x * (rs * rs) * (dy * x).mean(-1, keepdim=True))  # the affine-free RMSNorm's backward
    return dx.reshape(n * s, C)


#: N1 round 4 (development switch): the native global backward's casts and delta in one launch (gattn_tf32.cu swa_gprep)
GPREP = __import__("os").environ.get("SWA_GPREP", "1") == "1"


def _global_attn_bwd(qh, kh, vh, o, do, lse, seqused, pq, pk, cos, sin, dp, B):
    """The fp32 block's global attention backward (``block_fwd_tf32`` with a global half window saved the TF32 heads, O and the bf16
    attention's lse): dQ / dK / dV of the bf16 global attention (``attn_global_bwd``: FlashAttention-4, or PyTorch's FlashAttention-2 on a
    B200 without it) on the heads rounded to bf16 as the forward ran it, then :func:`_head_tail` for q and k -> dpq | dpk | dpv into the
    columns 0 .. 3C of dp, TF32-rounded as the window kernels write them. One modulation row per sequence row only (B = N)."""
    n, _, s, _ = qh.shape
    assert B == n, "global attention on the TF32 stages: one modulation / rotary row per sequence row (B = N)"
    # B200 2026-11-17 (round 3): the five casts and the three zero fills in one launch, the head-wise tail (:func:`_head_tail`'s equations)
    # and the v columns in one launch (gattn_tf32.cu; were ~50 torch launches per block)
    glue = _f.global_glue(_index(qh))
    if _sm.native_global_train(qh):
        # B200 (round 3, part 2): the native backward (gattn_bwd.cu) on the native forward's saves -- the three head casts, dO16 and its D
        # (swa_gdelta: D of the rounded dO), no zero fills (every row written)
        if GPREP and _f.N1_GLUE:
            b, do16, dd = glue.prep((qh, kh, vh), o.contiguous(), do.contiguous(), n, s)
            hv = lambda i: b[i].view(n, H, s, D)  # noqa: E731
        else:
            b = glue.cast((qh, kh, vh), 0)
            hv = lambda i: b[i].view(n, H, s, D)  # noqa: E731
            do16, dd = glue.delta(o.contiguous(), do.contiguous(), n, s)
        dqh, dkh, dvh = _sm.attn_global_bwd_native(hv(0), hv(1), hv(2), do16, lse, dd, seqused)
        glue.tail(dqh, dkh, dvh, pq.contiguous(), pk.contiguous(), cos.contiguous(), sin.contiguous(), dp, _f.EPS)
        return
    b = glue.cast((qh, kh, vh, o.contiguous(), do.contiguous()), 3)
    hv = lambda i: b[i].view(n, H, s, D)  # noqa: E731
    dqh, dkh, dvh = hv(5), hv(6), hv(7)
    _sm.attn_global_bwd(hv(0), hv(1), hv(2), b[3].view(n * s, C), b[4].view(n * s, C), lse, seqused, dqh, dkh, dvh)
    glue.tail(dqh, dkh, dvh, pq.contiguous(), pk.contiguous(), cos.contiguous(), sin.contiguous(), dp, _f.EPS)


def _widen(fn, *saves):
    """r3-swa part 19: the hybrid forward's twelve bf16 saves -> fp32 (``sm100.save32`` / ``save32_late``; exact)."""
    return fn([t.contiguous() for t in saves])


def block_bwd_tf32(
    dy, q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, qh, kh, vh, g, o, lse, q1, x, pq, pk, att, y, ffn, B, half_window=_f.HW, modb=None
):
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
        few = SCHED and M <= 128 * num_sms()
        if qh.dtype == torch.bfloat16 and not few:
            # r3-swa part 19: the hybrid forward's bf16 saves (widened under the gate kernel by _bwd_few; here in front of the backward)
            qh, kh, vh, g, o, q1, x, pq, pk, att, y, ffn = _widen(_sm.save32, qh, kh, vh, g, o, q1, x, pq, pk, att, y, ffn)
        if few:
            return _bwd_few(
                K,
                dy,
                q,
                mod,
                cos,
                sin,
                seqused,
                w,
                wab,
                wu,
                wdt,
                wot,
                qh,
                kh,
                vh,
                g,
                o,
                lse,
                q1,
                x,
                pq,
                pk,
                att,
                y,
                ffn,
                B,
                half_window,
                modb,
                dmod,
            )
        # FFN
        dq2 = dy.reshape(M, C)
        run, (dffn, hh, dab) = K.ffn_gate.bind(y, dq2, mod, wab, wdt, A, B)
        run()
        run, dq1 = K.adaln.bind(q1, mm_tf32(dab, wu), dq2, ffn, mod, dmod, A, B, eps, 3 * C, att, 2 * C)  # + d gate_a = sum_aug dq1 att
        run()
        # out-projection: one kernel (datt, dgated = datt Wo, dO, dG, gated, Dv)
        dP = torch.empty(M, 4 * C, device=dev, dtype=torch.float32)  # [dpq | dpk | dpv | dg]
        run, (datt, gated, dO, Dv) = K.oproj_fused.bind(dq1, o, g, mod, wot, dP, A, B)
        run()
        if half_window != _f.HW:  # global attention: its bf16 backward and the head-wise tail (_global_attn_bwd)
            _global_attn_bwd(qh, kh, vh, o, dO, lse, seqused, pq, pk, cos, sin, dP, B)
        else:
            # window attention (TF32 kernels, fp32 operands: the forward's saves as they are)
            # (the epilogues run the head-wise tail of the qkvg backward and write dpq | dpk | dpv into dP)
            run = K.attn.bind(qh, kh, vh, dO, lse, Dv, seqused, pq, pk, cos, sin, dP, B)
            run()
        # qkvg
        run, dq = K.adaln.bind(q.reshape(M, C), mm_tf32(dP, w), dq1, None, mod, dmod, A, B, eps, 0)
        run()
        # the five weight gradients: one launch (dWqkv | dWg is one problem over dP, x read once) + the fixed-order reduction
        run, (dwqkv, dwo, dwu, dwd) = K.dw.bind(dP, x.view(M, C), datt, gated.view(M, C), dab, y.view(M, C), dffn, hh, split=True)
        dwg = run.dwg
        if modb is None:
            run()
            return [dq.view(N, S, C), dmod, dwqkv, dwg, dwo, dwu, dwd]
        # B200 (2026-11-17): the modulation backward (c, Wmod fp32) from dmod, its dc and dW chains on side streams beside the weight gradients
        c_mod, w_mod, dca = (*modb, None)[:3]
        res = {}

        def mod_dc():
            res["dc"] = K.mod.dsilu(mm_tf32(dmod, w_mod), c_mod)

        def mod_dw():
            res["dw"] = mm_tf32(dmod.t(), K.mod.silu(c_mod))

        if dca is not None:  # round 3 part 14: the stack's dc sum (DcAcc) on the current stream
            _sm._fork_join([mod_dw], run)
            dqx, dcx = dca[0].step(K, dca[1], lambda: mm_tf32(dmod, w_mod), c_mod, dq, dmod)
            return [dqx.view(N, S, C), dmod, dwqkv, dwg, dwo, dwu, dwd, dcx, res["dw"]]
        _sm._fork_join([mod_dc, mod_dw], run)
        return [dq.view(N, S, C), dmod, dwqkv, dwg, dwo, dwu, dwd, res["dc"], res["dw"]]


#: the few-tile schedule of :func:`block_bwd_tf32` (development switch)
SCHED = 2
#: round 3 part 6 (development switch): the few-tile schedule's dy GEMM inside the gate kernel (``FfnBwdGateTf32.bind(dyf=True)``)
DYF = __import__("os").environ.get("SWA_DYF", "1") == "1"


def _bwd_few(
    K, dy, q, mod, cos, sin, seqused, w, wab, wu, wdt, wot, qh, kh, vh, g, o, lse, q1, x, pq, pk, att, y, ffn, B, half_window, modb, dmod
):
    """:func:`block_bwd_tf32` for at most one 128-row tile per SM (B200 2026-11-17, round 3 part 4): the same kernels and operands (so the
    same bits), launched as the dependencies allow instead of one after the other -- every stage is a few CTAs there and the step is the
    length of its dependent chain. Side stream 1: the weight-gradient slices of a problem as soon as its operands exist (dWu / dWd after
    the FFN gate kernel, dWo after the out-projection, dWqkv | dWg after the attention), then their reduction. Side stream 2: dK / dV
    beside dQ, and the modulation's dW chain beside its dc chain. The current stream joins both at the end."""
    N, S, _ = q.shape
    M = N * S
    A = N // B
    dev = q.device
    eps = _f.EPS
    cur = torch.cuda.current_stream()
    s1, s2 = _sm._side_streams(_index(q))
    res = {}

    def fork(s, *fns):  # fns on stream s, after everything queued on the current stream (and on s) so far
        ev = torch.cuda.Event()
        ev.record(cur)
        s.wait_event(ev)
        with torch.cuda.stream(s):
            for fn in fns:
                fn()

    def join(*ss):
        for s in ss:
            ev = torch.cuda.Event()
            ev.record(s)
            cur.wait_event(ev)

    dq2 = dy.reshape(M, C)
    w16 = qh.dtype == torch.bfloat16
    if w16 and not DYF:  # (the late widening is wired for the DYF gate build only)
        qh, kh, vh, g, o, q1, x, pq, pk, att, y, ffn = _widen(_sm.save32, qh, kh, vh, g, o, q1, x, pq, pk, att, y, ffn)
        w16 = False
    run, (dffn, hh, dab, *dyp) = K.ffn_gate.bind(y, dq2, mod, wab, wdt, A, B, dyf=DYF, late_trigger=w16)
    run()
    if w16:
        # r3-swa part 19: the hybrid forward's bf16 saves widened now, under the gate kernel (it read y in bf16; save32_late waits for it
        # at its end, so every kernel after it still runs after the gate kernel)
        qh, kh, vh, g, o, q1, x, pq, pk, att, y, ffn = _widen(_sm.save32_late, qh, kh, vh, g, o, q1, x, pq, pk, att, y, ffn)
    if modb is not None:  # silu(c) of the modulation's dW: needs nothing of this backward
        # N2: every block of a stack shares one c: silu(c) made once (the first block's backward, on s2), the others' dWmod read it
        dcao = modb[2][0] if len(modb) > 2 and modb[2] is not None else None
        ckey = (modb[0].data_ptr(), tuple(modb[0].shape), modb[0].dtype)
        am = getattr(dcao, "amod", None) if (SILU_ONCE and dcao is not None) else None
        if am is not None and am[0] == ckey:
            a_mod = am[1]
        else:
            fork(s2, lambda: res.__setitem__("a", K.mod.silu(modb[0])))
            a_mod = res["a"]
            if SILU_ONCE and dcao is not None:
                dcao.amod = (ckey, a_mod)
    if DYF:  # dy = the gate kernel's partials, summed by the adaLN backward (no GEMM launch)
        run, dq1 = K.adaln.bind(q1, dyp[0], dq2, ffn, mod, dmod, A, B, eps, 3 * C, att, 2 * C, ndy=dyp[0].shape[0])
    else:
        run, dq1 = K.adaln.bind(q1, mm_tf32(dab, wu), dq2, ffn, mod, dmod, A, B, eps, 3 * C, att, 2 * C)
    run()
    fx = DCX and modb is not None and not EARLY_DC and K.dcx.ok(A, M, q, dq1, modb[0], w, modb[1], mod)
    dwm = None
    extra = []  # side streams of this backward beyond s1 / s2 (joined / held with them)
    if fx and DWM_SPLIT:
        # round 3 part 16: dWmod = dmod^T silu(c) in two parts -- the rows 2C .. 6C (dmod columns written by the FFN adaLN above) now, on a
        # stream of their own beside the out-projection / attention; the rows 0 .. 2C after the one-kernel tail (they need its adaLN)
        dwm = torch.empty(6 * C, C, device=dev, dtype=torch.float32)
        ds = _dc_stream(_index(q))
        ds.wait_stream(cur)
        ds.wait_stream(s2)  # silu(c)
        with torch.cuda.stream(ds):
            mm_tf32_out(dmod[:, 2 * C :].t(), a_mod, dwm[2 * C :])
        ev_m = torch.cuda.Event()
        ev_m.record(ds)
    dP = torch.empty(M, 4 * C, device=dev, dtype=torch.float32)
    run, (datt, gated, dO, Dv) = K.oproj_fused.bind(dq1, o, g, mod, wot, dP, A, B)
    dw, (dwqkv, dwo, dwu, dwd) = K.dw.bind(dP, x.view(M, C), datt, gated.view(M, C), dab, y.view(M, C), dffn, hh, split=True)
    run()
    dce = None
    ga = N * (S // 128) * H  # the CTAs of an attention pass
    par = SCHED >= 2 and half_window == _f.HW and 2 * ga <= num_sms()  # dK / dV beside dQ: both passes fit the SMs at once
    early = par and 2 * ga + dw.ctas(1, 4) <= num_sms()  # and dWo / dWu / dWd beside them
    dcg_room = early and 2 * ga + dw.ctas(1, 4) + (M // 128) * K.dcg.NS_EARLY <= num_sms()
    if modb is not None and not fx and DCG and dcg_room and not EARLY_DC and K.dcg.ok(M, dmod, modb[1], modb[0]):
        # r3-swa part 19: E = dmod[:, 2C:] Wmod[2C:] now, on a stream of its own beside the attention backward (DcgTf32.early); the
        # current stream waits for it where it already joins the dK / dV stream (no wait in front of a PDL launch)
        e_acc = torch.empty(M, C, device=dev, dtype=torch.float32)
        se = _dce_stream(_index(q))
        se.wait_stream(cur)
        with torch.cuda.stream(se):
            e_keep = K.dcg.early(dmod, modb[1], e_acc)
        ev_dce = torch.cuda.Event()
        ev_dce.record(se)
        dce = (e_acc, ev_dce, e_keep)
    dc_early = None
    if modb is not None and EARLY_DC and not (KSPLIT_DC > 1 and M >= KSPLIT_MIN_ROWS):
        # round 3 part 14: dc's GEMM over the modulation columns 2C .. 6C (written by the FFN adaLN backward above: gate_a, shift_f,
        # scale_f, gate_f) on a stream of its own now; after the qkvg adaLN only the columns 0 .. 2C remain (K 256 of 768); the dsilu
        # launch adds the two partials in order (acc_sum)
        ds = _dc_stream(_index(q))
        ds.wait_stream(cur)
        acc2 = torch.empty(2, dmod.shape[0], C, device=dev, dtype=torch.float32)
        with torch.cuda.stream(ds):
            mm_tf32_out(dmod[:, 2 * C :], modb[1][2 * C :], acc2[1])
        ev_e = torch.cuda.Event()
        ev_e.record(ds)
        dc_early = (acc2, ev_e)
    ga = N * (S // 128) * H  # the CTAs of an attention pass
    par = SCHED >= 2 and half_window == _f.HW and 2 * ga <= num_sms()  # dK / dV beside dQ: both passes fit the SMs at once
    early = par and 2 * ga + dw.ctas(1, 4) <= num_sms()  # and dWo / dWu / dWd beside them
    # r3-swa part 21 (DWS): dWqkvg's m-blocks as their operands arrive -- dWg (tile 3) with the early tiles after the out-projection,
    # dWq (tile 0) after the dQ pass on a stream of its own, dWk / dWv (tiles 1, 2) after dK / dV on clusters of DWS_CL
    dws = DWS and early and hasattr(dw, "tiles")
    if early:
        fork(s1, (lambda: dw.tiles(3, 10)) if dws else (lambda: dw.part(1, 4)))
    xp = None
    if DXP and early and not fx and K.attn.dxp_ok(N, S):  # (where the tail after dK / dV ends the step; L 2048: level / +2 us)
        # r3-swa part 21: dx = dP W as eight per-head partials written by the attention passes' epilogues (no dx GEMM launch)
        xp = torch.empty(8, M, C, device=dev, dtype=torch.float32)
    if half_window != _f.HW:
        _global_attn_bwd(qh, kh, vh, o, dO, lse, seqused, pq, pk, cos, sin, dP, B)
    else:
        at = K.attn.bind(qh, kh, vh, dO, lse, Dv, seqused, pq, pk, cos, sin, dP, B, w=w if xp is not None else None, xp=xp)
        if par:
            fork(s2, at.k)
            at.q()
            if dws:
                s3 = _xstream(_index(q), 1)
                fork(s3, lambda: dw.tiles(0, 1))
                extra.append(s3)
            join(s2)
        else:
            at()
    if dce is not None:  # (beside the dK / dV join: the next launch is the cuBLAS dx GEMM)
        cur.wait_event(dce[1])

    def dw_tail():  # (round 3 part 14: the reduce writes dWg into storage of its own -- no clone launch)
        if dws:
            dw.tiles(1, 3, DWS_CL)
            return
        dw.part(0, 1 if early else 4)
        dw.reduce()

    fork(s1, dw_tail)
    if fx:
        # round 3 part 16: dx GEMM -> adaLN -> dc GEMM -> silu' in one cluster kernel (DcxTf32)
        c_mod, w_mod, dca = (*modb, None)[:3]
        frun, dq = K.dcx.bind(dP, w, dmod, w_mod, q.reshape(M, C), dq1, mod, c_mod, eps)
        out = [dq.view(N, S, C), dmod, dwqkv, dw.dwg, dwo, dwu, dwd]
        if dca is not None:
            dqx, dcx = dca[0].fused(dca[1], frun, dq)
            out[0] = dqx.view(N, S, C)
        else:
            dcx = frun()
        if dwm is not None:
            s2.wait_event(ev_m)
            fork(s2, lambda: mm_tf32_out(dmod[:, : 2 * C].t(), a_mod, dwm[: 2 * C]))
            res["dw"] = dwm
        else:
            fork(s2, lambda: res.__setitem__("dw", mm_tf32(dmod.t(), a_mod)))
        out += [dcx, res["dw"]]
        if dca is not None and len(dca) > 2 and dca[2] and dca[1] != 0:
            dca[0].hold([s1, s2], [dw.keep[:-1], dP, dmod, a_mod, frun.keep])
            return out
        join(s2)
        join(s1)
        return out
    ksx = _ks(M, KSPLIT_DX) if KSPLIT_DX > 1 and M >= KSPLIT_MIN_ROWS else 1
    if xp is not None:
        dysrc, ndy = xp, 8
    elif ksx > 1:
        dysrc, ndy = mm_tf32_ks(dP, w, ksx), ksx
    else:
        dysrc, ndy = mm_tf32(dP, w), 1
    run, dq = K.adaln.bind(q.reshape(M, C), dysrc, dq1, None, mod, dmod, A, B, eps, 0, ndy=ndy)
    run()
    out = [dq.view(N, S, C), dmod, dwqkv, dw.dwg, dwo, dwu, dwd]
    if modb is not None:
        c_mod, w_mod, dca = (*modb, None)[:3]

        def mod_dw():
            if PROBE_SKIP & 1:
                res["dw"] = torch.zeros(6 * C, C, device=dev)
            elif DWM_KERNEL and K.dwmod.ok(dmod, a_mod):
                res["dw"], res["dwkeep"] = K.dwmod.run(dmod, a_mod)
            else:
                res["dw"] = mm_tf32(dmod.t(), a_mod)

        fork(s2, mod_dw)
        ksc = _ks(dmod.shape[0], KSPLIT_DC) if KSPLIT_DC > 1 and M >= KSPLIT_MIN_ROWS else 1

        def make_acc():
            if PROBE_SKIP & 2:
                return torch.zeros(M, C, device=dev)
            if dc_early is not None:  # (on whichever stream runs the dsilu: it waits for the early half)
                acc2, ev_e = dc_early
                torch.cuda.current_stream().wait_event(ev_e)
                mm_tf32_out(dmod[:, : 2 * C], w_mod[: 2 * C], acc2[0])
                return acc2
            return mm_tf32_ks(dmod, w_mod, ksc) if ksc > 1 else mm_tf32(dmod, w_mod)

        if dce is not None:  # r3-swa part 19: the GEMM's last third + silu' (+ the stack's sum) in one launch (DcgTf32.late)
            lrun = K.dcg.late(dmod, w_mod, dce[0], c_mod)
            if dca is not None:
                dqx, dcx = dca[0].late(dca[1], lrun, dq)
                out[0] = dqx.view(N, S, C)
            else:
                dcx = lrun()
            out += [dcx, res["dw"]]
        elif dca is not None:  # round 3 part 14: the stack's dc sum (DcAcc): no autograd add / cast after the blocks
            dqx, dcx = dca[0].step(K, dca[1], make_acc, c_mod, dq, dmod)
            out[0] = dqx.view(N, S, C)
            out += [dcx, res["dw"]]
        else:
            out += [K.mod.dsilu(make_acc(), c_mod), res["dw"]]
        if dca is not None and len(dca) > 2 and dca[2] and dca[1] != 0:
            # round 3 part 14 (DEFER_DW): the weight gradients' side streams joined by the stack's first block (the last backward), not
            # here: the next block's backward starts after this block's dq. Only when every weight's .grad is None and unhooked (the
            # gradients are then stolen by AccumulateGrad without a kernel; nothing reads them before the stack's backward ends)
            # (keep the kernels' inputs and workspace, never the returned gradients: a second reference makes AccumulateGrad copy them)
            mk = [xp] if xp is not None else []
            dca[0].hold([s1, s2, *extra], [dw.keep[:-1], dP, dmod, a_mod] + ([dce[0], dce[2]] if dce is not None else []) + mk)
            return out
        join(s2)
    join(s1, *extra)
    return out
