"""Fused bias-only token DiT inference over a stack of ``modules.bias_only_dit.BiasOnlyDiTBlock``: CUDA and cuBLAS only.

    runner = FusedBiasOnlyDiT(blocks)          # packs every weight once
    P = runner.hoist(pair, mask)               # once per sample(): every block's attention weights, softmax(pair bias)
    single = runner.step(single, cond, P)      # once per solver step

What each call hoists, and why it may:

  hoist   The logits are the pair bias alone, and the pair carries no noise level: every block's attention weights are the
          same at every step and for every sample. One LayerNorm pass over the pair rows (the blocks share its statistics;
          each block's LayerNorm weight is folded into its projection), ONE cuBLAS GEMM for every block's bias, and a row
          softmax that writes P [nb H, L, L] bf16 in place of the bias.
  step    (the 12-launch step; bf16 serves the three-kernel step below by default) Per block: cuBLAS v|g GEMM ->
          ``pv_gate_inf`` (sigmoid(g) * P v, a tcgen05 GEMM with the gate in its epilogue) -> cuBLAS out GEMM -> residual +
          gate + AdaLN rows -> expand GEMM + SwiGLU (``gemm_swiglu2_sm100`` from 3840 rows; below it cuBLAS + the SwiGLU
          rows) -> cuBLAS squeeze GEMM -> residual + gate (+ the next block's AdaLN) rows. The row passes are this family's (``bias_only_dit_rows.cu``, one warp per row). The conditioning tables (AdaLN scale /
          shift, output gates) of every block come from two GEMMs over L rows when the samples share one conditioning, over
          S L rows when each has its own. The residual stream is fp32.

fp32 (``dtype=torch.float32``, the TF32 recipe): the same schedule with every activation, table and weight fp32. The hoist is
``pair_bias`` (LN(pair) Wf^T in exact fp32, ``bias_only_dit_f32_rows.cu``) and the fp32 softmax, P [nb H, L, L] fp32; the core is
``pv_gate_tf32`` (kind::tf32 MMAs, P streamed through shared memory); GEMMs are cuBLAS on TF32 tensor cores (``tf32_gemms``); the
expand GEMM is cuBLAS + the fp32 SwiGLU rows; the conditioning tables and the opt-in fused kernels (cond tables, GEMM + residual +
AdaLN) stay bf16-only -- the fp32 step runs the default composition.

fp32, three kernels per block (the default; settings.bias_only_dit_inf3=0 keeps the composition above; read per call, so both
steps run in one process): the
conditioning tables of every block are HOISTED out of the step -- [T, nb, 6, 768] = (gate1, gate2, s1, s2, sh1, sh2) with the four
sigmoids applied, made by the LayerNorm rows + two cuBLAS TF32 GEMMs + one cat + one sigmoid once per conditioning tensor
(``_tables3``: ``kernels._capture.lookup_inputs``, keyed on the tensor's address / version / layout with a weak reference; inside a
CUDA-graph capture scoped to the capture, i.e. remade by every replay, unless ``static_inputs()`` declares the conditioning fixed --
then a replay runs none of those kernels) -- and each block is ``bo_front_tf32`` (LN + AdaLN + the v|g GEMM) -> ``pv_gate_tf32
-DPDL_INF`` -> ``bo_tail_tf32`` (out GEMM, residual + gate, LN + AdaLN, a|b GEMM, SwiGLU, squeeze GEMM, residual + gate), chained by
programmatic dependent launch, on weights rounded to the nearest TF32 once per pack (``_pack3``). A failed build warns once and keeps
the composition above. Front and tail exchange their GEMM operands inside each cluster through L2 scratch (stores, one release,
TMA-fed GEMM; ``_buffers3``).

bf16, three kernels per block (the default; settings.bias_only_dit_inf3_bf16=0 keeps the 12-launch step above, as does a failed
build; read per call): the conditioning tables HOISTED as for fp32 but bf16 ([T, nb, 6, 768], the four sigmoids applied: ``_tables3b``) and each block ``bo_front_bf16``
(LN + AdaLN + v|g GEMM) -> ``pv_gate_inf -DPDL_INF`` -> ``bo_tail_bf16`` or, where it takes fewer rounds, the pair tail
``bo_tail2_bf16`` (out GEMM, residual + gate, LN + AdaLN, a|b GEMM, SwiGLU, squeeze GEMM, residual + gate); bf16 operands, fp32
accumulators, LayerNorm and residual; the block input is the step's bf16 single, then the previous tail's fp32 output; the last
tail writes the step's dtype.

Tried and not kept: splitting the samples over two or three CUDA streams (each chain leaves SMs idle at these sizes, but the
core and cuBLAS's kernels each fill an SM's shared memory, so the chains did not overlap: L384 70 -> 80 us), and the
output-gate table GEMM on a side stream (no change).
"""

from __future__ import annotations

import weakref

import torch

from miniworld_engine.kernels._device import num_sms

from miniworld_engine.kernels.bias_only_dit import cuda as C

EPS = 1e-5
#: Code switches for two measured-slower kernel paths kept for A/B (formerly settings.bias_only_dit_cond / _resln, default off).
_COND_TABLES = False
_RESLN_GEMM = False
#: fp32 hoist of the 256-wide pair (``FusedBiasOnlyDiT.hoist``): products of the TF32 pair-bias kernel (``pair_bias_tc``), 3 = split fp32
_HOIST_TC_TERMS = 3


def _tf32():
    """The fp32 path's kernels (``cuda/tf32.py``)."""
    from miniworld_engine.kernels.bias_only_dit.cuda import tf32

    return tf32


def _rows():
    """The token DiT's CUDA row kernels: the pair LayerNorm (C = 128 or 256) of the hoist."""
    from miniworld_engine.kernels.conditioned_transition import cuda as rows

    return rows


class _Lazy:
    """A per-key pack made on first use (``make(key)``) and kept -- only when made in the capture state this holder was created in (a
    pack recorded inside a capture is not computed until a replay; one made eagerly is not remade by a replay): otherwise made per call."""

    def __init__(self, make):
        self.make, self.made, self.capturing = make, {}, torch.cuda.is_current_stream_capturing()

    def __getitem__(self, key):
        if torch.cuda.is_current_stream_capturing() != self.capturing:
            return self.make(key)
        if key not in self.made:
            self.made[key] = self.make(key)
        return self.made[key]


#: B200 inference hoist, bf16: the pair LayerNorm + bias projection in ONE pass (the training block's ``pair_bias`` kernel) at pair 128 / 256
_B200_HOIST1 = True
#: B200 inference hoist, fp32: the tensor-core pair-bias kernel (``pair_bias_tc``, _HOIST_TC_TERMS products) at pair 128 as at 256
_B200_HOIST_TC = True


def _b200(device) -> bool:
    return torch.cuda.get_device_capability(device) == (10, 0)


class _TailForms:
    """One weight's tail-kernel forms per cluster size (B200 three-kernel step): the constructor's pack launch makes those of the step's
    cluster (``add``); another size is made on first use by a pack launch of its own, kept only when made in the runner's own capture
    state (``FusedBiasOnlyDiT._lazy_ok``; ``_Lazy``'s rule)."""

    def __init__(self, runner, make):
        self.runner, self.make, self.made = runner, make, {}

    def add(self, pack, cl):
        self.made[cl] = self.make(pack, cl)

    def __getitem__(self, cl):
        if cl in self.made:
            return self.made[cl]
        from miniworld_engine.kernels.bias_only_dit.cuda import pack_sm100

        P = pack_sm100.Pack(self.runner.dtype)
        t = self.make(P, cl)
        P.launch(t.device)
        if self.runner._lazy_ok():
            self.made[cl] = t
        return t


class FusedBiasOnlyDiT:
    def __init__(self, blocks, dtype=torch.bfloat16, rows: int | None = None):
        """``rows``: the step's S L rows when known (B200: the pack then makes the tail's weight forms for that row count's cluster only,
        others on first use)."""
        assert dtype in (torch.bfloat16, torch.float32), "bf16, or fp32 on TF32 tensor cores"
        blocks = list(blocks)
        a0 = blocks[0].attention
        self.nb = len(blocks)
        self.h = a0.n_head
        self.d = a0.to_out.weight.shape[0]  # the single width
        self.da = a0.to_value.weight.shape[0]  # the attention's: n_head x head width (768 or 1024)
        self.dc = a0.ada_ln_in.ln_cond.weight.shape[0]
        self.dp = a0.ln_pair.weight.shape[0]
        self.fp32 = dtype is torch.float32
        dev = a0.to_value.weight.device
        self._p3 = None  # the fp32 three-kernel step's TF32-rounded weight pack
        self._p3b = None  # the bf16 three-kernel step's weight pack
        self._capturing = torch.cuda.is_current_stream_capturing()  # built inside a CUDA-graph capture (``_lazy_ok``)
        self.dtype = dtype
        self._ops: dict = {}
        self._pack_side = None  # the side stream of a captured pack, until the first step joins it (``_join``)
        self._hoist_side = None  # the side stream of a captured pack's pair projection and of the hoist (``_join_hoist``)
        self._tab_bias = None  # (b1, [b2 | b1]): the one-block table pass's bias vector (``_tab_b200``; the B200 pack makes it)
        self._blocks = blocks  # B200: the 12-launch step's natural pack is made from them on first use (``_step_natural``)
        self.w1 = None
        if not self._pack_b200(blocks, dtype, dev, rows):
            self._pack_torch(blocks, dtype, dev)
        # the one-kernel conditioning tables (cond_tables_sm100.cu): [W1; W2], their biases, and the column sums of the bf16 W1
        # rows (LN(c) W1^T = rstd (c W1^T - mu colsum(W1)))
        self.n_g1 = self.nb * 4 * self.d
        self._wcc = None  # (wc, bc, csc): made on first use (``_cond_pack``; the B200 three-kernel step never reads them)
        self._buf: dict = {}
        self._tab3: dict = {}  # hoisted conditioning tables of the three-kernel step

    def _pack_torch(self, blocks, dtype, dev):
        """The weight pack by torch ops (every card; on B200 where ``_pack_b200`` does not apply)."""
        f32 = lambda t: t.detach().float()
        w1, b1, w2, b2, pw = [], [], [], [], []
        self.per = []
        for blk in blocks:
            at, tr = blk.attention, blk.transition
            la, lt = f32(at.ada_ln_in.ln_cond.weight), f32(tr.ada_ln_in.ln_cond.weight)
            # AdaLN scale / shift of both halves on LayerNorm(cond), the norm weight folded in
            w1 += [
                f32(at.ada_ln_in.to_scale.weight) * la,
                f32(at.ada_ln_in.to_bias.weight) * la,
                f32(tr.ada_ln_in.to_scale.weight) * lt,
                f32(tr.ada_ln_in.to_bias.weight) * lt,
            ]
            z = torch.zeros(self.d, device=dev)
            b1 += [
                f32(at.ada_ln_in.to_scale.bias),
                z,
                f32(tr.ada_ln_in.to_scale.bias),
                z,
            ]
            # the two output gates read the raw conditioning
            w2 += [f32(at.to_scale.weight), f32(tr.to_scale.weight)]
            b2 += [f32(at.to_scale.bias), f32(tr.to_scale.bias)]
            pw.append(f32(at.to_bias.weight) * f32(at.ln_pair.weight))  # [H, dp], ln_pair weight folded
            self.per.append(
                dict(
                    wvg=torch.cat([at.to_value.weight, at.to_gate.weight], 0).detach().to(dtype).contiguous(),
                    wo=at.to_out.weight.detach().to(dtype).contiguous(),
                    wab=torch.cat([tr.expand_a.weight, tr.expand_b.weight], 0).detach().to(dtype).contiguous(),
                    ws=tr.squeeze.weight.detach().to(dtype).contiguous(),
                )
            )
        self.w1 = torch.cat(w1, 0).to(dtype).contiguous()  # [nb 4 d, dc]
        self.b1 = torch.cat(b1, 0).to(dtype).contiguous()
        self.w2 = torch.cat(w2, 0).to(dtype).contiguous()  # [nb 2 d, dc]
        self.b2 = torch.cat(b2, 0).to(dtype).contiguous()
        self.pw = torch.cat(pw, 0).to(dtype).contiguous()  # [nb H, dp]

    def _step_natural(self, single, cond, P, out_dtype, kmask):
        """The 12-launch step on a B200 runner, whose constructor packed only the three-kernel step's forms (``_pack_b200``; reached only
        when that step does not serve): the natural pack by torch ops first, kept only in the runner's own capture state (``_lazy_ok``),
        else made for this call and dropped after it."""
        self._join(single.device)
        self._join_hoist(single.device)
        if self.w1 is not None:
            return self._step(single, cond, P, out_dtype, kmask)
        saved = (self.w2, self.b2, self.pw, self.per)
        self._pack_torch(self._blocks, self.dtype, self.w2.device)
        try:
            return self._step(single, cond, P, out_dtype, kmask)
        finally:
            if not self._lazy_ok():
                self.w2, self.b2, self.pw, self.per = saved
                self.w1 = None
                self._buf.clear()

    def _pack_b200(self, blocks, dtype, dev, rows=None) -> bool:
        """B200, where the step runs the three-kernel path (bf16 ``_inf3b_ok`` / fp32 ``_inf3_ok``): the pack that step reads -- the AdaLN
        table weights in its order, their biases, the gate projections, the folded pair projection, per block [Wv; Wg], [Wa; Wb] (bf16
        as cast; fp32 rounded to the nearest TF32) and Wo / Wsq in the tail kernels' pair-packed forms -- in ONE launch per block
        (``bo_pack_sm100.cu``), the same values as ``_pack_torch`` + ``_pack3`` / ``_pack3b``, bit for bit. A CUDA-graph replay re-runs the
        pack (it is scoped to the capture), so it makes only what the step reads (2026-11-17: r104 L128 S5 pack 36 / 42 us -> see
        docs/b200/bias-only.md): the tail forms for the cluster of ``rows`` (all of them when None; another one on first use,
        ``_TailForms``), none of the 12-launch step's natural forms (``_step_natural`` on first use). False: not this card / path (the torch
        pack then)."""
        if torch.cuda.get_device_capability(dev) != (10, 0) or self.d != 768:
            return False
        if not (self._inf3_ok(dev) if self.fp32 else self._inf3b_ok(dev)):
            return False
        from miniworld_engine.kernels.bias_only_dit.cuda import inf3_bf16 as B16
        from miniworld_engine.kernels.bias_only_dit.cuda import pack_sm100

        nb, D, DA, dc, H, dp = self.nb, self.d, self.da, self.dc, self.h, self.dp
        e = lambda *shape: torch.empty(*shape, device=dev, dtype=dtype)  # noqa: E731
        self.w2, self.pw, w1r = e(nb * 2 * D, dc), e(nb * H, dp), e(nb * 4 * D, dc)
        bb = e(nb * 6 * D)  # [b2 | b1r]: one block's table pass reads them as one vector (``_tab_b200``)
        self.b2, b1r = bb[: nb * 2 * D], bb[nb * 2 * D :]
        if nb == 1:
            self._tab_bias = (b1r, bb)
        rnd = 2 if self.fp32 else None  # the tail kernels' operands: TF32-rounded fp32, or bf16 as they are
        tail = (self._ops3(dev) if self.fp32 else self._ops3b(dev))[2]
        tpair = _tf32().TAIL_PAIR if self.fp32 else B16.TAIL_PAIR
        if rows is not None and rows % 128 == 0:
            cl = tail.cluster(rows // 128)
            want = {8} if cl == tpair else {cl}
        else:
            want = set(_tf32().TAIL_CLUSTERS if self.fp32 else B16.TAIL_CLUSTERS)
        pair = rows is not None and rows % 128 == 0 and tail.cluster(rows // 128) == tpair
        bw = 32 if self.fp32 else 64
        # Inside a CUDA-graph capture (the pack is then re-run by every replay) the block step's weights (~5 M of the ~7 M elements) are
        # packed on a side stream forked here and joined by the runner's first step (``_join``): the pack is latency-bound (~3 TB/s),
        # and the hoist and the conditioning tables before the first block read none of them. Eagerly: one stream.
        # The pair projection (a few K elements) goes on a second stream forked here, on which ``hoist`` then runs the pair part beside the
        # conditioning tables (``side.py``): the tables and the pair part read nothing of each other until the first core.
        from miniworld_engine.kernels.bias_only_dit.cuda import side as SD

        side, hside = SD.fork(dev), SD.fork(dev)
        self.per, packs = [], []
        for i, blk in enumerate(blocks):
            at, tr = blk.attention, blk.transition
            # P: what the tables read; Q: the block step's weights; Ph: what the hoist reads
            P, Q, Ph = pack_sm100.Pack(dtype), pack_sm100.Pack(dtype), pack_sm100.Pack(dtype)
            la, lt = at.ada_ln_in.ln_cond.weight, tr.ada_ln_in.ln_cond.weight
            ws1, wb1 = at.ada_ln_in.to_scale.weight, at.ada_ln_in.to_bias.weight
            ws2, wb2 = tr.ada_ln_in.to_scale.weight, tr.ada_ln_in.to_bias.weight
            r4 = lambda t, j: t[(4 * i + j) * D : (4 * i + j + 1) * D]  # noqa: E731
            for j, (w, s) in enumerate(((ws1, la), (ws2, lt), (wb1, la), (wb2, lt))):  # the three-kernel order: s1, s2, sh1, sh2
                P.copy(w, r4(w1r, j), s)
            for j, bias in enumerate((at.ada_ln_in.to_scale.bias, tr.ada_ln_in.to_scale.bias, None, None)):
                P.copy(bias, r4(b1r, j)) if bias is not None else P.zero(r4(b1r, j))
            for j, (w, bias) in enumerate(((at.to_scale.weight, at.to_scale.bias), (tr.to_scale.weight, tr.to_scale.bias))):
                P.copy(w, self.w2[(2 * i + j) * D : (2 * i + j + 1) * D])
                P.copy(bias, self.b2[(2 * i + j) * D : (2 * i + j + 1) * D])
            Ph.copy(at.to_bias.weight, self.pw[i * H : (i + 1) * H], at.ln_pair.weight)  # ln_pair weight folded
            wvg, wab = e(2 * DA, D), e(4 * D, D)  # bf16: cast; fp32: rounded to the nearest TF32
            Q.copy(at.to_value.weight, wvg[:DA], ddt=rnd)
            Q.copy(at.to_gate.weight, wvg[DA:], ddt=rnd)
            Q.copy(tr.expand_a.weight, wab[: 2 * D], ddt=rnd)
            Q.copy(tr.expand_b.weight, wab[2 * D :], ddt=rnd)
            wo, wsq = at.to_out.weight, tr.squeeze.weight
            fp32 = self.fp32
            blk_pack = dict(
                wvg=wvg,
                wab=wab,
                wo=_TailForms(self, lambda Q, cl, w=wo: Q.pairs(w, cl, bw, ddt=rnd)),
                # bf16 CL 8 tail: Wsq in each CTA's z order
                wsq=_TailForms(
                    self, lambda Q, cl, w=wsq: Q.pairs(w, cl, bw, zo=(2 * D // cl // 64 if (cl == 8 and not fp32) else 0), ddt=rnd)
                ),
            )
            if not self.fp32:
                blk_pack["wsqn"] = _TailForms(self, lambda Q, cl, w=wsq: Q.pairs(w, 8, bw))  # the pair tail: natural k order
            for cl in sorted(want):
                blk_pack["wo"].add(Q, cl)
                if self.fp32 or not pair or cl != 8:
                    blk_pack["wsq"].add(Q, cl)
            if pair and not self.fp32:
                blk_pack["wsqn"].add(Q, 8)
            with SD.on(hside):
                Ph.launch(dev)
            P.launch(dev)
            with SD.on(side):  # inside a capture: on a side stream, beside the hoist and the tables (joined by the first step)
                Q.launch(dev)
            packs.append(blk_pack)
        self._pack_side, self._hoist_side = side, hside
        if self.fp32:
            self._p3 = (w1r, b1r, packs)
        else:
            self._p3b = (w1r, b1r, packs)
        return True

    def _join(self, device) -> None:
        """The first step after a captured pack waits for its side stream (``_pack_b200``); later steps of the capture are ordered after
        that one."""
        side = self._pack_side
        if side is not None and torch.cuda.is_current_stream_capturing():
            torch.cuda.current_stream(device).wait_stream(side)
            self._pack_side = None

    def _join_hoist(self, device) -> None:
        """A step's first core after a captured pack / hoist waits for their side stream (``_pack_b200``, ``hoist``)."""
        side = self._hoist_side
        if side is not None and torch.cuda.is_current_stream_capturing():
            torch.cuda.current_stream(device).wait_stream(side)
            self._hoist_side = None

    def _cond_pack(self):
        """[W1; W2], their fp32 biases and the fp32 column sums of the W1 rows (zeros for W2): the one-kernel conditioning tables'
        operands, made on first use and kept with the runner (a captured runner makes them inside its capture, as the rest of its pack)."""
        assert self.w1 is not None, "the 12-launch step's natural pack (_step_natural)"

        def build():
            dev = self.w1.device
            return (
                torch.cat([self.w1, self.w2], 0).contiguous(),
                torch.cat([self.b1, self.b2], 0).float().contiguous(),
                torch.cat([self.w1.float().sum(1), torch.zeros(self.w2.shape[0], device=dev)], 0).contiguous(),
            )

        if not self._lazy_ok():  # made in the other capture state than the runner: not kept (see _lazy_ok)
            return build()
        if self._wcc is None:
            self._wcc = build()
        return self._wcc

    def _lazy_ok(self) -> bool:
        """A part of the pack made after the constructor is kept only when made in the runner's own capture state: inside a capture the
        kernels only record (an eager runner read there would keep unwritten memory), eagerly they run once (a captured runner's replays
        would not remake it)."""
        return torch.cuda.is_current_stream_capturing() == self._capturing

    @property
    def wc(self):
        return self._cond_pack()[0]

    @property
    def bc(self):
        return self._cond_pack()[1]

    @property
    def csc(self):
        return self._cond_pack()[2]

    # ------------------------------------------------------------------ once per sample()
    def hoist(self, pair, mask=None):
        """pair [1, L, L, dp] -> every block's attention weights P [nb H, L, L] in the runner's dtype (rows sum to 1 over the keys).
        ``mask`` [L] bool marks the real tokens; the other keys get zero weight. Inside the capture of a B200 pack: on that pack's hoist
        stream (``_pack_b200``), joined by the step's first core (the caller keeps ``pair`` / ``mask`` alive until then: a step follows)."""
        from miniworld_engine.kernels.bias_only_dit.cuda import side as SD

        hs = self._hoist_side if torch.cuda.is_current_stream_capturing() else None
        with SD.on(hs):
            return self._hoist(pair, mask)

    def _hoist(self, pair, mask=None):
        L = pair.shape[1]
        z2d = pair.reshape(L * L, self.dp)
        m = None if mask is None else mask.reshape(L).to(torch.bool).contiguous()
        if self.fp32:
            # one pass of the pair per block: its LayerNorm and the H biases on the FMA pipe, exact fp32 (no LN(pair) in memory)
            R32 = _tf32().rows32()
            P = torch.empty(self.nb * self.h, L, L, device=pair.device, dtype=torch.float32)
            z32 = z2d.float().contiguous()
            # the 256-wide pair: the same LayerNorm + projection with the products on TF32 tensor cores, split fp32 (3 terms, ~exact)
            # -- on the FMA pipe its 2 x 256 x H FMAs per pair row were the whole step (~1 ms of 1.85 at L768, the block no faster
            # than PyTorch); 128 keeps the FMA kernel off H100 -- on H100 the FMA kernel was 0.16 of a 0.46 ms step at L 512 (behind the
            # Triton block), so both widths take the tensor-core kernel there
            # B200 (2026-11-17): the tensor-core kernel at 128 too (_B200_HOIST_TC: the FMA kernel was 90.7 of a 299 us step at r104 L512 S5)
            tc = self.dp == 256 or C.h100_enabled(pair.device) or (_B200_HOIST_TC and _b200(pair.device))
            pst = torch.empty(L * L, 2, device=pair.device) if tc else None
            for b in range(self.nb):
                if pst is not None:
                    R32.pair_bias_tc_cuda(
                        z32,
                        self.pw[b * self.h : (b + 1) * self.h].contiguous(),
                        P[b * self.h : (b + 1) * self.h],
                        pst,
                        EPS,
                        _HOIST_TC_TERMS,
                    )
                    continue
                R32.pair_bias_cuda(
                    z32,
                    self.pw[b * self.h : (b + 1) * self.h],
                    P[b * self.h : (b + 1) * self.h],
                    None,
                    EPS,
                )
            R32.softmax_rows_cuda(P.view(-1, L), P.view(-1, L), m)
            return P
        if (
            self.dtype == torch.bfloat16
            and (L * L) % 128 == 0
            and self.h in (12, 16, 24)
            and ((self.dp == 256 and C.h100_enabled(pair.device)) or (_B200_HOIST1 and self.dp in (128, 256) and _b200(pair.device)))
        ):
            # B200 (2026-11-17, _B200_HOIST1): the same one-pass kernel at pair 128 and 256 -- the LayerNorm rows + GEMM were 39.0 + 13.1 of a
            # 190 us step at r104 master L512 S5
            # the 256-wide bf16 pair in ONE pass (the training block's pair_bias kernel, bias_only_dit_train_rows.cu: LayerNorm on chip,
            # the H biases on mma.sync, head-major): the LayerNorm rows below write LN(pair) (as large as the pair) and the GEMM reads it
            # back -- three passes over a 256-wide pair, 0.31 of a 0.59 ms step at L 768 (H100), slower than the Triton block's one pass
            from miniworld_engine.kernels.bias_only_dit.cuda.train import ext as train_ext

            P = torch.empty(self.nb * self.h, L, L, device=pair.device, dtype=self.dtype)
            pst = torch.empty(L * L, 2, device=pair.device, dtype=torch.float32)  # the row statistics the kernel writes (unused here)
            z = z2d.contiguous()
            for b in range(self.nb):
                train_ext().pair_bias_cuda(
                    z, self.pw[b * self.h : (b + 1) * self.h].contiguous(), P[b * self.h : (b + 1) * self.h], pst, EPS
                )
            C.softmax_rows(P.view(-1, L), P.view(-1, L), m)
            return P
        zh = torch.empty(L * L, self.dp, device=pair.device, dtype=self.dtype)
        if self.dp == 128:
            _rows()._ext().layernorm128_rows(z2d, zh, EPS, False)
        else:  # the same row kernel template at 256 channels (layernorm_rows_kernel<256>)
            _rows()._ext().layernorm_rows_cuda(z2d, zh, EPS)
        P = torch.empty(self.nb * self.h, L, L, device=pair.device, dtype=self.dtype)
        torch.mm(self.pw, zh.t(), out=P.view(self.nb * self.h, L * L))
        C.softmax_rows(P.view(-1, L), P.view(-1, L), m)
        return P

    def _buffers(self, S, L, dev):
        key = (S, L, dev)
        if key not in self._buf:
            M, D = S * L, self.d
            e = lambda n, dt=self.dtype: torch.empty(M, n, device=dev, dtype=dt)
            self._buf[key] = dict(
                x=e(D, torch.float32),
                xa=e(D),
                vg=e(2 * self.da),
                vt=torch.empty(self.da, M, device=dev, dtype=self.dtype),  # Hopper fp32: v^T (K-major TF32 operand)
                gg=e(self.da),  # Hopper fp32: the gate g = xa Wg^T
                a=e(self.da),
                y=e(D),
                ab=e(self.per[0]["wab"].shape[0]),
                h=e(self.per[0]["wab"].shape[0] // 2),
            )
        return self._buf[key]

    def _core(self, device):
        idx = device.index if device.index is not None else torch.cuda.current_device()
        if ("core", idx) not in self._ops:
            self._ops[("core", idx)] = (
                (
                    C.PvGateCoreH100TF32(idx, nh=self.h, dh=self.da // self.h)
                    if C.h100_enabled(idx)
                    else _tf32().PvGateCoreTF32(idx, nh=self.h, dh=self.da // self.h)
                )
                if self.fp32
                else C.PvGateCore(idx, nh=self.h, dh=self.da // self.h)
            )
        return self._ops[("core", idx)]

    def _cond(self, device):
        """The one-kernel conditioning tables (settings.bias_only_dit_cond=1); off by default: slower than LayerNorm rows + two
        cuBLAS GEMMs at every measured shape (the sigmoids of the tables on 256 epilogue threads per SM dominate its time)."""
        if not _COND_TABLES or self.fp32:
            return None
        idx = device.index if device.index is not None else torch.cuda.current_device()
        if ("cond", idx) not in self._ops:
            self._ops[("cond", idx)] = C.CondTables(idx)
        return self._ops[("cond", idx)]

    def _resln(self, device, final, presig):
        """The width-768 GEMM with residual + gate + AdaLN (or the final output) in its epilogue; None: cuBLAS + rows.
        Off by default until the conditioning tables carry their sigmoids (settings.bias_only_dit_resln=1 turns it on)."""
        if not _RESLN_GEMM or self.fp32:
            return None
        idx = device.index if device.index is not None else torch.cuda.current_device()
        key = ("resln", idx, final, presig)
        if key not in self._ops:
            self._ops[key] = C.ResLnGemm(idx, final=final, presig=presig)
        return self._ops[key]

    def _gemm_swiglu(self, device, xa):
        """The sm_100a expand GEMM with the SwiGLU epilogue where it fits (M >= its row threshold); None: cuBLAS + rows."""
        idx = device.index if device.index is not None else torch.cuda.current_device()
        key = ("gsw", idx, xa.shape[0])
        if self.fp32:
            return None
        if C.h100_enabled(idx):
            # Hopper: expand + SwiGLU as one sm_90a GEMM (the token DiT's ``gemm_swiglu_sm90``): [a | b] never reaches HBM and
            # the SwiGLU row pass (10 us at L768) disappears
            if key not in self._ops:
                from miniworld_engine.kernels.conditioned_transition.cuda import gemm_swiglu

                wab = self.per[0]["wab"]
                # its 128 x 256 tiles under-fill the SMs at few rows (<= half a wave) and lose a whole second wave just past one:
                # there cuBLAS + the SwiGLU rows is faster (2026-10-19, us: M128 10.6 -> 6.0, M640 10.6 -> 9.7, M1536 19.2 -> 17.6)
                tiles = (xa.shape[0] // 128) * (wab.shape[0] // 256)
                nsm = num_sms(idx)
                small = _H100_EXPAND_ROWS and (tiles <= nsm // 2 or nsm < tiles <= nsm * 11 // 10)
                self._ops[key] = gemm_swiglu.GemmSwigluSm90(idx) if gemm_swiglu.supported_sm90(xa, wab) and not small else None
            return self._ops[key]
        if key not in self._ops:
            from miniworld_engine.kernels.conditioned_transition.cuda import gemm_swiglu

            wab = self.per[0]["wab"]
            self._ops[key] = gemm_swiglu.GemmSwiglu(idx, K=wab.shape[1], H=wab.shape[0] // 2) if gemm_swiglu.supported(xa, wab) else None
        return self._ops[key]

    # ------------------------------------------------------------------ once per solver step
    def step(self, single, cond, P, out_dtype=None, kmask=None):
        """One solver step: single [S, 1, L, D], cond [S or 1, 1, L, dc] (one conditioning shared by the samples when
        its sample axis is 1 or has stride 0), P from ``hoist``. Returns [S, 1, L, D] in ``out_dtype`` (single's). ``kmask``: the bool key
        mask [1, L] P was made with (``kernels._key_len``) or None: the H100 TF32 core (fp32) skips the key blocks past its last attended key,
        where P is 0 in every row (the same bits; the sm_100 three-kernel steps take it and leave their cores as they are)."""
        if not self.fp32:
            if self._inf3b_ok(single.device):
                return self._step3b(single, cond, P, out_dtype, kmask)
            return self._step_natural(single, cond, P, out_dtype, kmask)
        with _tf32().tf32_gemms():
            if self._inf3_ok(single.device):
                return self._step3(single, cond, P, out_dtype, kmask)
            return self._step_natural(single, cond, P, out_dtype, kmask)

    # ------------------------------------------------------------------ fp32, three kernels per block (default; INF3=0: off)
    def _inf3_ok(self, device) -> bool:
        T = _tf32()
        if not (self.fp32 and T.inf3_on() and self.d == 768 and self.da in (768, 1024)):
            return False
        if C.h100_enabled(device):  # the three-kernel step is tcgen05 (sm_100a)
            return False
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return T.inf3_ready(idx, self.h, self.da // self.h)

    def _pack3(self):
        """The three-kernel step's weights, once per runner (i.e. per weight version): per block [Wv; Wg], Wo, [Wa; Wb], Wsq rounded to
        the nearest TF32; the AdaLN table weights reordered so that one GEMM gives (s1, s2, sh1, sh2) per block."""
        if self._p3 is None:
            rnd, pairs = _tf32().round_tf32, _tf32().pack_pairs
            nb, D, dc = self.nb, self.d, self.dc
            # (s1, sh1, s2, sh2) -> (s1, s2, sh1, sh2) by slices + cat: device ops only (a Python index list would make a host index
            # tensor, which a CUDA-graph capture refuses to copy -- the pack runs inside a capture whenever the weights are not static)
            w4, b4 = self.w1.view(nb, 4, D, dc), self.b1.view(nb, 4, D)
            w1 = torch.cat([w4[:, 0:1], w4[:, 2:3], w4[:, 1:2], w4[:, 3:4]], 1).reshape(nb * 4 * D, dc).contiguous()
            b1 = torch.cat([b4[:, 0:1], b4[:, 2:3], b4[:, 1:2], b4[:, 3:4]], 1).reshape(nb * 4 * D).contiguous()
            # Wo / Wsq pair-packed per tail cluster size (8 / 6): the tail loads two k-blocks of a CTA's output rows as one TMA box
            tcl = _tf32().TAIL_CLUSTERS
            blocks = []
            for p in self.per:
                wo, ws = rnd(p["wo"]), rnd(p["ws"])
                blocks.append(
                    dict(
                        wvg=rnd(p["wvg"]),
                        wab=rnd(p["wab"]),
                        wo={cl: pairs(wo, cl) for cl in tcl},
                        wsq={cl: pairs(ws, cl) for cl in tcl},
                    )
                )
            self._p3 = (w1, b1, blocks)
        return self._p3

    def _tables3(self, cond, S, L):
        """The hoisted conditioning tables [T, nb, 6, 768] fp32 (T = L when the samples share one conditioning, else S L): per block
        sigmoid(gate1), sigmoid(gate2), sigmoid(s1), sigmoid(s2), sh1, sh2 -- made once per conditioning tensor (address, in-place
        version, layout; a weak reference guards against a freed address reused by another tensor) and reused by every later step
        that passes it unchanged. ``kernels._capture.lookup_inputs`` scopes the entries to a CUDA-graph capture (a replay remakes
        them) unless ``static_inputs()`` is on."""
        from miniworld_engine.kernels import _capture

        nb, D = self.nb, self.d
        shared = cond.shape[0] == 1 or cond.stride(0) == 0
        w1, b1, _ = self._pack3()

        def build():
            c = (cond[0, 0] if shared else cond.reshape(S * L, self.dc)).float()
            T = c.shape[0]
            cn = torch.empty(T, self.dc, device=c.device, dtype=torch.float32)
            C.ln_rows(c, cn, EPS)
            if self._tab_one():
                return self._tab_b200(cn, c, w1, b1, T)
            g1 = torch.addmm(b1, cn, w1.t()).view(T, nb, 4, D)  # s1, s2, sh1, sh2
            g2 = torch.addmm(self.b2, c, self.w2.t()).view(T, nb, 2, D)  # gate1, gate2
            tab = torch.cat([g2, g1], 2)  # gate1, gate2, s1, s2, sh1, sh2
            tab[:, :, :4].sigmoid_()
            return tab

        key = (
            "tab3",
            cond.data_ptr(),
            cond._version,
            tuple(cond.shape),
            cond.stride(),
            cond.dtype,
            cond.device,
            S,
            L,
        )
        entry = _capture.lookup_inputs(
            self._tab3,
            key,
            lambda: (weakref.ref(cond), build()),
            limit=16,
            valid=lambda e: e[0]() is cond,
            alive=lambda e: e[0]() is not None,
            inputs=(cond,),
        )
        return entry[1]

    def _tab_one(self) -> bool:
        """B200, one block: the three-kernel step's table GEMMs write into the table and ``bo_tab_sm100.cu`` adds the biases and the sigmoids
        in one pass (``_tab_b200``)."""
        return self.nb == 1 and torch.cuda.get_device_capability(self.w2.device) == (10, 0)

    def _tab_b200(self, cn, c, w1, b1, T):
        """The table [T, 1, 6, 768] (gate1, gate2, s1, s2, sh1, sh2; the first four through their sigmoid) of one block: the two GEMMs
        (no bias) straight into its columns (cuBLAS takes the row pitch), then ONE pass (``bo_tab_sm100.cu``) adds [b2 | b1] and applies
        the sigmoids -- in place of two biased GEMMs, a cat and a strided sigmoid (four launches, two strided elementwise kernels: 26 us at
        r104 master L512 S5). The bias vector [b2 | b1] is made once per runner."""
        from miniworld_engine.kernels.bias_only_dit.cuda import pack_sm100

        D, dev = self.d, cn.device
        tab = torch.empty(T, 1, 6, D, device=dev, dtype=self.dtype)
        t2 = tab.view(T, 6 * D)
        torch.mm(c, self.w2.t(), out=t2[:, : 2 * D])  # gate1, gate2
        torch.mm(cn, w1.t(), out=t2[:, 2 * D :])  # s1, s2, sh1, sh2
        bias = self._tab_bias
        if bias is None or bias[0] is not b1:
            bias = (b1, torch.cat([self.b2, b1]).contiguous())
            if torch.cuda.is_current_stream_capturing() == self._capturing:
                self._tab_bias = bias
        n = T * 6 * D // 4
        pack_sm100.tab_kernel(dev)(((n + 255) // 256, 1, 1), (256, 1, 1), t2, bias[1], T, 6 * D, 4 * D, int(self.fp32))
        return tab

    def _ops3(self, device):
        """(front, core, tail) of the three-kernel step."""
        T = _tf32()
        idx = device.index if device.index is not None else torch.cuda.current_device()
        key = ("inf3", idx)
        if key not in self._ops:
            self._ops[key] = (
                T.FrontTF32(idx, self.da),
                T.PvGateCoreTF32(idx, nh=self.h, dh=self.da // self.h, pdl=True),
                T.TailTF32(idx, self.da),
            )
        return self._ops[key]

    def _buffers3(self, S, L, dev):
        """v|g, a, the residual ping-pong of a multi-block stack, and the clusters' operand-exchange scratch (xa, xt [M, 768] rows, h
        [M 48, 32] blocked k-block-major: written and read within one kernel, L2-resident; xt / h padded to whole tile pairs for the
        pair tail, whose missing odd tile writes its own padding rows)."""
        key = ("inf3", S, L, dev)
        if key not in self._buf:
            M, f = S * L, torch.float32
            e = lambda n: torch.empty(M, n, device=dev, dtype=f)  # noqa: E731
            # the scratch rows padded by 128 B: their natural strides (3 / 6 KB) are multiples of 2 KB, so a TMA box's 128 rows
            # would fall on the same address bits above the line (round-4 experiment against L2 slice camping; harmless otherwise)
            pad = lambda n, rows=M: torch.empty(rows, n + 32, device=dev, dtype=f)[:, :n]  # noqa: E731
            Mp = _tf32().tail_rows(M, _tf32().TAIL_PAIR)
            self._buf[key] = dict(
                vg=e(2 * self.da),
                a=e(self.da),
                x=[e(self.d) for _ in range(2 if self.nb > 1 else 0)],
                xa=pad(self.d),
                xt=pad(self.d, Mp),
                h=torch.empty(Mp * 2 * self.d // 32, 32, device=dev, dtype=f),
            )
        return self._buf[key]

    def _step3(self, single, cond, P, out_dtype=None, kmask=None):
        """The fp32 step as three kernels per block: front (LN + AdaLN + v|g GEMM) -> core -> tail (everything after the core)."""
        S, B, L, D = single.shape
        assert B == 1 and L % 128 == 0 and D == self.d
        M, H, DA, dev = S * L, self.h, self.da, single.device
        tab = self._tables3(cond, S, L)
        T = tab.shape[0]
        _, _, packs = self._pack3()
        front_op, core, tail = self._ops3(dev)
        tcl = tail.cluster(M // 128)  # 8, or 6 / the pair tail where it takes fewer rounds
        pair = tcl == _tf32().TAIL_PAIR
        wk, Mt = (
            (8 if pair else tcl),
            _tf32().tail_rows(M, tcl),
        )  # the pair tail: the CL 8 packs, scratch of whole tile pairs
        buf = self._buffers3(S, L, dev)
        self._join(dev)
        self._join_hoist(dev)  # before the front: the core is its programmatic dependent launch and reads P
        x = single.reshape(M, D).float().contiguous()
        out = torch.empty(M, D, device=dev, dtype=torch.float32)
        vg, a = buf["vg"], buf["a"]
        for b, p in enumerate(packs):
            y = out if b + 1 == self.nb else buf["x"][b % 2]
            tb = tab[:, b]
            front_op(x, tb, p["wvg"], vg, T, xa=buf["xa"])
            core(vg[:, :DA], P[b * H : (b + 1) * H].view(H * L, L), a, S, g=vg[:, DA:])  # sigmoid(g) * (P v), TF32-rounded
            tail(
                a,
                x,
                tb,
                p["wo"][wk],
                p["wab"],
                p["wsq"][wk],
                y,
                T,
                xt=buf["xt"][:Mt],
                h=buf["h"][: Mt * 48],
                cl=tcl,
            )
            x = y
        res = out.view(S, 1, L, D)
        return res if out_dtype in (None, torch.float32) else res.to(out_dtype)

    # ------------------------------------------------------------------ bf16, three kernels per block (default; INF3_BF16=0: off)
    def _inf3b_ok(self, device) -> bool:
        from miniworld_engine.kernels.bias_only_dit.cuda import inf3_bf16 as B16

        if (
            self.fp32
            or torch.cuda.get_device_capability(device) != (10, 0)  # sm_100a cubins: an H100 keeps the 12-launch step, no load attempt
            or not B16.inf3_bf16_on()
            or self.d != 768
            or self.da not in (768, 1024)
        ):
            return False
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return B16.inf3_bf16_ready(idx, self.h, self.da // self.h)

    def _pack3b(self):
        """The bf16 three-kernel step's weights, once per runner: the AdaLN table weights reordered to (s1, s2, sh1, sh2) per block
        (slices + cat, capture-safe), per block Wo / Wsq pair-packed per tail cluster size (Wsq in each CTA's z order), [Wv; Wg] and [Wa; Wb]
        as they are."""
        if self._p3b is None:
            from miniworld_engine.kernels.bias_only_dit.cuda import inf3_bf16 as B16

            nb, D, dc = self.nb, self.d, self.dc
            w4, b4 = self.w1.view(nb, 4, D, dc), self.b1.view(nb, 4, D)
            w1 = torch.cat([w4[:, 0:1], w4[:, 2:3], w4[:, 1:2], w4[:, 3:4]], 1).reshape(nb * 4 * D, dc).contiguous()
            b1 = torch.cat([b4[:, 0:1], b4[:, 2:3], b4[:, 1:2], b4[:, 3:4]], 1).reshape(nb * 4 * D).contiguous()
            blocks = [
                dict(
                    wvg=p["wvg"],
                    wab=p["wab"],
                    # per tail cluster size, made when a step first takes that size (the step takes one or two of the five)
                    wo=_Lazy(lambda cl, w=p["wo"]: B16.pack_pairs_bf16(w, cl)),
                    wsq=_Lazy(lambda cl, w=p["ws"]: B16.pack_pairs_bf16(w, cl, z_order=True)),
                    wsqn=_Lazy(lambda cl, w=p["ws"]: B16.pack_pairs_bf16(w, 8)),
                )
                for p in self.per
            ]  # the pair tail: natural k order
            self._p3b = (w1, b1, blocks)
        return self._p3b

    def _tables3b(self, cond, S, L):
        """The hoisted bf16 conditioning tables [T, nb, 6, 768] (gate1, gate2, s1, s2, sh1, sh2; the first four through their sigmoid,
        rounded to bf16 once): the 12-launch step's LayerNorm rows and two GEMMs, once per conditioning tensor (``_tables3``'s
        ``lookup_inputs`` rules)."""
        from miniworld_engine.kernels import _capture

        nb, D = self.nb, self.d
        shared = cond.shape[0] == 1 or cond.stride(0) == 0
        w1, b1, _ = self._pack3b()

        def build():
            c = (cond[0, 0] if shared else cond.reshape(S * L, self.dc)).to(self.dtype)
            T = c.shape[0]
            cn = torch.empty(T, self.dc, device=c.device, dtype=self.dtype)
            C.ln_rows(c, cn, EPS)
            if self._tab_one():
                return self._tab_b200(cn, c, w1, b1, T)
            g1 = torch.addmm(b1, cn, w1.t()).view(T, nb, 4, D)  # s1, s2, sh1, sh2
            g2 = torch.addmm(self.b2, c, self.w2.t()).view(T, nb, 2, D)  # gate1, gate2
            tab = torch.cat([g2, g1], 2)  # gate1, gate2, s1, s2, sh1, sh2
            tab[:, :, :4].sigmoid_()
            return tab

        key = (
            "tab3b",
            cond.data_ptr(),
            cond._version,
            tuple(cond.shape),
            cond.stride(),
            cond.dtype,
            cond.device,
            S,
            L,
        )
        entry = _capture.lookup_inputs(
            self._tab3,
            key,
            lambda: (weakref.ref(cond), build()),
            limit=16,
            valid=lambda e: e[0]() is cond,
            alive=lambda e: e[0]() is not None,
            inputs=(cond,),
        )
        return entry[1]

    def _ops3b(self, device):
        """(front, core, tail) of the bf16 three-kernel step: ``bo_front_bf16``, ``pv_gate_inf -DPDL_INF``, ``bo_tail_bf16`` /
        ``bo_tail2_bf16``."""
        from miniworld_engine.kernels.bias_only_dit.cuda import inf3_bf16 as B16

        idx = device.index if device.index is not None else torch.cuda.current_device()
        key = ("inf3b", idx)
        if key not in self._ops:
            # p_early: the core loads P before its PDL wait (-DP_EARLY). Invariant (_step3b): P is ``hoist``'s output (or the
            # integration's per-call softmax of a hoisted bias, ``_infer_bias``), written before the step by no kernel of its PDL chain, and the step's FIRST front is launched without the programmatic attribute
            # (``serial``) whenever the core takes P early -- so every kernel of the chain starts after all earlier stream work,
            # P's writer included, is complete and visible, also where hoist runs right before the step (a P-cache miss, or both
            # in one captured graph). Never for a P made inside a PDL chain (the training backward's P^T: PvGateCore's default).
            self._ops[key] = (
                B16.FrontBF16(idx, self.da),
                C.PvGateCore(idx, nh=self.h, dh=self.da // self.h, pdl=True, p_early=True),
                B16.TailBF16(idx, self.da),
            )
        return self._ops[key]

    def _buffers3b(self, S, L, dev):
        """v|g, a, the fp32 hand-over between blocks, and the clusters' exchange scratch (xa / xt [M, 768] bf16 rows padded by 128 B,
        h [M 24, 64] bf16 blocked k-block-major; xt / h padded to whole tile pairs for the pair tail)."""
        from miniworld_engine.kernels.bias_only_dit.cuda import inf3_bf16 as B16

        key = ("inf3b", S, L, dev)
        if key not in self._buf:
            M, bf, f32 = S * L, torch.bfloat16, torch.float32
            Mp = B16.tail_rows(M, B16.TAIL_PAIR)
            self._buf[key] = dict(
                xa=torch.empty(M, self.d + 64, device=dev, dtype=bf)[:, : self.d],
                vg=torch.empty(M, 2 * self.da, device=dev, dtype=bf),
                a=torch.empty(M, self.da, device=dev, dtype=bf),
                xn=[torch.empty(M, self.d, device=dev, dtype=f32) for _ in range(2 if self.nb > 1 else 0)],
                xt=torch.empty(Mp, self.d + 64, device=dev, dtype=bf)[:, : self.d],
                h=torch.empty(Mp * 2 * self.d // 64, 64, device=dev, dtype=bf),
            )
        return self._buf[key]

    def _step3b(self, single, cond, P, out_dtype=None, kmask=None):
        """The bf16 step as three kernels per block: front (LN + AdaLN + v|g GEMM) -> core -> tail (everything after the core); the
        block input the step's bf16 single, then the previous tail's fp32 output; the last tail writes the step's dtype."""
        from miniworld_engine.kernels.bias_only_dit.cuda import inf3_bf16 as B16

        S, B, L, D = single.shape
        assert B == 1 and L % 128 == 0 and D == self.d
        M, H, DA, dev = S * L, self.h, self.da, single.device
        tab = self._tables3b(cond, S, L)
        T = tab.shape[0]
        _, _, packs = self._pack3b()
        front_op, core, tail = self._ops3b(dev)
        tcl = tail.cluster(M // 128)  # 8, 6, or the pair tail where it takes fewer rounds
        pair, Mt = tcl == B16.TAIL_PAIR, B16.tail_rows(M, tcl)
        buf = self._buffers3b(S, L, dev)
        self._join(dev)
        self._join_hoist(dev)  # before the front: the core is its programmatic dependent launch and reads P
        xin = single.reshape(M, D)
        out = torch.empty(M, D, device=dev, dtype=out_dtype or single.dtype)
        xa, vg, a = buf["xa"], buf["vg"], buf["a"]
        early = core.early()  # the core reads P before its PDL wait: see _ops3b
        assert not early or P.shape == (self.nb * H, L, L), P.shape  # the whole hoisted P (hoist's layout), not a chain's own
        for b, p in enumerate(packs):
            y = out if b + 1 == self.nb else buf["xn"][b % 2]
            tb = tab[:, b]
            front_op(xin, tb, p["wvg"], vg, T, xa=xa, serial=early and b == 0)
            core(vg[:, :DA], P[b * H : (b + 1) * H].view(H * L, L), a, S, g=vg[:, DA:])  # sigmoid(g) * (P v)
            wo, wsq = (p["wo"][8], p["wsqn"][8]) if pair else (p["wo"][tcl], p["wsq"][tcl])
            tail(
                a,
                xin,
                tb,
                wo,
                p["wab"],
                wsq,
                y,
                T,
                xt=buf["xt"][:Mt],
                h=buf["h"][: Mt * 24],
                cl=tcl,
            )
            xin = y
        return out.view(S, 1, L, D)

    def _step(self, single, cond, P, out_dtype=None, kmask=None):
        S, B, L, D = single.shape
        assert B == 1 and L % 128 == 0 and D == self.d
        M, nb = S * L, self.nb
        dev = single.device
        shared = cond.shape[0] == 1 or cond.stride(0) == 0
        c = cond[0, 0] if shared else cond.reshape(M, self.dc)
        c = c.to(self.dtype)
        T = c.shape[0]
        cond_op = self._cond(dev)
        if cond_op is not None:  # one kernel; the sigmoid tables arrive as sigmoids
            wc, bc, csc = self._cond_pack()
            G = torch.empty(T, wc.shape[0], device=dev, dtype=self.dtype)
            cond_op(c, wc, bc, csc, G, self.n_g1, EPS)
            g1, g2 = (
                G[:, : self.n_g1].view(T, nb, 4, D),
                G[:, self.n_g1 :].view(T, nb, 2, D),
            )
        else:
            cn = torch.empty(T, self.dc, device=dev, dtype=self.dtype)
            C.ln_rows(c, cn, EPS)
            if _H100_COND1 and C.h100_enabled(dev) and not self.fp32 and T % 128 == 0 and T <= _H100_COND1_ROWS:
                # both table GEMMs as one sm_90a launch (bo_gemm_sm90.cu cond_tables; the weights prefetched before its PDL wait)
                G = torch.empty(T, self.wc.shape[0], device=dev, dtype=self.dtype)
                C.h100_gemm_ext().cond_tables(cn, c.contiguous(), self.wc, self.bc, G, self.n_g1, True, 0)
                g1, g2 = G[:, : self.n_g1].view(T, nb, 4, D), G[:, self.n_g1 :].view(T, nb, 2, D)
            else:
                g1 = torch.addmm(self.b1, cn, self.w1.t()).view(T, nb, 4, D)
                g2 = torch.addmm(self.b2, c, self.w2.t()).view(T, nb, 2, D)
        out = torch.empty(M, D, device=dev, dtype=out_dtype or single.dtype)
        self._run(single.reshape(M, D), g1, g2, P, out, S, L, presig=cond_op is not None, kmask=kmask)
        return out.view(S, 1, L, D)

    def _run(self, single, g1, g2, P, out, S, L, presig=False, kmask=None):
        """The blocks for S samples: single, out [S L, D]; g1, g2 the conditioning tables (row period T = their rows);
        ``presig``: their scale and gate tables already hold the sigmoids. ``kmask``: as ``step``'s (read by the H100 cores)."""
        H, T, DA = self.h, g1.shape[0], self.da
        dev = single.device
        buf = self._buffers(S, L, dev)
        x, xa, vg, a, y, ab, h = (buf[k] for k in ("x", "xa", "vg", "a", "y", "ab", "h"))
        core, gsw = self._core(dev), self._gemm_swiglu(dev, xa)
        resln, resln_final = (
            self._resln(dev, False, presig),
            self._resln(dev, True, presig),
        )
        ps = presig
        hopper = C.h100_enabled(dev)
        # Hopper: no fp32 copy of the input; the first residual pass reads it as it is (resgate_adaln_rows_in)
        nocopy = hopper and _H100_NOCOPY and resln is None
        C.adaln_in_rows(single, x[:0] if nocopy else x, g1[:, 0, 0], g1[:, 0, 1], xa, T, EPS, ps)
        sqro = hopper and _H100_SQ_RESOUT and resln is None and not ps and not self.fp32
        h100_tf32 = self.fp32 and hopper
        # the H100 TF32 core skips the key blocks past the last attended key (the bf16 pvx core reads no mask: its skip cost every unpadded call)
        km = {"kmask": kmask} if kmask is not None and h100_tf32 else {}
        for b, p in enumerate(self.per):
            if h100_tf32:
                # TF32 wgmma reads K-major operands only: v^T = Wv xa^T from cuBLAS, g = xa Wg^T beside it
                torch.mm(p["wvg"][:DA], xa.t(), out=buf["vt"])
                torch.mm(xa, p["wvg"][DA:].t(), out=buf["gg"])
                core(buf["vt"], P[b * H : (b + 1) * H].view(H * L, L), a, S, g=buf["gg"], **km)
            else:
                torch.mm(xa, p["wvg"].t(), out=vg)
                core(
                    vg[:, :DA],
                    P[b * H : (b + 1) * H].view(H * L, L),
                    a,
                    S,
                    g=vg[:, DA:],
                )  # sigmoid(g) * (P v)
            if resln is not None:  # out GEMM + residual + gate + AdaLN, one kernel
                resln(a, p["wo"], x, g2[:, b, 0], g1[:, b, 2], g1[:, b, 3], xa, T, EPS)
            else:
                torch.mm(a, p["wo"].t(), out=y)
                if nocopy and b == 0:
                    C.resgate_adaln_rows_in(single, x, y, g2[:, b, 0], g1[:, b, 2], g1[:, b, 3], xa, T, EPS, ps)
                else:
                    C.resgate_adaln_rows(x, y, g2[:, b, 0], g1[:, b, 2], g1[:, b, 3], xa, T, EPS, ps)
            if gsw is not None:
                gsw(xa, p["wab"], h)  # expand + SwiGLU, one sm_100a kernel
            else:
                torch.mm(xa, p["wab"].t(), out=ab)
                if self.fp32:
                    _tf32().rows32().swiglu_cuda(ab, h)
                else:
                    C.swiglu_rows(ab, h)
            last = b + 1 == self.nb
            if resln is not None:  # squeeze GEMM + residual + gate (+ AdaLN)
                if last:
                    resln_final(h, p["ws"], x, g2[:, b, 1], None, None, out, T, EPS)
                else:
                    resln(
                        h,
                        p["ws"],
                        x,
                        g2[:, b, 1],
                        g1[:, b + 1, 0],
                        g1[:, b + 1, 1],
                        xa,
                        T,
                        EPS,
                    )
            elif last and sqro:  # squeeze GEMM + the last residual + gate, one sm_90a kernel (bo_gemm_sm90.cu)
                C.h100_gemm_ext().sq_resout(h, p["ws"], x, g2[:, b, 1], out, False, True, 0)
            else:
                torch.mm(h, p["ws"].t(), out=y)
                if last:
                    C.resgate_out_rows(x, y, g2[:, b, 1], out, T, ps)
                else:
                    C.resgate_adaln_rows(
                        x,
                        y,
                        g2[:, b, 1],
                        g1[:, b + 1, 0],
                        g1[:, b + 1, 1],
                        xa,
                        T,
                        EPS,
                        ps,
                    )


#: Hopper 12-launch step: no fp32 copy of the step input (the first residual pass reads the input itself).
_H100_NOCOPY = True
#: Hopper 12-launch step: the expand GEMM as cuBLAS + the SwiGLU rows where gemm_swiglu_sm90 under-fills the SMs (see _gemm_swiglu).
_H100_EXPAND_ROWS = True
#: Hopper 12-launch step: the two conditioning-table GEMMs as one sm_90a launch (bo_gemm_sm90.cu cond_tables) instead of two cuBLAS, up to
#: _H100_COND1_ROWS table rows (2026-10-19, us per block: L128 A1 37.0 -> 35.4, L384 A1 48.6 -> 47.9; slower from 768 rows: L768 A1 72.0 -> 78.4)
_H100_COND1 = True
_H100_COND1_ROWS = 384
#: Hopper 12-launch step: the squeeze GEMM + the last residual as one kernel (bo_gemm_sm90.cu sq_resout) instead of cuBLAS + the rows;
#: off: 1.1-1.5x slower (its epilogue runs after the mainloop with the tensor core idle; 2026-10-19 record).
_H100_SQ_RESOUT = False


__all__ = ["FusedBiasOnlyDiT"]
