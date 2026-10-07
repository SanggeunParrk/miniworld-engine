"""Fused bias-only token DiT inference over a stack of ``modules.bias_only_dit.BiasOnlyDiTBlock``: CUDA and cuBLAS only.

    runner = FusedBiasOnlyDiT(blocks)          # packs every weight once
    P = runner.hoist(pair, mask)               # once per sample(): every block's attention weights, softmax(pair bias)
    single = runner.step(single, cond, P)      # once per solver step

What each call hoists, and why it may:

  hoist   The logits are the pair bias alone, and the pair carries no noise level: every block's attention weights are the
          same at every step and for every sample. One LayerNorm pass over the pair rows (the blocks share its statistics;
          each block's LayerNorm weight is folded into its projection), ONE cuBLAS GEMM for every block's bias, and a row
          softmax that writes P [nb H, L, L] bf16 in place of the bias.
  step    Per block: cuBLAS v|g GEMM -> ``pv_gate_inf`` (sigmoid(g) * P v, a tcgen05 GEMM with the gate in its epilogue) ->
          cuBLAS out GEMM -> residual + gate + AdaLN rows -> expand GEMM + SwiGLU (``gemm_swiglu2_sm100`` from 3840 rows;
          below it cuBLAS + the SwiGLU rows) -> cuBLAS squeeze GEMM -> residual + gate (+ the next block's AdaLN) rows. The
          row passes are this family's (``bias_only_dit_rows.cu``, one warp per row). The conditioning tables (AdaLN scale /
          shift, output gates) of every block come from two GEMMs over L rows when the samples share one conditioning, over
          S L rows when each has its own. The residual stream is fp32.

fp32 (``dtype=torch.float32``, the TF32 recipe): the same schedule with every activation, table and weight fp32. The hoist is
``pair_bias`` (LN(pair) Wf^T in exact fp32, ``bias_only_dit_f32_rows.cu``) and the fp32 softmax, P [nb H, L, L] fp32; the core is
``pv_gate_tf32`` (kind::tf32 MMAs, P streamed through shared memory); GEMMs are cuBLAS on TF32 tensor cores (``tf32_gemms``); the
expand GEMM is cuBLAS + the fp32 SwiGLU rows; the conditioning tables and the opt-in fused kernels (cond tables, GEMM + residual +
AdaLN) stay bf16-only -- the fp32 step runs the default composition.

fp32, three kernels per block (the default; MINIWORLD_BIAS_ONLY_DIT_INF3=0 keeps the composition above; read per call, so both
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

Tried and not kept: splitting the samples over two or three CUDA streams (each chain leaves SMs idle at these sizes, but the
core and cuBLAS's kernels each fill an SM's shared memory, so the chains did not overlap: L384 70 -> 80 us), and the
output-gate table GEMM on a side stream (no change).
"""

from __future__ import annotations

import os
import weakref

import torch

from miniworld_engine.kernels.bias_only_dit import cuda as C

EPS = 1e-5


def _tf32():
    """The fp32 path's kernels (``cuda/tf32.py``)."""
    from miniworld_engine.kernels.bias_only_dit.cuda import tf32
    return tf32


def _rows():
    """The token DiT's CUDA row kernels: the pair LayerNorm (C = 128) of the hoist."""
    from miniworld_engine.kernels.conditioned_transition import cuda as rows
    return rows


class FusedBiasOnlyDiT:
    def __init__(self, blocks, dtype=torch.bfloat16):
        assert dtype in (torch.bfloat16, torch.float32), "bf16, or fp32 on TF32 tensor cores"
        blocks = list(blocks)
        a0 = blocks[0].attention
        self.nb = len(blocks)
        self.h = a0.n_head
        self.d = a0.to_out.weight.shape[0]                       # the single width
        self.da = a0.to_value.weight.shape[0]                    # the attention's: n_head x head width (768 or 1024)
        self.dc = a0.ada_ln_in.ln_cond.weight.shape[0]
        self.dp = a0.ln_pair.weight.shape[0]
        self.fp32 = dtype is torch.float32
        dev = a0.to_value.weight.device
        f32 = lambda t: t.detach().float()
        w1, b1, w2, b2, pw = [], [], [], [], []
        self.per = []
        for blk in blocks:
            at, tr = blk.attention, blk.transition
            la, lt = f32(at.ada_ln_in.ln_cond.weight), f32(tr.ada_ln_in.ln_cond.weight)
            # AdaLN scale / shift of both halves on LayerNorm(cond), the norm weight folded in
            w1 += [f32(at.ada_ln_in.to_scale.weight) * la, f32(at.ada_ln_in.to_bias.weight) * la,
                   f32(tr.ada_ln_in.to_scale.weight) * lt, f32(tr.ada_ln_in.to_bias.weight) * lt]
            z = torch.zeros(self.d, device=dev)
            b1 += [f32(at.ada_ln_in.to_scale.bias), z, f32(tr.ada_ln_in.to_scale.bias), z]
            # the two output gates read the raw conditioning
            w2 += [f32(at.to_scale.weight), f32(tr.to_scale.weight)]
            b2 += [f32(at.to_scale.bias), f32(tr.to_scale.bias)]
            pw.append(f32(at.to_bias.weight) * f32(at.ln_pair.weight))            # [H, dp], ln_pair weight folded
            self.per.append(dict(
                wvg=torch.cat([at.to_value.weight, at.to_gate.weight], 0).detach().to(dtype).contiguous(),
                wo=at.to_out.weight.detach().to(dtype).contiguous(),
                wab=torch.cat([tr.expand_a.weight, tr.expand_b.weight], 0).detach().to(dtype).contiguous(),
                ws=tr.squeeze.weight.detach().to(dtype).contiguous()))
        self.w1 = torch.cat(w1, 0).to(dtype).contiguous()          # [nb 4 d, dc]
        self.b1 = torch.cat(b1, 0).to(dtype).contiguous()
        self.w2 = torch.cat(w2, 0).to(dtype).contiguous()          # [nb 2 d, dc]
        self.b2 = torch.cat(b2, 0).to(dtype).contiguous()
        self.pw = torch.cat(pw, 0).to(dtype).contiguous()          # [nb H, dp]
        # the one-kernel conditioning tables (cond_tables_sm100.cu): [W1; W2], their biases, and the column sums of the bf16 W1
        # rows (LN(c) W1^T = rstd (c W1^T - mu colsum(W1)))
        self.n_g1 = self.w1.shape[0]
        self.wc = torch.cat([self.w1, self.w2], 0).contiguous()
        self.bc = torch.cat([self.b1, self.b2], 0).float().contiguous()
        self.csc = torch.cat([self.w1.float().sum(1), torch.zeros(self.w2.shape[0], device=dev)], 0).contiguous()
        self.dtype = dtype
        self._buf: dict = {}
        self._ops: dict = {}
        self._tab3: dict = {}                                   # hoisted conditioning tables of the three-kernel step
        self._p3 = None                                         # its TF32-rounded weight pack

    # ------------------------------------------------------------------ once per sample()
    def hoist(self, pair, mask=None):
        """pair [1, L, L, dp] -> every block's attention weights P [nb H, L, L] in the runner's dtype (rows sum to 1 over the keys).
        ``mask`` [L] bool marks the real tokens; the other keys get zero weight."""
        L = pair.shape[1]
        z2d = pair.reshape(L * L, self.dp)
        m = None if mask is None else mask.reshape(L).to(torch.bool).contiguous()
        if self.fp32:
            # one pass of the pair per block: its LayerNorm and the H biases on the FMA pipe, exact fp32 (no LN(pair) in memory)
            R32 = _tf32().rows32()
            P = torch.empty(self.nb * self.h, L, L, device=pair.device, dtype=torch.float32)
            z32 = z2d.float().contiguous()
            for b in range(self.nb):
                R32.pair_bias_cuda(z32, self.pw[b * self.h:(b + 1) * self.h], P[b * self.h:(b + 1) * self.h], None, EPS)
            R32.softmax_rows_cuda(P.view(-1, L), P.view(-1, L), m)
            return P
        zh = torch.empty(L * L, self.dp, device=pair.device, dtype=self.dtype)
        _rows()._ext().layernorm128_rows(z2d, zh, EPS)
        P = torch.empty(self.nb * self.h, L, L, device=pair.device, dtype=self.dtype)
        torch.mm(self.pw, zh.t(), out=P.view(self.nb * self.h, L * L))
        C.softmax_rows(P.view(-1, L), P.view(-1, L), m)
        return P

    def _buffers(self, S, L, dev):
        key = (S, L, dev)
        if key not in self._buf:
            M, D = S * L, self.d
            e = lambda n, dt=self.dtype: torch.empty(M, n, device=dev, dtype=dt)
            self._buf[key] = dict(x=e(D, torch.float32), xa=e(D), vg=e(2 * self.da), a=e(self.da), y=e(D),
                                  ab=e(self.per[0]["wab"].shape[0]), h=e(self.per[0]["wab"].shape[0] // 2))
        return self._buf[key]

    def _core(self, device):
        idx = device.index if device.index is not None else torch.cuda.current_device()
        if ("core", idx) not in self._ops:
            self._ops[("core", idx)] = (_tf32().PvGateCoreTF32(idx, nh=self.h, dh=self.da // self.h) if self.fp32 else
                                        C.PvGateCore(idx, nh=self.h, dh=self.da // self.h))
        return self._ops[("core", idx)]

    def _cond(self, device):
        """The one-kernel conditioning tables (MINIWORLD_BIAS_ONLY_DIT_COND=1); off by default: slower than LayerNorm rows + two
        cuBLAS GEMMs at every measured shape (the sigmoids of the tables on 256 epilogue threads per SM dominate its time)."""
        if os.environ.get("MINIWORLD_BIAS_ONLY_DIT_COND", "0") == "0" or self.fp32:
            return None
        idx = device.index if device.index is not None else torch.cuda.current_device()
        if ("cond", idx) not in self._ops:
            self._ops[("cond", idx)] = C.CondTables(idx)
        return self._ops[("cond", idx)]

    def _resln(self, device, final, presig):
        """The width-768 GEMM with residual + gate + AdaLN (or the final output) in its epilogue; None: cuBLAS + rows.
        Off by default until the conditioning tables carry their sigmoids (MINIWORLD_BIAS_ONLY_DIT_RESLN=1 turns it on)."""
        if os.environ.get("MINIWORLD_BIAS_ONLY_DIT_RESLN", "0") == "0" or self.fp32:
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
        if key not in self._ops:
            from miniworld_engine.kernels.conditioned_transition.cuda import gemm_swiglu
            wab = self.per[0]["wab"]
            self._ops[key] = (gemm_swiglu.GemmSwiglu(idx, K=wab.shape[1], H=wab.shape[0] // 2)
                              if gemm_swiglu.supported(xa, wab) else None)
        return self._ops[key]

    # ------------------------------------------------------------------ once per solver step
    def step(self, single, cond, P, out_dtype=None):
        """One solver step: single [S, 1, L, D], cond [S or 1, 1, L, dc] (one conditioning shared by the samples when
        its sample axis is 1 or has stride 0), P from ``hoist``. Returns [S, 1, L, D] in ``out_dtype`` (single's)."""
        if not self.fp32:
            return self._step(single, cond, P, out_dtype)
        with _tf32().tf32_gemms():
            if self._inf3_ok(single.device):
                return self._step3(single, cond, P, out_dtype)
            return self._step(single, cond, P, out_dtype)

    # ------------------------------------------------------------------ fp32, three kernels per block (default; INF3=0: off)
    def _inf3_ok(self, device) -> bool:
        T = _tf32()
        if not (self.fp32 and T.inf3_on() and self.d == 768 and self.da in (768, 1024)):
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
                blocks.append(dict(wvg=rnd(p["wvg"]), wab=rnd(p["wab"]), wo={cl: pairs(wo, cl) for cl in tcl},
                                   wsq={cl: pairs(ws, cl) for cl in tcl}))
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
            g1 = torch.addmm(b1, cn, w1.t()).view(T, nb, 4, D)          # s1, s2, sh1, sh2
            g2 = torch.addmm(self.b2, c, self.w2.t()).view(T, nb, 2, D)  # gate1, gate2
            tab = torch.cat([g2, g1], 2)                                 # gate1, gate2, s1, s2, sh1, sh2
            tab[:, :, :4].sigmoid_()
            return tab

        key = ("tab3", cond.data_ptr(), cond._version, tuple(cond.shape), cond.stride(), cond.dtype, cond.device, S, L)
        entry = _capture.lookup_inputs(self._tab3, key, lambda: (weakref.ref(cond), build()), limit=16,
                                       valid=lambda e: e[0]() is cond, alive=lambda e: e[0]() is not None)
        return entry[1]

    def _ops3(self, device):
        """(front, core, tail) of the three-kernel step."""
        T = _tf32()
        idx = device.index if device.index is not None else torch.cuda.current_device()
        key = ("inf3", idx)
        if key not in self._ops:
            self._ops[key] = (T.FrontTF32(idx, self.da), T.PvGateCoreTF32(idx, nh=self.h, dh=self.da // self.h, pdl=True),
                              T.TailTF32(idx, self.da))
        return self._ops[key]

    def _buffers3(self, S, L, dev):
        """v|g, a, the residual ping-pong of a multi-block stack, and the clusters' operand-exchange scratch (xa, xt [M, 768] rows, h
        [M 48, 32] blocked k-block-major: written and read within one kernel, L2-resident)."""
        key = ("inf3", S, L, dev)
        if key not in self._buf:
            M, f = S * L, torch.float32
            e = lambda n: torch.empty(M, n, device=dev, dtype=f)                                 # noqa: E731
            # the scratch rows padded by 128 B: their natural strides (3 / 6 KB) are multiples of 2 KB, so a TMA box's 128 rows
            # would fall on the same address bits above the line (round-4 experiment against L2 slice camping; harmless otherwise)
            pad = lambda n: torch.empty(M, n + 32, device=dev, dtype=f)[:, :n]                   # noqa: E731
            self._buf[key] = dict(vg=e(2 * self.da), a=e(self.da), x=[e(self.d) for _ in range(2 if self.nb > 1 else 0)],
                                  xa=pad(self.d), xt=pad(self.d), h=torch.empty(M * 2 * self.d // 32, 32, device=dev, dtype=f))
        return self._buf[key]

    def _step3(self, single, cond, P, out_dtype=None):
        """The fp32 step as three kernels per block: front (LN + AdaLN + v|g GEMM) -> core -> tail (everything after the core)."""
        S, B, L, D = single.shape
        assert B == 1 and L % 128 == 0 and D == self.d
        M, H, DA, dev = S * L, self.h, self.da, single.device
        tab = self._tables3(cond, S, L)
        T = tab.shape[0]
        _, _, packs = self._pack3()
        front, core, tail = self._ops3(dev)
        tcl = tail.cluster(M // 128)                                  # 8, or 6 where it fits the tiles in fewer rounds
        buf = self._buffers3(S, L, dev)
        x = single.reshape(M, D).float().contiguous()
        out = torch.empty(M, D, device=dev, dtype=torch.float32)
        vg, a = buf["vg"], buf["a"]
        for b, p in enumerate(packs):
            y = out if b + 1 == self.nb else buf["x"][b % 2]
            tb = tab[:, b]
            front(x, tb, p["wvg"], vg, T, xa=buf["xa"])
            core(vg[:, :DA], P[b * H:(b + 1) * H].view(H * L, L), a, S, g=vg[:, DA:])     # sigmoid(g) * (P v), TF32-rounded
            tail(a, x, tb, p["wo"][tcl], p["wab"], p["wsq"][tcl], y, T, xt=buf["xt"], h=buf["h"], cl=tcl)
            x = y
        res = out.view(S, 1, L, D)
        return res if out_dtype in (None, torch.float32) else res.to(out_dtype)

    def _step(self, single, cond, P, out_dtype=None):
        S, B, L, D = single.shape
        assert B == 1 and L % 128 == 0 and D == self.d
        M, nb = S * L, self.nb
        dev = single.device
        shared = cond.shape[0] == 1 or cond.stride(0) == 0
        c = cond[0, 0] if shared else cond.reshape(M, self.dc)
        c = c.to(self.dtype)
        T = c.shape[0]
        cond_op = self._cond(dev)
        if cond_op is not None:                                  # one kernel; the sigmoid tables arrive as sigmoids
            G = torch.empty(T, self.wc.shape[0], device=dev, dtype=self.dtype)
            cond_op(c, self.wc, self.bc, self.csc, G, self.n_g1, EPS)
            g1, g2 = G[:, :self.n_g1].view(T, nb, 4, D), G[:, self.n_g1:].view(T, nb, 2, D)
        else:
            cn = torch.empty(T, self.dc, device=dev, dtype=self.dtype)
            C.ln_rows(c, cn, EPS)
            g1 = torch.addmm(self.b1, cn, self.w1.t()).view(T, nb, 4, D)
            g2 = torch.addmm(self.b2, c, self.w2.t()).view(T, nb, 2, D)
        out = torch.empty(M, D, device=dev, dtype=out_dtype or single.dtype)
        self._run(single.reshape(M, D), g1, g2, P, out, S, L, presig=cond_op is not None)
        return out.view(S, 1, L, D)

    def _run(self, single, g1, g2, P, out, S, L, presig=False):
        """The blocks for S samples: single, out [S L, D]; g1, g2 the conditioning tables (row period T = their rows);
        ``presig``: their scale and gate tables already hold the sigmoids."""
        H, T, DA = self.h, g1.shape[0], self.da
        dev = single.device
        buf = self._buffers(S, L, dev)
        x, xa, vg, a, y, ab, h = (buf[k] for k in ("x", "xa", "vg", "a", "y", "ab", "h"))
        core, gsw = self._core(dev), self._gemm_swiglu(dev, xa)
        resln, resln_final = self._resln(dev, False, presig), self._resln(dev, True, presig)
        ps = presig
        C.adaln_in_rows(single, x, g1[:, 0, 0], g1[:, 0, 1], xa, T, EPS, ps)
        for b, p in enumerate(self.per):
            torch.mm(xa, p["wvg"].t(), out=vg)
            core(vg[:, :DA], P[b * H:(b + 1) * H].view(H * L, L), a, S, g=vg[:, DA:])   # sigmoid(g) * (P v)
            if resln is not None:                                         # out GEMM + residual + gate + AdaLN, one kernel
                resln(a, p["wo"], x, g2[:, b, 0], g1[:, b, 2], g1[:, b, 3], xa, T, EPS)
            else:
                torch.mm(a, p["wo"].t(), out=y)
                C.resgate_adaln_rows(x, y, g2[:, b, 0], g1[:, b, 2], g1[:, b, 3], xa, T, EPS, ps)
            if gsw is not None:
                gsw(xa, p["wab"], h)                                      # expand + SwiGLU, one sm_100a kernel
            else:
                torch.mm(xa, p["wab"].t(), out=ab)
                if self.fp32:
                    _tf32().rows32().swiglu_cuda(ab, h)
                else:
                    C.swiglu_rows(ab, h)
            last = b + 1 == self.nb
            if resln is not None:                                         # squeeze GEMM + residual + gate (+ AdaLN)
                if last:
                    resln_final(h, p["ws"], x, g2[:, b, 1], None, None, out, T, EPS)
                else:
                    resln(h, p["ws"], x, g2[:, b, 1], g1[:, b + 1, 0], g1[:, b + 1, 1], xa, T, EPS)
            else:
                torch.mm(h, p["ws"].t(), out=y)
                if last:
                    C.resgate_out_rows(x, y, g2[:, b, 1], out, T, ps)
                else:
                    C.resgate_adaln_rows(x, y, g2[:, b, 1], g1[:, b + 1, 0], g1[:, b + 1, 1], xa, T, EPS, ps)


__all__ = ["FusedBiasOnlyDiT"]
