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

Tried and not kept: splitting the samples over two or three CUDA streams (each chain leaves SMs idle at these sizes, but the
core and cuBLAS's kernels each fill an SM's shared memory, so the chains did not overlap: L384 70 -> 80 us), and the
output-gate table GEMM on a side stream (no change).
"""

from __future__ import annotations

import os

import torch

from miniworld_engine.kernels.bias_only_dit import cuda as C

EPS = 1e-5


def _rows():
    """The token DiT's CUDA row kernels: the pair LayerNorm (C = 128) of the hoist."""
    from miniworld_engine.kernels.conditioned_transition import cuda as rows
    return rows


class FusedBiasOnlyDiT:
    def __init__(self, blocks, dtype=torch.bfloat16):
        blocks = list(blocks)
        a0 = blocks[0].attention
        self.nb = len(blocks)
        self.h = a0.n_head
        self.d = a0.to_out.weight.shape[0]                       # the single width
        self.da = a0.to_value.weight.shape[0]                    # the attention's: n_head x head width (768 or 1024)
        self.dc = a0.ada_ln_in.ln_cond.weight.shape[0]
        self.dp = a0.ln_pair.weight.shape[0]
        assert dtype is torch.bfloat16, "the core is bf16 only"
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

    # ------------------------------------------------------------------ once per sample()
    def hoist(self, pair, mask=None):
        """pair [1, L, L, dp] -> every block's attention weights P [nb H, L, L] bf16 (rows sum to 1 over the keys).
        ``mask`` [L] bool marks the real tokens; the other keys get zero weight."""
        L = pair.shape[1]
        z2d = pair.reshape(L * L, self.dp)
        zh = torch.empty(L * L, self.dp, device=pair.device, dtype=self.dtype)
        _rows()._ext().layernorm128_rows(z2d, zh, EPS)
        P = torch.empty(self.nb * self.h, L, L, device=pair.device, dtype=self.dtype)
        torch.mm(self.pw, zh.t(), out=P.view(self.nb * self.h, L * L))
        m = None if mask is None else mask.reshape(L).to(torch.bool).contiguous()
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
            self._ops[("core", idx)] = C.PvGateCore(idx, nh=self.h, dh=self.da // self.h)
        return self._ops[("core", idx)]

    def _cond(self, device):
        """The one-kernel conditioning tables (MINIWORLD_BIAS_ONLY_DIT_COND=1); off by default: slower than LayerNorm rows + two
        cuBLAS GEMMs at every measured shape (the sigmoids of the tables on 256 epilogue threads per SM dominate its time)."""
        if os.environ.get("MINIWORLD_BIAS_ONLY_DIT_COND", "0") == "0":
            return None
        idx = device.index if device.index is not None else torch.cuda.current_device()
        if ("cond", idx) not in self._ops:
            self._ops[("cond", idx)] = C.CondTables(idx)
        return self._ops[("cond", idx)]

    def _resln(self, device, final, presig):
        """The width-768 GEMM with residual + gate + AdaLN (or the final output) in its epilogue; None: cuBLAS + rows.
        Off by default until the conditioning tables carry their sigmoids (MINIWORLD_BIAS_ONLY_DIT_RESLN=1 turns it on)."""
        if os.environ.get("MINIWORLD_BIAS_ONLY_DIT_RESLN", "0") == "0":
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
