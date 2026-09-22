"""Fused token DiT inference over a stack of engine ``modules.dit.DiTBlock`` (AF3 Alg. 23), no QK-norm, no mask.

    runner = FusedTokenDiT(blocks)            # packs every weight once
    bias = runner.hoist(pair)                 # once per sample(): every block's pair bias, head-major
    single = runner.step(single, cond, bias)  # once per solver step

What each call hoists, and why it may:

  hoist   The pair representation carries no noise level, so every block's pair bias is the same at every
          step. All 24 blocks share the LayerNorm statistics of the pair rows, so with each block's LayerNorm
          weight folded into its projection they are ONE GEMM over one pass of the pair.
  step    The single conditioning carries the noise level and changes every step, but at a step every sample
          sees the same one: it is computed for L token rows, not S * L, and for all 24 blocks in two GEMMs
          (LayerNorm(cond) statistics are shared too; each block's cond-LayerNorm weight is folded).
"""
import torch
import torch.nn.functional as F

from . import kernels as K
from .attn import attention_gated_in_place


def _t(w):
    return w.detach().t().contiguous()


def attention_in_place(q, k, v, bias, mask, m, key):
    """The engine's forward attention kernel, launched with the output written over q.

    ``_attn_fwd`` computes one base offset from q's strides and applies it to k, v and the output too, so all four
    must share strides. Here q, k, v are strided views of one [M, 4D] GEMM output; the engine launcher allocates a
    contiguous output, which does not share them and is written out of bounds. Writing over q does: a program reads
    its own q tile before it writes that same tile, and no other program reads it.
    """
    import triton
    from miniworld_engine.autotune.shape_key import pack
    from miniworld_engine.kernels.augmented_attention.triton.main import _attn_fwd
    A, B, L, H, D = q.shape
    qf, kf, vf = (t.view(A * B, L, H, D) for t in (q, k, v))
    grid = lambda META: (triton.cdiv(L, META["BLOCK_M1"]), A * B * H, triton.cdiv(D, META["HEAD_DIM_PAD"]))
    _attn_fwd[grid](qf, kf, vf, bias, mask, D ** -0.5, m, qf, *qf.stride(), *kf.stride(), *vf.stride(), *qf.stride(),
                    *bias.stride(), *mask.stride()[:2], A, B, H, L, D,
                    HEAD_DIM_PAD=max(16, triton.next_power_of_2(D)), shape_key=pack(key, H=H, HEAD_DIM=D))
    return q


class FusedTokenDiT:
    def __init__(self, blocks, dtype=torch.bfloat16, core="gated", prescale=True, core_precision="tf32"):
        """``dtype`` is the activation / weight dtype of the whole path: bf16, or fp32 (MiniWorld's v1 diffusion recipe).
        The residual stream is fp32 either way. fp32 GEMMs follow ``torch.backends.cuda.matmul.allow_tf32`` -- the caller's
        policy, as for any torch matmul -- and ``core_precision`` sets the attention core's MMA precision for fp32."""
        self.core_precision = core_precision
        self.prescale = prescale     # fold sm_scale*log2(e) into Wq,bq and log2(e) into the pair-bias weights (gated core only)
        self.core = core            # "gated": tdit.attn (sample-fastest grid, gate epilogue); "engine": the engine kernel + gate pass
        blocks = list(blocks)
        a0 = blocks[0].attention
        assert not a0.use_qk_norm, "QK-norm is not implemented on this path"
        self.nb = len(blocks)
        self.h = a0.n_head
        self.d = a0.to_query.weight.shape[0]
        self.dc = a0.ada_ln_in.ln_cond.weight.shape[0]
        self.dp = a0.ln_pair.weight.shape[0]
        self.eps = 1e-5
        dev = a0.to_query.weight.device
        f32 = lambda t: t.detach().float()
        w1, b1, w2, b2, pw = [], [], [], [], []
        self.per = []
        for blk in blocks:
            at, tr = blk.attention, blk.transition
            la, lt = f32(at.ada_ln_in.ln_cond.weight), f32(tr.ada_ln_in.ln_cond.weight)
            # AdaLN scale / shift of both halves on LayerNorm(cond) with the norm weight folded in
            w1 += [f32(at.ada_ln_in.to_scale.weight) * la, f32(at.ada_ln_in.to_bias.weight) * la,
                   f32(tr.ada_ln_in.to_scale.weight) * lt, f32(tr.ada_ln_in.to_bias.weight) * lt]
            z = torch.zeros(self.d, device=dev)
            b1 += [f32(at.ada_ln_in.to_scale.bias), z, f32(tr.ada_ln_in.to_scale.bias), z]
            # the two output gates read the raw conditioning
            w2 += [f32(at.to_scale.weight), f32(tr.to_scale.weight)]
            b2 += [f32(at.to_scale.bias), f32(tr.to_scale.bias)]
            pw.append(f32(at.to_bias.weight) * f32(at.ln_pair.weight))            # [H, dp], ln_pair weight folded
            wqkvg = torch.cat([at.to_query.weight, at.to_key.weight, at.to_value.weight, at.to_gate.weight], 0)
            bqkvg = torch.cat([at.to_query.bias.detach(), torch.zeros(3 * self.d, device=dev, dtype=at.to_query.bias.dtype)])
            if prescale and core == "gated":
                qs = (self.d // self.h) ** -0.5 * 1.4426950408889634          # sm_scale * log2(e)
                wqkvg = wqkvg.detach().float().clone(); bqkvg = bqkvg.float().clone()
                wqkvg[: self.d] *= qs
                bqkvg[: self.d] *= qs
            self.per.append(dict(
                wqkvg_t=_t(wqkvg).to(dtype), bqkvg=bqkvg.to(dtype).contiguous(),
                wo_t=_t(at.to_out.weight).to(dtype),
                wa_t=_t(tr.expand_a.weight).to(dtype), wb_t=_t(tr.expand_b.weight).to(dtype),
                ws_t=_t(tr.squeeze.weight).to(dtype),
                # v2 operands, nn.Linear layout [out, in] for torch.addmm(b, x, W.t())
                wqkvg=wqkvg.detach().to(dtype).contiguous(), wo=at.to_out.weight.detach().to(dtype).contiguous(),
                wab=torch.cat([tr.expand_a.weight, tr.expand_b.weight], 0).detach().to(dtype).contiguous(),
                ws=tr.squeeze.weight.detach().to(dtype).contiguous()))
        self.w1 = torch.cat(w1, 0).to(dtype).contiguous()          # [nb*4*d, dc]
        self.b1 = torch.cat(b1, 0).to(dtype).contiguous()
        self.w2 = torch.cat(w2, 0).to(dtype).contiguous()          # [nb*2*d, dc]
        self.b2 = torch.cat(b2, 0).to(dtype).contiguous()
        pwc = torch.cat(pw, 0)
        if prescale and core == "gated":
            pwc = pwc * 1.4426950408889634                                  # log2(e): the core works in the exp2 domain
        self.pw_t = pwc.t().to(dtype).contiguous()                          # [dp, nb*H]
        self.dtype = dtype
        self._buf = {}

    # ------------------------------------------------------------------ once per sample()
    def hoist(self, pair):
        """pair [1, L, L, dp] -> every block's bias, [nb*H, L, L] head-major."""
        L = pair.shape[1]
        out = torch.empty(self.nb * self.h, L, L, device=pair.device, dtype=self.dtype)
        K.pair_bias_all(pair.reshape(L * L, self.dp), self.pw_t, out, L, self.eps)
        return out

    def _buffers(self, S, L, dev):
        key = (S, L, dev)
        if key not in self._buf:
            M = S * L
            T = self.d // K.STAT_W
            self._buf[key] = dict(
                x=torch.empty(M, self.d, device=dev, dtype=torch.float32),
                mean=torch.empty(M, T, device=dev), m2=torch.empty(M, T, device=dev),
                qkvg=torch.empty(4, M, self.d, device=dev, dtype=self.dtype),
                h=torch.empty(M, self.per[0]["wa_t"].shape[1], device=dev, dtype=self.dtype),
                # the attention core allocates and fills an all-true mask per call when handed None
                keep=torch.ones(S, 1, L, device=dev, dtype=torch.bool),
                lse=torch.empty(S, 1, self.h, L, device=dev, dtype=torch.float32),
                # v2
                xa=torch.empty(M, self.d, device=dev, dtype=self.dtype),
                qkvg2=torch.empty(M, 4 * self.d, device=dev, dtype=self.dtype),
                a=torch.empty(M, self.d, device=dev, dtype=self.dtype),
                y=torch.empty(M, self.d, device=dev, dtype=self.dtype),
                ab=torch.empty(M, 2 * self.per[0]["wa_t"].shape[1], device=dev, dtype=self.dtype))
        return self._buf[key]

    # ------------------------------------------------------------------ once per solver step
    def step_v1(self, single, cond, bias, out_dtype=None):
        """v1: AdaLN in the GEMM prologue (Triton). Kept as the measured negative result; see kernels.py v2 note."""
        assert self.dtype == torch.bfloat16, "v1 kernels are bf16-only"
        S, B, L, D = single.shape
        assert B == 1
        M = S * L
        buf = self._buffers(S, L, single.device)
        x, mean, m2, qkvg, h = buf["x"], buf["mean"], buf["m2"], buf["qkvg"], buf["h"]
        c = cond[0, 0]                                                 # [L, dc]: shared by every sample at a step
        cn = F.layer_norm(c.float(), (self.dc,), eps=self.eps).to(self.dtype)
        g1 = torch.addmm(self.b1, cn, self.w1.t()).view(L, self.nb, 4, D)
        g1[:, :, 0::2].sigmoid_()                                      # AdaLN scales of both halves
        g2 = torch.addmm(self.b2, c.to(self.dtype), self.w2.t()).sigmoid_().view(L, self.nb, 2, D)
        x.copy_(single.reshape(M, D))
        K.row_stats(x, mean, m2)
        from miniworld_engine.kernels.augmented_attention import triton_augmented_attention_pair_bias as attn
        q, k, v, g = (qkvg[i].view(S, 1, L, self.h, D // self.h) for i in range(4))
        for b, p in enumerate(self.per):
            K.adaln_qkvg(x, mean, m2, g1[:, b, 0], g1[:, b, 1], p["wqkvg_t"], p["bqkvg"], qkvg, L, self.eps)
            bb = bias[b * self.h:(b + 1) * self.h].permute(1, 2, 0).unsqueeze(0)   # [1,L,L,H] view of head-major
            o = attn(q, k, v, bb, buf["keep"])
            K.gate_resgate(o.view(M, D), qkvg[3], p["wo_t"], g2[:, b, 0], x, mean, m2, L)
            K.adaln_swiglu(x, mean, m2, g1[:, b, 2], g1[:, b, 3], p["wa_t"], p["wb_t"], h, L, self.eps)
            K.gate_resgate(h, None, p["ws_t"], g2[:, b, 1], x, mean, m2, L)
        return x.view(S, 1, L, D).to(out_dtype or single.dtype)

    def _cond(self, cond, L, D):
        """Raw logits: the v2 row kernels apply the sigmoid as they load, where it costs nothing (they are
        memory-bound); a separate in-place pass over the strided scale columns cost 5.3 us a block."""
        c = cond[0, 0]                                                 # [L, dc]: shared by every sample at a step
        cn = F.layer_norm(c.float(), (self.dc,), eps=self.eps).to(self.dtype)
        g1 = torch.addmm(self.b1, cn, self.w1.t()).view(L, self.nb, 4, D)
        g2 = torch.addmm(self.b2, c.to(self.dtype), self.w2.t()).view(L, self.nb, 2, D)
        return g1, g2

    def step(self, single, cond, bias, out_dtype=None):
        """v2: four cuBLAS GEMMs, the attention core, and four row kernels per block."""
        from miniworld_engine.autotune.shape_key import atom_key
        S, B, L, D = single.shape
        assert B == 1
        M, H = S * L, self.h
        buf = self._buffers(S, L, single.device)
        x, xa, qkvg, a, y, ab, h = (buf[k] for k in ("x", "xa", "qkvg2", "a", "y", "ab", "h"))
        g1, g2 = self._cond(cond, L, D)
        x.copy_(single.reshape(M, D))
        K.adaln_rows(x, g1[:, 0, 0], g1[:, 0, 1], xa, L, self.eps)
        # q, k, v, g as strided views of ONE [M, 4D] GEMM output; the core launcher honours strides
        q, k, v = (qkvg.view(S, 1, L, 4 * D)[..., i * D:(i + 1) * D].unflatten(-1, (H, D // H)) for i in range(3))
        key = atom_key(L)
        q4, k4, v4, g4 = (qkvg.view(S, L, 4 * D)[..., i * D:(i + 1) * D].unflatten(-1, (H, D // H)) for i in range(4))
        keep2 = buf["keep"].view(S, L)
        for b, p in enumerate(self.per):
            torch.addmm(p["bqkvg"], xa, p["wqkvg"].t(), out=qkvg)
            if self.core == "gated":
                attention_gated_in_place(q4, k4, v4, g4, bias[b * H:(b + 1) * H], keep2, self.prescale, self.core_precision)   # sigmoid(g)*o over q
                torch.mm(qkvg[:, :D], p["wo"].t(), out=y)
            else:
                attention_in_place(q, k, v, bias[b * H:(b + 1) * H].unsqueeze(0), buf["keep"], buf["lse"], key)
                K.gate_rows(qkvg[:, :D], qkvg[:, 3 * D:], a)            # o sits where q was
                torch.mm(a, p["wo"].t(), out=y)
            K.resgate_adaln_rows(x, y, g2[:, b, 0], g1[:, b, 2], g1[:, b, 3], xa, L, self.eps)
            torch.mm(xa, p["wab"].t(), out=ab)
            K.swiglu_rows(ab, h)
            torch.mm(h, p["ws"].t(), out=y)
            last = b + 1 == self.nb
            K.resgate_adaln_rows(x, y, g2[:, b, 1], None if last else g1[:, b + 1, 0],
                                 None if last else g1[:, b + 1, 1], xa, L, self.eps)
        return x.view(S, 1, L, D).to(out_dtype or single.dtype)
