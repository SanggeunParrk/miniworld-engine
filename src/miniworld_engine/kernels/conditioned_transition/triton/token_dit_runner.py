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

from miniworld_engine.kernels.conditioned_transition.triton import token_dit_kernels as K
from miniworld_engine.kernels.conditioned_transition.triton.token_dit_attn import attention_gated_in_place, attention_gated_in_place2, bias_descriptor


def _row_kernels(device):
    """The row passes between the GEMMs: CUDA on B200 (``kernels/conditioned_transition/cuda``), Triton elsewhere or
    when the extension does not build. MINIWORLD_TOKEN_DIT_ROWS_CUDA=0 keeps Triton on B200 too."""
    import os
    if (device.type == "cuda" and torch.cuda.get_device_capability(device) == (10, 0)
            and os.environ.get("MINIWORLD_TOKEN_DIT_ROWS_CUDA", "1") != "0"):
        try:
            from miniworld_engine.kernels.conditioned_transition import cuda as cuda_rows
            cuda_rows.available()
            return cuda_rows
        except Exception as exc:  # noqa: BLE001 -- a failed build keeps the Triton rows
            import warnings
            warnings.warn(f"token DiT CUDA row kernels unavailable, keeping Triton: {exc!r}", RuntimeWarning, stacklevel=2)
    return K


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
    def __init__(self, blocks, dtype=torch.bfloat16, core="gated2", prescale=True, core_precision="tf32"):
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
        self.K = _row_kernels(dev)
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
            if prescale and core in ("gated", "gated2"):
                qs = (self.d // self.h) ** -0.5 * 1.4426950408889634          # sm_scale * log2(e)
                wqkvg = wqkvg.detach().float().clone(); bqkvg = bqkvg.float().clone()
                wqkvg[: self.d] *= qs
                bqkvg[: self.d] *= qs
            # (the v1 path's transposed copies -- wqkvg_t, wo_t, wa_t, wb_t, ws_t -- were read by nothing but two buffer
            # shapes and cost five weight-sized copies on every pack; the research archive keeps v1)
            self.per.append(dict(
                bqkvg=bqkvg.to(dtype).contiguous(),
                # v2 operands, nn.Linear layout [out, in] for torch.addmm(b, x, W.t())
                wqkvg=wqkvg.detach().to(dtype).contiguous(), wo=at.to_out.weight.detach().to(dtype).contiguous(),
                wab=torch.cat([tr.expand_a.weight, tr.expand_b.weight], 0).detach().to(dtype).contiguous(),
                ws=tr.squeeze.weight.detach().to(dtype).contiguous()))
        self.w1 = torch.cat(w1, 0).to(dtype).contiguous()          # [nb*4*d, dc]
        self.b1 = torch.cat(b1, 0).to(dtype).contiguous()
        self.w2 = torch.cat(w2, 0).to(dtype).contiguous()          # [nb*2*d, dc]
        self.b2 = torch.cat(b2, 0).to(dtype).contiguous()
        pwc = torch.cat(pw, 0)
        if prescale and core in ("gated", "gated2"):
            pwc = pwc * 1.4426950408889634                                  # log2(e): the core works in the exp2 domain
        self.pw_t = pwc.t().to(dtype).contiguous()                          # [dp, nb*H]
        self.dtype = dtype
        self._buf = {}
        self.streams = 1
        self._streams = []

    # ------------------------------------------------------------------ once per sample()
    def hoist(self, pair, mask=None):
        """pair [1, L, L, dp] -> every block's bias, [nb*H, L, L] head-major. ``mask`` [L] bool marks the real tokens;
        padded keys get -inf here, once per sample, so the attention core never touches a mask."""
        L = pair.shape[1]
        out = torch.empty(self.nb * self.h, L, L, device=pair.device, dtype=self.dtype)
        if getattr(self.K, "PAIR_BIAS_MASK", False):              # the CUDA rows fold the mask into the same pass
            self.K.pair_bias_all(pair.reshape(L * L, self.dp), self.pw_t, out, L, self.eps, mask=mask)
            return out
        self.K.pair_bias_all(pair.reshape(L * L, self.dp), self.pw_t, out, L, self.eps)
        if mask is not None:
            out[:, :, ~mask.reshape(L)] = float("-inf")
        return out

    def _buffers(self, S, L, dev, tag=0):
        key = (S, L, dev, tag)
        if key not in self._buf:
            M = S * L
            T = self.d // self.K.STAT_W
            self._buf[key] = dict(
                x=torch.empty(M, self.d, device=dev, dtype=torch.float32),
                mean=torch.empty(M, T, device=dev), m2=torch.empty(M, T, device=dev),
                qkvg=torch.empty(4, M, self.d, device=dev, dtype=self.dtype),
                h=torch.empty(M, self.per[0]["wab"].shape[0] // 2, device=dev, dtype=self.dtype),
                # the attention core allocates and fills an all-true mask per call when handed None
                keep=torch.ones(S, 1, L, device=dev, dtype=torch.bool),
                lse=torch.empty(S, 1, self.h, L, device=dev, dtype=torch.float32),
                # v2
                xa=torch.empty(M, self.d, device=dev, dtype=self.dtype),
                qkvg2=torch.empty(M, 4 * self.d, device=dev, dtype=self.dtype),
                a=torch.empty(M, self.d, device=dev, dtype=self.dtype),
                y=torch.empty(M, self.d, device=dev, dtype=self.dtype),
                ab=torch.empty(M, self.per[0]["wab"].shape[0], device=dev, dtype=self.dtype))
        return self._buf[key]

    # ------------------------------------------------------------------ once per solver step
    def _cond(self, cond, L, D):
        """Raw logits: the v2 row kernels apply the sigmoid as they load, where it costs nothing (they are
        memory-bound); a separate in-place pass over the strided scale columns cost 5.3 us a block."""
        c = cond[0, 0]                                                 # [L, dc]: shared by every sample at a step
        cn = F.layer_norm(c.float(), (self.dc,), eps=self.eps).to(self.dtype)
        g1 = torch.addmm(self.b1, cn, self.w1.t()).view(L, self.nb, 4, D)
        g2 = torch.addmm(self.b2, c.to(self.dtype), self.w2.t()).view(L, self.nb, 2, D)
        return g1, g2

    def step(self, single, cond, bias, out_dtype=None, streams=None):
        """One solver step. ``streams`` > 1 splits the samples into that many groups, each on its own CUDA stream: the
        samples share only the conditioning and the pair bias, so one group's memory-bound passes can run under another
        group's GEMMs and attention."""
        S, B, L, D = single.shape
        assert B == 1
        streams = min(streams or self.streams, S)
        g1, g2 = self._cond(cond, L, D)
        if streams <= 1:
            # copy=True: the fp32 residual is this runner's buffer; returning it uncopied (fp32 out) let the next step on a
            # reused runner overwrite an earlier result
            return self._run(single, g1, g2, bias, 0).view(S, 1, L, D).to(out_dtype or single.dtype, copy=True)
        main = torch.cuda.current_stream()
        if len(self._streams) < streams:
            self._streams += [torch.cuda.Stream() for _ in range(streams - len(self._streams))]
        ready = torch.cuda.Event()
        ready.record(main)
        cuts = [round(i * S / streams) for i in range(streams + 1)]
        outs = []
        for i in range(streams):
            st = self._streams[i]
            st.wait_event(ready)
            with torch.cuda.stream(st):
                outs.append(self._run(single[cuts[i]:cuts[i + 1]], g1, g2, bias, i + 1))
            done = torch.cuda.Event()
            done.record(st)
            main.wait_event(done)
        return torch.cat([o.view(-1, 1, L, D) for o in outs], 0).to(out_dtype or single.dtype)

    def _run(self, single, g1, g2, bias, tag):
        """The blocks for one group of samples; returns the fp32 residual, [S*L, D]."""
        from miniworld_engine.autotune.shape_key import atom_key
        S, B, L, D = single.shape
        M, H = S * L, self.h
        buf = self._buffers(S, L, single.device, tag)
        x, xa, qkvg, a, y, ab, h = (buf[k] for k in ("x", "xa", "qkvg2", "a", "y", "ab", "h"))
        x.copy_(single.reshape(M, D))
        self.K.adaln_rows(x, g1[:, 0, 0], g1[:, 0, 1], xa, L, self.eps)
        # q, k, v, g as strided views of ONE [M, 4D] GEMM output; the core launcher honours strides
        q, k, v = (qkvg.view(S, 1, L, 4 * D)[..., i * D:(i + 1) * D].unflatten(-1, (H, D // H)) for i in range(3))
        key = atom_key(L)
        q4, k4, v4, g4 = (qkvg.view(S, L, 4 * D)[..., i * D:(i + 1) * D].unflatten(-1, (H, D // H)) for i in range(4))
        keep2 = buf["keep"].view(S, L)
        sm100 = self.core == "gated2" and self.prescale and self._sm100_core(single.device, L, D, H)
        gsw = self._gemm_swiglu(single.device, xa)
        if self.core == "gated2" and not sm100:
            assert self.prescale, "the v2 core expects pre-scaled logits"
            key_d = (bias.data_ptr(), tuple(bias.shape))
            if getattr(self, "_bdesc_key", None) != key_d:
                self._bdesc, self._bdesc_key = bias_descriptor(bias), key_d
            bdesc = self._bdesc
        for b, p in enumerate(self.per):
            self._mm(xa, p["wqkvg"], qkvg, p["bqkvg"])
            if sm100:
                sm100(qkvg, bias, b, S)                                   # sigmoid(g)*o over q, one sm_100a kernel
                self._mm(qkvg[:, :D], p["wo"], y)
            elif self.core == "gated2":
                attention_gated_in_place2(q4, k4, v4, g4, bdesc, b, self.core_precision)  # sigmoid(g)*o over q
                self._mm(qkvg[:, :D], p["wo"], y)
            elif self.core == "gated":
                attention_gated_in_place(q4, k4, v4, g4, bias[b * H:(b + 1) * H], keep2, self.prescale, self.core_precision)   # sigmoid(g)*o over q
                torch.mm(qkvg[:, :D], p["wo"].t(), out=y)
            else:
                attention_in_place(q, k, v, bias[b * H:(b + 1) * H].unsqueeze(0), buf["keep"], buf["lse"], key)
                self.K.gate_rows(qkvg[:, :D], qkvg[:, 3 * D:], a)            # o sits where q was
                torch.mm(a, p["wo"].t(), out=y)
            self.K.resgate_adaln_rows(x, y, g2[:, b, 0], g1[:, b, 2], g1[:, b, 3], xa, L, self.eps)
            if gsw is not None:
                gsw(xa, p["wab"], h)                                  # expand + SwiGLU, one sm_100a kernel
            else:
                torch.mm(xa, p["wab"].t(), out=ab)
                self.K.swiglu_rows(ab, h)
            self._mm(h, p["ws"], y)
            last = b + 1 == self.nb
            self.K.resgate_adaln_rows(x, y, g2[:, b, 1], None if last else g1[:, b + 1, 0],
                                 None if last else g1[:, b + 1, 1], xa, L, self.eps)
        return x

    @staticmethod
    def _blackwell_bf16(device, dtype):
        return dtype is torch.bfloat16 and device.type == "cuda" and torch.cuda.get_device_capability(device) == (10, 0)

    def _gemm_swiglu(self, device, xa):
        """The sm_100a expand GEMM with the SwiGLU epilogue (``kernels/conditioned_transition/cuda/gemm_swiglu.py``) where
        it fits and builds; cuBLAS + the SwiGLU row pass otherwise."""
        if not self._blackwell_bf16(device, self.dtype) or torch.compiler.is_compiling():
            return None
        idx = device.index if device.index is not None else torch.cuda.current_device()
        cache = self.__dict__.setdefault("_gemm_swiglu_ops", {})
        key = (idx, xa.shape[0])                                      # the choice depends on M (gemm_swiglu.MIN_ROWS)
        if key not in cache:
            from miniworld_engine.kernels.conditioned_transition.cuda import gemm_swiglu
            op = None
            wab = self.per[0]["wab"]
            if gemm_swiglu.supported(xa, wab):
                try:
                    op = gemm_swiglu.GemmSwiglu(idx, K=wab.shape[1], H=wab.shape[0] // 2)
                except Exception as exc:  # noqa: BLE001 -- a failed build keeps cuBLAS + the row pass
                    import warnings
                    warnings.warn(f"sm100 SwiGLU GEMM unavailable, keeping cuBLAS: {exc!r}", RuntimeWarning, stacklevel=2)
            cache[key] = op
        return cache[key]

    def _sm100_core(self, device, L, D, H):
        """The sm_100a gated core (``augmented_attention/cuda/sm100``, attn_inf.cu) where it fits -- bf16, 16 x 48, L a
        multiple of 128, B200 -- and builds; the Triton gated2 core otherwise. Same contract: pre-scaled logits in,
        sigmoid(g) * o written over q."""
        idx = device.index if device.index is not None else torch.cuda.current_device()
        cores = self.__dict__.setdefault("_sm100_cores", {})
        if idx not in cores:
            from miniworld_engine.kernels.augmented_attention.cuda import sm100
            core = None
            if sm100.inference_core_supported(self.dtype, L, D, H, idx) and not torch.compiler.is_compiling():
                try:
                    core = sm100.GatedInferenceCore(idx)
                except Exception as exc:  # noqa: BLE001 -- a failed build keeps the Triton core
                    import warnings
                    warnings.warn(f"sm100 token DiT core unavailable, keeping the Triton core: {exc!r}", RuntimeWarning, stacklevel=2)
            cores[idx] = core
        core = cores[idx]
        return core if core is not None and L % 128 == 0 else None

    def _mm(self, A, W, out, bias=None):
        """out = A @ W^T (+ bias), cuBLAS."""
        if bias is None:
            torch.mm(A, W.t(), out=out)
        else:
            torch.addmm(bias, A, W.t(), out=out)
