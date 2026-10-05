"""Fused token DiT inference over a stack of engine ``modules.dit.DiTBlock`` (AF3 Alg. 23), QK-norm on or off.

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
import contextlib

import torch
import torch.nn.functional as F

from miniworld_engine.kernels.conditioned_transition.triton import token_dit_kernels as K
from miniworld_engine.kernels.conditioned_transition.triton.token_dit_attn import attention_gated_in_place, attention_gated_in_place2, bias_descriptor


@contextlib.contextmanager
def _tf32(on: bool):
    """cuBLAS on TF32 tensor cores for the fp32 path's GEMMs, whatever the caller's allow_tf32 (restored after): the fp32
    path is the TF32 recipe -- its attention core is TF32 too -- and IEEE fp32 GEMMs ran the step 5-8x slower."""
    if not on:
        yield
        return
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def _row_kernels(device):
    """The row passes between the GEMMs: CUDA on H100 and B200 (``kernels/conditioned_transition/cuda``), Triton elsewhere
    or when the extension does not build. MINIWORLD_TOKEN_DIT_ROWS_CUDA=0 keeps Triton there too."""
    import os
    if (device.type == "cuda" and torch.cuda.get_device_capability(device) in ((9, 0), (10, 0))
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


def _norm_eps(norm):
    """The eps an engine RMSNorm applies to bf16 / fp32 input (its float32 default when unset), or a plain ``eps``."""
    if hasattr(norm, "effective_eps"):
        return float(norm.effective_eps(torch.float32))
    return float(norm.eps)


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
        The residual stream is fp32 either way. fp32 GEMMs run on TF32 tensor cores (``_tf32``: the fp32 path is the TF32
        recipe), and ``core_precision`` sets the Triton attention core's MMA precision for fp32."""
        self.core_precision = core_precision
        self.prescale = prescale     # fold sm_scale*log2(e) into Wq,bq and log2(e) into the pair-bias weights (gated core only)
        self.core = core            # "gated": tdit.attn (sample-fastest grid, gate epilogue); "engine": the engine kernel + gate pass
        blocks = list(blocks)
        a0 = blocks[0].attention
        self.nb = len(blocks)
        self.h = a0.n_head
        self.d = a0.to_query.weight.shape[0]
        self.dc = a0.ada_ln_in.ln_cond.weight.shape[0]
        self.dp = a0.ln_pair.weight.shape[0]
        self.eps = 1e-5
        dev = a0.to_query.weight.device
        self.K = _row_kernels(dev)
        # QK-norm (RMSNorm of every q / k head after the projection) runs as one in-place CUDA row pass; the
        # sm_scale log2 e fold then moves from Wq / bq, which the norm would cancel, into the q norm's weight
        self.qk = bool(a0.use_qk_norm)
        assert all(bool(b.attention.use_qk_norm) == self.qk for b in blocks), "QK-norm must be on for every block or none"
        if self.qk:
            if not hasattr(self.K, "qknorm_rows"):
                raise NotImplementedError("QK-norm on the fused token DiT step needs the CUDA row kernels (H100 or B200)")
            eq, ek = getattr(a0, "qk_eps", None) or (a0.norm_query.effective_eps(dtype), a0.norm_key.effective_eps(dtype))
            self.eq, self.ek = float(eq), float(ek)
        f32 = lambda t: t.detach().float()
        # column blocks of the projection GEMM output. fp32 packs q | k | g | v: its sm_100a core (TF32) takes q | k | g and
        # v^T from a second GEMM over the last quarter of the weight (see _run), with no copy of the pack
        order = ("q", "k", "g", "v") if dtype is torch.float32 else ("q", "k", "v", "g")
        self.col = {n: i for i, n in enumerate(order)}
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
            proj = dict(q=at.to_query.weight, k=at.to_key.weight, v=at.to_value.weight, g=at.to_gate.weight)
            wqkvg = torch.cat([proj[n] for n in order], 0)
            bqkvg = torch.cat([at.to_query.bias.detach(), torch.zeros(3 * self.d, device=dev, dtype=at.to_query.bias.dtype)])  # q first
            qs = (self.d // self.h) ** -0.5 * 1.4426950408889634 if prescale and core in ("gated", "gated2") else 1.0
            if qs != 1.0 and not self.qk:                                   # sm_scale * log2(e) into Wq, bq
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
            if self.qk:                                                     # sm_scale * log2(e) into the q norm's weight
                self.per[-1].update(nq=(f32(at.norm_query.weight) * qs).contiguous(), nk=f32(at.norm_key.weight).contiguous())
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
        with _tf32(self.dtype is torch.float32):
            return self._hoist(pair, mask)

    def _hoist(self, pair, mask):
        L = pair.shape[1]
        out = torch.empty(self.nb * self.h, L, L, device=pair.device, dtype=self.dtype)
        # the TF32 core reads its bias with the key columns permuted (``key_perm``); the LayerNorm lays them out so
        self._bias_perm = self._tf32_core(pair.device, L) is not None
        if getattr(self.K, "PAIR_BIAS_MASK", False):              # the CUDA rows fold the mask into the same pass
            self.K.pair_bias_all(pair.reshape(L * L, self.dp), self.pw_t, out, L, self.eps, mask=mask, perm=self._bias_perm)
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
        # A sampling step shares one conditioning across the samples (cond [1, ...] or expanded, stride 0): L rows. A
        # conditioning per sample: S L rows. The row kernels read these tables at row % period with period = their row
        # count (see _run), so both cases run the same kernels.
        shared = cond.shape[0] == 1 or cond.stride(0) == 0
        c = cond[0, 0] if shared else cond.reshape(-1, self.dc)       # [L or S L, dc]
        rows = c.shape[0]
        if hasattr(self.K, "cond_rows") and self.dc == 384:
            cn = torch.empty(rows, self.dc, device=c.device, dtype=self.dtype)
            cc = torch.empty_like(cn)
            self.K.cond_rows(c, cn, cc, self.eps)
        elif hasattr(self.K, "layernorm_rows"):                      # other CUDA row providers
            cn = torch.empty(rows, self.dc, device=c.device, dtype=self.dtype)
            self.K.layernorm_rows(c, cn, self.eps)
            cc = c.to(self.dtype)
        else:
            cn = F.layer_norm(c.float(), (self.dc,), eps=self.eps).to(self.dtype)
            cc = c.to(self.dtype)
        g1 = torch.addmm(self.b1, cn, self.w1.t()).view(rows, self.nb, 4, D)
        g2 = torch.addmm(self.b2, cc, self.w2.t()).view(rows, self.nb, 2, D)
        return g1, g2

    def step(self, single, cond, bias, out_dtype=None, streams=None):
        """One solver step. ``streams`` > 1 splits the samples into that many groups, each on its own CUDA stream: the
        samples share only the conditioning and the pair bias, so one group's memory-bound passes can run under another
        group's GEMMs and attention."""
        with _tf32(self.dtype is torch.float32):
            return self._step(single, cond, bias, out_dtype or single.dtype, streams)

    def _step(self, single, cond, bias, want, streams):
        S, B, L, D = single.shape
        assert B == 1
        streams = min(streams or self.streams, S)
        g1, g2 = self._cond(cond, L, D)
        if g1.shape[0] != L:
            streams = 1                                    # per-sample tables are not split across sample groups
        if streams <= 1:
            out, fresh = self._run(single, g1, g2, bias, 0, want)
            # copy=True otherwise: the fp32 residual is this runner's buffer; returning it uncopied (fp32 out) let the next
            # step on a reused runner overwrite an earlier result
            return out.view(S, 1, L, D) if fresh else out.view(S, 1, L, D).to(want, copy=True)
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
                outs.append(self._run(single[cuts[i]:cuts[i + 1]], g1, g2, bias, i + 1, want)[0])
            done = torch.cuda.Event()
            done.record(st)
            main.wait_event(done)
        return torch.cat([o.view(-1, 1, L, D) for o in outs], 0).to(want)

    def _run(self, single, g1, g2, bias, tag, want):
        """The blocks for one group of samples: (output [S*L, D], fresh). With the CUDA rows the input is read and the
        output written by the first / last row pass (a fresh ``want`` tensor); otherwise (the fp32 residual buffer, False)."""
        from miniworld_engine.autotune.shape_key import atom_key
        S, B, L, D = single.shape
        M, H = S * L, self.h
        P = g1.shape[0]                                    # AdaLN / gate table period: L (shared cond) or S L (per sample)
        buf = self._buffers(S, L, single.device, tag)
        x, xa, qkvg, a, y, ab, h = (buf[k] for k in ("x", "xa", "qkvg2", "a", "y", "ab", "h"))
        fused_io = hasattr(self.K, "adaln_in_rows")
        if fused_io:                                       # x = single (into fp32) and the first AdaLN, one pass
            self.K.adaln_in_rows(single.reshape(M, D), x, g1[:, 0, 0], g1[:, 0, 1], xa, P, self.eps)
        else:
            x.copy_(single.reshape(M, D))
            self.K.adaln_rows(x, g1[:, 0, 0], g1[:, 0, 1], xa, P, self.eps)
        # q, k, v, g as strided views of ONE [M, 4D] GEMM output (column blocks in self.col order); the core launcher
        # honours strides
        c = self.col
        q, k, v = (qkvg.view(S, 1, L, 4 * D)[..., c[n] * D:(c[n] + 1) * D].unflatten(-1, (H, D // H)) for n in "qkv")
        key = atom_key(L)
        q4, k4, v4, g4 = (qkvg.view(S, L, 4 * D)[..., c[n] * D:(c[n] + 1) * D].unflatten(-1, (H, D // H)) for n in "qkvg")
        cuda_core = self._cuda_core(single.device, L, D, H) if self.core == "gated2" and self.prescale else None
        tf32 = self._tf32_core(single.device, L)
        tf32_core = (cuda_core or tf32) and self.dtype is torch.float32
        if tf32_core:
            # Both architectures' TF32 cores take q | k | g and v^T. The upstream fp32 pack has q | k | g | v order.
            qkg, vt = qkvg.view(-1)[:3 * M * D].view(M, 3 * D), qkvg.view(-1)[3 * M * D:].view(D, M)
            if tf32:
                assert self._bias_perm, "the H100 TF32 core needs a key-permuted bias"
        keep2 = buf["keep"].view(S, L)
        gsw = self._gemm_swiglu(single.device, xa)
        if self.core == "gated2" and not cuda_core and not tf32:
            assert self.prescale, "the v2 core expects pre-scaled logits"
            key_d = (bias.data_ptr(), tuple(bias.shape))
            if getattr(self, "_bdesc_key", None) != key_d:
                self._bdesc, self._bdesc_key = bias_descriptor(bias), key_d
            bdesc = self._bdesc
        for b, p in enumerate(self.per):
            if not tf32_core:
                self._mm(xa, p["wqkvg"], qkvg, p["bqkvg"])
                if self.qk:
                    self.K.qknorm_rows(qkvg, p["nq"], p["nk"], self.eq, self.ek, D)
            if tf32_core:
                self._mm(xa, p["wqkvg"][:3 * D], qkg, p["bqkvg"][:3 * D])    # q | k | g
                if self.qk:
                    self.K.qknorm_rows(qkg, p["nq"], p["nk"], self.eq, self.ek, D)
                torch.mm(p["wqkvg"][3 * D:], xa.t(), out=vt)                 # v^T (v has no bias)
                if tf32:
                    tf32(qkg, vt, bias, b, S, 2 * D)                      # H100: key-permuted bias, gate at column 2D
                else:
                    assert cuda_core is not None
                    cuda_core(qkg, bias, b, S, vt)                       # B200: upstream TF32 core and layouts
                self._mm(qkg[:, :D], p["wo"], y)
            elif cuda_core:
                cuda_core(qkvg, bias, b, S)                               # sigmoid(g)*o over q, native bf16 core
                self._mm(qkvg[:, :D], p["wo"], y)
            elif self.core == "gated2":
                attention_gated_in_place2(q4, k4, v4, g4, bdesc, b, self.core_precision)  # sigmoid(g)*o over q
                self._mm(qkvg[:, :D], p["wo"], y)
            elif self.core == "gated":
                attention_gated_in_place(q4, k4, v4, g4, bias[b * H:(b + 1) * H], keep2, self.prescale, self.core_precision)   # sigmoid(g)*o over q
                torch.mm(qkvg[:, :D], p["wo"].t(), out=y)
            else:
                attention_in_place(q, k, v, bias[b * H:(b + 1) * H].unsqueeze(0), buf["keep"], buf["lse"], key)
                self.K.gate_rows(qkvg[:, :D], qkvg[:, c["g"] * D:(c["g"] + 1) * D], a)   # o sits where q was
                torch.mm(a, p["wo"].t(), out=y)
            self.K.resgate_adaln_rows(x, y, g2[:, b, 0], g1[:, b, 2], g1[:, b, 3], xa, P, self.eps)
            if gsw is not None:
                gsw(xa, p["wab"], h)                                  # expand + SwiGLU, one CUDA kernel
            else:
                torch.mm(xa, p["wab"].t(), out=ab)
                self.K.swiglu_rows(ab, h)
            self._mm(h, p["ws"], y)
            last = b + 1 == self.nb
            if last and fused_io:                          # the last residual straight into the output
                out = torch.empty(M, D, device=x.device, dtype=want)
                self.K.resgate_out_rows(x, y, g2[:, b, 1], out, P)
                return out, True
            self.K.resgate_adaln_rows(x, y, g2[:, b, 1], None if last else g1[:, b + 1, 0],
                                 None if last else g1[:, b + 1, 1], xa, P, self.eps)
        return x, False

    @staticmethod
    def _blackwell_bf16(device, dtype):
        return dtype is torch.bfloat16 and device.type == "cuda" and torch.cuda.get_device_capability(device) == (10, 0)

    def _gemm_swiglu(self, device, xa):
        """The expand GEMM with the SwiGLU epilogue (``kernels/conditioned_transition/cuda/gemm_swiglu.py``: sm_100a, or
        sm_90a) where it fits and builds; cuBLAS + the SwiGLU row pass otherwise."""
        cap = torch.cuda.get_device_capability(device) if device.type == "cuda" else None
        if (torch.compiler.is_compiling() or cap not in ((9, 0), (10, 0))
                or not (self.dtype is torch.bfloat16 or (self.dtype is torch.float32 and cap == (9, 0)))):   # fp32: sm_90 (TF32)
            return None
        idx = device.index if device.index is not None else torch.cuda.current_device()
        cache = self.__dict__.setdefault("_gemm_swiglu_ops", {})
        key = (idx, xa.shape[0])                                      # the choice depends on M (gemm_swiglu.MIN_ROWS)
        if key not in cache:
            from miniworld_engine.kernels.conditioned_transition.cuda import gemm_swiglu
            op = None
            wab = self.per[0]["wab"]
            if gemm_swiglu.supported(xa, wab) or gemm_swiglu.supported_sm90(xa, wab):
                try:
                    op = (gemm_swiglu.GemmSwiglu(idx, K=wab.shape[1], H=wab.shape[0] // 2)
                          if torch.cuda.get_device_capability(device) == (10, 0) else gemm_swiglu.GemmSwigluSm90(idx))
                except Exception as exc:  # noqa: BLE001 -- a failed build keeps cuBLAS + the row pass
                    import warnings
                    warnings.warn(f"SwiGLU GEMM unavailable, keeping cuBLAS: {exc!r}", RuntimeWarning, stacklevel=2)
            cache[key] = op
        return cache[key]

    def _cuda_core(self, device, L, D, H):
        """The upstream B200 bf16 / TF32 cores and layouts, or the H100 bf16 16 x 48 core where each is supported.
        Same contract: pre-scaled logits in, sigmoid(g) * o written over q. Unsupported calls keep Triton."""
        idx = device.index if device.index is not None else torch.cuda.current_device()
        cores = self.__dict__.setdefault("_cuda_cores", {})
        if idx not in cores:
            from miniworld_engine.kernels.augmented_attention import cuda as sm90
            from miniworld_engine.kernels.augmented_attention.cuda import sm100
            core = None
            if not torch.compiler.is_compiling():
                try:
                    if sm100.inference_core_supported(self.dtype, L, D, H, idx):
                        core = sm100.GatedInferenceCore(idx, self.dtype, H, D // H)
                    elif sm90.inference_core_supported(self.dtype, L, D, H, idx):
                        sm90._ext("attn_fwd")
                        core = sm90.gated_inference
                except Exception as exc:  # noqa: BLE001 -- a failed build keeps the Triton core
                    import warnings
                    warnings.warn(f"CUDA token DiT core unavailable, keeping the Triton core: {exc!r}", RuntimeWarning, stacklevel=2)
            cores[idx] = core
        core = cores[idx]
        multiple = 128 if torch.cuda.get_device_capability(device) == (9, 0) else 8
        return core if core is not None and L % multiple == 0 else None

    def _tf32_core(self, device, L):
        """The fp32 step's TF32 CUDA core (``augmented_attention/cuda``, attn_tf32.cu) where it fits and builds: fp32,
        pre-scaled logits, H100, L a multiple of 128, the CUDA rows (they lay out its key-permuted bias). None otherwise
        (the Triton gated2 core, TF32 too)."""
        if (self.dtype is not torch.float32 or self.core != "gated2" or not self.prescale or not hasattr(self.K, "key_perm")
                or L % 128 or torch.compiler.is_compiling()):
            return None
        idx = device.index if device.index is not None else torch.cuda.current_device()
        cores = self.__dict__.setdefault("_tf32_cores", {})
        if idx not in cores:
            from miniworld_engine.kernels.augmented_attention import cuda as sm90
            core = None
            if sm90.tf32_inference_core_supported(self.dtype, L, self.d, self.h, idx):
                try:
                    sm90._ext("attn_tf32")                                  # build now: a failure keeps Triton
                    core = sm90.tf32_gated_inference
                except Exception as exc:  # noqa: BLE001 -- a failed build keeps the Triton core
                    import warnings
                    warnings.warn(f"TF32 token DiT core unavailable, keeping the Triton core: {exc!r}", RuntimeWarning, stacklevel=2)
            cores[idx] = core
        return cores[idx]

    def _mm(self, A, W, out, bias=None):
        """out = A @ W^T (+ bias), cuBLAS."""
        if bias is None:
            torch.mm(A, W.t(), out=out)
        else:
            torch.addmm(bias, A, W.t(), out=out)
