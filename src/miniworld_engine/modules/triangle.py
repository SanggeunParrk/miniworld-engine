"""Triangle updates of the pair representation: triangle attention (AlphaFold 3 Algorithms 14 / 15) and the triangle multiplicative update
(Algorithms 12 / 13), each single-direction and bidirectional, with the residual and the row / column dropout.

Each is a :class:`~miniworld_engine.modules.primitives.BackendModule`; ``_forward_pytorch`` is the reference.

Model variants are separate classes; :class:`TriangleAttention` and :class:`TriangleMultiplication` select one by name (``model=``) and are
themselves the variant most models run::

    TriangleAttention(128, 4)                         # MiniWorld, Boltz-2, Protenix, OpenDDE, ESMFold2 (bias of the transposed pair)
    TriangleAttention(256, 4, model="chai1")          # -> TriangleAttentionChai1(256, 4)
    TriangleMultiplication(128, model="rf3")          # -> TriangleMultiplicationRF3(128)

=====================================  ====================================================================================================
class                                  models (``model=``)
=====================================  ====================================================================================================
TriangleAttention                      miniworld, boltz2, protenix, protenix_v2, opendde, esmfold2, openfold3_preview2: the ending node's
                                       bias from the transposed pair (``protenix_v2`` is an alias of ``protenix``: Protenix-v2 runs
                                       Protenix's triangle attention and multiplication unchanged; TriangleMultiplication takes it too)
TriangleAttentionAF3                   af3, openfold3 (v0.5), intellifold2: the ending node's bias from the pair as it is (AF3's code)
TriangleAttentionRF3                   rf3: gate / out-projection biases, AF3's bias orientation, fp32 softmax
TriangleAttentionChai1                 chai1: AF3's bias orientation, the ending node's update in the transposed orientation, fp32 softmax
TriangleAttentionChai1Confidence       chai1_confidence: a second, transposed out-projection (Chai-1's confidence head), fp32 softmax
BidirectionalTriangleAttention         MiniWorld: starting + ending node in one block
TriangleMultiplication                 every model but rf3 (RoseTTAFold3's fast mode: ``PairformerConfig(model="rf3", trimul_model="miniworld")``)
TriangleMultiplicationRF3              rf3: the contraction divided by the real sequence length
BidirectionalTriangleMultiplication    MiniWorld: outgoing + incoming in one block
=====================================  ====================================================================================================
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

import torch
import torch.nn.functional as F
from einops import rearrange

from miniworld_engine.modules.functional import dropout_scale, inverse_length, sigmoid_gate
from miniworld_engine.modules.norm import LayerNorm, RMSNorm
from miniworld_engine.modules.primitives import BackendModule, Linear, MasterWeights, master_autocast, pair_inputs, wide


def _select(registry: dict[str, type], kind: str, model: str) -> type:
    try:
        return registry[model]
    except KeyError:
        raise ValueError(f"no {kind} for model {model!r}; known: {', '.join(sorted(registry))}") from None


#: pair_fused triangle-attention classes (C, H, D, C_out) the library refused at run time in this process (its Refused: a cell that cannot
#: build on this stack); the anthropic backends take the core provider for them from then on.
_PAIR_FUSED_REFUSED: dict[tuple, str] = {}


def _key_mask5(mask: torch.Tensor | None, q: torch.Tensor) -> torch.Tensor | None:
    """The module's key mask ``[B, L]`` as the libraries' ``[B, N, 1, 1, K]`` (the same keys for every row ``N`` of ``q [B, N, H, Q, d]``)."""
    if mask is None:
        return None
    B, N, _, _, _ = q.shape
    return mask.bool()[:, None, None, None, :].expand(B, N, 1, 1, mask.shape[-1]).contiguous()


def _cueq_core(q, k, v, bias, mask):
    """The triangle-attention core on ``cuequivariance_torch.triangle_attention``: q / k / v ``[B, N, H, Q, d]``, bias ``[B, 1, H, Q, K]`` (fp32,
    as the kernel takes it), key mask ``[B, K]`` -> ``[B, N, H, Q, d]``."""
    from miniworld_engine.modules import _cueq

    return _cueq.torch_api().triangle_attention(q, k, v, bias.float(), _key_mask5(mask, q), q.shape[-1] ** -0.5)


def _triton_core(q, k, v, bias, mask):
    """The triangle-attention core on the engine's Triton kernel (``kernels/triangle_attention/triton/main``, forward + backward), in the
    cuEquivariance signature of :func:`_cueq_core`; masked keys at the bias dtype's min (the reference's fill)."""
    from miniworld_engine.kernels.triangle_attention.triton.main import triton_triangle_attention_pair_bias

    b = bias[:, 0]  # [B, H, Q, K]
    if mask is not None:
        b = b.masked_fill(~mask.bool()[:, None, None, :], torch.finfo(b.dtype).min)
    out = triton_triangle_attention_pair_bias(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), b)  # [B, H, N, Q, d]
    return out.transpose(1, 2)


# ================================================================================================================== triangle attention
class _TriangleAttentionBase(BackendModule, MasterWeights):
    """The parameters and the residual / dropout of every single-direction triangle attention; a variant defines :meth:`_update` (the raw
    update, no residual) and its class constants.

    Backends (self-attention only; the bias-only form has no query / key): the attention CORE ``softmax_k(q_ij . k_ik / sqrt(d) + b_jk) v_ik``
    runs in a library kernel, every other statement (LayerNorm, projections, bias orientation, gate, out-projection(s), residual, dropout) is
    the module's own. ``"cueq"``: ``cuequivariance_torch.triangle_attention`` (forward + backward). ``"anthropic"``: the shared core's
    triangle-attention provider (``opt_core.kernels.triattn``, the af3_torch / rosettafold3 / chai1 kits' core) by its ``fast`` tier word,
    inference only; for :class:`TriangleAttention` / :class:`TriangleAttentionAF3` first the whole fused block the Boltz-2 / Protenix /
    OpenFold3 kits install (``opt_core.attn.pair_fused.tri_attn_block``: LN + projections prologue, core ``tier:fast``, gate + W_o
    epilogue; the residual the module's), the core provider where the block refuses.
    Masked keys: the kernels drop them (a large negative logit / -inf); the reference fills the dtype's minimum (RF3 / Chai-1: -inf) -- the
    same softmax for every row with at least one unmasked key.

    The engine's own backends. ``"cuda"`` (H100 sm_90a, bf16 activations and parameters): the whole update (LN, projections, core, gate,
    out-projection, residual, dropout) on ``kernels/integrations/triattn_h100_sm90`` (d_pair <= 512 multiple of 8, heads of 16 / 32
    channels -- training also 64 --, <= 16 heads, L % 16 == 0, no q / k norm; training for every variant: AF3's ending-node bias, RF3's
    projection biases, Chai-1's untransposed / dual output) or, inference only, ``kernels/triangle_attention/cuda/h100_dh`` (heads of
    32 / 64 channels, Chai-1's output variants).
    ``"triton"``: the attention core on ``kernels/triangle_attention/triton/main`` (flash-style, forward + backward), every other statement
    the module's own (as the library cores), bf16 only.

    The bias-only form (``use_self_attention=False``; :class:`TriangleAttention` only) has its own engine paths (engine 2.4's, restored
    2026-11-12): ``"cuda"`` on BidirectionalTriangleAttention's H100 / A100 kernels for one direction
    (``kernels/integrations/bidir_triattn_h100.forward_single``: the glue LayerNorm, one [Wv; Wg; Wb] GEMM, the softmax per (sample, head),
    the contraction as plain GEMMs, the gate, the out-projection with the residual / dropout; bf16 activations), ``"triton"`` on the fused
    LayerNorm + projection and the gate-folded out-projection (:meth:`_bias_only_triton`). The libraries (``"cueq"``, ``"anthropic"``) are
    q.k attention and refuse it.
    """

    backends = ("pytorch", "anthropic", "cueq", "cuda", "triton")
    backend_versions: ClassVar[dict[str, int]] = {
        "anthropic": 3,  # 3 (2026-11-13): the block folds the residual (z += block(z)); the cu129 triattn prebuilts
        "cueq": 2,
        "cuda:a100_sm80": 4,
        "cuda:h100_sm90": 5,
        "triton": 2,
    }  # 2026-11-10: kits; sm_80 paths; 3: A100 AF3 flip in the fronts, inference casts / packs read by captures; cuda:a100_sm80 4 / triton 2
    # (2026-11-12): row-blocked inference past a whole-pair budget (FoldForge OpenDDE refiner, 2976 / 3968 tokens at d_pair 384; A100: also
    # L % 128 != 0, padded); cuda:h100_sm90 2 (2026-11-12): row blocks of <= 2 GiB of q | k | v | g far past the working-set cap -- served
    # sizes up to L2048 keep their path; cuda:h100_sm90 3 (2026-11-12): the training dropout scale drawn by the H100 Philox kernel in one
    # launch (kernels/triangle_attention/cuda/h100_triattn.drop_scale; the same bits as the torch draw)
    # cuda:h100_sm90 4 (2026-11-13): d_pair 128 x 4 heads inference: the tail projects the gate from x (fp32, never stored; the front
    # writes q | k | v only) -- outputs change within rounding, see the classes' `version`; d_pair 256 x 8 heads: the wide front's CTA
    # groups kept in step (FW_LAG; bitwise the same)
    # cuda:h100_sm90 5 (2026-11-13): d_pair 256 x 8 heads inference: the LayerNorm folded into the wide front's weight pack (the GEMM
    # reads the pair as loaded; outputs change within rounding, backend-only); d_pair 128 x 4 heads ending node: the tail reads o
    # transposed and x / the output as stored (bitwise the same); AF3 d_pair 64 (4 heads x 16): the one-kernel front up to L2048 (was 768)

    #: A numerically dead instance of a checkpoint (set by a model's checkpoint loader): inference returns the residual input.
    dead: bool = False
    #: The orientation the ending node takes its pair bias from: ``True`` the transposed pair (the frame its queries and keys are in, the
    #: supplementary-information and OpenFold convention), ``False`` the pair as it is, read in the transposed frame (AlphaFold 3's released
    #: code). The starting node has one orientation.
    transposed_bias: bool = True
    #: Chai-1: the ending node returns its update as the transposed pair has it.
    untransposed_output: bool = False
    #: Chai-1 confidence head: a second out-projection (``to_out_t``) whose update lands transposed.
    dual_output: bool = False

    def __init__(
        self,
        d_pair: int,
        n_head: int,
        *,
        d_hidden: int | None,
        starting: bool,
        use_self_attention: bool,
        use_qk_norm: bool,
        p_drop: float,
        gate_bias: bool,
        out_bias: bool,
        master_weights: bool,
        backend: str,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.starting = starting
        #: Ending node on AlphaFold 3's bias orientation: the bias of the pair as it is, read in the transposed frame.
        self._bias_untransposed = not starting and not self.transposed_bias
        self.use_self_attention = use_self_attention
        self.use_qk_norm = use_qk_norm
        # The residual is unconditional; the dropout (drop_row for starting, drop_col for ending) acts in training only.
        self.p_drop = p_drop
        self._drop_dim = 1 if starting else 2

        if d_hidden is None:
            d_hidden = d_pair

        if d_hidden % n_head != 0:
            msg = f"d_hidden ({d_hidden}) must be divisible by n_head ({n_head})"
            raise ValueError(msg)

        self.ln_pair = LayerNorm(d_pair)
        if use_self_attention:
            self.to_query = Linear(d_pair, d_hidden, bias=False, init="glorot")
            self.to_key = Linear(d_pair, d_hidden, bias=False, init="glorot")

            if use_qk_norm:
                d_head = d_hidden // n_head
                self.norm_query = RMSNorm(d_head)
                self.norm_key = RMSNorm(d_head)

        self.to_value = Linear(d_pair, d_hidden, bias=False, init="glorot")
        self.to_bias = Linear(d_pair, n_head, bias=False, init="default")
        self.to_gate = Linear(d_pair, d_hidden, bias=gate_bias, init="gating")
        self.to_out = Linear(d_hidden, d_pair, bias=out_bias, init="zero")
        if self.dual_output:
            self.to_out_t = Linear(d_hidden, d_pair, bias=out_bias, init="zero")
        if master_weights:
            self.set_master_weights()
        self.set_backend(backend)

    def _forward_pytorch(self, pair: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``pair`` ``[B, L, L, d_pair]``, ``mask`` ``[B, L]`` (bool). ALWAYS returns the residual output ``pair + drop(update(pair))``; the
        raw update without the residual is :meth:`_update`."""
        return self._residual(pair, mask, None)

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        return pair_inputs(
            self.ln_pair.normalized_shape[0], length=length, samples=samples, dtype=dtype, device=device, generator=generator
        )

    def _residual(self, pair: torch.Tensor, mask: torch.Tensor | None, core) -> torch.Tensor:
        """``pair + drop(update(pair))`` with the attention core ``core`` (``None``: the reference's own statements; else a function of the
        cuEquivariance signature, see :func:`_cueq_core`)."""
        if self.dead and not (self.untransposed_output or self.dual_output or self.training or torch.is_grad_enabled()):
            return pair
        out = self._update(pair, mask, core)
        if self.p_drop > 0.0 and self.training:
            out = out * dropout_scale(pair, self.p_drop, self._drop_dim)
        return pair + out

    def _update(self, pair: torch.Tensor, mask: torch.Tensor | None = None, core=None) -> torch.Tensor:
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------- library cores
    def _head_dim(self) -> int:
        return self.to_value.weight.shape[0] // self.n_head

    def _core_refusal(self, pair: torch.Tensor, mask: torch.Tensor | None) -> str | None:
        """What the attention cores need from the call (the call's shape / mask; the library's own range is checked by its backend). The
        projections are the module's own statements: fp32 masters next to a bf16 pair run them under autocast (:func:`master_autocast`)."""
        if not self.use_self_attention:
            return "bias-only attention (use_self_attention=False) has no query / key: the library op is q.k attention"
        if pair.dim() != 4 or pair.shape[1] != pair.shape[2]:
            return f"pair must be [B, L, L, d_pair], got {tuple(pair.shape)}"
        if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != tuple(pair.shape[:2])):
            return "mask must be a bool [B, L] key mask"
        return None

    def _supports_cueq(self, pair: torch.Tensor, mask: torch.Tensor | None = None) -> str | None:
        from miniworld_engine.modules import _cueq

        why = _cueq.unavailable(pair.device) or self._core_refusal(pair, mask)
        if why is not None:
            return why
        d, length = self._head_dim(), pair.shape[1]
        # cuequivariance_torch.triangle_attention runs its PyTorch formula outside these bounds (its kernels' head widths; sequences up to
        # its fallback threshold, raised to 200 below a head width of 32): the library's own rules, read from the library
        threshold = int(_cueq.ops("triangle_attention").CUEQ_TRIATTN_FALLBACK_THRESHOLD)
        if pair.dtype == torch.bfloat16:
            if d > 128 or d % 8:
                return f"head width {d}: the bf16 kernel takes widths <= 128 divisible by 8"
        elif pair.dtype == torch.float32:
            if d > 32 or d % 4:
                return f"head width {d}: the fp32 kernel takes widths <= 32 divisible by 4"
            if torch.is_grad_enabled() and (pair.requires_grad or any(p.requires_grad for p in self.parameters())):
                return "fp32 training: the op's fp32 backward exists only in TF32 (it refuses IEEE fp32); fp32 is inference only"
        else:
            return f"dtype {pair.dtype}: the backend is verified for bfloat16 / float32"
        if d < 32:
            threshold = max(threshold, 200)
        if length <= threshold:
            return f"L={length} <= {threshold}: cuEquivariance runs its PyTorch formula for sequences this short (head width {d})"
        return None

    def _forward_cueq(self, pair: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        with master_autocast(pair, self):  # fp32 master matrices under bf16 activations: the projections run bf16, as the reference's do
            return self._residual(pair, mask, _cueq_core)

    #: The variants whose update is the fused triangle-attention BLOCK of ``opt_core.attn.pair_fused`` (``tri_attn_block``: LN -> q | k | v |
    #: g | pair-bias prologue, the attention core, the gate x o @ W_o epilogue; the ending node reads z^T by address math). It is the pair
    #: block the Boltz-2 (``boltz2_opt.pairblock`` / ``pairfuse``), Protenix-v1 (``levers_ptx1._tri_forward``) and OpenFold3 kits
    #: (``cells/pairfused.tri_att_start_end``; v0.5 / ob0 with ``bias_frame="z"``) install on their models' triangle attention.
    _anthropic_block: bool = False

    def _anthropic_select(self, pair: torch.Tensor) -> tuple[Any, tuple[Any, str | None]]:
        from miniworld_engine.modules import _anthropic

        prov = _anthropic.load("opt_core.kernels.triattn")
        return prov, _anthropic.kernel_selection(
            prov,
            "{}.{}".format(*torch.cuda.get_device_capability(pair.device)),
            {torch.bfloat16: "bf16", torch.float32: "fp32"}[pair.dtype],
            self._head_dim(),
            self.n_head,
            int(pair.shape[1]),
            "fwd",
            word="fast",
            form="keypad",
        )

    def _anthropic_block_call(self, pair: torch.Tensor, mask: torch.Tensor | None) -> tuple[Any, Any, Any, str | None]:
        """``(PF, W, kwargs, None)`` for ``pair_fused.tri_attn_block`` on this call, or ``(None, None, None, reason)``."""
        from miniworld_engine.modules import _anthropic

        if not self._anthropic_block:
            return None, None, None, "the variant's update (fp32 softmax / output forms) is not pair_fused's block"
        if self.use_qk_norm:
            return None, None, None, "q / k RMSNorm is not part of pair_fused's prologue"
        why = _anthropic.missing("opt_core.attn.pair_fused")
        if why is not None:
            return None, None, None, why
        PF = _anthropic.load("opt_core.attn.pair_fused")
        key = tuple((p.data_ptr(), p._version, p.dtype) for p in self.parameters())
        if getattr(self, "_anthropic_pf_key", None) != (key, pair.device):
            self._anthropic_pf = PF.pack_triattn_weights(
                ln_w=self.ln_pair.weight,
                ln_b=self.ln_pair.bias,
                w_q=self.to_query.weight,
                w_k=self.to_key.weight,
                w_v=self.to_value.weight,
                w_g=self.to_gate.weight,
                w_b=self.to_bias.weight,
                w_o=self.to_out.weight,
                b_g=self.to_gate.bias,
                b_o=self.to_out.bias,
                n_heads=self.n_head,
                head_dim=self._head_dim(),
                eps=self.ln_pair.eps,
                device=pair.device,
            )
            self._anthropic_pf_key = (key, pair.device)
        W = self._anthropic_pf
        pow2 = all(v and not v & (v - 1) for v in (W.C, W.H * W.D, W.D, W.C_out))
        # the fused kernels take power-of-two widths; the torch surround (its declared construction for c_z 384 / 12 heads) the others
        kw = {"ending": not self.starting, "impl": "fpf" if pow2 else "torch", "core": "tier:fast"}
        m5 = None if mask is None else mask[:, None, None, None, :].expand(pair.shape[0], pair.shape[1], 1, 1, pair.shape[1])
        if (W.C, W.H, W.D, W.C_out) in _PAIR_FUSED_REFUSED:
            return None, None, None, _PAIR_FUSED_REFUSED[(W.C, W.H, W.D, W.C_out)]
        ok, why = PF.supported_triattn(pair, W, m5, ln="fused", residual=False, **kw)
        if not ok:
            return None, None, None, f"pair_fused.tri_attn_block refuses: {why}"
        # the ending node's bias: from the transposed pair ('x', TriangleAttention) or from the pair as it is ('z', AlphaFold 3's code)
        kw.update(mask=m5, ln="fused", residual=False, scale=self._head_dim() ** -0.5, bias_frame="z" if self._bias_untransposed else "x")
        return PF, W, kw, None

    def _supports_anthropic(self, pair: torch.Tensor, mask: torch.Tensor | None = None) -> str | None:
        from miniworld_engine.modules import _anthropic

        why = _anthropic.unavailable("opt_core.kernels.triattn", pair.device) or self._core_refusal(pair, mask)
        if why is not None:
            return why
        if torch.is_grad_enabled():
            return "inference only (the provider's backward cells name the cuEquivariance op): call under torch.no_grad() / inference_mode"
        if self.training and self.p_drop > 0.0:
            return "the row / column dropout is live (training mode): inference only"
        if pair.dtype not in (torch.bfloat16, torch.float32):
            return f"dtype {pair.dtype}: the provider has bfloat16 / float32 cells only"
        with master_autocast(pair, self):
            block_why = self._anthropic_block_call(pair, mask)[3]
        if block_why is None:
            return None
        core_why = self._anthropic_select(pair)[1][1]
        return None if core_why is None else f"{block_why}; the attention-core provider: {core_why}"

    def _forward_anthropic(self, pair: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.dead and not (self.untransposed_output or self.dual_output):  # inference only (as the reference's dead instance)
            return pair
        with master_autocast(pair, self):
            PF, W, kw, block_why = self._anthropic_block_call(pair, mask)
            if block_why is None:
                try:
                    if pair.dtype == torch.bfloat16:
                        # the kits' block statement: z += block(z), the residual folded in the epilogue (the ending node scattered back by
                        # address math). The module returns a new tensor, so the block updates a copy of the pair in place.
                        return PF.tri_attn_block(pair.clone(), W, **{**kw, "residual": True})
                    return pair + PF.tri_attn_block(pair, W, **kw).to(pair.dtype)
                except PF.Refused as e:
                    # the library's own run-time refusal (a cell whose build fails on this stack, e.g. the c512 h8 d64 epilogue's ptxas
                    # error): it names the lever unusable in this process, so the class is the core provider's from here on
                    _PAIR_FUSED_REFUSED[(W.C, W.H, W.D, W.C_out)] = block_why = f"pair_fused refused at run time: {str(e)[:200]}"
        prov, (sel, why) = self._anthropic_select(pair)
        if sel is None:
            from miniworld_engine.modules.primitives import BackendUnsupported

            raise BackendUnsupported(f"{block_why}; the attention-core provider: {why}")
        stock = None
        if sel.row in prov.NEEDS_STOCK:  # a stock row the provider names (the cuEquivariance op ...): the kits hand the face their stock op
            from miniworld_engine.modules import _cueq

            stock = _cueq.torch_api().triangle_attention

        def core(q, k, v, bias, key_mask):
            m5 = _key_mask5(key_mask, q)
            return prov.triangle_attention(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                bias.float().contiguous(),
                m5,
                q.shape[-1] ** -0.5,
                word=sel.word,
                selection=sel,
                stock=stock,
            )

        with master_autocast(pair, self):  # fp32 master matrices under bf16 activations: the projections run bf16, as the reference's do
            return self._residual(pair, mask, core)

    # ------------------------------------------------------------------------------------------------------------- engine kernels
    def _make_drop_scale(self, pair: torch.Tensor, p: float) -> torch.Tensor:
        """The module's broadcast dropout scale (drop_row starting / drop_col ending), as the H100 kernels take it."""
        return dropout_scale(pair, p, self._drop_dim)

    def _trains(self, pair: torch.Tensor) -> bool:
        grad = torch.is_grad_enabled() and (pair.requires_grad or any(p.requires_grad for p in self.parameters()))
        return grad or (self.training and self.p_drop > 0.0)

    def _cuda_route(self, pair: torch.Tensor, mask: torch.Tensor | None) -> tuple[str | None, str | None]:
        """``("sm90" | "dh", None)``: the H100 path that serves the call, else ``(None, reason)``. Shapes / dtypes / flags only."""
        if not pair.is_cuda:
            return None, "the kernels run on CUDA tensors only"
        if not self.use_self_attention:
            # the bias-only form: BidirectionalTriangleAttention's H100 / A100 kernels, one direction (bidir_triattn_h100.forward_single)
            from miniworld_engine.kernels.integrations import bidir_triattn_h100

            why = bidir_triattn_h100.single_refusal(self, pair, mask)
            return (None, why) if why is not None else ("bias_only", None)
        cap = torch.cuda.get_device_capability(pair.device)
        if cap == (8, 0):
            return self._cuda_route_sm80(pair, mask)
        if cap != (9, 0):
            return None, "the cuda backend is the H100 (sm_90a) / A100 (sm_80) kernels"
        why = self._core_refusal(pair, mask)
        if why is not None:
            return None, why
        if self.use_qk_norm:
            return None, "q / k RMSNorm is not part of the H100 kernels"
        if pair.dtype != torch.bfloat16:
            return None, f"dtype {pair.dtype}: the H100 TriangleAttention kernels are bf16 only (no fp32 kernel)"
        trains = self._trains(pair)
        c, d = pair.shape[-1], self._head_dim()
        from miniworld_engine.kernels.integrations import triattn_h100_sm90

        if trains:
            # every variant trains on the sm90 route (bias flip, projection biases, Chai-1's output layouts in its glue; 64-channel
            # heads on the h100_dh core)
            if triattn_h100_sm90.serves(self, pair, mask):
                return "sm90", None
            return None, (
                "training: the H100 kernels take d_pair = to_query in-features <= 512 (multiple of 8), heads of 16 / 32 / 64 "
                f"channels, <= 16 heads, L % 16 == 0 (got d_pair {c}, head width {d}, {self.n_head} heads, L {pair.shape[1]})"
            )
        if not (self.untransposed_output or self.dual_output) and triattn_h100_sm90.serves(self, pair, mask):
            return "sm90", None
        if d in (32, 64) and c % 8 == 0 and c <= 512 and self.to_query.weight.shape[1] == c:
            return "dh", None
        return None, f"d_pair {c}, head width {d}, L {pair.shape[1]}: outside the H100 kernels (heads 16 / 32 / 64, d_pair <= 512, L % 16)"

    def _cuda_route_sm80(self, pair: torch.Tensor, mask: torch.Tensor | None) -> tuple[str | None, str | None]:
        """``("sm80" | "sm80_train", None)``: the A100 path (``kernels/integrations/triattn_sm80``: the whole update, inference or with
        autograd), else ``(None, reason)``. bf16, d_pair 128 / 4 x 32 or d_pair 64 .. 512 with heads of 16 / 32 / 64 channels, L % 128 == 0;
        every variant (AF3 / RF3 / Chai-1 bias orientation, RF3 projection biases, Chai-1 untransposed / dual output) on the variant path."""
        why = self._core_refusal(pair, mask)
        if why is not None:
            return None, why
        from miniworld_engine.kernels.integrations import triattn_sm80

        trains = self._trains(pair)
        why = triattn_sm80.refusal(self, pair, mask, train=trains)
        if why is not None:
            return None, why
        return ("sm80_train" if trains else "sm80"), None

    def _supports_cuda(self, pair: torch.Tensor, mask: torch.Tensor | None = None) -> str | None:
        return self._cuda_route(pair, mask)[1]

    def _forward_cuda(self, pair: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.dead and not (self.untransposed_output or self.dual_output or self.training or torch.is_grad_enabled()):
            return pair
        route, _ = self._cuda_route(pair, mask)
        if route == "bias_only":
            from miniworld_engine.kernels.integrations import bidir_triattn_h100

            return bidir_triattn_h100.forward_single(self, pair, mask)
        if route in ("sm80", "sm80_train"):
            from miniworld_engine.kernels.integrations import triattn_sm80

            return triattn_sm80.forward(self, pair, mask) if route == "sm80" else triattn_sm80.forward_train(self, pair, mask)
        if route == "sm90":
            from miniworld_engine.kernels.integrations import triattn_h100_sm90

            return triattn_h100_sm90.forward(self, pair, mask)
        from miniworld_engine.kernels.triangle_attention.cuda import h100_dh

        return h100_dh.forward(self, pair, mask)

    def _supports_triton(self, pair: torch.Tensor, mask: torch.Tensor | None = None) -> str | None:
        if not pair.is_cuda:
            return "the Triton kernels run on CUDA tensors only"
        if not self.use_self_attention:
            return self._bias_only_triton_refusal(pair, mask)
        why = self._core_refusal(pair, mask)
        if why is not None:
            return why
        if pair.dtype != torch.bfloat16:
            return f"dtype {pair.dtype}: the Triton triangle-attention core is bf16 only"
        return None

    def _forward_triton(self, pair: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        with master_autocast(pair, self):  # fp32 master matrices under bf16 activations: the projections run bf16, as the reference's do
            if self._rows_blocked(pair):
                return self._attention_rows(pair, mask, _triton_core)
            return self._residual(pair, mask, _triton_core if self.use_self_attention else self._bias_only_triton)

    #: An inference pair larger than this runs :meth:`_attention_rows` (the Triton core; FoldForge OpenDDE's refiner: d_pair 384 at 2976 /
    #: 3968 structural tokens, 6.3 / 11.3 GiB); requirement sizes up to L2048 (3 GiB at d_pair 384) keep the whole-pair composition.
    ROWS_BLOCKED_BYTES = 4 << 30
    #: The rows of one block of :meth:`_attention_rows`: about this many bytes of the attended frame per pair-shaped temporary.
    ROWS_BLOCK_BYTES = 1 << 30

    def _rows_blocked(self, pair: torch.Tensor) -> bool:
        """Whether the call runs :meth:`_attention_rows`: inference (no autograd, no live dropout), the self-attention update of
        :meth:`_attention` (not the fp32-softmax variants, not a dead instance), a pair past ``ROWS_BLOCKED_BYTES``."""
        return (
            self.use_self_attention
            and not isinstance(self, _TriangleAttentionFp32Softmax)
            and not self.dead
            and not (self.training and self.p_drop > 0.0)
            and not (torch.is_grad_enabled() and (pair.requires_grad or any(p.requires_grad for p in self.parameters())))
            and pair.numel() * pair.element_size() > self.ROWS_BLOCKED_BYTES
        )

    def _attention_rows(self, pair: torch.Tensor, mask: torch.Tensor | None, core) -> torch.Tensor:
        """``pair + update(pair)`` of :meth:`_attention` (with ``core``) in blocks of the attended frame's rows i: the rows are
        independent given the bias b_jk, which is made first (block by block, [B, H, L, L], the masked keys filled once), then each block
        normalises its rows, projects q / k / v / g, runs the core against every key, gates, projects out and adds its residual into its
        rows of the output. No pair-sized temporary: the whole-pair composition holds the normalised pair, q, k, v, the gate, the core
        output and the update (7 pair sizes; 79 GiB at 3968 tokens, d_pair 384), this holds one block of each. The same statements per
        row (a projection GEMM over fewer rows may pick another cuBLAS kernel)."""
        B, L, _, C = pair.shape
        H = self.n_head
        frame = pair if self.starting else pair.transpose(1, 2)
        out = torch.empty_like(pair)
        frame_out = out if self.starting else out.transpose(1, 2)
        rows = max(1, min(L, self.ROWS_BLOCK_BYTES // (L * C * pair.element_size())))
        bias = pair.new_empty((B, H, L, L))
        for i0 in range(0, L, rows):
            x = self.ln_pair(frame[:, i0 : i0 + rows].contiguous())
            bias[:, :, i0 : i0 + rows] = self.to_bias(x).permute(0, 3, 1, 2)
        if self._bias_untransposed:
            bias = bias.transpose(-1, -2).contiguous()
        if mask is not None:
            bias.masked_fill_(~mask.bool()[:, None, None, :], torch.finfo(bias.dtype).min)  # the cores' own fill, once
        for i0 in range(0, L, rows):
            x = self.ln_pair(frame[:, i0 : i0 + rows].contiguous())
            n = x.shape[1]

            def heads(t, n=n):  # [B, n, L, H * d] -> [B, n, H, L, d]
                return t.view(B, n, L, H, -1).transpose(2, 3)

            q, k, v = heads(self.to_query(x)), heads(self.to_key(x)), heads(self.to_value(x))
            if self.use_qk_norm:
                q, k = self.norm_query(q), self.norm_key(k)
            o = core(q, k, v, bias[:, None], None).transpose(2, 3).reshape(B, n, L, -1)
            del q, k, v
            frame_out[:, i0 : i0 + n] = frame[:, i0 : i0 + n] + self.to_out(sigmoid_gate(self.to_gate(x), o))
            del x, o
        return out

    # The bias-only form (``use_self_attention=False``; engine 2.4's ``_bias_only_inference`` and its portable training path): the attention
    # weights ``softmax_k(b_jk)`` do not depend on the row i, so the core is one batched GEMM per (batch, head). Triton: the LayerNorm fused
    # into ONE projection over the concatenated [W_v | W_b | W_g] (``layernorm_linear``: the normalised pair never reaches memory; trainable:
    # its backward is cuBLAS + the Triton LayerNorm backward), the softmax and the GEMM as the reference's statements, then the gate folded
    # into the out-projection GEMM (``bias_only_attention/triton/gate_out.fused_gate_out``; at a hidden width above 128 the one-pass sigmoid
    # gate + a cuBLAS out-projection). bf16 / fp32 activations, d_pair <= 1024, a bool [B, L] key mask or none, no projection biases.
    _GATE_FUSED_MAX_DH = 128

    def _bias_only_triton_refusal(self, pair: torch.Tensor, mask: torch.Tensor | None) -> str | None:
        if pair.dim() != 4 or pair.shape[1] != pair.shape[2]:
            return f"pair must be [B, L, L, d_pair], got {tuple(pair.shape)}"
        if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != tuple(pair.shape[:2])):
            return "mask must be a bool [B, L] key mask"
        if pair.dtype not in (torch.bfloat16, torch.float32):
            return f"dtype {pair.dtype} (bfloat16 / float32)"
        if pair.shape[-1] > 1024:
            return f"d_pair {pair.shape[-1]} > 1024 (the fused LayerNorm + projection kernel)"
        if self.to_gate.bias is not None or self.to_out.bias is not None or self.dual_output or self.untransposed_output:
            return "gate / out-projection biases or a second out-projection (the bias-only Triton path has neither)"
        return None

    def _bias_only_triton(self, pair: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """The bias-only update (no residual) of the RAW pair, as :meth:`_attention` computes it."""
        from miniworld_engine.kernels.bias_only_attention.triton.gate_out import fused_gate_out
        from miniworld_engine.kernels.gated_projection.triton.main import sigmoid_gate_fused
        from miniworld_engine.kernels.layernorm_linear.autograd import LayerNormLinearTritonFn
        from miniworld_engine.kernels.layernorm_linear.triton.fused import layernorm_linear_triton_fwd

        x = pair if self.starting else rearrange(pair, "B I J D -> B J I D").contiguous()
        L = x.shape[1]
        dt, H = x.dtype, self.n_head
        dv = self.to_value.weight.shape[0]
        w_cat = torch.cat([self.to_value.weight, self.to_bias.weight, self.to_gate.weight], 0).to(dt)
        grad = torch.is_grad_enabled() and (pair.requires_grad or any(p.requires_grad for p in self.parameters()))
        ln = self.ln_pair
        with torch.autocast("cuda", enabled=False):
            if grad:
                proj = LayerNormLinearTritonFn.apply(x, ln.weight, ln.bias, w_cat, None, ln.eps, L)
            else:
                proj = layernorm_linear_triton_fwd(x, ln.weight, ln.bias, w_cat, None, ln.eps)
            value, bias, gate = proj.split([dv, H, dv], dim=-1)
            value = rearrange(value, "B L L2 (H D) -> B H L L2 D", H=H)
            bias = rearrange(bias, "B L L2 H -> B H L L2")
            if self._bias_untransposed:
                bias = bias.transpose(-1, -2)
            if mask is not None:
                bias = bias.masked_fill(~mask[:, None, None, :], torch.finfo(bias.dtype).min)
            attention = torch.softmax(bias.float(), dim=-1).to(dt)  # independent of i: one GEMM per (b, h)
            out = rearrange(torch.einsum("bhjk,bhikd->bhijd", attention, value), "B H L L2 D -> B L L2 (H D)")
            wo = self.to_out.weight.to(dt)
            if dv <= self._GATE_FUSED_MAX_DH:
                out = fused_gate_out(gate, out, wo)
            else:
                out = F.linear(sigmoid_gate_fused(gate, out.contiguous()), wo)
        return out if self.starting else rearrange(out, "B J I D -> B I J D").contiguous()

    def _attention_core(self, x: torch.Tensor, mask: torch.Tensor | None, core) -> torch.Tensor:
        """``_attention``'s q / k / v / bias of the normalised ``x`` (already in the attended frame) through ``core`` -> ``[B, L, L, H * d]``."""
        B, L, L2, _ = x.shape

        def heads(t):  # [B, L, L2, H * d] -> [B, L (rows i), H, L2 (queries j), d]: the library layout
            return t.view(B, L, L2, self.n_head, -1).transpose(2, 3)

        query, key, value = heads(self.to_query(x)), heads(self.to_key(x)), heads(self.to_value(x))
        if self.use_qk_norm:
            query, key = self.norm_query(query), self.norm_key(key)
        bias = self.to_bias(x).permute(0, 3, 1, 2)  # [B, H, j, k]
        if self._bias_untransposed:
            bias = bias.transpose(-1, -2)
        out = core(query, key, value, bias[:, None], mask)  # [B, i, H, j, d]
        return out.transpose(2, 3).reshape(B, L, L2, -1)

    def _attention(self, pair: torch.Tensor, mask: torch.Tensor | None = None, core=None) -> torch.Tensor:
        """The update of :class:`TriangleAttention` / :class:`TriangleAttentionAF3` (no residual / dropout): products and softmax in the
        activation dtype, masked keys at the dtype's min. The ending node attends over the transposed pair."""
        if core is not None and not self.use_self_attention:
            return core(pair, mask)  # the bias-only update on the raw pair: its own frame, the LayerNorm fused into the projection
        if not self.starting:
            pair = rearrange(pair, "B I J D -> B J I D").contiguous()

        pair = self.ln_pair(pair)
        if core is not None:
            out = self._attention_core(pair, mask, core)
            return self._attention_out(pair, out)
        value = self.to_value(pair)
        bias = self.to_bias(pair)
        value = rearrange(value, "B L L2 (H D) -> B H L L2 D", H=self.n_head)
        bias = rearrange(bias, "B L L2 H -> B H L L2")
        if self._bias_untransposed:
            # AF3: the bias of the pair as it is, read in the transposed frame (see ``transposed_bias``).
            bias = bias.transpose(-1, -2).contiguous()
        if mask is not None:
            bias = bias.masked_fill(~mask[:, None, None, :], torch.finfo(bias.dtype).min)

        if self.use_self_attention:
            query = self.to_query(pair)
            key = self.to_key(pair)
            query = rearrange(query, "B L L2 (H D) -> B H L L2 D", H=self.n_head)
            key = rearrange(key, "B L L2 (H D) -> B H L L2 D", H=self.n_head)
            if self.use_qk_norm:
                query = self.norm_query(query)
                key = self.norm_key(key)
            query.mul_(query.shape[-1] ** -0.5)
            attention = torch.einsum("bhijd,bhikd->bhijk", query, key)
            attention = attention + bias[:, :, None, :, :]
            attention = F.softmax(attention, dim=-1)
            out = torch.einsum("bhijk,bhikd->bhijd", attention, value)
        else:
            # softmax(bias) is independent of i: one GEMM per (b, h)
            attention = F.softmax(bias, dim=-1)
            out = torch.einsum("bhjk,bhikd->bhijd", attention, value)

        out = rearrange(out, "B H L L2 D -> B L L2 (H D)")
        return self._attention_out(pair, out)

    def _attention_out(self, pair: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """The gate and out-projection of :meth:`_attention` (``pair`` the normalised input in the attended frame), laid back."""
        gate = self.to_gate(pair)
        out = sigmoid_gate(gate, out)
        out = self.to_out(out)

        if not self.starting:
            out = rearrange(out, "B J I D -> B I J D").contiguous()
        return out

    def _attention_fp32_softmax(self, pair: torch.Tensor, mask: torch.Tensor | None = None, core=None) -> torch.Tensor:
        """The update of the RoseTTAFold3 / Chai-1 variants (no residual / dropout): products in the parameters' dtype, logits / softmax /
        PV in fp32, masked keys at -inf (a fully masked row gives 0). The ending node attends over the transposed pair; its bias comes from
        the transposed pair (``transposed_bias``) or from the pair as it is; the output is laid back as the variant has it."""
        B, L, _, _ = pair.shape
        H = self.n_head
        x = self.ln_pair(pair)
        xt = x if self.starting else x.transpose(1, 2)
        bias = self.to_bias(x if (self.starting or not self.transposed_bias) else xt).permute(0, 3, 1, 2)  # [B, H, L, L]
        q, k, v = (proj(xt).view(B, L, L, H, -1) for proj in (self.to_query, self.to_key, self.to_value))
        if core is not None:  # the library layout [B, i, H, j, d]; the kernel's logits / softmax are fp32
            o = core(q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3), bias[:, None], mask).transpose(2, 3).reshape(B, L, L, -1)
            o = o.to(pair.dtype)
        else:
            d = q.shape[-1]
            acc = wide(pair)
            logits = torch.einsum("bijhd,bikhd->bihjk", q.to(acc), k.to(acc)) * d**-0.5 + bias.to(acc)[:, None]
            if mask is not None:
                logits = logits.masked_fill(~mask.bool()[:, None, None, None, :], float("-inf"))
            o = torch.einsum("bihjk,bikhd->bijhd", torch.nan_to_num(logits.softmax(-1)), v.to(acc)).reshape(B, L, L, -1).to(pair.dtype)
        u = torch.sigmoid(self.to_gate(xt)) * o
        out = self.to_out(u)
        if self.dual_output:
            other = self.to_out_t(u)
            return out + other.transpose(1, 2) if self.starting else out.transpose(1, 2) + other
        return out if self.starting or self.untransposed_output else out.transpose(1, 2)


class TriangleAttention(_TriangleAttentionBase):
    """Triangular gated self-attention (AF3 Algorithms 14 / 15) on the supplementary-information / OpenFold bias orientation::

        x = LN(pair)            (ending node: LN(pair^T))
        o = softmax_k(q_ij . k_ik / sqrt(d) + b_jk) v_ik      per head; masked keys at the dtype's min
        return pair + drop(W_o (sigmoid(W_g x) * o))          (ending node: the update transposed back)

    Models: MiniWorld, Boltz-2, Protenix, OpenDDE, ESMFold2. ``TriangleAttention(..., model=name)`` builds the class of ``model`` instead
    (:data:`TRIANGLE_ATTENTION_MODELS`), with the remaining arguments.

    Parameters
    ----------
    d_pair : int
        Dimension of pair representation.
    n_head : int
        Number of attention heads.
    d_hidden : int | None
        Total hidden dimension for QKV projections. Defaults to ``d_pair`` when *None*. Must be divisible by ``n_head``.
    starting : bool
        Whether the attention is around the "starting" node.
    use_self_attention : bool
        Whether to use self-attention (``False``: the attention weights are ``softmax(bias)`` alone, no query / key).
    use_qk_norm : bool
        Whether to apply RMSNorm to query and key projections.
    p_drop : float
        Dropout of the update (training only): drop_row (broadcast over i) for the starting node, drop_col (over j) for the ending node.
    master_weights : bool
        Keep fp32 master parameters under a bf16 cast (:class:`MasterWeights`).
    model : str | None
        Select a model's variant (see above); ``None`` is this class.
    backend : str
        See :class:`BackendModule`.

    """

    version = 2  # human-managed: bump when a change alters this class's accuracy or speed results
    #: 2: (2026-11-13) the H100 d_pair 128 x 4 heads inference gate in fp32 inside the tail (backend_versions cuda:h100_sm90 4)

    _anthropic_block = True

    def __new__(cls, *args, model: str | None = None, **kwargs):
        target = cls if model is None else _select(TRIANGLE_ATTENTION_MODELS, "triangle attention", model)
        if cls is TriangleAttention and target is not TriangleAttention:
            return target(*args, **kwargs)  # not a TriangleAttention: Python does not call __init__ on it again
        return super().__new__(cls)

    def __init__(
        self,
        d_pair: int = 128,
        n_head: int = 4,
        *,
        d_hidden: int | None = None,
        starting: bool = True,
        use_self_attention: bool = True,
        use_qk_norm: bool = False,
        p_drop: float = 0.25,
        master_weights: bool = False,
        model: str | None = None,
        backend: str = "pytorch",
    ) -> None:
        super().__init__(
            d_pair,
            n_head,
            d_hidden=d_hidden,
            starting=starting,
            use_self_attention=use_self_attention,
            use_qk_norm=use_qk_norm,
            p_drop=p_drop,
            gate_bias=False,
            out_bias=False,
            master_weights=master_weights,
            backend=backend,
        )

    def _update(self, pair: torch.Tensor, mask: torch.Tensor | None = None, core=None) -> torch.Tensor:
        return self._attention(pair, mask, core)


class TriangleAttentionAF3(_TriangleAttentionBase):
    """:class:`TriangleAttention` on AlphaFold 3's released bias orientation: the ending node projects its pair bias from the pair as it is
    and reads it in the transposed frame (the transpose of :class:`TriangleAttention`'s bias). The starting node is the same as
    :class:`TriangleAttention`'s. Models: AlphaFold 3, OpenFold3 v0.5 (OpenBind; preview-2 is :class:`TriangleAttention`), IntelliFold2.

    Parameters as :class:`TriangleAttention` (always self-attention, no q / k norm).
    """

    version = 3  # human-managed: bump when a change alters this class's accuracy or speed results
    #: 2026-11-12: the H100 inference core of 64-channel heads (h100_dh) on a 128-query tile from L 384 (outputs bitwise unchanged); 3: the
    #: training dropout scale by the H100 Philox kernel (as the base class)
    backend_versions: ClassVar[dict[str, int]] = {**_TriangleAttentionBase.backend_versions, "cuda:h100_sm90": 5}  # 5: as the base class

    transposed_bias = False
    _anthropic_block = True

    def __init__(
        self,
        d_pair: int = 128,
        n_head: int = 4,
        *,
        d_hidden: int | None = None,
        starting: bool = True,
        p_drop: float = 0.25,
        master_weights: bool = False,
        backend: str = "pytorch",
    ) -> None:
        super().__init__(
            d_pair,
            n_head,
            d_hidden=d_hidden,
            starting=starting,
            use_self_attention=True,
            use_qk_norm=False,
            p_drop=p_drop,
            gate_bias=False,
            out_bias=False,
            master_weights=master_weights,
            backend=backend,
        )

    def _update(self, pair: torch.Tensor, mask: torch.Tensor | None = None, core=None) -> torch.Tensor:
        return self._attention(pair, mask, core)


class _TriangleAttentionFp32Softmax(_TriangleAttentionBase):
    """The RoseTTAFold3 / Chai-1 variants: AlphaFold 3's bias orientation, the softmax in fp32 (:meth:`_attention_fp32_softmax`)."""

    transposed_bias = False
    #: 2026-11-12: the H100 inference core of 64-channel heads (h100_dh) on a 128-query tile from L 384 (outputs bitwise unchanged); 3: the
    #: training dropout scale by the H100 Philox kernel (as the base class)
    #: 4: as the base class (RF3 at d_pair 128); 5: as the base class (RF3 at d_pair 128: the ending node's tail, bitwise the same)
    backend_versions: ClassVar[dict[str, int]] = {**_TriangleAttentionBase.backend_versions, "cuda:h100_sm90": 5}
    #: RoseTTAFold3: the gate and the out-projection carry a bias.
    _projection_biases: bool = False

    def __init__(
        self,
        d_pair: int = 128,
        n_head: int = 4,
        *,
        d_hidden: int | None = None,
        starting: bool = True,
        p_drop: float = 0.25,
        master_weights: bool = False,
        backend: str = "pytorch",
    ) -> None:
        super().__init__(
            d_pair,
            n_head,
            d_hidden=d_hidden,
            starting=starting,
            use_self_attention=True,
            use_qk_norm=False,
            p_drop=p_drop,
            gate_bias=self._projection_biases,
            out_bias=self._projection_biases,
            master_weights=master_weights,
            backend=backend,
        )

    def _update(self, pair: torch.Tensor, mask: torch.Tensor | None = None, core=None) -> torch.Tensor:
        return self._attention_fp32_softmax(pair, mask, core)


class TriangleAttentionRF3(_TriangleAttentionFp32Softmax):
    """RoseTTAFold3's triangle attention: :class:`TriangleAttentionAF3`'s bias orientation, the gate and the out-projection with a bias,
    logits / softmax / PV in fp32 (masked keys -inf)::

        o = softmax_k(q_ij . k_ik / sqrt(d) + b_jk) v_ik                       (fp32)
        return pair + drop(W_o (sigmoid(W_g x + b_g) * o) + b_o)

    Parameters as :class:`TriangleAttentionAF3`.
    """

    version = 3  # human-managed: bump when a change alters this class's accuracy or speed results
    #: 3: (2026-11-13) the H100 d_pair 128 x 4 heads inference gate in fp32 inside the tail

    _projection_biases = True

    # ------------------------------------------------------------------------------------------------------------- anthropic
    # The rosettafold3 kit's triangle attention (``fpf_rf3_adapter._tri_forward_v2``, arm gflash): ``lnl_fused.ln_linear`` (LN + the bias
    # projection in one kernel, the normalised pair written in the attention frame), ONE q | k | v | g GEMM (the gate bias only), the core
    # provider's row (the module's key mask handed to it: RF3's stock call has none), ``gate_transpose`` (sigmoid(g) * o, laid back for the
    # ending node) and the module's ``to_out`` (with its bias). C 64 / 128 (the kernel's widths); elsewhere the base path (core only).
    def _anthropic_rf3_ok(self, pair: torch.Tensor) -> bool:
        from miniworld_engine.modules import _anthropic

        return pair.shape[-1] in (64, 128) and pair.dtype == torch.bfloat16 and _anthropic.missing("opt_core.kernels.lnl_fused") is None

    def _forward_anthropic(self, pair: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.dead or not self._anthropic_rf3_ok(pair):
            return super()._forward_anthropic(pair, mask)
        import torch.nn.functional as Fn

        from miniworld_engine.modules import _anthropic

        rfu = _anthropic.load("opt_core.kernels.lnl_fused")
        prov, (sel, _why) = self._anthropic_select(pair)
        stock = None
        if sel.row in prov.NEEDS_STOCK:
            from miniworld_engine.modules import _cueq

            stock = _cueq.torch_api().triangle_attention
        key = tuple((p.data_ptr(), p._version, p.dtype) for p in self.parameters())
        if getattr(self, "_anthropic_rf3_key", None) != key:  # the kit's _tri_weights
            with torch.no_grad():
                bf = torch.bfloat16
                wcat = (
                    torch.cat([self.to_query.weight, self.to_key.weight, self.to_value.weight, self.to_gate.weight], 0).to(bf).contiguous()
                )
                nq = self.to_query.weight.shape[0]
                bcat = torch.zeros(wcat.shape[0], device=wcat.device, dtype=bf)
                bcat[3 * nq :] = self.to_gate.bias.to(bf)
                wb = torch.zeros((max(16, 1 << (self.n_head - 1).bit_length()), wcat.shape[1]), device=wcat.device, dtype=bf)
                wb[: self.n_head] = self.to_bias.weight.to(bf)
                self._anthropic_rf3 = {
                    "Wcat": wcat,
                    "bcat": bcat,
                    "nq": nq,
                    "Wb": wb,
                    "lnw": self.ln_pair.weight.float().contiguous(),
                    "lnb": self.ln_pair.bias.float().contiguous(),
                }
            self._anthropic_rf3_key = key
        W = self._anthropic_rf3
        B, L, _, _ = pair.shape
        H, nq = self.n_head, W["nq"]
        with master_autocast(pair, self):
            y16, b16 = rfu.ln_linear(
                pair.contiguous(), W["lnw"], W["lnb"], W["Wb"], eps=self.ln_pair.eps, write_y=True, transpose=not self.starting
            )
            bias = (
                b16[..., :H].permute(0, 3, 1, 2)[:, None].float().contiguous()
            )  # [B, 1, H, i, j]: from LN(pair) as it is (AF3's orientation)
            qkvg = Fn.linear(y16, W["Wcat"], W["bcat"]).contiguous()  # [B, i', j', 4 nq] in the attention frame
            q, k, v = (qkvg[..., s * nq : (s + 1) * nq].view(B, L, L, H, -1).transpose(2, 3).contiguous() for s in range(3))
            out = prov.triangle_attention(
                q, k, v, bias, _key_mask5(mask, q), q.shape[-1] ** -0.5, word=sel.word, selection=sel, stock=stock
            )
            if out.is_contiguous() and out.dtype == torch.bfloat16:
                y = rfu.gate_transpose(out, qkvg, 3 * nq, transpose=not self.starting)
            else:
                y = torch.sigmoid(qkvg[..., 3 * nq :]) * out.transpose(2, 3).reshape(B, L, L, -1).to(qkvg.dtype)
                y = y if self.starting else y.transpose(1, 2)
            return pair + self.to_out(y).to(pair.dtype)


class TriangleAttentionChai1(_TriangleAttentionFp32Softmax):
    """Chai-1's trunk / template triangle attention: :class:`TriangleAttentionAF3`'s bias orientation, the softmax in fp32 (masked keys
    -inf), and the ending node returns its update as the transposed pair has it (not transposed back).

    Parameters as :class:`TriangleAttentionAF3`.
    """

    version = 2  # human-managed: bump when a change alters this class's accuracy or speed results
    #: cuda:h100_sm90 stays 3: no Chai-1 shape takes the d_pair 128 x 4 heads or the d_pair 256 x 8 heads path
    backend_versions: ClassVar[dict[str, int]] = {**_TriangleAttentionFp32Softmax.backend_versions, "cuda:h100_sm90": 3}

    untransposed_output = True


class TriangleAttentionChai1Confidence(_TriangleAttentionFp32Softmax):
    """Chai-1's confidence-head triangle attention: as :class:`TriangleAttentionChai1` with a second out-projection ``to_out_t`` whose update
    lands transposed (starting: ``W_o u + (W_t u)^T``; ending: ``(W_o u)^T + W_t u``).

    Parameters as :class:`TriangleAttentionAF3`.
    """

    version = 2  # human-managed: bump when a change alters this class's accuracy or speed results
    #: cuda:h100_sm90 stays 3 (as TriangleAttentionChai1)
    backend_versions: ClassVar[dict[str, int]] = {**_TriangleAttentionFp32Softmax.backend_versions, "cuda:h100_sm90": 3}

    dual_output = True


#: ``TriangleAttention(model=name)`` -> the class it builds.
TRIANGLE_ATTENTION_MODELS: dict[str, type[_TriangleAttentionBase]] = {
    **dict.fromkeys(("miniworld", "boltz2", "protenix", "protenix_v2", "opendde", "esmfold2", "openfold3_preview2"), TriangleAttention),
    **dict.fromkeys(("af3", "openfold3", "intellifold2"), TriangleAttentionAF3),
    "rf3": TriangleAttentionRF3,
    "chai1": TriangleAttentionChai1,
    "chai1_confidence": TriangleAttentionChai1Confidence,
}


class BidirectionalTriangleAttention(BackendModule, MasterWeights):
    """Triangular gated attention computing starting + ending in one block; returns ``pair + update``.

    Mirrors :class:`BidirectionalTriangleMultiplication`: one shared input LayerNorm, the value / bias (and query / key) projections doubled
    so the channels split in half -- the first half computes the **starting** attention, the second half the **ending** attention -- then
    the two are concatenated to ``2 * d_hidden`` and projected down to ``d_pair``.

    Bias-only (``use_self_attention=False``, the default): the attention weights come only from the bias projection. The two directions
    are different contractions of the *same* shape (no input transpose)::

        starting:  out_s[i,j,d] = sum_k softmax_k(bias_s[j,k]) * value_s[i,k,d]   (contract axis 2)
        ending:    out_e[i,j,d] = sum_k softmax_k(bias_e[k,i]) * value_e[k,j,d]   (contract axis 1)

    The gate follows :class:`TriangleAttention` (on the hidden-width output, applied *before* ``to_out``), not the trimul convention.
    Model: MiniWorld.

    ``"cuda"`` (H100 sm_90a and A100 sm_80, bias-only, bf16 activations; forward and backward): ``kernels/integrations/bidir_triattn_h100`` -- each
    direction the TriangleAttention family's node form in its own frame (the ending node on the transposed pair), the family's glue
    LayerNorm / gate kernels around cuBLAS GEMMs over channel planes (the softmax formed once per head: it does not depend on the row).
    """

    version = 1  # human-managed: bump when a change alters this class's accuracy or speed results
    backend_versions: ClassVar[dict[str, int]] = {"cuda:a100_sm80": 2}  # 2026-11-10: the sm_80 build of the H100 path

    backends = ("pytorch", "cuda")

    def _supports_cuda(self, pair: torch.Tensor, mask: torch.Tensor | None = None) -> str | None:
        from miniworld_engine.kernels.integrations import bidir_triattn_h100

        return bidir_triattn_h100.refusal(self, pair, mask)

    def _forward_cuda(self, pair: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        from miniworld_engine.kernels.integrations import bidir_triattn_h100

        return bidir_triattn_h100.forward(self, pair, mask)

    def __init__(
        self,
        d_pair: int = 128,
        n_head: int = 4,
        *,
        d_hidden: int | None = None,
        use_self_attention: bool = False,
        master_weights: bool = False,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.d_pair = d_pair
        self.d_hidden = d_hidden if d_hidden is not None else d_pair
        self.use_self_attention = use_self_attention

        if self.d_hidden % n_head != 0:
            msg = f"d_hidden ({self.d_hidden}) must be divisible by n_head ({n_head})"
            raise ValueError(msg)

        d2 = 2 * self.d_hidden

        self.ln_pair = LayerNorm(d_pair)
        # Doubled-width value/bias projections: [starting | ending] channels.
        self.to_value = Linear(d_pair, d2, bias=False, init="glorot")
        self.to_bias = Linear(d_pair, 2 * n_head, bias=False, init="default")
        self.to_gate = Linear(d_pair, d2, bias=False, init="gating")
        self.to_out = Linear(d2, d_pair, bias=False, init="zero")
        if use_self_attention:
            # Doubled-width Q/K projections: [starting | ending] channels, like value.
            self.to_query = Linear(d_pair, d2, bias=False, init="glorot")
            self.to_key = Linear(d_pair, d2, bias=False, init="glorot")
        if master_weights:
            self.set_master_weights()
        self.set_backend(backend)

    def _attend(self, value: torch.Tensor, bias: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Bias-only: the two direction einsums on the split channels. value ``[B, L, L, 2*d_hidden]``, bias ``[B, L, L, 2*n_head]`` ->
        ``[B, L, L, 2*d_hidden]``."""
        H = self.n_head
        value = rearrange(value, "B L L2 (G D) -> B G L L2 D", G=2 * H)
        bias = rearrange(bias, "B L L2 G -> B G L L2", G=2 * H)
        value_s, value_e = value[:, :H], value[:, H:]
        bias_s, bias_e = bias[:, :H], bias[:, H:]

        if mask is not None:
            bias_s = bias_s.masked_fill(~mask[:, None, None, :], float("-inf"))
            bias_e = bias_e.masked_fill(~mask[:, None, :, None], float("-inf"))

        out_s = torch.einsum("bhjk,bhikd->bhijd", F.softmax(bias_s, dim=-1), value_s)
        out_e = torch.einsum("bhki,bhkjd->bhijd", F.softmax(bias_e, dim=-2), value_e)
        out = torch.cat([out_s, out_e], dim=1)  # [B, 2H, L, L, D]
        return rearrange(out, "B G L L2 D -> B L L2 (G D)")

    @staticmethod
    def _attn_one(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """One-direction (starting-frame) triangle self-attention: q, k, v ``[B, H, L, L, D]``, bias ``[B, H, L, L]`` -> ``[B, H, L, L, D]``."""
        scale = q.shape[-1] ** -0.5
        attn = torch.einsum("bhijd,bhikd->bhijk", q * scale, k)
        attn = attn + bias[:, :, None, :, :]
        attn = F.softmax(attn, dim=-1)
        return torch.einsum("bhijk,bhikd->bhijd", attn, v)

    def _attend_sa(self, pln: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Self-attention (Q/K/V) on the split channels -> ``[B, L, L, 2*d_hidden]``. The ENDING direction is the transpose-dual of
        STARTING: the same single-direction attention on the (i<->j)-transposed ending-half tensors, the result transposed back."""
        H = self.n_head
        q = rearrange(self.to_query(pln), "B L L2 (G D) -> B G L L2 D", G=2 * H)
        k = rearrange(self.to_key(pln), "B L L2 (G D) -> B G L L2 D", G=2 * H)
        v = rearrange(self.to_value(pln), "B L L2 (G D) -> B G L L2 D", G=2 * H)
        bias = rearrange(self.to_bias(pln), "B L L2 G -> B G L L2", G=2 * H)

        # starting
        qs, ks, vs, bs = q[:, :H], k[:, :H], v[:, :H], bias[:, :H]
        if mask is not None:
            bs = bs.masked_fill(~mask[:, None, None, :], float("-inf"))
        out_s = self._attn_one(qs, ks, vs, bs)

        # ending: (i<->j)-transpose the projected tensors -> starting frame, attend, transpose back
        qe = q[:, H:].transpose(2, 3)
        ke = k[:, H:].transpose(2, 3)
        ve = v[:, H:].transpose(2, 3)
        be = bias[:, H:].transpose(2, 3)
        if mask is not None:
            be = be.masked_fill(~mask[:, None, None, :], float("-inf"))
        out_e = self._attn_one(qe, ke, ve, be).transpose(2, 3)

        out = torch.cat([out_s, out_e], dim=1)  # [B, 2H, L, L, D]
        return rearrange(out, "B G L L2 D -> B L L2 (G D)")

    def _forward_pytorch(self, pair: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``pair`` ``[B, L, L, d_pair]``, ``mask`` ``[B, L]`` (bool). ALWAYS returns the residual output ``pair + bidir_attention(pair)``."""
        with master_autocast(pair, self):
            pln = self.ln_pair(pair)
            out = self._attend_sa(pln, mask) if self.use_self_attention else self._attend(self.to_value(pln), self.to_bias(pln), mask)
            return pair + self.to_out(sigmoid_gate(self.to_gate(pln), out))

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        return pair_inputs(self.d_pair, length=length, samples=samples, dtype=dtype, device=device, generator=generator)


# ======================================================================================================= triangle multiplicative update
def _trimul_triton_refusal(
    module, pair: torch.Tensor, mask: torch.Tensor | None, d_hidden: int, *, bidirectional: bool = False
) -> str | None:
    """The range of the portable Triton TriMul (unidirectional and bidirectional): CUDA, bf16 activations (the matrices are cast to the pair's dtype at the call: fp32 masters take their gradient in fp32),
    a square pair, a bool [B, L] mask, d_hidden != d_pair only within ``contract.hidden_refusal``."""
    if not pair.is_cuda:
        return "the Triton kernels run on CUDA tensors only"
    if pair.dtype != torch.bfloat16:
        return f"dtype {pair.dtype}: the engine's TriMul kernels are bf16 only (no fp32 TriMul)"
    if pair.dim() != 4 or pair.shape[1] != pair.shape[2]:
        return f"pair must be [B, L, L, d_pair], got {tuple(pair.shape)}"
    if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != tuple(pair.shape[:2])):
        return "mask must be a bool [B, L] residue mask"
    # Both pipelines index in 64 bits above 2**31 elements: the training one's LN_out kernels over the m-major (L * L, width) contraction
    # view (layernorm_linear te_style / mmajor_bwd / layernorm persistent) take a WIDE constexpr there. Unidirectional training past the
    # bound is tested (tests/gpu/test_trimul_big_plane.py, L 4200 d_hidden 128); the bidirectional one cannot be (the smallest plane past
    # it, width 2 * 128 at L 3000, needs more than 80 GB in training), so it keeps the refusal there.
    width = 2 * d_hidden
    if bidirectional and (torch.is_grad_enabled() or module.training) and pair.shape[1] * pair.shape[2] * width >= 2**31:
        return (
            f"training (grad enabled or train mode) at L {pair.shape[1]}, width {width}: the bidirectional Triton TriMul training is "
            "untested past L * L * width = 2**31 elements (no plane that large fits an 80 GB card in training; inference: no bound)"
        )
    from miniworld_engine.kernels.trimul_inproj.triton.contract import hidden_refusal

    return hidden_refusal(pair.shape[-1], d_hidden)


def _trimul_cuda_route(module, pair, mask, dropout_p, *, bidirectional: bool) -> tuple[str | None, str | None]:
    """``("infer" | "train", None)``: the H100 TriMul path (``kernels/integrations/trimul_h100``) that serves the call, else
    ``(None, reason)``. bf16 only (the H100 TriMul policy: no fp32 TriMul kernel). Shapes / dtypes / flags only."""
    if not pair.is_cuda:
        return None, "the kernels run on CUDA tensors only"
    cap = torch.cuda.get_device_capability(pair.device)
    if cap == (8, 0):
        return _trimul_cuda_route_sm80(module, pair, mask)
    if cap != (9, 0):
        return None, "the cuda backend is the H100 (sm_90a) / A100 (sm_80) kernels"
    if pair.dtype != torch.bfloat16:
        return None, f"dtype {pair.dtype}: the H100 TriMul kernels are bf16 only (no fp32 TriMul, by policy)"
    if pair.dim() != 4 or pair.shape[1] != pair.shape[2]:
        return None, f"pair must be [B, L, L, d_pair], got {tuple(pair.shape)}"
    if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != tuple(pair.shape[:2])):
        return None, "mask must be a bool [B, L] residue mask"
    from miniworld_engine.kernels.integrations import trimul_h100

    dropout_p = module.p_drop if dropout_p is None else dropout_p
    live_drop = bool(dropout_p) and dropout_p > 0.0 and module.training
    if not torch.is_grad_enabled() and not live_drop and trimul_h100.serves_inference(module, pair, bidirectional=bidirectional):
        return "infer", None
    train_ok = trimul_h100.serves(module, pair) if bidirectional else trimul_h100.serves_single(module, pair)
    if train_ok:
        return "train", None
    return None, (
        f"d_pair {pair.shape[-1]}, d_hidden {module.d_hidden}, L {pair.shape[1]}, grad {torch.is_grad_enabled()}, dropout "
        f"{live_drop}: outside the H100 TriMul contract (trimul_h100.serves_inference / serves"
        f"{'' if bidirectional else '_single'}: tile table, lengths, eps 1e-5, fp32 LN affine)"
    )


def _trimul_cuda_route_sm80(module, pair, mask) -> tuple[str | None, str | None]:
    """``("sm80", None)``: the A100 TriMul (``kernels/integrations/trimul_sm80``: the D128 fused set, the wide set for D 64 / 256 / 384 / 512,
    inference and training, one plane per launch), else ``(None, reason)``. bf16 only (no fp32 TriMul, by policy)."""
    if pair.dtype != torch.bfloat16:
        return None, f"dtype {pair.dtype}: the A100 TriMul kernels are bf16 only (no fp32 TriMul, by policy)"
    if pair.dim() != 4 or pair.shape[1] != pair.shape[2]:
        return None, f"pair must be [B, L, L, d_pair], got {tuple(pair.shape)}"
    if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != tuple(pair.shape[:2])):
        return None, "mask must be a bool [B, L] residue mask"
    from miniworld_engine.kernels.integrations import trimul_sm80

    why = trimul_sm80.refusal(module, pair, mask)
    return (None, why) if why is not None else ("sm80", None)


def _trimul_cuda_forward(module, pair, mask, dropout_p, *, bidirectional: bool) -> torch.Tensor:
    from miniworld_engine.kernels.integrations import trimul_h100

    route, _ = _trimul_cuda_route(module, pair, mask, dropout_p, bidirectional=bidirectional)
    if route == "sm80":
        from miniworld_engine.kernels.integrations import trimul_sm80

        dropout_p = module.p_drop if dropout_p is None else dropout_p
        scale = dropout_scale(pair, dropout_p, 1) if dropout_p and dropout_p > 0.0 and module.training else None
        return trimul_sm80.update(module, pair, mask, scale, bidirectional=bidirectional)
    if route == "infer":
        return trimul_h100.update_inference(module, pair, mask, bidirectional=bidirectional)
    dropout_p = module.p_drop if dropout_p is None else dropout_p
    # the reference's row dropout draw ([B, 1, L, D], drop_row; one H100 Philox launch, the same bits); the kernels fold it with the
    # residual into their epilogue
    scale = trimul_h100.dropout_scale(pair, dropout_p) if dropout_p and dropout_p > 0.0 and module.training else None
    return trimul_h100.update(module, pair, mask, scale, bidirectional=bidirectional)


def _per_sample(module, pair, mask, dropout_p, one):
    """``one(x, mask, dropscale)`` per sample (the Triton TriMul takes one square plane), concatenated; the row dropout scale drawn per
    sample as the reference draws it (``dropout_scale(pair, p, 1)``), training only."""
    dropout_p = module.p_drop if dropout_p is None else dropout_p
    scale = dropout_scale(pair, dropout_p, 1) if dropout_p and dropout_p > 0.0 and module.training else None
    outs = [
        one(pair[b : b + 1].contiguous(), None if mask is None else mask[b : b + 1], None if scale is None else scale[b : b + 1])
        for b in range(pair.shape[0])
    ]
    return outs[0] if len(outs) == 1 else torch.cat(outs)


class _TriangleMultiplicationBase(BackendModule, MasterWeights):
    """The single-direction triangle multiplicative update; a variant sets :attr:`divide_by_length`."""

    backends = ("pytorch",)

    #: A numerically dead instance of a checkpoint (set by a model's checkpoint loader): inference returns the residual input.
    dead: bool = False
    #: RoseTTAFold3: the contraction is multiplied by ``1 / (number of unmasked residues)`` of its sample (``1 / L`` without a mask) before
    #: ``ln_out``.
    divide_by_length: bool = False

    def __init__(
        self,
        d_pair: int = 128,
        *,
        d_hidden: int | None = None,
        outgoing: bool = True,
        p_drop: float = 0.25,
        master_weights: bool = False,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        self.outgoing = outgoing
        # The residual is unconditional; the row-broadcast dropout (drop_row, broadcast_dim=1) acts in training only.
        self.p_drop = p_drop

        if d_hidden is None:
            d_hidden = d_pair

        # The front projects d_pair -> d_hidden, ``to_out`` projects d_hidden -> d_pair (AF3 Alg. 12).
        self.d_hidden = d_hidden
        self.ln_pair = LayerNorm(d_pair)
        self.to_left = Linear(d_pair, d_hidden, bias=False, init="default")
        self.to_left_gate = Linear(d_pair, d_hidden, bias=False, init="zero")
        self.to_right = Linear(d_pair, d_hidden, bias=False, init="default")
        self.to_right_gate = Linear(d_pair, d_hidden, bias=False, init="zero")

        # LN_out normalises the CONTRACTION output, whose width is d_hidden, not d_pair.
        self.ln_out = LayerNorm(d_hidden)
        self.to_gate = Linear(d_pair, d_pair, bias=False, init="zero")
        self.to_out = Linear(d_hidden, d_pair, bias=False, init="zero")
        if master_weights:
            self.set_master_weights()
        self.set_backend(backend)

    def _forward_pytorch(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor | None = None,
        dropout_p: float | None = None,
    ) -> torch.Tensor:
        """``pair`` ``[B, L, L, d_pair]``, ``mask`` ``[B, L]`` (bool). ALWAYS returns the residual output ``pair + drop_row(trimul(pair))``.
        ``dropout_p`` overrides ``p_drop`` for this call (None: ``p_drop``); the dropout acts in training only."""
        if self.dead and not (self.training or torch.is_grad_enabled()):
            return pair
        dropout_p = self.p_drop if dropout_p is None else dropout_p
        pair_in = pair  # the residual
        scale = dropout_scale(pair, dropout_p, 1) if dropout_p and dropout_p > 0.0 and self.training else None
        with master_autocast(pair, self):
            pair = self.ln_pair(pair)
            left = sigmoid_gate(self.to_left_gate(pair), self.to_left(pair))
            right = sigmoid_gate(self.to_right_gate(pair), self.to_right(pair))

            if mask is not None:
                mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)
                left = left * mask_2d[..., None]
                right = right * mask_2d[..., None]

            out = torch.einsum("bikd,bjkd->bijd", left, right) if self.outgoing else torch.einsum("bkid,bkjd->bijd", left, right)

            if self.divide_by_length:
                out = out * inverse_length(out, mask)
            out = self.ln_out(out)
            out = sigmoid_gate(self.to_gate(pair), self.to_out(out))
            if scale is not None:
                out = out * scale
            return out + pair_in

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        return pair_inputs(
            self.ln_pair.normalized_shape[0], length=length, samples=samples, dtype=dtype, device=device, generator=generator
        )


class TriangleMultiplication(_TriangleMultiplicationBase):
    """Triangular multiplicative update (AF3 Algorithms 12 / 13)::

        x = LN(pair)
        a = sigmoid(W_ag x) * (W_a x),  b = sigmoid(W_bg x) * (W_b x)        # masked by mask_i & mask_j
        t = sum_k a_ik b_jk   (outgoing)  |  sum_k a_ki b_kj   (incoming)
        return pair + drop_row(sigmoid(W_g x) * W_o LN_out(t))

    Models: every model but RoseTTAFold3. ``TriangleMultiplication(..., model=name)`` builds the class of ``model`` instead
    (:data:`TRIANGLE_MULTIPLICATION_MODELS`), with the remaining arguments.

    Parameters
    ----------
    d_pair : int
        Dimension of pair representation.
    d_hidden : int | None
        Hidden dimension for left/right projections. Defaults to ``d_pair`` when *None*.
    outgoing : bool
        Whether to use outgoing edges.
    p_drop : float
        Row-broadcast dropout of the update (training only).
    master_weights : bool
        Keep fp32 master parameters under a bf16 cast (:class:`MasterWeights`).
    model : str | None
        Select a model's variant (see above); ``None`` is this class.
    backend : str
        See :class:`BackendModule`.

    """

    version = 1  # human-managed: bump when a change alters this class's accuracy or speed results

    def __new__(cls, *args, model: str | None = None, **kwargs):
        target = cls if model is None else _select(TRIANGLE_MULTIPLICATION_MODELS, "triangle multiplication", model)
        if cls is TriangleMultiplication and target is not TriangleMultiplication:
            return target(*args, **kwargs)  # not a TriangleMultiplication: Python does not call __init__ on it again
        return super().__new__(cls)

    def __init__(
        self,
        d_pair: int = 128,
        *,
        d_hidden: int | None = None,
        outgoing: bool = True,
        p_drop: float = 0.25,
        master_weights: bool = False,
        model: str | None = None,
        backend: str = "pytorch",
    ) -> None:
        super().__init__(d_pair, d_hidden=d_hidden, outgoing=outgoing, p_drop=p_drop, master_weights=master_weights, backend=backend)

    # ------------------------------------------------------------------------------------------------------------- library ops
    #: ``"cueq"``: ``cuequivariance_torch.triangle_multiplicative_update`` (the whole update: LN_in, gated + masked projections, contraction,
    #: LN_out, output projection x output gate; forward + backward); the residual and the row dropout are the module's. ``"anthropic"``: the
    #: shared core's TriMul provider (``opt_core.kernels.trimul``, the af3_torch kit's lever ``trimul``) by its ``fast`` tier word, the
    #: residual folded into its epilogue, inference only. (:class:`TriangleMultiplicationRF3` divides the contraction by the length before
    #: LN_out, which neither library op does: no cueq backend; its anthropic backend folds the 1 / n into W_b.) ``"triton"``: the engine's portable fused pipeline
    #: ``kernels/trimul_inproj/triton/unidirectional.trimul_triton`` (the whole update, residual and row dropout folded into its gate store;
    #: forward-only inference, an autograd Function in training), one launch per sample (the kernel takes one square plane). ``"cuda"``:
    #: the H100 TriMul kernels (``kernels/trimul_inproj/cuda``: the B200-style sm90 family with the per-cell legacy selection) through
    #: ``kernels/integrations/trimul_h100`` (``serves_inference`` / ``update_inference`` without grad or live dropout, else
    #: ``serves_single`` / ``update``), bf16 only.
    backends = ("pytorch", "anthropic", "cueq", "cuda", "triton")
    backend_versions: ClassVar[dict[str, int]] = {
        "anthropic": 3,  # 3 (2026-11-13): the sealed CUDA-13 rows served wherever the driver loads them
        "cueq": 2,
        "cuda:a100_sm80": 3,
        "cuda:h100_sm90": 6,
    }  # 2026-11-10: rewired to the model kits; 2026-11-11: A100 inference pack read by captures; 3: A100 D64 persistent front (k1w64);
    # cuda:h100_sm90 2 (2026-11-12): inference planes past 8 GiB (OpenDDE refiner, d_pair 384 at 2976 / 3968) run K1 + the contraction in two
    # halves of the hidden channels; cuda:h100_sm90 3 (2026-11-12): the training row-dropout scale by the H100 Philox kernel in one launch (the
    # same bits as the torch draw) and the small-L inference K1 / K3 tiles of the TriMul table (trimul_inproj h100_sm90 version 4);
    # cuda:h100_sm90 5 (2026-11-13): deterministic training backward at D256-512 (W_o column sums and the wide LayerNorm dgamma / dbeta by
    # per-block partials reduced in block order, no atomics); cuda:h100_sm90 6 (2026-11-13): the D256 port's K1 in n128 weight blocks
    # (schedule 1, eight-slot ring) and its inference K3 with the weights in a ring of their own (k3.cu WSTAGES 3); outputs bitwise unchanged

    def _supports_cuda(self, pair: torch.Tensor, mask: torch.Tensor | None = None, dropout_p: float | None = None) -> str | None:
        return _trimul_cuda_route(self, pair, mask, dropout_p, bidirectional=False)[1]

    def _forward_cuda(self, pair: torch.Tensor, mask: torch.Tensor | None = None, dropout_p: float | None = None) -> torch.Tensor:
        if self.dead and not (self.training or torch.is_grad_enabled()):
            return pair
        return _trimul_cuda_forward(self, pair, mask, dropout_p, bidirectional=False)

    #: Rows excluded by name on this H100 stack (driver 575 / CUDA 12.9, torch 2.13 + cu129): the sealed ``trimul_native`` package's
    #: rows ``native`` / ``native_exact`` and the sealed ``esm_v61`` package (CUDA-13 cubins) dump core when launched; ``tx_sm90a`` /
    #: ``tx_sm90a_exact`` have no prebuilt binary for this ABI and the provider's step-aside from them lands on a dumping row. The
    #: provider's next measured row of the cell serves instead (``select(exclude=...)``, the provider's own step-aside mechanism).
    _ANTHROPIC_EXCLUDED_ROWS = ("native", "native_exact", "native:f32in", "esm_v61", "tx_sm90a", "tx_sm90a_exact")

    def _library_refusal(self, pair: torch.Tensor, mask: torch.Tensor | None) -> str | None:
        if pair.dim() != 4 or pair.shape[1] != pair.shape[2]:
            return f"pair must be [B, L, L, d_pair], got {tuple(pair.shape)}"
        if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != tuple(pair.shape[:2])):
            return "mask must be a bool [B, L] residue mask"
        if self.ln_pair.eps != self.ln_out.eps:
            return "the library op takes one LayerNorm epsilon for LN_in and LN_out"
        return None

    def _pair_mask(self, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor | None:
        return None if mask is None else (mask[:, :, None] & mask[:, None, :]).to(pair.dtype)

    def _supports_cueq(self, pair: torch.Tensor, mask: torch.Tensor | None = None, dropout_p: float | None = None) -> str | None:
        from miniworld_engine.modules import _cueq

        why = _cueq.unavailable(pair.device) or self._library_refusal(pair, mask)
        if why is not None:
            return why
        if pair.dtype not in (torch.bfloat16, torch.float32):
            return f"dtype {pair.dtype}: the backend is verified for bfloat16 / float32"
        if self.d_hidden % 32 or pair.shape[-1] % 32:
            return "the op's kernels take widths that are multiples of 32"
        # The op runs its PyTorch reference formula for the calls its routing gives no kernel (short sequences; the bf16 hidden-256 band below
        # 2**15 pair rows) unless an inference owner takes the call: the library's own routing functions decide, nothing restated here.
        tmu = _cueq.ops("triangle_multiplicative_update")
        precision = tmu.Precision.NONE.value  # the op's default precision (fp32: TF32x3 unless torch allows TF32)
        if tmu._no_owner_path(pair, self.d_hidden, precision) == "reference":
            grad = torch.is_grad_enabled() and (pair.requires_grad or any(p.requires_grad for p in self.parameters()))
            owner = not grad and tmu.triangle_multiplicative_update_is_supported(
                pair,
                "outgoing" if self.outgoing else "incoming",
                None,
                c_hidden=self.d_hidden,
                precision=None,
            )
            if not owner:
                return (
                    f"L={pair.shape[1]}, d_pair={pair.shape[-1]}, d_hidden={self.d_hidden}, {pair.dtype}: cuEquivariance runs its PyTorch "
                    "reference formula for this call (no kernel owner; its routing's reference band)"
                )
        return None

    def _forward_cueq(self, pair: torch.Tensor, mask: torch.Tensor | None = None, dropout_p: float | None = None) -> torch.Tensor:
        from miniworld_engine.modules import _cueq

        if self.dead and not (self.training or torch.is_grad_enabled()):
            return pair
        dropout_p = self.p_drop if dropout_p is None else dropout_p
        scale = dropout_scale(pair, dropout_p, 1) if dropout_p and dropout_p > 0.0 and self.training else None
        w = lambda lin: lin.weight.to(pair.dtype)  # fp32 master matrices: cast in autograd, the gradient lands on the master
        update = _cueq.torch_api().triangle_multiplicative_update(
            pair,
            direction="outgoing" if self.outgoing else "incoming",
            mask=self._pair_mask(pair, mask),
            norm_in_weight=self.ln_pair.weight,
            norm_in_bias=self.ln_pair.bias,
            p_in_weight=torch.cat([w(self.to_left), w(self.to_right)]),  # rows [a; b]
            g_in_weight=torch.cat([w(self.to_left_gate), w(self.to_right_gate)]),
            norm_out_weight=self.ln_out.weight,
            norm_out_bias=self.ln_out.bias,
            p_out_weight=w(self.to_out),
            g_out_weight=w(self.to_gate),
            eps=self.ln_pair.eps,
            precision=None,  # the op's default
        )
        if scale is not None:
            update = update * scale
        return update + pair

    def _supports_triton(self, pair: torch.Tensor, mask: torch.Tensor | None = None, dropout_p: float | None = None) -> str | None:
        return _trimul_triton_refusal(self, pair, mask, self.d_hidden)

    def _forward_triton(self, pair: torch.Tensor, mask: torch.Tensor | None = None, dropout_p: float | None = None) -> torch.Tensor:
        from miniworld_engine.kernels.trimul_inproj.triton.unidirectional import trimul_triton

        if self.dead and not (self.training or torch.is_grad_enabled()):
            return pair

        def one(x, m, ds):
            w = lambda lin: lin.weight.to(x.dtype)  # fp32 master matrices: cast in autograd, the gradient lands on the master
            return trimul_triton(
                x,
                w(self.to_left),
                w(self.to_left_gate),
                w(self.to_right),
                w(self.to_right_gate),
                w(self.to_gate),
                w(self.to_out),
                self.ln_pair.weight,
                self.ln_pair.bias,
                self.ln_out.weight,
                self.ln_out.bias,
                self.ln_pair.eps,
                self.ln_out.eps,
                self.d_hidden,
                self.outgoing,
                mask=m,
                dropscale=ds,
            )

        return _per_sample(self, pair, mask, dropout_p, one)

    def _anthropic_select(self, pair: torch.Tensor) -> tuple[Any, tuple[Any, str | None]]:
        from miniworld_engine.modules import _anthropic

        prov = _anthropic.load("opt_core.kernels.trimul")
        return prov, _anthropic.kernel_selection(
            prov,
            "{}.{}".format(*torch.cuda.get_device_capability(pair.device)),
            {torch.bfloat16: "bf16", torch.float32: "fp32"}[pair.dtype],
            int(pair.shape[-1]),
            self.d_hidden,
            int(pair.shape[1]),
            "outgoing" if self.outgoing else "incoming",
            word="fast",
            exclude=None if _anthropic.cuda13_cubins_load() else self._ANTHROPIC_EXCLUDED_ROWS,
            batch=int(pair.shape[0]),
        )

    def _supports_anthropic(self, pair: torch.Tensor, mask: torch.Tensor | None = None, dropout_p: float | None = None) -> str | None:
        from miniworld_engine.modules import _anthropic

        why = _anthropic.unavailable("opt_core.kernels.trimul", pair.device) or self._library_refusal(pair, mask)
        if why is not None:
            return why
        if torch.is_grad_enabled():
            return "inference only (the provider's backward cells name the cuEquivariance op): call under torch.no_grad() / inference_mode"
        if self.training and (self.p_drop if dropout_p is None else dropout_p):
            return "the row dropout is live (training mode): inference only"
        if pair.dtype not in (torch.bfloat16, torch.float32):
            return f"dtype {pair.dtype}: the provider has bfloat16 / float32 cells only"
        return self._anthropic_select(pair)[1][1]

    def _forward_anthropic(self, pair: torch.Tensor, mask: torch.Tensor | None = None, dropout_p: float | None = None) -> torch.Tensor:
        if self.dead:  # inference only: a dead instance returns the residual input (as the reference does)
            return pair
        prov, (sel, _why) = self._anthropic_select(pair)
        # the parameters as they are: the face casts fp32 masters to a bf16 pair's dtype itself ("weights follow z's dtype")
        weights = {
            "ln_in_w": self.ln_pair.weight,
            "ln_in_b": self.ln_pair.bias,
            "w_ag": self.to_left_gate.weight,
            "w_ap": self.to_left.weight,
            "w_bg": self.to_right_gate.weight,
            "w_bp": self.to_right.weight,
            "ln_out_w": self.ln_out.weight,
            "ln_out_b": self.ln_out.bias,
            "w_o": self.to_out.weight,
            "w_og": self.to_gate.weight,
        }
        key = tuple((t.data_ptr(), t._version, t.dtype) for t in weights.values())
        if (
            getattr(self, "_anthropic_key", None) != key
        ):  # the face caches its weight packs by tensor: a new dict once the parameters change
            self._anthropic_key, self._anthropic_cache = key, {}
        cache = self._anthropic_cache
        return prov.triangle_multiplication(
            pair.contiguous(),
            self._pair_mask(pair, mask),
            direction="outgoing" if self.outgoing else "incoming",
            weights={k: t.detach() for k, t in weights.items()},
            selection=sel,
            residual=True,
            cache=cache,
            eps=self.ln_pair.eps,
        )


class TriangleMultiplicationRF3(_TriangleMultiplicationBase):
    """RoseTTAFold3's triangle multiplicative update: :class:`TriangleMultiplication` with the contraction multiplied by
    ``1 / (number of unmasked residues)`` of its sample (``1 / L`` without a mask, :func:`~miniworld_engine.modules.functional.inverse_length`)
    before ``ln_out``.

    Parameters as :class:`TriangleMultiplication`.
    """

    version = 1  # human-managed: bump when a change alters this class's accuracy or speed results

    divide_by_length = True

    #: ``"cuda"``: :class:`TriangleMultiplication`'s H100 TriMul kernels (``kernels/integrations/trimul_h100``) with the length factor
    #: folded into the left projection: ``W_a -> W_a / n`` scales ``a`` and so the contraction by ``1 / n`` exactly (the gate, the mask and
    #: the backward are linear in it; ``W_a``'s gradient comes back through the product), one launch per sample when the samples' masks
    #: differ; on an A100 the same on ``kernels/integrations/trimul_sm80``. ``"triton"``: the portable Triton TriMul the same way.
    #: ``"anthropic"``: see the anthropic section below.
    backends = ("pytorch", "anthropic", "cuda", "triton")
    backend_versions: ClassVar[dict[str, int]] = {
        "anthropic": 3,  # 2026-11-13: the sealed CUDA-13 rows served wherever the driver loads them
        "cuda:a100_sm80": 4,
        "cuda:h100_sm90": 3,
    }  # 2026-11-10: sm_80 paths; 3: the inference pack read by captures; 4: A100 D64 persistent front (k1w64); cuda:h100_sm90 2
    # (2026-11-12): the training row-dropout scale by the H100 Philox kernel, the small-L inference tiles of the TriMul table; cuda:h100_sm90 3 (2026-11-13): fp32-master weight casts once per parameter version

    def _supports_cuda(self, pair: torch.Tensor, mask: torch.Tensor | None = None, dropout_p: float | None = None) -> str | None:
        if pair.is_cuda and torch.cuda.get_device_capability(pair.device) not in ((9, 0), (8, 0)):
            return "the RF3 length factor is wired to the H100 (sm_90a) / A100 (sm_80) TriMul kernels only"
        one = pair[:1] if mask is not None and pair.shape[0] > 1 else pair  # per-sample launches: one sample's contract
        return _trimul_cuda_route(self, one, None if mask is None else mask[: one.shape[0]], dropout_p, bidirectional=False)[1]

    def _forward_cuda(self, pair: torch.Tensor, mask: torch.Tensor | None = None, dropout_p: float | None = None) -> torch.Tensor:
        if self.dead and not (self.training or torch.is_grad_enabled()):
            return pair
        factor = inverse_length(pair, mask).reshape(-1)  # [B], the reference's factor (in the pair's dtype)
        if mask is None or pair.shape[0] == 1:
            return _trimul_cuda_forward(_LeftScaled(self, factor[:1]), pair, mask, dropout_p, bidirectional=False)
        from miniworld_engine.kernels.integrations import trimul_h100

        p = self.p_drop if dropout_p is None else dropout_p
        drop = trimul_h100.dropout_scale(pair, p) if p and p > 0.0 and self.training else None  # drawn for the batch, as the reference does
        outs = []
        for b in range(pair.shape[0]):
            view, x, m = _LeftScaled(self, factor[b : b + 1]), pair[b : b + 1], mask[b : b + 1]
            route = _trimul_cuda_route(view, x, m, p, bidirectional=False)[0]
            if route == "sm80":  # A100: the same folded factor on kernels/integrations/trimul_sm80
                from miniworld_engine.kernels.integrations import trimul_sm80

                outs.append(trimul_sm80.update(view, x, m, None if drop is None else drop[b : b + 1], bidirectional=False))
            elif route == "infer":
                outs.append(trimul_h100.update_inference(view, x, m, bidirectional=False))
            else:
                outs.append(trimul_h100.update(view, x, m, None if drop is None else drop[b : b + 1], bidirectional=False))
        return torch.cat(outs)

    def _supports_triton(self, pair: torch.Tensor, mask: torch.Tensor | None = None, dropout_p: float | None = None) -> str | None:
        return _trimul_triton_refusal(self, pair, mask, self.d_hidden)

    def _forward_triton(self, pair: torch.Tensor, mask: torch.Tensor | None = None, dropout_p: float | None = None) -> torch.Tensor:
        from miniworld_engine.kernels.trimul_inproj.triton.unidirectional import trimul_triton

        if self.dead and not (self.training or torch.is_grad_enabled()):
            return pair

        def one(x, m, ds):
            w = lambda lin: lin.weight.to(x.dtype)  # fp32 master matrices: cast in autograd, the gradient lands on the master
            left = (self.to_left.weight * inverse_length(x, m).reshape(()).to(self.to_left.weight.dtype)).to(x.dtype)
            return trimul_triton(
                x,
                left,
                w(self.to_left_gate),
                w(self.to_right),
                w(self.to_right_gate),
                w(self.to_gate),
                w(self.to_out),
                self.ln_pair.weight,
                self.ln_pair.bias,
                self.ln_out.weight,
                self.ln_out.bias,
                self.ln_pair.eps,
                self.ln_out.eps,
                self.d_hidden,
                self.outgoing,
                mask=m,
                dropscale=ds,
            )

        return _per_sample(self, pair, mask, dropout_p, one)

    # ------------------------------------------------------------------------------------------------------------- anthropic
    #: ``"anthropic"`` (inference only): the shared core's TriMul provider (``opt_core.kernels.trimul``, the face every kit's TriMul calls --
    #: the rosettafold3 kit's fused rows ``fpf_rf3_trimul_rows.serve_fast`` serve RF3's cuEquivariance statement, which has no 1 / L) on the
    #: same math restated: the division of the contraction before ``ln_out`` is linear in the ``b`` operand, so each sample runs with its
    #: ``W_b`` (the right projection, before the gate product) scaled by ``1 / (unmasked residues)`` -- ``t / n = sum_k a_ik (b_jk / n)``;
    #: one provider call per sample.
    _ANTHROPIC_EXCLUDED_ROWS = TriangleMultiplication._ANTHROPIC_EXCLUDED_ROWS
    _library_refusal = TriangleMultiplication._library_refusal
    _pair_mask = TriangleMultiplication._pair_mask

    def _anthropic_select(self, pair: torch.Tensor) -> tuple[Any, tuple[Any, str | None]]:
        return TriangleMultiplication._anthropic_select(cast(TriangleMultiplication, self), pair[:1])  # one sample per call

    def _supports_anthropic(self, pair: torch.Tensor, mask: torch.Tensor | None = None, dropout_p: float | None = None) -> str | None:
        return TriangleMultiplication._supports_anthropic(cast(TriangleMultiplication, self), pair, mask, dropout_p)

    def _forward_anthropic(self, pair: torch.Tensor, mask: torch.Tensor | None = None, dropout_p: float | None = None) -> torch.Tensor:
        if self.dead:  # inference only: a dead instance returns the residual input (as the reference does)
            return pair
        prov, (sel, _why) = self._anthropic_select(pair)
        B, L = pair.shape[:2]
        count = (
            mask.sum(-1).clamp_min(1).double() if mask is not None else torch.full((B,), float(L), device=pair.device, dtype=torch.float64)
        )
        inv = count.reciprocal().float()  # [B] on the device (no host read)
        pm = self._pair_mask(pair, mask)
        base = {
            "ln_in_w": self.ln_pair.weight,
            "ln_in_b": self.ln_pair.bias,
            "w_ag": self.to_left_gate.weight,
            "w_ap": self.to_left.weight,
            "w_bg": self.to_right_gate.weight,
            "ln_out_w": self.ln_out.weight,
            "ln_out_b": self.ln_out.bias,
            "w_o": self.to_out.weight,
            "w_og": self.to_gate.weight,
        }
        outs = []
        with torch.no_grad():
            for b in range(B):
                weights = {**{k: t.detach() for k, t in base.items()}, "w_bp": self.to_right.weight.detach().float() * inv[b]}
                outs.append(
                    prov.triangle_multiplication(
                        pair[b : b + 1].contiguous(),
                        None if pm is None else pm[b : b + 1],
                        direction="outgoing" if self.outgoing else "incoming",
                        weights=weights,
                        selection=sel,
                        residual=True,
                        cache={},
                        eps=self.ln_pair.eps,
                    )
                )
        return outs[0] if B == 1 else torch.cat(outs)


class _LeftScaled:
    """:class:`TriangleMultiplicationRF3` as the TriMul kernels see it: ``to_left.weight`` multiplied by the sample's length factor
    (a tensor in the autograd graph of the master), every other attribute -- the parameters, the LayerNorms, the flags -- the module's."""

    def __init__(self, module: TriangleMultiplicationRF3, scale: torch.Tensor) -> None:
        from types import SimpleNamespace

        self._module = module
        self.to_left = SimpleNamespace(weight=module.to_left.weight * scale.to(module.to_left.weight.dtype))

    def __getattr__(self, name):
        return getattr(self._module, name)


#: ``TriangleMultiplication(model=name)`` -> the class it builds.
TRIANGLE_MULTIPLICATION_MODELS: dict[str, type[_TriangleMultiplicationBase]] = {
    **dict.fromkeys(TRIANGLE_ATTENTION_MODELS, TriangleMultiplication),
    "rf3": TriangleMultiplicationRF3,
}


class BidirectionalTriangleMultiplication(BackendModule, MasterWeights):
    """Triangular multiplicative update computing outgoing + incoming in one block; returns ``pair + drop_row(update)``.

    One module shares one input LayerNorm and projects the pair to ``2 * d_hidden`` channels; the hidden channels split in half -- the first
    ``d_hidden`` compute the **outgoing** product (``bikd,bjkd->bijd``), the second ``d_hidden`` the **incoming** product
    (``bkid,bkjd->bijd``). The two are concatenated to ``2 * d_hidden``, normalised together (``ln_out``) and projected down to ``d_pair``.
    Model: MiniWorld.

    ``"triton"``: ``kernels/trimul_inproj/triton/bidirectional.bidirectional_trimul_triton`` (the whole update with the residual and the row
    dropout), one launch per sample; bf16 activations (fp32 master matrices cast at the call). ``"cuda"``: the H100 TriMul kernels through
    ``kernels/integrations/trimul_h100`` (``serves_inference`` / ``update_inference`` without grad or live dropout, else ``serves`` /
    ``update``), bf16 activations only.
    """

    version = 1  # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "cuda", "triton")
    backend_versions: ClassVar[dict[str, int]] = {
        "cuda:a100_sm80": 2,
        "cuda:h100_sm90": 4,
    }  # 2026-11-10: sm_80 k1w64 front (D64); cuda:h100_sm90 2 (2026-11-12): the training row-dropout scale by the H100 Philox kernel, the
    # small-L inference tiles of the TriMul table; cuda:h100_sm90 3 (2026-11-13): fp32-master weight casts once per parameter version;
    # cuda:h100_sm90 4 (2026-11-13): deterministic training backward at D256-512 (W_o column sums and the wide LayerNorm dgamma / dbeta by
    # per-block partials reduced in block order, no atomics)

    def _supports_cuda(self, pair: torch.Tensor, mask: torch.Tensor | None = None, dropout_p: float | None = None) -> str | None:
        return _trimul_cuda_route(self, pair, mask, dropout_p, bidirectional=True)[1]

    def _forward_cuda(self, pair: torch.Tensor, mask: torch.Tensor | None = None, dropout_p: float | None = None) -> torch.Tensor:
        return _trimul_cuda_forward(self, pair, mask, dropout_p, bidirectional=True)

    def _supports_triton(self, pair: torch.Tensor, mask: torch.Tensor | None = None, dropout_p: float | None = None) -> str | None:
        return _trimul_triton_refusal(self, pair, mask, self.d_hidden, bidirectional=True)

    def _forward_triton(self, pair: torch.Tensor, mask: torch.Tensor | None = None, dropout_p: float | None = None) -> torch.Tensor:
        from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import bidirectional_trimul_triton

        def one(x, m, ds):
            w = lambda lin: lin.weight.to(x.dtype)  # fp32 master matrices: cast in autograd, the gradient lands on the master
            return bidirectional_trimul_triton(
                x,
                w(self.to_left),
                w(self.to_left_gate),
                w(self.to_right),
                w(self.to_right_gate),
                w(self.to_gate),
                w(self.to_out),
                self.ln_pair.weight,
                self.ln_pair.bias,
                self.ln_out.weight,
                self.ln_out.bias,
                self.ln_pair.eps,
                self.ln_out.eps,
                self.d_hidden,
                mask=m,
                dropscale=ds,
            )

        return _per_sample(self, pair, mask, dropout_p, one)

    def __init__(
        self,
        d_pair: int = 128,
        d_hidden: int | None = None,
        *,
        p_drop: float = 0.25,
        master_weights: bool = False,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        # The residual is unconditional; the row-broadcast dropout (drop_row, broadcast_dim=1) acts in training only.
        self.p_drop = p_drop
        self.d_pair = d_pair
        self.d_hidden = d_hidden if d_hidden is not None else d_pair
        d2 = 2 * self.d_hidden

        self.ln_pair = LayerNorm(d_pair)
        # Doubled-width left/right projections: [outgoing | incoming] channels.
        self.to_left = Linear(d_pair, d2, bias=False, init="default")
        self.to_left_gate = Linear(d_pair, d2, bias=False, init="zero")
        self.to_right = Linear(d_pair, d2, bias=False, init="default")
        self.to_right_gate = Linear(d_pair, d2, bias=False, init="zero")

        self.ln_out = LayerNorm(d2)
        self.to_gate = Linear(d_pair, d_pair, bias=False, init="zero")
        self.to_out = Linear(d2, d_pair, bias=False, init="zero")
        if master_weights:
            self.set_master_weights()
        self.set_backend(backend)

    def _forward_pytorch(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor | None = None,
        dropout_p: float | None = None,
    ) -> torch.Tensor:
        """``pair`` ``[B, L, L, d_pair]``, ``mask`` ``[B, L]`` (bool). ALWAYS returns the residual output
        ``pair + drop_row(bidir_trimul(pair))``. ``dropout_p`` overrides ``p_drop`` for this call; the dropout acts in training only."""
        dropout_p = self.p_drop if dropout_p is None else dropout_p
        pair_in = pair
        scale = dropout_scale(pair, dropout_p, 1) if dropout_p and dropout_p > 0.0 and self.training else None
        with master_autocast(pair, self):
            pair = self.ln_pair(pair)
            left = sigmoid_gate(self.to_left_gate(pair), self.to_left(pair))
            right = sigmoid_gate(self.to_right_gate(pair), self.to_right(pair))

            if mask is not None:
                mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)
                left = left * mask_2d[..., None]
                right = right * mask_2d[..., None]

            # Split hidden channels: first half -> outgoing, second half -> incoming.
            h = self.d_hidden
            left_out, left_in = left[..., :h], left[..., h:]
            right_out, right_in = right[..., :h], right[..., h:]

            out_outgoing = torch.einsum("bikd,bjkd->bijd", left_out, right_out)
            out_incoming = torch.einsum("bkid,bkjd->bijd", left_in, right_in)
            out = torch.cat([out_outgoing, out_incoming], dim=-1)

            out = self.ln_out(out)
            out = sigmoid_gate(self.to_gate(pair), self.to_out(out))
            if scale is not None:
                out = out * scale
            return out + pair_in

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        return pair_inputs(
            self.ln_pair.normalized_shape[0], length=length, samples=samples, dtype=dtype, device=device, generator=generator
        )
