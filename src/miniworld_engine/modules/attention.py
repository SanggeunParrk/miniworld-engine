"""Attentions over the token / atom single stream, each with its residual: ``AttentionPairBias`` (the Pairformer's single attention: AlphaFold 3
Algorithm 24 without the conditioning), ``AugmentedAttentionPairBias`` (Algorithm 24 with the adaptive conditioning: the DiT's attention) with
its model variants ``FlatQKNormAugmentedAttention`` (RoseTTAFold3: LayerNorm of the flat q / k) and ``UngatedAugmentedAttention`` (Chai-1: no
gating query), and ``BiasOnlyAttention`` (the bias-only DiT's attention: the pair bias alone is the logits, no query and no key).

Each is a :class:`~miniworld_engine.modules.primitives.BackendModule`; ``_forward_pytorch`` is the reference.
"""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from miniworld_engine.kernels import _capture
from miniworld_engine.modules.functional import sigmoid_gate
from miniworld_engine.modules.norm import AdaptiveLayerNorm, LayerNorm, RMSNorm, _activation_dtype_reason, _wants_grad, triton_unsupported
from miniworld_engine.modules.primitives import BackendModule, Linear, MasterWeights, key_mask, master_autocast, random_input

_capture.instance_cache_attr("_anthropic_params")  # the anthropic producer's fp32 parameter copies (clear_weight_caches)

# ------------------------------------------------------------------------------------------------------------------- "triton" core
#: the head widths the Triton pair-bias attention core (kernels/augmented_attention/triton/main.py) is built for
TRITON_MAX_HEAD_DIM = 64


def _triton_core_reason(x: torch.Tensor, head_dim: int) -> str | None:
    """Why the Triton pair-bias attention core cannot take a call on ``x`` (its activations), or None. Forward and backward (autograd)."""
    if not x.is_cuda:
        return "CUDA tensors only"
    dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
    if dt not in (torch.bfloat16, torch.float32):
        return f"dtype {dt} (bfloat16 / float32 only)"
    if head_dim > TRITON_MAX_HEAD_DIM:
        return f"head dim {head_dim} > {TRITON_MAX_HEAD_DIM} (the Triton core's limit)"
    return None


def _triton_core(q, k, v, bias, mask):
    """``softmax(q k^T / sqrt(D) + bias) v`` on the Triton kernel: q / k / v [A, B, L, H, D], bias [B, L, L, H] (shared by the A samples),
    mask [A, B, L] bool or None (masked keys off). Differentiable in q, k, v and bias."""
    from miniworld_engine.kernels.augmented_attention.triton.main import triton_augmented_attention_pair_bias

    dt = q.dtype
    return triton_augmented_attention_pair_bias(q, k.to(dt), v.to(dt), bias.to(dt), mask)


#: the requirements stream whose ``make_inputs`` adds an ``extra_pair_bias`` (OpenDDE's structural-token refiner: an [L, L] logit bias on every
#: head, FoldForge R3)
EXTRA_BIAS_STREAM = "token_single_extra_bias"


def _extra_planes_reason(extra: torch.Tensor, b: int, h: int, n: int) -> str | None:
    """Why ``extra_pair_bias`` cannot be read as ``[B, H, L, L]`` head-major logit planes, or None: ``[L, L]``, ``[H | 1, L, L]`` or
    ``[B | 1, H | 1, L, L]`` (a size-1 axis is shared; stride-0 expanded views are fine), a floating dtype."""
    if not extra.is_floating_point():
        return f"extra_pair_bias dtype {extra.dtype} (a floating dtype)"
    if extra.ndim not in (2, 3, 4) or tuple(extra.shape[-2:]) != (n, n):
        return f"extra_pair_bias must be [L, L], [H, L, L] or [B, H, L, L] with L = {n}, got {tuple(extra.shape)}"
    lead = (1,) * (4 - extra.ndim) + tuple(extra.shape[:-2])
    if lead[0] not in (1, b) or lead[1] not in (1, h):
        return f"extra_pair_bias {tuple(extra.shape)} does not broadcast to [B, H, L, L] = [{b}, {h}, {n}, {n}]"
    return None


def _extra_planes(extra: torch.Tensor, b: int, h: int, n: int) -> torch.Tensor:
    """``extra_pair_bias`` as an (expanded, possibly stride-0) ``[B, H, L, L]`` view (:func:`_extra_planes_reason` must be None)."""
    return extra.reshape((1,) * (4 - extra.ndim) + tuple(extra.shape)).expand(b, h, n, n)


class AttentionPairBias(BackendModule, MasterWeights):
    """Attention with pair bias::

        s = LN(single); q = W_q s + b_q, k = W_k s, v = W_v s       # [B, H, L, D] (q, k RMS-normed with ``use_qk_norm``)
        bias = W_b LN_p(pair) [+ extra_pair_bias], masked keys at the dtype's min
        o = sigmoid(W_g s + gate_bias) * sdpa(q, k, v, bias)
        return single + W_o o

    ``ln_pair`` has no offset: it feeds only the logits (``to_bias`` has no bias), and an offset adds one constant per head to every logit,
    which the softmax cancels.

    Parameters
    ----------
    d_single : int
        Dimension of single representation.
    d_pair : int
        Dimension of pair representation.
    n_head : int
        Number of attention heads.
    d_hidden : int, optional
        Per-head hidden dimension. Defaults to ``d_single // n_head``.
    use_qk_norm : bool
        Whether to apply RMSNorm to query and key projections.
    gate_bias : float
        A constant added to the gate logits (Chai-1's pairformer single attention: 1.0).
    master_weights : bool
        Keep fp32 master parameters under a bf16 cast (:class:`MasterWeights`).
    backend : str
        See :class:`BackendModule`.

    """

    version = 3  # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "anthropic", "cueq", "cuda", "triton")
    backend_versions: ClassVar[dict[str, int]] = {
        "anthropic": 2,
        "cuda:a100_sm80": 6,
        "cuda:h100_sm90": 2,
        "triton:a100_sm80": 2,
        "triton:h100_sm90": 2,
    }  # 2026-11-10: kits; sm_80 paths; 3: inference packs read by captures; triton:a100_sm80 / triton:h100_sm90 2 (2026-11-11): the Triton core returns dbias head-major on the A100 / H100; cuda:a100_sm80 4 / cuda:h100_sm90 2 (2026-11-11): B > 1 served (one step per batch row, FoldForge 5I28 confidence head); cuda:a100_sm80 5 (2026-11-12): the composed fp32 path's pair backward is one TF32 tensor-core launch for the batch (pair_bias_gen_bwd_tc); cuda:a100_sm80 6 (2026-11-12): the generic sm_80 pair-bias backward (pair_bias_gen_bwd, bf16 pairs) 1.4-2.1x faster, the same values

    #: A numerically dead instance of a checkpoint (set by a model's checkpoint loader): inference returns the residual input.
    dead: bool = False

    def __init__(
        self,
        d_single: int = 384,
        d_pair: int = 128,
        n_head: int = 8,
        d_hidden: int | None = None,
        *,
        use_qk_norm: bool = False,
        gate_bias: float = 0.0,
        master_weights: bool = False,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.gate_bias = float(gate_bias)
        self.use_qk_norm = use_qk_norm
        if d_hidden is None:
            if d_single % n_head != 0:
                msg = f"{d_single=} must be divisible by {n_head=}"
                raise ValueError(msg)
            d_hidden = d_single // n_head

        self.ln_single = LayerNorm(d_single)
        self.to_query = Linear(d_single, d_hidden * n_head, bias=True, init="glorot")
        self.to_key = Linear(d_single, d_hidden * n_head, bias=False, init="glorot")
        self.to_value = Linear(d_single, d_hidden * n_head, bias=False, init="glorot")

        if use_qk_norm:
            self.norm_query = RMSNorm(d_hidden)
            self.norm_key = RMSNorm(d_hidden)

        self.ln_pair = LayerNorm(d_pair, bias=False)
        self.to_bias = Linear(d_pair, n_head, bias=False, init="default")
        self.to_gate = Linear(d_single, d_hidden * n_head, bias=False, init="gating")
        self.to_out = Linear(d_hidden * n_head, d_single, bias=False, init="zero")
        if master_weights:
            self.set_master_weights()
        self.set_backend(backend)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # checkpoints written before the offset was removed carry ln_pair.bias; it had no effect on the output, so dropping it is exact
        state_dict.pop(prefix + "ln_pair.bias", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _forward_pytorch(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        mask: torch.Tensor | None = None,
        extra_pair_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``single`` ``[B, L, d_single]``, ``pair`` ``[B, L, L, d_pair]``, ``mask`` ``[B, L]`` (bool). ALWAYS returns the residual output
        ``single + attention_pair_bias(single, pair)``.

        ``extra_pair_bias`` ``[B, n_head, L, L]`` (head-major; a head axis of 1 is shared by every head) is added to the pair-bias logits
        after ``to_bias``, before the key mask and the softmax; gradient flows to it."""
        if self.dead and not (self.training or torch.is_grad_enabled()):
            return single
        with master_autocast(single, self):
            single_res = single  # residual == the ORIGINAL input (before ln_single rebinds `single`)
            single = self.ln_single(single)
            query = self.to_query(single)
            key = self.to_key(single)
            value = self.to_value(single)

            query = rearrange(query, "B L (H D) -> B H L D", H=self.n_head)
            key = rearrange(key, "B L (H D) -> B H L D", H=self.n_head)
            value = rearrange(value, "B L (H D) -> B H L D", H=self.n_head)

            if self.use_qk_norm:
                query = self.norm_query(query)
                key = self.norm_key(key)

            pair = self.ln_pair(pair)
            bias = self.to_bias(pair)
            bias = rearrange(bias, "B L L2 H -> B H L L2")
            if extra_pair_bias is not None:  # before the key mask: a masked key stays at the minimum
                bias = bias + extra_pair_bias.to(bias.dtype)
            if mask is not None:
                bias = bias.masked_fill(~mask[:, None, None, :], torch.finfo(bias.dtype).min)
            out = F.scaled_dot_product_attention(query, key, value, bias)

            gate = self.to_gate(single)
            if self.gate_bias:
                gate = gate + self.gate_bias
            out = rearrange(out, "B H L D -> B L (H D)")
            out = sigmoid_gate(gate, out)
            out = self.to_out(out)
            return single_res + out

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        d_single, d_pair = self.ln_single.normalized_shape[0], self.ln_pair.normalized_shape[0]
        return {
            "single": random_input((samples, length, d_single), dtype=dtype, device=device, generator=generator),
            "pair": random_input((samples, length, length, d_pair), dtype=dtype, device=device, generator=generator),
            "mask": key_mask(samples, length, device=device),
            # OpenDDE's refiner (FoldForge R3): one [L, L] bias expanded over the samples and the heads, as FoldForge hands it
            **(
                {
                    "extra_pair_bias": random_input((length, length), dtype=dtype, device=device, generator=generator)[None, None].expand(
                        samples, self.n_head, length, length
                    )
                }
                if stream == EXTRA_BIAS_STREAM
                else {}
            ),
        }

    def _extra_reason(self, single, extra_pair_bias) -> str | None:
        if extra_pair_bias is None or single.dim() != 3:
            return None
        return _extra_planes_reason(extra_pair_bias, single.shape[0], self.n_head, single.shape[1])

    # ------------------------------------------------------------------------------------------------------------- "triton"
    # The reference's statements with the attention core (q k^T / sqrt(D) + pair bias [+ extra], key mask, softmax, P v) on the Triton
    # pair-bias attention kernel (forward and backward); the LayerNorms / projections are the module's. bf16 / fp32 / master, head dim <= 64.
    def _supports_triton(self, single, pair, mask=None, extra_pair_bias=None) -> str | None:
        why = self._call_refusal(single, pair, mask) if single.dim() == 3 else "single must be [B, L, d]"
        return why or self._extra_reason(single, extra_pair_bias) or _triton_core_reason(single, self._head_dim())

    def _forward_triton(self, single, pair, mask=None, extra_pair_bias=None):
        if self.dead and not (self.training or torch.is_grad_enabled()):
            return single
        with master_autocast(single, self):
            b, n = single.shape[:2]
            h, d = self.n_head, self._head_dim()
            s = self.ln_single(single)
            q, k, v = (t.view(b, n, h, d) for t in (self.to_query(s), self.to_key(s), self.to_value(s)))
            if self.use_qk_norm:
                q, k = self.norm_query(q), self.norm_key(k)
            bias = self.to_bias(self.ln_pair(pair))  # [B, L, L, H]
            if extra_pair_bias is not None:
                bias = bias + _extra_planes(extra_pair_bias, b, h, n).to(bias.dtype).permute(0, 2, 3, 1)
            out = _triton_core(q[None], k[None], v[None], bias, None if mask is None else mask[None])[0]
            gate = self.to_gate(s)
            if self.gate_bias:
                gate = gate + self.gate_bias
            out = sigmoid_gate(gate, out.reshape(b, n, h * d).to(gate.dtype))
            return single + self.to_out(out)

    # ------------------------------------------------------------------------------------------------------------- "cuda"
    # H100 (sm_90): the whole op as hand CUDA + cuBLAS (kernels/integrations/attention_pair_bias_b200.py, its H100 branches; the residual is
    # folded there). No gradient -> the inference op (LayerNorm rows, one packed q | k | v | g GEMM, pair LayerNorm + head projection pass,
    # the gated sm_90a wgmma core, to_out accumulated onto the residual); a gradient -> one autograd Function (forward + backward on the same
    # rows / cores, fp32 weight gradients onto the parameters). Range = ``refusal``: bf16 activations (parameters bf16 or fp32 masters) or
    # float32 activations with float32 parameters (the TF32 path: attn_tf32 core, fp32 row / pair kernels, fp32 GEMMs), any batch B
    # (the one-row step per batch row inside each op, the weight packs made once; FoldForge 5I28 confidence head: its samples as B),
    # 8 x 48 / 12 x 32 / 16 x 24 / 24 x 16 heads at d_single 384 or 16 x 32 at 512, d_pair 128, 256 / 384 (8 or 16 heads) or 512 (16 heads),
    # gate_bias, no QK-norm, L a multiple of 128; everything else is refused, never run elsewhere. extra_pair_bias (eval only, FoldForge R3:
    # OpenDDE's refiner): added to the bias planes the pair pass writes, before the core, in the core's units (exp2; the fp32 core's key order).
    # A100 (sm_80): kernels/integrations/attention_pair_bias_sm80.py -- the fused C++ step (bf16 activations, any B, the same head layouts,
    # d_pair 128, any L, no gate_bias) and, for fp32 (TF32 core), the wider pairs (256 / 384 / 512) and gate_bias, the path composed from
    # the sm_80 attention kernels (any B, L a multiple of 128, head dims 16-64; both one step per batch row); no QK-norm; inference and
    # training (the weight gradients summed over B). extra_pair_bias (eval only): the composed path, its planes added to the bias tensor
    # it makes.
    # Its ``refusal`` is the range there.
    def _supports_cuda(self, single, pair, mask=None, extra_pair_bias=None) -> str | None:
        from miniworld_engine.kernels.integrations import attention_pair_bias_b200 as native

        why = self._call_refusal(single, pair, mask) or self._extra_reason(single, extra_pair_bias)
        if why is not None:
            return why
        grad = self._cuda_grad(single, pair, extra_pair_bias)
        if extra_pair_bias is not None and grad:
            # the training Functions keep the bias inside their passes and return no gradient for an additive logit term ("triton" does)
            return "extra_pair_bias with a gradient: the cuda backend folds it into the inference ops' bias planes only (eval)"
        if single.is_cuda and torch.cuda.get_device_capability(single.device) == (8, 0):
            from miniworld_engine.kernels.integrations import attention_pair_bias_sm80 as sm80

            if extra_pair_bias is not None:  # the fused C++ step makes its bias inside the step: the composed path, whose bias is a tensor
                generic = sm80._generic_refusal(self, single, pair, mask, False)
                return None if generic is None else f"extra_pair_bias (A100: the composed path only): {generic}"
            return sm80.refusal(self, single, pair, mask, grad)
        return native.refusal(self, single, pair, mask, grad)

    def _cuda_grad(self, single, pair, extra_pair_bias=None) -> bool:
        return torch.is_grad_enabled() and (
            single.requires_grad
            or pair.requires_grad
            or (extra_pair_bias is not None and extra_pair_bias.requires_grad)
            or any(p.requires_grad for p in self.parameters())
        )

    def _forward_cuda(self, single, pair, mask=None, extra_pair_bias=None):
        from miniworld_engine.kernels.integrations import attention_pair_bias_b200 as native

        if self.dead and not (self.training or torch.is_grad_enabled()):
            return single
        # extra_pair_bias (eval only, ``_supports_cuda``): [B, H, L, L] planes added to the kernels' hoisted bias before the core (FoldForge R3)
        extra = None if extra_pair_bias is None else _extra_planes(extra_pair_bias, single.shape[0], self.n_head, single.shape[1])
        if torch.cuda.get_device_capability(single.device) == (8, 0):
            from miniworld_engine.kernels.integrations import attention_pair_bias_sm80 as sm80

            if extra is not None:
                return sm80.update_generic(self, single, pair, mask, False, extra=extra)
            if self._cuda_grad(single, pair):
                return sm80.update_train(self, single, pair, mask)
            return sm80.update_inference(self, single, pair, mask)
        if self._cuda_grad(single, pair):
            return native.update_train(self, single, pair, mask)
        return native.update_inference(self, single, pair, mask, extra=extra)

    # ------------------------------------------------------------------------------------------------------------- library backends
    def _head_dim(self) -> int:
        return self.to_query.weight.shape[0] // self.n_head

    def _call_refusal(self, single, pair, mask) -> str | None:
        if single.dim() != 3 or pair.dim() != 4 or pair.shape[:3] != (single.shape[0], single.shape[1], single.shape[1]):
            return f"single must be [B, L, d] and pair [B, L, L, d_pair], got {tuple(single.shape)} / {tuple(pair.shape)}"
        if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != tuple(single.shape[:2])):
            return "mask must be a bool [B, L] key mask"
        if single.dtype not in (torch.bfloat16, torch.float32):
            return f"dtype {single.dtype}: bfloat16 / float32 only"
        return None

    # "anthropic": the kits' Pairformer single attention (af3_torch's lever ``apb``: ``_pair_logits_fused`` / ``_apb_planes`` + ``_selfattn_forward`` /
    # ``_apb_attention``; openfold3's ``cells/apb_trunk`` and protenix's ``pf_attn`` bind the same two faces) on the shared core's pair-bias
    # provider (opt_core.kernels.apb), by the kits' tier word "fast":
    #   * the PRODUCER ``pair_bias_planes`` -- bias = W_b LN_p(pair), head-major, on the row the provider selects for the cell
    #     bias_c<d_pair>h<H> (ln_proj / lnl_ln_linear / fpf_pf_bias; a geometry no cell lists: the provider names its stock row);
    #   * the CORE ``pair_bias_attention`` -- sigmoid(g) * softmax(q k^T / sqrt(D) + bias) v, the cell pf_h16d24 (other geometries: no cell
    #     word, the provider names its boundary's row), the key mask folded as -1e9 as the kits do.
    # Every row served as the release serves it; LN(single), q / k / v / gate and the out-projection are the module's statements (the kits'
    # too). The module is model-agnostic and every kit binds the same provider, so one wiring serves every requirement row. ``ln_pair`` has no
    # offset: the producer gets a zero LayerNorm bias (exact). Forward only: no kit carries a backward for this op (every carried row of the
    # provider's table is ``backward: false``; only its stock rows -- SDPA, cuEquivariance, torch -- differentiate), so a call with a
    # gradient is refused.
    def _anthropic_core_select(self, single):
        from miniworld_engine.modules import _anthropic

        prov = _anthropic.load("opt_core.kernels.apb")
        h, d = self.n_head, self._head_dim()
        cell = prov.cell_word("pf", h, d)
        dt = torch.bfloat16 if (single.dtype == torch.bfloat16 or torch.is_autocast_enabled("cuda")) else single.dtype
        return (
            prov,
            cell,
            _anthropic.kernel_selection(
                prov,
                tuple(torch.cuda.get_device_capability(single.device)),
                dt,
                cell,
                int(single.shape[1]),
                word="fast",
                samples=int(single.shape[0]),
                head_dim=d,
                heads=h,
            ),
        )

    def _anthropic_producer_select(self, pair):
        from miniworld_engine.modules import _anthropic

        prov = _anthropic.load("opt_core.kernels.apb")
        h, c = self.n_head, int(pair.shape[-1])
        cell = prov.cell_word("bias", heads=h, c_z=c)
        # no producer cell for this geometry: the producer boundary's stock row by name (a cell-less tier word would name the CORE's stock row)
        word = "fast" if cell is not None else prov.STOCK_OF_BOUNDARY["producer"]
        return _anthropic.kernel_selection(
            prov, tuple(torch.cuda.get_device_capability(pair.device)), pair.dtype, cell, int(pair.shape[1]), word=word, c_z=c, heads=h
        )

    def _anthropic_producer_params(self):
        """fp32 contiguous LayerNorm weight / zero offset / projection the producer rows pack or cast from (the kits hand fp32 copies too),
        rebuilt only when a parameter changes (identity + version), so the provider's packed-weight cache keeps hitting."""
        w_ln, w_b = self.ln_pair.weight, self.to_bias.weight
        key = (id(w_ln), w_ln._version, id(w_b), w_b._version)
        held = self.__dict__.get("_anthropic_params")
        if held is None or held[0] != key:
            lnw = w_ln.detach().float().contiguous()
            held = (key, (lnw, torch.zeros_like(lnw), w_b.detach().float().contiguous()), {})
            self.__dict__["_anthropic_params"] = held
        return held[1], held[2]

    def _supports_anthropic(self, single, pair, mask=None, extra_pair_bias=None) -> str | None:
        from miniworld_engine.modules import _anthropic

        why = _anthropic.unavailable("opt_core.kernels.apb", single.device) or self._call_refusal(single, pair, mask)
        if why is not None:
            return why
        if torch.is_grad_enabled():
            return (
                "inference only: no kit carries a backward for the pair-bias attention (the provider's carried rows are all forward only): "
                "call under torch.no_grad() / inference_mode"
            )
        why = self._anthropic_producer_select(pair)[1]
        if why is not None:
            return f"pair-bias producer: {why}"
        return self._anthropic_core_select(single)[2][1]

    def _forward_anthropic(self, single, pair, mask=None, extra_pair_bias=None):
        if self.dead:
            return single
        with master_autocast(single, self):
            prov, cell, (sel, _why) = self._anthropic_core_select(single)
            psel, _pwhy = self._anthropic_producer_select(pair)
            b, n = single.shape[:2]
            h, d = self.n_head, self._head_dim()
            s = self.ln_single(single)
            q, k, v = (t.view(b, n, h, d) for t in (self.to_query(s), self.to_key(s), self.to_value(s)))
            if self.use_qk_norm:
                q, k = self.norm_query(q), self.norm_key(k)
            gate = self.to_gate(s)
            if self.gate_bias:
                gate = gate + self.gate_bias
            (lnw, lnb, w_b), cache = self._anthropic_producer_params()
            bias, _served = prov.pair_bias_planes(
                pair.contiguous(), lnw, lnb, w_b, selection=psel, eps=self.ln_pair.eps, out_layout="hij", out_dtype=q.dtype, cache=cache
            )
            bias = bias[..., :h, :, :n]  # rows writing [B, H, L, ld] planes (ld = L rounded to 8)
            if extra_pair_bias is not None:
                bias = bias + extra_pair_bias.to(bias.dtype)
            if mask is not None:
                bias = bias.masked_fill(~mask[:, None, None, :], -1e9)
            bias = bias.to(q.dtype).expand(b, h, n, n).contiguous()
            o, _served = prov.pair_bias_attention(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                bias,
                None,
                gate=gate.view(b, n, h, d).contiguous(),
                selection=sel,
                scale=d**-0.5,
                layout="snhd",
                cell=cell,
            )
            return single + self.to_out(o.reshape(b, n, h * d))

    # "cueq": cuequivariance_torch.attention_pair_bias -- the whole op (LayerNorm of single and pair, q / k / v / gate / pair-bias projections,
    # masked softmax attention, gate, out-projection) on the module's parameters; the residual is the module's. Which route the library takes
    # for a call (a kernel, or its own native PyTorch formula) is the library's choice.
    def _supports_cueq(self, single, pair, mask=None, extra_pair_bias=None) -> str | None:
        from miniworld_engine.modules import _cueq

        why = _cueq.unavailable(single.device) or self._call_refusal(single, pair, mask)
        if why is not None:
            return why
        if self.use_qk_norm:
            return "the per-head q / k RMSNorm: the op's q / k norm is one LayerNorm over all heads"
        if extra_pair_bias is not None:
            return "extra_pair_bias: the op has no additive logit input"
        if self.ln_single.eps != self.ln_pair.eps:
            return "the op takes one LayerNorm epsilon for the single and the pair"
        return None

    def _forward_cueq(self, single, pair, mask=None, extra_pair_bias=None):
        from miniworld_engine.modules import _cueq

        if self.dead and not (self.training or torch.is_grad_enabled()):
            return single
        with master_autocast(single, self):
            gate_bias = None
            if self.gate_bias:
                gate_bias = torch.full(
                    (self.to_gate.weight.shape[0],), self.gate_bias, dtype=self.to_gate.weight.dtype, device=single.device
                )
            out = _cueq.torch_api().attention_pair_bias(
                single,
                pair,
                None if mask is None else mask.to(single.dtype),
                self.n_head,
                self.ln_single.weight,
                self.ln_single.bias,
                self.to_query.weight,
                self.to_query.bias,
                self.to_key.weight,
                self.to_value.weight,
                self.to_gate.weight,
                self.to_out.weight,
                w_proj_z=self.to_bias.weight,
                w_ln_z=self.ln_pair.weight,
                b_ln_z=None,
                eps=self.ln_single.eps,
                b_proj_g=gate_bias,
            )
            out = out[0] if isinstance(out, tuple) else out
            return single + out.to(single.dtype)


class AugmentedAttentionPairBias(BackendModule):
    """Augmented attention with pair bias and adaptive conditioning (AlphaFold 3 Algorithm 24), with the residual::

        a = AdaLN(single, cond); q = W_q a + b_q, k = W_k a, v = W_v a     # [A, B, L, H, D] (q, k RMS-normed with ``use_qk_norm``)
        logits = q k^T / sqrt(D) + W_b LN_p(pair) [+ extra_pair_bias], masked keys at the dtype's min
        o = sigmoid(W_g a) * softmax(logits) v
        return single + sigmoid(W_sc cond + b_sc) * W_o o

    ``single`` / ``cond`` carry the augmentation axis ``A`` (the diffusion samples) in front of the batch; the pair bias is shared by them.
    ``ln_pair`` has no offset: its bias reaches the loss only through ``to_bias`` as a per-head constant on every logit, which the softmax
    cancels (AlphaFold 3: ``create_offset=False``).

    Parameters
    ----------
    d_single : int
        Dimension of single representation.
    d_cond : int
        Dimension of conditioning representation.
    d_pair : int
        Dimension of pair representation.
    n_head : int
        Number of attention heads.
    use_qk_norm : bool
        Whether to apply RMSNorm to query and key projections.
    query_bias : bool
        Whether the query projection carries a bias (AlphaFold 3: yes; RoseTTAFold3's ``to_q``: no).
    backend : str
        See :class:`BackendModule`.

    """

    backends = ("pytorch", "triton")
    #: The K / V AdaLN of the atom DiT blocks whose keys and values have their own conditioning (set by modules/atom_dit.py; absent
    #: otherwise). Declared for the type checker only.
    ada_ln_kv: AdaptiveLayerNorm

    def __init__(
        self,
        d_single: int,
        d_cond: int,
        d_pair: int,
        n_head: int,
        *,
        use_qk_norm: bool = False,
        query_bias: bool = True,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.use_qk_norm = use_qk_norm

        d_hidden = d_single // n_head

        self.ada_ln_in = AdaptiveLayerNorm(d_single, d_cond)
        self.to_query = Linear(d_single, d_hidden * n_head, bias=query_bias)
        self.to_key = Linear(d_single, d_hidden * n_head, bias=False)
        self.to_value = Linear(d_single, d_hidden * n_head, bias=False)

        if use_qk_norm:
            self.norm_query = RMSNorm(d_hidden)
            self.norm_key = RMSNorm(d_hidden)

        self.ln_pair = LayerNorm(d_pair, bias=False)
        self.to_bias = Linear(d_pair, n_head, bias=False, init="zero")
        self.to_gate = Linear(d_single, d_hidden * n_head, bias=False, init="gating")
        self.to_out = Linear(d_hidden * n_head, d_single, bias=False, init="zero")

        self.to_scale = Linear(d_cond, d_single, bias=True, init="default", bias_init=-2.0)  # sigmoid(-2) ~ 0.1: the update starts small
        self.set_backend(backend)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        """Drop ``ln_pair.bias`` from checkpoints written before it was removed (its gradient was identically zero, so a trained checkpoint
        carries it at its zero init and discarding it changes nothing numerically)."""
        state_dict.pop(f"{prefix}ln_pair.bias", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _attention_core(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor | None = None,
        compute_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """``softmax(q k^T / sqrt(D) + bias) v`` on ``query`` / ``key`` / ``value`` ``[A, B, L, H, D]``, ``bias`` ``[B, L, L, H]``, ``mask``
        ``[A, B, L]``. ``compute_dtype`` is the precision the core runs in (``None``: the caller's)."""
        if compute_dtype is not None:
            query, key, value, bias = (t.to(compute_dtype) for t in (query, key, value, bias))
        query = query * query.shape[-1] ** -0.5
        attention = torch.einsum("abihd,abjhd->abhij", query, key)
        bias = bias.permute(0, 3, 1, 2).contiguous()  # (B, H, L, L)
        attention = attention + bias[None]
        if mask is not None:
            # finfo.min (not -inf): a fully-masked row of -inf softmaxes to NaN
            attention = attention.masked_fill(~mask[:, :, None, None, :], torch.finfo(attention.dtype).min)
        attention = F.softmax(attention, dim=-1)
        return torch.einsum("abhij,abjhd->abihd", attention, value)

    def _forward_pytorch(
        self,
        single: torch.Tensor,
        cond: torch.Tensor,
        pair: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        compute_dtype: torch.dtype | None = None,
        extra_pair_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``single`` ``[A, B, L, d_single]``, ``cond`` ``[A, B, L, d_cond]``, ``pair`` ``[B, L, L, d_pair]``, ``mask`` ``[B, L]`` or
        ``[A, B, L]`` (bool). ALWAYS returns the residual output ``single + delta(single, cond, pair, mask)``; a caller that composes the
        update some other way takes :meth:`delta`."""
        return single + self.delta(single, cond, pair, mask, compute_dtype=compute_dtype, extra_pair_bias=extra_pair_bias)

    #: the fp32 attention logits one call may hold, A x B x H x L x L x 4 bytes: above it the augmentation axis is sliced (``delta``)
    LOGIT_BUDGET_BYTES = 8 << 30

    def augment_chunk(self, single: torch.Tensor) -> int:
        """How many samples (the augmentation axis A) one call takes: all of them, unless their logits exceed ``LOGIT_BUDGET_BYTES``."""
        a, b, length = single.shape[:3]
        per_sample = b * self.n_head * length * length * 4
        return max(1, min(a, self.LOGIT_BUDGET_BYTES // max(per_sample, 1)))

    def delta(
        self,
        single: torch.Tensor,
        cond: torch.Tensor,
        pair: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        compute_dtype: torch.dtype | None = None,
        extra_pair_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The update alone (no residual).

        ``extra_pair_bias`` ``[L, L]``: an additive logit term for every head, sample and batch row, on top of the pair's own bias and before
        the key mask, in the dtype of the bias; differentiable.

        ``compute_dtype`` sets the precision the attention core runs in, per call (``None``: the caller's dtype). The projections, the gate
        and the output stay in the module's dtype; the core's result comes back in the input dtype.
        """
        chunk = self.augment_chunk(single)
        if chunk < single.shape[0]:
            # The logits [A, B, H, L, L] would not fit: the samples are independent, so the call runs on slices of the augmentation axis
            # (a slice size that is a function of the shape only) and the outputs are concatenated. Under autograd every slice is
            # checkpointed, so the backward does not keep every slice's attention at once.
            from torch.utils.checkpoint import checkpoint

            parts = []
            grad = torch.is_grad_enabled() and (single.requires_grad or any(p.requires_grad for p in self.parameters()))
            for i in range(0, single.shape[0], chunk):
                m = mask[i : i + chunk] if mask is not None and mask.dim() == 3 else mask

                def piece(s, c, _m=m):
                    return self.delta(s, c, pair, _m, compute_dtype=compute_dtype, extra_pair_bias=extra_pair_bias)

                s, c = single[i : i + chunk], cond[i : i + chunk]
                parts.append(checkpoint(piece, s, c, use_reentrant=False) if grad else piece(s, c))
            return torch.cat(parts)
        in_dtype = single.dtype
        single = self.ada_ln_in(single, cond)
        query = self.to_query(single)
        key = self.to_key(single)
        value = self.to_value(single)
        query, key = self.flat_qk_norm(query, key)

        num_aug, batch, len_res = query.shape[:3]
        n_head, hidden_dim = self.n_head, query.shape[-1] // self.n_head

        query, key, value = [x.view(num_aug, batch, len_res, n_head, hidden_dim) for x in (query, key, value)]
        if self.use_qk_norm:
            query = self.norm_query(query)
            key = self.norm_key(key)

        bias = self.to_bias(self.ln_pair(pair))
        if extra_pair_bias is not None:
            if extra_pair_bias.ndim != 2 or tuple(extra_pair_bias.shape) != tuple(bias.shape[1:3]):
                raise ValueError(f"extra_pair_bias must be [L, L] = {tuple(bias.shape[1:3])}, got {tuple(extra_pair_bias.shape)}")
            bias = bias + extra_pair_bias.to(bias.dtype)[None, :, :, None]  # [B, L, L, H]: the same term for every head
        if mask is not None and mask.ndim == 2:
            mask = repeat(mask, "B L -> A B L", A=single.shape[0])
        core = self._attention_core_triton if self.backend == "triton" else self._attention_core
        out = core(query, key, value, bias, mask, compute_dtype=compute_dtype)
        # back to the caller's precision before the gate / out projections (the cast is scoped to the core)
        out = out.to(in_dtype)
        out = rearrange(out, "A B L H D -> A B L (H D)")

        out = self.gate_project(single, out)
        return sigmoid_gate(self.to_scale(cond), out)

    # ------------------------------------------------------------------------------------------------------------- "triton"
    # ``_forward_pytorch`` / ``delta`` with the attention core on the Triton pair-bias attention kernel (forward and backward); the AdaLN,
    # projections, q / k norms and gates are the module's statements. bf16 / fp32 / master, head dim <= 64.
    def _attention_core_triton(self, query, key, value, bias, mask=None, compute_dtype=None):
        if compute_dtype is not None:
            query, key, value, bias = (t.to(compute_dtype) for t in (query, key, value, bias))
        return _triton_core(query, key, value, bias, mask)

    def _supports_triton(self, single, cond, pair, mask=None, *, compute_dtype=None, extra_pair_bias=None) -> str | None:
        if compute_dtype not in (None, torch.bfloat16, torch.float32):
            return f"compute_dtype {compute_dtype} (bfloat16 / float32 only)"
        if mask is not None and (mask.dtype != torch.bool or mask.ndim not in (2, 3)):
            return "mask must be a bool [B, L] or [A, B, L] key mask"
        return _triton_core_reason(single, self.to_query.weight.shape[0] // self.n_head)

    def _forward_triton(self, single, cond, pair, mask=None, *, compute_dtype=None, extra_pair_bias=None):
        return self._forward_pytorch(single, cond, pair, mask, compute_dtype=compute_dtype, extra_pair_bias=extra_pair_bias)

    # ------------------------------------------------------------------------------------------- the parts a model's variant replaces
    def flat_qk_norm(self, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """A norm of the FLAT q / k projections ``[..., n_head * d]``, before the head split: none here (the per-head RMSNorm of
        ``use_qk_norm`` follows the split); :class:`FlatQKNormAugmentedAttention`: one LayerNorm over all heads."""
        return query, key

    def gate_project(self, a: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """The gating query and the output projection of the attention's result ``out``: ``W_o (sigmoid(W_g a) * out)``, ``a`` the
        AdaLN output. (:class:`UngatedAugmentedAttention`: no gating query.)"""
        return self.to_out(sigmoid_gate(self.to_gate(a), out))


class FlatQKNormAugmentedAttention(AugmentedAttentionPairBias):
    """RoseTTAFold3's DiT attention: :class:`AugmentedAttentionPairBias` with an affine LayerNorm of the flat q / k, over all heads at once
    (``norm_query`` / ``norm_key``, each ``LayerNorm(n_head * d)``, computed in fp32) in place of the per-head RMSNorm, and a bias-free query
    projection (RF3 ``AttentionPairBiasDiffusion.to_q``: ``MultiDimLinear(..., bias=False)``).

    Invariant: ``to_query.bias is None`` (the release has no query bias). A checkpoint layout that stores one for RoseTTAFold3 (an all-zero
    tensor) is the bias-free query: drop it after checking it is zero; a non-zero value is a conversion error. Used by
    :class:`~miniworld_engine.modules.token_dit.TokenDiTRoseTTAFold3` and :class:`~miniworld_engine.modules.atom_dit.AtomDiTRoseTTAFold3`."""

    def __init__(self, d_single: int, d_cond: int, d_pair: int, n_head: int, *, backend: str = "pytorch") -> None:
        super().__init__(d_single, d_cond, d_pair, n_head, query_bias=False, backend=backend)
        d_hidden = d_single // n_head
        self.norm_query = LayerNorm(d_hidden * n_head)
        self.norm_key = LayerNorm(d_hidden * n_head)

    def flat_qk_norm(self, query, key):
        return self.norm_query(query), self.norm_key(key)


class UngatedAugmentedAttention(AugmentedAttentionPairBias):
    """Chai-1's DiT attention: :class:`AugmentedAttentionPairBias` without the gating query (no ``to_gate``): ``W_o out``."""

    def __init__(self, d_single: int, d_cond: int, d_pair: int, n_head: int, *, backend: str = "pytorch") -> None:
        super().__init__(d_single, d_cond, d_pair, n_head, backend=backend)
        del self.to_gate

    def gate_project(self, a, out):
        return self.to_out(out)


class BiasOnlyAttention(BackendModule):
    """Pair-bias attention without queries and keys, with adaptive conditioning (the bias-only token DiT's attention).

    ``out = single + sigmoid(to_scale(cond)) * to_out(sigmoid(g) * softmax(bias) v)`` (the residual is the module's own input), with
    ``bias = to_bias(LN(pair))`` [B, H, L, L] shared by every augmented sample, ``v`` and ``g`` projections of ``AdaLN(single, cond)``. A key
    mask [B, L] enters as the largest negative finite logit (a fully masked row is uniform, not NaN). ``bias=`` [B, H, L, L] given (a hoisted
    pair bias, :func:`~miniworld_engine.modules.token_dit.pair_bias_all`): those are the logits, and ``ln_pair`` / ``to_bias`` / ``pair`` are
    not read.

    Everything around the core is :class:`AugmentedAttentionPairBias`'s (AdaLN, the value and gate projections, both sigmoid gates, the
    conditioning scale), minus ``to_query``, ``to_key`` and the QK-norm. With no query and no key the attention weights depend on neither the
    sample nor the single representation, only on the pair: ONE softmax serves every augmented sample.

    ``backend="triton"`` (engine 2.4's portable path, ``kernels/bias_only_attention/triton``): the AdaLN on its ``"triton"`` backend, the pair
    LayerNorm + bias projection in one pass (``pair_bias.pair_bias_logits``, head-major logits, forward and backward), the value and gate
    projections as ONE GEMM over the concatenated weights; with a gradient the feature-major training core (``core_t.bias_only_core_t``: the
    value / gate GEMM, ``softmax(bias) v`` as a batched GEMM over the heads, the sigmoid gate in Triton, ``to_out``; no permute copies), one pair
    for every sample or a pair per sample; inference: the batched einsum (one softmax for every sample, a cuBLAS GEMM per head -- the flash
    bias-only kernel of ``triton/main.py`` does not beat it, as engine 2.4 measured). bf16 or fp32 activations, CUDA.
    """

    backends = ("pytorch", "triton")

    def __init__(
        self,
        d_single: int,
        d_cond: int,
        d_pair: int,
        n_head: int,
        d_head: int | None = None,
        *,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        self.n_head = n_head
        d_hidden = d_head or d_single // n_head  # the head width; n_head x d_head may differ from d_single
        self.ada_ln_in = AdaptiveLayerNorm(d_single, d_cond)
        self.to_value = Linear(d_single, d_hidden * n_head, bias=False)
        self.to_gate = Linear(d_single, d_hidden * n_head, bias=False, init="gating")
        # no offset, as in AugmentedAttentionPairBias: a per-head constant on every logit cancels in the softmax
        self.ln_pair = LayerNorm(d_pair, bias=False)
        self.to_bias = Linear(d_pair, n_head, bias=False, init="zero")
        self.to_out = Linear(d_hidden * n_head, d_single, bias=False, init="zero")
        self.to_scale = Linear(d_cond, d_single, bias=True, init="default", bias_init=-2.0)  # sigmoid(-2) ~ 0.1: the update starts small
        self.set_backend(backend)

    def _attention_weights(self, pair, mask, bias, dtype):
        """softmax over the keys of the pair bias (masked keys at the largest negative finite logit): [B, H, L, L], shared by every sample."""
        # (B, H, L, L) from the pair, or the hoisted bias (pair_bias_all), head-major already
        logits = self.to_bias(self.ln_pair(pair)).permute(0, 3, 1, 2) if bias is None else bias.to(dtype)
        if mask is not None:
            logits = logits.masked_fill(~mask[:, None, None, :], torch.finfo(logits.dtype).min)
        return torch.softmax(logits, dim=-1)

    def _forward_pytorch(
        self,
        single: torch.Tensor,
        cond: torch.Tensor,
        pair: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        *,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``single`` ``[A, B, L, d_single]``, ``cond`` ``[A, B, L, d_cond]``, ``pair`` ``[B, L, L, d_pair]``, ``mask`` ``[B, L]``."""
        # residual == the ORIGINAL input
        single, single_res = self.ada_ln_in.forward_with_stream(single, cond)
        value, gate = self.to_value(single), self.to_gate(single)
        value = rearrange(value, "A B L (H D) -> A B L H D", H=self.n_head)
        # one softmax for every sample: the logits carry no augmentation axis
        attention = self._attention_weights(pair, mask, bias, value.dtype)
        out = torch.einsum("bhij,abjhd->abihd", attention, value).flatten(-2)
        out = self.to_out(sigmoid_gate(gate, out))
        return single_res + sigmoid_gate(self.to_scale(cond), out)

    # ------------------------------------------------------------------------------------------------------------- "triton"
    def set_backend(self, backend: str = "pytorch"):
        super().set_backend(backend)
        ada = getattr(self, "ada_ln_in", None)  # absent while nn.Module.__init__ runs the first set_backend
        if isinstance(ada, BackendModule):
            ada.set_backend(backend if backend in ada.backends else "pytorch")
        return self

    def _triton_core_t(self, single: torch.Tensor, src: torch.Tensor | None) -> bool:
        """The feature-major training core takes the call: a gradient, ``single`` [A, B, L, d] with ONE pair (B = 1) or a pair per sample
        (A = 1, the logits' batch = B)."""
        if single.ndim != 4 or not _wants_grad(self, single):
            return False
        return single.shape[1] == 1 or (single.shape[0] == 1 and src is not None and src.shape[0] == single.shape[1])

    def _supports_triton(self, single, cond, pair=None, mask=None, *, bias=None) -> str | None:
        from miniworld_engine.kernels.bias_only_attention.triton.pair_bias import supported as pair_bias_supported

        why = triton_unsupported(single) or _activation_dtype_reason(single, cond)
        if why is not None:
            return why
        if single.ndim != 4 or tuple(single.shape[:-1]) != tuple(cond.shape[:-1]):
            return f"single / cond must be [A, B, L, d] with the same leading shape, got {tuple(single.shape)} / {tuple(cond.shape)}"
        if bias is None:
            if pair is None:
                return "the pair or the block's precomputed bias"
            if pair.dtype != single.dtype:
                return f"pair dtype {pair.dtype} (the activations': {single.dtype})"
            if self.ln_pair.bias is not None or self.to_bias.bias is not None or self.ln_pair.weight is None:
                return "ln_pair with a weight and no offset, to_bias without a bias (the one-pass pair-bias kernel)"
            if not pair_bias_supported(pair, self.n_head):
                return "pair must be contiguous with a power-of-two width 16 .. 512 and at most 64 heads (the one-pass pair-bias kernel)"
        if mask is not None and (mask.dtype != torch.bool or mask.ndim != 2):
            return "mask must be a bool [B, L] key mask"
        return self.ada_ln_in._supports_triton(single, cond)

    def _forward_triton(self, single, cond, pair=None, mask=None, *, bias=None) -> torch.Tensor:
        from miniworld_engine.kernels.bias_only_attention.triton.core_t import bias_only_core_t
        from miniworld_engine.kernels.bias_only_attention.triton.pair_bias import pair_bias_logits

        x, single_res = self.ada_ln_in.forward_with_stream(single, cond)  # the AdaLN's "triton" backend (set_backend)
        dt = single.dtype
        wv, wg, wo = (m.weight.to(dt) for m in (self.to_value, self.to_gate, self.to_out))
        with torch.autocast("cuda", enabled=False):
            # [B, H, L, L], head-major: the one-pass pair LayerNorm + projection, or the hoisted bias
            if bias is None:
                assert pair is not None, "the pair or its hoisted bias"
                logits = pair_bias_logits(pair, self.ln_pair.weight, self.to_bias.weight, self.ln_pair.eps)
            else:
                logits = bias.to(dt)
            if mask is not None:
                logits = logits.masked_fill(~mask[:, None, None, :], torch.finfo(logits.dtype).min)
            attention = torch.softmax(logits.float(), dim=-1).to(dt)  # [B, H, L, L], one softmax for every sample
            if self._triton_core_t(single, pair if bias is None else bias):
                p = attention[0] if single.shape[1] == 1 else attention
                out = bias_only_core_t(x.reshape(-1, x.shape[-1]), wv, wg, wo, p).reshape(single.shape)
            else:
                value, gate = F.linear(x, torch.cat([wv, wg], 0)).split(wv.shape[0], dim=-1)
                value = rearrange(value, "A B L (H D) -> A B L H D", H=self.n_head)
                out = torch.einsum("bhij,abjhd->abihd", attention, value).flatten(-2)
                out = F.linear(sigmoid_gate(gate, out), wo)
            scale = F.linear(cond, self.to_scale.weight.to(dt), self.to_scale.bias.to(dt))
            return single_res + sigmoid_gate(scale, out)
