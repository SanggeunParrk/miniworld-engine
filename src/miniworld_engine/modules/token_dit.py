"""The token-track diffusion transformer (AlphaFold 3 Algorithm 23 and the models' variants of it): one class per model, picked by
:class:`TokenDiT`.

    TokenDiT(model="boltz2")                    # -> TokenDiTBoltz2()

Each class is ONE block in the engine layout and hard-codes its model's math (no option flags); :data:`TOKEN_DIT_MODELS` maps the model
names to them. ``attention`` is an :class:`~miniworld_engine.modules.attention.AugmentedAttentionPairBias` (or a model variant of it, or
``BiasOnlyAttention``), ``transition`` a :class:`~miniworld_engine.modules.transition.ConditionedTransition` (or a variant); each part owns
its AdaLN, its pair LayerNorm + projection and its residual. A model's stack is a list of blocks; ``single`` / ``cond`` [A, B, L, d] carry
the augmentation axis, the pair [B, L, L, d_pair] does not::

    sequential    a = attention(a, s, z); a = transition(a, s)                    a' = a + dA(a),  out = a' + dT(a')
    parallel      out = a + attention.delta(a, s, z) + transition.delta(a, s)      (RoseTTAFold3, Chai-1)

    TokenDiTProtenix          sequential, sigmoid AdaLN, gating query, SwiGLU transition; d_cond 384, pair 128
                              (AlphaFold 3, OpenFold3 v0.5 and preview-2, Protenix, Protenix-v2, OpenDDE, IntelliFold2, ESMFold2)
    TokenDiTMiniWorld         the same with the per-head q / k RMSNorm (MiniWorld's QK-norm token DiT)
    TokenDiTMiniWorldBiasOnly the attention without query and key: the logits are the pair bias (MiniWorld's bias-only token DiT)
    TokenDiTBoltz2            TokenDiTProtenix with the up-gate transition (``UpGatedConditionedTransition``); d_cond 768
    TokenDiTRoseTTAFold3      parallel; the affine LayerNorm of the flat q / k and a bias-free query (``FlatQKNormAugmentedAttention``)
    TokenDiTChai1             parallel; identity AdaLN (``IdentityAdaptiveLayerNorm``, epsilon 0.1) in both parts, no
                              gating query (``UngatedAugmentedAttention``); pair 256

A released checkpoint (AlphaFold 3's one shared pair LayerNorm with a projection per super block of four blocks, Chai-1's affine pair
LayerNorm) is converted to these blocks: the shared LayerNorm's weight is replicated per block, the super-block projection sliced into each
block's head rows, and the pair LayerNorm's offset dropped (it adds one constant per head to every key's logit, which the softmax cancels).

Each is a :class:`~miniworld_engine.modules.primitives.BackendModule`; ``_forward_pytorch`` is the reference.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from miniworld_engine.modules import _anthropic
from miniworld_engine.modules.attention import (
    AugmentedAttentionPairBias,
    BiasOnlyAttention,
    FlatQKNormAugmentedAttention,
    UngatedAugmentedAttention,
)
from miniworld_engine.modules.dit_variant_cuda import VariantTokenCuda
from miniworld_engine.modules.functional import sigmoid_gate
from miniworld_engine.modules.norm import IdentityAdaptiveLayerNorm
from miniworld_engine.modules.primitives import BackendModule, key_mask, random_input, wide
from miniworld_engine.modules.transition import ConditionedTransition, UpGatedConditionedTransition


def _token_dit_inputs(block, *, length, samples, dtype, device, generator) -> dict:
    """A token DiT block's measurement inputs: ``single`` / ``cond`` [S, 1, L, d] (S samples on the augmentation axis), ``pair`` [1, L, L, d_pair],
    the key ``mask`` [1, L]."""
    d_single, d_cond, d_pair = block.transition.d_hidden, block.transition.d_cond, block.attention.ln_pair.normalized_shape[0]
    return {"single": random_input((samples, 1, length, d_single), dtype=dtype, device=device, generator=generator),
            "cond": random_input((samples, 1, length, d_cond), dtype=dtype, device=device, generator=generator),
            "pair": random_input((1, length, length, d_pair), dtype=dtype, device=device, generator=generator),
            "mask": key_mask(1, length, device=device)}


# ================================================================================================= the "anthropic" backend (the release)
# One adapter per release op; the block's statements around the attention core are the ones of ``AugmentedAttentionPairBias.delta``
# (``_release_delta``), the core is the model's release kernel. Every release op here is forward only (no backward).


def _grad_reason(module: torch.nn.Module, *tensors) -> str | None:
    if torch.is_grad_enabled() and (any(p.requires_grad for p in module.parameters())
                                    or any(isinstance(t, torch.Tensor) and t.requires_grad for t in tensors)):
        return "the release op has no backward: call it under torch.no_grad() / inference_mode() (eval only)"
    return None


def _core_dtype(single: torch.Tensor) -> torch.dtype:
    """The dtype the attention core sees: the activation's, or the autocast dtype (the master-weight layout)."""
    return torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else single.dtype


def _token_mask_reason(mask: torch.Tensor | None, single: torch.Tensor) -> str | None:
    if mask is None:
        return None
    if mask.ndim not in (2, 3) or tuple(mask.shape[-2:]) != tuple(single.shape[1:3]):
        return f"mask must be [B, L] or [A, B, L], got {tuple(mask.shape)}"
    if not bool(mask.any(-1).all()):
        return "a (sample, batch) row with no valid key (the release kernels average such a row, the reference gives 0)"
    return None


def _release_delta(att, single, cond, pair, mask, extra_pair_bias, core, *, fused_gate: bool = False) -> torch.Tensor:
    """``AugmentedAttentionPairBias.delta`` (no residual) with ``core(q, k, v, bias, mask, gate)`` -> [A, B, L, H * D] as its attention core:
    q / k / v [A, B, L, H, D], bias [B, L, L, H] (+ ``extra_pair_bias``), mask [A, B, L] or None, gate the gating query's logits
    ``to_gate(a)`` [A, B, L, H * D] when the release op fuses ``sigmoid(gate) *`` (``fused_gate``; then only ``to_out`` follows), else None."""
    in_dtype = single.dtype
    x = att.ada_ln_in(single, cond)
    query, key = att.flat_qk_norm(att.to_query(x), att.to_key(x))
    value = att.to_value(x)
    a, b, n = query.shape[:3]
    h = att.n_head
    d = query.shape[-1] // h
    query, key, value = (t.view(a, b, n, h, d) for t in (query, key, value))
    bias = att.to_bias(att.ln_pair(pair))
    if extra_pair_bias is not None:
        bias = bias + extra_pair_bias.to(bias.dtype)[None, :, :, None]
    if mask is not None and mask.ndim == 2:
        mask = mask[None].expand(a, -1, -1)
    gate = att.to_gate(x) if fused_gate else None
    out = core(query, key, value, bias, mask, gate).to(in_dtype)
    out = att.to_out(out) if fused_gate else att.gate_project(x, out)
    return sigmoid_gate(att.to_scale(cond), out)


def _core_af3_apb(q, k, v, bias, mask, gate):
    """af3_torch's token core in 16 bit: ``opt_core.attn.apb_core.pair_bias_attention`` (the apb_attn Triton kernel the kit's DTK step and
    its ``apb`` lever call), one launch per batch row for its A samples sharing the row's bias."""
    apb = _anthropic.load("opt_core.attn.apb_core")
    outs = []
    for j in range(q.shape[1]):
        km = None if mask is None else mask[:, j]
        outs.append(apb.pair_bias_attention(q[:, j].transpose(1, 2), k[:, j].transpose(1, 2), v[:, j].transpose(1, 2),
                                            bias[j].permute(2, 0, 1), km, layout="shnd"))   # [A, L, H * D]
    return torch.stack(outs, 1)


def _core_dtk_flash(q, k, v, bias, mask, gate, *, dtype=None):
    """``opt_core.kernels.dtk_kernels.flash_bias_attn``, one [H, L, D] problem per (sample, batch row): af3_torch's DTK per-sample core
    (its fp32 path), and the core RoseTTAFold3's kit calls (``rosettafold3_opt.dtk.attn``: bias re-laid in the activation dtype, no mask)."""
    dk = _anthropic.load("opt_core.kernels.dtk_kernels")
    a, b, n, h, d = q.shape
    dt = dtype or q.dtype
    out = torch.empty(a, b, n, h * d, dtype=dt, device=q.device)
    for j in range(b):
        bh = bias[j].permute(2, 0, 1).to(dt).contiguous()                      # [H, L, L]
        for i in range(a):
            km = None if mask is None else mask[i, j].to(torch.int32)
            dk.flash_bias_attn(q[i, j].to(dt).transpose(0, 1), k[i, j].to(dt).transpose(0, 1), v[i, j].to(dt).transpose(0, 1),
                               bias=bh, key_mask=km, out=out[i, j], out_dtype=dt, scale=d ** -0.5)
    return out



# ================================================================================================= the "cuda" backend (the engine's H100 kernels)
# The fused H100 block paths of kernels/integrations (token_dit: inference; token_dit_train: the autograd Function; bias_only_dit /
# bias_only_dit_train likewise for the bias-only block). Each returns the whole block (both residuals) and reads the block's own
# parameters by name. Their gates (``serves``) are the kernels' ranges; they still test the attributes the removed dispatch set
# (``implementation`` / ``attention._backend``), which :func:`_kernel_view` supplies with the engine-kernel values.


class _View:
    """``module`` with some attributes overridden (read-only forwarding of everything else, parameters included)."""

    def __init__(self, module, **over):
        object.__setattr__(self, "_module", module)
        object.__setattr__(self, "_over", over)

    def __getattr__(self, name):
        over = object.__getattribute__(self, "_over")
        return over[name] if name in over else getattr(object.__getattribute__(self, "_module"), name)


def _kernel_view(block):
    return _View(block, implementation="miniworld", attention=_View(block.attention, _backend="portable"))


def _h100_reason(single: torch.Tensor) -> str | None:
    if not single.is_cuda:
        return "CUDA tensors only"
    if torch.cuda.get_device_capability(single.device) != (9, 0):
        return "the H100 (sm_90) kernels only"
    return None


def _wants_grad(module: torch.nn.Module, *tensors) -> bool:
    return torch.is_grad_enabled() and (any(isinstance(t, torch.Tensor) and t.requires_grad for t in tensors)
                                        or any(p.requires_grad for p in module.parameters()))


def _shared_key_mask(mask: torch.Tensor | None) -> tuple[torch.Tensor | None, bool]:
    """(the [B, L] key mask the samples share, True), or (``mask``, False) for an [A, B, L] mask that may differ per sample. An [A, B, L]
    mask is shared when A == 1 or it is a [B, L] mask expanded over A (stride 0): metadata only, no device read."""
    if mask is None or mask.ndim == 2:
        return mask, True
    if mask.ndim == 3 and (mask.shape[0] == 1 or mask.stride(0) == 0):
        return mask[0], True
    return mask, False


class TokenDiTProtenix(BackendModule):
    """AF3 Alg. 23 in the engine layout: pair-bias attention, then the conditioned transition::

        a = AugmentedAttentionPairBias(a, s, z, mask)     # a + attention(a, s, z)
        a = ConditionedTransition(a, s)                  # a + transition(a, s)

    ``forward(single, cond, pair, mask)`` -> ``single``'s shape. ``single`` and ``cond`` carry the augmentation axis (``A, B, L, d``);
    ``pair`` does not (``B, L, L, d_pair``), because the pair bias is shared across augmentations. Both parts build their own
    ``AdaptiveLayerNorm`` from their ``(d_hidden, d_cond)``. At atom widths (128 / 128 / pair 16 / 4 heads) it is also the dense atom DiT.

    AlphaFold 3's own token DiT is this block (registered as ``"af3"``).

    ``backend="anthropic"``: the release's af3_torch attention core (bf16: ``opt_core.attn.apb_core``; fp32: ``dtk_kernels.flash_bias_attn``),
    the rest of the block as ``"pytorch"``; eval only, CUDA H100, head dim <= 64 (:meth:`_supports_anthropic`).
    """
    version = 3   # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "anthropic", "cuda", "triton")

    #: The q / k RMSNorm of the attention (:class:`TokenDiTMiniWorld`).
    qk_norm: bool = False
    #: The parallel schedule ``a + dA(a) + dT(a)`` (:class:`TokenDiTRoseTTAFold3`, :class:`TokenDiTChai1`).
    parallel: bool = False
    #: The parts (a model's variant of each).
    attention_part: type[AugmentedAttentionPairBias] = AugmentedAttentionPairBias
    transition_part: type[ConditionedTransition] = ConditionedTransition

    def __init__(
        self,
        d_single: int = 768,
        d_cond: int = 384,
        d_pair: int = 128,
        n_head: int = 16,
        n: int = 2,
        *,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        qk = {"use_qk_norm": True} if self.qk_norm else {}
        self.attention = self.attention_part(d_single=d_single, d_cond=d_cond, d_pair=d_pair, n_head=n_head, **qk)
        self.transition = self.transition_part(d_hidden=d_single, d_cond=d_cond, n=n)
        self.set_backend(backend)

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
        """AF3 Alg. 23; each part returns its residual output (stream in, stream out).

        ``mask`` is a key mask, ``[B, L]`` or ``[A, B, L]``. ``extra_pair_bias`` ``[L, L]`` is added to every head's attention logits, on
        top of the pair's own bias (``AugmentedAttentionPairBias``'s ``extra_pair_bias``). ``compute_dtype``: the precision of the attention
        core."""
        kw = {"compute_dtype": compute_dtype} if compute_dtype is not None else {}
        if extra_pair_bias is not None:
            kw["extra_pair_bias"] = extra_pair_bias
        if self.parallel:
            return single + self.attention.delta(single, cond, pair, mask, **kw) + self.transition.delta(single, cond)
        single = self.attention(single, cond, pair, mask, **kw)
        return self.transition(single, cond)

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        return _token_dit_inputs(self, length=length, samples=samples, dtype=dtype, device=device, generator=generator)

    # ------------------------------------------------------------------------------------------------------------ "triton"
    # The block's own statements with the attention core on the Triton pair-bias attention kernel (forward and backward): the block's
    # backend is set on its attention part (``set_backend``); the transition and every projection are the module's statements.
    def set_backend(self, backend: str = "pytorch"):
        super().set_backend(backend)
        self.attention.set_backend("triton" if backend == "triton" else "pytorch")
        return self

    def _supports_triton(self, single, cond, pair, mask=None, *, compute_dtype=None, extra_pair_bias=None) -> str | None:
        return self.attention._supports_triton(single, cond, pair, mask, compute_dtype=compute_dtype, extra_pair_bias=extra_pair_bias)

    def _forward_triton(self, single, cond, pair, mask=None, *, compute_dtype=None, extra_pair_bias=None) -> torch.Tensor:
        return self._forward_pytorch(single, cond, pair, mask, compute_dtype=compute_dtype, extra_pair_bias=extra_pair_bias)

    # ------------------------------------------------------------------------------------------------------------ "cuda"
    # H100: no gradient -> kernels/integrations/token_dit (the fused inference block: hoisted pair bias, CUDA rows, cuBLAS, the sm_90a
    # gated attention core); a gradient -> kernels/integrations/token_dit_train (one autograd Function, forward + backward on CUDA rows,
    # cuBLAS and the sm_90a cores). Range = the two gates: bf16 / fp32, [A, 1, L, 768] (B > 1 trunk_seeds: bf16 inference only, one
    # pair per trunk_seed, the key mask expanded over B), 16 heads x 48, cond 384 or 768, pair 128 / 256 / 512 (the fused blocks' row
    # kernels and weight packs are built per cond / pair width), transition n = 2, LayerNorm eps 1e-5,
    # a key mask shared by the samples; inference L a multiple of 128, training any L (padded to 128).
    def _cuda_route(self, single, cond, pair, mask, compute_dtype, extra_pair_bias):
        """("infer" | "train", the [B, L] key mask) or (None, why not)."""
        why = _h100_reason(single)
        if why is not None:
            return None, why
        km, shared = _shared_key_mask(mask)
        if not shared:
            return None, "a key mask that differs across the samples (the fused blocks take one [B, L] mask)"
        view = _kernel_view(self)
        if _wants_grad(self, single, cond, pair, extra_pair_bias):
            from miniworld_engine.kernels.integrations import token_dit_train

            if token_dit_train.serves(view, single, cond, pair, km, compute_dtype, extra_pair_bias):
                return "train", km
            return None, ("training: outside the fused H100 training block's range (bf16 / fp32, [A, 1, L, d] with heads x head width "
                          "16 x 48 / 24 x 32 / 12 x 64 (d 768) or 16 x 64 (d 1024), cond 384 or 768, pair 128 / 256 / 512, n = 2, key mask [1, L], eps 1e-5)")
        if single.shape[1] > 1 and km is not None and km.stride(0) != 0:
            return None, "B > 1 trunk_seeds take one key mask expanded over B (stride 0)"
        from miniworld_engine.kernels.integrations import token_dit

        if token_dit.serves(view, single, cond, pair, compute_dtype, extra_pair_bias):
            return "infer", km
        return None, ("inference: outside the fused H100 inference block's range (bf16 / fp32, [A, 1, L, d] with L a multiple of 128 and "
                      "heads x head width 16 x 48 / 24 x 32 / 12 x 64 (d 768) or 16 x 64 (d 1024), cond 384 or 768, pair 128 / 256 / 512, n = 2, eps 1e-5)")

    def _supports_cuda(self, single, cond, pair, mask=None, *, compute_dtype=None, extra_pair_bias=None) -> str | None:
        route, why = self._cuda_route(single, cond, pair, mask, compute_dtype, extra_pair_bias)
        return why if route is None else None

    def _forward_cuda(self, single, cond, pair, mask=None, *, compute_dtype=None, extra_pair_bias=None) -> torch.Tensor:
        route, km = self._cuda_route(single, cond, pair, mask, compute_dtype, extra_pair_bias)
        view = _kernel_view(self)
        if route == "train":
            from miniworld_engine.kernels.integrations import token_dit_train

            return token_dit_train.block(view, single, cond, pair, km, compute_dtype, extra_pair_bias)
        from miniworld_engine.kernels.integrations import token_dit

        return token_dit.update(view, single, cond, pair, km, extra_pair_bias)

    # ------------------------------------------------------------------------------------------------------------ "anthropic"
    #: the release op this class's attention core runs on, and whether that op fuses the gating query's sigmoid
    _release_module = "opt_core.attn.apb_core"
    _release_path = "common/opt_core"
    _release_fused_gate = False

    def _release_core(self, q, k, v, bias, mask, gate):
        if q.dtype in (torch.bfloat16, torch.float16):
            return _core_af3_apb(q, k, v, bias, mask, gate)
        return _core_dtk_flash(q, k, v, bias, mask, gate)

    def _release_range(self, single, cond, pair, mask) -> str | None:
        """The class's own limits beyond the shared ones (dtype, mask, head width)."""
        if _core_dtype(single) not in (torch.bfloat16, torch.float32):
            return f"core dtype {_core_dtype(single)} (bfloat16 / float32 only)"
        if single.shape[-1] // self.attention.n_head > 64:
            return "head dim > 64 (the release kernels' limit)"
        return _anthropic.missing("opt_core.kernels.dtk_kernels")

    def _supports_anthropic(self, single, cond, pair, mask=None, *, compute_dtype=None, extra_pair_bias=None) -> str | None:
        why = _anthropic.unavailable(self._release_module, single.device, path=self._release_path)
        if why is not None:
            return why
        if compute_dtype is not None:
            return "compute_dtype (the release core runs in the activations' dtype)"
        if getattr(self.attention, "use_qk_norm", False):
            return "per-head q / k RMSNorm (no release op)"
        return (_grad_reason(self, single, cond, pair, extra_pair_bias) or _token_mask_reason(mask, single)
                or self._release_range(single, cond, pair, mask))

    def _forward_anthropic(self, single, cond, pair, mask=None, *, compute_dtype=None, extra_pair_bias=None) -> torch.Tensor:
        """``_forward_pytorch`` with the attention core on the release op (``_release_core``)."""
        delta = _release_delta(self.attention, single, cond, pair, mask, extra_pair_bias, self._release_core,
                               fused_gate=self._release_fused_gate)
        if self.parallel:
            return single + delta + self.transition.delta(single, cond)
        return self.transition(single + delta, cond)


class TokenDiTMiniWorld(TokenDiTProtenix):
    """MiniWorld's token DiT block: :class:`TokenDiTProtenix` with the per-head q / k RMSNorm (``attention.norm_query`` /
    ``attention.norm_key``). No ``"anthropic"`` backend: no release folder has the QK-RMSNorm token DiT."""
    version = 1   # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "cuda", "triton")
    qk_norm = True


class TokenDiTBoltz2(VariantTokenCuda, TokenDiTProtenix):
    """Boltz-2's token DiT block: :class:`TokenDiTProtenix` with the up-gate transition ``silu(W_a h) (W_b h) (W_u h)``
    (``transition.up``); the conditioning is 768 wide."""
    version = 2   # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "anthropic", "cuda", "triton")   # "cuda": kernels/dit_variants (modules/dit_variant_cuda.py)

    transition_part = UpGatedConditionedTransition

    def __init__(self, d_single: int = 768, d_cond: int = 768, d_pair: int = 128, n_head: int = 16, n: int = 2, *,
                 backend: str = "pytorch") -> None:
        super().__init__(d_single, d_cond, d_pair, n_head, n, backend=backend)

    # "anthropic": Boltz-2's own flash pair-bias kernel (boltz2/opt/forward/sampler/src/bz_sampler_dit.py ``_flash_bias_kernel``, the core of
    # its ``dit_fused`` sampler lever): rows [q | k | v | g], sigmoid(g) fused, one bias per batch row shared by the samples with the key mask
    # folded in as Boltz's ``(1 - mask) * -1e6``; bf16 operands (bf16) or ieee fp32 dots (fp32); head dim 48 only.
    _release_module = "bz_sampler_dit"
    _release_path = "boltz2/opt/forward/sampler/src"
    _release_fused_gate = True

    def _release_core(self, q, k, v, bias, mask, gate):
        import triton

        bz = _anthropic.load(self._release_module, path=self._release_path)
        a, b, n, h, d = q.shape
        c, dt = h * d, q.dtype
        prec = "bf16" if dt == torch.bfloat16 else "ieee"
        bq, bkv, warps, stages = bz._flash_cfg(n, prec)
        qb = torch.zeros(c, dtype=dt, device=q.device)              # the q bias is in q already (to_query's bias)
        out = torch.empty(a, b, n, c, dtype=dt, device=q.device)
        for j in range(b):
            rows = torch.cat([t[:, j].reshape(a * n, c) for t in (q, k, v, gate)], -1).contiguous()   # [A * L, 4 C]
            pb = bias[j].permute(2, 0, 1).float()
            if mask is not None:
                pb = pb + (1 - mask[0, j].float())[None, None, :] * -1e6
            pb = pb.to(dt).contiguous()
            ctx = torch.empty(a * n, c, dtype=dt, device=q.device)
            bz._flash_bias_kernel[(triton.cdiv(n, bq), h, a)](
                rows, qb, pb, ctx, n, rows.stride(0), pb.stride(0), pb.stride(1), ctx.stride(0), float(d ** -0.5),
                H=h, HD=d, D=c, PREC=3 if prec == "bf16" else 0, OUT_BF16=dt == torch.bfloat16, BQ=bq, BKV=bkv,
                num_warps=warps, num_stages=stages)
            out[:, j] = ctx.view(a, n, c)
        return out

    def _release_range(self, single, cond, pair, mask) -> str | None:
        if _core_dtype(single) not in (torch.bfloat16, torch.float32):
            return f"core dtype {_core_dtype(single)} (bfloat16 / float32 only)"
        if single.shape[-1] // self.attention.n_head != 48:
            return "head dim != 48 (Boltz-2's kernel is written for 32 + 16 columns per head)"
        if mask is not None and mask.ndim == 3 and not bool((mask == mask[:1]).all()):
            return "a key mask that differs across the samples (Boltz-2 folds one mask per batch row into the shared bias)"
        return None


class TokenDiTRoseTTAFold3(VariantTokenCuda, TokenDiTProtenix):
    """RoseTTAFold3's token DiT block (``af3_diffusion_transformer.py``): the parallel schedule ``a + dA(a) + dT(a)``
    (``no_residual_connection_between_attention_and_transition: true``), a bias-free query (``to_q``: ``bias=False``) and an affine LayerNorm
    of the flat q / k (``kq_norm``: ``attention.norm_query`` / ``attention.norm_key``, over all heads)."""
    version = 2   # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "anthropic", "cuda", "triton")   # "cuda": kernels/dit_variants (modules/dit_variant_cuda.py)

    parallel = True
    attention_part = FlatQKNormAugmentedAttention

    def make_inputs(self, **kwargs) -> dict:
        return {**super().make_inputs(**kwargs), "mask": None}   # the model calls the block without a key mask

    # "anthropic": the core RoseTTAFold3's kit runs (rosettafold3/opt/rosettafold3_opt/dtk.py ``attn``: opt_core's
    # ``dtk_kernels.flash_bias_attn`` per sample, the bias re-laid in the activation dtype, no key mask; the gate stays outside).
    _release_module = "opt_core.kernels.dtk_kernels"

    def _release_core(self, q, k, v, bias, mask, gate):
        return _core_dtk_flash(q, k, v, bias, None, None)

    def _release_range(self, single, cond, pair, mask) -> str | None:
        if _core_dtype(single) not in (torch.bfloat16, torch.float32):
            return f"core dtype {_core_dtype(single)} (bfloat16 / float32 only)"
        if single.shape[-1] // self.attention.n_head > 64:
            return "head dim > 64 (the kernel's limit)"
        if mask is not None and not bool(mask.all()):
            return "a key mask (RoseTTAFold3's release core takes none)"
        return None


class TokenDiTChai1(VariantTokenCuda, TokenDiTProtenix):
    """Chai-1's token DiT block: the parallel schedule, the identity AdaLN ``(1 + W_s s) LN(a) + W_b s`` in both parts
    (``attention.ada_ln_in``, ``transition.ada_ln_in``), no gating query; the pair is 256 wide. ``adaln_eps``: the epsilon of both AdaLNs'
    LayerNorm of ``a`` (Chai-1's release, its traced ``layer_norm(x, None, None, 0.1)``: 0.1, the default)."""
    version = 2   # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "anthropic", "cuda", "triton")   # "cuda": kernels/dit_variants (modules/dit_variant_cuda.py)

    parallel = True
    attention_part = UngatedAugmentedAttention

    def make_inputs(self, **kwargs) -> dict:
        return {**super().make_inputs(**kwargs), "mask": None}   # the model calls the block without a key mask

    def __init__(self, d_single: int = 768, d_cond: int = 384, d_pair: int = 256, n_head: int = 16, n: int = 2, *,
                 adaln_eps: float = 0.1, backend: str = "pytorch") -> None:
        super().__init__(d_single, d_cond, d_pair, n_head, n, backend=backend)
        self.attention.ada_ln_in = IdentityAdaptiveLayerNorm(d_single, d_cond, eps=adaln_eps)
        self.transition.ada_ln_in = IdentityAdaptiveLayerNorm(d_single, d_cond, eps=adaln_eps)

    # "anthropic": the core Chai-1's kit serves (chai1/opt/forward/dstep_megakernel/chai1_fastln/dit_attn.py ``_serve4``): opt_core's
    # pair-bias provider with the row it selects for fp32 cell dit_h16d48 under the tier word "fast" (graphed, capture), [S, H, N, D] with one
    # [1, H, N, N] bias, no gate, no key mask (Chai-1 folds its mask into the bias). A stock row the provider selects (SDPA ...) is served as the release serves it.
    _release_module = "opt_core.kernels.apb"
    _CELL = "dit_h16d48"

    def _chai_selection(self, n: int, samples: int, d: int, h: int, device):
        provider = _anthropic.load(self._release_module)
        cc = tuple(torch.cuda.get_device_capability(device))
        return _anthropic.kernel_selection(provider, cc, torch.float32, self._CELL, int(n), word="fast", samples=int(samples),
                                           timing="graphed", capture=True, head_dim=int(d), heads=int(h))

    def _release_core(self, q, k, v, bias, mask, gate):
        provider = _anthropic.load(self._release_module)
        a, b, n, h, d = q.shape
        sel, why = self._chai_selection(n, a, d, h, q.device)
        if sel is None:
            raise RuntimeError(why)
        out = torch.empty(a, b, n, h * d, dtype=q.dtype, device=q.device)
        for j in range(b):
            o, _ = provider.pair_bias_attention(q[:, j].transpose(1, 2), k[:, j].transpose(1, 2), v[:, j].transpose(1, 2),
                                                bias[j].permute(2, 0, 1)[None], selection=sel, layout="shnd", cell=self._CELL, capture=True)
            out[:, j] = o.transpose(1, 2).reshape(a, n, h * d)
        return out

    def _release_range(self, single, cond, pair, mask) -> str | None:
        if _core_dtype(single) != torch.float32:
            return f"core dtype {_core_dtype(single)} (Chai-1's release core serves float32 only)"
        if mask is not None and not bool(mask.all()):
            return "a key mask (Chai-1's release core takes the mask inside its bias, not as a mask)"
        h = self.attention.n_head
        return self._chai_selection(single.shape[2], single.shape[0], single.shape[-1] // h, h, single.device)[1]


class TokenDiTMiniWorldBiasOnly(BackendModule):
    """MiniWorld's bias-only token DiT block: AF3 Alg. 23 with the attention's query-key half removed::

        a = BiasOnlyAttention(a, s, z, mask)           a + gated softmax(pair bias) v -- no query, no key
        a = ConditionedTransition(a, s)                a + transition(a, s)

    ``forward(single, cond, pair, mask)`` -> ``single``'s shape. ``single`` and ``cond`` carry the augmentation axis (``A, B, L, d``);
    ``pair`` does not (``B, L, L, d_pair``). The token widths are the defaults (768 / cond 384 / pair 128 / 16 heads x 48 / transition
    n = 2), the same as :class:`TokenDiTProtenix`. ``forward(..., bias=b)``: the block's pair bias precomputed ([B, H, L, L], from
    :func:`pair_bias_all`); ``pair`` may be None. A pair of its own per sample is ``single`` [1, B, L, d] with ``pair`` [B, L, L, d_pair].
    No ``"anthropic"`` backend: no release folder has the bias-only attention.
    """
    version = 2   # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "cuda")

    def __init__(
        self,
        d_single: int = 768,
        d_cond: int = 384,
        d_pair: int = 128,
        n_head: int = 16,
        n: int = 2,
        *,
        d_head: int | None = None,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        self.attention = BiasOnlyAttention(d_single=d_single, d_cond=d_cond, d_pair=d_pair, n_head=n_head, d_head=d_head)
        self.transition = ConditionedTransition(d_hidden=d_single, d_cond=d_cond, n=n)
        self.set_backend(backend)

    def _forward_pytorch(
        self,
        single: torch.Tensor,
        cond: torch.Tensor,
        pair: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        *,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Each part returns its residual output (stream in, stream out). ``bias``: this block's hoisted pair bias (:func:`pair_bias_all`) in
        place of ``to_bias(ln_pair(pair))``."""
        if bias is None and pair is None:
            raise ValueError("TokenDiTMiniWorldBiasOnly: pass the pair or the block's precomputed bias")
        single = self.attention(single, cond, pair, mask) if bias is None else self.attention(single, cond, pair, mask, bias=bias)
        return self.transition(single, cond)

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        return _token_dit_inputs(self, length=length, samples=samples, dtype=dtype, device=device, generator=generator)

    # ------------------------------------------------------------------------------------------------------------ "cuda"
    # H100: no gradient -> kernels/integrations/bias_only_dit (the fused inference block: P = softmax(pair bias) once per pair, the
    # pv_gate core, CUDA rows, cuBLAS); a gradient -> kernels/integrations/bias_only_dit_train (one autograd Function on the CUDA rows,
    # cuBLAS and the bias-only cores). Range = the two gates: bf16 / fp32 (fp32 = fp32 weights, no autocast), [A, 1, L, 768] with L a
    # multiple of 128 (training L <= 768), 16 heads x 48, cond 384 or 768 (training also 128 / 256), pair 128 / 256, n = 2, eps 1e-5,
    # key mask [1, L] or none, the pair or the block's hoisted bias [1, H, L, L].
    def _cuda_route(self, single, cond, pair, mask, bias):
        why = _h100_reason(single)
        if why is not None:
            return None, why
        view = _kernel_view(self)
        if _wants_grad(self, single, cond, pair, bias):
            from miniworld_engine.kernels.integrations import bias_only_dit_train

            if bias_only_dit_train.serves(view, single, cond, pair, mask, bias=bias):
                return "train", None
            return None, ("training: outside the fused H100 bias-only training block's range (bf16 / fp32 weights, [A, 1, L, 768] with L "
                          "a multiple of 128 <= 768, heads x head width 16 x 48 / 24 x 32 / 12 x 64 / 16 x 64, cond 128 / 256 / 384 / 768, "
                          "pair 128 / 256, n = 2, key mask [1, L], eps 1e-5)")
        from miniworld_engine.kernels.integrations import bias_only_dit

        if bias_only_dit.serves(view, single, cond, pair, mask, bias=bias):
            return "infer", None
        return None, ("inference: outside the fused H100 bias-only inference block's range (bf16 / fp32 weights without autocast, "
                      "[A, 1, L, 768] with L a multiple of 128, heads x head width 16 x 48 / 24 x 32 / 12 x 64 / 16 x 64, cond 384 or 768, "
                      "pair 128 / 256, n = 2, key mask [1, L], eps 1e-5)")

    def _supports_cuda(self, single, cond, pair=None, mask=None, *, bias=None) -> str | None:
        route, why = self._cuda_route(single, cond, pair, mask, bias)
        return why if route is None else None

    def _forward_cuda(self, single, cond, pair=None, mask=None, *, bias=None) -> torch.Tensor:
        route, _ = self._cuda_route(single, cond, pair, mask, bias)
        view = _kernel_view(self)
        if route == "train":
            from miniworld_engine.kernels.integrations import bias_only_dit_train

            return bias_only_dit_train.block(view, single, cond, pair, mask, bias=bias)
        from miniworld_engine.kernels.integrations import bias_only_dit

        return bias_only_dit.update(view, single, cond, pair, mask, bias=bias)


def pair_bias_all(blocks, pair: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Every bias-only block's pair bias at once: ``tuple(to_bias(ln_pair(pair)).permute(0, 3, 1, 2) for each block)`` -- [B, H, L, L] each
    -- from one LayerNorm and one GEMM, differentiable in ``pair`` and in every block's ``ln_pair.weight`` / ``to_bias.weight``.

    ``ln_pair`` has no offset, so with LN0 the LayerNorm without affine and the fold W'_b = W_b diag(gamma_b), ``bias_b = LN0(pair) W'_b^T``:
    the normalization is the same for every block, only gamma differs. Pass block b its bias (``block(single, cond, pair, mask,
    bias=biases[b])``) and the block touches neither ``ln_pair`` nor ``to_bias``; the gradients reach them, and the pair, through this op.
    Same parameters, the same result up to rounding (gamma applied to the weight instead of the activation). Under activation checkpointing
    call it OUTSIDE the checkpointed blocks.

    ``blocks``: :class:`TokenDiTMiniWorldBiasOnly` blocks (or their ``attention`` modules); they must share the LayerNorm eps and the head
    count, and ``ln_pair`` must have no offset."""
    mods = list(blocks)
    if not mods:
        raise ValueError("pair_bias_all: no blocks")
    attns = [m.attention if hasattr(m, "attention") else m for m in mods]
    eps, H = attns[0].ln_pair.eps, attns[0].to_bias.weight.shape[0]
    for a in attns:
        if (a.ln_pair.eps != eps or a.to_bias.weight.shape[0] != H or getattr(a.ln_pair, "bias", None) is not None
                or getattr(a.to_bias, "bias", None) is not None or a.ln_pair.weight is None):
            raise ValueError("pair_bias_all: the blocks must share eps and head count, ln_pair must have a weight and no offset, "
                             "to_bias no bias")
    # the "cuda" backend on a B200: the fused hoist (one LayerNorm and one GEMM for every block, the blocks' bias gradients written into
    # one buffer); pair [1, L, L, 128] bf16 or fp32 -- any other pair (and every H100 call) takes the PyTorch fold below
    if getattr(mods[0], "backend", "pytorch") == "cuda":
        from miniworld_engine.kernels.integrations import bias_only_dit_hoist

        if bias_only_dit_hoist.serves("miniworld", attns, pair):
            return bias_only_dit_hoist.pair_bias_all(attns, pair)
    # LN0 in at least fp32 (as the engine's LayerNorm computes), one matmul against W'_all, per-block head-major views [B, H, L, L] of the
    # [B, L, L, N] product
    dp = pair.shape[-1]
    ln0 = F.layer_norm(pair.to(wide(pair)), (dp,), None, None, eps).to(pair.dtype)
    wf = torch.cat([a.to_bias.weight * a.ln_pair.weight.to(a.to_bias.weight.dtype)[None, :] for a in attns], 0)   # [N, dp]
    b = torch.matmul(ln0, wf.t().to(ln0.dtype))                                                                   # [B, L, L, N]
    return tuple(b[..., i * H:(i + 1) * H].permute(0, 3, 1, 2) for i in range(len(attns)))


# ======================================================================================================================== selector
#: model name -> its token DiT class. Several models share a class where their blocks are the same math.
TOKEN_DIT_MODELS: dict[str, type[BackendModule]] = {
    "af3": TokenDiTProtenix,
    "boltz2": TokenDiTBoltz2,
    "rosettafold3": TokenDiTRoseTTAFold3,
    "chai1": TokenDiTChai1,
    "openfold3": TokenDiTProtenix,
    "openfold3_preview2": TokenDiTProtenix,
    "protenix": TokenDiTProtenix,
    "protenix_v2": TokenDiTProtenix,
    "opendde": TokenDiTProtenix,
    "intellifold2": TokenDiTProtenix,
    "esmfold2": TokenDiTProtenix,
    "miniworld": TokenDiTMiniWorld,
    "miniworld_bias_only": TokenDiTMiniWorldBiasOnly,
}


class TokenDiT:
    """``TokenDiT(model=..., *args, **kwargs)``: builds ``TOKEN_DIT_MODELS[model](*args, **kwargs)`` (the model's own class, not a
    wrapper: its state dict has the model class's keys)."""

    models = TOKEN_DIT_MODELS

    def __new__(cls, model: str, *args, **kwargs):
        try:
            klass = TOKEN_DIT_MODELS[model]
        except KeyError:
            raise ValueError(f"TokenDiT: unknown model {model!r}; known: {', '.join(sorted(TOKEN_DIT_MODELS))}") from None
        return klass(*args, **kwargs)


__all__ = [
    "TOKEN_DIT_MODELS",
    "TokenDiT",
    "TokenDiTBoltz2",
    "TokenDiTChai1",
    "TokenDiTMiniWorld",
    "TokenDiTMiniWorldBiasOnly",
    "TokenDiTProtenix",
    "TokenDiTRoseTTAFold3",
    "pair_bias_all",
]
