"""The atom-track diffusion transformer: one class per model, picked by :class:`AtomDiT`.

    AtomDiT(model="chai1")                      # -> AtomDiTChai1()

Each class hard-codes its model's math (no option flags); :data:`ATOM_DIT_MODELS` maps the model names to them. Two families:

* the engine's block-local AF3 atom block (AF3 Alg. 23 with the block-local attention of Alg. 24; one block per instance), the parameters
  of :class:`~miniworld_engine.modules.token_dit.TokenDiTProtenix` at atom widths, and where the model has one, a K / V AdaLN
  ``attention.ada_ln_kv``. The defaults are each model's release:

      AtomDiTOpenFold3      K / V AdaLN of the raw atoms (parallel: AF3 ``cross_attention`` 'q' / 'k' adaptive layernorms); the slid
                            128-key window (AlphaFold 3, OpenFold3 v0.4.1 / v0.5 / preview-2, IntelliFold2's JAX release, which runs AF3's
                            code); ``key_window="slide_qblock"``: IntelliFold2's torch release
      AtomDiTProtenix       K / V AdaLN of the already normalised atoms (chained: Protenix ``layernorm_kv(a=layernorm_a(a))``); the clipped
                            window (Protenix v1 / v2, OpenDDE)
      AtomDiTBoltz2         ONE AdaLN for q and k / v (Boltz-2 ``k_in = to_keys(adaln(a, s))``), the up-gate transition
                            (``UpGatedConditionedTransition``), the clipped window
      AtomDiTRoseTTAFold3   ONE AdaLN for q and k / v (RF3 ``ada_ln_1``), the parallel schedule, the affine LayerNorm of the flat q / k and a
                            bias-free query (``FlatQKNormAugmentedAttention``), the clipped window
      AtomDiTChai1          ONE AdaLN for q and k / v, the parallel schedule, the identity AdaLN (``IdentityAdaptiveLayerNorm``, epsilon 0.1)
                            everywhere, no gating query, no output projection, the clipped window and the same-conformer key mask (``uid``)

  Atom ``i`` belongs to query window ``w = i // 32`` and attends to the 128 atoms of its key window, with a per-window pair bias. The
  pair is the *trunked* atom pair ``[B, nwin, 32, 128, d_pair]`` (:func:`~miniworld_engine.modules.functional.to_windows` makes it from a
  dense pair). ``single`` and ``cond`` carry the augmentation axis ``[A, B, N, d]``, the pair does not. ``B`` is the multi-fold axis (B folds of one
  input, each its own pair and conditioning, one key mask expanded over B: docs/modules.md, multi-fold contract). Scores and softmax run in fp32
  (fp64 for an fp64 call). ``key_window`` (the real atoms ``R`` come first; a masked key is off for every query):

      "slide"          ``[32 w - 48, 32 w + 80)`` shifted back inside the real atoms (AF3 ``features.py`` AtomCrossAtt.compute_features;
                       OF3 ``get_block_indices``)
      "pad"            the same window clipped to ``[0, N)``, keys outside it off (Protenix ``create_local_attn_bias``, Boltz-2
                       ``get_indexing_matrix``, RF3 ``atom_attention``, Chai-1 ``get_qkv_indices_for_blocks``: its ``% N`` only indexes,
                       the wrapped keys are masked by ``get_block_atom_pair_mask``)
      "slide_qblock"   "slide" against ``R`` rounded up to a whole query window of 32 (IntelliFold2's torch release,
                       ``concat_previous_and_later_windows``)

  The trunked pair is read as laid out from the window's first key in every case (its producer gathers it with the same window).

  A released stack (one shared pair LayerNorm and one ``Linear(d_pair, 4 n_blocks)`` for the stack: AlphaFold 3, Boltz-2, Chai-1) is
  converted to these blocks: the LayerNorm weight replicated per block, the projection's rows ``4 b .. 4 b + 3`` to block ``b``, the
  LayerNorm offset (Chai-1) dropped (a per-head constant over a window's keys, which the softmax cancels).

* the pair-free sliding-window atom block (adaLN-Zero, 3D RoPE, SwiGLU; ESMFold2's release ``SWAAtomBlock``, which MiniWorld adopted) and
  its stack:

      AtomDiTMiniWorld      one block (``SWA3DRoPEAttention`` + ``SwiGLUFFN``)
      AtomDiTESMFold2       ESMFold2's release atom block: the same block (its own class, ESMFold2's constructor names)
      SWAAtomTransformer    ``n_block`` of them

* FoldForge's ESMFold2 atom block with the atom pair (FoldForge R6), :class:`AtomDiTESMFold2Pair`: the AF3 block above over a 32 x 192
  trunked pair with AdaptiveRMSNorm AdaLNs, raw gates, the +/- 64 rank window and q / k RMSNorm + rotary (``AtomDiT(model="esmfold2_pair")``).

Each is a :class:`~miniworld_engine.modules.primitives.BackendModule`; ``_forward_pytorch`` is the reference.
"""

from __future__ import annotations

from typing import ClassVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from torch.utils.checkpoint import checkpoint_sequential

from miniworld_engine.kernels import _capture
from miniworld_engine.modules import _anthropic
from miniworld_engine.modules.attention import (
    AugmentedAttentionPairBias,
    FlatQKNormAugmentedAttention,
    UngatedAugmentedAttention,
)
from miniworld_engine.modules.dit_variant_cuda import MasterEntryCast, VariantAtomCuda, fp32_conditioning
from miniworld_engine.modules.functional import (
    ATOM_KEYS,
    ATOM_QUERIES,
    apply_rotary_emb_3d,
    build_3d_rope,
    build_attention_params,
    slide_starts,
)
from miniworld_engine.modules.norm import RMS_EPS, AdaptiveLayerNorm, AdaptiveRMSNorm, IdentityAdaptiveLayerNorm, LayerNorm, RMSNorm
from miniworld_engine.modules.primitives import (
    BackendModule,
    Linear,
    MPLinear,
    key_mask,
    master_autocast,
    master_linear,
    random_input,
    shared_cond_stream,
    wide,
)
from miniworld_engine.modules.transition import ConditionedTransition, UpGatedConditionedTransition

for _attr in ("_katom_key", "_katom_packs"):  # the ESMFold2 kit's atom packs (clear_weight_caches)
    _capture.instance_cache_attr(_attr)

QUERIES, KEYS, KEY_OFFSET = ATOM_QUERIES, ATOM_KEYS, 48
_BF, _F32 = torch.bfloat16, torch.float32


def _h100(t: torch.Tensor) -> bool:
    """A CUDA tensor on an H100 (sm_90): the device the ``"cuda"`` backends' hand-written kernels are built for."""
    return t.is_cuda and torch.cuda.get_device_capability(t.device) == (9, 0)


def _a100(t: torch.Tensor) -> bool:
    """A CUDA tensor on an A100 (sm_80): the card of the ``"cuda"`` backends' sm_80 paths (``kernels/integrations/*_sm80.py``)."""
    return t.is_cuda and torch.cuda.get_device_capability(t.device) == (8, 0)


class _A100View:
    """The block as the sm_80 atom block integration reads it (``local_dit_sm80_block``, written against the earlier ``LocalDiTBlock``):
    ``cross_attention`` = a K / V AdaLN of its own; everything else (parameters, submodules, ``get_parameter``) is the module's."""

    def __init__(self, module) -> None:
        object.__setattr__(self, "_module", module)
        object.__setattr__(self, "cross_attention", module.kv_norm != "shared")

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_module"), name)


def _needs_grad(module: nn.Module, *tensors) -> bool:
    return torch.is_grad_enabled() and (
        any(p.requires_grad for p in module.parameters()) or any(isinstance(t, torch.Tensor) and t.requires_grad for t in tensors)
    )


def _one_row(t: torch.Tensor, folds: int) -> bool:
    """``t`` [1, N], or [B, N] that is one row expanded over the B folds (stride 0: metadata only, no device read)."""
    return t.shape[0] == 1 or (t.shape[0] == folds and t.stride(0) == 0)


def _folds(single: torch.Tensor) -> int:
    """B, the fold axis of ``single`` [A, B, N, d] (1 for any other rank: the shape checks refuse those)."""
    return single.shape[1] if single.ndim == 4 else 1


def _multi_fold_reason(module: nn.Module, single, cond, pair, mask, uid) -> str | None:
    """Why the ``"cuda"`` backend does not take this B > 1 fold call (the multi-fold contract, docs/modules.md), or None. The kernels keep
    one pair bias per launch outside the bf16 inference path: bf16 activations, no gradient; one key mask / ``uid`` row for the folds."""
    folds = _folds(single)
    if (single.dtype, cond.dtype, pair.dtype) != (_BF, _BF, _BF):
        return "B > 1 folds: bfloat16 activations only (the TF32 kernels keep one pair bias)"
    if _needs_grad(module, single, cond, pair):
        return "B > 1 folds: inference only (the backward keeps one pair bias)"
    if cond.ndim != 4 or tuple(cond.shape[:3]) != tuple(single.shape[:3]) or pair.shape[0] != folds:
        return f"B > 1 folds: cond [A, B, N, d] and pair [B, ...] with B = {folds}, got {tuple(cond.shape)} / {tuple(pair.shape)}"
    for name, t in (("mask", mask), ("uid", uid)):
        if t is not None and (t.ndim != 2 or t.shape[1] != single.shape[2] or not _one_row(t, folds)):
            return f"B > 1 folds: {name} must be one [1, N] row (or that row expanded over the B folds), got {tuple(t.shape)}"
    return None


def _fold(single, cond, pair, mask, uid, b: int):
    """Fold ``b`` of a multi-fold call as a one-fold call: single [A, 1, N, d] (a contiguous copy), cond [A, 1, N, d] (a sample-expanded
    conditioning stays the view: its hoisted table is keyed on the fold's storage and reused over the sampler's steps), pair [1, ...],
    the one mask / uid row."""
    c = cond[:, b : b + 1]
    if not (c.shape[0] > 1 and c.stride(0) == 0):
        c = c.contiguous()
    return (
        single[:, b : b + 1].contiguous(),
        c,
        pair[b : b + 1],
        None if mask is None else mask[:1],
        None if uid is None else uid[:1],
    )


def windows(n: int) -> int:
    return (n + QUERIES - 1) // QUERIES


def _key_windows(t: torch.Tensor, n: int, value: float = 0.0, keys: int = KEYS) -> torch.Tensor:
    """[..., N, C] -> [..., nwin, keys, C]: the key window of every query window (out-of-range positions are ``value``)."""
    nwin = windows(n)
    offset = (keys - QUERIES) // 2
    padded = F.pad(t, (0, 0, offset, nwin * QUERIES + keys - QUERIES - offset - n), value=value)
    return padded.unfold(-2, keys, QUERIES).transpose(-1, -2)


#: The constructor's ``key_window`` choices (see the module docstring).
KEY_WINDOWS = ("pad", "slide", "slide_qblock")

#: Where a block's keys / values come from: ``"parallel"`` (their own AdaLN of the raw atoms, AF3), ``"chained"`` (their own AdaLN of the
#: query AdaLN's output, Protenix), ``"shared"`` (the query AdaLN's output itself: one AdaLN, Boltz-2 / RF3 / Chai-1).
KV_NORMS = ("parallel", "chained", "shared")


def _dense_f32(t: torch.Tensor) -> torch.Tensor:
    """``t`` as a contiguous fp32 tensor in one pass (a strided view of the bias GEMM's output, any dtype)."""
    return torch.empty(t.shape, dtype=torch.float32, device=t.device).copy_(t)


def _gather_windows(t: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """[A, B, N, C] and the key positions ``index`` [B, nwin, keys] -> [A, B, nwin, keys, C] (a position outside ``[0, N)`` reads 0)."""
    n = t.shape[2]
    inside = (index >= 0) & (index < n)
    batch = torch.arange(t.shape[1], device=t.device)[:, None, None]
    out = t[:, batch, index.clamp(0, n - 1)]
    return out * inside[None, ..., None].to(out.dtype)


def _window_positions(key_window: str, keep: torch.Tensor, nwin: int, keys: int) -> torch.Tensor:
    """The key positions [B, nwin, keys] of every query window under a gathered ``key_window`` (``keep`` [B, N], real atoms first):

    * ``"slide"``: :func:`~miniworld_engine.modules.functional.slide_starts` of the real atom count ``R`` (AlphaFold 3's rule);
    * ``"slide_qblock"``: the same rule against ``R`` rounded up to a whole query window (IntelliFold2's torch release), so the last
      window may reach past ``R`` into padding, which the key mask turns off."""
    n_real = keep.sum(-1).long()
    if key_window == "slide_qblock":
        n_real = (n_real + QUERIES - 1) // QUERIES * QUERIES
    return slide_starts(n_real, nwin, keys)[:, :, None] + torch.arange(keys, device=keep.device)


# =================================================================================================== the engine's block-local AF3 block
class _LocalAtomDiT(BackendModule):
    """The block-local AF3 atom block: ``a = a + attention_delta(a, s, z)``, then ``a = ConditionedTransition(a, s)`` (or, ``parallel``,
    ``a + attention_delta(a) + transition_delta(a)``). A model class sets the parts, the K / V source (``kv_norm``), the key window and
    the conditioning forms.

    ``backend="anthropic"`` (:class:`AtomDiTOpenFold3`: af3_torch / openfold3's fp32 atom island ``atom_window``; :class:`AtomDiTProtenix`:
    Protenix's provider row ``opt_core.kernels.apb.atom_attention`` for its clipped window, the atom island for "slide";
    :class:`AtomDiTBoltz2`: Boltz-2's own kernel): the attention on the release op, the rest as ``"pytorch"``; eval only, fp32 or the master
    layout (:meth:`_supports_anthropic`).

    ``backend="triton"`` (every class): the 32 x 128 window attention core on the engine's Triton kernel
    (``kernels/augmented_attention/triton/window.py``, forward and backward), the block's other statements (AdaLN, projections, gates, the
    transition) as ``"pytorch"``'s (:meth:`_supports_triton`).

    ``backend="cuda"`` (every atom DiT class): on H100 the whole block on the one atom DiT implementation
    (``kernels/integrations/local_dit_h100.py``: conditioning tables, pre / post row kernels on wgmma, the mma.sync window attention;
    bf16 or TF32; a model's differences are its options, ``local_dit_h100.options``), forward and backward, every key window (the slid
    windows' edges on ``kernels/dit_variants/cuda/edge.cu``); the range is :meth:`_supports_cuda`. On A100 (sm_80):
    ``kernels/integrations/local_dit_sm80_block`` or the sm_80 builds of ``kernels/dit_variants/cuda`` (``modules/dit_variant_cuda.py``),
    see the comment above ``_a100_rows``.
    """

    backends = ("pytorch", "anthropic", "triton")
    backend_versions: ClassVar[dict[str, int]] = {
        "anthropic": 2,
        "cuda:a100_sm80": 4,
    }  # 2026-11-10: kits; sm_80 paths; cuda:a100_sm80 3 (2026-11-13): a CTA whose keys are all masked writes its zero rows without computing (bitwise the same)
    # cuda:a100_sm80 4 (2026-11-13): "slide" / "slide_qblock": the query windows past the last real one (padding queries only) read the slid
    # keys [R - 128, R) as the reference does (the tail kernels of dit_variants edge.cu); they kept the clipped window. Real rows unchanged.

    #: The default of the constructor's ``key_window``: one of :data:`KEY_WINDOWS` (see the module docstring).
    key_window: str = "slide"
    #: The K / V source: one of :data:`KV_NORMS`. ``"shared"`` builds no ``attention.ada_ln_kv``.
    kv_norm: str = "parallel"
    #: The parallel schedule ``a + dA(a) + dT(a)`` (:class:`AtomDiTRoseTTAFold3`, :class:`AtomDiTChai1`).
    parallel: bool = False
    #: The parts (a model's variant of each).
    attention_part: type[AugmentedAttentionPairBias] = AugmentedAttentionPairBias
    transition_part: type[ConditionedTransition] = ConditionedTransition

    def __init__(
        self,
        d_single: int = 128,
        d_cond: int = 128,
        d_pair: int = 16,
        n_head: int = 4,
        n: int = 2,
        *,
        key_window: str | None = None,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        if key_window is not None:
            if key_window not in KEY_WINDOWS:
                raise ValueError(f"key_window must be one of {', '.join(map(repr, KEY_WINDOWS))}, got {key_window!r}")
            self.key_window = key_window
        self.attention = self.attention_part(d_single=d_single, d_cond=d_cond, d_pair=d_pair, n_head=n_head)
        self.transition = self.transition_part(d_hidden=d_single, d_cond=d_cond, n=n)
        if self.kv_norm != "shared":
            self.attention.ada_ln_kv = AdaptiveLayerNorm(d_single, d_cond)
        self.set_backend(backend)

    # --------------------------------------------------------------------------------------------------------------------- the math
    def attention_operands(self, single, cond, pair):
        """The attention's operands: ``x`` (the AdaLN output), ``q`` / ``k`` / ``v`` [A, B, N, H * D] (after the model's q / k norm) and the
        trunked pair bias [B, H, nwin, 32, 128]."""
        att = self.attention
        x = att.ada_ln_in(single, cond)

        b, n = x.shape[1:3]
        nwin = windows(n)
        if tuple(pair.shape[:4]) != (b, nwin, QUERIES, KEYS):
            raise ValueError(f"pair must be the trunked atom pair [B, {nwin}, {QUERIES}, {KEYS}, d_pair], got {tuple(pair.shape)}")
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = F.layer_norm(pair, (pair.shape[-1],), att.ln_pair.weight.to(pair.dtype), None, att.ln_pair.eps)
        bias = F.linear(z, att.to_bias.weight.to(z.dtype)).permute(0, 4, 1, 2, 3)
        q = att.to_query(x)
        xkv = x if self.kv_norm == "shared" else att.ada_ln_kv(single if self.kv_norm == "parallel" else x, cond)
        k = att.to_key(xkv)
        v = att.to_value(xkv)
        q, k = att.flat_qk_norm(q, k)
        return x, q, k, v, bias

    def attention_delta(self, single, cond, pair, mask=None, uid=None) -> torch.Tensor:
        """The attention's update (no residual): AdaLN, q / k / v / gate, softmax over each 32 x 128 window, gates. ``uid`` [B, N]
        (Chai-1's conformer ids): a real query sees only the keys of its own conformer."""
        att = self.attention
        x, q, k, v, bias = self.attention_operands(single, cond, pair)
        a, b, n, _ = x.shape
        h, d = att.n_head, x.shape[-1] // att.n_head
        nwin, keys = windows(n), KEYS
        w = wide(x)
        qw = F.pad(q, (0, 0, 0, nwin * QUERIES - n)).reshape(a, b, nwin, QUERIES, h, d)
        keep = torch.ones(b, n, device=x.device) if mask is None else mask.float()
        gathered = self.key_window != "pad"
        if gathered:
            key_pos = _window_positions(self.key_window, keep, nwin, keys)  # [B, nwin, keys]
            kw = _gather_windows(k, key_pos).reshape(a, b, nwin, keys, h, d)
            vw = _gather_windows(v, key_pos).reshape(a, b, nwin, keys, h, d)
        else:
            kw = _key_windows(k, n, keys=keys).reshape(a, b, nwin, keys, h, d)
            vw = _key_windows(v, n, keys=keys).reshape(a, b, nwin, keys, h, d)
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = torch.einsum("abwqhd,abwkhd->abhwqk", qw.to(w), kw.to(w)) * d**-0.5 + bias.to(w)[None]
        if gathered:
            # the slid window lies inside the real atoms (only a system under 128 atoms has keys past them); slide_qblock's last window may
            # reach into padding: the key mask turns those keys off
            valid = _gather_windows(keep[None, ..., None], key_pos)[0, ..., 0] > 0
        else:
            valid = _key_windows(keep[..., None], n, keys=keys)[..., 0] > 0  # [B, nwin, keys]
            key_pos = (
                torch.arange(nwin, device=x.device)[:, None] * QUERIES - (keys - QUERIES) // 2 + torch.arange(keys, device=x.device)
            )[None].expand(b, -1, -1)
        valid = valid[:, :, None, :]  # [B, nwin, 1, keys]
        if uid is not None:  # a real query sees only the keys of its own conformer (a padding query: every valid key)
            pad = nwin * QUERIES - n
            uid_k = torch.gather(uid, 1, key_pos.clamp(0, n - 1).reshape(b, -1)).view(b, nwin, 1, keys)
            uid_q = F.pad(uid, (0, pad)).view(b, nwin, QUERIES, 1)
            real_q = F.pad(keep, (0, pad)).view(b, nwin, QUERIES, 1) > 0
            valid = valid & ((uid_q == uid_k) | ~real_q)  # [B, nwin, 32, keys]
        scores = scores.masked_fill(~valid[None, :, None], float("-inf"))
        probs = torch.nan_to_num(torch.softmax(scores, dim=-1))
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = torch.einsum("abhwqk,abwkhd->abwqhd", probs, vw.to(w)).to(x.dtype)
        out = out.reshape(a, b, nwin * QUERIES, h * d)[:, :, :n]
        out = att.gate_project(x, out)
        return self._out_gate(cond, out)

    def _out_gate(self, cond: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """The adaLN-Zero output gate ``sigmoid(W_sc cond + b_sc) * out``."""
        return torch.sigmoid(self.attention.to_scale(cond)) * out

    def transition_delta(self, single, cond) -> torch.Tensor:
        """The transition's update (no residual)."""
        return self.transition.delta(single, cond)

    def _step(self, single, cond, pair, mask=None, *, delta=None, **kw):
        """The block under the model's schedule (both residuals included). ``delta``: the attention update to use (default
        :meth:`attention_delta`, the reference)."""
        delta = self.attention_delta if delta is None else delta
        if self.parallel:
            return single + delta(single, cond, pair, mask, **kw) + self.transition_delta(single, cond)
        single = single + delta(single, cond, pair, mask, **kw)
        return self.transition(single, cond)

    # ------------------------------------------------------------------------------------------------------------ "triton"
    def _supports_triton(self, single, cond, pair, mask=None, uid=None, **kw) -> str | None:
        # ``uid`` positional as in the forward that takes it (AtomDiTChai1): FoldForge asks ``unsupported(*forward args)``
        if kw:
            return f"{', '.join(sorted(kw))} (not an input of the block)"
        if not single.is_cuda:
            return "the Triton window attention runs on CUDA tensors only"
        if single.dtype not in (_BF, _F32):
            return f"dtype {single.dtype} (bfloat16 / float32 only)"
        if single.ndim != 4 or cond.ndim != 4:
            return f"single / cond must be [A, B, N, d], got {tuple(single.shape)} / {tuple(cond.shape)}"
        b, n, c = single.shape[1:]
        h = self.attention.n_head
        if c % h or c // h > 64:
            return f"head dim {c / h:g} (the Triton window kernel takes a whole head dim <= 64)"
        if tuple(pair.shape[:4]) != (b, windows(n), QUERIES, KEYS):
            return f"pair must be the trunked atom pair [B, {windows(n)}, {QUERIES}, {KEYS}, d_pair], got {tuple(pair.shape)}"
        if mask is not None and tuple(mask.shape) != (b, n):
            return f"mask must be [B, N], got {tuple(mask.shape)}"
        if uid is not None and tuple(uid.shape) != (b, n):
            return f"uid must be [B, N], got {tuple(uid.shape)}"
        return None

    def _attention_delta_triton(self, single, cond, pair, mask=None, uid=None) -> torch.Tensor:
        """:meth:`attention_delta` with the window softmax on the Triton kernel (``window_attention``: the same key windows -- the clipped
        ``32 w - 48`` start, or the gathered window's first key position -- the same key mask, conformer mask and empty-row zero)."""
        from miniworld_engine.kernels.augmented_attention.triton.window import window_attention

        att = self.attention
        x, q, k, v, bias = self.attention_operands(single, cond, pair)
        a, b, n, _ = x.shape
        h, d = att.n_head, x.shape[-1] // att.n_head
        nwin = windows(n)
        keep = torch.ones(b, n, dtype=torch.bool, device=x.device) if mask is None else mask.bool()
        if self.key_window == "pad":
            first = torch.arange(nwin, device=x.device) * QUERIES - (KEYS - QUERIES) // 2
            starts = first[None].expand(b, -1).to(torch.int32).contiguous()
        else:
            starts = _window_positions(self.key_window, keep.float(), nwin, KEYS)[..., 0].to(torch.int32).contiguous()
        q, k, v = (t.to(x.dtype).reshape(a, b, n, h, d) for t in (q, k, v))
        out = window_attention(q, k, v, bias, starts, keep, uid=uid, keys=KEYS).to(x.dtype).reshape(a, b, n, h * d)
        out = att.gate_project(x, out)
        return self._out_gate(cond, out)

    def _forward_triton(self, single, cond, pair, mask=None):
        return self._step(single, cond, pair, mask, delta=self._attention_delta_triton)

    # -------------------------------------------------------------------------------------------------------------- "cuda"
    # The H100 warp-level block (``kernels/integrations/local_dit_h100.run_block``): the module's 27 parameters by name
    # (``local_dit.WEIGHTS`` + ``CROSS_WEIGHTS``), the K / V AdaLN chained (Protenix) or parallel (AlphaFold 3), the clipped window or AlphaFold
    # 3's slid one or IntelliFold2's slide_qblock (their edge windows recomputed and differentiated in torch), inference and training.
    # Listed only by the classes with a K / V AdaLN (Protenix, OpenFold3).
    def _supports_cuda(self, single, cond, pair, mask=None, uid=None, **kw) -> str | None:
        if kw:
            return f"{', '.join(kw)} (not an input of the H100 atom block)"
        if _a100(single):
            if _folds(single) > 1:  # B > 1 folds: one sm_80 call per fold (the sm_80 atom paths take one pair)
                why = _multi_fold_reason(self, single, cond, pair, mask, uid)
                if why is not None:
                    return f"A100: {why}"
                single, cond, pair, mask, uid = _fold(single, cond, pair, mask, uid, 0)
            return self._a100_reason(single, cond, pair, mask)
        if not _h100(single):
            return "the H100 atom block kernels run on an H100 (sm_90) CUDA tensor only"
        from miniworld_engine.kernels.integrations import local_dit_h100

        if local_dit_h100.options(self) is None:
            return (
                "outside the H100 atom block's options (local_dit_h100.options: mixed AdaLN forms or epsilons, a per-head q / k norm, "
                "an up-gate weight other than [256, 128] without bias, ...)"
            )
        folds = single.shape[1] if single.ndim == 4 else 1
        if uid is not None and (uid.ndim != 2 or uid.shape[1] != single.shape[2] or not _one_row(uid, folds)):
            return f"uid must be [1, N] (or one [1, N] row expanded over the B folds), got {tuple(uid.shape)}"
        att, tr = self.attention, self.transition
        params = list(self.parameters())
        dts = (single.dtype, cond.dtype, pair.dtype)
        if dts == (_F32, _F32, _F32):
            if torch.is_autocast_enabled("cuda"):
                return "float32 activations under autocast"
            if any(p.dtype != _F32 for p in params):
                return "float32 activations need float32 parameters (the TF32 kernels)"
        elif dts == (_BF, _BF, _BF):
            if any(p.dtype not in (_BF, _F32) for p in params):
                return "bfloat16 activations take bfloat16 or float32 parameters"
        else:
            return f"dtypes {tuple(str(t) for t in dts)} (single / cond / pair all bfloat16 or all float32)"
        if single.ndim != 4 or single.shape[3] != 128 or single.shape[2] == 0:
            return f"single must be [A, B, N, 128], got {tuple(single.shape)}"
        n = single.shape[2]
        if folds > 1:  # the multi-fold contract (B trunk_seeds of one input, each its own pair): the kernels' grouped-bias path
            if dts != (_BF, _BF, _BF):
                return "B > 1 folds: bfloat16 activations only (the TF32 kernels keep one pair bias)"
            if _needs_grad(self, single, cond, pair):
                return "B > 1 folds: inference only (the backward keeps one pair bias)"
        if tuple(cond.shape) != (*single.shape[:3], 128):
            return f"cond must be [A, B, N, 128], got {tuple(cond.shape)}"
        if tuple(pair.shape) != (folds, windows(n), QUERIES, KEYS, 16):
            return f"pair must be [B, {windows(n)}, {QUERIES}, {KEYS}, 16] with B = {folds}, got {tuple(pair.shape)}"
        if mask is not None and (
            mask.dtype != torch.bool or mask.ndim != 2 or mask.shape[1] != n or mask.shape[0] not in (1, folds) or not _one_row(mask, folds)
        ):
            return "mask must be a bool [1, N] key mask (B > 1 folds: the folds' one mask, [1, N] or expanded to [B, N])"
        geometry = (
            att.n_head,
            att.to_query.weight.shape[0],
            att.to_query.weight.shape[1],
            att.to_bias.weight.shape[1],
            tr.expand_a.weight.shape[0],
            tr.d_cond,
        )
        if geometry != (4, 128, 128, 16, 256, 128):
            return "geometry: the kernels are built for 128 / cond 128 / pair 16, 4 heads x 32, transition n = 2"
        if att.ln_pair.eps != 1e-5:
            return "pair LayerNorm eps != 1e-5"
        if any(not p.is_contiguous() for p in params):
            return "a non-contiguous parameter"
        return None

    # A100 (sm_80), two paths:
    #   * the chained K / V AdaLN (Protenix), the clipped "pad" window, bf16 activations (bf16 or fp32 master parameters): the whole block on
    #     kernels/integrations/local_dit_sm80_block (conditioning, pre / post row kernels and the transition on mma.sync, the 32 x 128 window
    #     attention and the trunked pair bias of kernels/augmented_attention/cuda/sm80_atom_local; forward and backward), within its gate;
    #   * everything else (the parallel K / V AdaLN of OpenFold3, the slid windows "slide" / "slide_qblock", fp32 on TF32): the atom
    #     DiT-variant path on its sm_80 builds (modules/dit_variant_cuda.VariantAtomCuda: kernels/dit_variants/cuda/train.py, inference
    #     included).
    def _a100_rows(self, single, cond, pair, mask) -> bool:
        from miniworld_engine.kernels.integrations import local_dit_sm80_block

        return (
            self.kv_norm == "chained"
            and self.key_window == "pad"
            and (single.dtype, cond.dtype, pair.dtype) == (_BF, _BF, _BF)
            and not (torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") != _BF)
            and local_dit_sm80_block.serves(_A100View(self), single, cond, pair, mask)
        )

    def _a100_reason(self, single, cond, pair, mask) -> str | None:
        if self._a100_rows(single, cond, pair, mask):
            return None
        fn, why = VariantAtomCuda._atom_serving(cast(VariantAtomCuda, self), single, cond, pair, mask, None)
        return None if fn is not None else why

    def _forward_cuda(self, single, cond, pair, mask=None, uid=None):
        if _a100(single) and _folds(single) > 1:  # B > 1 folds (``_supports_cuda``): one sm_80 call per fold, in fold order
            return torch.cat([self._forward_cuda(*_fold(single, cond, pair, mask, uid, b)) for b in range(_folds(single))], 1)
        if _a100(single):
            if not self._a100_rows(single, cond, pair, mask):
                from miniworld_engine.modules import dit_variant_cuda as dv

                fn, call = VariantAtomCuda._atom_serving(cast(VariantAtomCuda, self), single, cond, pair, mask, None)
                return dv._run(
                    fn,
                    dv._stack(self, atom=True),
                    single,
                    cond,
                    call,
                    wants_grad=dv._wants_grad(self, single, cond, pair),
                    keep_autocast=True,
                    per_sample=True,
                )
            from miniworld_engine.kernels.integrations import local_dit_sm80_block

            with torch.autocast("cuda", enabled=False):  # the kernels run on bf16 packs of their own
                return local_dit_sm80_block.block(_A100View(self), single, cond, pair, mask)
        from miniworld_engine.kernels.integrations import local_dit_h100

        if not (cond.is_contiguous() or local_dit_h100._shared(cond)):
            cond = cond.contiguous()
        # B > 1 folds: the folds' one key mask / conformer ids (``_supports_cuda``); pass the one row, not a materialised [B, N] copy, so
        # the hoisted pair bias (keyed on the mask's storage) is made once per pair, not once per call
        if mask is not None and mask.shape[0] > 1:
            mask = mask[:1]
        if uid is not None and uid.shape[0] > 1:
            uid = uid[:1]
        # the kernels' precision is their own (bf16 operands, or TF32 for fp32): an enclosing autocast must not recast their torch glue
        with torch.autocast("cuda", enabled=False):
            return local_dit_h100.run_block(
                self, single.contiguous(), cond, pair.contiguous(), None if mask is None else mask.contiguous(), uid
            )

    def _forward_pytorch(self, single, cond, pair, mask=None):
        """``single`` [A, B, N, d_single], ``cond`` [A, B, N, d_cond], trunked ``pair`` [B, nwin, 32, keys, d_pair], key ``mask`` [B, N]
        (real atoms first) -> the stream after the attention and the transition (both residuals included)."""
        return self._step(single, cond, pair, mask)

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        """``single`` / ``cond`` [S, 1, N, d] (S samples on the augmentation axis), the trunked ``pair`` [1, nwin, 32, 128, d_pair], the key
        ``mask`` [1, N] (real atoms first). Stream ``atom_folds`` (the multi-fold contract, docs/modules.md): ``depth`` = F folds (trunk_seeds of
        one input), ``single`` [S, F, N, d], ``cond`` one per fold [1, F, N, d] expanded over the S samples, ``pair`` [F, nwin, 32, 128, d_pair],
        one key mask [1, N] expanded over the folds (FoldForge's grouped sampler step). A ``*_shared_cond`` stream (``atom_single_shared_cond``,
        ``atom_folds_shared_cond``): ``cond`` [1, F, N, d] expanded (stride 0) over the S samples, as FoldForge's ``AtomDiTStack`` calls the
        block; the measurers keep it expanded (requirements/README.md)."""
        d_single, d_cond, d_pair = self.transition.d_hidden, self.transition.d_cond, self.attention.ln_pair.normalized_shape[0]
        stream, shared = shared_cond_stream(stream)
        if shared and stream == "atom_single":
            return {
                "single": random_input((samples, 1, length, d_single), dtype=dtype, device=device, generator=generator),
                "cond": random_input((1, 1, length, d_cond), dtype=dtype, device=device, generator=generator).expand(samples, -1, -1, -1),
                "pair": random_input((1, windows(length), QUERIES, KEYS, d_pair), dtype=dtype, device=device, generator=generator),
                "mask": key_mask(1, length, device=device, pad=QUERIES - 8),
            }
        if stream == "atom_folds":
            f = depth or 1
            return {
                "single": random_input((samples, f, length, d_single), dtype=dtype, device=device, generator=generator),
                "cond": random_input((1, f, length, d_cond), dtype=dtype, device=device, generator=generator).expand(samples, -1, -1, -1),
                "pair": random_input((f, windows(length), QUERIES, KEYS, d_pair), dtype=dtype, device=device, generator=generator),
                "mask": key_mask(1, length, device=device, pad=QUERIES - 8).expand(f, length),
            }
        return {
            "single": random_input((samples, 1, length, d_single), dtype=dtype, device=device, generator=generator),
            "cond": random_input((samples, 1, length, d_cond), dtype=dtype, device=device, generator=generator),
            "pair": random_input((1, windows(length), QUERIES, KEYS, d_pair), dtype=dtype, device=device, generator=generator),
            "mask": key_mask(1, length, device=device, pad=QUERIES - 8),
        }  # padding inside the last query window

    # ------------------------------------------------------------------------------------------------------------ "anthropic"
    # Each class on its own model kit's atom attention (the ops start at the model's q / k / v, computed here, except the OF3 island):
    #   provider  AtomDiTProtenix, clipped "pad" window (Protenix v1 / v2, OpenDDE): opt_core's pair-bias provider's windowed face
    #             ``opt_core.kernels.apb.atom_attention`` by the tier word "fast" (protenix_v2 ``apb_core._make_atom_forward``, opendde
    #             ``odde_apb_bind.atom_views``): the row it selects for cell atom_h4d32w32x128 (fpf_atom / dtk_window / sdpa_gather), all A
    #             samples of a batch row in one call, the windowed bias [H, nwin, 32, 128] fp32 with the key mask folded in as -inf (the
    #             kits' trunked bias), the gate fused; W_o and the AdaLN-Zero gate follow here. fp32 or bf16 (the master layout's autocast
    #             operands, as the kit's bf16 cells take them).
    #   island    AtomDiTOpenFold3 (every window) and AtomDiTProtenix "slide": the fp32 atom-attention island of af3_torch / openfold3
    #             (``opt_core.kernels.atom_window``; af3_torch ``diffusion_transformer.forward_windowed``, openfold3 ``of3_sampler.atom_window``):
    #             ``ln_qkvg`` (OpenFold3's parallel K / V AdaLN: LayerNorm, both AdaLN modulations and the four projections in one pass;
    #             Protenix's chained K / V AdaLN is not its form, so its q / k / v come from the module's statements) then ``window_attn``
    #             (keys [ks[w], ks[w] + 128) read in place, valid below n_real, gate, W_o, the AdaLN-Zero gate and the residual fused). The
    #             kits run it in fp32 at precision ``tf32rn`` (af3_torch_api ``precision``; ``ieee`` when the engine's TF32 rule is off);
    #             the master layout's bf16 activations are widened into it as the kits' island widens them. Conditioning rows
    #             [N, C] are per atom: one launch for all samples when they share the conditioning (A == 1 or stride 0), else one per sample.
    _release_module = "opt_core.kernels.atom_window"
    _release_path = "common/opt_core"
    _PROVIDER_CELL = "atom_h4d32w32x128"

    def _release_route(self) -> str:
        return "provider" if self.kv_norm == "chained" and self.key_window == "pad" else "island"

    def _provider_selection(self, single):
        provider = _anthropic.load("opt_core.kernels.apb")
        h, c = self.attention.n_head, single.shape[-1]
        cc = tuple(torch.cuda.get_device_capability(single.device))
        dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else single.dtype
        return _anthropic.kernel_selection(
            provider,
            cc,
            dt,
            self._PROVIDER_CELL,
            max(1, single.shape[2] // 8),
            word="fast",
            samples=int(single.shape[0]),
            timing="graphed",
            capture=True,
            head_dim=c // h,
            heads=h,
        )

    def _release_range(self, single, cond, pair, mask) -> str | None:
        """The class's own limits (the dtypes and windows its release ops serve)."""
        if single.dtype not in (torch.float32, torch.bfloat16):
            return f"dtype {single.dtype} (bfloat16 / float32 only)"
        if self._release_route() == "provider":
            why = _anthropic.missing("opt_core.kernels.apb")
            return why or self._provider_selection(single)[1]
        if any(p.dtype != torch.float32 for p in self.parameters()):
            return "the atom island (atom_window) takes fp32 parameters (float32 or the master layout)"
        return None

    def _supports_anthropic(self, single, cond, pair, mask=None, **kw) -> str | None:
        why = _anthropic.unavailable(self._release_module, single.device, path=self._release_path)
        if why is not None:
            return why
        if kw:
            return f"{', '.join(kw)} (not an input of the release op)"
        if torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") != torch.bfloat16:
            return "autocast to a dtype other than bfloat16"
        if torch.is_grad_enabled() and (
            any(p.requires_grad for p in self.parameters()) or any(t is not None and t.requires_grad for t in (single, cond, pair))
        ):
            return "the release op has no backward: call it under torch.no_grad() / inference_mode() (eval only)"
        h, c = self.attention.n_head, single.shape[-1]
        if c > 128 or c // h not in (16, 32, 64):
            return f"geometry {h} heads x {c // h} (the release ops: 32 x 128 windows, H * D <= 128, D in 16 / 32 / 64)"
        if single.ndim != 4 or cond.shape[:3] != single.shape[:3]:
            return f"single / cond must be [A, B, N, d], got {tuple(single.shape)} / {tuple(cond.shape)}"
        why = self._release_range(single, cond, pair, mask)
        if why is not None:
            return why
        n = single.shape[2]
        keep = torch.ones(single.shape[1], n, dtype=torch.bool, device=single.device) if mask is None else mask.bool()
        if mask is not None and tuple(mask.shape) != (single.shape[1], n):
            return f"mask must be [B, N], got {tuple(mask.shape)}"
        if self._release_route() == "island":
            n_real = keep.sum(-1)
            if not bool((keep == (torch.arange(n, device=keep.device)[None] < n_real[:, None])).all()):
                return "a key mask whose real atoms are not a prefix (the release op cuts keys at n_real)"
            if self.key_window != "pad" and int(n_real.min()) < KEYS:
                return f"fewer than {KEYS} real atoms (the release's window start goes negative there, the engine's clamps at 0)"
        if self.key_window == "pad":
            seen = _key_windows(keep[..., None].float(), n)[..., 0].amax(-1)  # [B, nwin]: a valid key in the window
            if not bool((seen > 0).all()):
                return "a query window with no valid key (the release ops average such a window, the reference gives 0)"
        return None

    def _release_window_starts(self, n: int, n_real: torch.Tensor, device) -> torch.Tensor:
        """The key-window starts [nwin] int32; ``n_real`` the real atom count as a 1-element int32 device tensor (no host read)."""
        aw = _anthropic.load("opt_core.kernels.atom_window")
        nwin = windows(n)
        if self.key_window == "pad":
            return (torch.arange(nwin, device=device) * QUERIES - (KEYS - QUERIES) // 2).to(torch.int32)
        real = n_real if self.key_window == "slide" else (n_real + QUERIES - 1) // QUERIES * QUERIES
        return aw.window_starts(n, real, QUERIES, KEYS, device)

    def _release_provider(self, single, cond, pair, mask):
        """``single + attention_delta(...)`` on the provider's windowed face (Protenix's kit)."""
        provider = _anthropic.load("opt_core.kernels.apb")
        sel, why = self._provider_selection(single)
        if sel is None:
            raise RuntimeError(why)
        att = self.attention
        x, q, k, v, bias = self.attention_operands(single, cond, pair)
        g = att.to_gate(x)
        a, b, n, c = q.shape
        h = att.n_head
        d = c // h
        keep = torch.ones(b, n, device=x.device) if mask is None else mask.float()
        valid = _key_windows(keep[..., None], n)[..., 0] > 0  # [B, nwin, 128]
        o = torch.empty(a, b, n, c, dtype=q.dtype, device=x.device)
        for j in range(b):
            bj = _dense_f32(bias[j])  # [H, nwin, 32, 128] fp32
            bj.masked_fill_(~valid[j][None, :, None, :], float("-inf"))
            oj, _ = provider.atom_attention(
                q[:, j].unflatten(-1, (h, d)),
                k[:, j].unflatten(-1, (h, d)),
                v[:, j].unflatten(-1, (h, d)),
                bj,
                g[:, j].unflatten(-1, (h, d)),
                word="fast",
                selection=sel,
                n_queries=QUERIES,
                n_keys=KEYS,
                scale=d**-0.5,
                capture=True,
                out_dtype=q.dtype,
            )
            o[:, j] = oj.reshape(a, n, c)
        return single + self._out_gate(cond, att.to_out(o))

    def _release_island(self, single, cond, pair, mask):
        """``single + attention_delta(...)`` on the fp32 atom island (af3_torch / openfold3's ``atom_window``)."""
        aw = _anthropic.load("opt_core.kernels.atom_window")
        att = self.attention
        precision = "tf32rn" if torch.backends.cuda.matmul.allow_tf32 else "ieee"
        in_dtype = single.dtype
        with torch.autocast("cuda", enabled=False):
            s32, c32 = single.float(), cond.float()  # the pair stays in its dtype: its bias is formed as the reference forms it
            a, b, n, c = s32.shape
            h = att.n_head
            d = c // h
            fused = self.kv_norm == "parallel" and att.ada_ln_in.ln_in.eps == att.ada_ln_kv.ln_in.eps
            if fused:  # OpenFold3: the conditioning of both AdaLNs per atom, ln_qkvg does the rest
                cq, ck = att.ada_ln_in.ln_cond(c32), att.ada_ln_kv.ln_cond(c32)
                cond_rows = (
                    torch.sigmoid(att.ada_ln_in.to_scale(cq)),
                    att.ada_ln_in.to_bias(cq),
                    torch.sigmoid(att.ada_ln_kv.to_scale(ck)),
                    att.ada_ln_kv.to_bias(ck),
                )
                z = F.layer_norm(pair, (pair.shape[-1],), att.ln_pair.weight.to(pair.dtype), None, att.ln_pair.eps)
                bias = F.linear(z, att.to_bias.weight.to(z.dtype)).permute(0, 4, 1, 2, 3)  # [B, H, nwin, 32, 128] view
            else:  # Protenix (chained K / V AdaLN): the module's own q / k / v / gate statements
                x, q, k, v, bias = self.attention_operands(s32, c32, pair)
                qkvg_all = torch.cat([q * d**-0.5, k, v, torch.sigmoid(att.to_gate(x))], -1)  # [A, B, N, 4 C]
            planes = _dense_f32(bias.transpose(1, 2))  # [B, nwin, H, 32, 128] fp32
            zgate = torch.sigmoid(att.to_scale(c32))  # the AdaLN-Zero gate [A, B, N, C]
            ones = torch.ones(1, n, dtype=torch.float32, device=s32.device)
            counts = torch.full((b,), n, dtype=torch.int32, device=s32.device) if mask is None else mask.sum(-1, dtype=torch.int32)
            shared = a == 1 or (cond.stride(0) == 0)
            out = torch.empty(a, b, n, c, dtype=torch.float32, device=s32.device)
            for j in range(b):
                n_real = counts[j : j + 1]
                ks = self._release_window_starts(n, n_real, s32.device)
                groups = [slice(0, a)] if shared else [slice(i, i + 1) for i in range(a)]
                for sl in groups:
                    i0 = sl.start
                    if fused:
                        rows = [t[i0, j].contiguous() for t in cond_rows]
                        qkvg = aw.ln_qkvg(
                            s32[sl, j],
                            *rows,
                            att.to_query,
                            att.to_key,
                            att.to_value,
                            att.to_gate,
                            att.ada_ln_in.ln_in.eps,
                            d**-0.5,
                            precision=precision,
                        )
                    else:
                        qkvg = qkvg_all[sl, j]
                    aw.window_attn(
                        qkvg,
                        s32[sl, j],
                        planes[j],
                        ks,
                        n_real,
                        ones,
                        zgate[i0, j].contiguous(),
                        att.to_out.weight,
                        att.to_out.bias,
                        h,
                        QUERIES,
                        KEYS,
                        out=out[sl, j],
                        precision=precision,
                    )
        return out.to(in_dtype)

    def _release_attention(self, single, cond, pair, mask):
        """``single + attention_delta(...)`` on the class's kit op."""
        if self._release_route() == "provider":
            return self._release_provider(single, cond, pair, mask)
        return self._release_island(single, cond, pair, mask)

    def _forward_anthropic(self, single, cond, pair, mask=None):
        """``_forward_pytorch`` with the attention on the release op."""
        if self.parallel:
            raise NotImplementedError  # no parallel class has this backend
        return self.transition(self._release_attention(single, cond, pair, mask), cond)


class AtomDiTProtenix(_LocalAtomDiT):
    """Protenix's / OpenDDE's atom transformer block: the K / V AdaLN of the atoms the first AdaLN has already normalised (chained:
    Protenix ``AttentionPairBias.forward``, ``kv = layernorm_kv(a=layernorm_a(a, s), s)``).

    ``key_window`` says where a query window's 128 keys start: ``"pad"`` at ``32 w - 48``, clipped to the atoms that exist, so an edge
    window sees fewer keys (the release: ``create_local_attn_bias``, the default), or ``"slide"`` shifted back inside the real atoms (the
    first two windows start at 0, the last ones end at atom ``N - 1``); the trunked pair is read as laid out from the window's first key
    either way. ``"slide"`` takes the real atom count ``N`` from the key mask, whose real atoms must come first; under 128 real atoms there is
    no valid window and the keys past ``N`` are masked.
    """

    version = 2  # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "anthropic", "cuda", "triton")
    key_window = "pad"
    kv_norm = "chained"
    #: 2026-11-10: H100 "slide": the five edge windows on the edge kernels (dit_variants edge.cu; were torch)
    #: 2026-11-12: 3, H100 conditioning at 8448 .. 16832 rows (cond90_split grid.y 2) gave the out gates of LN(c): grid.y 4
    #: 2026-11-13: 4, H100 "slide" / "slide_qblock": the query windows past the last real one read the slid keys [R - 128, R) (the tail
    #: kernels of dit_variants edge.cu; they kept the clipped window). Real rows unchanged.
    backend_versions: ClassVar[dict[str, int]] = {**_LocalAtomDiT.backend_versions, "cuda:h100_sm90": 4}


class AtomDiTOpenFold3(_LocalAtomDiT):
    """AlphaFold 3's atom transformer block in the engine layout (registered as ``"af3"``, ``"openfold3"``, ``"openfold3_preview2"``,
    ``"intellifold2"``): the K / V AdaLN of the raw atoms next to the query one (parallel: AF3 ``cross_attention``'s 'q' / 'k' adaptive
    layernorms, OF3 ``CrossAttentionPairBias.layer_norm_a_q`` / ``_k``), the slid 128-key window (AF3 ``features.py``; IntelliFold2's JAX
    release runs this code). ``key_window="slide_qblock"``: IntelliFold2's torch release; ``"pad"``: the clipped window.

    ``mask_form`` (FoldForge R13): ``"keys"`` (the default, as every atom DiT block) switches every masked key off for every query; ``"af3"``
    is AlphaFold 3's own form, whose mask term touches only the pairs where BOTH the query and the key are padding (``+1e9`` there), so a
    real query reads every key of its window, padding included. The forms differ only below 128 real atoms: from 128 on the slid window
    (``key_window="slide"``, the only window this form is defined for) lies inside the real atoms; below it every window holds the N real
    atoms and 128 - N padding keys (AlphaFold 3 ``features.py`` AtomCrossAtt: its windows shifted to start at N - 128 < 0 index padding by
    wrapping; which padding slots does not matter, as below). One pass, the key mask widened to the first 128 atoms (still a prefix: the
    window stays [0, 128); the padded atom count must be >= 128), on every backend. Exact for the real rows when the caller's padding atoms
    are what AlphaFold 3's gathers make them: zero ``single`` / ``cond`` rows (``atom_layout.convert`` fills a padding slot with 0, so every
    padding key is the same key) and the pair of a padding key the one AlphaFold 3 computes there. The padding rows are not AlphaFold 3's
    (its padding queries read only padding keys); AlphaFold 3 never reads them -- its keys come from the masked gather -- so zero them
    before the next block."""

    version = 3  # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "anthropic", "cuda", "triton")
    #: 2026-11-10: H100 "slide" / "slide_qblock": the five edge windows on the edge kernels (dit_variants edge.cu; were torch)
    #: 2026-11-12: 3, H100 conditioning at 8448 .. 16832 rows (cond90_split grid.y 2) gave the out gates of LN(c): grid.y 4
    #: 2026-11-13: 4, H100 "slide" / "slide_qblock": the query windows past the last real one read the slid keys [R - 128, R) (the tail
    #: kernels of dit_variants edge.cu; they kept the clipped window). Real rows unchanged.
    backend_versions: ClassVar[dict[str, int]] = {**_LocalAtomDiT.backend_versions, "cuda:h100_sm90": 4}
    #: ``"keys"`` or ``"af3"`` (see the class docstring)
    mask_form: str = "keys"

    def __init__(
        self,
        d_single: int = 128,
        d_cond: int = 128,
        d_pair: int = 16,
        n_head: int = 4,
        n: int = 2,
        *,
        key_window: str | None = None,
        mask_form: str = "keys",
        backend: str = "pytorch",
    ) -> None:
        super().__init__(d_single, d_cond, d_pair, n_head, n, key_window=key_window, backend=backend)
        if mask_form not in ("keys", "af3"):
            raise ValueError(f"mask_form must be 'keys' or 'af3', got {mask_form!r}")
        if mask_form == "af3" and self.key_window != "slide":
            raise ValueError(f"mask_form='af3' is AlphaFold 3's form on its slid window (key_window='slide'), got {self.key_window!r}")
        self.mask_form = mask_form

    def _mask(self, mask):
        """The key mask every backend runs under: the caller's (``"keys"``), or (``"af3"``) widened to the first 128 atoms, the window below 128
        real atoms -- a prefix still, so the slid window stays [0, 128); from 128 real atoms on it is the caller's mask."""
        if self.mask_form == "keys" or mask is None:
            return mask
        keep = mask.bool()
        return keep | (torch.arange(keep.shape[-1], device=keep.device) < KEYS)

    # each backend's entry (the measurers call them directly) takes the form's mask
    def _forward_pytorch(self, single, cond, pair, mask=None):
        return super()._forward_pytorch(single, cond, pair, self._mask(mask))

    def _supports_triton(self, single, cond, pair, mask=None, uid=None, **kw) -> str | None:
        return super()._supports_triton(single, cond, pair, self._mask(mask), uid, **kw)

    def _forward_triton(self, single, cond, pair, mask=None):
        return super()._forward_triton(single, cond, pair, self._mask(mask))

    def _supports_cuda(self, single, cond, pair, mask=None, uid=None, **kw) -> str | None:
        return super()._supports_cuda(single, cond, pair, self._mask(mask), uid, **kw)

    def _forward_cuda(self, single, cond, pair, mask=None, uid=None):
        return super()._forward_cuda(single, cond, pair, self._mask(mask), uid)

    def _supports_anthropic(self, single, cond, pair, mask=None, **kw) -> str | None:
        return super()._supports_anthropic(single, cond, pair, self._mask(mask), **kw)

    def _forward_anthropic(self, single, cond, pair, mask=None):
        return super()._forward_anthropic(single, cond, pair, self._mask(mask))


class AtomDiTBoltz2(VariantAtomCuda, _LocalAtomDiT):
    """Boltz-2's atom transformer block: ONE AdaLN for the queries and the keys / values (``DiffusionTransformerLayer``: ``b = adaln(a, s)``,
    ``k_in = to_keys(b)``; no ``attention.ada_ln_kv``), the up-gate transition ``silu(W_a h) (W_b h) (W_u h)`` (``transition.up``) and the
    clipped window (``get_indexing_matrix``: keys ``[32 w - 48, 32 w + 80)`` of zero-padded halves; ``key_window="pad"``, the default).

    ``backend="anthropic"``: Boltz-2's own window kernel (boltz2/opt/forward/atom/src/boltz_atom_kernels.py ``launch_win_attn``, its
    ``BOLTZ_ATOM=fused`` lever): rows [q | k | v] 384 wide and sigmoid(g) 128 wide, the clipped "pad" window, the key mask Boltz's
    ``(1 - mask) * -1e6``, the bias read in place from the trunked [B, nwin, 32, 128, 4]; bf16 operands (bf16, and the master layout's
    autocast operands) or fp32 dots (Boltz-2's ``tf32`` word under the engine's TF32 rule, else ``ieee``); the [q | k | v] rows from one
    packed GEMM; 128 = 4 x 32 wide, N a multiple of 32.

    Invariant (shared K / V AdaLN, ``kv_norm = "shared"``): the release computes the queries AND the keys / values from ONE AdaLN output
    (``b = adaln(a, s)``; ``to_queries(b)``, ``to_keys(b)``, ``to_values(b)``), so the block has no ``attention.ada_ln_kv``. A checkpoint
    layout that stores a K / V AdaLN beside the query one (e.g. an AF3-style ``q_adaptive_layernorm`` / ``k_adaptive_layernorm`` pair) holds
    two equal copies of it for this model: load the query copy, and treat a K / V copy that differs as a conversion error, not as a second
    AdaLN to keep (tests/cpu/test_requests_api.py asserts the layout)."""

    version = 2  # human-managed: bump when a change alters this class's accuracy or speed results

    backends = (
        "pytorch",
        "anthropic",
        "cuda",
        "triton",
    )  # "cuda": H100 local_dit_h100, A100 kernels/dit_variants (modules/dit_variant_cuda.py)
    key_window = "pad"
    kv_norm = "shared"
    transition_part = UpGatedConditionedTransition
    #: 2026-11-10: H100 on the one atom DiT implementation (local_dit_h100; was kernels/dit_variants): 2 "pad", 3 "slide"
    #: 2026-11-12: 4, H100 conditioning at 8448 .. 16832 rows (cond90_split grid.y 2) gave the out gates of LN(c): grid.y 4
    #: 2026-11-13: 5, H100 "slide" / "slide_qblock": the query windows past the last real one read the slid keys [R - 128, R) (the tail
    #: kernels of dit_variants edge.cu; they kept the clipped window). Real rows unchanged.
    backend_versions: ClassVar[dict[str, int]] = {**_LocalAtomDiT.backend_versions, "cuda:h100_sm90": 5}
    _release_module = "boltz_atom_kernels"
    _release_path = "boltz2/opt/forward/atom/src"

    def _release_range(self, single, cond, pair, mask) -> str | None:
        dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else single.dtype
        if dt not in (torch.bfloat16, torch.float32):
            return f"dtype {dt} (bfloat16 / float32 only)"
        if self.key_window != "pad":
            return f"key_window {self.key_window!r} (Boltz-2's kernel reads the clipped 'pad' window)"
        if single.shape[-1] != 128 or self.attention.n_head != 4:
            return "width != 128 = 4 heads x 32 (Boltz-2's kernel is written for it)"
        if single.shape[2] % QUERIES:
            return "N not a multiple of 32 (Boltz-2 pads its atoms to whole windows)"
        return None

    def _release_attention(self, single, cond, pair, mask):
        from miniworld_engine.modules.token_dit import _packed_linear

        bk = _anthropic.load(self._release_module, path=self._release_path)
        att = self.attention
        x = att.ada_ln_in(single, cond)
        a, b, n, c = x.shape
        if tuple(pair.shape[:4]) != (b, windows(n), QUERIES, KEYS):
            raise ValueError(f"pair must be the trunked atom pair [B, {windows(n)}, {QUERIES}, {KEYS}, d_pair], got {tuple(pair.shape)}")
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = F.layer_norm(pair, (pair.shape[-1],), att.ln_pair.weight.to(pair.dtype), None, att.ln_pair.eps)
        zb = F.linear(z, att.to_bias.weight.to(z.dtype))  # [B, nwin, 32, 128, H]: Boltz-2's own layout

        def rows(t):  # [A, B, N, C'] -> [B * A * N, C'] (a view for B = 1)
            return t.transpose(0, 1).reshape(b * a * n, t.shape[-1])

        qkv = rows(_packed_linear(att, ("to_query", "to_key", "to_value"), x))  # ONE GEMM -> the [q | k | v] rows Boltz-2 reads
        dt = qkv.dtype
        g = rows(torch.sigmoid(att.to_gate(x))).to(dt).contiguous()
        pad = (torch.ones(b, n, device=x.device) if mask is None else mask.float()).reshape(-1).contiguous()
        ob = torch.empty(b * a * n, c, dtype=dt, device=x.device)
        prec = "bf16" if dt == torch.bfloat16 else ("tf32" if torch.backends.cuda.matmul.allow_tf32 else "ieee")
        bk.launch_win_attn(
            qkv=qkv.contiguous(),
            g=g,
            bias=zb.contiguous(),
            ld=att.n_head,
            loff=0,
            bmult=a,
            pad=pad,
            ob=ob,
            R=b * a * n,
            M=n,
            K=windows(n),
            cmult=a,
            prec=prec,
        )
        o = ob.view(b, a, n, c).transpose(0, 1)
        return single + self._out_gate(cond, att.to_out(o))


class AtomDiTRoseTTAFold3(VariantAtomCuda, _LocalAtomDiT):
    """RoseTTAFold3's atom transformer block (``AttentionPairBiasDiffusion.atom_attention``): ONE AdaLN for the queries and the keys /
    values (``ada_ln_1``; no ``attention.ada_ln_kv``), a bias-free query and an affine LayerNorm of the flat q / k (``kq_norm``:
    ``attention.norm_query`` / ``attention.norm_key``, over all heads; ``FlatQKNormAugmentedAttention``), the parallel schedule
    (``no_residual_connection_between_attention_and_transition: true``) and the clipped window (``indicesK``/``maskK``;
    ``key_window="pad"``, the default). No ``"anthropic"`` backend: RoseTTAFold3's release keeps its stock atom attention (no kernel).

    Invariants: (1) shared K / V AdaLN as :class:`AtomDiTBoltz2` (the release's ``ada_ln_1`` output feeds q, k and v; no
    ``attention.ada_ln_kv``; a separate K / V AdaLN in a checkpoint layout must equal the query one). (2) No query bias: the release's
    ``to_q`` is ``MultiDimLinear(..., bias=False)`` (:class:`~miniworld_engine.modules.attention.FlatQKNormAugmentedAttention`, so
    ``attention.to_query.bias is None``); a checkpoint that carries a query bias for it must hold zeros, which are dropped (a non-zero
    one is a conversion error). tests/cpu/test_requests_api.py asserts both."""

    version = 2  # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "cuda", "triton")  # "cuda": H100 local_dit_h100, A100 kernels/dit_variants (modules/dit_variant_cuda.py)
    key_window = "pad"
    kv_norm = "shared"
    parallel = True
    attention_part = FlatQKNormAugmentedAttention
    #: 2026-11-10: H100 on the one atom DiT implementation (local_dit_h100; was kernels/dit_variants): 2 "pad", 3 "slide"
    #: 2026-11-12: 4, H100 conditioning at 8448 .. 16832 rows (cond90_split grid.y 2) gave the out gates of LN(c): grid.y 4
    #: 2026-11-13: 5, H100 "slide" / "slide_qblock": the query windows past the last real one read the slid keys [R - 128, R) (the tail
    #: kernels of dit_variants edge.cu; they kept the clipped window). Real rows unchanged.
    backend_versions: ClassVar[dict[str, int]] = {**_LocalAtomDiT.backend_versions, "cuda:h100_sm90": 5}


class AtomDiTChai1(MasterEntryCast, VariantAtomCuda, _LocalAtomDiT):
    """Chai-1's atom transformer block (its TorchScript ``diffusion_module.pt``): ONE identity AdaLN ``(1 + W_s c) LN(a) + W_b c`` for the
    queries and the keys / values (``single_layer_norm``; no ``attention.ada_ln_kv``) and another for the transition
    (``transition.ada_ln_in``), both LayerNorms of the atoms at ``adaln_eps`` (the release: 0.1, the default), no gating query and no output
    projection (``attention.to_out`` is the identity), the parallel schedule, the clipped window (``get_qkv_indices_for_blocks``: the wrapped
    keys are masked; ``key_window="pad"``, the default) and ``forward(..., uid=uid)`` [B, N]: a real query sees only the keys of its own
    conformer (``get_blocked_atom_pair_dists`` ANDs the same-``ref_space_uid`` mask into the block mask in place). No ``"anthropic"`` backend:
    Chai-1's release keeps its stock atom attention (no kernel).

    Invariant (shared K / V AdaLN): the release's ``single_layer_norm`` output feeds the queries and the keys / values (no
    ``attention.ada_ln_kv``); as for :class:`AtomDiTBoltz2`, a separate K / V AdaLN in a checkpoint layout must equal the query one.

    There is no circular window (FoldForge R9): chai_lab 0.6.1 ``get_qkv_indices_for_blocks`` returns ``kv_mask = (kv_indices < N) &
    (kv_indices >= 0)`` before ``kv_indices % N`` (the ``%`` only indexes) and ``get_block_atom_pair_mask`` ANDs it in, so a wrapped key is never
    attended: ``"pad"`` + ``uid`` is the release for every system, also one whose conformer covers both ends of the atom list and one under
    128 atoms (tests/cpu/test_chai1_key_window.py compares the attended keys with a transcription of the release's mask)."""

    version = 2  # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "cuda", "triton")  # "cuda": H100 local_dit_h100, A100 kernels/dit_variants (modules/dit_variant_cuda.py)
    key_window = "pad"
    kv_norm = "shared"
    parallel = True
    attention_part = UngatedAugmentedAttention
    #: 2026-11-10: H100 on the one atom DiT implementation (local_dit_h100; was kernels/dit_variants): 2 "pad", 3 "slide"
    #: 2026-11-13: 4, H100 "slide" / "slide_qblock": the query windows past the last real one read the slid keys [R - 128, R) (the tail
    #: kernels of dit_variants edge.cu; they kept the clipped window). Real rows unchanged.
    backend_versions: ClassVar[dict[str, int]] = {**_LocalAtomDiT.backend_versions, "cuda:h100_sm90": 4}

    def __init__(
        self,
        d_single: int = 128,
        d_cond: int = 128,
        d_pair: int = 16,
        n_head: int = 4,
        n: int = 2,
        *,
        key_window: str | None = None,
        adaln_eps: float = 0.1,
        backend: str = "pytorch",
    ) -> None:
        super().__init__(d_single, d_cond, d_pair, n_head, n, key_window=key_window, backend=backend)
        att = self.attention
        att.ada_ln_in = IdentityAdaptiveLayerNorm(d_single, d_cond, eps=adaln_eps)  # pyright: ignore[reportAttributeAccessIssue]  -- Chai-1 swaps in the identity AdaLN (same call signature)
        att.to_out = nn.Identity()  # pyright: ignore[reportAttributeAccessIssue]  -- Chai-1: no output projection (same call signature)
        self.transition.ada_ln_in = IdentityAdaptiveLayerNorm(d_single, d_cond, eps=adaln_eps)  # pyright: ignore[reportAttributeAccessIssue]  -- Chai-1 swaps in the identity AdaLN (same call signature)

    def _forward_pytorch(self, single, cond, pair, mask=None, uid=None):
        """As :class:`AtomDiTOpenFold3`'s, with the conformer ids ``uid`` [B, N] (None: no conformer mask)."""
        return self._step(single, cond, pair, mask, uid=uid)

    def _forward_triton(self, single, cond, pair, mask=None, uid=None):
        return self._step(single, cond, pair, mask, uid=uid, delta=self._attention_delta_triton)

    def make_inputs(self, **kwargs) -> dict:
        inputs = fp32_conditioning(super().make_inputs(**kwargs), kwargs.get("stream"), kwargs["generator"])
        return {
            **inputs,
            "uid": (torch.arange(inputs["mask"].shape[-1], device=inputs["mask"].device) // 24)[None].expand(inputs["mask"].shape[0], -1),
        }  # conformers of 24 atoms (one row; expanded over the folds of an "atom_folds" call)


# ======================================================================================== MiniWorld's sliding-window (SWA) atom block
class SWA3DRoPEAttention(BackendModule):
    """Sliding-window self-attention with 3D RoPE -- ESMFold2 Algorithm 8 (the attention of :class:`AtomDiTMiniWorld`).

    A replacement for the AF3 block-local atom attention + atom-pair bias: inter-atom geometry enters the queries and keys through 3D rotary
    position embeddings (:func:`~miniworld_engine.modules.functional.build_3d_rope`, from the reference conformer: constant across diffusion
    steps and samples) instead of an additive pair bias, and each atom attends only to the atoms within ``half_window`` of it by rank among
    the valid atoms of its row (a negative ``half_window``, or one at least the row length, is global attention).

    Wqkv -> RMSNorm(q, k) -> 3D RoPE -> sliding-window attention -> sigmoid(gate_proj) gate -> out_proj. qk RMSNorm is over head_dim,
    non-affine; Wqkv/out_proj/gate_proj are bias-free with default init (the block's adaLN-Zero supplies the identity-at-init residual gate).
    Faithful to the ESMFold2 reference code (Biohub/transformers modeling_esmfold2_common.py).

    Operates on a flattened ``[N, S, d]`` sequence where ``N = A * B`` packs the augmentation and batch axes. ``cos``/``sin`` and the
    validity mask are precomputed once and passed in via ``attention_params``
    (:func:`~miniworld_engine.modules.functional.build_attention_params`). The core is a dense band-masked SDPA (O(S^2) memory per row).

    ``backend="anthropic"``: the core on ESMFold2's release (esmfold2/opt/esmfold2_opt/atom_swa.py ``banded_window_attention``, its
    ``atom_swa`` lever): the same band (rank among the valid atoms, the diagonal) evaluated per 2048-query block over its key band, plain
    torch SDPA, so eval and train; ``half_window >= 0``.
    """

    backends = ("pytorch", "anthropic")

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        half_window: int = 64,
        *,
        magnitude_preserving: bool = False,
        mp_full: bool = False,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            msg = f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            raise ValueError(msg)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.half_window = half_window

        magnitude_preserving = magnitude_preserving or mp_full
        dense = MPLinear if magnitude_preserving else Linear
        self.Wqkv = dense(d_model, 3 * d_model, bias=False, init="normal")
        out_cls = MPLinear if mp_full else Linear
        self.out_proj = out_cls(d_model, d_model, bias=False, init="normal")
        self.gate_proj = out_cls(d_model, d_model, bias=False, init="normal")
        self.set_backend(backend)

    def _forward_pytorch(self, x: torch.Tensor, attention_params: tuple) -> torch.Tensor:
        """``x`` [N, S, d]; ``attention_params`` = (cos, sin, seqused, cu_seqlens, max_seqlen, valid) -> [N, S, d]."""
        n, s = x.shape[:2]
        cos, sin, _seqused, _cu_seqlens, _max_seqlen, valid = attention_params

        qkv = self.Wqkv(x).view(n, s, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 1, 3, 4).unbind(0)  # each [N, S, H, D]
        eps = torch.finfo(torch.float32).eps  # float32's epsilon whatever the dtype; the norm in fp32 (fp64 for an fp64 call)
        w = wide(q)
        q = F.rms_norm(q.to(w), (self.head_dim,), eps=eps).to(q.dtype)
        k = F.rms_norm(k.to(w), (self.head_dim,), eps=eps).to(k.dtype)
        q = apply_rotary_emb_3d(q, cos, sin)
        k = apply_rotary_emb_3d(k, cos, sin)

        out = self._sdpa_band(q, k, v, valid)
        out = out.reshape(n, s, -1)
        gate = self.gate_proj(x)
        out = out * torch.sigmoid(gate)
        return self.out_proj(out)

    _release = ("esmfold2_opt.atom_swa", "esmfold2/opt")

    def _supports_anthropic(self, x: torch.Tensor, attention_params: tuple) -> str | None:
        why = _anthropic.missing(*self._release[:1], path=self._release[1])
        if why is not None:
            return why
        if self.half_window < 0:
            return "a negative half_window"
        return None

    def _forward_anthropic(self, x: torch.Tensor, attention_params: tuple) -> torch.Tensor:
        """``_forward_pytorch`` with the band core on ESMFold2's ``banded_window_attention``."""
        swa = _anthropic.load(self._release[0], path=self._release[1])
        n, s = x.shape[:2]
        cos, sin, _seqused, _cu_seqlens, _max_seqlen, valid = attention_params
        qkv = self.Wqkv(x).view(n, s, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 1, 3, 4).unbind(0)
        eps = torch.finfo(torch.float32).eps
        q = F.rms_norm(q.float(), (self.head_dim,), eps=eps).to(q.dtype)
        k = F.rms_norm(k.float(), (self.head_dim,), eps=eps).to(k.dtype)
        q = apply_rotary_emb_3d(q, cos, sin)
        k = apply_rotary_emb_3d(k, cos, sin)
        out = swa.banded_window_attention(q, k, v, valid, self.half_window, self.scale) * valid.unsqueeze(-1).unsqueeze(-1)
        out = out.reshape(n, s, -1)
        return self.out_proj(out * torch.sigmoid(self.gate_proj(x)))

    def _sdpa_band(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Band mask over valid atoms (by rank), dense SDPA. Padding rows come back zero."""
        _n, s = q.shape[:2]
        allowed = valid.unsqueeze(1) & valid.unsqueeze(2)
        if self.half_window >= 0:  # a negative half_window is global attention (FlashAttention's window -1; MiniWorld's dense band)
            rank = torch.cumsum(valid.to(torch.long), dim=1) - 1  # [N, S]
            allowed = allowed & ((rank.unsqueeze(2) - rank.unsqueeze(1)).abs() <= self.half_window)
        allowed = allowed | torch.eye(s, dtype=torch.bool, device=q.device).unsqueeze(0)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=allowed.unsqueeze(1),
            scale=self.scale,
        ).transpose(1, 2)
        return out * valid.unsqueeze(-1).unsqueeze(-1)


def _swa_inputs(width: int, d_cond: int, rope, *, length, samples, dtype, device, generator) -> tuple:
    """An SWA atom block's measurement inputs over ``S`` rows of ``N`` atoms: the stream [S, N, width], the conditioning [S, N, d_cond] and the
    attention params (3D RoPE of a random reference conformer [1, N, 3] with conformers of 8 atoms, the front-packed valid mask [S, N])."""
    pos = random_input((1, length, 3), dtype=torch.float32, device=device, generator=generator, scale=5.0)
    cos, sin = rope(pos, (torch.arange(length, device=device) // 8)[None])
    params = build_attention_params(cos, sin, key_mask(samples, length, device=device), samples)
    return (
        random_input((samples, length, width), dtype=dtype, device=device, generator=generator),
        random_input((samples, length, d_cond), dtype=dtype, device=device, generator=generator),
        params,
    )


class SwiGLUFFN(nn.Module):
    """ESMFold2 SwiGLU with the hidden width rounded up to a multiple of 256: ``w_down(silu(x1) * x2)``, ``[x1 | x2] = w_up(x)``. A part of
    :class:`AtomDiTMiniWorld`."""

    def __init__(self, d_model: int, expansion_ratio: int = 2) -> None:
        super().__init__()
        hidden = ((expansion_ratio * (d_model // 3) * 2) + 255) // 256 * 256
        self.w_up = Linear(d_model, 2 * hidden, bias=False, init="normal")
        self.w_down = Linear(hidden, d_model, bias=False, init="normal")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(x1) * x2)


class AtomDiTMiniWorld(BackendModule):
    """MiniWorld's ESMFold2-style SWA atom block (``block_style="esmfold2"``; ESMFold2 Algorithm 8): adaLN-Zero, sliding-window 3D RoPE
    attention and SwiGLU, no atom-pair representation.

    ``n`` is MiniWorld's ``expansion_ratio``. Inputs have atom-length layout ``[N, S, d]``, where ``N = A * B``. Modulation predicts
    shift, scale, and residual gate for each of attention and FFN. Both gates start at zero, so the block is initially the identity::

        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = W silu(cond)
        x = x + gate_a * attn(rmsnorm(x) * (1 + scale_a) + shift_a)
        x = x + gate_f * ffn(rmsnorm(x) * (1 + scale_f) + shift_f)

    Parameter names follow the ESMFold2 block. A bf16 CUDA activation next to fp32 master matrices runs the block under bf16 autocast
    (:func:`~miniworld_engine.modules.primitives.master_autocast`). :meth:`forward_hoisted` takes the augment-invariant conditioning
    [B, S, d_cond] instead.

    ``backend="anthropic"`` sets its attention's (:class:`SWA3DRoPEAttention`); the block's own statements stay ``"pytorch"``'s.
    """

    # "triton": 2026-11-12, the fused block's Triton stages (engine 2.4's SWADiTBlock)
    backends = ("pytorch", "anthropic", "cuda", "triton")
    # 2026-11-10: kits; sm_80 paths; H100 inference (cached weight forms, slim modulation, fp32 global on the fp32 stages, PDL)
    # cuda:a100_sm80 6 (2026-11-12): bf16 training: the weight forms in one launch (train_forms), row-count launch shapes and K splits, the
    # few-row modulation backward on cuBLAS (sm80.train_cfg)
    backend_versions: ClassVar[dict[str, int]] = {
        "anthropic": 2,
        "cuda:a100_sm80": 7,
        "cuda:h100_sm90": 2,
    }  # cuda:a100_sm80 7 (2026-11-13): the sm_80 window attention exits on a tile past seqused (zero rows, the same bits)

    def __init__(
        self,
        d_atom: int = 128,
        d_cond: int = 128,
        n_head: int = 4,
        n: int = 2,
        half_window: int = 64,
        *,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_atom, elementwise_affine=False)
        self.ffn_norm = RMSNorm(d_atom, elementwise_affine=False)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            Linear(d_cond, 6 * d_atom, bias=False, init="zero"),
        )
        self.attn = SWA3DRoPEAttention(d_atom, n_head, half_window=half_window)
        self.ffn = SwiGLUFFN(d_atom, expansion_ratio=n)
        self.set_backend(backend)

    def set_backend(self, backend: str = "pytorch"):
        super().set_backend(backend)
        self.attn.set_backend(backend if backend in self.attn.backends else "pytorch")
        return self

    # "anthropic": eval -> ESMFold2's fused SWAAtomBlock (esmfold2/opt/forward/fast_inference/driver/ef2_atom.py ``_fused_block``, the kit's
    # fast-tier lever a5 ``fused_block``): K1 [rms-adaLN + the q|k|v|gate GEMM + q / k RMSNorm + 3D RoPE], K2 [window attention per
    # (query tile, head), bf16 x bf16 -> fp32, * sigmoid(gate)], K3 [out_proj + gated residual + rms-adaLN + SwiGLU + gated residual], the
    # GEMMs on tensor cores by the kit's GEMM word (master: "bf16"; fp32: "tf32" under the engine's TF32 rule, else the fp32-faithful
    # "tf32x3"); the modulation silu(c) W^T is the module's statement (the kit hoists it per fold). Its weight packs are re-made when a
    # weight changes. A gradient (the kernels have no backward) or a call outside its range (4 heads x 32 over 128, SwiGLU hidden 256,
    # half_window >= 0, the 6-tuple of build_attention_params with front-packed rows) runs the block's statements with the attention on
    # ESMFold2's ``atom_swa`` band core (:class:`SWA3DRoPEAttention`'s "anthropic").
    _EF2_ATOM = ("ef2_atom", "esmfold2/opt/forward/fast_inference/driver")

    def _fused_reason(self, x: torch.Tensor, cond: torch.Tensor, attention_params: tuple) -> str | None:
        why = _anthropic.unavailable(self._EF2_ATOM[0], x.device, path=self._EF2_ATOM[1])
        if why is not None:
            return why
        if torch.is_grad_enabled() and (x.requires_grad or cond.requires_grad or any(p.requires_grad for p in self.parameters())):
            return "a gradient (the fused block is forward only)"
        at = self.attn
        if (
            (at.n_heads, at.head_dim, x.shape[-1]) != (4, 32, 128)
            or tuple(self.ffn.w_up.weight.shape) != (512, 128)
            or tuple(self.ffn.w_down.weight.shape) != (128, 256)
            or at.half_window < 0
        ):
            return "geometry (the fused block: 128 = 4 heads x 32, SwiGLU hidden 256, half_window >= 0)"
        eps = torch.finfo(torch.float32).eps
        if any(nm.eps not in (None, eps) for nm in (self.attn_norm, self.ffn_norm)):
            return "an RMSNorm eps other than float32's epsilon"
        if x.dtype not in (torch.float32, torch.bfloat16) or x.ndim != 3 or len(attention_params) != 6:
            return "x [N, S, 128] float32 / bfloat16 with the 6-tuple of build_attention_params"
        cos, seqused = attention_params[0], attention_params[2]
        if cos.shape[-1] != 16 or seqused.ndim != 1 or seqused.shape[0] != x.shape[0]:
            return "cos [.., S, 16] and seqused [N]"
        return None

    def _supports_anthropic(self, x: torch.Tensor, cond: torch.Tensor, attention_params: tuple) -> str | None:
        if self._fused_reason(x, cond, attention_params) is None:
            return None
        return self.attn._supports_anthropic(x, attention_params)

    def _forward_anthropic(self, x: torch.Tensor, cond: torch.Tensor, attention_params: tuple) -> torch.Tensor:
        if self._fused_reason(x, cond, attention_params) is not None:
            return self._forward_pytorch(x, cond, attention_params)  # the attention runs its own backend (atom_swa)
        ef = _anthropic.load(*self._EF2_ATOM[:1], path=self._EF2_ATOM[1])
        n, s, d = x.shape
        m = n * s
        with master_autocast(x, self):
            mod = self.adaln_modulation(cond)
        lowp = x.dtype == torch.bfloat16 or torch.is_autocast_enabled("cuda")
        ef._CFG["gemm"] = "bf16" if lowp else ("tf32" if torch.backends.cuda.matmul.allow_tf32 else "tf32x3")
        if not ef.__dict__.get("_engine_sms"):
            ef._sm_count()
            ef._engine_sms = True  # pyright: ignore[reportAttributeAccessIssue]  -- a flag on the kit module, set once per process
        key = tuple((p.data_ptr(), p._version) for p in self.parameters())
        if self.__dict__.get("_katom_key") != key:  # the kit packs once per model load; a changed weight is a new pack
            self.__dict__.pop("_katom_packs", None)
            self.__dict__["_katom_key"] = key
        cos, sin, seqused = attention_params[0], attention_params[1], attention_params[2]
        fs = {
            "M": m,
            "N": s,
            "Bp": n,
            "seqlen": seqused.to(torch.int32).contiguous(),
            "cos": cos.float().expand(n, s, -1).reshape(m, -1).contiguous(),
            "sin": sin.float().expand(n, s, -1).reshape(m, -1).contiguous(),
        }
        with torch.autocast("cuda", enabled=False):
            xin = x.float().reshape(m, d).contiguous()
            out, _ = ef._fused_block(self, xin, mod.float().reshape(m, -1).contiguous(), fs)
        return out.view(n, s, d).to(x.dtype)

    # ------------------------------------------------------------------------------------------------------------ "cuda"
    # The whole block on the fused SWA kernels (kernels/swa_dit: the H100 sm_90a stages -- modulation silu(c) W^T, qkvg + q / k RMSNorm +
    # 3D RoPE, the window attention, out-projection + gated residual + SwiGLU FFN; a gradient: SWADiTBlockFunction / SWADiTModulationH100,
    # their backwards on the same family; global attention (a half window >= the global threshold) on FlashAttention-4 between the native
    # stages, an fp32 inference call's between the fp32 block's own row stages). Inference (``_h100_eval``) caches the weight casts / packs
    # per weight version and skips the backward's modulation copies; a stack shares silu(c) over its blocks. Range: H100, bf16 / fp32 / master activations (weights in the activation dtype or fp32), d_atom 128 = 4 heads x 32, d_cond
    # 128, SwiGLU hidden 256, half_window 64 or global (FlashAttention-4 installed), the 6-tuple of build_attention_params (fp32 RoPE without
    # gradient, int32 front-packed seqused [N]). fp32: TF32 composed stages (swa_fp32.cu) when TF32 is allowed, the native stages with exact
    # fp32 cuBLAS projections under matmul precision "highest" (kernels/swa_dit/dispatch.native_h100_stages).
    def _supports_cuda(self, x: torch.Tensor, cond: torch.Tensor, attention_params: tuple) -> str | None:
        from miniworld_engine.kernels.swa_dit._common import D_ATOM, HALF_WINDOW, N_HEAD, N_HIDDEN, is_global

        if _a100(x):
            return self._a100_reason(x, cond, attention_params)
        if not x.is_cuda or torch.cuda.get_device_capability(x.device) != (9, 0):
            return "the H100 (sm_90) kernels only"
        if len(attention_params) != 6:
            return "attention_params must be the 6-tuple of build_attention_params"
        if x.dtype not in (torch.bfloat16, torch.float32) or cond.dtype != x.dtype:
            return f"x / cond dtype {x.dtype} / {cond.dtype} (bfloat16 or float32, the same)"
        if x.ndim != 3 or x.shape[-1] != D_ATOM or self.attn.n_heads != N_HEAD:
            return f"x must be [N, S, {D_ATOM}] with {N_HEAD} heads"
        if tuple(cond.shape) != (*x.shape[:2], D_ATOM):
            return f"cond must be [N, S, {D_ATOM}]"
        hw = self.attn.half_window
        if is_global(hw):
            from miniworld_engine.kernels.swa_dit.cuda import sm100

            if not sm100.FA4_AVAILABLE:
                return "global attention needs FlashAttention-4 (flash_attn.cute), not installed"
        elif hw != HALF_WINDOW:
            return f"half_window {hw} (the kernels: {HALF_WINDOW} or global)"
        linears = (self.adaln_modulation[1], self.attn.Wqkv, self.attn.gate_proj, self.attn.out_proj, self.ffn.w_up, self.ffn.w_down)
        if any(type(m) is not Linear or m.bias is not None for m in linears):
            return "the kernels read bias-free (non magnitude-preserving) Linear weights"
        if tuple(self.ffn.w_up.weight.shape) != (2 * N_HIDDEN, D_ATOM):
            return f"SwiGLU hidden {self.ffn.w_up.weight.shape[0] // 2} (the kernels: {N_HIDDEN})"
        if any(m.weight.dtype not in (x.dtype, torch.float32) for m in linears):
            return "weights must be in the activation dtype or fp32"
        cos, sin, seqused = attention_params[:3]
        for t in (cos, sin):
            if t.dtype != torch.float32 or t.shape[-1] != D_ATOM // N_HEAD // 2 or (t.requires_grad and torch.is_grad_enabled()):
                return "cos / sin must be fp32 [..., head_dim / 2] without gradient"
        if seqused.dtype != torch.int32 or tuple(seqused.shape) != (x.shape[0],):
            return "seqused must be int32 [N]"
        return None

    # A100 (sm_80): kernels/swa_dit/cuda/sm80. bf16 inference with the window: the fused stages (modulation mod_fwd, then block_fwd:
    # qkvg + q / k RMSNorm + 3D RoPE, the window attention, out-projection + gated residual + SwiGLU FFN; mma.sync). Training, fp32 and
    # global attention: sm80/composed.py (the same bf16 stages and their backwards; fp32 on the row stages around cuBLAS projections and
    # the bf16 attention, as the H100 native fp32 block; global attention on PyTorch's FlashAttention-2). fp32 inference under TF32: the two
    # fused TF32 row kernels (sm80/fp32_fwd_sm80.cuh: qkvg, out-projection + FFN) around the attention; a bf16 global inference call takes
    # Q / K / V token-major from the qkvg kernel (no transposes for FlashAttention-2). Range: bf16 or fp32 activations
    # (weights in that dtype or fp32 masters), d_atom 128 = 4 heads x 32, d_cond 128, SwiGLU hidden 256, half_window 64 or global.
    def _a100_linears(self):
        return (self.adaln_modulation[1], self.attn.Wqkv, self.attn.gate_proj, self.attn.out_proj, self.ffn.w_up, self.ffn.w_down)

    def _a100_reason(self, x: torch.Tensor, cond: torch.Tensor, attention_params: tuple) -> str | None:
        from miniworld_engine.kernels.swa_dit._common import D_ATOM, HALF_WINDOW, N_HEAD, N_HIDDEN, is_global

        if len(attention_params) != 6:
            return "attention_params must be the 6-tuple of build_attention_params"
        if x.dtype not in (_BF, _F32) or cond.dtype != x.dtype:
            return f"A100: x / cond dtype {x.dtype} / {cond.dtype} (bfloat16 or float32, the same)"
        if torch.is_autocast_enabled("cuda") and (x.dtype != _BF or torch.get_autocast_dtype("cuda") != _BF):
            return "A100: autocast: bfloat16 activations under bfloat16 autocast only"
        linears = self._a100_linears()
        if x.ndim != 3 or x.shape[-1] != D_ATOM or self.attn.n_heads != N_HEAD or tuple(cond.shape) != (*x.shape[:2], D_ATOM):
            return f"A100: x / cond must be [N, S, {D_ATOM}] with {N_HEAD} heads"
        hw = self.attn.half_window
        if is_global(hw):
            if not torch.backends.cuda.is_flash_attention_available():
                return "A100: global attention needs PyTorch's FlashAttention-2 (not in this build)"
        elif hw != HALF_WINDOW:
            return f"A100: half_window {hw} (the sm_80 kernels: {HALF_WINDOW} or global)"
        if any(type(m) is not Linear or m.bias is not None for m in linears):
            return "the kernels read bias-free (non magnitude-preserving) Linear weights"
        if tuple(self.ffn.w_up.weight.shape) != (2 * N_HIDDEN, D_ATOM):
            return f"SwiGLU hidden {self.ffn.w_up.weight.shape[0] // 2} (the kernels: {N_HIDDEN})"
        if any(m.weight.dtype not in (x.dtype, _F32) for m in linears):
            return "weights must be in the activation dtype or fp32"
        cos, sin, seqused = attention_params[:3]
        for t in (cos, sin):
            if (
                t.dtype != _F32
                or t.shape[-1] != D_ATOM // N_HEAD // 2
                or t.numel() != x.shape[0] * x.shape[1] * t.shape[-1]
                or (t.requires_grad and torch.is_grad_enabled())
            ):
                return "cos / sin must be fp32 [N, S, head_dim / 2] without gradient"
        if seqused.dtype != torch.int32 or tuple(seqused.shape) != (x.shape[0],):
            return "seqused must be int32 [N]"
        return None

    def _forward_a100(self, x: torch.Tensor, cond: torch.Tensor, attention_params: tuple) -> torch.Tensor:
        from miniworld_engine.kernels.swa_dit._common import FP32_EPS, HALF_WINDOW
        from miniworld_engine.kernels.swa_dit.cuda import sm80
        from miniworld_engine.kernels.swa_dit.cuda.sm80 import composed

        n = x.shape[0]
        cos, sin, seqused = attention_params[:3]
        hw = self.attn.half_window
        weights = [m.weight for m in self._a100_linears()]
        if _needs_grad(self, x, cond):
            return composed.SWABlockA100.apply(x, cond, cos, sin, seqused.contiguous(), hw, *weights)
        if x.dtype != _BF or hw != HALF_WINDOW:
            return composed.block(x, cond, cos, sin, seqused.contiguous(), hw, weights)
        from miniworld_engine.kernels.swa_dit.dispatch import eval_form

        cos, sin = cos.reshape(-1, cos.shape[-1]).contiguous(), sin.reshape(-1, sin.shape[-1]).contiguous()
        # the bf16 forms of the (fp32 master) weights once per weight version; a capture reads the eager forms (no six casts per replay)
        wmod, *weights = eval_form("swa_a100_bf16", weights, lambda: [w.to(_BF).contiguous() for w in weights])  # pyright: ignore[reportCallIssue]  -- tensors of the weight list (untyped Module attributes)
        with torch.autocast("cuda", enabled=False):
            mod = sm80.mod_fwd(cond.reshape(-1, cond.shape[-1]).contiguous(), wmod, False)[0]
            # one modulation row per sequence row: b = N
            return sm80.block_fwd(x.contiguous(), mod, cos, sin, seqused.contiguous(), *weights, n, FP32_EPS, FP32_EPS, False)[0]  # pyright: ignore[reportCallIssue]  -- the weight list unpacked into the launcher's fixed parameters

    def _forward_cuda(self, x: torch.Tensor, cond: torch.Tensor, attention_params: tuple) -> torch.Tensor:
        if _a100(x):
            return self._forward_a100(x, cond, attention_params)
        from miniworld_engine.kernels.swa_dit.autograd import SWADiTBlockFunction, SWADiTModulationH100
        from miniworld_engine.kernels.swa_dit.dispatch import native_h100_silu

        n, _s = x.shape[:2]
        cos, sin, seqused = attention_params[:3]
        cos, sin = cos.reshape(-1, cos.shape[-1]).contiguous(), sin.reshape(-1, sin.shape[-1]).contiguous()
        c = cond.reshape(-1, cond.shape[-1]).contiguous()
        wmod = self.adaln_modulation[1].weight.contiguous()  # pyright: ignore[reportCallIssue]  -- adaln_modulation[1] is the Linear (nn.Sequential indexing is untyped)
        weights = [
            w.contiguous()
            for w in (
                self.attn.Wqkv.weight,
                self.attn.gate_proj.weight,
                self.attn.out_proj.weight,
                self.ffn.w_up.weight,
                self.ffn.w_down.weight,
            )
        ]
        q = x.contiguous()
        if torch.is_grad_enabled() and any(t.requires_grad for t in (q, c, wmod, *weights)):
            from miniworld_engine.kernels.integrations.token_dit_train import _native_caller_output

            mod = _native_caller_output(SWADiTModulationH100.apply(c, wmod), cond, wmod, force=True)
            # one modulation row per sequence row: B = N (no augment structure is known here)
            return SWADiTBlockFunction.apply(q, mod, cos, sin, seqused, *weights, n, self.attn.half_window)
        return self._h100_eval(q, native_h100_silu(c), cos, sin, seqused)

    def _h100_eval(self, q: torch.Tensor, a: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seqused: torch.Tensor) -> torch.Tensor:
        """The H100 inference block from ``a = silu(cond)`` (``native_h100_silu``, [N*S, d_cond] in q's dtype; a stack conditioned on one
        cond computes it once): the modulation GEMM, then the fused block, its weights cast / packed once per weight version
        (``dispatch.eval_form``). q [N, S, 128] contiguous, cos / sin [N*S, 16] -> [N, S, 128]."""
        from miniworld_engine.kernels.swa_dit._common import FP32_EPS
        from miniworld_engine.kernels.swa_dit.dispatch import native_h100_eval_weights, native_h100_mod_eval, swa_dit_block_fwd

        mod = native_h100_mod_eval(a, self.adaln_modulation[1].weight)  # pyright: ignore[reportArgumentType]  -- adaln_modulation[1] is the Linear (nn.Sequential indexing is untyped)
        weights = native_h100_eval_weights(
            (self.attn.Wqkv.weight, self.attn.gate_proj.weight, self.attn.out_proj.weight, self.ffn.w_up.weight, self.ffn.w_down.weight),
            q.dtype,
        )
        return swa_dit_block_fwd(q, mod, cos, sin, seqused, *weights, q.shape[0], self.attn.half_window, FP32_EPS, False)[0]  # pyright: ignore[reportCallIssue]  -- the weight list unpacked into the launcher's fixed parameters

    # ------------------------------------------------------------------------------------------------------------ "triton"
    # The whole block on the fused SWA block's Triton stages (kernels/swa_dit/triton, any card: ``swa_dit_block_fwd`` / ``_bwd`` with
    # ``portable=True``): qkvg + q / k RMSNorm + 3D RoPE, the window attention, out-projection + gated residual + SwiGLU FFN, and their
    # backwards (bf16; fp32 on the fp32 row kernels around the bf16-operand window attention); the weight gradients and the modulation
    # ``silu(cond) W^T`` (fp32, one row per sequence row) are cuBLAS. Engine 2.4's SWADiTBlock ran these off H100 / B200. Range: bf16 or
    # fp32 activations (weights in that dtype or fp32 masters), d_atom 128 = 4 heads x 32, SwiGLU hidden 256, half_window 64 or global
    # (2026-11-12: the same window kernels with every key in reach, dispatch.TRITON_GLOBAL_HW), the 6-tuple of build_attention_params.
    def _supports_triton(self, x: torch.Tensor, cond: torch.Tensor, attention_params: tuple) -> str | None:
        from miniworld_engine.kernels.swa_dit._common import D_ATOM, HALF_WINDOW, N_HEAD, N_HIDDEN, is_global

        if not x.is_cuda:
            return "the Triton kernels run on CUDA tensors only"
        if len(attention_params) != 6:
            return "attention_params must be the 6-tuple of build_attention_params"
        if x.dtype not in (_BF, _F32) or cond.dtype != x.dtype:
            return f"x / cond dtype {x.dtype} / {cond.dtype} (bfloat16 or float32, the same)"
        if torch.is_autocast_enabled("cuda") and (x.dtype != _BF or torch.get_autocast_dtype("cuda") != _BF):
            return "autocast: bfloat16 activations under bfloat16 autocast only"
        linears = self._a100_linears()
        if x.ndim != 3 or x.shape[-1] != D_ATOM or self.attn.n_heads != N_HEAD or tuple(cond.shape[:2]) != tuple(x.shape[:2]):
            return f"x / cond must be [N, S, {D_ATOM}] / [N, S, d_cond] with {N_HEAD} heads"
        if self.attn.half_window != HALF_WINDOW and not is_global(self.attn.half_window):
            return f"half_window {self.attn.half_window} (the Triton window attention: {HALF_WINDOW} or global)"
        if any(type(m) is not Linear or m.bias is not None for m in linears):
            return "the kernels read bias-free (non magnitude-preserving) Linear weights"
        if tuple(self.ffn.w_up.weight.shape) != (2 * N_HIDDEN, D_ATOM):
            return f"SwiGLU hidden {self.ffn.w_up.weight.shape[0] // 2} (the kernels: {N_HIDDEN})"
        if any(m.weight.dtype not in (x.dtype, _F32) for m in linears):
            return "weights must be in the activation dtype or fp32"
        cos, sin, seqused = attention_params[:3]
        for t in (cos, sin):
            if (
                t.dtype != _F32
                or t.shape[-1] != D_ATOM // N_HEAD // 2
                or t.numel() != x.shape[0] * x.shape[1] * t.shape[-1]
                or (t.requires_grad and torch.is_grad_enabled())
            ):
                return "cos / sin must be fp32 [N, S, head_dim / 2] without gradient"
        if seqused.dtype != torch.int32 or tuple(seqused.shape) != (x.shape[0],):
            return "seqused must be int32 [N]"
        return None

    def _forward_triton(self, x: torch.Tensor, cond: torch.Tensor, attention_params: tuple) -> torch.Tensor:
        from miniworld_engine.kernels.swa_dit._common import FP32_EPS
        from miniworld_engine.kernels.swa_dit.autograd import SWADiTBlockFunction
        from miniworld_engine.kernels.swa_dit.dispatch import swa_dit_block_fwd

        n = x.shape[0]
        cos, sin, seqused = attention_params[:3]
        cos, sin = cos.reshape(-1, cos.shape[-1]).contiguous(), sin.reshape(-1, sin.shape[-1]).contiguous()
        wmod, *weights = (cast(torch.Tensor, m.weight) for m in self._a100_linears())
        with torch.autocast("cuda", enabled=False):
            # the modulation in fp32 (the kernels' operand), one row per sequence row (B = N); differentiable in cond and its weight
            mod = F.linear(F.silu(cond.reshape(-1, cond.shape[-1]).float()), wmod.float())
            if _needs_grad(self, x, cond):
                return SWADiTBlockFunction.apply(
                    x.contiguous(), mod, cos, sin, seqused.contiguous(), *weights, n, self.attn.half_window, True
                )
            weights = [w.to(x.dtype).contiguous() for w in weights]
            args = (x.contiguous(), mod, cos, sin, seqused.contiguous(), *weights, n, self.attn.half_window, FP32_EPS, False, True)
            return swa_dit_block_fwd(*args)[0]  # pyright: ignore[reportCallIssue]  -- the weight list unpacked into fixed parameters

    def forward_hoisted(self, x: torch.Tensor, c_base: torch.Tensor, attention_params: tuple) -> torch.Tensor:
        """The block over ``N = A * B`` rows whose conditioning repeats over the augments: ``cond[a*B + b] == c_base[b]``, with
        ``c_base`` [B, S, d_cond]. The same function as ``forward(x, c_base.repeat(A, 1, 1), attention_params)``."""
        return self(x, c_base.repeat(x.shape[0] // c_base.shape[0], 1, 1), attention_params)

    def _forward_pytorch(self, x: torch.Tensor, cond: torch.Tensor, attention_params: tuple) -> torch.Tensor:
        """``x`` [N, S, d_atom], ``cond`` [N, S, d_cond], ``attention_params`` from ``build_attention_params`` -> [N, S, d_atom]."""
        with master_autocast(x, self):
            shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.adaln_modulation(cond).chunk(6, dim=-1)
            attn_in = self.attn_norm(x) * (1 + scale_a) + shift_a
            out = x + gate_a * self.attn(attn_in, attention_params)
            ffn_in = self.ffn_norm(out) * (1 + scale_f) + shift_f
            out = out + gate_f * self.ffn(ffn_in)
        return out.to(x.dtype)

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        x, cond, params = _swa_inputs(
            self.attn_norm.normalized_shape[0],
            self.adaln_modulation[1].in_features,  # pyright: ignore[reportArgumentType]  -- adaln_modulation[1] is the Linear (nn.Sequential indexing is untyped)
            lambda pos, uid: build_3d_rope(pos, uid, self.attn.head_dim),
            length=length,
            samples=samples,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        return {"x": x, "cond": cond, "attention_params": params}


class AtomDiTESMFold2(AtomDiTMiniWorld):
    """ESMFold2's release atom block (Biohub transformers ``modeling_esmfold2_common.py`` ``SWAAtomBlock``, the block of its
    ``SWAAtomTransformer`` in the atom encoder and decoder): pair-free, ``adaln_modulation = Sequential(SiLU, Linear(no bias))`` giving
    shift / scale / raw gate for the attention and the FFN, an affine-free RMSNorm of the atoms, ``SWA3DRoPEAttention`` (``Wqkv`` without
    bias, an affine-free q / k RMSNorm, the 3D rotary, the band ``|rank_i - rank_j| <= 64`` among the valid atoms, ``sigmoid(gate_proj)`` of
    the normalised input, ``out_proj``) and ``SwiGLUFFN``. It is :class:`AtomDiTMiniWorld` (MiniWorld adopted this block), under
    ESMFold2's constructor: ``forward(x [A * B, S, d_single], cond [A * B, S, d_cond], attention_params)``
    (:func:`~miniworld_engine.modules.functional.build_attention_params`). ``backend="anthropic"``: ESMFold2's own band core.

    Not FoldForge's ESMFold2 atom stack (FoldForge R6): that one is the AF3 block over a 32 x 192 atom pair (a pair bias this block has no
    input for), :class:`AtomDiTESMFold2Pair` (``AtomDiT(model="esmfold2_pair")``); no option turns one into the other."""

    version = 1  # human-managed: bump when a change alters this class's accuracy or speed results

    def __init__(self, d_single: int = 128, d_cond: int = 128, n_head: int = 4, n: int = 2, *, backend: str = "pytorch") -> None:
        super().__init__(d_single, d_cond, n_head, n, half_window=64, backend=backend)


def _legacy_options_removed(where: str, **options: bool) -> None:
    """The legacy magnitude-preserving options were removed: accepted as false (configs that spell them out keep loading), they raise when on."""
    on = sorted(name for name, value in options.items() if value)
    if on:
        names = ", ".join(on)
        msg = f"{where}: the legacy option(s) {names} were removed; restore them from git history if needed."
        raise NotImplementedError(msg)


class SWAAtomTransformer(BackendModule):
    """Stack of :class:`AtomDiTMiniWorld` blocks (ESMFold2 Algorithm 8; MiniWorld's atom transformer and the stack ``InputFeatureEmbedder``
    holds). Each block is a canonical DiT adaLN-Zero block (Peebles & Xie, arXiv:2212.09748), not the AF3 conditioning (``sigmoid``-scale
    AdaLN + ``bias=-2`` output gate): ESMFold2 reverts the atom transformer to the original DiT block while keeping AF3's attention with pair
    bias only at the token level. ``backend="anthropic"`` sets every block's (their attention on ESMFold2's release band core)."""

    version = 1  # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "anthropic", "cuda", "triton")  # "triton": every block's (AtomDiTMiniWorld._forward_triton)
    # 2026-11-10: kits; sm_80 paths; H100 inference (cached weight forms, slim modulation, fp32 global on the fp32 stages, PDL)
    # cuda:a100_sm80 6 (2026-11-12): bf16 training: the weight forms in one launch (train_forms), row-count launch shapes and K splits, the
    # few-row modulation backward on cuBLAS (sm80.train_cfg)
    # triton 2 (2026-11-12): global attention (half_window >= 65536 or negative) served, on the window kernels with every key in reach
    # cuda:h100_sm90 3 (2026-11-12): bf16 / master global attention on the composed bf16 stages around FlashAttention-4 (dispatch._h100_global_composed;
    # was the native stages): master train L 1024 S 48 2.19 -> 1.24 ms
    backend_versions: ClassVar[dict[str, int]] = {
        "anthropic": 2,
        "cuda:a100_sm80": 7,
        "cuda:h100_sm90": 3,
        "triton": 2,
    }  # cuda:a100_sm80 7 (2026-11-13): as AtomDiTMiniWorld

    class Config(BaseModel):
        """Configuration for the SWA atom transformer."""

        d_atom: int = 128
        d_cond: int = 128
        n_block: int = 3
        # Activation-checkpoint the atom blocks (None = off; 1..n_block). The atom DiT activations (A*B*L_atom x d_atom) dominate the
        # fp32 diffusion memory at large num_augment, so this is the primary lever for fitting big crops.
        n_checkpoint_segments: int | None = None
        n_head: int = 4
        swa_window_size: int = 128
        expansion_ratio: int = 2
        # "esmfold2": ESMFold2 SWAAtomBlock (adaLN-Zero + SwiGLU). "af3": MiniWorld's AF3 atom block with the 3D-RoPE sliding-window
        # attention core (read by the caller; this stack builds AtomDiTMiniWorld blocks either way).
        block_style: str = "esmfold2"
        # global_attn: with block_style "af3", GLOBAL self-attention (all atoms attend to all) instead of the sliding window (read by the
        # caller).
        global_attn: bool = False
        # local_structure_attn: with block_style "af3", attend to explicit neighbors (sequence + noisy-coordinate nearest atoms) through
        # ``functional.sparse_neighbor_attention`` (read by the caller).
        local_structure_attn: bool = False
        seq_neighbors: int = 128
        structure_neighbors: int = 128
        structure_query_chunk_size: int = 128
        sparse_attention_query_chunk_size: int = 64
        n_spatial_rope_pairs_per_axis: int = 2
        spatial_rope_base_frequency: float = 20.0
        n_uid_rope_pairs: int = 10
        uid_rope_base_frequency: float = 10000.0
        magnitude_preserving: bool = False
        use_rotation: bool = False
        mp_full: bool = False
        mp_residual: bool = False
        residual_t: float = 0.3

    def __init__(self, config: Config, *, backend: str = "pytorch") -> None:
        super().__init__()
        _legacy_options_removed(
            "SWAAtomTransformer", magnitude_preserving=config.magnitude_preserving, mp_full=config.mp_full, mp_residual=config.mp_residual
        )
        self.config = config
        self.head_dim = config.d_atom // config.n_head
        self.blocks = nn.ModuleList(
            [
                AtomDiTMiniWorld(
                    config.d_atom,
                    config.d_cond,
                    config.n_head,
                    n=config.expansion_ratio,
                    half_window=config.swa_window_size // 2,
                )
                for _ in range(config.n_block)
            ],
        )
        self.set_backend(backend)

    def set_backend(self, backend: str = "pytorch"):
        super().set_backend(backend)
        for block in cast("list[AtomDiTMiniWorld]", self.blocks):
            block.set_backend(backend)
        return self

    # "cuda": every block on the fused SWA kernels (:meth:`AtomDiTMiniWorld._forward_cuda`); the range is the blocks' (all alike).
    def _supports_cuda(self, q, c, attention_params, c_base=None) -> str | None:
        return cast(AtomDiTMiniWorld, self.blocks[0])._supports_cuda(q, c, attention_params) if len(self.blocks) else None

    def _forward_cuda(self, q, c, attention_params, c_base=None):
        blocks = cast("list[AtomDiTMiniWorld]", list(self.blocks))
        if (
            len(blocks) > 1
            and _h100(q)
            and self.config.n_checkpoint_segments is None
            and not (torch.is_grad_enabled() and (q.requires_grad or c.requires_grad or any(p.requires_grad for p in self.parameters())))
        ):
            # H100 inference: every block is conditioned on the same c, so silu(c) -- the modulation GEMM's operand -- is made once
            from miniworld_engine.kernels.swa_dit.dispatch import native_h100_silu

            cos, sin, seqused = attention_params[:3]
            cos, sin = cos.reshape(-1, cos.shape[-1]).contiguous(), sin.reshape(-1, sin.shape[-1]).contiguous()
            a = native_h100_silu(c.reshape(-1, c.shape[-1]).contiguous())
            q = q.contiguous()
            for block in blocks:
                q = block._h100_eval(q, a, cos, sin, seqused)
            return q
        return self._forward_pytorch(q, c, attention_params, c_base)  # the blocks run their own backend

    def _forward_anthropic(self, q, c, attention_params, c_base=None):
        return self._forward_pytorch(q, c, attention_params, c_base)  # the blocks run their own backend

    # "triton": every block on the fused block's Triton stages (:meth:`AtomDiTMiniWorld._forward_triton`); the range is the blocks'.
    def _supports_triton(self, q, c, attention_params, c_base=None) -> str | None:
        return cast(AtomDiTMiniWorld, self.blocks[0])._supports_triton(q, c, attention_params) if len(self.blocks) else None

    def _forward_triton(self, q, c, attention_params, c_base=None):
        return self._forward_pytorch(q, c, attention_params, c_base)  # the blocks run their own backend

    def build_rope(self, ref_pos: torch.Tensor, ref_space_uid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The (cos, sin) [B, N, head_dim // 2] 3D RoPE tensors for one batch (no augment axis): ``ref_pos`` [B, N, 3], ``ref_space_uid``
        [B, N]."""
        return build_3d_rope(
            ref_pos,
            ref_space_uid,
            self.head_dim,
            n_spatial_per_axis=self.config.n_spatial_rope_pairs_per_axis,
            n_uid_pairs=self.config.n_uid_rope_pairs,
            spatial_base_freq=self.config.spatial_rope_base_frequency,
            uid_base_freq=self.config.uid_rope_base_frequency,
        )

    def _forward_pytorch(
        self,
        q: torch.Tensor,
        c: torch.Tensor,
        attention_params: tuple,
        c_base: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass over the flattened ``N = A * B`` atom sequence: ``q`` [N, S, d_atom], ``c`` [N, S, d_cond]. ``c_base``
        [B, S, d_cond], the augment-invariant conditioning that ``c`` repeats, is accepted for the caller's contract and not needed."""
        del c_base
        if self.config.n_checkpoint_segments is None:
            for block in self.blocks:
                q = block(q, c, attention_params)
            return q

        # Activation checkpointing over the atom blocks: only segment-boundary q is kept; c / attention_params are per-call constants,
        # closed over here.
        def run_module(module):
            def forward(x):
                return module(x, c, attention_params)

            return forward

        return checkpoint_sequential(
            [run_module(b) for b in self.blocks],
            segments=self.config.n_checkpoint_segments,
            use_reentrant=False,
            input=q,
        )

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        q, c, params = _swa_inputs(
            self.config.d_atom,
            self.config.d_cond,
            self.build_rope,
            length=length,
            samples=samples,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        return {"q": q, "c": c, "attention_params": params}


# ================================================================ ESMFold2's atom block with the atom pair (FoldForge R6)
#: The 192-key subset and the +/- 64 rank window of :class:`AtomDiTESMFold2Pair`.
ESM_KEYS, ESM_HALF = 192, 64


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary on the head axis ``x [..., N, h, d]`` with per-atom tables ``cos`` / ``sin`` [..., N, r] (r <= d / 2), split into halves:
    channel ``c < r`` pairs with ``c + r``, ``(a, b) -> (a cos - b sin, b cos + a sin)``; channels from ``2 r`` on pass unchanged (FoldForge
    ``diffusion_transformer.apply_rope``: tiled tables, not interleaved)."""
    r = cos.shape[-1]
    c, s = cos[..., None, :].to(x.dtype), sin[..., None, :].to(x.dtype)
    a, b = x[..., :r], x[..., r : 2 * r]
    return torch.cat([a * c - b * s, b * c + a * s, x[..., 2 * r :]], dim=-1)


def _rms(x: torch.Tensor) -> torch.Tensor:
    """Affine-free RMSNorm over the last axis at float32's epsilon (FoldForge ``RMSNorm``, ESMFold2's ``rope_norm``)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS)


class _ESMAttention(nn.Module):
    """The attention half of :class:`AtomDiTESMFold2Pair` (parameters only; the math is the block's)."""

    def __init__(self, d_single: int, d_cond: int, d_pair: int, n_head: int) -> None:
        super().__init__()
        self.n_head = n_head
        self.ada_ln_in = AdaptiveRMSNorm(d_single, d_cond)
        self.ada_ln_kv = AdaptiveRMSNorm(d_single, d_cond)
        self.to_query = Linear(d_single, d_single, bias=True)
        self.to_key = Linear(d_single, d_single, bias=False)
        self.to_value = Linear(d_single, d_single, bias=False)
        self.ln_pair = LayerNorm(d_pair, bias=False)
        self.to_bias = Linear(d_pair, n_head, bias=False, init="zero")
        self.to_gate = Linear(d_single, d_single, bias=False, init="gating")
        self.to_out = Linear(d_single, d_single, bias=False, init="zero")
        self.to_scale = Linear(d_cond, d_single, bias=False, init="zero")


class _ESMTransition(nn.Module):
    """The transition half of :class:`AtomDiTESMFold2Pair` (parameters only)."""

    def __init__(self, d_hidden: int, d_cond: int, n: int) -> None:
        super().__init__()
        self.d_hidden, self.d_cond = d_hidden, d_cond
        self.ada_ln_in = AdaptiveRMSNorm(d_hidden, d_cond)
        self.expand_a = Linear(d_hidden, n * d_hidden, bias=False)
        self.expand_b = Linear(d_hidden, n * d_hidden, bias=False)
        self.squeeze = Linear(n * d_hidden, d_hidden, bias=False, init="zero")
        self.to_scale = Linear(d_cond, d_hidden, bias=False, init="zero")


class AtomDiTESMFold2Pair(BackendModule):
    """FoldForge's ESMFold2 / ESMFold2-fast atom block (FoldForge R6; ``dense/diffusion_transformer.py`` ``CrossAttention`` +
    ``DiffusionTransition`` under ``spec.ESMFOLD2``): AlphaFold 3's atom block -- parallel K / V AdaLN of the raw atoms, the slid key
    window, a pair bias from the trunked atom pair, the sequential schedule -- with four changes:

    * every AdaLN is :class:`~miniworld_engine.modules.norm.AdaptiveRMSNorm`: ``(1 + W_s silu(c)) * rms(x) + W_b silu(c)`` (no biases);
    * both output gates are raw ``W silu(c)`` (no sigmoid, no bias);
    * the key window is a 192-key subset slid AlphaFold 3's way (``[32 w - 80, 32 w + 112)``,
      :func:`~miniworld_engine.modules.functional.slide_starts` with ``keys=192``) in which query ``i`` sees key ``j`` only when
      ``|i - j| <= 64`` (the rank among the real atoms, which come first);
    * q and k pass an affine-free per-head RMSNorm and a rotary with per-atom tables the caller supplies: ``forward(..., rope=(cos, sin))``,
      each ``[B, N, r]``, ``r <= head_dim / 2`` (the keys take their own atoms' rows; FoldForge ``build_atom_rope``).

    ``single`` / ``cond`` ``[A, B, N, d]``, trunked ``pair`` ``[B, nwin, 32, 192, d_pair]`` (the window's keys laid out from its first key,
    as FoldForge's ``pair_cond`` gathered with the same subsets), key ``mask`` ``[B, N]`` (real atoms first). Not :class:`AtomDiTESMFold2`:
    that is ESMFold2's released pair-free ``SWAAtomBlock`` (fused ``Wqkv``, adaLN-Zero, 3D RoPE inside); this is the block FoldForge's
    ESMFold2 port builds (``AtomDiT(model="esmfold2_pair")``).

    Backends: ``"pytorch"`` (the reference), ``"triton"`` (the window attention core on ``kernels/augmented_attention/triton/window.py``, the
    rest as ``"pytorch"``), ``"cuda"`` (``kernels/integrations/local_dit_esm.py``: H100 inference on the sm90_atom row kernels built with
    ``RMS``, the q / k RMSNorm + rotary of ``qknorm_local.cu`` and the 192-key rank-window attention of ``lattn90_esm.cu``; H100 training and
    A100 inference / training on the row passes of ``dit_variants/cuda/train_rows.cu`` + cuBLAS around the same attention kernels).
    """

    version = 1  # human-managed: bump when a change alters this class's accuracy or speed results
    backends = ("pytorch", "triton", "cuda")

    def __init__(
        self, d_single: int = 128, d_cond: int = 128, d_pair: int = 16, n_head: int = 4, n: int = 2, *, backend: str = "pytorch"
    ) -> None:
        super().__init__()
        self.attention = _ESMAttention(d_single, d_cond, d_pair, n_head)
        self.transition = _ESMTransition(d_single, d_cond, n)
        self.set_backend(backend)

    # --------------------------------------------------------------------------------------------------------------------- the math
    def attention_operands(self, single, cond, pair, mask, rope):
        """``x`` (the q AdaLN output), q / k (normalised and rotated, fp32 or wider) and v [A, B, N, H, D], the pair bias
        [B, H, nwin, 32, 192], the window starts [B, nwin] and the key mask [B, N] (float)."""
        att = self.attention
        x = att.ada_ln_in(single, cond)
        xkv = att.ada_ln_kv(single, cond)
        a, b, n, c = x.shape
        h = att.n_head
        d = c // h
        nwin = windows(n)
        if tuple(pair.shape[:4]) != (b, nwin, QUERIES, ESM_KEYS):
            raise ValueError(f"pair must be the trunked atom pair [B, {nwin}, {QUERIES}, {ESM_KEYS}, d_pair], got {tuple(pair.shape)}")
        if rope is None or len(rope) != 2:
            raise ValueError("this block rotates q and k: pass rope=(cos, sin), each [B, N, r]")
        cos, sin = rope
        if tuple(cos.shape[:2]) not in ((b, n), (1, n)) or cos.shape != sin.shape or 2 * cos.shape[-1] > d:
            raise ValueError(f"rope tables must be [B, N, r] with r <= {d // 2}, got {tuple(cos.shape)} / {tuple(sin.shape)}")
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = F.layer_norm(pair, (pair.shape[-1],), att.ln_pair.weight.to(pair.dtype), None, att.ln_pair.eps)
        bias = F.linear(z, att.to_bias.weight.to(z.dtype)).permute(0, 4, 1, 2, 3)
        w = wide(x)
        q = master_linear(att.to_query, x)
        k = master_linear(att.to_key, xkv)
        v = master_linear(att.to_value, xkv).reshape(a, b, n, h, d)
        q = apply_rope(_rms(q.to(w).reshape(a, b, n, h, d)), cos.to(w), sin.to(w))
        k = apply_rope(_rms(k.to(w).reshape(a, b, n, h, d)), cos.to(w), sin.to(w))
        keep = torch.ones(b, n, device=x.device) if mask is None else mask.float()
        starts = slide_starts(keep.sum(-1).long(), nwin, ESM_KEYS)
        return x, q, k, v, bias, starts, keep

    def _gates(self, single, cond, x, out):
        """``W_sc silu(c) * W_o (sigmoid(W_g x) * out)``: the attention's update from its core output ``out`` [A, B, N, H * D]."""
        att = self.attention
        out = master_linear(att.to_out, torch.sigmoid(master_linear(att.to_gate, x)) * out)
        return master_linear(att.to_scale, F.silu(cond)) * out

    def attention_delta(self, single, cond, pair, mask=None, rope=None) -> torch.Tensor:
        """The attention's update (no residual), PyTorch: softmax over each window's 192 keys, the +/- 64 band and the key mask."""
        x, q, k, v, bias, starts, keep = self.attention_operands(single, cond, pair, mask, rope)
        a, b, n, h, d = q.shape
        nwin, keys = windows(n), ESM_KEYS
        w = q.dtype
        key_pos = starts[:, :, None] + torch.arange(keys, device=x.device)  # [B, nwin, keys]
        qw = F.pad(q.reshape(a, b, n, h * d), (0, 0, 0, nwin * QUERIES - n)).reshape(a, b, nwin, QUERIES, h, d)
        kw = _gather_windows(k.reshape(a, b, n, h * d), key_pos).reshape(a, b, nwin, keys, h, d)
        vw = _gather_windows(v.reshape(a, b, n, h * d), key_pos).reshape(a, b, nwin, keys, h, d)
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = torch.einsum("abwqhd,abwkhd->abhwqk", qw, kw.to(w)) * d**-0.5 + bias.to(w)[None]
        valid = _gather_windows(keep[None, ..., None], key_pos)[0, ..., 0] > 0  # [B, nwin, keys]
        query_pos = torch.arange(nwin * QUERIES, device=x.device).view(1, nwin, QUERIES, 1)
        valid = valid[:, :, None, :] & ((query_pos - key_pos[:, :, None, :]).abs() <= ESM_HALF)  # [B, nwin, 32, keys]
        scores = scores.masked_fill(~valid[None, :, None], float("-inf"))
        probs = torch.nan_to_num(torch.softmax(scores, dim=-1))
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = torch.einsum("abhwqk,abwkhd->abwqhd", probs, vw.to(w)).to(x.dtype)
        out = out.reshape(a, b, nwin * QUERIES, h * d)[:, :, :n]
        return self._gates(single, cond, x, out)

    def transition_delta(self, single, cond) -> torch.Tensor:
        """The transition's update (no residual): AdaptiveRMSNorm, SwiGLU, the raw gate ``W silu(c)``."""
        tr = self.transition
        x = tr.ada_ln_in(single, cond)
        hidden = F.silu(master_linear(tr.expand_a, x)) * master_linear(tr.expand_b, x)
        return master_linear(tr.to_scale, F.silu(cond)) * master_linear(tr.squeeze, hidden)

    def _step(self, single, cond, pair, mask, rope, delta):
        single = single + delta(single, cond, pair, mask, rope)
        return single + self.transition_delta(single, cond)

    def _forward_pytorch(self, single, cond, pair, mask=None, rope=None):
        """``single`` [A, B, N, d_single], ``cond`` [A, B, N, d_cond], trunked ``pair`` [B, nwin, 32, 192, d_pair], key ``mask`` [B, N],
        ``rope`` (cos, sin) [B, N, r] -> the stream after the attention and the transition (both residuals included)."""
        return self._step(single, cond, pair, mask, rope, self.attention_delta)

    # ------------------------------------------------------------------------------------------------------------ "triton"
    def _supports_triton(self, single, cond, pair, mask=None, rope=None) -> str | None:
        if not single.is_cuda:
            return "the Triton window attention runs on CUDA tensors only"
        if single.dtype not in (_BF, _F32):
            return f"dtype {single.dtype} (bfloat16 / float32 only)"
        if single.ndim != 4 or cond.ndim != 4:
            return f"single / cond must be [A, B, N, d], got {tuple(single.shape)} / {tuple(cond.shape)}"
        b, n, c = single.shape[1:]
        h = self.attention.n_head
        if c % h or c // h > 64:
            return f"head dim {c / h:g} (the Triton window kernel takes a whole head dim <= 64)"
        if tuple(pair.shape[:4]) != (b, windows(n), QUERIES, ESM_KEYS):
            return f"pair must be the trunked atom pair [B, {windows(n)}, {QUERIES}, {ESM_KEYS}, d_pair], got {tuple(pair.shape)}"
        if mask is not None and tuple(mask.shape) != (b, n):
            return f"mask must be [B, N], got {tuple(mask.shape)}"
        if rope is None:
            return "rope=(cos, sin) is required"
        return None

    def _attention_delta_triton(self, single, cond, pair, mask=None, rope=None) -> torch.Tensor:
        """:meth:`attention_delta` with the window softmax on the Triton kernel (same windows, band, key mask and empty-row zero)."""
        from miniworld_engine.kernels.augmented_attention.triton.window import window_attention

        x, q, k, v, bias, starts, keep = self.attention_operands(single, cond, pair, mask, rope)
        a, b, n, h, d = q.shape
        q, k = q.to(x.dtype), k.to(x.dtype)
        out = window_attention(q, k, v, bias, starts.to(torch.int32).contiguous(), keep > 0, keys=ESM_KEYS, window_half=ESM_HALF)
        return self._gates(single, cond, x, out.to(x.dtype).reshape(a, b, n, h * d))

    def _forward_triton(self, single, cond, pair, mask=None, rope=None):
        return self._step(single, cond, pair, mask, rope, self._attention_delta_triton)

    # -------------------------------------------------------------------------------------------------------------- "cuda"
    def _supports_cuda(self, single, cond, pair, mask=None, rope=None) -> str | None:
        from miniworld_engine.kernels.integrations import local_dit_esm

        return local_dit_esm.reason(self, single, cond, pair, mask, rope)

    def _forward_cuda(self, single, cond, pair, mask=None, rope=None):
        from miniworld_engine.kernels.integrations import local_dit_esm

        # the kernels' precision is their own (bf16 operands, or TF32 for fp32): an enclosing autocast must not recast their torch glue
        with torch.autocast("cuda", enabled=False):
            return local_dit_esm.run_block(self, single, cond, pair, mask, rope)

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        """``single`` / ``cond`` [S, 1, N, d] (S samples), the trunked ``pair`` [1, nwin, 32, 192, d_pair], the key ``mask`` [1, N] (real atoms
        first, padding inside the last query window) and the rotary tables ``rope`` [1, N, head_dim / 2] (fp32: FoldForge builds them so).
        Stream ``atom_single_shared_cond``: ``cond`` [1, 1, N, d] expanded (stride 0) over the S samples (requirements/README.md)."""
        d_single, d_cond = self.transition.d_hidden, self.transition.d_cond
        shared = shared_cond_stream(stream)[1]
        d_pair = self.attention.ln_pair.normalized_shape[0]
        half = d_single // self.attention.n_head // 2
        angle = 3 * random_input((1, length, half), dtype=torch.float32, device=device, generator=generator)
        rdt = torch.float64 if dtype == torch.float64 else torch.float32
        return {
            "single": random_input((samples, 1, length, d_single), dtype=dtype, device=device, generator=generator),
            "cond": (
                random_input((1, 1, length, d_cond), dtype=dtype, device=device, generator=generator).expand(samples, -1, -1, -1)
                if shared
                else random_input((samples, 1, length, d_cond), dtype=dtype, device=device, generator=generator)
            ),
            "pair": random_input((1, windows(length), QUERIES, ESM_KEYS, d_pair), dtype=dtype, device=device, generator=generator),
            "mask": key_mask(1, length, device=device, pad=QUERIES - 8),
            "rope": (angle.cos().to(rdt), angle.sin().to(rdt)),
        }


# ======================================================================================================================== selector
#: model name -> its atom DiT class. Several models share a class where their blocks are the same math.
ATOM_DIT_MODELS: dict[str, type[BackendModule]] = {
    "af3": AtomDiTOpenFold3,
    "boltz2": AtomDiTBoltz2,
    "rosettafold3": AtomDiTRoseTTAFold3,
    "chai1": AtomDiTChai1,
    "openfold3": AtomDiTOpenFold3,
    "openfold3_preview2": AtomDiTOpenFold3,
    "intellifold2": AtomDiTOpenFold3,
    "protenix": AtomDiTProtenix,
    "protenix_v2": AtomDiTProtenix,
    "opendde": AtomDiTProtenix,
    "esmfold2": AtomDiTESMFold2,
    "esmfold2_pair": AtomDiTESMFold2Pair,
    "miniworld": AtomDiTMiniWorld,
}


class AtomDiT:
    """``AtomDiT(model=..., *args, **kwargs)``: builds ``ATOM_DIT_MODELS[model](*args, **kwargs)`` (the model's own class, not a wrapper:
    its state dict has the model class's keys)."""

    models = ATOM_DIT_MODELS

    def __new__(cls, model: str, *args, **kwargs):
        try:
            klass = ATOM_DIT_MODELS[model]
        except KeyError:
            raise ValueError(f"AtomDiT: unknown model {model!r}; known: {', '.join(sorted(ATOM_DIT_MODELS))}") from None
        return klass(*args, **kwargs)


__all__ = [
    "ATOM_DIT_MODELS",
    "ESM_HALF",
    "ESM_KEYS",
    "KEYS",
    "KEY_OFFSET",
    "KEY_WINDOWS",
    "QUERIES",
    "AtomDiT",
    "AtomDiTBoltz2",
    "AtomDiTChai1",
    "AtomDiTESMFold2",
    "AtomDiTESMFold2Pair",
    "AtomDiTMiniWorld",
    "AtomDiTOpenFold3",
    "AtomDiTProtenix",
    "AtomDiTRoseTTAFold3",
    "SWA3DRoPEAttention",
    "SWAAtomTransformer",
    "SwiGLUFFN",
    "windows",
]
