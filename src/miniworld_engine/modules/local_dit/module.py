"""The AF3 atom transformer block (Alg. 23 with the block-local attention of Alg. 24): 32 queries x 128 keys per window.

    a = a + AttentionPairBias(a, s, z)     a = a + ConditionedTransition(a, s)

Atom ``i`` belongs to query window ``w = i // 32`` and attends to the 128 atoms ``[32 w - 48, 32 w + 80)`` (the window clipped to
``[0, N)``), with a per-window pair bias. The pair tensor is the *trunked* atom pair ``[B, nwin, 32, 128, d_pair]``,
``pair[b, w, i, j] = z[b, 32 w + i, 32 w - 48 + j]`` with ``nwin = ceil(N / 32)`` (:func:`to_windows` makes it from a dense pair;
entries outside ``[0, N)`` are never read). The block has the parameters of :class:`~miniworld_engine.modules.dit.DiTBlock` at atom
widths (d_single = d_cond = 128, d_pair = 16, 4 heads, transition n = 2), so the dense and the local block share a checkpoint layout;
only the attention differs. With ``cross_attention=True`` (AF3's / Protenix's atom blocks) the keys and values are projected from a second
AdaLN of the already normalised atoms, ``attention.ada_ln_kv`` (four more parameters). ``single`` and ``cond`` carry the augmentation axis ``[A, B, N, d]``, the pair does not.

ESMFold2's atom block (:data:`ESMFOLD2`) is the AF3 one with four changes: every AdaLN is :class:`AdaptiveRMSNorm` (affine-free RMSNorm
of the atoms, SiLU of the conditioning, ``(1 + s)`` scale, no biases), both output gates are raw ``W silu(s)`` (no sigmoid, no bias),
q and k pass a per-head RMSNorm and a rotary with per-atom tables the caller supplies, and a window holds 192 keys of which a query
sees those within +/- 64 of it by rank among the real atoms.

The reference path below is plain PyTorch (the window gather by ``unfold``); on a B200 ``integrations/local_dit`` serves bf16 calls
with the sm_100a kernels of ``kernels/augmented_attention/cuda/sm100_atom_local``, and fp32 calls (fp32 parameters) with their TF32 twins.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from miniworld_engine.integrations import h100_master as _master
from miniworld_engine.integrations import local_dit as _local
from miniworld_engine.modules.adaptive_layernorm import (
    AdaptiveLayerNorm,
    AdaptiveRMSNorm,
)
from miniworld_engine.modules.adaptive_layernorm.module import RMS_EPS
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.primitives import Linear

QUERIES, KEYS, KEY_OFFSET = 32, 128, 48

#: ESMFold2's atom block (FoldForge ``spec.ESMFOLD2`` / ``ESMFOLD2_FAST``): AlphaFold 3's parallel K/V AdaLN and slid window, at 192
#: keys with the +/- 64 rank window, RMS / SiLU conditioning with raw gates, and q / k RMSNorm + rotary (``forward(..., rope=(cos, sin))``).
ESMFOLD2 = {
    "cross_attention": True,
    "kv_norm": "parallel",
    "key_window": "slide",
    "keys": 192,
    "window_half": 64,
    "conditioning": "rms_silu",
    "use_qk_norm": True,
    "rope": True,
}


def windows(n: int) -> int:
    return (n + QUERIES - 1) // QUERIES


def _key_windows(t: torch.Tensor, n: int, value: float = 0.0, keys: int = KEYS) -> torch.Tensor:
    """[..., N, C] -> [..., nwin, keys, C]: the key window of every query window (out-of-range positions are ``value``)."""
    nwin = windows(n)
    offset = (keys - QUERIES) // 2
    padded = F.pad(t, (0, 0, offset, nwin * QUERIES + keys - QUERIES - offset - n), value=value)
    return padded.unfold(-2, keys, QUERIES).transpose(-1, -2)



def slide_starts(n_real: torch.Tensor, nwin: int, keys: int = KEYS) -> torch.Tensor:
    """[B] real atom counts -> [B, nwin] first key of every query window, AlphaFold 3's rule: the window ``[32 w - 48, 32 w + 80)``
    shifted back inside the real atoms ``[0, N)`` (``featurisation: AtomCrossAtt.compute_features``). A window short of the start
    begins at 0, one past the end at ``N - 128``. Below 128 real atoms the rule has no valid window; the window then begins at 0 and
    the keys past ``N`` are masked (AlphaFold 3's own indices wrap there).

    A wider subset (``keys`` 192: ``[32 w - 80, 32 w + 112)``) takes the rule exactly as written there: a window short of the start
    begins at 0, otherwise it ends at atom ``N - 1`` at the latest, so below ``keys`` real atoms a window may begin before atom 0
    (AlphaFold 3's negative indices, padding atoms; :func:`slide_windows` masks them)."""
    first = QUERIES * torch.arange(nwin, device=n_real.device) - (keys - QUERIES) // 2
    if keys == KEYS:
        first = first.clamp(min=0)
        return torch.minimum(first[None], (n_real - KEYS)[:, None]).clamp(min=0)
    return torch.where(first[None] < 0, 0, torch.minimum(first[None], (n_real - keys)[:, None]))


def slide_windows(t: torch.Tensor, starts: torch.Tensor, keys: int = KEYS) -> torch.Tensor:
    """[A, B, N, C] and starts [B, nwin] -> [A, B, nwin, keys, C]: the slid key window of every query window (a position outside
    ``[0, N)`` reads 0)."""
    n = t.shape[2]
    index = starts[:, :, None] + torch.arange(keys, device=t.device)               # [B, nwin, keys]
    inside = (index >= 0) & (index < n)
    batch = torch.arange(t.shape[1], device=t.device)[:, None, None]
    out = t[:, batch, index.clamp(0, n - 1)]
    return out * inside[None, ..., None].to(out.dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary on the head axis ``x [..., N, h, d]`` with per-atom tables ``cos`` / ``sin`` [..., N, r] (r <= d / 2), split into halves:
    channel ``c < r`` pairs with ``c + r``, ``(a, b) -> (a cos - b sin, b cos + a sin)``; channels from ``2 r`` on pass unchanged."""
    r = cos.shape[-1]
    c, s = cos[..., None, :].to(x.dtype), sin[..., None, :].to(x.dtype)
    a, b = x[..., :r], x[..., r : 2 * r]
    return torch.cat([a * c - b * s, b * c + a * s, x[..., 2 * r :]], dim=-1)


def to_windows(pair: torch.Tensor, keys: int = KEYS) -> torch.Tensor:
    """Dense pair [B, N, N, d] -> trunked pair [B, nwin, 32, keys, d] (the clipped window; positions outside [0, N) are zero)."""
    n = pair.shape[1]
    nwin = windows(n)
    query = torch.arange(nwin * QUERIES, device=pair.device).view(nwin, QUERIES)
    key = torch.arange(nwin, device=pair.device)[:, None] * QUERIES - (keys - QUERIES) // 2 + torch.arange(keys, device=pair.device)[None]
    valid = (query < n)[:, :, None] & ((key >= 0) & (key < n))[:, None, :]
    out = pair[:, query.clamp(max=n - 1)[:, :, None], key.clamp(0, n - 1)[:, None, :]]
    return out * valid[None, :, :, :, None].to(pair.dtype)


class LocalDiTBlock(DiTBlock):
    """AF3 atom transformer block: block-local pair-bias attention, then the conditioned transition.

    ``cross_attention`` adds the K / V AdaLN (``attention.ada_ln_kv``). ``kv_norm`` says what it normalises: ``"chained"`` the atoms the
    first AdaLN has already normalised (Protenix, the default), ``"parallel"`` the raw atoms next to it (AlphaFold 3).
    ``key_window`` says where a query window's 128 keys start: ``"pad"`` at ``32 w - 48``, clipped to the atoms that exist, so an edge
    window sees fewer keys (the default), or ``"slide"`` shifted back inside the real atoms (AlphaFold 3: the first two windows start at 0,
    the last ones end at atom ``N - 1``); the trunked pair is read as laid out from the window's first key either way. ``"slide"`` takes
    the real atom count ``N`` from the key mask, whose real atoms must come first; under 128 real atoms there is no valid window and the
    keys past ``N`` are masked. The H100 mma kernels serve both for inference; a call with grad runs the composed path.

    ESMFold2's options (:data:`ESMFOLD2` holds the set): ``keys`` is the window's key count (128, or 192 centred the same way:
    ``[32 w - 80, 32 w + 112)``, and the pair is ``[B, nwin, 32, keys, d_pair]``); ``window_half`` lets query ``i`` see key ``j`` only when
    ``|i - j| <= window_half`` (positions, which are the ranks among the real atoms since those come first); ``conditioning="rms_silu"``
    makes the three AdaLNs :class:`AdaptiveRMSNorm` and the attention's and the transition's output gates raw ``W silu(s)`` (no sigmoid, no
    bias); ``rope`` rotates q and k (after the qk RMSNorm when ``use_qk_norm``) with per-atom tables, ``forward(..., rope=(cos, sin))``,
    each ``[B, N, r]`` with ``r <= head_dim / 2`` (the keys take their own atoms' rows).
    """

    def __init__(
        self,
        d_single: int = 128,
        d_cond: int = 128,
        d_pair: int = 16,
        n_head: int = 4,
        n: int = 2,
        *,
        cross_attention: bool = False,
        kv_norm: str = "chained",
        key_window: str = "pad",
        keys: int = KEYS,
        window_half: int | None = None,
        conditioning: str = "layernorm",
        rope: bool = False,
        use_qk_norm: bool = False,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__(d_single, d_cond, d_pair, n_head, n, use_qk_norm=use_qk_norm, implementation=implementation)
        self.cross_attention = cross_attention
        if kv_norm not in ("chained", "parallel"):
            raise ValueError(f"kv_norm must be 'chained' or 'parallel', got {kv_norm!r}")
        if key_window not in ("pad", "slide"):
            raise ValueError(f"key_window must be 'pad' or 'slide', got {key_window!r}")
        if kv_norm == "parallel" and not cross_attention:
            raise ValueError("kv_norm='parallel' needs cross_attention=True (there is no K/V AdaLN otherwise)")
        if conditioning not in ("layernorm", "rms_silu"):
            raise ValueError(f"conditioning must be 'layernorm' or 'rms_silu', got {conditioning!r}")
        if keys < QUERIES or (keys - QUERIES) % 2:
            raise ValueError(f"keys must be at least {QUERIES} and centre the query window, got {keys}")
        if window_half is not None and not 0 <= window_half <= (keys - QUERIES) // 2:
            raise ValueError(f"window_half must lie in [0, {(keys - QUERIES) // 2}] for {keys} keys, got {window_half}")
        self.kv_norm = kv_norm
        self.key_window = key_window
        self.keys = keys
        self.window_half = window_half
        self.conditioning = conditioning
        self.rope = rope
        att, tr = self.attention, self.transition
        if conditioning == "rms_silu":
            att.ada_ln_in = AdaptiveRMSNorm(d_single, d_cond)
            att.to_scale = Linear(d_cond, d_single, bias=False, init="zero")
            tr.ada_ln_in = AdaptiveRMSNorm(d_single, d_cond)
            tr.to_scale = Linear(d_cond, d_single, bias=False, init="zero")
            if use_qk_norm:   # float32's epsilon whatever the dtype, like every RMSNorm of this family
                att.norm_query.eps = att.norm_key.eps = RMS_EPS
        if cross_attention:
            att.ada_ln_kv = (AdaptiveRMSNorm(d_single, d_cond) if conditioning == "rms_silu"
                             else AdaptiveLayerNorm(d_single, d_cond, implementation=implementation))

    def attention_delta(self, single, cond, pair, mask=None, rope=None) -> torch.Tensor:
        """The attention's update (no residual), PyTorch: AdaLN, q / k / v / gate, softmax over each 32 x ``keys`` window, gates."""
        att = self.attention
        rms = self.conditioning == "rms_silu"
        x = att.ada_ln_in(single, cond)
        master = (att.implementation != ImplementationType.PYTORCH and x.dtype == torch.bfloat16
                  and att.to_query.weight.dtype == torch.float32 and _master.is_h100(x.device))
        wq = wk = wv = wg = wo = ws = wb = bq = bs = None
        if master and not rms:
            wq, wk, wv, wg, wo, ws, wb, bq, bs = _master.pack([
                att.to_query.weight, att.to_key.weight, att.to_value.weight,
                att.to_gate.weight, att.to_out.weight, att.to_scale.weight,
                att.to_bias.weight, att.to_query.bias, att.to_scale.bias,
            ])

        def proj(module, t, weight=None, bias=None):
            if not master:
                return module(t)
            return _master.linear(t, module, weight, bias) if weight is not None else _master.master_linear(module, t)

        a, b, n, _ = x.shape
        h, d = att.n_head, x.shape[-1] // att.n_head
        nwin, keys = windows(n), self.keys
        if tuple(pair.shape[:4]) != (b, nwin, QUERIES, keys):
            raise ValueError(f"pair must be the trunked atom pair [B, {nwin}, {QUERIES}, {keys}, d_pair], got {tuple(pair.shape)}")
        if self.rope and rope is None:
            raise ValueError("this block rotates q and k: pass rope=(cos, sin), each [B, N, r]")
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = F.layer_norm(pair, (pair.shape[-1],), att.ln_pair.weight.to(pair.dtype), None, att.ln_pair.eps)
        bias = (proj(att.to_bias, z, wb) if master else F.linear(z, att.to_bias.weight.to(z.dtype))).permute(0, 4, 1, 2, 3)
        q = proj(att.to_query, x, wq, bq)
        # AF3 cross-attention mode: a second AdaLN for K / V (ty types the nn.Module attribute as Tensor). Chained: of the already
        # normalised atoms (Protenix); parallel: of the raw atoms, next to the first one (AlphaFold 3).
        kv_input = single if self.kv_norm == "parallel" else x
        xkv = att.ada_ln_kv(kv_input, cond) if self.cross_attention else x  # ty: ignore[call-non-callable]
        k = proj(att.to_key, xkv, wk)
        v = proj(att.to_value, xkv, wv)
        gate = proj(att.to_gate, x, wg)
        if att.use_qk_norm:   # per head, over the head width, one weight per tensor
            q = att.norm_query(q.reshape(a, b, n, h, d)).reshape(a, b, n, h * d)
            k = att.norm_key(k.reshape(a, b, n, h, d)).reshape(a, b, n, h * d)
        wide = torch.float64 if x.dtype == torch.float64 else torch.float32
        if self.rope:   # after the norm, in the score precision; the keys are rotated per atom, before the window gather
            cos, sin = rope
            if tuple(cos.shape[:2]) != (b, n) or cos.shape != sin.shape or 2 * cos.shape[-1] > d:
                raise ValueError(f"rope tables must be [B, N, r] with r <= {d // 2}, got {tuple(cos.shape)} / {tuple(sin.shape)}")
            q = apply_rope(q.to(wide).reshape(a, b, n, h, d), cos.to(wide), sin.to(wide)).reshape(a, b, n, h * d)
            k = apply_rope(k.to(wide).reshape(a, b, n, h, d), cos.to(wide), sin.to(wide)).reshape(a, b, n, h * d)
        qw = F.pad(q, (0, 0, 0, nwin * QUERIES - n)).reshape(a, b, nwin, QUERIES, h, d)
        keep = torch.ones(b, n, device=x.device) if mask is None else mask.float()
        if self.key_window == "slide":
            n_real = keep.sum(-1).long()
            starts = slide_starts(n_real, nwin, keys)
            kw = slide_windows(k, starts, keys).reshape(a, b, nwin, keys, h, d)
            vw = slide_windows(v, starts, keys).reshape(a, b, nwin, keys, h, d)
        else:
            kw = _key_windows(k, n, keys=keys).reshape(a, b, nwin, keys, h, d)
            vw = _key_windows(v, n, keys=keys).reshape(a, b, nwin, keys, h, d)
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = torch.einsum("abwqhd,abwkhd->abhwqk", qw.to(wide), kw.to(wide)) * d**-0.5 + bias.to(wide)[None]
        if self.key_window == "slide":
            # the slid window lies inside the real atoms; only a system under `keys` atoms has keys past them
            valid = slide_windows(keep[None, ..., None], starts, keys)[0, ..., 0] > 0
            key_pos = starts[:, :, None] + torch.arange(keys, device=x.device)                       # [B, nwin, keys]
        else:
            valid = _key_windows(keep[..., None], n, keys=keys)[..., 0] > 0                # [B, nwin, keys]
            key_pos = (torch.arange(nwin, device=x.device)[:, None] * QUERIES - (keys - QUERIES) // 2
                       + torch.arange(keys, device=x.device))[None].expand(b, -1, -1)
        valid = valid[:, :, None, :]                                                               # [B, nwin, 1, keys]
        if self.window_half is not None:   # +/- window_half by rank among the real atoms = by position (the real atoms come first)
            query_pos = torch.arange(nwin * QUERIES, device=x.device).view(1, nwin, QUERIES, 1)
            valid = valid & ((query_pos - key_pos[:, :, None, :]).abs() <= self.window_half)
        scores = scores.masked_fill(~valid[None, :, None], float("-inf"))
        probs = torch.nan_to_num(torch.softmax(scores, dim=-1))
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = torch.einsum("abhwqk,abwkhd->abwqhd", probs, vw.to(wide)).to(x.dtype)
        out = out.reshape(a, b, nwin * QUERIES, h * d)[:, :, :n]
        out = proj(att.to_out, torch.sigmoid(gate) * out, wo)
        if rms:   # ESMFold2: the raw gate of the activated conditioning
            return proj(att.to_scale, F.silu(cond)) * out
        scale = proj(att.to_scale, cond, ws, bs)
        return torch.sigmoid(scale) * out

    def transition_delta(self, single, cond) -> torch.Tensor:
        """The transition's update (no residual). ``conditioning="rms_silu"``: AdaptiveRMSNorm, SwiGLU, then the raw gate ``W silu(s)``;
        otherwise the ConditionedTransition's own."""
        tr = self.transition
        if self.conditioning != "rms_silu":
            return tr.delta(single, cond)
        x = tr.ada_ln_in(single, cond)
        hidden = F.silu(_master.master_linear(tr.expand_a, x)) * _master.master_linear(tr.expand_b, x)
        return _master.master_linear(tr.to_scale, F.silu(cond)) * _master.master_linear(tr.squeeze, hidden)

    def forward(self, single, cond, pair, mask=None, rope=None):
        """``single`` [A, B, N, d_single], ``cond`` [A, B, N, d_cond], trunked ``pair`` [B, nwin, 32, keys, d_pair], key ``mask`` [B, N],
        ``rope`` (cos, sin) [B, N, r] when the block rotates q / k -> the stream after the attention and the transition (both residuals
        included)."""
        if _local.serves(self, single, cond, pair, mask, rope):
            return _local.block(self, single, cond, pair, mask, rope)
        single = single + self.attention_delta(single, cond, pair, mask, rope)
        if self.conditioning == "rms_silu":
            return single + self.transition_delta(single, cond)
        return self.transition(single, cond)
