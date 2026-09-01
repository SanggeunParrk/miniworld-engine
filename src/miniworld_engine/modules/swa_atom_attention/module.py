"""Sliding-window atom attention with 3D RoPE (ESMFold2 style).

This is the attention primitive of the ESMFold2 ``SWAAtomTransformer``
(Algorithm 8) — a replacement for the AF3 block-local atom attention +
atom-pair bias used by :class:`AugmentedAttentionPairBias`. Two differences
from the AF3 path:

* **No atom-pair tensor.** Inter-atom geometry is injected into the queries and
  keys via 3D rotary position embeddings (:func:`build_3d_rope`) instead of an
  additive ``[B, L, L, n_head]`` pair bias. This removes the O(L_atom^2) pair
  tensor entirely.
* **Sliding window.** Each atom attends only to a local window of ``2*half``
  neighbours in atom-index order, via FlashAttention-4's native sliding-window
  support. Padding is handled by ``seqused_k`` over the full padded ``[N, S]``
  layout (fixed-stride varlen) — no unpad/gather, no ``torch.nonzero``, fully
  static shape (CUDA-graph capturable). A dense SDPA band-mask fallback is used
  when FlashAttention is unavailable (e.g. CPU unit tests).

The 3D RoPE is derived from the *reference conformer* coordinates
(``reference.pos``) and per-atom space UIDs (``reference.space_uid``), both of
which are fixed inputs — the cos/sin are therefore constant across diffusion
steps and augmentation samples.

Faithful to the ESMFold2 reference code (Biohub/transformers
modeling_esmfold2_common.py): sliding-window attention with qk RMSNorm and a
sigmoid output gate; Wqkv/out_proj/gate_proj are bias-free default-init Linears.

Ref: ESMFold2, "Language Modeling Materializes a World Model of Protein
Biology", Algorithm 8 / Algorithm 9; RoPE: Su et al. (arXiv:2104.09864).
"""

from __future__ import annotations

import importlib.util
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int

from miniworld_engine import settings
from miniworld_engine._typecheck import typecheck
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.modules.primitives import Linear, MPLinear


# FlashAttention-4 (CuTeDSL) is the Hopper/Blackwell windowed-attention backend
# (FA2 has no sm_100 / B200 support). Its import pulls CUTLASS, so we only *probe*
# availability here — ``find_spec`` locates the module without executing it — and
# import ``flash_attn_varlen_func`` lazily inside :func:`flash_window_seqused`. This
# keeps ``import miniworld_engine.modules`` PyTorch-first (SDPA fallback) and free of any GPU
# backend until the flash path actually runs. Optimizing this via a
# ``miniworld_engine`` op is the intended future path; for now the fallback is torch.
#: Which flash backend this process can actually use: "fa4", "fa2" or None.
#:
#: Resolved once, from what is INSTALLED and what the card can run -- not from the package name
#: alone. `find_spec("flash_attn") is not None` was both: FA4 and FA2 share the top-level package,
#: so it could not tell them apart, and it never asked the card, so an FA4 wheel on an A100 sent
#: sm80 down a Hopper/Blackwell path. FA4 (CuTeDSL) needs sm90+; FA2 covers sm80 and has no sm100
#: support, which is why both are kept rather than one replacing the other.
#:
#: Probed with `find_spec`, which locates a module without executing it: importing FA4 pulls
#: CUTLASS, and `import miniworld_engine.modules` stays PyTorch-first and GPU-backend-free until
#: the flash path actually runs. The real import stays lazy, inside `flash_window_seqused`.
def _has_submodule(parent: str, child: str) -> bool:
    """Is ``parent.child`` installed, WITHOUT importing either.

    `find_spec("flash_attn.cute")` would import `flash_attn` to find its submodule -- and raise
    ModuleNotFoundError outright when the parent is absent, which is the common case. Looking for
    the file under the parent's search locations keeps the probe free of both.
    """
    try:
        spec = importlib.util.find_spec(parent)
    except (ImportError, ValueError):  # a broken or shadowed install
        return False
    if spec is None or not spec.submodule_search_locations:
        return False
    for root in spec.submodule_search_locations:
        base = pathlib.Path(root)
        if (base / child / "__init__.py").exists() or (base / f"{child}.py").exists():
            return True
        if any(base.glob(f"{child}.*.so")):  # a compiled extension module
            return True
    return False


_FA4_SPEC = _has_submodule("flash_attn", "cute")
_FA2_SPEC = _has_submodule("flash_attn", "flash_attn_interface")


def _flash_backend(device: torch.device | None = None) -> str | None:
    """"fa4" | "fa2" | None for `device` (default: the current CUDA device)."""
    if not torch.cuda.is_available():
        return None
    major = torch.cuda.get_device_capability(device)[0]
    if _FA4_SPEC and major >= 9:
        return "fa4"
    if _FA2_SPEC and major >= 8:
        return "fa2"
    return None


#: Kept for callers that only ask "is there any flash path at all".
_FLASH_AVAILABLE = _FA4_SPEC or _FA2_SPEC

# Opt-in runtime guard (``settings.swa_check_front_packed``) that asserts the seqused_k
# precondition — valid atoms are front-packed in each row. Off by default so it
# adds no per-step GPU sync / torch.compile graph break in production.
def _check_front_packed() -> bool:
    """Read at call time; see layernorm.compile_native._ln_bwd_override."""
    return settings.current().swa_check_front_packed


def _flash_window_seqused_fake(q, k, v, cu_seqlens, seqused, max_seqlen, valid, n, s, scale,
                               half_window):
    """[N, S, H, D] — the attention output carries q's shape and dtype."""
    return torch.empty_like(q)


# FA4's CuTeDSL kernel is opaque to dynamo. Under compile_wrap="disable" this is a clean
# eager leaf so torch.compile breaks here once (no per-trace resume churn / cache pressure)
# and CUDA graphs (reduce-overhead) capture the compiled regions around it; under
# "custom_op" it is an opaque graph node instead and nothing breaks at all.
@opaque(fake=_flash_window_seqused_fake, name="swa_atom_attention_flash_window")
def flash_window_seqused(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cu_seqlens: torch.Tensor, seqused: torch.Tensor, max_seqlen: int,
    valid: torch.Tensor, n: int, s: int, scale: float, half_window: int,
) -> torch.Tensor:
    """Static-shape sliding-window attention via FA4. q/k/v: [N, S, H, D] -> [N, S, H, D].

    Keeps the full padded ``[N, S]`` layout — no unpad/gather/scatter. Each row is
    one fixed-stride length-``S`` sequence (``cu_seqlens = [0, S, 2S, ...]``, static)
    and ``seqused_k`` (= per-row valid-atom count) tells FA4 that only the first
    ``seqused[i]`` positions are real keys, so padding keys are excluded *exactly*
    (no probability mass leaks to them). Requires valid atoms to be front-packed in
    each row — enforced in :func:`build_attention_params`.

    Everything is static-shape: no ``torch.nonzero``, no data-dependent gather, so
    the surrounding compiled region never recompiles and the call is CUDA-graph
    capturable. Padding query rows are zeroed on the way out (matches the old path).
    """
    backend = _flash_backend(q.device)
    if backend == "fa4":
        from flash_attn.cute import (  # ty: ignore[unresolved-import]  # optional extra
            flash_attn_varlen_func,  # lazy — pulls CUTLASS
        )
    elif backend == "fa2":
        # FA2's varlen entry point is a different module from FA4's; the argument names below
        # are the ones both accept.
        from flash_attn.flash_attn_interface import (  # ty: ignore[unresolved-import]
            flash_attn_varlen_func,
        )
    else:  # pragma: no cover - the caller checks first
        msg = "no usable flash backend for this device; the caller should have taken _sdpa_band"
        raise RuntimeError(msg)

    in_dtype = q.dtype
    nh, hd = q.shape[2], q.shape[3]
    # Zero the padding rows of q/k/v (positions >= seqused). Padding keys are already
    # excluded by seqused_k, but padding-query rows are skipped by seqused_q and left
    # with undefined activations/LSE that poison the backward with NaN (NaN*0 into
    # dk/dv). Selecting zeros here (not multiplying) gives a clean, differentiable
    # zero for padding rows: their grad path is cut, valid rows are untouched.
    row_mask = valid.reshape(n * s, 1, 1)

    def _clean(t):
        t = t.reshape(n * s, nh, hd).to(torch.bfloat16)
        return torch.where(row_mask, t, torch.zeros_like(t))
    q, k, v = _clean(q), _clean(k), _clean(v)
    # half_window < 0 -> global attention (no sliding window).
    window = (-1, -1) if half_window < 0 else (half_window, half_window)
    # BOTH seqused_q and seqused_k are required: with a sliding window, passing only
    # seqused_k misaligns the window against the fixed-stride sequence (verified: it
    # diverges from the packed/SDPA reference; passing both matches to bf16 tol).
    # BOTH seqused_q and seqused_k where they exist. FA2 gained them at different releases, so
    # pass only what this build's signature declares rather than assuming: a missing keyword is a
    # TypeError, and silently dropping seqused_k would let padding keys take probability mass.
    import inspect

    accepted = inspect.signature(flash_attn_varlen_func).parameters
    kw = {name: seqused for name in ("seqused_q", "seqused_k") if name in accepted}
    if "seqused_k" in kw:
        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            softmax_scale=scale,
            window_size=window,
            **kw,
        )
        if isinstance(out, tuple):  # return_lse / aux outputs
            out = out[0]
        out = out.reshape(n, s, nh, hd)
    else:
        # NO seqused on this build -- FA2 2.8.3's varlen entry point takes neither, and its
        # release line never gained them. Dropping them is not an option: with fixed-stride
        # cu_seqlens the padding positions stay in the sequence, their zeroed keys score 0
        # against every query, and softmax hands them real probability mass.
        #
        # So this branch removes the padding instead of describing it: unpad to a densely
        # packed [total_valid, H, D] with the true per-row lengths in cu_seqlens, attend, and
        # scatter back. A padding key cannot take mass because it is not there, and the window
        # is measured inside each real sequence rather than across a padded stride.
        #
        # The cost is the property the FA4 path exists to keep. `unpad_input` calls
        # `torch.nonzero`, so this is a data-dependent shape: it syncs, it is NOT CUDA-graph
        # capturable, and a compiled region around it re-traces when the valid count moves.
        # That is the price of an sm80 card, and it is still the fast path -- `_sdpa_band`
        # materialises an [N, S, S] mask, which is 24 GiB at the shape this block runs.
        from flash_attn.bert_padding import (  # ty: ignore[unresolved-import]
            index_first_axis,
            pad_input,
            unpad_input,
        )

        q4 = q.reshape(n, s, nh, hd)
        q_un, indices, cu_var, max_var = unpad_input(q4, valid)[:4]
        flat = (n * s, nh, hd)
        k_un = index_first_axis(k.reshape(*flat), indices)
        v_un = index_first_axis(v.reshape(*flat), indices)
        out = flash_attn_varlen_func(
            q_un, k_un, v_un,
            cu_seqlens_q=cu_var, cu_seqlens_k=cu_var,
            max_seqlen_q=max_var, max_seqlen_k=max_var,
            softmax_scale=scale,
            window_size=window,
        )
        if isinstance(out, tuple):
            out = out[0]
        # pad_input writes zeros at the padding rows, which is what the `where` below wants.
        out = pad_input(out, indices, n, s).reshape(n, s, nh, hd)
    # seqused_q skips padding-query rows (position >= seqused), leaving them
    # uninitialized (can be NaN). SELECT with where (not multiply) so that garbage
    # is discarded rather than turned into NaN*0=NaN; matches the old pad_input zeros.
    mask = valid.unsqueeze(-1).unsqueeze(-1)
    return torch.where(mask, out, torch.zeros_like(out)).to(in_dtype)


def sparse_neighbor_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    neighbor_idx: torch.Tensor,
    neighbor_mask: torch.Tensor,
    valid: torch.Tensor,
    scale: float,
    *,
    query_chunk_size: int = 64,
) -> torch.Tensor:
    """Attention over explicit per-query neighbor indices.

    q/k/v are [N, S, H, D], neighbor_idx/mask are [N, S, K]. The query axis is
    chunked so local+structure attention stays O(N * chunk * K) in peak memory
    instead of materializing [N, S, K, H, D] for the whole atom crop.
    """
    n, s, h, d = q.shape
    k_nei = neighbor_idx.shape[-1]
    in_dtype = q.dtype
    attn_dtype = torch.bfloat16 if q.is_cuda else q.dtype
    q_attn = q.to(attn_dtype)
    k_attn = k.to(attn_dtype)
    v_attn = v.to(attn_dtype)
    k_flat = k_attn.reshape(n * s, h, d)
    v_flat = v_attn.reshape(n * s, h, d)
    batch_offset = (torch.arange(n, device=q.device, dtype=torch.long) * s).view(n, 1, 1)
    out = q.new_zeros(n, s, h, d)

    for start in range(0, s, query_chunk_size):
        stop = min(start + query_chunk_size, s)
        idx_chunk = neighbor_idx[:, start:stop].to(torch.long)
        mask_chunk = neighbor_mask[:, start:stop]
        flat_idx = (idx_chunk + batch_offset).reshape(-1)

        k_g = k_flat[flat_idx].view(n, stop - start, k_nei, h, d)
        v_g = v_flat[flat_idx].view(n, stop - start, k_nei, h, d)
        q_chunk = q_attn[:, start:stop]

        scores = torch.einsum("nchd,nckhd->nchk", q_chunk, k_g) * scale
        valid_query = valid[:, start:stop].unsqueeze(-1)
        allowed = mask_chunk & valid_query
        scores = scores.masked_fill(
            ~allowed.unsqueeze(2), torch.finfo(scores.dtype).min
        )
        # Invalid padded queries have no legal keys; avoid all -inf softmax rows.
        scores = torch.where(valid_query.unsqueeze(2), scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, dim=-1).to(v_g.dtype)
        out_chunk = torch.einsum("nchk,nckhd->nchd", weights, v_g)
        out[:, start:stop] = out_chunk.to(in_dtype)

    return out * valid.unsqueeze(-1).unsqueeze(-1)


@torch.no_grad()
def build_local_structure_neighbor_indices(
    x_t: Float[torch.Tensor, "N S 3"],
    valid: Bool[torch.Tensor, "N S"],
    *,
    seq_neighbors: int = 128,
    structure_neighbors: int = 128,
    query_chunk_size: int = 128,
) -> tuple[Int[torch.Tensor, "N S K"], Bool[torch.Tensor, "N S K"]]:
    """Build sequence-local plus x_t-nearest neighbor indices.

    The sequence-local set is the nearest ``seq_neighbors`` atoms in valid atom
    order and includes self. The structure set takes the nearest
    ``structure_neighbors`` atoms by current noisy coordinates after excluding
    the sequence-local set, so the concatenated set is duplicate-free.

    This avoids full [N, S, S] distance tensors; each batch row and query chunk
    is processed independently to keep peak memory bounded.
    """
    n, s = valid.shape
    device = valid.device
    if seq_neighbors < 1:
        raise ValueError("seq_neighbors must be >= 1 so every valid atom includes self.")
    if structure_neighbors < 0:
        raise ValueError("structure_neighbors must be >= 0.")

    seq_k = min(seq_neighbors, s)
    struct_k = min(structure_neighbors, s)
    total_k = seq_k + struct_k
    neighbor_idx = torch.zeros(n, s, total_k, dtype=torch.int32, device=device)
    neighbor_mask = torch.zeros(n, s, total_k, dtype=torch.bool, device=device)

    coords = x_t.to(torch.float32)
    for b in range(n):
        valid_pos = torch.nonzero(valid[b], as_tuple=False).flatten()
        m = int(valid_pos.numel())
        if m == 0:
            continue

        row_seq_k = min(seq_k, m)
        ranks = torch.arange(m, device=device)
        rank_dist = (ranks[:, None] - ranks[None, :]).abs().to(torch.float32)
        # Stable tie-break toward lower atom index for deterministic neighbors.
        rank_dist = rank_dist + valid_pos.view(1, m).to(torch.float32) * 1e-6
        _, seq_rank_idx = torch.topk(rank_dist, k=row_seq_k, dim=-1, largest=False)
        seq_atom_idx = valid_pos[seq_rank_idx]
        neighbor_idx[b, valid_pos, :row_seq_k] = seq_atom_idx.to(torch.int32)
        neighbor_mask[b, valid_pos, :row_seq_k] = True

        if struct_k == 0:
            continue

        row_struct_k = min(struct_k, max(m - row_seq_k, 0))
        if row_struct_k == 0:
            continue

        row_coords = coords[b, valid_pos]
        for q_start in range(0, m, query_chunk_size):
            q_stop = min(q_start + query_chunk_size, m)
            q_pos = valid_pos[q_start:q_stop]
            d2 = torch.cdist(row_coords[q_start:q_stop], row_coords, p=2).square()
            seq_excluded = torch.zeros(q_stop - q_start, m, dtype=torch.bool, device=device)
            seq_excluded.scatter_(1, seq_rank_idx[q_start:q_stop, :row_seq_k], True)
            d2 = d2.masked_fill(seq_excluded, float("inf"))
            d2 = d2 + valid_pos.view(1, m).to(torch.float32) * 1e-6
            struct_vals, struct_rank_idx = torch.topk(
                d2, k=row_struct_k, dim=-1, largest=False
            )
            offset = seq_k
            neighbor_idx[b, q_pos, offset:offset + row_struct_k] = valid_pos[
                struct_rank_idx
            ].to(torch.int32)
            neighbor_mask[b, q_pos, offset:offset + row_struct_k] = torch.isfinite(
                struct_vals
            )

    return neighbor_idx, neighbor_mask


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@typecheck
def apply_rotary_emb_3d(
    x: Float[torch.Tensor, "N S H D"],
    cos: Float[torch.Tensor, "N S half"],
    sin: Float[torch.Tensor, "N S half"],
) -> Float[torch.Tensor, "N S H D"]:
    """Apply RoPE to ``x`` using batch-dependent ``cos``/``sin``.

    Only the leading ``2*half`` channels of the head dim are rotated; any
    trailing channels (when the active rope frequencies underfill ``D/2``) pass
    through unchanged.
    """
    ro_dim = cos.shape[-1] * 2
    cos = cos.unsqueeze(2).repeat(1, 1, 1, 2)  # [N, S, 1, 2*half]
    sin = sin.unsqueeze(2).repeat(1, 1, 1, 2)
    x_rot = x[..., :ro_dim]
    # cos/sin are fp32 (angle precision), which promotes a bf16 x to fp32. Do the
    # rotation math in the promoted dtype but cast back to the input dtype, so q/k
    # keep the same dtype as the (unrotated) v — otherwise SDPA rejects the mixed
    # dtypes and flash silently upcasts. Mirrors _qk_norm's `.to(x.dtype)`.
    x_rot = (x_rot * cos + _rotate_half(x_rot) * sin).to(x.dtype)
    return torch.cat([x_rot, x[..., ro_dim:]], dim=-1)


@typecheck
@torch.no_grad()
def build_3d_rope(
    ref_pos: Float[torch.Tensor, "B N 3"],
    ref_space_uid: Int[torch.Tensor, "B N"],
    head_dim: int,
    *,
    n_spatial_per_axis: int = 2,
    n_uid_pairs: int = 10,
    spatial_base_freq: float = 20.0,
    uid_base_freq: float = 10000.0,
) -> tuple[Float[torch.Tensor, "B N half"], Float[torch.Tensor, "B N half"]]:
    """Build cos/sin for 3D spatial RoPE + per-residue UID RoPE.

    The spatial part rotates by the reference (x, y, z) coordinates so that
    attention scores depend on relative atomic geometry; the UID part rotates by
    ``ref_space_uid`` so atoms in different spaces (chains / fragments) are
    separated. ``3 * n_spatial_per_axis + n_uid_pairs`` frequencies are used and
    zero-padded up to ``head_dim // 2``.
    """
    device = ref_pos.device
    b, n = ref_pos.shape[:2]
    half_dim = head_dim // 2
    n_spatial_total = 3 * n_spatial_per_axis

    spatial_inv_freq = 1.0 / (
        spatial_base_freq
        ** (torch.arange(0, n_spatial_per_axis, dtype=torch.float32, device=device) / n_spatial_per_axis)
    )
    uid_inv_freq = 1.0 / (
        uid_base_freq
        ** (torch.arange(0, n_uid_pairs, dtype=torch.float32, device=device) / n_uid_pairs)
    )

    pos_f32 = ref_pos.float()
    spatial_freqs = torch.einsum("bna,k->bnak", pos_f32, spatial_inv_freq)
    spatial_freqs = spatial_freqs.reshape(b, n, n_spatial_total)

    uid_freqs = torch.einsum("bn,k->bnk", ref_space_uid.float(), uid_inv_freq)

    n_active = n_spatial_total + n_uid_pairs
    if n_active > half_dim:
        msg = (
            f"3D RoPE active frequencies ({n_active}) exceed head_dim//2 ({half_dim}); "
            f"reduce n_spatial_per_axis / n_uid_pairs or increase head_dim."
        )
        raise ValueError(msg)
    freqs = torch.cat([spatial_freqs, uid_freqs], dim=-1)
    if n_active < half_dim:
        pad = torch.zeros(b, n, half_dim - n_active, device=device, dtype=torch.float32)
        freqs = torch.cat([freqs, pad], dim=-1)

    return freqs.cos(), freqs.sin()


def _qk_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),)).to(x.dtype)


class SWA3DRoPEAttention(nn.Module):
    """Sliding-window self-attention with 3D RoPE — ESMFold2 Algorithm 8.

    Attention: Wqkv -> RMSNorm(q,k) -> 3D RoPE -> sliding-window attention ->
    sigmoid(gate_proj) gate -> out_proj. This mirrors the ESMFold2 reference
    code: qk RMSNorm (over head_dim, non-affine) and the sigmoid output gate are
    both present; Wqkv/out_proj/gate_proj are bias-free with default init (the
    block's adaLN-Zero supplies the identity-at-init residual gate).

    Operates on a flattened ``[N, S, d]`` sequence where ``N = A * B`` packs the
    augmentation and batch axes. ``cos``/``sin`` and the varlen unpadding tensors
    are precomputed once (they are step-invariant) and passed in via
    ``attention_params``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        half_window: int = 64,
        *,
        magnitude_preserving: bool = False,
        mp_full: bool = False,
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

    def forward(
        self,
        x: Float[torch.Tensor, "N S d"],
        attention_params: tuple,
    ) -> Float[torch.Tensor, "N S d"]:
        """Forward pass. ``attention_params`` = (cos, sin, seqused, cu_seqlens, max_seqlen, valid)."""
        n, s = x.shape[:2]
        cos, sin, seqused, cu_seqlens, max_seqlen, valid = attention_params

        qkv = self.Wqkv(x).view(n, s, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 1, 3, 4).unbind(0)  # each [N, S, H, D]
        q, k = _qk_norm(q), _qk_norm(k)
        q = apply_rotary_emb_3d(q, cos, sin)
        k = apply_rotary_emb_3d(k, cos, sin)

        # FA4 if the card can run it, else FA2, else the dense band-mask. The choice is the
        # card's as much as the install's -- see `_flash_backend`.
        if _flash_backend(x.device) is not None:
            out = self._flash_window(q, k, v, cu_seqlens, seqused, max_seqlen, valid, n, s)
        else:
            out = self._sdpa_band(q, k, v, valid)

        out = out.reshape(n, s, -1)
        gate = self.gate_proj(x)
        if out.is_cuda:
            # `sigmoid(gate) * out` in ONE triton pass instead of torch's sigmoid-then-multiply,
            # which reads and writes the whole [N, S, d] twice. The kernel is gated_projection's
            # `_sigmul`, already public as `kernels.sigmoid_gate_fused` and already tuned on this
            # card -- nothing new to build or register. It carries its own autograd, and its
            # shape_key is `both_key(rows_of(...))`, which is the convention its own family uses.
            from miniworld_engine import kernels

            out = kernels.sigmoid_gate_fused(gate, out)
        else:
            out = out * torch.sigmoid(gate)   # CPU unit tests: no triton
        return self.out_proj(out)

    def _flash_window(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seqused: torch.Tensor,
        max_seqlen: int,
        valid: torch.Tensor,
        n: int,
        s: int,
    ) -> torch.Tensor:
        return flash_window_seqused(
            q, k, v, cu_seqlens, seqused, max_seqlen, valid, n, s,
            self.scale, self.half_window,
        )

    def _sdpa_band(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Dense fallback: band mask over valid atoms. O(S^2) — test/CPU only."""
        _n, s = q.shape[:2]
        rank = torch.cumsum(valid.to(torch.long), dim=1) - 1  # [N, S]
        within = (rank.unsqueeze(2) - rank.unsqueeze(1)).abs() <= self.half_window
        allowed = within & valid.unsqueeze(1) & valid.unsqueeze(2)
        allowed = allowed | torch.eye(s, dtype=torch.bool, device=q.device).unsqueeze(0)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=allowed.unsqueeze(1),
            scale=self.scale,
        ).transpose(1, 2)
        return out * valid.unsqueeze(-1).unsqueeze(-1)


@typecheck
def build_attention_params(
    cos: Float[torch.Tensor, "B N half"],
    sin: Float[torch.Tensor, "B N half"],
    valid: Bool[torch.Tensor, "N S"],
    num_aug: int,
) -> tuple:
    """Assemble the per-call SWA attention params from RoPE + validity mask.

    ``cos``/``sin`` are computed per batch element (no augmentation axis, since
    they come from the fixed reference conformer); they are expanded to the
    flattened ``N = num_aug * B`` axis here.

    The windowed attention runs on the full padded ``[N, S]`` layout (no unpadding).
    We hand FA4 a fixed-stride varlen layout: every row is one length-``S`` sequence
    (``cu_seqlens = [0, S, 2S, ...]`` — static, depends only on ``N``/``S``) and
    ``seqused`` (= per-row valid-atom count) marks how many leading positions are
    real. No ``torch.nonzero`` (data-dependent shape) and no gather, so the caller's
    compiled region never recompiles and the path is CUDA-graph capturable.

    Precondition: valid atoms are FRONT-PACKED in each row (``valid[i]`` is
    ``True`` for the first ``seqused[i]`` positions, then ``False``). seqused_k
    masking relies on this; set ``settings.swa_check_front_packed`` to assert it at runtime.
    """
    cos = cos.repeat(num_aug, 1, 1)
    sin = sin.repeat(num_aug, 1, 1)
    n, s = valid.shape
    seqused = valid.sum(dim=-1, dtype=torch.int32)  # [N] valid atoms per row
    cu_seqlens = torch.arange(0, (n + 1) * s, s, dtype=torch.int32, device=valid.device)
    max_seqlen = s
    if _check_front_packed():
        expected = torch.arange(s, device=valid.device).unsqueeze(0) < seqused.unsqueeze(1)
        if not torch.equal(valid, expected):
            raise RuntimeError(
                "SWA seqused_k attention requires front-packed valid atoms, but the "
                "atom_mask has interior gaps. Use a gather/rank-based path instead."
            )
    return cos, sin, seqused, cu_seqlens, max_seqlen, valid
