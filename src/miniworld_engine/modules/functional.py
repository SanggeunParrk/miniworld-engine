"""Pure functions of the model math (no parameters): the gates the modules share, rotary embeddings, the sampler's rigid-motion glue
(Kabsch by Horn's quaternion method, churn, augmentation, the Euler step, atom pooling), the smooth-LDDT loss and RoseTTAFold3's
triangle-multiplication length factor.

These are the PyTorch definitions; a kernel of one of them is checked against the function here.
"""

from __future__ import annotations

import torch

from miniworld_engine.modules.primitives import wide


# ------------------------------------------------------------------------------------------------------------------------------- gates
def sigmoid_gate(gate: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Sigmoid gating mechanism: ``sigmoid(gate) * x``."""
    return torch.sigmoid(gate) * x


def swish_gate(gate: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Swish gating mechanism: ``gate * sigmoid(gate) * x`` (SwiGLU with ``gate`` the activated half)."""
    return gate * torch.sigmoid(gate) * x


def dropout_scale(x: torch.Tensor, p: float, broadcast_dim: int) -> torch.Tensor:
    """Broadcast dropout scale ``(rand > p) / (1 - p)`` in ``x``'s dtype, of ``x``'s shape with ``broadcast_dim`` set to 1 (drop_row:
    ``broadcast_dim=1``, drop_col: ``broadcast_dim=2``). The caller multiplies its update by it (training only)."""
    shape = list(x.shape)
    shape[broadcast_dim] = 1
    keep = torch.rand(shape, device=x.device, dtype=x.dtype) > p
    return keep.to(x.dtype) / (1.0 - p)


# -------------------------------------------------------------------------------------------------------------------------------- rope
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb_3d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to ``x`` [N, S, H, D] using batch-dependent ``cos``/``sin`` [N, S, half].

    Only the leading ``2*half`` channels of the head dim are rotated; any trailing channels (when the active rope frequencies underfill
    ``D/2``) pass through unchanged.
    """
    ro_dim = cos.shape[-1] * 2
    cos = cos.unsqueeze(2).repeat(1, 1, 1, 2)  # [N, S, 1, 2*half]
    sin = sin.unsqueeze(2).repeat(1, 1, 1, 2)
    x_rot = x[..., :ro_dim]
    # cos/sin are fp32 (angle precision), which promotes a bf16 x to fp32. Do the rotation math in the promoted dtype but cast back to
    # the input dtype, so q/k keep the same dtype as the (unrotated) v.
    x_rot = (x_rot * cos + _rotate_half(x_rot) * sin).to(x.dtype)
    return torch.cat([x_rot, x[..., ro_dim:]], dim=-1)


@torch.no_grad()
def build_3d_rope(
    ref_pos: torch.Tensor,
    ref_space_uid: torch.Tensor,
    head_dim: int,
    *,
    n_spatial_per_axis: int = 2,
    n_uid_pairs: int = 10,
    spatial_base_freq: float = 20.0,
    uid_base_freq: float = 10000.0,
    angles_bf16_compat: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin [B, N, head_dim // 2] for 3D spatial RoPE + per-residue UID RoPE from ``ref_pos`` [B, N, 3] and
    ``ref_space_uid`` [B, N] (the tables of :func:`apply_rotary_emb_3d` in ESMFold2's SWA atom attention).

    The spatial part rotates by the reference (x, y, z) coordinates so that attention scores depend on relative atomic geometry; the UID
    part rotates by ``ref_space_uid`` so atoms in different spaces (chains / fragments) are separated. ``3 * n_spatial_per_axis +
    n_uid_pairs`` frequencies are used and zero-padded up to ``head_dim // 2``.

    ``angles_bf16_compat`` (formerly ``settings.rope_angles_bf16_compat``): the angles as einsums, which run as bf16 matmuls under an
    autocast region -- for checkpoints trained before the angles were pinned to fp32.
    """
    device = ref_pos.device
    b, n = ref_pos.shape[:2]
    half_dim = head_dim // 2
    n_spatial_total = 3 * n_spatial_per_axis

    w = wide(ref_pos)   # the angles in at least fp32 (fp64 for an fp64 call)
    spatial_inv_freq = 1.0 / (
        spatial_base_freq ** (torch.arange(0, n_spatial_per_axis, dtype=w, device=device) / n_spatial_per_axis)
    )
    uid_inv_freq = 1.0 / (uid_base_freq ** (torch.arange(0, n_uid_pairs, dtype=w, device=device) / n_uid_pairs))

    # Broadcast multiplies, not einsums: these are outer products, and under an autocast region ``einsum`` runs them as bf16 matmuls
    # (UID angles are ids in the thousands times a frequency; coordinates times a frequency) -- the angles must stay fp32.
    pos_w = ref_pos.to(w)
    if angles_bf16_compat:
        spatial_freqs = torch.einsum("bna,k->bnak", pos_w, spatial_inv_freq)
        spatial_freqs = spatial_freqs.reshape(b, n, n_spatial_total)
        uid_freqs = torch.einsum("bn,k->bnk", ref_space_uid.to(w), uid_inv_freq)
    else:
        spatial_freqs = pos_w[..., None] * spatial_inv_freq
        spatial_freqs = spatial_freqs.reshape(b, n, n_spatial_total)
        uid_freqs = ref_space_uid.to(w)[..., None] * uid_inv_freq

    n_active = n_spatial_total + n_uid_pairs
    if n_active > half_dim:
        msg = (
            f"3D RoPE active frequencies ({n_active}) exceed head_dim//2 ({half_dim}); "
            f"reduce n_spatial_per_axis / n_uid_pairs or increase head_dim."
        )
        raise ValueError(msg)
    freqs = torch.cat([spatial_freqs, uid_freqs], dim=-1)
    if n_active < half_dim:
        pad = torch.zeros(b, n, half_dim - n_active, device=device, dtype=freqs.dtype)
        freqs = torch.cat([freqs, pad], dim=-1)

    return freqs.cos(), freqs.sin()


# ------------------------------------------------------------------------------------------------------------- atom attention windows
# The AF3 block-local atom attention: query window w holds atoms [32 w, 32 w + 32) and reads a key window of ``keys`` atoms (128, or
# ESMFold2's 192) centred on it.
ATOM_QUERIES, ATOM_KEYS = 32, 128


def slide_starts(n_real: torch.Tensor, nwin: int, keys: int = ATOM_KEYS) -> torch.Tensor:
    """[B] real atom counts -> [B, nwin] first key of every query window, AlphaFold 3's rule: the window ``[32 w - 48, 32 w + 80)``
    shifted back inside the real atoms ``[0, N)`` (``featurisation: AtomCrossAtt.compute_features``). A window short of the start
    begins at 0, one past the end at ``N - 128``. Below 128 real atoms the rule has no valid window; the window then begins at 0 and
    the keys past ``N`` are masked (AlphaFold 3's own indices wrap there).

    A wider subset (``keys`` 192: ``[32 w - 80, 32 w + 112)``) takes the rule exactly as written there: a window short of the start
    begins at 0, otherwise it ends at atom ``N - 1`` at the latest, so below ``keys`` real atoms a window may begin before atom 0
    (AlphaFold 3's negative indices, padding atoms)."""
    first = ATOM_QUERIES * torch.arange(nwin, device=n_real.device) - (keys - ATOM_QUERIES) // 2
    if keys == ATOM_KEYS:
        first = first.clamp(min=0)
        return torch.minimum(first[None], (n_real - ATOM_KEYS)[:, None]).clamp(min=0)
    return torch.where(first[None] < 0, 0, torch.minimum(first[None], (n_real - keys)[:, None]))


def to_windows(pair: torch.Tensor, keys: int = ATOM_KEYS) -> torch.Tensor:
    """Dense atom pair [B, N, N, d] -> trunked pair [B, nwin, 32, keys, d], ``nwin = ceil(N / 32)``, ``out[b, w, i, j] = pair[b, 32 w + i,
    32 w - (keys - 32) / 2 + j]`` (the clipped window; positions outside [0, N) are zero)."""
    n = pair.shape[1]
    nwin = (n + ATOM_QUERIES - 1) // ATOM_QUERIES
    query = torch.arange(nwin * ATOM_QUERIES, device=pair.device).view(nwin, ATOM_QUERIES)
    key = (torch.arange(nwin, device=pair.device)[:, None] * ATOM_QUERIES - (keys - ATOM_QUERIES) // 2
           + torch.arange(keys, device=pair.device)[None])
    valid = (query < n)[:, :, None] & ((key >= 0) & (key < n))[:, None, :]
    out = pair[:, query.clamp(max=n - 1)[:, :, None], key.clamp(0, n - 1)[:, None, :]]
    return out * valid[None, :, :, :, None].to(pair.dtype)


# ---------------------------------------------------------------------------------------------------------------- SWA atom attention
def build_attention_params(
    cos: torch.Tensor,
    sin: torch.Tensor,
    valid: torch.Tensor,
    num_aug: int,
    *,
    check_front_packed: bool = False,
) -> tuple:
    """Assemble the per-call SWA attention params from RoPE + validity mask.

    ``cos``/``sin`` [B, N, half] are computed per batch element (no augmentation axis, since they come from the fixed reference
    conformer); they are expanded to the flattened ``N = num_aug * B`` axis here. ``valid`` [N, S] bool.

    Returns ``(cos, sin, seqused, cu_seqlens, max_seqlen, valid)``: ``seqused`` the per-row valid-atom count, ``cu_seqlens`` the
    fixed-stride varlen offsets ``[0, S, 2S, ...]`` (the layout of a varlen attention kernel; the band attention reads only ``valid``).

    Precondition: valid atoms are FRONT-PACKED in each row (``valid[i]`` is ``True`` for the first ``seqused[i]`` positions, then
    ``False``). ``check_front_packed`` (formerly ``settings.swa_check_front_packed``) asserts it (a device sync).
    """
    cos = cos.repeat(num_aug, 1, 1)
    sin = sin.repeat(num_aug, 1, 1)
    n, s = valid.shape
    seqused = valid.sum(dim=-1, dtype=torch.int32)  # [N] valid atoms per row
    cu_seqlens = torch.arange(0, (n + 1) * s, s, dtype=torch.int32, device=valid.device)
    max_seqlen = s
    if check_front_packed:
        expected = torch.arange(s, device=valid.device).unsqueeze(0) < seqused.unsqueeze(1)
        if not torch.equal(valid, expected):
            raise RuntimeError(
                "SWA seqused_k attention requires front-packed valid atoms, but the "
                "atom_mask has interior gaps. Use a gather/rank-based path instead."
            )
    return cos, sin, seqused, cu_seqlens, max_seqlen, valid


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
    """Attention over explicit per-query neighbor indices (MiniWorld's local-structure atom attention).

    q/k/v are [N, S, H, D], neighbor_idx/mask are [N, S, K]. The query axis is chunked so local+structure attention stays
    O(N * chunk * K) in peak memory instead of materializing [N, S, K, H, D] for the whole atom crop.
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
        scores = scores.masked_fill(~allowed.unsqueeze(2), torch.finfo(scores.dtype).min)
        # Invalid padded queries have no legal keys; avoid all -inf softmax rows.
        scores = torch.where(valid_query.unsqueeze(2), scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, dim=-1).to(v_g.dtype)
        out_chunk = torch.einsum("nchk,nckhd->nchd", weights, v_g)
        out[:, start:stop] = out_chunk.to(in_dtype)

    return out * valid.unsqueeze(-1).unsqueeze(-1)


# ----------------------------------------------------------------------------------------------------------------------------- sampler
# Kabsch rotation by Horn's quaternion method: the dominant eigenvector of the symmetric 4x4 N(H) by repeated squaring of N + c I (no SVD,
# no host read, so a CUDA-graph capture allows it). A proper rotation always results. Coordinates are fp32 ``[B, N, 3]``, one rigid motion
# per sample.

#: ``(N + cI)^(2**KABSCH_SQUARINGS)``: the ratio of the two largest eigenvalues is raised to ``2**KABSCH_SQUARINGS``.
KABSCH_SQUARINGS = 12


def horn_matrix(h: torch.Tensor) -> torch.Tensor:
    """The symmetric 4x4 ``N`` of Horn (1987) from ``h[..., i, j] = sum_n w_n a[n, i] b[n, j]``."""
    sxx, sxy, sxz = h[..., 0, 0], h[..., 0, 1], h[..., 0, 2]
    syx, syy, syz = h[..., 1, 0], h[..., 1, 1], h[..., 1, 2]
    szx, szy, szz = h[..., 2, 0], h[..., 2, 1], h[..., 2, 2]
    rows = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ]
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def dominant_quaternion(n: torch.Tensor) -> torch.Tensor:
    """Unit eigenvector of the largest eigenvalue of the symmetric 4x4 ``n`` (``[..., 4]``)."""
    # |eigenvalue| <= the trace norm of H <= sqrt(3) |H|_F, and the spectrum of N is the sums of the
    # signed singular values, so shifting by the Frobenius bound makes the matrix positive definite.
    shift = n.square().sum((-2, -1), keepdim=True).sqrt() * 1.0
    m = n + shift * torch.eye(4, dtype=n.dtype, device=n.device)
    for _ in range(KABSCH_SQUARINGS):
        m = m @ m
        m = m / m.abs().amax((-2, -1), keepdim=True).clamp_min(torch.finfo(n.dtype).tiny)
    # m ~ lambda^k q q^T: the column of largest norm is q up to scale (and sign).
    columns = m.transpose(-2, -1)
    norms = columns.square().sum(-1)
    pick = norms.argmax(-1, keepdim=True)
    q = torch.gather(columns, -2, pick[..., None].expand(*pick.shape, 4)).squeeze(-2)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(n.dtype).tiny)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Rotation matrix (column-vector convention) of the unit quaternion ``q = (w, x, y, z)``."""
    w, x, y, z = q.unbind(-1)
    rows = [
        [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (y * x + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
        [2 * (z * x - w * y), 2 * (z * y + w * x), w * w - x * x - y * y + z * z],
    ]
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def kabsch_rotation(covariance: torch.Tensor) -> torch.Tensor:
    """``R`` with ``a @ R ~ b`` (row vectors) for ``covariance = a^T W b``; the same ``R`` as ``U D V^T`` of its SVD."""
    q = dominant_quaternion(horn_matrix(covariance))
    return quaternion_to_matrix(q).transpose(-2, -1)


def churn(sigma, sigma_next, *, window, gamma, gamma_min, sigma_min, sigma_max, noise_scale, variance_floor, clamp):
    """``(sigma_hat, noise magnitude)`` of one churn step; device scalars in, device scalars out (no host read)."""
    one = torch.ones((), dtype=sigma.dtype, device=sigma.device)
    on = (sigma >= sigma_min) & (sigma <= sigma_max) if window else sigma_next > gamma_min
    sigma_hat = sigma * torch.where(on, one * (1 + gamma), one)
    variance = sigma_hat**2 - sigma**2
    if clamp:
        variance = variance.clamp(min=variance_floor)
    return sigma_hat, noise_scale * torch.sqrt(variance)


def augment_noise(coords, mask, rv, tr, noise, sigma, sigma_next, **cfg):
    """Centre (mask-weighted), rotate by the Gram-Schmidt frame of ``rv`` ``[B, 2, 3]``, translate by ``tr`` ``[B, 3]``, mask, add
    ``magnitude * noise``. Returns ``(noisy, sc)`` with ``sc[0] = sigma_hat``."""
    sigma_hat, magnitude = churn(sigma, sigma_next, **cfg)
    w = mask[None, :, None]
    centre = (w * coords).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp(min=1e-6)
    v0, v1 = rv.unbind(-2)
    e0 = v0 / torch.clamp(torch.linalg.norm(v0, dim=-1, keepdim=True), min=1e-10)
    v1 = v1 - e0 * (v1 * e0).sum(dim=-1, keepdim=True)
    e1 = v1 / torch.clamp(torch.linalg.norm(v1, dim=-1, keepdim=True), min=1e-10)
    rot = torch.stack([e0, e1, torch.cross(e0, e1, dim=-1)], dim=-2)
    aug = (torch.einsum("bni,bij->bnj", coords - centre, rot) + tr[:, None, :]) * w
    return aug + magnitude * noise, torch.stack([sigma_hat, magnitude])


def align_euler(noisy, denoised, mask, sc, sigma_next, step_scale):
    """Kabsch-align ``noisy`` onto ``denoised`` (weights ``mask``), then the Euler update with ``sigma_hat = sc[0]``."""
    w = mask[None, :, None]
    total = w.sum(1, keepdim=True)
    cm = (noisy * w).sum(1, keepdim=True) / total
    cr = (denoised * w).sum(1, keepdim=True) / total
    a, b = noisy - cm, denoised - cr
    cov = torch.einsum("bnc,bnd->bcd", a * w, b)
    aligned = a @ kabsch_rotation(cov) + cr
    sigma_hat = sc[0]
    direction = (aligned - denoised) / sigma_hat
    return aligned + step_scale * (sigma_next - sigma_hat) * direction


def pool_atoms(x, idx, gather_mask, atom_mask):
    """The gather (``x[:, idx]`` masked by ``gather_mask``), relu and masked mean over the atoms, as the encoder spells them."""
    g = x[:, idx.long()] * gather_mask.to(x.dtype)[None, :, :, None]
    m = atom_mask.to(wide(x))[None, :, :, None]
    return ((m * torch.relu(g)).sum(-2) / torch.clamp(m.sum(-2), min=1e-10)).to(x.dtype)


# ------------------------------------------------------------------------------------------------------------------------- smooth LDDT
LDDT_BINS = (0.5, 1.0, 2.0, 4.0)
LDDT_NUC_CUTOFF, LDDT_OTHER_CUTOFF = 30.0, 15.0


def smooth_lddt_loss(x_pred: torch.Tensor, gt: torch.Tensor, is_nuc: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """AlphaFold 3 Algorithm 27 written out densely (an [N, N] tensor per augment): x_pred [A, B, N, 3], gt [B, N, 3] (the same rigid
    structure for every augment), is_nuc / mask [B, N] bool -> scalar ``mean_{a, b} (1 - smooth_lddt)``."""
    exact = "donot_use_mm_for_euclid_dist"   # the matmul form cancels |a|^2 + |b|^2 - 2ab: errors ~1e-4 A at structure scale
    d_pred = torch.cdist(x_pred, x_pred, compute_mode=exact)               # [A, B, N, N]
    d_gt = torch.cdist(gt, gt, compute_mode=exact)                         # [B, N, N]
    diff = (d_pred - d_gt).abs()
    score = sum(torch.sigmoid(t - diff) for t in LDDT_BINS) / len(LDDT_BINS)
    cutoff = torch.where(is_nuc, LDDT_NUC_CUTOFF, LDDT_OTHER_CUTOFF).unsqueeze(-1)   # radius of atom i (the row)
    n = gt.shape[-2]
    keep = (d_gt < cutoff) & mask.unsqueeze(-1) & mask.unsqueeze(-2) & ~torch.eye(n, dtype=torch.bool, device=gt.device)
    lddt = (score * keep).sum(dim=(-1, -2)) / keep.sum(dim=(-1, -2)).clamp(min=1)
    return (1.0 - lddt).mean()


# ----------------------------------------------------------------------------------------------------------------------------- trimul
def inverse_length(like: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """``1 / (real sequence length)`` as ``[B, 1, 1, 1]`` in ``like``'s dtype (RoseTTAFold3's triangle-multiplication factor): the number of
    unmasked residues of each sample (at least 1), ``L`` when there is no mask. Computed on the device (no host read), so a captured graph
    takes each replay's mask."""
    if mask is None:
        return like.new_full((like.shape[0], 1, 1, 1), 1.0 / like.shape[1])
    count = mask.sum(-1).clamp_min(1)
    return count.to(wide(like)).reciprocal().to(like.dtype).view(-1, 1, 1, 1)
