import numpy as np
import torch


@torch.no_grad()
def weighted_align(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Align x to y using weighted least squares."""
    if x.shape != y.shape:
        msg = "x and y must have the same shape."
        raise ValueError(msg)
    if weight.shape != x.shape[:-1]:
        msg = (
            "weight must have the same shape as x and y except for the last dimension."
        )
        raise ValueError(msg)
    if x.ndim < 2:
        msg = "x and y must have at least 2 dimensions."
        raise ValueError(msg)
    if x.shape[-1] != 3:
        msg = "Last dimension of x and y must be of size 3."
        raise ValueError(msg)

    L = x.shape[-2]  # Length of the sequence
    original_shape = x.shape
    x, y = x.reshape(-1, L, 3), y.reshape(-1, L, 3)  # (AB, L, 3)
    weight = weight.reshape(-1, L)  # (AB, L)

    if weight is None:
        weight = torch.ones_like(x[..., 0])

    # Compute the weighted centroids
    w_sum = weight.sum(dim=-1, keepdim=True)
    weight = weight.unsqueeze(-1)  # (AB, L, 1)
    x_centroid = (x * weight).sum(dim=-2) / w_sum
    y_centroid = (y * weight).sum(dim=-2) / w_sum

    x_centroid = x_centroid.unsqueeze(-2)  # (AB, 1, 3)
    y_centroid = y_centroid.unsqueeze(-2)  # (AB, 1, 3)

    # Center the points
    x_centered = x - x_centroid
    y_centered = y - y_centroid

    # Compute the covariance matrix
    cov_matrix = torch.einsum("bni,bnj->bij", x_centered * weight, y_centered)

    # Singular Value Decomposition
    u, _, v = torch.linalg.svd(cov_matrix)
    v = v.mH

    rotation_matrix = torch.einsum("bij,bkj -> bik", u, v)
    F = torch.eye(3, dtype=cov_matrix.dtype, device=cov_matrix.device)[None].repeat(
        x.shape[0],
        1,
        1,
    )
    F[:, -1, -1] = torch.where(
        torch.det(rotation_matrix) < 0,
        torch.tensor(-1.0, dtype=rotation_matrix.dtype, device=rotation_matrix.device),
        torch.tensor(1.0, dtype=rotation_matrix.dtype, device=rotation_matrix.device),
    )
    rotation_matrix = torch.einsum(
        "bij, bjk, blk -> bil",
        u,
        F,
        v,
    )

    aligned_x = (
        torch.einsum("bni, bij -> bnj", x_centered, rotation_matrix) + y_centroid
    )

    # restore original shape
    return aligned_x.reshape(*original_shape)


def kabsch_rmsd(coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Calculate NxN RMSD matrix for N coordinate sets (NumPy version).
    coords: (N, L, 3)
    mask:  (N, L) or (N, L, 1) boolean
    """
    # shape
    _n, _l, _ = coords.shape

    # ensure mask shape
    if mask.ndim == 3:
        mask = mask.squeeze(-1)
    mask = mask.astype(bool)  # (N, L)

    # (1) centroid using masked atoms
    counts = np.sum(mask, axis=1, keepdims=True)  # (N,1)
    counts = np.clip(counts, 1, None)
    centroid = (coords * mask[..., None]).sum(axis=1, keepdims=True) / counts[..., None]

    # (2) center and zero invalid atoms
    coords_centered = (coords - centroid) * mask[..., None]

    # (3) broadcast
    P = coords_centered[:, None, :, :]  # (N,1,L,3) -> (N,N,L,3)
    Q = coords_centered[None, :, :, :]  # (1,N,L,3) -> (N,N,L,3)

    # pairwise common-atom mask
    pair_mask = mask[:, None, :] & mask[None, :, :]  # (N,N,L)
    L_pair = np.sum(pair_mask, axis=-1).clip(min=1e-8)  # (N,N)

    # masked coords
    Pm = P * pair_mask[..., None]
    Qm = Q * pair_mask[..., None]

    # norms
    P_norm_sq = np.sum(Pm**2, axis=(2, 3))
    Q_norm_sq = np.sum(Qm**2, axis=(2, 3))

    # covariance H (N,N,3,3)
    H = np.einsum("ijkl,ijml->ijkm", Pm, Qm)

    # stabilizer
    H += np.eye(3)[None, None, :, :] * 1e-2

    # (4) batch SVD
    # NumPy 1.21+ supports batched SVD
    U, S, Vt = np.linalg.svd(H)  # (N,N,3,3)

    det_sign = np.sign(
        np.linalg.det(
            np.einsum(
                "...ij,...jk->...ik", Vt.transpose(0, 1, 3, 2), U.transpose(0, 1, 2, 3)
            )
        )
    )

    # correct improper rotation
    S[..., -1] *= det_sign

    trace_S = S.sum(axis=-1)  # (N,N)

    rmsd_sq = (P_norm_sq + Q_norm_sq - 2 * trace_S) / L_pair
    rmsd_sq = np.clip(rmsd_sq, a_min=0.0, a_max=None)

    return np.sqrt(rmsd_sq)  # (N,N)


def kabsch_rmsd_ref(
    coords: np.ndarray, mask: np.ndarray, ref_idx: int = 0, eps: float = 1e-7
) -> np.ndarray:
    """Fully-vectorized NxN RMSD by aligning all structures to a single reference (coords[ref_idx]).
    - Input: NumPy arrays
      coords: (N, L, 3) float
      mask:   (N, L) or (N, L, 1) bool/0-1
    - Output: NumPy array (N, N) float
    - No Python loops; uses PyTorch batched ops internally for stability.
    """
    # --- to torch (float64 for stable SVD) ---
    if mask.ndim == 3:
        mask = np.squeeze(mask, axis=-1)
    tC = torch.from_numpy(coords).to(torch.float64)
    tM = torch.from_numpy(mask.astype(bool))

    N, _l, _ = tC.shape
    device = tC.device

    # --- reference-centered coords (zero-out invalid) ---
    ref_m = tM[ref_idx]  # (L,)
    # ref_cnt = torch.clamp(ref_m.sum(), min=1)
    ref_centroid = tC[ref_idx, ref_m].mean(dim=0, keepdim=True)  # (1,3)
    ref_centered = tC[ref_idx] - ref_centroid  # (L,3)
    ref_centered = torch.where(
        ref_m[:, None], ref_centered, torch.zeros(1, 3, dtype=tC.dtype, device=device)
    )

    # --- intersection mask vs reference; per-structure centroids ---
    inter = tM & ref_m[None, :]  # (N,L)
    cnts = torch.clamp(inter.sum(dim=1, keepdim=True), min=1)  # (N,1)
    centroids = (tC * inter[..., None]).sum(dim=1, keepdim=True) / cnts[
        ..., None
    ]  # (N,1,3)

    # --- centered inputs with mask ---
    P = (tC - centroids) * inter[..., None]  # (N,L,3)
    Q = ref_centered[None, :, :] * inter[..., None]  # (N,L,3)

    # --- covariance S = P^T Q (N,3,3) ---
    S = torch.einsum("nlc,nld->ncd", P, Q)  # (N,3,3)
    if eps > 0:
        S = S + eps * torch.eye(3, dtype=S.dtype, device=device)[None, :, :]

    # --- batched SVD -> rotation (reflection fixed) ---
    U, _, Vh = torch.linalg.svd(S, full_matrices=False)  # (N,3,3)
    R = Vh.transpose(-2, -1) @ U.transpose(-2, -1)  # (N,3,3)
    detR = torch.linalg.det(R)
    neg = detR < 0
    if neg.any():
        Vh_fix = Vh.clone()
        Vh_fix[neg, -1, :] *= -1
        R = Vh_fix.transpose(-2, -1) @ U.transpose(-2, -1)

    # --- handle degenerate intersections (<3 common atoms): fall back to I ---
    valid = inter.sum(dim=1) >= 3  # (N,)
    I3 = torch.eye(3, dtype=R.dtype, device=device).expand(N, 3, 3)
    R = torch.where(valid[:, None, None], R, I3)

    # --- apply rotations; build aligned coords ---
    aligned = (tC - centroids) @ R  # (N,L,3)

    # --- pairwise RMSD with per-pair mask ---
    diff = aligned[:, None, :, :] - aligned[None, :, :, :]  # (N,N,L,3)
    diff_sq = (diff * diff).sum(dim=-1)  # (N,N,L)
    pair_mask = tM[:, None, :] & tM[None, :, :]  # (N,N,L)
    Lpair = torch.clamp(pair_mask.sum(dim=-1), min=1)  # (N,N)
    rmsd = torch.sqrt(
        (diff_sq.masked_fill(~pair_mask, 0.0).sum(dim=-1) / Lpair).clamp_min(0.0)
    )  # (N,N)

    # --- back to numpy ---
    return rmsd.cpu().numpy()
